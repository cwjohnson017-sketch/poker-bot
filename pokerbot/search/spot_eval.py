"""Exploitability of flop search decisions with an exact river (``docs/value_net.md``, section 5).

For one flop decision (a :class:`Spot`), :func:`evaluate_spot`

1. runs the **value-net search** (``leaf.mode: value_net``) for a fixed number
   of iterations, on a tree built with ``tree.leaf_budget_cost = 1 + k`` so that
   its trunk (flop and turn betting) is exactly the rollout tree's;
2. runs the default **rollout search** with the same settings (``leaf.seed``
   set, everything else the default leaf config);
3. optionally runs variants on the same trunk (:func:`parse_variant`):
   ``value_net_every<n>`` (``leaf.net_every n``) and ``value_net_budget`` (a
   production-style run under a time budget instead of a fixed iteration
   count);
4. checks that every tree has the value-net tree's decision nodes
   (:func:`~pokerbot.search.exact_eval.node_key`), child actions and depth-limit
   leaves, and that the root ranges agree;
5. puts every strategy on the value-net tree: the searches' average strategies
   with :func:`~pokerbot.search.exact_eval.map_sigma` (strict), the blueprint
   with :func:`blueprint_sigma`;
6. scores each profile with
   :func:`~pokerbot.search.exact_eval.trunk_exploitability`: every (leaf, river
   card) river subgame solved exactly, best responses backed up through the
   trunk, on the plain root ranges (no gadget). No leaf model is involved.

The scoring game uses ``eval_max_runouts`` all-in run-outs per board (default
1176 = every turn and river card of a flop, so flop all-ins are exact; the
search itself samples ``solver.max_runouts`` = 48). The value-net search's own
exploitability in the game its leaf model defines (``solver.exploitability()``)
is reported as a sanity reference: the gap to the exact score is how much the
net's game differs from the real one.

Spots (:func:`exploit_spots`) are those of ``runs/search_noise/exploit.py``:
boards from ``np.random.default_rng(seed)`` permutations, a 2.5x button open
and a call, then "BB first", "BTN vs check" and "BTN vs 1/2 lead". The trunk
is the whole flop tree from the street root, so a "BTN" spot also scores the
BB's flop strategy that the search computed there.

:func:`run_evaluation` drives a list of spots, writes the JSON after every spot
(resumable) and renders the markdown report (:func:`render_markdown`). The CLI
is ``scripts/eval_search_exploit.py``.
"""

from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import re
import time
import traceback
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .abstract import RAISE, legal_options
from .blueprint import policy_matrix
from .combos import NUM_COMBOS, blocked_sum
from .config import SearchConfig, search_config
from .exact_eval import map_sigma, node_key, trunk_exploitability
from .solver import RangeSolver
from .tree import DECISION, LEAF, VALUE, SubgameTree
from .value_leaf import FixedLeafValues

C = NUM_COMBOS

SPOT_TYPES = ("bb_first", "btn_vs_check", "btn_vs_lead")
SPOT_NAMES = {
    "bb_first": "BB first",
    "btn_vs_check": "BTN vs check",
    "btn_vs_lead": "BTN vs 1/2 lead",
}
BASE_PROFILES = ("value_net", "rollout", "blueprint")
PROFILE_NAMES = {"value_net": "value-net", "rollout": "rollout search", "blueprint": "blueprint"}
EXACT_FLOP_RUNOUTS = math.comb(49, 2)  # every turn + river of a flop: 1176
HUGE_BUDGET = 1e6


# -- spots ------------------------------------------------------------------------


@dataclass
class Spot:
    label: str  # e.g. "board0 BTN vs check"
    board_index: int
    spot_type: str  # one of SPOT_TYPES
    state: Any  # engine GameState at the decision


def exploit_spots(
    engine: Any,
    game_config: Any,
    boards: int = 2,
    seed: int = 5,
    types: Sequence[str] = SPOT_TYPES,
) -> list[Spot]:
    """The flop decisions of ``runs/search_noise``: per board (a permutation from
    ``np.random.default_rng(seed)``, seat 0 on the button) a 2.5x open
    (raise to 250) and a call, then the types in ``types``. More boards extend
    the same sequence, so the first ``n`` boards never change."""
    bad = [t for t in types if t not in SPOT_TYPES]
    if bad:
        raise ValueError(f"unknown spot types {bad} (expected {SPOT_TYPES})")
    rng = np.random.default_rng(seed)
    out = []
    for b in range(boards):
        deck = rng.permutation(52).tolist()
        base = engine.GameState.new_hand(game_config, 0, deck)  # seat 0 on the button
        base.apply(engine.Action.raise_to(250))
        base.apply(engine.Action.check_call())
        states = {"bb_first": base.clone()}
        states["btn_vs_check"] = base.clone()
        states["btn_vs_check"].apply(engine.Action.check_call())
        states["btn_vs_lead"] = base.clone()
        states["btn_vs_lead"].apply(engine.Action.raise_to(250))  # half-pot lead (pot 500)
        for t in SPOT_TYPES:
            if t in types:
                out.append(Spot(f"board{b} {SPOT_NAMES[t]}", b, t, states[t]))
    return out


# -- settings ---------------------------------------------------------------------


@dataclass
class EvalSettings:
    iters: int = 300  # fixed solver iterations of the searches
    max_nodes: int = 6000  # tree.max_nodes
    leaf_seed: int = 1  # rollout search leaf.seed
    seed: int = 0  # search seed (gadget rollouts)
    river_iters: int = 400  # BatchRiverSolver iterations per river subgame
    mix: float = 0.05
    min_mass: float = 1e-7
    river_batch: int = 256  # halved automatically on CUDA out-of-memory
    eval_max_runouts: int | None = EXACT_FLOP_RUNOUTS  # None: the search's solver.max_runouts
    budget: float = 4.0  # seconds per decision for value_net_budget
    device: str = "auto"
    variants: tuple[str, ...] = ()
    turn_net: str | None = None  # turn-end net checkpoint for the value_net_turn* variants
    search: dict = field(default_factory=dict)  # extra search_config overrides (every search)

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def comparable(self) -> dict:
        """The settings that change the numbers (for resuming)."""
        d = asdict(self)
        for k in ("river_batch", "device"):
            d.pop(k)
        d["variants"] = list(d["variants"])
        return d


_EVERY = re.compile(r"^value_net(_turn)?_every(\d+)$")


def parse_variant(name: str) -> dict:
    """``value_net_every<n>``: value-net search with ``leaf.net_every n``;
    ``value_net_budget``: value-net search under ``EvalSettings.budget`` seconds
    with the default iteration settings (cap and ``min_iterations`` from the
    config) instead of a fixed count, on the same tree. ``value_net_turn``,
    ``value_net_turn_budget`` and ``value_net_turn_every<n>``: the same with the
    turn-end net ``EvalSettings.turn_net`` as the leaf model."""
    if name in ("value_net_budget", "value_net_turn_budget"):
        return {"net_every": 1, "budget": True, "turn": "_turn" in name}
    if name == "value_net_turn":
        return {"net_every": 1, "budget": False, "turn": True}
    m = _EVERY.match(name)
    if m and int(m.group(2)) >= 1:
        return {"net_every": int(m.group(2)), "budget": False, "turn": bool(m.group(1))}
    raise ValueError(
        f"unknown variant {name!r} (value_net[_turn]_every<n>, value_net[_turn]_budget, "
        "value_net_turn)"
    )


def _merge(base: dict, extra: dict | None) -> dict:
    """``extra`` over ``base``, merging nested sections one level deep (as
    :func:`~pokerbot.search.config.search_config` does)."""
    out = {k: dict(v) if isinstance(v, dict) else v for k, v in base.items()}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def search_overrides(
    settings: EvalSettings,
    leaf_mode: str = "rollouts",
    net: str | None = None,
    net_every: int = 1,
    budget: float | None = None,
    num_continuations: int = 4,
) -> dict:
    """``search_config`` overrides for one search run, as ``exploit.py``'s
    ``run_search``: fixed ``settings.iters`` iterations under a huge time budget
    (or ``budget`` seconds with the default iteration settings), the
    blueprint's actions, ``settings.max_nodes``. A value-net run counts every
    leaf ``1 + num_continuations`` nodes, so its trunk equals the rollout tree's."""
    dev = str(settings.torch_device())
    o: dict[str, Any] = {
        "device": dev,
        "fallback_on_error": False,
        "seed": settings.seed,
        "tree": {"actions": "blueprint", "max_nodes": settings.max_nodes},
    }
    if budget is None:
        o["min_iterations"] = settings.iters
        o["time_budget"] = {"flop": HUGE_BUDGET, "turn": HUGE_BUDGET, "river": HUGE_BUDGET}
        o["solver"] = {"iterations": settings.iters}
    else:
        o["time_budget"] = {"flop": budget, "turn": budget, "river": budget}
    if leaf_mode == "rollouts":
        o["leaf"] = {"seed": settings.leaf_seed}
    elif leaf_mode == "value_net":
        o["leaf"] = {"mode": "value_net", "net": net, "net_every": int(net_every)}
        o["tree"]["leaf_budget_cost"] = 1 + int(num_continuations)
    else:
        raise ValueError(f"unknown leaf mode {leaf_mode!r}")
    o = _merge(o, settings.search)
    if leaf_mode == "value_net":  # keep the run's own leaf mode and trunk cost
        o["leaf"] = {**o["leaf"], "mode": "value_net", "net": net, "net_every": int(net_every)}
        o["tree"]["leaf_budget_cost"] = 1 + int(num_continuations)
    return o


# -- search runs ------------------------------------------------------------------


@dataclass
class SearchRun:
    name: str
    solver: RangeSolver
    agent: Any
    stats: dict
    seconds: float  # wall time of the act() call


@contextlib.contextmanager
def capture_solvers() -> Iterator[list[RangeSolver]]:
    """Collect every :class:`RangeSolver` that :mod:`pokerbot.search.agent` builds
    (monkeypatches ``pokerbot.search.agent.RangeSolver``, as ``exploit.py``)."""
    import pokerbot.search.agent as agent_mod

    got: list[RangeSolver] = []
    orig = agent_mod.RangeSolver

    class Capture(orig):  # type: ignore[valid-type, misc]
        def __init__(self, *a: Any, **k: Any) -> None:
            super().__init__(*a, **k)
            got.append(self)

    agent_mod.RangeSolver = Capture
    try:
        yield got
    finally:
        agent_mod.RangeSolver = orig


def run_search(
    name: str,
    blueprint: Any,
    state: Any,
    game_config: Any,
    config: SearchConfig | dict,
    value_predictor: Any = None,
) -> SearchRun:
    """One search decision at ``state`` for the player to act (a fresh
    :class:`~pokerbot.search.agent.SearchAgent`, as ``make_agent("search:...")``
    builds it); returns the captured solver, the agent and its ``last_stats``."""
    from ..eval.masking import MaskedState
    from .agent import SearchAgent

    cfg = config if isinstance(config, SearchConfig) else search_config(**config)
    agent = SearchAgent(blueprint, cfg, name=f"search:{name}", value_predictor=value_predictor)
    seat = int(state.current_player)
    with capture_solvers() as got:
        agent.new_hand(seat, game_config)
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        agent.act(MaskedState(state, seat, game_config, rng), seat, rng)
        dt = time.perf_counter() - t0
    if not got:
        raise RuntimeError(f"{name}: the agent built no solver (did it search?)")
    return SearchRun(name, got[-1], agent, dict(agent.last_stats), dt)


# -- trunk identity ---------------------------------------------------------------


def _decisions(tree: SubgameTree) -> dict[tuple, list]:
    ids = (tree.kind == DECISION).nonzero().flatten().tolist()
    return {node_key(tree, n): sorted(tree.child_actions(n)) for n in ids}


def _leaves(tree: SubgameTree) -> set:
    ids = ((tree.kind == LEAF) | (tree.kind == VALUE)).nonzero().flatten().tolist()
    return {node_key(tree, n) for n in ids}


def trunk_differences(a: SubgameTree, b: SubgameTree, limit: int = 5) -> list[str]:
    """Why ``a`` and ``b`` do not share a trunk: decision nodes (by
    :func:`~pokerbot.search.exact_eval.node_key`), their child actions, and the
    depth-limit leaves (``LEAF`` or ``VALUE``). Empty when they do."""
    out: list[str] = []
    da, db = _decisions(a), _decisions(b)
    only_a = [k for k in da if k not in db]
    only_b = [k for k in db if k not in da]
    if only_a or only_b:
        out.append(
            f"decision nodes differ: {len(only_a)} only in the first tree, {len(only_b)} only "
            f"in the second (e.g. {(only_a or only_b)[:limit]})"
        )
    diff = [k for k in da if k in db and da[k] != db[k]]
    if diff:
        out.append(
            f"child actions differ at {len(diff)} nodes, e.g. "
            + "; ".join(f"{k}: {da[k]} vs {db[k]}" for k in diff[:limit])
        )
    la, lb = _leaves(a), _leaves(b)
    if la != lb:
        out.append(f"depth-limit leaves differ: {len(la - lb)} vs {len(lb - la)} unmatched")
    return out


def check_same_trunk(a: SearchRun, b: SearchRun, rtol: float = 1e-5) -> dict:
    """Raise ``AssertionError`` unless ``a`` and ``b`` share a trunk and root
    ranges; returns the check's numbers."""
    diffs = trunk_differences(a.solver.tree, b.solver.tree)
    if diffs:
        raise AssertionError(f"{a.name} and {b.name} trees differ: " + " | ".join(diffs))
    ra, rb = a.solver.ranges, b.solver.ranges.to(a.solver.ranges)
    scale = float(ra.abs().max().clamp(min=1e-30))
    gap = float((ra - rb).abs().max())
    if gap > rtol * scale:
        raise AssertionError(f"{a.name} and {b.name} root ranges differ by {gap:.3g}")
    return {
        "decision_nodes": len(_decisions(a.solver.tree)),
        "leaves": len(_leaves(a.solver.tree)),
        "ranges_max_diff": gap,
    }


# -- blueprint profile --------------------------------------------------------------


def blueprint_profile(solver: RangeSolver, bp: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """The blueprint's strategy on every decision node of ``solver``'s tree,
    ``[D, A, C]``, and the blueprint mass on actions the tree lacks, ``[D, C]``.

    Children are matched to the blueprint's legal options at the node's state by
    concrete action; mass on blueprint actions missing from the tree (sizes the
    node budget dropped) is renormalised over the tree's children (uniform where
    none match), as ``exploit.py`` did. A ``LEAF`` node of a rollout tree plays
    the plain blueprint continuation (index 0)."""
    tree = solver.tree
    dev, dt = solver.device, solver.dtype
    sigma = solver.uniform.clone()
    dropped = torch.zeros(solver.Dn, C, device=dev, dtype=dt)
    for d, node in enumerate(solver.dec_nodes.tolist()):
        kind = int(tree.kind[node])
        if kind == LEAF:
            sigma[d] = 0
            sigma[d, 0] = 1
            continue
        if kind != DECISION:
            continue
        n = int(tree.num_children[node])
        st = tree.states[node]
        P = policy_matrix(bp, st, int(tree.actor[node])).to(dev, dt).expand(C, -1)  # [C, A]
        col = {(o.kind, o.amount): o.index for o in legal_options(st, bp.spec)}
        S = torch.zeros(C, n, device=dev, dtype=dt)
        for j, (k, amt) in enumerate(tree.child_actions(node)):
            i = col.get((int(k), int(amt) if int(k) == RAISE else 0))
            if i is not None:
                S[:, j] = P[:, i]
        tot = S.sum(1, keepdim=True)
        dropped[d] = (1 - tot[:, 0]).clamp(min=0)
        S = torch.where(tot > 0, S / tot.clamp(min=1e-30), torch.full_like(S, 1.0 / n))
        sigma[d] = 0
        sigma[d, :n] = S.t()
    return sigma, dropped


def blueprint_sigma(solver: RangeSolver, bp: Any) -> torch.Tensor:
    """The blueprint's ``[D, A, C]`` strategy on ``solver``'s tree (see
    :func:`blueprint_profile`)."""
    return blueprint_profile(solver, bp)[0]


@torch.no_grad()
def offtree_mass(solver: RangeSolver, sigma: torch.Tensor, dropped: torch.Tensor) -> float:
    """Expected number of decisions per hand, under ``sigma`` on the plain root
    ranges, at which the blueprint would have picked an action the tree lacks
    (``dropped`` from :func:`blueprint_profile`)."""
    tree = solver.tree
    root = solver.ranges
    reach = solver.forward(sigma, root)
    # probability of the chance cards along the path, given both hands
    w = torch.ones(tree.num_nodes, device=solver.device, dtype=solver.dtype)
    cw = tree.chance_weight.to(solver.device, solver.dtype)
    ls = tree.level_start
    for d in range(1, len(ls) - 1):  # depth by depth: parents first
        ids = torch.arange(ls[d], ls[d + 1], device=solver.device)
        w[ids] = w[tree.parent[ids]] * cw[ids]
    nodes = solver.dec_nodes
    actor = tree.actor[nodes]
    own = reach[actor, nodes]  # [D, C]
    opp = blocked_sum(reach[1 - actor, nodes])  # [D, C]
    Z = solver.pair_mass(root).clamp(min=1e-30)
    return float((own * opp * dropped).sum(1).mul(w[nodes]).sum() / Z)


# -- scoring --------------------------------------------------------------------------


def scoring_solver(solver: RangeSolver, max_runouts: int | None = None) -> RangeSolver:
    """A solver on ``solver``'s tree and plain root ranges for
    :func:`~pokerbot.search.exact_eval.trunk_exploitability`, with
    ``max_runouts`` all-in run-outs per board (``None``: the search's) and
    placeholder leaf values (the scorer replaces them)."""
    cfg = solver.cfg if max_runouts is None else replace(solver.cfg, max_runouts=int(max_runouts))
    L = int((solver.tree.kind == VALUE).sum())
    zeros = FixedLeafValues(torch.zeros(2, L, C, device=solver.device, dtype=solver.dtype))
    return RangeSolver(solver.tree, solver.ranges, cfg, value_leaves=zeros)


def score_profile(
    eval_solver: RangeSolver,
    sigma: torch.Tensor,
    river_spec: Any,
    game_config: Any,
    settings: EvalSettings,
    log: Callable[[str], None] | None = None,
    batch: int | None = None,
) -> dict:
    """:func:`~pokerbot.search.exact_eval.trunk_exploitability` of ``sigma`` with
    chips converted to mbb/hand (``1000 * chips / big blind``). ``batch``
    overrides ``settings.river_batch``."""
    batch = settings.river_batch if batch is None else int(batch)
    res = trunk_exploitability(
        eval_solver,
        sigma,
        river_spec,
        game_config,
        iterations=settings.river_iters,
        mix=settings.mix,
        min_mass=settings.min_mass,
        batch=batch,
        log=log,
    )
    mbb = 1000.0 / float(game_config.big_blind)
    return {
        "chips": float(res["exploitability"]),
        "mbb": float(res["exploitability"]) * mbb,
        "br": [float(x) for x in res["br"]],
        "skip_bound": float(res["skip_bound"]),
        "skip_bound_mbb": float(res["skip_bound"]) * mbb,
        "skipped": int(res["skipped"]),
        "instances": int(res["instances"]),
        "leaves": int(res["leaves"]),
        "river_exploit_max": float(res["river_exploit_max"]),
        "river_seconds": float(res["river_seconds"]),
        "seconds": float(res["seconds"]),
        "river_batch": batch,
    }


def _self_exploitability(run: SearchRun, game_config: Any) -> dict:
    t0 = time.perf_counter()
    e = run.solver.exploitability()
    return {
        "chips": float(e["exploitability"]),
        "mbb": 1000.0 * float(e["exploitability"]) / float(game_config.big_blind),
        "br": [float(x) for x in e["br"]],
        "seconds": time.perf_counter() - t0,
    }


def _jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [_jsonable(v) for v in x]
    if isinstance(x, torch.Tensor):
        return _jsonable(x.tolist())
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return x


# -- one spot -----------------------------------------------------------------------


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _score_with_retry(
    batch: list[int], say: Callable[[str], None], *args: Any, **kwargs: Any
) -> dict:
    """:func:`score_profile` with the river batch ``batch[0]``, halved (and kept
    halved in ``batch``) on CUDA out-of-memory. ``trunk_exploitability`` builds
    each batch's river solver while the previous one is alive, so its peak is
    about two batches (about 3 GB each at 256 instances of a 237-node river tree)."""
    while True:
        try:
            return score_profile(*args, batch=batch[0], **kwargs)
        except torch.cuda.OutOfMemoryError:
            if batch[0] <= 16:
                raise
        _free()  # outside the except block, so the failed solver is released
        batch[0] //= 2
        say(f"    CUDA out of memory: retrying with river batch {batch[0]}")


def evaluate_spot(
    spot: Spot,
    blueprint: Any,
    game_config: Any,
    settings: EvalSettings,
    predictor_factory: Callable[[], Any] = lambda: None,
    net_path: str | None = None,
    river_spec: Any = None,
    log: Callable[[str], None] = print,
    keep: dict | None = None,
) -> dict:
    """Everything for one spot (see the module docstring), as a JSON-ready dict.

    ``predictor_factory()`` gives each value-net agent its predictor (``None``:
    the agent loads ``leaf.net = net_path`` itself). ``river_spec`` defaults to
    ``blueprint.spec``. ``keep`` (a dict) receives the runs, the scoring solver
    and the sigmas, for tests."""
    t_spot = time.perf_counter()
    state = spot.state
    river_spec = blueprint.spec if river_spec is None else river_spec
    ro_over = search_overrides(settings, "rollouts")
    k = search_config(**ro_over).tree.num_continuations

    def vn_over(net_every: int = 1, budget: float | None = None, path: Any = net_path) -> dict:
        return search_overrides(settings, "value_net", path, net_every, budget, k)

    def search(name: str, over: dict, vn: bool, own_net: bool = False) -> SearchRun:
        pred = predictor_factory() if vn and not own_net else None
        run = run_search(name, blueprint, state, game_config, over, pred)
        s = run.stats
        log(
            f"  {name}: {s.get('total_seconds', run.seconds):.2f}s "
            f"({s.get('iterations')} iterations, {s.get('nodes')} nodes, "
            f"tree {s.get('tree_seconds', 0):.2f}s, setup {s.get('setup_seconds', 0):.2f}s, "
            f"solve {s.get('solve_seconds', 0):.2f}s)"
        )
        return run

    log(f"{spot.label}: board {list(state.board)}, seat {int(state.current_player)} to act")
    # the value-net search first: its blueprint queries then start cold, as in play
    runs: dict[str, SearchRun] = {"value_net": search("value_net", vn_over(), True)}
    runs["rollout"] = search("rollout", ro_over, False)
    for name in settings.variants:
        v = parse_variant(name)
        if v["turn"] and not settings.turn_net:
            raise ValueError(f"variant {name} needs EvalSettings.turn_net (--turn-net)")
        path = settings.turn_net if v["turn"] else net_path
        over = vn_over(v["net_every"], settings.budget if v["budget"] else None, path)
        runs[name] = search(name, over, True, own_net=v["turn"])
    vrun = runs["value_net"]
    vsolver = vrun.solver
    check = check_same_trunk(vrun, runs["rollout"])
    for name in settings.variants:
        check_same_trunk(vrun, runs[name])
    log(
        f"  trunk check ok: {check['decision_nodes']} decision nodes, {check['leaves']} leaves, "
        f"ranges max diff {check['ranges_max_diff']:.2g}"
    )

    # strategies on the value-net tree
    sigmas: dict[str, torch.Tensor] = {"value_net": vsolver.average_strategy()}
    sigmas["rollout"], _ = map_sigma(runs["rollout"].solver, vsolver, strict=True)
    for name in settings.variants:
        sigmas[name], _ = map_sigma(runs[name].solver, vsolver, strict=True)
    t_bp = time.perf_counter()
    sigmas["blueprint"], dropped = blueprint_profile(vsolver, blueprint)
    t_bp = time.perf_counter() - t_bp

    searches: dict[str, dict] = {}
    for name, run in runs.items():
        searches[name] = {
            **run.stats,
            "act_seconds": run.seconds,
            "self_exploitability": _self_exploitability(run, game_config),
        }
    nodes = {name: int(run.solver.tree.num_nodes) for name, run in runs.items()}
    tree = vsolver.tree
    eval_solver = scoring_solver(vsolver, settings.eval_max_runouts)
    if keep is not None:
        keep.update(runs=runs, eval_solver=eval_solver, sigmas=sigmas, dropped=dropped)
    del runs, vrun, vsolver  # the searches' solvers are not needed for scoring
    _free()
    offtree = offtree_mass(eval_solver, sigmas["blueprint"], dropped)
    profiles: dict[str, dict] = {}
    order = [*BASE_PROFILES, *settings.variants]
    batch = [settings.river_batch]
    for name in order:
        res = _score_with_retry(
            batch,
            log,
            eval_solver,
            sigmas[name],
            river_spec,
            game_config,
            settings,
            log=lambda m, n=name: log(f"    [{n}] {m.lstrip('# ').strip()}"),
        )
        profiles[name] = res
        log(
            f"  {name}: {res['mbb']:.0f} mbb/hand (BR {res['br'][0]:.1f} / {res['br'][1]:.1f} "
            f"chips, {res['instances']} river subgames, {res['skipped']} skipped, "
            f"bound {res['skip_bound_mbb']:.2g} mbb, scored in {res['seconds']:.0f}s)"
        )
    out = {
        "label": spot.label,
        "board_index": spot.board_index,
        "spot_type": spot.spot_type,
        "board": [int(c) for c in state.board],
        "seat": int(state.current_player),
        "hole": [int(c) for c in state.hole_cards(int(state.current_player))],
        "pot": int(tree.contrib[0].sum()),
        "nodes": nodes,
        "leaves": int((tree.kind == VALUE).sum()),
        "decision_nodes": check["decision_nodes"],
        "street_actions": [[list(a) for a in s] for s in tree.street_actions],
        "raise_caps": list(tree.raise_caps),
        "chance_cards": tree.chance_cards_used,
        "trunk_check": {"ok": True, **check},
        "searches": searches,
        "reference_sanity": searches["value_net"]["self_exploitability"],
        "profiles": profiles,
        "blueprint_offtree": offtree,
        "blueprint_seconds": t_bp,
        "seconds": time.perf_counter() - t_spot,
    }
    return _jsonable(out)


# -- driver -------------------------------------------------------------------------


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def run_evaluation(
    spots: Sequence[Spot],
    blueprint: Any,
    game_config: Any,
    settings: EvalSettings,
    meta: dict,
    out_json: str | Path,
    out_md: str | Path | None = None,
    predictor_factory: Callable[[], Any] = lambda: None,
    net_path: str | None = None,
    resume: bool = False,
    warmup: Spot | None = None,
    log: Callable[[str], None] = print,
) -> dict:
    """Evaluate ``spots`` in order, rewriting ``out_json`` (and ``out_md``) after
    every spot. With ``resume`` an existing ``out_json`` written with the same
    comparable settings and meta keys ``blueprint`` / ``leaf_model`` keeps its
    finished spots, and only the others run. A failing spot is logged and
    recorded under ``errors`` (and retried on resume). ``warmup`` (a spot) is
    searched once with a few iterations first so CUDA start-up costs stay out
    of the timings."""
    out_json = Path(out_json)
    data = {
        "settings": settings.comparable(),
        "meta": meta,
        "spots": [],
        "errors": [],
    }
    if resume and out_json.exists():
        old = json.loads(out_json.read_text())
        same = old.get("settings") == data["settings"] and all(
            old.get("meta", {}).get(k) == meta.get(k) for k in ("blueprint", "leaf_model")
        )
        if not same:
            raise ValueError(
                f"{out_json} was written with other settings; "
                "use another --out-json or drop --resume"
            )
        data["spots"] = list(old.get("spots", []))
        data["meta"] = {**old.get("meta", {}), **meta, "resumed": True}
    done = {s["label"] for s in data["spots"]}
    todo = [s for s in spots if s.label not in done]
    if done:
        log(f"resuming: {len(done)} spots done, {len(todo)} to go")
    if warmup is not None and todo:
        t0 = time.perf_counter()
        warm = replace(settings, iters=2, variants=())
        k = search_config(**search_overrides(warm)).tree.num_continuations
        over = search_overrides(warm, "value_net", net_path, 1, None, k)
        run_search("warmup", blueprint, warmup.state, game_config, over, predictor_factory())
        _free()
        log(f"warm-up search: {time.perf_counter() - t0:.1f}s")
    t_all = time.perf_counter()
    for i, spot in enumerate(todo):
        try:
            res = evaluate_spot(
                spot,
                blueprint,
                game_config,
                settings,
                predictor_factory,
                net_path,
                log=log,
            )
            data["spots"].append(res)
            data["errors"] = [e for e in data["errors"] if e["label"] != spot.label]
            log(
                f"{spot.label} done in {res['seconds']:.0f}s "
                f"({i + 1}/{len(todo)}, {time.perf_counter() - t_all:.0f}s so far)"
            )
        except Exception as exc:  # noqa: BLE001 - keep going, record it
            log(f"{spot.label} FAILED: {exc!r}")
            data["errors"].append(
                {"label": spot.label, "error": repr(exc), "traceback": traceback.format_exc()}
            )
        _free()
        data["meta"]["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        order = {s.label: j for j, s in enumerate(spots)}
        data["spots"].sort(key=lambda s: order.get(s["label"], len(order)))
        _write_json(out_json, data)
        if out_md is not None:
            _write_text(Path(out_md), render_markdown(data))
    return data


# -- markdown -----------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _f(x: Any, nd: int = 0) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    return f"{x:.{nd}f}"


def _columns(spots: Sequence[dict]) -> list[str]:
    cols = list(BASE_PROFILES)
    for s in spots:
        for name in s["profiles"]:
            if name not in cols:
                cols.append(name)
    return cols


def render_markdown(data: dict) -> str:
    """The report: exploitability per spot (mbb/hand) with means overall and per
    spot type, timings per search, and the scoring details."""
    st = data["settings"]
    meta = data.get("meta", {})
    spots = data["spots"]
    cols = _columns(spots)
    lines = ["# Flop search exploitability with an exact river", ""]
    lines.append(
        f"Blueprint `{meta.get('blueprint')}`, leaf model `{meta.get('leaf_model')}`, "
        f"device {meta.get('device')}{', GPU ' + meta['gpu'] if meta.get('gpu') else ''}."
    )
    runouts = st.get("eval_max_runouts")
    allin = "the search's run-outs" if runouts is None else f"up to {runouts} run-outs per board"
    lines.append(
        f"Searches: {st['iters']} iterations, trees <= {st['max_nodes']} nodes "
        f"(value-net trees count a leaf as 1 + k nodes, so all trunks are equal), "
        f"rollout leaf seed {st['leaf_seed']}. Scoring: every (leaf, river card) river "
        f"subgame solved exactly ({st['river_iters']} DCFR iterations, {st['mix']:g} uniform "
        f"mixed into both river ranges), best responses backed up through the trunk on the "
        f"plain root ranges; all-ins over {allin}. "
        f"Exploitability = (BR_0 + BR_1) / 2 in mbb/hand (1 bb = 100 chips)."
    )
    if any(v == "value_net_budget" for v in st.get("variants", [])):
        lines.append(
            f"`value_net_budget`: value-net search under a {st['budget']:g}s budget "
            "(default iteration settings) on the same tree."
        )
    lines += ["", "## Exploitability (mbb/hand)", ""]
    head = ["spot", "pot", "leaves"] + [PROFILE_NAMES.get(c, c) for c in cols]
    head.append("value-net self (net's game)")
    lines.append("| " + " | ".join(head) + " |")
    lines.append("|---|" + "---:|" * (len(head) - 1))

    def prof(s: dict, c: str) -> float | None:
        p = s["profiles"].get(c)
        return None if p is None else p["mbb"]

    for s in spots:
        row = [s["label"], str(s["pot"]), str(s["leaves"])]
        row += [_f(prof(s, c)) for c in cols]
        row.append(_f(s["reference_sanity"]["mbb"]))
        lines.append("| " + " | ".join(row) + " |")
    groups = [("**mean**", spots)]
    for t in SPOT_TYPES:
        g = [s for s in spots if s["spot_type"] == t]
        if g and len(g) < len(spots):
            groups.append((f"mean {SPOT_NAMES[t]} ({len(g)})", g))
    for name, g in groups:
        row = [name, "", ""]
        row += [_f(_mean([prof(s, c) for s in g])) for c in cols]
        row.append(_f(_mean([s["reference_sanity"]["mbb"] for s in g])))
        lines.append("| " + " | ".join(row) + " |")
    if spots:
        bpo = _mean([s.get("blueprint_offtree") for s in spots])
        lines += [
            "",
            f"Blueprint column: the blueprint's strategy on the trunk, with its mass on "
            f"sizes the tree lacks renormalised over the tree's actions (on average "
            f"{_f(bpo, 3)} such decisions per hand). Value-net self: the value-net search's "
            f"exploitability in its own game (as judged by its leaf model).",
        ]

    # timing
    lines += ["", "## Timing per decision (seconds)", ""]
    th = [
        "spot",
        "search",
        "total",
        "tree",
        "setup",
        "leaf",
        "solve",
        "iterations",
        "ms/iter",
        "nodes",
        "value leaves",
    ]
    lines.append("| " + " | ".join(th) + " |")
    lines.append("|---|---|" + "---:|" * (len(th) - 2))
    names: list[str] = []
    for s in spots:
        for name in s["searches"]:
            if name not in names:
                names.append(name)

    def trow(label: str, name: str, xs: Sequence[dict]) -> str:
        def m(key: str) -> float | None:
            return _mean([x.get(key) for x in xs])

        it = m("iterations")
        solve = m("solve_seconds")
        ms = None if not it or solve is None else 1000 * solve / it
        cells = [
            label,
            name,
            _f(m("total_seconds"), 2),
            _f(m("tree_seconds"), 2),
            _f(m("setup_seconds"), 2),
            _f(m("leaf_seconds"), 2),
            _f(solve, 2),
            _f(it, 0),
            _f(ms, 1),
            _f(m("nodes"), 0),
            _f(m("value_leaves"), 0),
        ]
        return "| " + " | ".join(cells) + " |"

    for s in spots:
        for name in names:
            if name in s["searches"]:
                lines.append(trow(s["label"], name, [s["searches"][name]]))
    for name in names:
        xs = [s["searches"][name] for s in spots if name in s["searches"]]
        if xs:
            lines.append(trow("**mean**", name, xs))
    lines += [
        "",
        "total = the agent's whole decision (tree, gadget terminate values, leaf setup, "
        "solve); setup includes leaf (rollouts or the value-net evaluator) and the gadget.",
    ]

    # scoring details
    lines += ["", "## Scoring details", ""]
    dh = [
        "spot",
        "profile",
        "mbb/hand",
        "BR0 (chips)",
        "BR1 (chips)",
        "river subgames",
        "skipped",
        "skip bound (mbb)",
        "max river expl. (% pot)",
        "scoring s",
    ]
    lines.append("| " + " | ".join(dh) + " |")
    lines.append("|---|---|" + "---:|" * (len(dh) - 2))
    for s in spots:
        for c in cols:
            p = s["profiles"].get(c)
            if p is None:
                continue
            cells = [
                s["label"],
                PROFILE_NAMES.get(c, c),
                _f(p["mbb"]),
                _f(p["br"][0], 1),
                _f(p["br"][1], 1),
                str(p["instances"]),
                str(p["skipped"]),
                f"{p['skip_bound_mbb']:.2g}",
                _f(100 * p["river_exploit_max"], 2),
                _f(p["seconds"], 0),
            ]
            lines.append("| " + " | ".join(cells) + " |")
    total = sum(s["seconds"] for s in spots)
    lines += ["", f"{len(spots)} spots, {total / 60:.1f} min of evaluation."]
    if data.get("errors"):
        lines += ["", "## Errors", ""]
        for e in data["errors"]:
            lines.append(f"* {e['label']}: `{e['error']}`")
    return "\n".join(lines) + "\n"


__all__ = [
    "EXACT_FLOP_RUNOUTS",
    "EvalSettings",
    "SPOT_TYPES",
    "SearchRun",
    "Spot",
    "blueprint_profile",
    "blueprint_sigma",
    "capture_solvers",
    "check_same_trunk",
    "evaluate_spot",
    "exploit_spots",
    "offtree_mass",
    "parse_variant",
    "render_markdown",
    "run_evaluation",
    "run_search",
    "score_profile",
    "scoring_solver",
    "search_overrides",
    "trunk_differences",
]
