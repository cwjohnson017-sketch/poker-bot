"""Flop (and turn) tree size under a fixed time budget, scored with an exact river.

The question: under the production flop budget (4 s), do bigger flop trees
(more bet sizes, fewer DCFR iterations) give a less exploitable strategy than
the production 6,000-node trees? More generally, which of several named search
**variants** plays the better flop: each a node budget and search-config
overrides of its own (:class:`Variant`), e.g. the production 20,000-node tree
next to ``tree.keep_open`` trees of 20,000 and 34,170 nodes. The same for turn
decisions (``SizeSettings.street = 2``; see "Turn decisions" below).

For one flop decision (a :class:`~pokerbot.search.spot_eval.Spot`),
:func:`evaluate_sizes`

1. runs the production value-net search (``configs/search_value_net.yaml``:
   value-net leaves at the end of the turn, each counting one node) once per
   budget variant (``SizeSettings.variants`` in their order; without them one
   per ``max_nodes`` in ``SizeSettings.sizes``, named by it, smallest first),
   under the ``budget`` flop time budget with the config's iteration cap and
   ``min_iterations``, so the iteration count depends on the tree; with
   ``iters`` also once per variant for that fixed number of iterations
   (``<name>_it<iters>``, a reference without the time limit). A variant's
   overrides go over ``SizeSettings.search``; its ``max_nodes`` (as
   ``tree.max_nodes`` and ``tree.max_nodes_flop``), the time budget and the
   leaf settings go over both, one level deep (:func:`size_overrides`: a
   variant's ``leaf.turn_net`` stays);
2. takes one budget search's tree as the scoring trunk (``SizeSettings.trunk``,
   default the last variant: without variants, the largest size) and puts
   every search's average strategy on it with
   :func:`~pokerbot.search.tree_map.translate_sigma`. Where the histories
   agree this is exact. Opponent sizes that a smaller tree lacks map onto its
   sizes by the deterministic pseudo-harmonic rule, and unmatched nodes (none
   in the production trees) play the blueprint. The blueprint is mapped with
   :func:`~pokerbot.search.spot_eval.blueprint_profile`;
3. **re-searches** as the agent does in play (``research``, budget searches
   but the trunk's): at every first off-tree opponent flop edge of the trunk
   (the opponent picks a size the search's tree lacks, after a history that
   tree has exactly; :func:`~pokerbot.search.tree_map.offtree_edges`), the
   state is replayed to that point and the SAME agent acts again. Its tree is
   rooted at the flop with the observed path forced in, so the size is an
   extra branch; the root info (ranges, gadget terminate values) comes from
   the agent's cache, and the searcher's earlier flop decisions on the path
   are locked to what it played (the first search's root strategy, and for
   any later decisions on the path the first search's average strategy there,
   seeded into ``SearchAgent._played``). The ``<name>_research`` profile takes
   each re-search's strategy below its edge (translated by the same walk,
   exact along the path) and the first search's translated strategy
   everywhere else;
4. with ``resolve_iters``, **re-solves** every budget variant's final profile
   (``<name>_research`` where it has re-searches, else the translated one; the
   trunk search's own strategy for the trunk) below the flop
   (:func:`resolve_below_root`): on the trunk, every flop decision of both
   players is locked to the profile and every later decision is solved again
   for ``resolve_iters`` DCFR iterations with the trunk search's own leaf
   model (``SearchAgent.value_leaf_provider``) and solver settings, on its
   plain root ranges (unsafe: no gadget). In play, separate turn searches
   replace a flop search's turn strategy, which is only a plan; the
   ``<name>_resolved`` profiles differ only in their flop strategies. Where a
   profile never takes a flop action (a size its tree lacks), nothing reaches
   the turn below it, and the other player's turn decisions there would keep
   the uniform strategy: ``resolve_mix`` > 0 locks the flop to ``(1 - mix) *
   profile + mix * uniform`` while solving (the result still plays the
   profile on the flop), so those lines are solved as well;
5. scores every profile with :func:`~pokerbot.search.spot_eval.score_profile`.
   Every (leaf, river card) river subgame is solved exactly, and best
   responses are backed up through the trunk on the plain root ranges. The
   **one-sided** number is the best response of the searcher's opponent (the
   searcher is the player to act); its differences between profiles are exact
   differences in the searcher's exploitability in the trunk's game. The
   **two-sided** number is ``(BR_0 + BR_1) / 2``. A ``_research`` profile with
   no re-search (the trunk's own, or one whose tree has all the trunk's
   opponent flop sizes) is the translated one and is not scored twice. With
   ``score_translated=False`` a translated profile that has a re-searched one
   (with re-searches) is not scored either (``{"not_scored": true}``); the
   trunk's always is.

The trunk's game gives the opponent every size of the trunk's tree. The
translated profiles charge a smaller tree for answering a size it lacks with
its strategy for the nearest size it has; the re-searched profiles answer it
as the agent would. Only off-tree edges are re-searched: the agent re-searches
at its on-tree decisions too, for every size, which this leaves out.

**Turn decisions** (``street=2``; spots from :func:`select_turn_spots`, after
the flop lines of :data:`~pokerbot.search.spot_eval.TURN_LINES`). Searches are
rooted at the turn with the turn time budget: solved to showdown by default,
or with ``tree.depth_streets_turn: 0`` ending at value-net leaves at the end of
turn betting (valued by ``leaf.turn_net`` when set, else ``leaf.net``). The
trunk must be such a depth-0 search's tree (:func:`trunk_problem`; checked
before any search runs), so every (leaf, river card) river subgame is solved
exactly as on the flop. A search solved to showdown is scored on its turn
strategy alone: its turn decisions match the trunk's by history (where it
deals the river the trunk has a leaf, which is never matched, and its river
decisions have no trunk node), and its river plan gives way to the exact river
equilibrium, as river searches replace it in play. Re-searches happen at
off-tree opponent turn actions; the subset check compares the trunk's betting
streets only; there is nothing to re-solve below a turn trunk
(``resolve_iters`` is refused).

Before the measured searches of a spot, a short search on it (the smallest
tree's variant, :func:`warmup_config`) warms the blueprint's caches
(``spot_warmup``), so every variant starts from the same cache state.

:func:`run_size_evaluation` drives a list of spots, rewrites the JSON after
every spot (resumable) and renders the markdown report
(:func:`render_markdown`). The CLI is ``scripts/eval_tree_size.py``.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import REPO_ROOT
from ..engine_select import get_engine
from .abstract import make_state, to_action
from .config import SearchConfig, search_config
from .gadget import history_key
from .solver import RangeSolver, SolverConfig
from .spot_eval import (
    EXACT_FLOP_RUNOUTS,
    HUGE_BUDGET,
    SPOT_NAMES,
    SPOT_TYPES,
    TURN_LINES,
    EvalSettings,
    SearchRun,
    Spot,
    _f,
    _free,
    _jsonable,
    _mean,
    _merge,
    _score_with_retry,
    _self_exploitability,
    _write_json,
    _write_text,
    blueprint_profile,
    capture_solvers,
    exploit_spots,
    offtree_mass,
    run_search,
    scoring_solver,
    turn_spots,
)
from .tree import DECISION, VALUE, SubgameTree
from .tree_map import (
    NodeMatch,
    StrategySnapshot,
    action_set_differences,
    match_nodes,
    offtree_edges,
    path_nodes,
    subtree_nodes,
    translate_sigma,
)

DEFAULT_CONFIG = "configs/search_value_net.yaml"
DEFAULT_SIZES = (6000, 10000, 20000)

RESEARCH = "_research"
RESOLVED = "_resolved"
STREET_NAMES = ("preflop", "flop", "turn", "river")
STREETS = {"flop": 1, "turn": 2}  # the streets whose decisions can be evaluated


# -- settings ---------------------------------------------------------------------


@dataclass
class Variant:
    """One budget search: its name (a column of the report), its node budget
    (``tree.max_nodes`` and ``tree.max_nodes_flop``) and ``search_config``
    overrides of its own, merged over ``SizeSettings.search`` (see
    :func:`size_overrides`)."""

    name: str
    max_nodes: int
    search: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.max_nodes = int(self.max_nodes)
        self.search = dict(self.search or {})
        n = self.name
        if not n or n == "blueprint" or "_it" in n or n.endswith((RESEARCH, RESOLVED)):
            raise ValueError(
                f"bad variant name {n!r}: it must be non-empty, not 'blueprint', without "
                f"'_it' and not end with {RESEARCH!r} or {RESOLVED!r}"
            )
        if self.max_nodes <= 0:
            raise ValueError(f"variant {n!r}: max_nodes must be positive, got {self.max_nodes}")


@dataclass
class SizeSettings:
    sizes: tuple[int, ...] = DEFAULT_SIZES  # tree.max_nodes per search (without variants)
    budget: float = 4.0  # time budget of the evaluated street (seconds per decision)
    iters: int | None = None  # also a fixed-iteration search per variant
    config: str = DEFAULT_CONFIG  # base search config (YAML)
    seed: int = 0  # search seed (gadget rollouts)
    river_iters: int = 200  # BatchRiverSolver iterations per river subgame
    mix: float = 0.05
    min_mass: float = 1e-7
    river_batch: int = 256  # halved automatically on CUDA out-of-memory
    eval_max_runouts: int | None = EXACT_FLOP_RUNOUTS  # None: the search's solver.max_runouts
    strict: bool = False  # raise when a search's strategy loses mass on the trunk
    research: bool = True  # re-search at off-tree opponent actions (budget searches)
    spot_warmup: bool = True  # warm the blueprint caches on each spot first
    device: str = "auto"
    search: dict = field(default_factory=dict)  # extra search_config overrides (every search)
    # named budget searches in report order (Variant or its dict); empty: one per size,
    # named str(size), smallest first. Given, they replace sizes (set to their max_nodes)
    variants: tuple[Variant, ...] = ()
    trunk: str | None = None  # the variant whose tree is the scoring trunk (None: the last)
    score_translated: bool = True  # False: skip translated profiles that have re-searches
    resolve_iters: int = 0  # > 0: also <name>_resolved profiles (re-solved below the flop)
    resolve_mix: float = 0.0  # uniform mixed into the flop locks while re-solving
    street: int = 1  # the decisions evaluated: 1 flop, 2 turn (turn spots, a depth-0 trunk)

    def __post_init__(self) -> None:
        self.variants = tuple(
            v if isinstance(v, Variant) else Variant(**v) for v in (self.variants or ())
        )
        if self.variants:
            names = [v.name for v in self.variants]
            dup = sorted({n for n in names if names.count(n) > 1})
            if dup:
                raise ValueError(f"duplicate variant names {dup}")
            self.sizes = tuple(v.max_nodes for v in self.variants)
        sizes = sorted({int(s) for s in self.sizes})
        if not sizes:
            raise ValueError("SizeSettings.sizes is empty")
        self.sizes = tuple(sizes)
        if self.trunk is not None:
            self.trunk = str(self.trunk)
            names = [v.name for v in self.budget_variants()]
            if self.trunk not in names:
                raise ValueError(f"trunk {self.trunk!r} is not a variant (one of {names})")
        self.resolve_iters = int(self.resolve_iters or 0)
        if self.resolve_iters < 0:
            raise ValueError(f"resolve_iters must be >= 0, got {self.resolve_iters}")
        self.resolve_mix = float(self.resolve_mix or 0.0)
        if not 0.0 <= self.resolve_mix < 1.0:
            raise ValueError(f"resolve_mix must be in [0, 1), got {self.resolve_mix}")
        self.street = int(self.street)
        if self.street not in STREETS.values():
            raise ValueError(f"street must be 1 (flop) or 2 (turn), got {self.street}")
        if self.street == 2 and self.resolve_iters:
            raise ValueError(
                "resolve_iters needs decisions below the root street, and a turn trunk (leaves "
                "at the end of the turn) has none: drop resolve_iters for turn evaluations"
            )

    def budget_variants(self) -> tuple[Variant, ...]:
        """The budget searches in report order: ``variants``, or one per size
        (named ``str(size)``, no overrides), smallest first."""
        if self.variants:
            return self.variants
        return tuple(Variant(search_name(s), s) for s in self.sizes)

    def trunk_name(self) -> str:
        """The name of the variant whose tree is the scoring trunk."""
        return self.trunk if self.trunk is not None else self.budget_variants()[-1].name

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def comparable(self) -> dict:
        """The settings that change the numbers (for resuming). The variant
        options are left out at their defaults, so a run without them reads as
        before they existed."""
        d = asdict(self)
        for k in ("river_batch", "device"):
            d.pop(k)
        d["sizes"] = list(d["sizes"])
        d["variants"] = [dict(v) for v in d["variants"]]
        defaults = {
            "variants": [],
            "trunk": None,
            "score_translated": True,
            "resolve_iters": 0,
            "resolve_mix": 0.0,
            "street": 1,
        }
        for k, v in defaults.items():
            if d[k] == v:
                d.pop(k)
        return d

    def eval_settings(self) -> EvalSettings:
        """The scoring settings for :func:`~pokerbot.search.spot_eval.score_profile`."""
        return replace(
            EvalSettings(),
            river_iters=self.river_iters,
            mix=self.mix,
            min_mass=self.min_mass,
            river_batch=self.river_batch,
            eval_max_runouts=self.eval_max_runouts,
            device=self.device,
        )


def config_path(path: str | Path) -> Path:
    """``path`` as given if it exists, else relative to the repository root."""
    p = Path(path)
    if p.exists():
        return p
    q = REPO_ROOT / p
    if q.exists():
        return q
    raise FileNotFoundError(f"search config {path} not found (also tried {q})")


def search_name(size: int | str, iters: int | None = None) -> str:
    """``"6000"`` for a budget search, ``"6000_it300"`` for a fixed-iteration one
    (``size`` may be a variant's name: ``"open34k_it300"``)."""
    base = size if isinstance(size, str) else f"{int(size)}"
    return base if iters is None else f"{base}_it{int(iters)}"


def search_runs(settings: SizeSettings) -> list[tuple[str, Variant, int | None]]:
    """``(name, variant, fixed iterations or None)`` of every search: the budget
    searches in variant order, then the fixed-iteration references."""
    vs = settings.budget_variants()
    out: list[tuple[str, Variant, int | None]] = [(v.name, v, None) for v in vs]
    if settings.iters:
        it = int(settings.iters)
        out += [(search_name(v.name, it), v, it) for v in vs]
    return out


def search_plan(settings: SizeSettings) -> list[tuple[str, int, int | None]]:
    """``(name, max_nodes, fixed iterations or None)`` of every search, budget
    searches first, in variant order (without variants: smallest tree first)."""
    return [(name, v.max_nodes, iters) for name, v, iters in search_runs(settings)]


def size_overrides(
    settings: SizeSettings,
    max_nodes: int,
    net_path: str | None = None,
    iters: int | None = None,
    extra: dict | None = None,
) -> dict:
    """``search_config`` overrides of one search over ``settings.config``:
    ``settings.search`` first, then ``extra`` (a variant's own overrides), then
    this run's ``tree.max_nodes`` (and ``max_nodes_flop``), its time budget on
    ``settings.street`` (or ``iters`` fixed iterations under a huge budget),
    value-net leaves and ``leaf.net = net_path`` when given. Sections merge one
    level deep, so e.g. a variant's ``leaf.turn_net`` stays."""
    o: dict[str, Any] = {
        "device": str(settings.torch_device()),
        "fallback_on_error": False,
        "seed": settings.seed,
    }
    o = _merge(o, settings.search)
    o = _merge(o, extra)
    o = _merge(o, {"tree": {"max_nodes": int(max_nodes), "max_nodes_flop": int(max_nodes)}})
    if iters is None:
        street = STREET_NAMES[settings.street]
        o = _merge(o, {"time_budget": {street: float(settings.budget)}})
    else:
        huge = {"flop": HUGE_BUDGET, "turn": HUGE_BUDGET, "river": HUGE_BUDGET}
        o = _merge(o, {"min_iterations": int(iters), "time_budget": huge})
        o = _merge(o, {"solver": {"iterations": int(iters)}})
    o = _merge(o, {"leaf": {"mode": "value_net", **({"net": net_path} if net_path else {})}})
    return o


def size_config(
    settings: SizeSettings,
    max_nodes: int,
    net_path: str | None = None,
    iters: int | None = None,
    extra: dict | None = None,
) -> SearchConfig:
    """A fresh :class:`SearchConfig` for one search (see :func:`size_overrides`)."""
    over = size_overrides(settings, max_nodes, net_path, iters, extra)
    return search_config(config_path(settings.config), **over)


def variant_config(
    settings: SizeSettings, variant: Variant, net_path: str | None = None, iters: int | None = None
) -> SearchConfig:
    """:func:`size_config` of ``variant``: its node budget and overrides."""
    return size_config(settings, variant.max_nodes, net_path, iters, variant.search)


def warmup_config(settings: SizeSettings, net_path: str | None = None) -> SearchConfig:
    """A two-iteration search of the budget variant with the smallest tree (the
    first such), for the warm-up searches."""
    v = min(settings.budget_variants(), key=lambda v: v.max_nodes)
    return variant_config(settings, v, net_path, iters=2)


def variant_from_arg(text: str) -> Variant:
    """A :class:`Variant` from ``NAME=MAX_NODES`` or ``NAME=MAX_NODES:JSON``
    (``JSON`` a search-override dict, e.g. ``open=34170:{"tree": {"keep_open":
    true}}``), as the CLI's ``--variant`` takes it."""
    name, eq, rest = str(text).partition("=")
    nodes, colon, js = rest.partition(":")
    if not eq or not name.strip() or not nodes.strip().isdigit():
        raise ValueError(f"bad variant {text!r}: expected NAME=MAX_NODES or NAME=MAX_NODES:JSON")
    over = {}
    if colon:
        try:
            over = json.loads(js)
        except json.JSONDecodeError as exc:
            raise ValueError(f"bad variant {text!r}: its overrides are not JSON ({exc})") from exc
        if not isinstance(over, dict):
            raise ValueError(f"bad variant {text!r}: its overrides must be a JSON object")
    return Variant(name.strip(), int(nodes), over)


def select_spots(
    engine: Any,
    game_config: Any,
    boards: int = 1,
    seed: int = 5,
    types: Sequence[str] = SPOT_TYPES,
    picks: Sequence[str] | None = None,
    street: int = 1,
    lines: Sequence[str] = tuple(TURN_LINES),
) -> list[Spot]:
    """:func:`~pokerbot.search.spot_eval.exploit_spots` (``boards`` x ``types``),
    or with ``picks`` exactly those spots in that order, each ``"<board>:<type>"``
    (e.g. ``"0:bb_first"``, ``"1:btn_vs_check"``) on the same board sequence.
    ``street=2``: the turn spots of :func:`select_turn_spots` instead."""
    if street == 2:
        return select_turn_spots(engine, game_config, boards, seed, lines, types, picks)
    if street != 1:
        raise ValueError(f"street must be 1 (flop) or 2 (turn), got {street}")
    if not picks:
        return exploit_spots(engine, game_config, boards, seed, types)
    want = []
    for p in picks:
        b, _, t = str(p).partition(":")
        if not b.isdigit() or t not in SPOT_TYPES:
            raise ValueError(f"bad spot {p!r}: expected <board>:<type>, type one of {SPOT_TYPES}")
        want.append((int(b), t))
    every = exploit_spots(engine, game_config, max(b for b, _ in want) + 1, seed, SPOT_TYPES)
    by = {(s.board_index, s.spot_type): s for s in every}
    return [by[w] for w in want]


def spot_line(label: str) -> str | None:
    """The flop line of a turn spot's label (``"board0 xbc BTN vs check"`` ->
    ``"xbc"``; :data:`~pokerbot.search.spot_eval.TURN_LINES`), else ``None``."""
    parts = str(label).split()
    return parts[1] if len(parts) > 2 and parts[1] in TURN_LINES else None


def select_turn_spots(
    engine: Any,
    game_config: Any,
    boards: int = 1,
    seed: int = 5,
    lines: Sequence[str] = tuple(TURN_LINES),
    types: Sequence[str] = SPOT_TYPES,
    picks: Sequence[str] | None = None,
) -> list[Spot]:
    """:func:`~pokerbot.search.spot_eval.turn_spots` (``boards`` x ``lines`` x
    ``types``), or with ``picks`` exactly those spots in that order, each
    ``"<board>:<line>:<type>"`` (e.g. ``"0:xbc:btn_vs_check"``) on the same
    board sequence."""
    if not picks:
        return turn_spots(engine, game_config, boards, seed, lines, types)
    want = []
    for p in picks:
        parts = str(p).split(":")
        b, line, t = (parts + ["", ""])[:3]
        if len(parts) != 3 or not b.isdigit() or line not in TURN_LINES or t not in SPOT_TYPES:
            raise ValueError(
                f"bad turn spot {p!r}: expected <board>:<line>:<type>, line one of "
                f"{tuple(TURN_LINES)}, type one of {SPOT_TYPES}"
            )
        want.append((int(b), line, t))
    every = turn_spots(engine, game_config, max(w[0] for w in want) + 1, seed)
    by = {(s.board_index, spot_line(s.label), s.spot_type): s for s in every}
    return [by[w] for w in want]


def spot_street_errors(spots: Sequence[Spot], street: int) -> list[str]:
    """Labels of ``spots`` whose state is not on ``street`` (spots without a
    state are not checked)."""
    return [s.label for s in spots if s.state is not None and int(s.state.street) != street]


def trunk_problem(settings: SizeSettings) -> str | None:
    """Why the trunk variant cannot be scored exactly on ``settings.street``,
    from its config (``None`` when it can): turn evaluations need turn trees
    that end at value-net leaves at the end of the turn (``tree.depth_streets_turn:
    0``, or ``depth_streets: 0`` without it), since
    :func:`~pokerbot.search.exact_eval.trunk_exploitability` solves every river
    exactly from the trunk's turn-end leaves. Flop trunks are not checked."""
    if settings.street != 2:
        return None
    name = settings.trunk_name()
    v = next(v for v in settings.budget_variants() if v.name == name)
    tc = variant_config(settings, v).tree
    depth = tc.depth_streets if tc.depth_streets_turn is None else tc.depth_streets_turn
    if int(depth) <= 0:  # as TreeBuilder: turn trees with depth >= 1 are solved to showdown
        return None
    return (
        f"the scoring trunk {name!r} solves turn trees to showdown, but turn spots are scored "
        "on a trunk with value-net leaves at the end of the turn (every river solved exactly): "
        'use a variant with {"tree": {"depth_streets_turn": 0}} as the trunk'
    )


def check_settings(settings: SizeSettings, spots: Sequence[Spot] = ()) -> None:
    """``ValueError`` unless every spot is on ``settings.street`` and the trunk
    can be scored there (:func:`trunk_problem`)."""
    bad = spot_street_errors(spots, settings.street)
    if bad:
        raise ValueError(
            f"spots {bad} are not {STREET_NAMES[settings.street]} decisions "
            f"(settings.street {settings.street})"
        )
    problem = trunk_problem(settings)
    if problem:
        raise ValueError(problem)


# -- re-searches at off-tree opponent actions --------------------------------------


def research_name(name: str) -> str:
    """``"6000"`` -> ``"6000_research"``: the profile with the re-searches."""
    return f"{name}{RESEARCH}"


def resolved_name(name: str) -> str:
    """``"6000"`` -> ``"6000_resolved"``: the profile re-solved below the flop."""
    return f"{name}{RESOLVED}"


def street_root_state(engine: Any, game_config: Any, state: Any) -> Any:
    """``state`` at the start of its street: the same button, hole cards, board
    and earlier streets, with none of this street's actions."""
    street = int(state.street)
    pre = [h for h in state.history if int(h[0]) < street]
    holes = [list(state.hole_cards(0)), list(state.hole_cards(1))]
    return make_state(engine, game_config, state.button, list(state.board), pre, holes)


def replay(engine: Any, game_config: Any, state: Any, history: Sequence) -> Any:
    """``state``'s street start with ``history`` applied: ``(street, player, kind,
    amount)`` steps from the street root, as in ``SubgameTree.histories``."""
    s = street_root_state(engine, game_config, state)
    for _street, player, kind, amount in history:
        if int(s.current_player) != int(player):
            raise ValueError(f"history {history} does not fit the state (player {player})")
        s.apply(to_action(engine, int(kind), int(amount)))
    return s


def played_key(state: Any) -> tuple:
    """The key of ``SearchAgent._roots`` and ``_played`` for ``state``'s street."""
    street = int(state.street)
    pre = [h for h in state.history if int(h[0]) < street]
    return (history_key(pre), tuple(int(c) for c in state.board))


def act_again(name: str, agent: Any, state: Any, game_config: Any) -> SearchRun:
    """Another decision of ``agent`` in the same hand, at ``state``, without
    ``new_hand`` (its root info, played strategies and caches carry over), with
    the solver captured."""
    from ..eval.masking import MaskedState

    seat = int(state.current_player)
    with capture_solvers() as got:
        rng = np.random.default_rng(0)
        t0 = time.perf_counter()
        agent.act(MaskedState(state, seat, game_config, rng), seat, rng)
        dt = time.perf_counter() - t0
    if not got:
        raise RuntimeError(f"{name}: the agent built no solver (did it search?)")
    return SearchRun(name, got[-1], agent, dict(agent.last_stats), dt)


def seed_played(
    agent: Any,
    key: tuple,
    first: StrategySnapshot,
    match: NodeMatch,
    trunk: SubgameTree,
    node: int,
    seat: int,
) -> int:
    """Put the first search's average strategy into ``agent._played`` at every
    decision of ``seat`` on the root street on the path to trunk node ``node``
    that it has not played yet (only the first search's own decision is there),
    so the re-search locks them to it. The path must be in ``first``'s tree
    exactly. Returns how many were added."""
    s_index = {n: d for d, n in enumerate(first.dec_nodes.tolist())}
    added = 0
    for q in path_nodes(trunk, node):
        if int(trunk.kind[q]) != DECISION or int(trunk.actor[q]) != seat:
            continue
        if int(trunk.street[q]) != trunk.root_street:
            continue
        k = (key, trunk.histories[q])
        if k in agent._played:
            continue
        s = match.src_of[q]
        if s < 0 or match.translated[q]:
            raise ValueError(f"trunk node {q} is not in the first search's tree exactly")
        n = int(first.tree.num_children[s])
        strat = first.sigma[s_index[s], :n].t().contiguous().cpu()
        agent._played[k] = (first.tree.child_actions(s), strat)
        added += 1
    return added


@dataclass
class Research:
    edge: int  # the trunk node the off-tree opponent action leads to
    history: tuple  # its history from the street root (ending with that action)
    stats: dict  # the agent's last_stats
    seconds: float  # wall time of act()
    snapshot: StrategySnapshot
    seeded: int  # played strategies seeded into the agent for its locks


def research_offtree(
    name: str,
    agent: Any,
    first: StrategySnapshot,
    trunk: SubgameTree,
    state: Any,
    game_config: Any,
    engine: Any = None,
    match: NodeMatch | None = None,
    log: Callable[[str], None] = print,
) -> list[Research]:
    """Re-search with ``agent`` (the one whose first decision at ``state`` gave
    ``first``) at every first off-tree opponent edge of ``trunk`` on this street
    (:func:`~pokerbot.search.tree_map.offtree_edges` of ``first``'s tree against
    the trunk): replay the hand to just after the opponent's action and act
    again (see the module docstring)."""
    engine = engine or get_engine()
    seat = int(state.current_player)
    m = match_nodes(first.tree, trunk) if match is None else match
    key = played_key(state)
    out = []
    for n in offtree_edges(m, trunk, 1 - seat):
        hist = tuple(trunk.histories[n])
        seeded = seed_played(agent, key, first, m, trunk, n, seat)
        s = replay(engine, game_config, state, hist)
        if int(s.current_player) != seat:
            raise ValueError(f"after {hist} seat {int(s.current_player)} acts, not {seat}")
        run = act_again(f"{name} at {hist}", agent, s, game_config)
        out.append(
            Research(n, hist, run.stats, run.seconds, StrategySnapshot.of(run.solver), seeded)
        )
        st = run.stats
        log(
            f"    {name} re-search after {_show_action(hist[-1])} (history {len(hist)} "
            f"actions): {st.get('total_seconds', run.seconds):.2f}s, {st.get('iterations')} "
            f"iterations, {st.get('nodes')} nodes, {seeded} locks seeded"
        )
        del run
        _free()
    return out


def compose_research(
    base: torch.Tensor,
    researches: Sequence[Research],
    trunk_solver: Any,
    seat: int,
    strict: bool = False,
    fallback: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict]]:
    """``base`` (a strategy on ``trunk_solver``'s tree) with every decision node
    at or below each re-search's edge replaced by that re-search's strategy,
    translated onto the trunk (:func:`~pokerbot.search.tree_map.translate_sigma`
    restricted to the subtree). Returns it (on ``base``'s device) and one
    translation report per re-search, with ``path_exact`` (the re-search's tree
    has the edge's history exactly, as it must) and ``subtree_decision_nodes``."""
    out = base.clone()
    tree = trunk_solver.tree
    d_index = {n: d for d, n in enumerate(trunk_solver.dec_nodes.tolist())}
    reports = []
    for r in researches:
        sub = [x for x in subtree_nodes(tree, r.edge) if x in d_index]
        m = match_nodes(r.snapshot.tree, tree)
        sig, rep = translate_sigma(
            r.snapshot, trunk_solver, seat, strict=strict, fallback=fallback, match=m, nodes=sub
        )
        ids = torch.tensor([d_index[x] for x in sub], dtype=torch.long)
        out[ids] = sig[ids.to(sig.device)].to(out.device, out.dtype)
        rep["path_exact"] = bool(m.src_of[r.edge] >= 0 and not m.translated[r.edge])
        rep["subtree_decision_nodes"] = len(sub)
        reports.append(rep)
        del sig
    return out, reports


def _show_action(step: Sequence) -> str:
    _street, player, kind, amount = (int(x) for x in step)
    what = {0: "fold", 1: "check/call"}.get(kind, f"raise to {amount}")
    return f"seat {player} {what}"


def _research_summary(researches: Sequence[Research], reports: Sequence[dict]) -> dict:
    secs = [r.stats.get("total_seconds", r.seconds) for r in researches]
    its = [r.stats.get("iterations") for r in researches]
    edges = []
    for r, rep in zip(researches, reports, strict=True):
        t = r.snapshot.tree
        edges.append(
            {
                "node": r.edge,
                "history": [list(h) for h in r.history],
                "action": _show_action(r.history[-1]),
                "iterations": r.stats.get("iterations"),
                "nodes": r.stats.get("nodes"),
                "leaves": int((t.kind == VALUE).sum()),
                "flop_sizes": flop_sizes(t.street_actions),
                "total_seconds": r.stats.get("total_seconds", r.seconds),
                "solve_seconds": r.stats.get("solve_seconds"),
                "act_seconds": r.seconds,
                "seeded_locks": r.seeded,
                "translate": rep,
            }
        )
    fill: Counter = Counter()
    for rep in reports:
        fill.update(rep["fill"])
    return {
        "count": len(researches),
        "total_seconds": sum(secs),
        "mean_seconds": _mean(secs),
        "mean_iterations": _mean(its),
        "seeded_locks": sum(r.seeded for r in researches),
        "subtree_decision_nodes": sum(rep["subtree_decision_nodes"] for rep in reports),
        "all_paths_exact": all(rep["path_exact"] for rep in reports),
        "fill": dict(fill),
        "edges": edges,
    }


# -- re-solving below the root street -------------------------------------------------


def root_street_locks(
    tree: SubgameTree, dec_nodes: torch.Tensor, sigma: torch.Tensor
) -> dict[int, torch.Tensor]:
    """``{node: [C, n] strategy}`` of ``sigma`` (``[D, A, C]`` over
    ``dec_nodes``) at every decision node of both players on ``tree``'s root
    street: the ``locked`` argument of
    :class:`~pokerbot.search.solver.RangeSolver` (views of ``sigma``)."""
    kind = tree.kind.tolist()
    street = tree.street.tolist()
    nch = tree.num_children.tolist()
    out: dict[int, torch.Tensor] = {}
    for d, n in enumerate(dec_nodes.tolist()):
        if kind[n] == DECISION and street[n] == tree.root_street:
            out[n] = sigma[d, : nch[n]].t()
    return out


@torch.no_grad()
def resolve_below_root(
    trunk_solver: Any,
    sigma: torch.Tensor,
    value_leaves: Any,
    iterations: int,
    game_config: Any,
    cfg: SolverConfig | None = None,
    mix: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """``sigma`` (a profile on ``trunk_solver``'s tree, in its decision-node
    order) with every decision below the root street solved again: a fresh
    :class:`~pokerbot.search.solver.RangeSolver` on the trunk and its plain
    root ranges (no gadget: unsafe), with every root-street decision of both
    players locked to ``sigma`` (:func:`root_street_locks`; the solver
    renormalises each combo's row), leaves valued by ``value_leaves``, solver
    settings ``cfg`` (default ``trunk_solver.cfg``) without a time limit, run for
    ``iterations`` DCFR iterations.

    ``mix`` > 0 locks the root-street decisions to ``(1 - mix) * sigma + mix *
    uniform`` while solving, so every later-street subgame is reached by both
    players: where ``sigma`` never takes an action (e.g. a size the search's
    tree lacks), the other player's decisions below it otherwise get no
    counterfactual value at all and keep the uniform strategy. CFR is
    scale-free per subgame, so a small ``mix`` solves those subgames as fully as
    the rest. The result plays ``sigma`` exactly on the root street either way.

    Returns the average strategy (on the CPU, ``sigma``'s layout, ``sigma``
    itself at the locked nodes) and a report: ``iterations``, ``locked_nodes``,
    ``mix``, ``locked_max_diff`` (the largest change of a locked probability
    during the solve, ~0 unless a row of ``sigma`` was not normalised),
    ``seconds`` / ``setup_seconds`` / ``solve_seconds`` and
    ``self_exploitability`` (the result's exploitability in the trunk's game
    with that leaf model, as the searches report theirs)."""
    t0 = time.perf_counter()
    tree = trunk_solver.tree
    cfg = replace(trunk_solver.cfg if cfg is None else cfg, time_budget=None)
    sig = sigma.to(trunk_solver.device, getattr(torch, cfg.dtype))
    locked = root_street_locks(tree, trunk_solver.dec_nodes, sig)
    if mix > 0:
        locked = {n: (1 - mix) * s + mix / s.shape[1] for n, s in locked.items()}
    solver = RangeSolver(tree, trunk_solver.ranges, cfg, locked=locked, value_leaves=value_leaves)
    del locked
    if not torch.equal(solver.dec_nodes, trunk_solver.dec_nodes):
        raise RuntimeError("the re-solve's decision nodes are not the trunk's")
    t_setup = time.perf_counter() - t0
    solver.solve(iterations=int(iterations))
    # the locks held (the solver never updates a locked node); then sigma exactly there
    ids = solver.locked.nonzero().flatten()
    diff = 0.0
    for a in range(0, len(ids), 1024):  # chunks: [K, A, C] copies of the locked rows
        j = ids[a : a + 1024]
        want = (1 - mix) * sig[j] + mix * solver.uniform[j] if mix > 0 else sig[j]
        diff = max(diff, float((solver.sigma[j] - want).abs().max()))
        solver.sigma[j] = sig[j]
    del sig
    out = solver.average_strategy().cpu()
    info = {
        "iterations": solver.iterations_done,
        "locked_nodes": len(ids),
        "mix": float(mix),
        "locked_max_diff": diff,
        "seconds": time.perf_counter() - t0,
        "setup_seconds": t_setup,
        "solve_seconds": solver.solve_time,
    }
    run = SearchRun("resolve", solver, None, {}, 0.0)
    info["self_exploitability"] = _self_exploitability(run, game_config)
    del run, solver
    return out, info


# -- one spot -------------------------------------------------------------------------


def _blueprint_on(solver: Any, blueprint: Any, game_config: Any) -> tuple:
    """:func:`~pokerbot.search.spot_eval.blueprint_profile`, passing
    ``game_config`` when that version accepts it."""
    if "game_config" in inspect.signature(blueprint_profile).parameters:
        return blueprint_profile(solver, blueprint, game_config=game_config)
    return blueprint_profile(solver, blueprint)


def _check_trunk_tree(tree: SubgameTree, name: str) -> None:
    """``ValueError`` unless ``tree`` ends at value-net leaves at the end of the
    turn, as :func:`~pokerbot.search.exact_eval.trunk_exploitability` needs
    (a turn tree solved to showdown has no leaves; a flop tree with
    ``depth_streets: 0`` has them at the end of the flop)."""
    if tree.last_street != 2 or not bool((tree.kind == VALUE).any()):
        hint = (
            'use a variant with {"tree": {"depth_streets_turn": 0}} as the trunk'
            if tree.root_street == 2
            else "use a trunk with tree.depth_streets 1"
        )
        raise ValueError(
            f"the scoring trunk {name!r} must end at value-net leaves at the end of the turn "
            f"(its tree's last street is {STREET_NAMES[tree.last_street]}): {hint}"
        )


def _tree_info(tree: Any) -> dict:
    return {
        "decision_nodes": int((tree.kind == DECISION).sum()),
        "leaves": int((tree.kind == VALUE).sum()),
        "street_actions": [[list(a) for a in s] for s in tree.street_actions],
        "raise_caps": list(tree.raise_caps),
        "chance_cards": tree.chance_cards_used,
    }


def evaluate_sizes(
    spot: Spot,
    blueprint: Any,
    game_config: Any,
    settings: SizeSettings,
    predictor_factory: Callable[[], Any] = lambda: None,
    net_path: str | None = None,
    river_spec: Any = None,
    log: Callable[[str], None] = print,
    keep: dict | None = None,
) -> dict:
    """Everything for one spot (see the module docstring), as a JSON-ready dict.

    ``predictor_factory()`` gives each search its leaf predictor (``None``: the
    agent loads ``leaf.net`` itself, ``net_path`` when given). ``river_spec``
    defaults to ``blueprint.spec``. ``keep`` (a dict) receives the scoring
    solver, the snapshots, the re-searches and the sigmas, for tests."""
    t_spot = time.perf_counter()
    check_settings(settings, [spot])
    state = spot.state
    seat = int(state.current_player)
    river_spec = blueprint.spec if river_spec is None else river_spec
    plan = search_runs(settings)
    trunk_name = settings.trunk_name()
    log(f"{spot.label}: board {list(state.board)}, seat {seat} to act")

    warm_s = None
    if settings.spot_warmup:
        t0 = time.perf_counter()
        cfg = warmup_config(settings, net_path)
        run_search("spot-warmup", blueprint, state, game_config, cfg, predictor_factory())
        _free()
        warm_s = time.perf_counter() - t0
        log(f"  cache warm-up search: {warm_s:.1f}s")

    searches: dict[str, dict] = {}
    snaps: dict[str, StrategySnapshot] = {}
    agents: dict[str, Any] = {}  # first-decision agents of the variants to re-search
    eval_solver = None
    trunk_agent = None  # the trunk search's agent and solver settings, for the re-solves
    trunk_cfg = None
    for name, variant, iters in plan:
        cfg = variant_config(settings, variant, net_path, iters)
        run = run_search(name, blueprint, state, game_config, cfg, predictor_factory())
        s = run.stats
        info = {
            **s,
            "max_nodes": variant.max_nodes,
            "fixed_iterations": iters,
            "act_seconds": run.seconds,
            **_tree_info(run.solver.tree),
            "self_exploitability": _self_exploitability(run, game_config),
        }
        searches[name] = info
        log(
            f"  {name}: {s.get('total_seconds', run.seconds):.2f}s ({s.get('iterations')} "
            f"iterations, {s.get('nodes')} nodes, {info['leaves']} leaves, "
            f"setup {s.get('setup_seconds', 0):.2f}s, solve {s.get('solve_seconds', 0):.2f}s; "
            f"own-game exploitability {info['self_exploitability']['mbb']:.0f} mbb/hand)"
        )
        snaps[name] = StrategySnapshot.of(run.solver, "cpu")
        if name == trunk_name:
            _check_trunk_tree(run.solver.tree, name)
            eval_solver = scoring_solver(run.solver, settings.eval_max_runouts)
            if settings.resolve_iters:
                trunk_agent, trunk_cfg = run.agent, run.solver.cfg
        elif settings.research and iters is None:
            agents[name] = run.agent
        del run
        _free()
    assert eval_solver is not None
    trunk = snaps[trunk_name]
    ttree = trunk.tree
    solved = range(ttree.root_street, ttree.last_street + 1)  # the trunk's betting streets
    for name, snap in snaps.items():
        diffs = action_set_differences(snap.tree, ttree, solved)
        searches[name]["subset_of_trunk"] = not diffs
        searches[name]["subset_issues"] = diffs
        searches[name]["ranges_max_diff"] = float((snap.ranges - trunk.ranges).abs().max())
        if diffs:
            log(f"  WARNING: {name}'s actions are not a subset of the trunk's: {diffs}")

    # every profile on the trunk
    t_bp = time.perf_counter()
    bp_sigma, dropped = _blueprint_on(eval_solver, blueprint, game_config)
    t_bp = time.perf_counter() - t_bp
    sigmas: dict[str, torch.Tensor] = {}
    matches: dict[str, NodeMatch] = {}
    for name, snap in snaps.items():
        t0 = time.perf_counter()
        matches[name] = match_nodes(snap.tree, ttree)
        sig, rep = translate_sigma(
            snap,
            eval_solver,
            seat,
            strict=settings.strict,
            fallback=bp_sigma,
            match=matches[name],
        )
        rep["seconds"] = time.perf_counter() - t0
        searches[name]["translate"] = rep
        sigmas[name] = sig.cpu()
        del sig
        lost = rep["lost_mass"]
        log(
            f"  {name} -> trunk: translated nodes {rep['translated_nodes']['searcher']} / "
            f"{rep['translated_nodes']['opponent']} (searcher / opponent), unmatched "
            f"{sum(rep['unmatched_nodes'].values())}, lost-mass nodes "
            f"{lost['searcher']['nodes']} / {lost['opponent']['nodes']}"
        )

    # re-searches at the opponent's off-tree actions on the root street, as in play
    researched: dict[str, list[Research]] = {}
    if settings.research:
        for name, _variant, iters in plan:
            if iters is not None:
                continue
            if name not in agents:  # the trunk itself: nothing is off its tree
                searches[name]["research"] = _research_summary([], [])
                continue
            rs = research_offtree(
                name,
                agents.pop(name),
                snaps[name],
                ttree,
                state,
                game_config,
                match=matches[name],
                log=log,
            )
            comp, reports = compose_research(
                sigmas[name], rs, eval_solver, seat, settings.strict, bp_sigma
            )
            summary = _research_summary(rs, reports)
            searches[name]["research"] = summary
            if rs:
                sigmas[research_name(name)] = comp
            if keep is not None:
                researched[name] = rs
            del rs, comp
            log(
                f"  {name}: {summary['count']} re-searches ({_f(summary['total_seconds'], 1)}s), "
                f"{summary['subtree_decision_nodes']} trunk decision nodes below them"
                + ("" if summary["all_paths_exact"] else "; WARNING: a path was not exact")
            )
            _free()
    sigmas["blueprint"] = bp_sigma.cpu()
    offtree = offtree_mass(eval_solver, bp_sigma, dropped)
    if keep is not None:
        keep.update(
            eval_solver=eval_solver,
            snaps=snaps,
            sigmas=sigmas,
            dropped=dropped,
            researches=researched,
            matches=matches,
        )
    del snaps, trunk, bp_sigma, dropped, researched, agents
    _free()

    # every budget variant's final profile re-solved below the flop on the trunk
    if settings.resolve_iters:
        for name, _variant, iters in plan:
            if iters is not None:
                continue
            src = research_name(name) if research_name(name) in sigmas else name
            sig, rinfo = resolve_below_root(
                eval_solver,
                sigmas[src],
                trunk_agent.value_leaf_provider(eval_solver.tree),
                settings.resolve_iters,
                game_config,
                trunk_cfg,
                settings.resolve_mix,
            )
            searches[name]["resolve"] = {"source": src, **rinfo}
            sigmas[resolved_name(name)] = sig
            del sig
            log(
                f"  {name}: re-solved below the flop from {src} in {rinfo['seconds']:.1f}s "
                f"({rinfo['iterations']} iterations, {rinfo['locked_nodes']} flop decisions "
                f"locked; own-game exploitability {rinfo['self_exploitability']['mbb']:.0f} "
                "mbb/hand)"
            )
            _free()
    del trunk_agent
    _free()

    # scoring
    ev = settings.eval_settings()
    batch = [settings.river_batch]
    mbb = 1000.0 / float(game_config.big_blind)
    profiles: dict[str, dict] = {}
    for name in profile_order(settings):
        if name not in sigmas:  # a re-searched profile without re-searches: the same
            base = name[: -len(RESEARCH)]
            profiles[name] = {**profiles[base], "same_as": base}
            continue
        if not settings.score_translated and name != trunk_name and research_name(name) in sigmas:
            profiles[name] = {"not_scored": True}  # its re-searched profile is scored
            continue
        res = _score_with_retry(
            batch,
            log,
            eval_solver,
            sigmas[name].to(eval_solver.device),
            river_spec,
            game_config,
            ev,
            log=lambda m, n=name: log(f"    [{n}] {m.lstrip('# ').strip()}"),
        )
        res["one_sided_mbb"] = res["br"][1 - seat] * mbb
        profiles[name] = res
        log(
            f"  {name}: one-sided {res['one_sided_mbb']:.0f}, two-sided {res['mbb']:.0f} mbb/hand "
            f"(BR {res['br'][0]:.1f} / {res['br'][1]:.1f} chips, {res['instances']} river "
            f"subgames, {res['skipped']} skipped, scored in {res['seconds']:.0f}s)"
        )
    out = {
        "label": spot.label,
        "board_index": spot.board_index,
        "spot_type": spot.spot_type,
        **(
            {"street": settings.street, "line": spot_line(spot.label)}
            if settings.street != 1
            else {}
        ),
        "board": [int(c) for c in state.board],
        "seat": seat,
        "hole": [int(c) for c in state.hole_cards(seat)],
        "pot": int(ttree.contrib[0].sum()),
        "trunk": trunk_name,
        "trunk_nodes": int(ttree.num_nodes),
        **_tree_info(ttree),
        "searches": searches,
        "profiles": profiles,
        "blueprint_offtree": offtree,
        "blueprint_seconds": t_bp,
        "spot_warmup_seconds": warm_s,
        "seconds": time.perf_counter() - t_spot,
    }
    return _jsonable(out)


# -- driver -------------------------------------------------------------------------


def run_size_evaluation(
    spots: Sequence[Spot],
    blueprint: Any,
    game_config: Any,
    settings: SizeSettings,
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
    every spot. With ``resume``, an existing ``out_json`` written with the same
    comparable settings and meta keys ``blueprint`` / ``leaf_model`` keeps its
    finished spots and only the others run. A failing spot is logged and
    recorded under ``errors`` (and retried on resume). ``warmup`` (a spot) is
    searched once with two iterations first, so CUDA start-up stays out of the
    timings. Spots off ``settings.street`` and a trunk that cannot be scored
    there raise ``ValueError`` before anything runs (:func:`check_settings`)."""
    check_settings(settings, [*spots, *([warmup] if warmup is not None else [])])
    out_json = Path(out_json)
    data: dict[str, Any] = {
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
        cfg = warmup_config(settings, net_path)
        run_search("warmup", blueprint, warmup.state, game_config, cfg, predictor_factory())
        _free()
        log(f"warm-up search: {time.perf_counter() - t0:.1f}s")
    t_all = time.perf_counter()
    order = {s.label: j for j, s in enumerate(spots)}
    for i, spot in enumerate(todo):
        try:
            res = evaluate_sizes(
                spot, blueprint, game_config, settings, predictor_factory, net_path, log=log
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
        data["spots"].sort(key=lambda s: order.get(s["label"], len(order)))
        _write_json(out_json, data)
        if out_md is not None:
            _write_text(Path(out_md), render_markdown(data))
    return data


# -- markdown -------------------------------------------------------------------------


def profile_order(settings: SizeSettings) -> list[str]:
    """Profile names in report order: per budget variant the translated, the
    re-searched and the re-solved profile, then the fixed-iteration references,
    then the blueprint."""
    out = []
    for name, _variant, iters in search_runs(settings):
        out.append(name)
        if iters is None and settings.research:
            out.append(research_name(name))
        if iters is None and settings.resolve_iters:
            out.append(resolved_name(name))
    return [*out, "blueprint"]


def display_name(name: str) -> str:
    """``"6000"`` -> ``"6000"``, ``"6000_it300"`` -> ``"6000, 300 it"``,
    ``"6000_research"`` -> ``"6000, re-searched"``, ``"6000_resolved"`` ->
    ``"6000, re-solved"``."""
    if name.endswith(RESEARCH):
        return f"{name[: -len(RESEARCH)]}, re-searched"
    if name.endswith(RESOLVED):
        return f"{name[: -len(RESOLVED)]}, re-solved"
    size, _, it = name.partition("_it")
    return f"{size}, {it} it" if it else name


def flop_sizes(street_actions: Sequence[Sequence], street: int = 1) -> str:
    """The sized raises of ``street`` (default the flop), e.g. ``"open 0.75/1;
    rr 1"`` (all-in is always there)."""
    both, opens, rr = [], [], []
    for a in street_actions[street]:
        if a[0] not in ("raise", "raise_x"):
            continue
        s = f"{float(a[1]):g}" + ("x" if a[0] == "raise_x" else "")
        if len(a) < 3:
            both.append(s)
        else:
            (opens if a[2] == "open" else rr).append(s)
    parts = ["/".join(both)] if both else []
    parts += [f"open {'/'.join(opens)}"] if opens else []
    parts += [f"rr {'/'.join(rr)}"] if rr else []
    return "; ".join(parts) or "all-in only"


def _columns(spots: Sequence[dict]) -> list[str]:
    """Every profile name but the blueprint's, in first-seen order, without
    those scored on no spot (``not_scored``)."""
    cols: list[str] = []
    scored = set()
    for s in spots:
        for name, p in s["profiles"].items():
            if name != "blueprint" and name not in cols:
                cols.append(name)
            if not p.get("not_scored"):
                scored.add(name)
    return [c for c in cols if c in scored]


def _search_names(spots: Sequence[dict]) -> list[str]:
    """Every search's name, in first-seen order."""
    names: list[str] = []
    for s in spots:
        names += [n for n in s["searches"] if n not in names]
    return names


def _variant_rows(st: dict) -> list[tuple[str, int, dict]]:
    """``(name, max_nodes, overrides)`` of the budget variants of a settings dict
    (:meth:`SizeSettings.comparable`), also of one written before variants."""
    if st.get("variants"):
        return [(v["name"], v["max_nodes"], v.get("search") or {}) for v in st["variants"]]
    return [(search_name(s), s, {}) for s in st["sizes"]]


def _groups(spots: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    """The overall mean, then per spot type and per flop line (turn spots), each
    when it is a proper subset."""
    groups = [("**mean**", list(spots))]
    for t in SPOT_TYPES:
        g = [s for s in spots if s["spot_type"] == t]
        if g and len(g) < len(spots):
            groups.append((f"mean {SPOT_NAMES[t]} ({len(g)})", g))
    for line in TURN_LINES:
        g = [s for s in spots if s.get("line") == line]
        if g and len(g) < len(spots):
            groups.append((f"mean {line} ({len(g)})", g))
    return groups


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _metric(s: dict, c: str, key: str) -> float | None:
    p = s["profiles"].get(c)
    return None if p is None or p.get("not_scored") else p[key]


def _g(x: Any, nd: int = 0) -> str:
    """:func:`~pokerbot.search.spot_eval._f`, with ``"-"`` for a missing value."""
    return "-" if x is None else _f(x, nd)


def _table(
    spots: Sequence[dict],
    cols: Sequence[str],
    key: str,
    heads: Sequence[str],
    base: str | None = None,
) -> list[str]:
    """Rows per spot and group means of profile metric ``key`` for ``cols``
    (headed ``heads``) and the blueprint, plus each column minus ``base``."""
    diff = [c for c in cols if base is not None and c != base]
    bhead = heads[list(cols).index(base)] if base in cols else ""
    head = ["spot", *heads, "blueprint", *[f"{heads[list(cols).index(c)]} - {bhead}" for c in diff]]
    out = [_row(head), "|---|" + "---:|" * (len(head) - 1)]

    def row(label: str, g: Sequence[dict]) -> str:
        vals = {c: _mean([_metric(s, c, key) for s in g]) for c in [*cols, "blueprint"]}
        cells = [label] + [_g(vals[c]) for c in [*cols, "blueprint"]]
        b = vals.get(base) if base else None
        cells += [_g(None if vals[c] is None or b is None else vals[c] - b) for c in diff]
        return _row(cells)

    out += [row(s["label"], [s]) for s in spots]
    out += [row(name, g) for name, g in _groups(spots)]
    return out


def render_markdown(data: dict) -> str:
    """The report: one-sided and two-sided exploitability per spot and variant
    (mbb/hand) with means, re-searched, re-solved and translated, the
    re-searches, the re-solves, the searches (iterations, seconds, trees), the
    translation onto the trunk and the scoring details."""
    st = data["settings"]
    meta = data.get("meta", {})
    spots = data["spots"]
    cols = _columns(spots)
    res_cols = [c for c in cols if c.endswith(RESEARCH)]
    rsv_cols = [c for c in cols if c.endswith(RESOLVED)]
    tr_cols = [c for c in cols if c not in res_cols and c not in rsv_cols]
    names = _search_names(spots)
    budget_cols = [c for c in names if "_it" not in c]
    rows = _variant_rows(st)
    named = bool(st.get("variants") or st.get("trunk"))
    per = "variant" if st.get("variants") else "size"
    street = int(st.get("street", 1))
    sn = STREET_NAMES[street]
    lines = [f"# {sn.capitalize()} tree size under a fixed time budget", ""]
    lines.append(
        f"Blueprint `{meta.get('blueprint')}`, leaf model `{meta.get('leaf_model')}`, "
        f"device {meta.get('device')}{', GPU ' + meta['gpu'] if meta.get('gpu') else ''}. "
        f"Evaluated: {sn} decisions."
    )
    ref = f"; reference: {st['iters']} fixed iterations per {per}" if st.get("iters") else ""
    runouts = st.get("eval_max_runouts")
    allin = "the search's run-outs" if runouts is None else f"up to {runouts} run-outs per board"
    extra = f" Extra search overrides: `{json.dumps(st['search'])}`." if st.get("search") else ""
    if st.get("variants"):
        where = "as the variants " + "; ".join(
            f"`{n}` at max_nodes {m}" + (f" with `{json.dumps(o)}`" if o else "")
            for n, m, o in rows
        )
    else:
        where = f"at max_nodes {' / '.join(str(s) for s in st['sizes'])}"
    trunk = st.get("trunk") or rows[-1][0]
    trunk_txt = (
        f"the `{trunk}` search's tree" if named else f"the {st['sizes'][-1]}-node search's tree"
    )
    leaves = (
        "value-net leaves at the end of the turn"
        if street == 1
        else "turn trees solved to showdown, or with `tree.depth_streets_turn: 0` ending at "
        "value-net leaves at the end of the turn"
    )
    lines.append(
        f"Searches: `{st['config']}` ({leaves}) {where}, {sn} "
        f"budget {st['budget']:g} s with the config's iteration settings{ref}.{extra} "
        + (
            f"A short search on each spot first warms the blueprint caches, so every {per} "
            "starts from the same cache state. "
            if st.get("spot_warmup")
            else ""
        )
        + f"Scoring trunk: {trunk_txt}. Each search's average "
        "strategy is translated onto it (`tree_map.translate_sigma`: exact where the histories "
        "agree, sizes a smaller tree lacks mapped onto its sizes by the deterministic "
        "pseudo-harmonic rule, unmatched nodes play the blueprint); the blueprint is mapped "
        f"directly. Every (leaf, river card) river subgame solved exactly ({st['river_iters']} "
        f"DCFR iterations, {st['mix']:g} uniform mixed into both river ranges), best responses "
        f"backed up through the trunk on the plain root ranges; all-ins over {allin}."
    )
    if street == 2:
        lines += [
            "",
            "Turn spots: the trunk ends at the end of turn betting, so a search solved to "
            "showdown is scored on its turn strategy alone. Its river plan is dropped: every "
            "river is played by the exact river equilibrium of the scoring (the blueprint's "
            "river abstraction), as river searches would replace that plan in play.",
        ]
    if res_cols:
        lines += [
            "",
            f"**Re-searched** profiles are what the agent plays: when the opponent takes a {sn} "
            "size the search's tree lacks (after a history the tree has), the same agent "
            "searches again there, with that size forced into its tree, its root info reused "
            f"and its own earlier {sn} decisions locked to what it played. Below each such "
            "action the profile is the re-search's strategy; elsewhere the first search's. "
            "**Translated** profiles answer those sizes with the first search's strategy for "
            "the nearest size it has. Only off-tree actions are re-searched (the agent also "
            "re-searches at on-tree decisions, for every size)."
            + (
                " Translated profiles of searches with re-searches are not scored."
                if st.get("score_translated") is False
                else ""
            ),
        ]
    if rsv_cols:
        mix = st.get("resolve_mix") or 0.0
        lines += [
            "",
            "**Re-solved** profiles compare flop decisions alone: on the trunk, every flop "
            "decision of both players is locked to the search's profile (re-searched where it "
            "has re-searches, else translated; the trunk search's own strategy for the trunk), "
            "and every later-street decision is re-solved with the trunk search's leaf model "
            f"for {st.get('resolve_iters')} DCFR iterations on the plain root ranges (unsafe: "
            "no gadget), as separate turn searches replace a flop search's turn plan in play. "
            "So the columns differ only in the flop strategy. "
            + (
                f"While solving, the flop locks have {mix:g} uniform mixed in, so the turn "
                "below flop actions a profile never takes is solved too (the profile still "
                "plays its own flop)."
                if mix
                else "Below a flop action a profile never takes (a size its tree lacks), the "
                "other player's turn decisions get no counterfactual value and stay uniform "
                "(see `resolve_mix`)."
            ),
        ]
    one_txt = (
        "Best response of the opponent of the player to act against that player's trunk "
        "strategy, in the trunk's game. The game value is the same for every column, so "
        "differences are exact differences in the searcher's exploitability. Lower is better."
    )
    two_txt = (
        "(BR_0 + BR_1) / 2, both players playing the profile. With safe resolving the "
        "opponent's side carries the gadget's unrefined subgame strategy "
        "(see docs/value_net_eval.md)."
    )

    def sizes_of(cs: Sequence[str]) -> list[str]:
        out = []
        for c in cs:
            for suffix in (RESEARCH, RESOLVED):
                if c.endswith(suffix):
                    c = c[: -len(suffix)]
                    break
            else:
                c = display_name(c)
            out.append(c)
        return out

    tables = []
    if res_cols:
        tables.append(("re-searched (as played)", res_cols))
    if rsv_cols:
        tables.append(("re-solved", rsv_cols))
    tables.append(("translated only" if res_cols or rsv_cols else "", tr_cols))
    for metric, title, txt in (
        ("one_sided_mbb", "Best response against the searcher", one_txt),
        ("mbb", "Two-sided exploitability", two_txt),
    ):
        for kind, cs in tables:
            head = f"## {title}{', ' + kind if kind else ''} (mbb/hand)"
            base = cs[0] if cs else None
            lines += ["", head, "", txt, ""]
            lines += _table(spots, cs, metric, sizes_of(cs), base)

    # re-searches
    if res_cols:
        lines += ["", "## Re-searches at off-tree opponent actions", ""]
        rh = [
            "spot",
            "search",
            "re-searches",
            "trunk decision nodes below",
            "mean iterations",
            "mean s",
            "total s",
            "locks seeded",
            "opponent actions",
        ]
        lines += [_row(rh), "|---|---|" + "---:|" * 6 + "---|"]
        for s in spots:
            for name in budget_cols:
                r = s["searches"].get(name, {}).get("research")
                if r is None:
                    continue
                acts = ", ".join(e["action"].split(" ", 2)[-1] for e in r["edges"]) or "-"
                lines.append(
                    _row(
                        [
                            s["label"],
                            name,
                            str(r["count"]),
                            str(r["subtree_decision_nodes"]),
                            _f(r["mean_iterations"]),
                            _f(r["mean_seconds"], 2),
                            _f(r["total_seconds"], 1),
                            str(r["seeded_locks"]),
                            acts,
                        ]
                    )
                )
        lines += [
            "",
            "Each re-search is a full decision of the same agent (tree, setup and solve within "
            f"the {sn} budget); the opponent actions are raise-to amounts in chips.",
        ]

    # re-solves
    if any("resolve" in x for s in spots for x in s["searches"].values()):
        lines += ["", "## Re-solves below the flop on the trunk", ""]
        rh = [
            "spot",
            "search",
            "flop from",
            "locked decisions",
            "iterations",
            "s",
            "own game (mbb)",
        ]
        lines += [_row(rh), "|---|---|---|" + "---:|" * 4]
        for s in spots:
            for name in budget_cols:
                r = s["searches"].get(name, {}).get("resolve")
                if r is None:
                    continue
                lines.append(
                    _row(
                        [
                            s["label"],
                            name,
                            display_name(r["source"]),
                            str(r["locked_nodes"]),
                            str(r["iterations"]),
                            _f(r["seconds"], 1),
                            _f(r["self_exploitability"]["mbb"]),
                        ]
                    )
                )
        lines += [
            "",
            "own game = the re-solved profile's exploitability in the trunk's game with the "
            "trunk search's leaf model (both players best-respond on every street).",
        ]

    # searches
    lines += ["", "## Searches", ""]
    th = [
        "spot",
        "search",
        "nodes",
        "decision nodes",
        "leaves",
        f"{sn} sizes",
        f"{STREET_NAMES[street + 1]} sizes",
        "iterations",
        "total s",
        "setup s",
        "solve s",
        "ms/iter",
        "own game (mbb)",
    ]
    lines += [_row(th), "|---|---|" + "---:|" * 3 + "---|---|" + "---:|" * (len(th) - 7)]

    def srow(label: str, name: str, xs: Sequence[dict]) -> str:
        def m(key: str) -> float | None:
            return _mean([x.get(key) for x in xs])

        def sizes(s: int) -> str:
            if s == 3 and all(x.get("leaves") for x in xs):
                return "-"  # turn trees ending at leaves: no river
            got = {flop_sizes(x["street_actions"], s) for x in xs if x.get("street_actions")}
            return got.pop() if len(got) == 1 else "varies"

        it, solve = m("iterations"), m("solve_seconds")
        ms = None if not it or solve is None else 1000 * solve / it
        return _row(
            [
                label,
                display_name(name),
                _f(m("nodes")),
                _f(m("decision_nodes")),
                _f(m("leaves")),
                sizes(street),
                sizes(street + 1),
                _f(it),
                _f(m("total_seconds"), 2),
                _f(m("setup_seconds"), 2),
                _f(solve, 2),
                _f(ms, 1),
                _f(_mean([x["self_exploitability"]["mbb"] for x in xs])),
            ]
        )

    for s in spots:
        for name in names:
            if name in s["searches"]:
                lines.append(srow(s["label"], name, [s["searches"][name]]))
    for name in names:
        xs = [s["searches"][name] for s in spots if name in s["searches"]]
        if xs:
            lines.append(srow("**mean**", name, xs))
    lines += [
        "",
        "total = the agent's whole decision (tree, gadget terminate values, leaf setup, solve). "
        "own game = the search's exploitability in its own tree with its leaf model.",
    ]

    # translation
    lines += ["", "## Translation onto the trunk", ""]
    xh = [
        "spot",
        "search",
        "subset of trunk",
        "translated nodes (searcher / opp.)",
        "unmatched (searcher / opp.)",
        "lost-mass nodes (searcher / opp.)",
        "max lost",
        "children by pseudo-harmonic",
    ]
    lines += [_row(xh), "|---|---|---|" + "---:|" * (len(xh) - 3)]

    def pair(d: dict) -> str:
        return f"{d['searcher']} / {d['opponent']}"

    for s in spots:
        for name in names:
            x = s["searches"].get(name)
            if x is None or "translate" not in x:
                continue
            r = x["translate"]
            lm = r["lost_mass"]
            lines.append(
                _row(
                    [
                        s["label"],
                        display_name(name),
                        "yes" if x.get("subset_of_trunk") else "**no**",
                        pair(r["translated_nodes"]),
                        pair(r["unmatched_nodes"]),
                        pair({k: v["nodes"] for k, v in lm.items()}),
                        f"{max(lm['searcher']['max'], lm['opponent']['max']):.2g}",
                        str(r["fill"].get("harmonic", 0)),
                    ]
                )
            )
    if spots:
        n_dec = spots[0]["decision_nodes"]
        bpo = _mean([s.get("blueprint_offtree") for s in spots])
        lines += [
            "",
            f"Translated nodes: trunk decision nodes (of {n_dec} on the first spot) reached "
            "through at least one action the search's tree lacks, mapped onto its tree. "
            f"Blueprint: {_f(bpo, 3)} decisions per hand on average where it would pick a size "
            "the trunk lacks (renormalised over the trunk's actions).",
        ]

    # scoring details
    lines += ["", "## Scoring details", ""]
    dh = [
        "spot",
        "profile",
        "one-sided",
        "two-sided",
        "BR0 (chips)",
        "BR1 (chips)",
        "river subgames",
        "skipped",
        "skip bound (mbb)",
        "max river expl. (% pot)",
        "scoring s",
    ]
    lines += [_row(dh), "|---|---|" + "---:|" * (len(dh) - 2)]
    for s in spots:
        for c in [*cols, "blueprint"]:
            p = s["profiles"].get(c)
            if p is None or p.get("same_as") or p.get("not_scored"):
                continue
            lines.append(
                _row(
                    [
                        s["label"],
                        display_name(c),
                        _f(p["one_sided_mbb"]),
                        _f(p["mbb"]),
                        _f(p["br"][0], 1),
                        _f(p["br"][1], 1),
                        str(p["instances"]),
                        str(p["skipped"]),
                        f"{p['skip_bound_mbb']:.2g}",
                        _f(100 * p["river_exploit_max"], 2),
                        _f(p["seconds"], 0),
                    ]
                )
            )
    if res_cols:
        lines += [
            "",
            "A re-searched profile without re-searches (the trunk's own, or one whose tree has "
            f"every opponent {sn} size of the trunk's) is the translated one and is scored once.",
        ]
    total = sum(s["seconds"] for s in spots)
    lines += ["", f"{len(spots)} spots, {total / 60:.1f} min of evaluation."]
    if data.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"* {e['label']}: `{e['error']}`" for e in data["errors"]]
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SIZES",
    "RESEARCH",
    "RESOLVED",
    "STREETS",
    "STREET_NAMES",
    "Research",
    "SizeSettings",
    "Variant",
    "act_again",
    "check_settings",
    "compose_research",
    "config_path",
    "display_name",
    "evaluate_sizes",
    "flop_sizes",
    "played_key",
    "profile_order",
    "render_markdown",
    "replay",
    "research_name",
    "research_offtree",
    "resolve_below_root",
    "resolved_name",
    "root_street_locks",
    "run_size_evaluation",
    "search_name",
    "search_plan",
    "search_runs",
    "seed_played",
    "select_spots",
    "select_turn_spots",
    "size_config",
    "size_overrides",
    "spot_line",
    "spot_street_errors",
    "street_root_state",
    "trunk_problem",
    "variant_config",
    "variant_from_arg",
    "warmup_config",
]
