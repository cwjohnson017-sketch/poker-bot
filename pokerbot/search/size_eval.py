"""Flop tree size under a fixed time budget, scored with an exact river.

The question: under the production flop budget (4 s), do bigger flop trees
(more bet sizes, fewer DCFR iterations) give a less exploitable strategy than
the production 6,000-node trees?

For one flop decision (a :class:`~pokerbot.search.spot_eval.Spot`),
:func:`evaluate_sizes`

1. runs the production value-net search (``configs/search_value_net.yaml``:
   value-net leaves at the end of the turn, each counting one node) once per
   ``max_nodes`` in ``SizeSettings.sizes``, under the ``budget`` flop time
   budget with the config's iteration cap and ``min_iterations``, so the
   iteration count depends on the tree; with ``iters`` also once per size for
   that fixed number of iterations (a reference without the time limit);
2. takes the largest budget search's tree as the scoring trunk and puts every
   search's average strategy on it with
   :func:`~pokerbot.search.tree_map.translate_sigma`. Where the histories
   agree this is exact. Opponent sizes that a smaller tree lacks map onto its
   sizes by the deterministic pseudo-harmonic rule, and unmatched nodes (none
   in the production trees) play the blueprint. The blueprint is mapped with
   :func:`~pokerbot.search.spot_eval.blueprint_profile`;
3. scores every profile with :func:`~pokerbot.search.spot_eval.score_profile`.
   Every (leaf, river card) river subgame is solved exactly, and best
   responses are backed up through the trunk on the plain root ranges. The
   **one-sided** number is the best response of the searcher's opponent (the
   searcher is the player to act); its differences between profiles are exact
   differences in the searcher's exploitability in the trunk's game. The
   **two-sided** number is ``(BR_0 + BR_1) / 2``.

The trunk's game gives the opponent every size of the largest tree, so a
smaller tree is charged for its translation of the sizes it lacks. The real
agent would re-solve at its next decision instead (with the observed size
added to the tree), so the smaller trees' numbers are pessimistic there.

Before the measured searches of a spot, a short search on it warms the
blueprint's caches (``spot_warmup``), so every size starts from the same cache
state.

:func:`run_size_evaluation` drives a list of spots, rewrites the JSON after
every spot (resumable) and renders the markdown report
(:func:`render_markdown`). The CLI is ``scripts/eval_tree_size.py``.
"""

from __future__ import annotations

import inspect
import json
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import torch

from ..config import REPO_ROOT
from .config import SearchConfig, search_config
from .spot_eval import (
    EXACT_FLOP_RUNOUTS,
    HUGE_BUDGET,
    SPOT_NAMES,
    SPOT_TYPES,
    EvalSettings,
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
    exploit_spots,
    offtree_mass,
    run_search,
    scoring_solver,
)
from .tree import DECISION, VALUE
from .tree_map import StrategySnapshot, action_set_differences, translate_sigma

DEFAULT_CONFIG = "configs/search_value_net.yaml"
DEFAULT_SIZES = (6000, 10000, 20000)


# -- settings ---------------------------------------------------------------------


@dataclass
class SizeSettings:
    sizes: tuple[int, ...] = DEFAULT_SIZES  # tree.max_nodes per search
    budget: float = 4.0  # flop time budget (seconds per decision)
    iters: int | None = None  # also a fixed-iteration search per size
    config: str = DEFAULT_CONFIG  # base search config (YAML)
    seed: int = 0  # search seed (gadget rollouts)
    river_iters: int = 200  # BatchRiverSolver iterations per river subgame
    mix: float = 0.05
    min_mass: float = 1e-7
    river_batch: int = 256  # halved automatically on CUDA out-of-memory
    eval_max_runouts: int | None = EXACT_FLOP_RUNOUTS  # None: the search's solver.max_runouts
    strict: bool = False  # raise when a search's strategy loses mass on the trunk
    spot_warmup: bool = True  # warm the blueprint caches on each spot first
    device: str = "auto"
    search: dict = field(default_factory=dict)  # extra search_config overrides (every search)

    def __post_init__(self) -> None:
        sizes = sorted({int(s) for s in self.sizes})
        if not sizes:
            raise ValueError("SizeSettings.sizes is empty")
        self.sizes = tuple(sizes)

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def comparable(self) -> dict:
        """The settings that change the numbers (for resuming)."""
        d = asdict(self)
        for k in ("river_batch", "device"):
            d.pop(k)
        d["sizes"] = list(d["sizes"])
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


def search_name(size: int, iters: int | None = None) -> str:
    """``"6000"`` for a budget search, ``"6000_it300"`` for a fixed-iteration one."""
    return f"{int(size)}" if iters is None else f"{int(size)}_it{int(iters)}"


def search_plan(settings: SizeSettings) -> list[tuple[str, int, int | None]]:
    """``(name, max_nodes, fixed iterations or None)`` of every search, budget
    searches first, smallest tree first."""
    out: list[tuple[str, int, int | None]] = [(search_name(s), s, None) for s in settings.sizes]
    if settings.iters:
        out += [(search_name(s, settings.iters), s, int(settings.iters)) for s in settings.sizes]
    return out


def size_overrides(
    settings: SizeSettings, max_nodes: int, net_path: str | None = None, iters: int | None = None
) -> dict:
    """``search_config`` overrides of one search over ``settings.config``:
    ``settings.search`` first, then this run's ``tree.max_nodes``, its flop time
    budget (or ``iters`` fixed iterations under a huge budget), value-net leaves
    and ``leaf.net = net_path`` when given."""
    o: dict[str, Any] = {
        "device": str(settings.torch_device()),
        "fallback_on_error": False,
        "seed": settings.seed,
    }
    o = _merge(o, settings.search)
    o = _merge(o, {"tree": {"max_nodes": int(max_nodes)}})
    if iters is None:
        o = _merge(o, {"time_budget": {"flop": float(settings.budget)}})
    else:
        huge = {"flop": HUGE_BUDGET, "turn": HUGE_BUDGET, "river": HUGE_BUDGET}
        o = _merge(o, {"min_iterations": int(iters), "time_budget": huge})
        o = _merge(o, {"solver": {"iterations": int(iters)}})
    o = _merge(o, {"leaf": {"mode": "value_net", **({"net": net_path} if net_path else {})}})
    return o


def size_config(
    settings: SizeSettings, max_nodes: int, net_path: str | None = None, iters: int | None = None
) -> SearchConfig:
    """A fresh :class:`SearchConfig` for one search (see :func:`size_overrides`)."""
    over = size_overrides(settings, max_nodes, net_path, iters)
    return search_config(config_path(settings.config), **over)


def select_spots(
    engine: Any,
    game_config: Any,
    boards: int = 1,
    seed: int = 5,
    types: Sequence[str] = SPOT_TYPES,
    picks: Sequence[str] | None = None,
) -> list[Spot]:
    """:func:`~pokerbot.search.spot_eval.exploit_spots` (``boards`` x ``types``),
    or with ``picks`` exactly those spots in that order, each ``"<board>:<type>"``
    (e.g. ``"0:bb_first"``, ``"1:btn_vs_check"``) on the same board sequence."""
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


# -- one spot -------------------------------------------------------------------------


def _blueprint_on(solver: Any, blueprint: Any, game_config: Any) -> tuple:
    """:func:`~pokerbot.search.spot_eval.blueprint_profile`, passing
    ``game_config`` when that version accepts it."""
    if "game_config" in inspect.signature(blueprint_profile).parameters:
        return blueprint_profile(solver, blueprint, game_config=game_config)
    return blueprint_profile(solver, blueprint)


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
    solver, the snapshots and the sigmas, for tests."""
    t_spot = time.perf_counter()
    state = spot.state
    seat = int(state.current_player)
    river_spec = blueprint.spec if river_spec is None else river_spec
    plan = search_plan(settings)
    trunk_name = search_name(settings.sizes[-1])
    log(f"{spot.label}: board {list(state.board)}, seat {seat} to act")

    warm_s = None
    if settings.spot_warmup:
        t0 = time.perf_counter()
        cfg = size_config(settings, settings.sizes[0], net_path, iters=2)
        run_search("spot-warmup", blueprint, state, game_config, cfg, predictor_factory())
        _free()
        warm_s = time.perf_counter() - t0
        log(f"  cache warm-up search: {warm_s:.1f}s")

    searches: dict[str, dict] = {}
    snaps: dict[str, StrategySnapshot] = {}
    eval_solver = None
    for name, size, iters in plan:
        cfg = size_config(settings, size, net_path, iters)
        run = run_search(name, blueprint, state, game_config, cfg, predictor_factory())
        s = run.stats
        info = {
            **s,
            "max_nodes": size,
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
            eval_solver = scoring_solver(run.solver, settings.eval_max_runouts)
        del run
        _free()
    assert eval_solver is not None
    trunk = snaps[trunk_name]
    ttree = trunk.tree
    for name, snap in snaps.items():
        diffs = action_set_differences(snap.tree, ttree)
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
    for name, snap in snaps.items():
        t0 = time.perf_counter()
        sig, rep = translate_sigma(
            snap, eval_solver, seat, strict=settings.strict, fallback=bp_sigma
        )
        rep["seconds"] = time.perf_counter() - t0
        searches[name]["translate"] = rep
        sigmas[name] = sig.cpu()
        lost = rep["lost_mass"]
        log(
            f"  {name} -> trunk: translated nodes {rep['translated_nodes']['searcher']} / "
            f"{rep['translated_nodes']['opponent']} (searcher / opponent), unmatched "
            f"{sum(rep['unmatched_nodes'].values())}, lost-mass nodes "
            f"{lost['searcher']['nodes']} / {lost['opponent']['nodes']}"
        )
    sigmas["blueprint"] = bp_sigma.cpu()
    offtree = offtree_mass(eval_solver, bp_sigma, dropped)
    if keep is not None:
        keep.update(eval_solver=eval_solver, snaps=snaps, sigmas=sigmas, dropped=dropped)
    del snaps, trunk, bp_sigma, dropped
    _free()

    # scoring
    ev = settings.eval_settings()
    batch = [settings.river_batch]
    mbb = 1000.0 / float(game_config.big_blind)
    profiles: dict[str, dict] = {}
    for name in [*searches, "blueprint"]:
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
    timings."""
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
        cfg = size_config(settings, settings.sizes[0], net_path, iters=2)
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


def display_name(name: str) -> str:
    """``"6000"`` -> ``"6000"``, ``"6000_it300"`` -> ``"6000, 300 it"``."""
    size, _, it = name.partition("_it")
    return f"{size}, {it} it" if it else name


def flop_sizes(street_actions: Sequence[Sequence]) -> str:
    """The flop's sized raises, e.g. ``"open 0.75/1; rr 1"`` (all-in is always there)."""
    both, opens, rr = [], [], []
    for a in street_actions[1]:
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
    cols: list[str] = []
    for s in spots:
        for name in s["profiles"]:
            if name != "blueprint" and name not in cols:
                cols.append(name)
    return [*cols, "blueprint"] if spots else cols


def _groups(spots: Sequence[dict]) -> list[tuple[str, list[dict]]]:
    groups = [("**mean**", list(spots))]
    for t in SPOT_TYPES:
        g = [s for s in spots if s["spot_type"] == t]
        if g and len(g) < len(spots):
            groups.append((f"mean {SPOT_NAMES[t]} ({len(g)})", g))
    return groups


def _row(cells: Sequence[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def render_markdown(data: dict) -> str:
    """The report: one-sided and two-sided exploitability per spot and size
    (mbb/hand) with means, the searches (iterations, seconds, trees), the
    translation onto the trunk and the scoring details."""
    st = data["settings"]
    meta = data.get("meta", {})
    spots = data["spots"]
    cols = _columns(spots)
    budget_cols = [c for c in cols if c != "blueprint" and "_it" not in c]
    base = budget_cols[0] if budget_cols else None
    lines = ["# Flop tree size under a fixed time budget", ""]
    lines.append(
        f"Blueprint `{meta.get('blueprint')}`, leaf model `{meta.get('leaf_model')}`, "
        f"device {meta.get('device')}{', GPU ' + meta['gpu'] if meta.get('gpu') else ''}."
    )
    ref = f"; reference: {st['iters']} fixed iterations per size" if st.get("iters") else ""
    runouts = st.get("eval_max_runouts")
    allin = "the search's run-outs" if runouts is None else f"up to {runouts} run-outs per board"
    extra = f" Extra search overrides: `{json.dumps(st['search'])}`." if st.get("search") else ""
    lines.append(
        f"Searches: `{st['config']}` (value-net leaves at the end of the turn) at max_nodes "
        f"{' / '.join(str(s) for s in st['sizes'])}, flop budget {st['budget']:g} s with the "
        f"config's iteration settings{ref}.{extra} "
        + (
            "A short search on each spot first warms the blueprint caches, so every size "
            "starts from the same cache state. "
            if st.get("spot_warmup")
            else ""
        )
        + f"Scoring trunk: the {st['sizes'][-1]}-node search's tree. Each search's average "
        "strategy is translated onto it (`tree_map.translate_sigma`: exact where the histories "
        "agree, sizes a smaller tree lacks mapped onto its sizes by the deterministic "
        "pseudo-harmonic rule, unmatched nodes play the blueprint); the blueprint is mapped "
        f"directly. Every (leaf, river card) river subgame solved exactly ({st['river_iters']} "
        f"DCFR iterations, {st['mix']:g} uniform mixed into both river ranges), best responses "
        f"backed up through the trunk on the plain root ranges; all-ins over {allin}."
    )
    groups = _groups(spots)

    def one(s: dict, c: str) -> float | None:
        p = s["profiles"].get(c)
        return None if p is None else p["one_sided_mbb"]

    def two(s: dict, c: str) -> float | None:
        p = s["profiles"].get(c)
        return None if p is None else p["mbb"]

    diff_cols = [c for c in cols if c not in ("blueprint", base)]
    lines += [
        "",
        "## Best response against the searcher (mbb/hand)",
        "",
        "Best response of the opponent of the player to act against that player's trunk "
        "strategy, in the trunk's game. The game value is the same for every column, so "
        "differences are exact differences in the searcher's exploitability. Lower is better.",
        "",
    ]
    head = ["spot", *[display_name(c) for c in cols]]
    head += [f"{display_name(c)} - {base}" for c in diff_cols]
    lines += [_row(head), "|---|" + "---:|" * (len(head) - 1)]

    def orow(label: str, g: Sequence[dict]) -> str:
        vals = {c: _mean([one(s, c) for s in g]) for c in cols}
        cells = [label] + [_f(vals[c]) for c in cols]
        b = vals.get(base) if base else None
        cells += [_f(None if vals[c] is None or b is None else vals[c] - b) for c in diff_cols]
        return _row(cells)

    lines += [orow(s["label"], [s]) for s in spots]
    lines += [orow(name, g) for name, g in groups]

    lines += [
        "",
        "## Two-sided exploitability (mbb/hand)",
        "",
        "(BR_0 + BR_1) / 2, both players playing the translated search strategy. With safe "
        "resolving the opponent's side carries the gadget's unrefined subgame strategy "
        "(see docs/value_net_eval.md).",
        "",
    ]
    head = ["spot", *[display_name(c) for c in cols]]
    lines += [_row(head), "|---|" + "---:|" * (len(head) - 1)]
    for s in spots:
        lines.append(_row([s["label"], *[_f(two(s, c)) for c in cols]]))
    for name, g in groups:
        lines.append(_row([name, *[_f(_mean([two(s, c) for s in g])) for c in cols]]))

    # searches
    lines += ["", "## Searches", ""]
    th = [
        "spot",
        "search",
        "nodes",
        "decision nodes",
        "leaves",
        "flop sizes",
        "iterations",
        "total s",
        "setup s",
        "solve s",
        "ms/iter",
        "own game (mbb)",
    ]
    lines += [_row(th), "|---|---|" + "---:|" * 3 + "---|" + "---:|" * (len(th) - 6)]
    names = [c for c in cols if c != "blueprint"]

    def srow(label: str, name: str, xs: Sequence[dict]) -> str:
        def m(key: str) -> float | None:
            return _mean([x.get(key) for x in xs])

        it, solve = m("iterations"), m("solve_seconds")
        ms = None if not it or solve is None else 1000 * solve / it
        sizes = {flop_sizes(x["street_actions"]) for x in xs if x.get("street_actions")}
        return _row(
            [
                label,
                display_name(name),
                _f(m("nodes")),
                _f(m("decision_nodes")),
                _f(m("leaves")),
                sizes.pop() if len(sizes) == 1 else "varies",
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
        for c in cols:
            p = s["profiles"].get(c)
            if p is None:
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
    total = sum(s["seconds"] for s in spots)
    lines += ["", f"{len(spots)} spots, {total / 60:.1f} min of evaluation."]
    if data.get("errors"):
        lines += ["", "## Errors", ""]
        lines += [f"* {e['label']}: `{e['error']}`" for e in data["errors"]]
    return "\n".join(lines) + "\n"


__all__ = [
    "DEFAULT_CONFIG",
    "DEFAULT_SIZES",
    "SizeSettings",
    "config_path",
    "display_name",
    "evaluate_sizes",
    "flop_sizes",
    "render_markdown",
    "run_size_evaluation",
    "search_name",
    "search_plan",
    "select_spots",
    "size_config",
    "size_overrides",
]
