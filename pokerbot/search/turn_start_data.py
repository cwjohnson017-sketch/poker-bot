"""Turn-start value-net training data: batched turn solves whose turn-end
leaves are valued by the turn-end net (``docs/turn_start_net.md``).

The turn-start net ``N_TS`` predicts the counterfactual values at the **turn
root** (after the turn card, before any turn betting), so that a flop search
can stop at the end of flop betting (``depth_streets: 0``) and value its
leaves by the chance average of ``N_TS`` over the turn cards
(:class:`~.value_leaf.FlopEndLeafEvaluator`). As in DeepStack, ``N_TS`` is
bootstrapped from the next net: every sample is a turn subgame solved by
:class:`~.batch_turn_solver.BatchTurnSolver` with ``VALUE`` leaves at the end
of turn betting valued by the turn-end net ``N_TE`` (itself bootstrapped from
the river net).

Pipeline per shard (``scripts/gen_turn_start_data.py``):

1. **States** (:func:`~.turn_data.make_turn_states` with ``turn_start=True``):
   a ``mix`` of blueprint self-play stopped at the turn root (source 0,
   :func:`~.value_ranges.selfplay_river_states` with ``turn_start=True``),
   perturbed copies of more self-play ranges (source 1) and DeepStack-style
   random ranges along the turn strength order on random 4-card boards with
   log-uniform ``c`` (source 2).
2. **Batches** (:func:`~.value_data.make_batches`): sorted by ``c`` and
   chunked; each batch is solved at one ``c`` drawn log-uniformly between its
   smallest and largest ``c`` (seeded), snapped to a reachable amount. The
   turn tree depends only on ``c``; the ranges are inputs only.
3. **Solve** (:func:`solve_turn_batch`): ``BatchTurnSolver`` on the turn-root
   tree of the action spec (:func:`~.batch_turn_solver.turn_tree`, button in
   seat 1 so seat order is ``(OOP, IP)``), then the best-response values of
   each player at the root against the other's average strategy.

Shards have the river-shard format (:data:`~.value_data.SHARD_DTYPES`):

* ``boards`` uint8 ``[n, 4]``: the turn board;
* ``c``, ``stack`` int32 ``[n]``: chips each player committed at the turn
  root (the solved ``c``) and chips behind;
* ``ranges`` float16 ``[n, 2, 1326]``: normalised ``(OOP, IP)`` ranges, zero
  on combos that hit the board (exactly the values the solver ran on);
* ``targets`` float16 ``[n, 2, 1326]``: ``v_p(c) / (m_-p(c) * 2c)``, ``v_p``
  the root best-response counterfactual value in chips and ``m_-p`` the
  opponent mass disjoint from the combo; zero on board conflicts and where
  ``m_-p <= 1e-9``;
* ``exploit`` float32 ``[n]``: the instance's exploitability in pot units, in
  the game the turn-end net defines;
* ``source`` uint8 ``[n]``: 0 self-play, 1 perturbed self-play, 2 random;

plus ``meta.json`` with ``kind: "turn_start"``, the turn-end net, the spec
and the per-shard timing and exploitability. Shards are written atomically
and seeded per ``(seed, shard index)``; ``resume`` skips existing shards.
Train with :func:`.turn_net.train_turn_net` and ``kind="turn_start"``
(``scripts/train_turn_net.py --kind turn_start``).
"""

from __future__ import annotations

import json
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import value_ranges as vr
from .batch_turn_solver import BatchTurnSolver, turn_tree
from .combos import blocked_sum
from .turn_data import DEFAULT_GAME, make_turn_states, quantise
from .turn_net import START_KIND
from .value_data import (
    SHARD_DTYPES,
    exploit_summary,
    make_batches,
    shard_name,
    shard_seeds,
    take,
    write_atomic,
)

MASS_EPS = 1e-9
FORMAT_VERSION = 1


# --------------------------------------------------------------------------- solving


class TurnTreeCache:
    """Turn-root trees by ``c`` (LRU), built with :func:`~.batch_turn_solver.turn_tree`."""

    def __init__(self, game_config: Any, spec: Any, device: Any, size: int = 64) -> None:
        self.game_config = game_config
        self.spec = spec
        self.device = device
        self.size = int(size)
        self._trees: OrderedDict[int, Any] = OrderedDict()

    def get(self, c: int) -> Any:
        c = int(c)
        if c in self._trees:
            self._trees.move_to_end(c)
            return self._trees[c]
        tree = turn_tree(self.game_config, c, self.spec, button=1, device=self.device)
        self._trees[c] = tree
        while len(self._trees) > self.size:
            self._trees.popitem(last=False)
        return tree


@torch.no_grad()
def solve_turn_batch(
    states_batch: dict[str, torch.Tensor],
    c: int,
    spec: Any,
    game_config: Any,
    iterations: int,
    predictor: Any,
    device: torch.device | str = "cuda",
    solver_cfg: Any = None,
    tree: Any = None,
    leaf_every: int = 1,
) -> dict[str, torch.Tensor]:
    """Solve one batch of turn-root states at ``c`` and return its rows in the
    shard format (CPU).

    ``states_batch`` holds ``boards [B, 4]``, ``ranges [B, 2, 1326]`` in
    ``(OOP, IP)`` order and optionally ``source``. ``predictor`` values the
    turn-end leaves (a turn-end net, or a river net averaged over the river
    cards). ``tree`` reuses a turn tree built for this ``c`` (engine
    ``game_config``)."""
    dev = torch.device(device)
    c = int(c)
    start = int(vr._game(game_config)["stacks"][0])
    if tree is None:
        tree = turn_tree(game_config, c, spec, button=1, device=dev)
    boards = states_batch["boards"].long().to(dev)
    ranges = quantise(states_batch["ranges"].float(), boards).to(dev)
    B = boards.shape[0]
    solver = BatchTurnSolver(
        tree, boards, ranges, predictor, solver_cfg, device=dev, leaf_every=leaf_every
    )
    solver.solve(int(iterations))
    valid = vr.board_valid(boards)
    pot = 2.0 * c
    targets = []
    for p in (0, 1):
        v = solver.root_values(p, best_response=True).float()
        m = blocked_sum(ranges[:, 1 - p])
        ok = valid & (m > MASS_EPS)
        targets.append(torch.where(ok, v / m.clamp(min=MASS_EPS) / pot, 0.0))
    ex = solver.exploitability()["exploitability"].reshape(B).float() / pot
    source = states_batch.get("source")
    if source is None:
        source = torch.zeros(B, dtype=torch.uint8)
    out = {
        "boards": boards,
        "c": torch.full((B,), c),
        "stack": torch.full((B,), start - c),
        "ranges": ranges,
        "targets": torch.stack(targets, 1),
        "exploit": ex,
        "source": source,
    }
    return {k: v.to("cpu", SHARD_DTYPES[k]) for k, v in out.items()}


def _solver_config(overrides: dict[str, Any]) -> Any:
    if not overrides:
        return None
    from .solver import SolverConfig

    return SolverConfig(**overrides)


def solve_turn_states(
    states: dict[str, torch.Tensor],
    spec: Any,
    game_config: Any,
    iterations: int,
    batch_size: int,
    predictor: Any,
    device: torch.device | str,
    solver_cfg: Any = None,
    trees: TurnTreeCache | None = None,
    generator: torch.Generator | None = None,
    leaf_every: int = 1,
) -> dict[str, torch.Tensor]:
    """Solve all turn-root ``states`` in ``c``-sorted batches
    (:func:`~.value_data.make_batches`; ``generator`` draws each batch's ``c``
    log-uniformly within the batch, else the median); rows in input order."""
    game = vr._game(game_config)
    start = int(game["stacks"][0])
    n = states["c"].shape[0]
    out: dict[str, torch.Tensor] = {}
    batches = make_batches(states, batch_size, game["big_blind"], start - 1, generator)
    for idx, c in batches:
        tree = trees.get(c) if trees is not None else None
        rows = solve_turn_batch(
            take(states, idx),
            c,
            spec,
            game_config,
            iterations,
            predictor,
            device,
            solver_cfg,
            tree,
            leaf_every,
        )
        if not out:
            out = {k: torch.empty((n, *v.shape[1:]), dtype=v.dtype) for k, v in rows.items()}
        for k, v in rows.items():
            out[k][idx] = v
    return out


# --------------------------------------------------------------------------- shards


@dataclass
class TurnStartGenConfig:
    samples: int = 20_000
    mix: tuple[float, float, float] = (0.5, 0.25, 0.25)  # self-play, perturbed, random
    iterations: int = 300  # DCFR iterations per turn solve
    batch: int = 256  # instances per solve
    seed: int = 0
    shard_size: int = 4096
    explore: float = 0.0  # uniform mixing of the self-play behaviour policy
    n_envs: int = 4096
    c_range: tuple[int, int] = (vr.C_MIN, vr.C_MAX)  # random states' c (log-uniform)
    leaf_every: int = 1  # run the turn-end net every n regret updates (1 = exact)
    solver: dict[str, Any] = field(default_factory=dict)  # SolverConfig overrides

    def __post_init__(self) -> None:
        self.mix = tuple(float(x) for x in self.mix)
        if len(self.mix) != 3 or min(self.mix) < 0 or sum(self.mix) <= 0:
            raise ValueError(f"mix must be three non-negative weights, got {self.mix}")
        self.c_range = (int(self.c_range[0]), int(self.c_range[1]))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mix"] = list(self.mix)
        d["c_range"] = list(self.c_range)
        return d


# fields that must match when a run is resumed (samples may grow)
RESUME_KEYS = (
    "mix",
    "iterations",
    "batch",
    "seed",
    "shard_size",
    "explore",
    "n_envs",
    "c_range",
    "leaf_every",
    "solver",
)


def generate_turn_start_data(
    turn_predictor: Any,
    bp: Any,
    out: str | Path,
    cfg: TurnStartGenConfig,
    game_config: Any = None,
    device: torch.device | str = "cuda",
    resume: bool = False,
    turn_net: str | None = None,
    blueprint_path: str | None = None,
    spec: Any = None,
    log: Callable[[str], Any] | None = print,
) -> dict[str, Any]:
    """Generate ``cfg.samples`` solved turn-root states into shards under
    ``out`` (see the module docstring). ``turn_predictor`` values the turn-end
    leaves. ``spec`` (the turn tree's actions) defaults to the blueprint's;
    ``game_config`` to the blueprint's game (``bp.game``), or 100bb at 50/100
    without a blueprint. Without ``resume`` an existing ``meta.json`` is an
    error; with it, the settings must match and existing shards are skipped.
    Returns the meta dict."""
    from ..blueprint.deepcfr.config import spec_to_dict

    log = log or (lambda *_a, **_k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if spec is None:
        if bp is None:
            raise ValueError("pass spec (the turn tree's actions) when there is no blueprint")
        spec = bp.spec
    if game_config is None:
        game_config = bp.game if bp is not None and hasattr(bp, "game") else DEFAULT_GAME
    game = vr._game(game_config)
    engine_cfg = vr.engine_game_config(game)
    spec_d = spec_to_dict(spec)
    meta_path = out / "meta.json"
    if meta_path.exists():
        if not resume:
            raise FileExistsError(f"{meta_path} exists; pass resume=True (--resume) to continue")
        meta = json.loads(meta_path.read_text())
        old, new = meta.get("config", {}), cfg.to_dict()
        bad = [k for k in RESUME_KEYS if old.get(k) != new.get(k)]
        if meta.get("kind") != START_KIND:
            bad.append("kind")
        if meta.get("spec") != spec_d:
            bad.append("spec")
        if bad:
            raise ValueError(f"cannot resume: settings differ from meta.json: {bad}")
        meta["config"]["samples"] = cfg.samples
    else:
        meta = {
            "format": FORMAT_VERSION,
            "kind": START_KIND,
            "turn_net": turn_net,
            "blueprint": blueprint_path,
            "spec": spec_d,
            "game": game,
            "config": cfg.to_dict(),
            "keys": {k: str(v).replace("torch.", "") for k, v in SHARD_DTYPES.items()},
            "boards": "[n, 4] turn boards at the turn root (before turn betting)",
            "sources": {"0": "self-play", "1": "perturbed self-play", "2": "random"},
            "targets": "turn-root best-response cfv / opponent disjoint mass / (2c), (OOP, IP); "
            "turn solves with turn-end net leaves",
            "shards": {},
        }
    write_atomic(meta_path, meta, as_json=True)
    dev = torch.device(device)
    trees = TurnTreeCache(engine_cfg, spec, dev)
    solver_cfg = _solver_config(cfg.solver)
    n_shards = math.ceil(cfg.samples / cfg.shard_size)
    t_run = time.time()
    done_now = 0
    for k in range(n_shards):
        path = out / shard_name(k)
        if path.exists():
            continue
        n = min(cfg.shard_size, cfg.samples - k * cfg.shard_size)
        seeds = shard_seeds(cfg.seed, k)
        t0 = time.time()
        sp_stats: dict[str, float] = {}
        states = make_turn_states(
            bp,
            engine_cfg,
            n,
            cfg.mix,
            seeds,
            dev,
            cfg.explore,
            cfg.n_envs,
            cfg.c_range,
            sp_stats,
            log,
            turn_start=True,
        )
        t1 = time.time()
        c_gen = torch.Generator().manual_seed(seeds[2] + 1)
        rows = solve_turn_states(
            states,
            spec,
            engine_cfg,
            cfg.iterations,
            cfg.batch,
            turn_predictor,
            dev,
            solver_cfg,
            trees,
            c_gen,
            cfg.leaf_every,
        )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = time.time()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seeds[2]))
        rows = {key: v[perm].contiguous() for key, v in rows.items()}
        if not bool(torch.isfinite(rows["targets"].float()).all()):
            raise FloatingPointError(f"non-finite targets in shard {k}")
        write_atomic(path, rows)
        dt = time.time() - t0
        ex = exploit_summary(rows["exploit"])
        c_med = float(rows["c"].double().median())
        info = {
            "samples": n,
            "seeds": seeds,
            "seconds": dt,
            "states_s": t1 - t0,
            "solve_s": t2 - t1,
            **ex,
            "c_median": c_med,
            "selfplay": {
                key: (round(v, 3) if isinstance(v, float) else v) for key, v in sp_stats.items()
            },
        }
        meta["shards"][shard_name(k)] = info
        done_now += n
        total_s = time.time() - t_run
        meta["timing"] = {
            "samples_written": sum(int(s["samples"]) for s in meta["shards"].values()),
            "last_run_seconds": total_s,
            "last_run_samples": done_now,
            "last_run_samples_per_s": done_now / max(total_s, 1e-9),
        }
        write_atomic(meta_path, meta, as_json=True)
        log(
            f"# {shard_name(k)} ({k + 1}/{n_shards}): {n} samples in {dt:.1f}s "
            f"({n / dt:.1f}/s; states {t1 - t0:.1f}s, solve {t2 - t1:.1f}s), exploit/pot "
            f"mean {ex['exploit_mean']:.4f} p90 {ex['exploit_p90']:.4f} "
            f"max {ex['exploit_max']:.4f}, c median {c_med:.0f}; "
            f"run {done_now / max(total_s, 1e-9):.1f} samples/s"
        )
    return meta


__all__ = [
    "TurnStartGenConfig",
    "TurnTreeCache",
    "generate_turn_start_data",
    "solve_turn_batch",
    "solve_turn_states",
]
