"""Training data for the leaf value net: batched exact river solves
(``docs/value_net.md``, section 3).

Pipeline per shard (``scripts/gen_value_data.py``):

1. **States** (:func:`make_states`): a ``mix`` of blueprint self-play river
   states (source 0), perturbed copies of further self-play ranges (source 1)
   and DeepStack-style random ranges on random boards with log-uniform ``c``
   (source 2); see :mod:`.value_ranges`.
2. **Batches** (:func:`make_batches`): the states are sorted by ``c`` and
   chunked; each batch is solved at its median ``c`` (the river tree depends
   only on ``c``, and the ranges are inputs only, so moving a sample's ``c`` a
   little is fine). The stored ``c`` and ``stack`` are the solved ones.
3. **Solve** (:func:`solve_batch`): ``BatchRiverSolver`` on the river-root
   tree of the blueprint's action spec, then the best-response values of each
   player against the other's average strategy, per unit of opponent mass and
   pot.

Shard format: a directory of ``shard_XXXXX.pt`` files, each a ``torch.save``
of a dict (``n`` samples, shuffled):

* ``boards`` uint8 ``[n, 5]``;
* ``c`` int32 ``[n]``: chips each player has committed at the river root;
* ``stack`` int32 ``[n]``: chips behind (start stack minus ``c``);
* ``ranges`` float16 ``[n, 2, 1326]``: normalised ``(OOP, IP)`` ranges, zero
  on board conflicts (exactly the values the solver was run on);
* ``targets`` float16 ``[n, 2, 1326]``: ``v_p(c) / m_{-p}(c) / (2c)`` with
  ``v_p`` the best-response counterfactual value in chips and ``m_{-p}`` the
  opponent's range mass disjoint from the combo; zero on board conflicts and
  where ``m_{-p} <= 1e-9``;
* ``exploit`` float32 ``[n]``: the instance's exploitability in pot units;
* ``source`` uint8 ``[n]``: 0 self-play, 1 perturbed self-play, 2 random.

plus ``meta.json`` (spec, iterations, mix, seeds, blueprint path, per-shard
timing and exploitability). Shards are written atomically, generation is
deterministic per shard (seeded from ``(seed, shard index)``), and a resumed
run skips the shards that exist.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import value_ranges as vr
from .combos import blocked_sum

SOURCE_SELFPLAY, SOURCE_PERTURBED, SOURCE_RANDOM = 0, 1, 2
SHARD_DTYPES = {
    "boards": torch.uint8,
    "c": torch.int32,
    "stack": torch.int32,
    "ranges": torch.float16,
    "targets": torch.float16,
    "exploit": torch.float32,
    "source": torch.uint8,
}
MASS_EPS = 1e-9
FORMAT_VERSION = 1


@dataclass
class GenConfig:
    samples: int = 20_000
    mix: tuple[float, float, float] = (0.5, 0.25, 0.25)  # self-play, perturbed, random
    iterations: int = 400
    batch: int = 512
    seed: int = 0
    shard_size: int = 4096
    explore: float = 0.0  # uniform mixing of the self-play behaviour policy
    n_envs: int = 4096
    c_range: tuple[int, int] = (vr.C_MIN, vr.C_MAX)  # random states' c (log-uniform)
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
RESUME_KEYS = ("mix", "iterations", "batch", "seed", "shard_size", "explore", "n_envs", "c_range")


def shard_seeds(seed: int, shard: int) -> list[int]:
    """Three independent 31-bit seeds of shard ``shard`` (self-play,
    perturbation, random states)."""
    st = np.random.SeedSequence([int(seed), int(shard)]).generate_state(3, np.uint32)
    return [int(x) & 0x7FFFFFFF for x in st]


def mix_counts(n: int, mix: tuple[float, float, float]) -> tuple[int, int, int]:
    """Split ``n`` samples by the ``mix`` weights (largest remainders)."""
    w = np.asarray(mix, dtype=np.float64)
    raw = w / w.sum() * n
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: n - counts.sum()]:
        counts[i] += 1
    return int(counts[0]), int(counts[1]), int(counts[2])


def make_states(
    bp: Any,
    game_config: Any,
    n: int,
    mix: tuple[float, float, float],
    seeds: list[int],
    device: torch.device | str = "cpu",
    explore: float = 0.0,
    n_envs: int = 4096,
    c_range: tuple[int, int] = (vr.C_MIN, vr.C_MAX),
    stats: dict[str, float] | None = None,
    log: Callable[[str], Any] | None = None,
) -> dict[str, torch.Tensor]:
    """``n`` river states (CPU): ``boards``, ``c``, ``stack``, ``ranges`` and
    ``source``, in source order (self-play, perturbed, random)."""
    dev = torch.device(device)
    n_sp, n_pert, n_rand = mix_counts(n, mix)
    game = vr._game(game_config)
    start = int(game["stacks"][0])
    parts = []
    if n_sp + n_pert:
        sp = vr.selfplay_river_states(
            bp,
            game_config,
            n_sp + n_pert,
            dev,
            seed=seeds[0],
            explore=explore,
            n_envs=n_envs,
            stats=stats,
            log=log,
        )
        src = torch.full((n_sp + n_pert,), SOURCE_SELFPLAY, dtype=torch.uint8)
        ranges = sp["ranges"]
        if n_pert:
            g = torch.Generator(device=dev).manual_seed(seeds[1])
            pert = vr.perturb_ranges(ranges[n_sp:].to(dev), sp["boards"][n_sp:].to(dev), g)
            ranges = torch.cat([ranges[:n_sp], pert.cpu()])
            src[n_sp:] = SOURCE_PERTURBED
        parts.append(
            {
                "boards": sp["boards"],
                "c": sp["c"],
                "stack": sp["stack"],
                "ranges": ranges,
                "source": src,
            }
        )
    if n_rand:
        g = torch.Generator(device=dev).manual_seed(seeds[2])
        rs = vr.random_states(n_rand, g, start, c_range[0], c_range[1], game["big_blind"])
        rs = {k: v.cpu() for k, v in rs.items()}
        rs["source"] = torch.full((n_rand,), SOURCE_RANDOM, dtype=torch.uint8)
        parts.append(rs)
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    out["boards"] = out["boards"].long()
    out["c"] = out["c"].long()
    out["stack"] = out["stack"].long()
    out["ranges"] = out["ranges"].float()
    return out


def take(states: dict[str, torch.Tensor], idx: torch.Tensor) -> dict[str, torch.Tensor]:
    return {k: v[idx] for k, v in states.items()}


def make_batches(
    states: dict[str, torch.Tensor],
    batch_size: int,
    big_blind: int = 100,
    max_c: int | None = None,
) -> list[tuple[torch.Tensor, int]]:
    """Sort the states by ``c`` and chunk them into batches of ``batch_size``:
    a list of ``(indices, c)`` where ``c`` is the batch's median rounded to a
    chip amount (the ``c`` the batch is solved at). The median is snapped to a
    reachable river-root amount (:func:`.value_ranges.reachable_c`: ``bb`` or
    at least ``2 * bb``, so at least 100 at 50/100) and capped at ``max_c``."""
    c = states["c"].long()
    order = torch.argsort(c, stable=True)
    out = []
    for lo in range(0, order.numel(), int(batch_size)):
        idx = order[lo : lo + int(batch_size)]
        med = int(round(float(torch.quantile(c[idx].double(), 0.5))))
        cc = int(vr.reachable_c(torch.tensor(med), big_blind))
        if max_c is not None:
            cc = min(cc, int(max_c))
        out.append((idx, cc))
    return out


def quantise_ranges(ranges: torch.Tensor) -> torch.Tensor:
    """Normalised ranges rounded to float16 (the stored values), as float32."""
    r = ranges.double()
    r = r / r.sum(-1, keepdim=True)
    return r.half().float()


def solve_batch(
    states_batch: dict[str, torch.Tensor],
    c: int,
    spec: Any,
    game_config: Any,
    iterations: int,
    device: torch.device | str = "cuda",
    solver_cfg: Any = None,
    tree: Any = None,
) -> dict[str, torch.Tensor]:
    """Solve one batch at ``c`` and return its rows in the shard format (CPU).

    ``states_batch`` holds ``boards [B, 5]``, ``ranges [B, 2, 1326]`` in
    ``(OOP, IP)`` order and optionally ``source``. The tree is the river-root
    tree with the button in seat 1, so seat order is ``(OOP, IP)``. ``tree``
    reuses a tree built for this ``c``.
    """
    from .batch_solver import BatchRiverSolver, river_tree

    dev = torch.device(device)
    c = int(c)
    start = int(vr._game(game_config)["stacks"][0])
    if tree is None:
        tree = river_tree(game_config, c, spec, button=1, device=dev)
    boards = states_batch["boards"].long().to(dev)
    ranges = quantise_ranges(states_batch["ranges"].to(dev))
    B = boards.shape[0]
    solver = BatchRiverSolver(tree, boards, ranges, solver_cfg, device=dev)
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


class TreeCache:
    """River trees by ``c`` (LRU), built with ``river_tree``."""

    def __init__(self, game_config: Any, spec: Any, device: Any, size: int = 64) -> None:
        self.game_config = game_config
        self.spec = spec
        self.device = device
        self.size = int(size)
        self._trees: OrderedDict[int, Any] = OrderedDict()

    def get(self, c: int) -> Any:
        from .batch_solver import river_tree

        c = int(c)
        if c in self._trees:
            self._trees.move_to_end(c)
            return self._trees[c]
        tree = river_tree(self.game_config, c, self.spec, button=1, device=self.device)
        self._trees[c] = tree
        while len(self._trees) > self.size:
            self._trees.popitem(last=False)
        return tree


def _solver_config(overrides: dict[str, Any]) -> Any:
    if not overrides:
        return None
    from .solver import SolverConfig

    return SolverConfig(**overrides)


def solve_states(
    states: dict[str, torch.Tensor],
    spec: Any,
    game_config: Any,
    iterations: int,
    batch_size: int,
    device: torch.device | str,
    solver_cfg: Any = None,
    trees: TreeCache | None = None,
) -> dict[str, torch.Tensor]:
    """Solve all ``states`` in ``c``-sorted batches; rows in the input order."""
    game = vr._game(game_config)
    start = int(game["stacks"][0])
    n = states["c"].shape[0]
    out: dict[str, torch.Tensor] = {}
    for idx, c in make_batches(states, batch_size, game["big_blind"], max_c=start - 1):
        tree = trees.get(c) if trees is not None else None
        rows = solve_batch(
            take(states, idx), c, spec, game_config, iterations, device, solver_cfg, tree
        )
        if not out:
            out = {k: torch.empty((n, *v.shape[1:]), dtype=v.dtype) for k, v in rows.items()}
        for k, v in rows.items():
            out[k][idx] = v
    return out


def shard_name(k: int) -> str:
    return f"shard_{k:05d}.pt"


def write_atomic(path: Path, obj: Any, as_json: bool = False) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if as_json:
        tmp.write_text(json.dumps(obj, indent=1))
    else:
        torch.save(obj, tmp)
    os.replace(tmp, path)


def exploit_summary(exploit: torch.Tensor) -> dict[str, float]:
    e = exploit.double()
    return {
        "exploit_mean": float(e.mean()),
        "exploit_p90": float(e.quantile(0.9)),
        "exploit_max": float(e.max()),
    }


def _check_resume(meta: dict[str, Any], cfg: GenConfig, spec: dict[str, Any]) -> None:
    old = meta.get("config", {})
    new = cfg.to_dict()
    bad = [k for k in RESUME_KEYS if old.get(k) != new.get(k)]
    if meta.get("spec") != spec:
        bad.append("spec")
    if bad:
        diff = {k: (old.get(k), new.get(k)) for k in bad if k != "spec"}
        raise ValueError(f"cannot resume: settings differ from meta.json: {bad} {diff}")


def generate(
    bp: Any,
    out: str | Path,
    cfg: GenConfig,
    game_config: Any = None,
    device: torch.device | str = "cuda",
    resume: bool = False,
    blueprint_path: str | None = None,
    log: Callable[[str], Any] | None = print,
) -> dict[str, Any]:
    """Generate ``cfg.samples`` solved river states into shards under ``out``.

    ``game_config`` defaults to the blueprint's game (``bp.game``). Without
    ``resume`` an existing ``meta.json`` is an error; with it, the settings
    must match and existing shards are skipped. Returns the updated meta dict.
    """
    from ..blueprint.deepcfr.config import spec_to_dict

    log = log or (lambda *_a, **_k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    game_config = game_config if game_config is not None else bp.game
    game = vr._game(game_config)
    engine_cfg = vr.engine_game_config(game)
    spec = bp.spec
    spec_d = spec_to_dict(spec)
    meta_path = out / "meta.json"
    if meta_path.exists():
        if not resume:
            raise FileExistsError(f"{meta_path} exists; pass resume=True (--resume) to continue")
        meta = json.loads(meta_path.read_text())
        _check_resume(meta, cfg, spec_d)
        if blueprint_path is not None and meta.get("blueprint") != str(blueprint_path):
            log(f"# note: blueprint path {blueprint_path} differs from {meta.get('blueprint')}")
        meta["config"]["samples"] = cfg.samples
    else:
        meta = {
            "format": FORMAT_VERSION,
            "blueprint": None if blueprint_path is None else str(blueprint_path),
            "spec": spec_d,
            "game": game,
            "config": cfg.to_dict(),
            "keys": {k: str(v).replace("torch.", "") for k, v in SHARD_DTYPES.items()},
            "sources": {"0": "self-play", "1": "perturbed self-play", "2": "random"},
            "targets": "best-response cfv / opponent disjoint mass / (2c), (OOP, IP)",
            "shards": {},
        }
    write_atomic(meta_path, meta, as_json=True)
    dev = torch.device(device)
    trees = TreeCache(engine_cfg, spec, dev)
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
        states = make_states(
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
        )
        t1 = time.time()
        rows = solve_states(
            states, spec, engine_cfg, cfg.iterations, cfg.batch, dev, solver_cfg, trees
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
        written = sum(int(s["samples"]) for s in meta["shards"].values())
        meta["timing"] = {
            "samples_written": written,
            "last_run_seconds": total_s,
            "last_run_samples": done_now,
            "last_run_samples_per_s": done_now / max(total_s, 1e-9),
        }
        write_atomic(meta_path, meta, as_json=True)
        kept = sp_stats.get("kept", 0) / max(1, sp_stats.get("hands", 0))
        log(
            f"# {shard_name(k)} ({k + 1}/{n_shards}): {n} samples in {dt:.1f}s "
            f"({n / dt:.1f}/s; states {t1 - t0:.1f}s, solve {t2 - t1:.1f}s), exploit/pot "
            f"mean {ex['exploit_mean']:.4f} p90 {ex['exploit_p90']:.4f} "
            f"max {ex['exploit_max']:.4f}, c median {c_med:.0f}, self-play kept {kept:.3f}; "
            f"run {done_now / max(total_s, 1e-9):.1f} samples/s"
        )
    return meta
