"""Turn-end value-net training data, bootstrapped from a river net (no solving).

DeepStack's auxiliary-net trick: the turn-end net ``N_TE`` (:mod:`.turn_net`)
learns the river-card average of the river net ``N_R``, which is exactly what
:class:`~.value_leaf.ValueLeafEvaluator` computes at a flop search's leaves:

    ev_TE_p(c) = v_p(c) / (m_-p(c) * pot),
    v_p(c) = sum_x (1/44) [c avoids x] m^x_-p(c) * pot * N_R(b4 + x, r masked by x)_p(c)

``v_p`` comes from :func:`~.value_leaf.river_average` (48 river-net rows per
state), ``m_-p = blocked_sum(r_-p)`` on the 4-card board and ``pot = 2c``. With
:class:`~.value_leaf.ShowdownOracle` as ``N_R`` the targets are the exact
values of a checked-down river (:func:`turn_checkdown_samples`, tests).

**States** (:func:`make_turn_states`), as for river data (:mod:`.value_data`):
a ``mix`` of blueprint self-play stopped at the end of turn betting (source 0,
:func:`~.value_ranges.selfplay_river_states` with ``turn_end=True``),
perturbed copies of more self-play ranges (source 1) and DeepStack-style
random ranges on random 4-card boards (source 2, recursive splits along the
turn-end strength order, :func:`random_turn_states`). Turn-end strength is
the mean river percentile of :mod:`.turn_net`.

**Shards** (:func:`generate_turn_data`, ``scripts/gen_turn_data.py``) have the
river-shard format of :mod:`.value_data` (:data:`~.value_data.SHARD_DTYPES`)
except for the boards:

* ``boards`` uint8 ``[n, 4]``: the turn board (no river card);
* ``c``, ``stack`` int32 ``[n]``: chips committed by each player / behind;
* ``ranges`` float16 ``[n, 2, 1326]``: normalised ``(OOP, IP)`` ranges, zero
  on combos that hit the 4-card board (the targets are computed on exactly
  these values);
* ``targets`` float16 ``[n, 2, 1326]``: ``ev_TE`` above, zero on board
  conflicts and where ``m_-p <= 1e-9``;
* ``exploit`` float32 ``[n]``: 0 (nothing is solved);
* ``source`` uint8 ``[n]``: 0 self-play, 1 perturbed self-play, 2 random;

plus ``meta.json`` with ``kind: "turn_end"`` and the river net used. Shards are
written atomically and seeded per ``(seed, shard index)``; ``resume`` skips
existing shards. Train with :func:`.turn_net.train_turn_net`
(``scripts/train_turn_net.py``).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from . import value_ranges as vr
from .combos import NUM_COMBOS, blocked_sum
from .turn_net import KIND, TURN_VALID, turn_rank_tables
from .value_data import (
    SHARD_DTYPES,
    SOURCE_PERTURBED,
    SOURCE_RANDOM,
    SOURCE_SELFPLAY,
    mix_counts,
    shard_name,
    shard_seeds,
    write_atomic,
)
from .value_leaf import ShowdownOracle, river_average

C = NUM_COMBOS
MASS_EPS = 1e-9
FORMAT_VERSION = 1
DEFAULT_GAME = {"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100, "ante": 0}


# --------------------------------------------------------------------------- targets


def _normalise(ranges: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Rows normalised to sum 1 over ``valid`` in their own dtype (empty rows stay 0)."""
    r = ranges * valid[:, None, :]
    s = r.sum(-1, keepdim=True)
    return torch.where(s > 0, r / s.clamp(min=torch.finfo(r.dtype).tiny), torch.zeros_like(r))


@torch.no_grad()
def turn_targets(
    river_predictor: Any,
    boards: torch.Tensor,
    ranges: torch.Tensor,
    c: torch.Tensor,
    stack: torch.Tensor,
    chunk: int = 16384,
) -> torch.Tensor:
    """Bootstrapped turn-end targets ``ev_TE [n, 2, 1326]`` (pot units per unit
    of disjoint opponent mass, ``(OOP, IP)``) of ``boards [n, 4]``, ``ranges
    [n, 2, 1326]`` (``(OOP, IP)``, any non-negative scale), ``c`` and ``stack``,
    from a river predictor (``predict(boards [m, 5], ...)``). Computed in the
    dtype of ``ranges``; zero on board conflicts and where ``m_-p <= 1e-9``
    (ranges normalised)."""
    dev = ranges.device
    b = torch.as_tensor(boards, device=dev).long()
    valid = vr.board_valid(b)
    r = _normalise(ranges, valid)
    c = torch.as_tensor(c, device=dev).reshape(-1)
    stack = torch.as_tensor(stack, device=dev).reshape(-1)
    v = river_average(river_predictor, b, c, stack, r.transpose(0, 1), True, chunk)  # [2, n, C]
    m = torch.stack([blocked_sum(r[:, 1]), blocked_sum(r[:, 0])], 1)  # [n, 2, C]
    pot = 2.0 * c.to(r.dtype)[:, None, None]
    ok = valid[:, None, :] & (m > MASS_EPS)
    ev = v.transpose(0, 1) / (m.clamp(min=MASS_EPS) * pot)
    return torch.where(ok, ev, torch.zeros_like(ev))


class RiverAveragePredictor:
    """An exact turn-end predictor (``kind = "turn_end"``): the river-card
    average of a river predictor, :func:`turn_targets`. With
    :class:`~.value_leaf.ShowdownOracle` it is the exact check-down turn-end
    value; as the provider of :class:`~.value_leaf.TurnEndLeafEvaluator` it
    reproduces :class:`~.value_leaf.ValueLeafEvaluator` on the same river net."""

    kind = KIND

    def __init__(self, river: Any, chunk: int = 16384) -> None:
        self.river = river
        self.chunk = chunk

    def predict(
        self, boards: torch.Tensor, ranges: torch.Tensor, c: torch.Tensor, stack: torch.Tensor
    ) -> torch.Tensor:
        return turn_targets(self.river, boards, ranges, c, stack, self.chunk)


# --------------------------------------------------------------------------- states


def turn_strength(boards4: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``[n, 4]`` boards -> ``(key, pct)`` ``[n, 1326]``: the turn-end strength
    rank (mean river percentile, ``-1`` on board conflicts; a strength order for
    :func:`~.value_ranges.random_ranges`) and its percentile among the valid
    combos (0 on conflicts; for :func:`~.value_ranges.perturb_ranges`)."""
    mrank2, _ = turn_rank_tables(boards4)
    pct = mrank2.clamp(min=0).float() / (2.0 * (TURN_VALID - 1))
    return mrank2, pct


def random_turn_states(
    n: int,
    generator: torch.Generator | None = None,
    start_stack: int = 10000,
    c_lo: int = vr.C_MIN,
    c_hi: int = vr.C_MAX,
    big_blind: int = 100,
) -> dict[str, torch.Tensor]:
    """``n`` turn-end states: random 4-card boards, log-uniform ``c``
    (:func:`~.value_ranges.random_c`) and independent DeepStack-style random
    ranges along the turn-end strength order for both players (on the
    generator's device)."""
    boards = vr.random_boards(n, generator)[:, :4]
    c = vr.random_c(n, generator, c_lo, c_hi, big_blind=big_blind)
    key, _ = turn_strength(boards)
    r0 = vr.random_ranges(boards, generator, strengths=key)
    r1 = vr.random_ranges(boards, generator, strengths=key)
    return {
        "boards": boards,
        "c": c,
        "stack": int(start_stack) - c,
        "ranges": torch.stack([r0, r1], 1),
    }


def make_turn_states(
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
    turn_start: bool = False,
) -> dict[str, torch.Tensor]:
    """``n`` turn-end states (CPU): ``boards [n, 4]``, ``c``, ``stack``,
    ``ranges`` and ``source``, in source order (self-play, perturbed, random).
    ``bp`` may be ``None`` when ``mix`` has no self-play share. ``turn_start``
    stops self-play at the turn root instead (turn-start states,
    :mod:`.turn_start_data`); the random states are the same either way."""
    dev = torch.device(device)
    n_sp, n_pert, n_rand = mix_counts(n, mix)
    game = vr._game(game_config)
    start = int(game["stacks"][0])
    parts = []
    if n_sp + n_pert:
        if bp is None:
            raise ValueError("self-play states need a blueprint")
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
            turn_end=not turn_start,
            turn_start=turn_start,
        )
        src = torch.full((n_sp + n_pert,), SOURCE_SELFPLAY, dtype=torch.uint8)
        ranges = sp["ranges"]
        if n_pert:
            g = torch.Generator(device=dev).manual_seed(seeds[1])
            b = sp["boards"][n_sp:].to(dev)
            _, pct = turn_strength(b)
            pert = vr.perturb_ranges(ranges[n_sp:].to(dev), b, g, pct=pct)
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
        rs = random_turn_states(n_rand, g, start, c_range[0], c_range[1], game["big_blind"])
        rs = {k: v.cpu() for k, v in rs.items()}
        rs["source"] = torch.full((n_rand,), SOURCE_RANDOM, dtype=torch.uint8)
        parts.append(rs)
    out = {k: torch.cat([p[k] for p in parts]) for k in parts[0]}
    out["boards"] = out["boards"].long()
    out["c"] = out["c"].long()
    out["stack"] = out["stack"].long()
    out["ranges"] = out["ranges"].float()
    return out


def quantise(ranges: torch.Tensor, boards: torch.Tensor) -> torch.Tensor:
    """Ranges normalised over the board's valid combos and rounded to float16
    (the stored values), as float32."""
    valid = vr.board_valid(boards.to(ranges.device))
    return _normalise(ranges.double(), valid).half().float()


@torch.no_grad()
def label_states(
    river_predictor: Any,
    states: dict[str, torch.Tensor],
    device: torch.device | str = "cpu",
    chunk: int = 16384,
    batch: int = 2048,
) -> dict[str, torch.Tensor]:
    """Shard rows (CPU, :data:`~.value_data.SHARD_DTYPES`) of turn-end ``states``:
    quantised ranges and their bootstrapped targets, ``batch`` states at a time."""
    dev = torch.device(device)
    n = states["c"].shape[0]
    boards = states["boards"].long()
    ranges = quantise(states["ranges"].float(), boards)
    targets = torch.empty(n, 2, C, dtype=torch.float32)
    for s in range(0, n, batch):
        sl = slice(s, s + batch)
        t = turn_targets(
            river_predictor,
            boards[sl].to(dev),
            ranges[sl].to(dev),
            states["c"][sl].to(dev),
            states["stack"][sl].to(dev),
            chunk,
        )
        targets[sl] = t.float().cpu()
    source = states.get("source")
    if source is None:
        source = torch.zeros(n, dtype=torch.uint8)
    out = {
        "boards": boards,
        "c": states["c"],
        "stack": states["stack"],
        "ranges": ranges,
        "targets": targets,
        "exploit": torch.zeros(n),
        "source": source,
    }
    return {k: v.to("cpu", SHARD_DTYPES[k]) for k, v in out.items()}


@torch.no_grad()
def turn_checkdown_samples(
    n: int,
    seed: int = 0,
    device: torch.device | str = "cpu",
    boards: int | None = None,
    total: int = 10000,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    """``n`` synthetic turn-end samples (shard format, float tensors, CPU) whose
    river is checked down: targets from :func:`turn_targets` with
    :class:`~.value_leaf.ShowdownOracle`. ``boards`` distinct random turn
    boards (default ``n``) shared round-robin; log-uniform ``c`` in
    ``[100, total / 2]``. Range families (``source``): 0 uniform with
    log-normal noise and spread (draw / made-hand) tilts, 1 strength-correlated
    (tilts and percentile bands of the turn-end strength, with noise), 2
    DeepStack random splits along the turn-end strength order, perturbed."""
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, device=dev)

    nb = n if boards is None else max(1, int(boards))
    bset = rand(nb, 52).argsort(1)[:, :4]
    b = bset[torch.arange(n, device=dev) % nb]
    from .turn_net import turn_board_features

    f = turn_board_features(bset, 2)
    valid = f["valid"][torch.arange(n, device=dev) % nb]
    pct = f["pct"][torch.arange(n, device=dev) % nb]
    spread = f["vrank2"][torch.arange(n, device=dev) % nb].clamp(min=0).float() / (
        2.0 * (TURN_VALID - 1)
    )
    key = f["mrank2"][torch.arange(n, device=dev) % nb]
    fam = torch.randint(0, 3, (n,), generator=g, device=dev)
    v = valid[:, None, :].float().expand(n, 2, C)
    noise = torch.exp(rand(n, 2, 1) * torch.randn(n, 2, C, generator=g, device=dev))
    # 0: uniform with noise and a tilt by the river spread (draw-heavy or made-hand-heavy)
    gam = (rand(n, 2, 1) * 2 - 1) * 6
    uni = noise * torch.exp(gam * (spread[:, None, :] - 0.5))
    # 1: strength-correlated: tilt exp(beta * pct) or a band [lo, lo + width], with noise
    beta = (rand(n, 2, 1) * 2 - 1) * 12
    tilt = torch.exp(beta * (pct[:, None, :] - 0.5))
    lo = rand(n, 2, 1) * 0.8
    band = ((pct[:, None, :] >= lo) & (pct[:, None, :] <= lo + 0.05 + rand(n, 2, 1) * 0.5)).float()
    strength = torch.where(rand(n, 2, 1) < 0.5, tilt, band + 0.01) * noise
    # 2: random splits along the strength order, perturbed (with the turn pct)
    rr = torch.stack(
        [vr.random_ranges(b, g, strengths=key), vr.random_ranges(b, g, strengths=key)], 1
    )
    rr = vr.perturb_ranges(rr, b, g, pct=pct)
    r = torch.where(
        (fam == 0)[:, None, None], uni, torch.where((fam == 1)[:, None, None], strength, rr)
    )
    ranges = _normalise((r * v).to(dtype), valid)
    c = torch.exp(math.log(100) + rand(n) * math.log(total / 200)).round().long()
    stack = total - c
    oracle = ShowdownOracle(max_boards=1 << 14)
    targets = turn_targets(oracle, b, ranges, c, stack)
    return {
        "boards": b.cpu(),
        "c": c.cpu(),
        "stack": stack.cpu(),
        "ranges": ranges.cpu(),
        "targets": targets.cpu(),
        "exploit": torch.zeros(n),
        "source": fam.to(torch.uint8).cpu(),
    }


# --------------------------------------------------------------------------- shards


@dataclass
class TurnGenConfig:
    samples: int = 20_000
    mix: tuple[float, float, float] = (0.5, 0.25, 0.25)  # self-play, perturbed, random
    seed: int = 0
    shard_size: int = 4096
    explore: float = 0.0  # uniform mixing of the self-play behaviour policy
    n_envs: int = 4096
    c_range: tuple[int, int] = (vr.C_MIN, vr.C_MAX)  # random states' c (log-uniform)
    chunk: int = 16384  # river-net rows per predict call
    batch: int = 2048  # states per target computation

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


RESUME_KEYS = ("mix", "seed", "shard_size", "explore", "n_envs", "c_range")


def generate_turn_data(
    river_predictor: Any,
    bp: Any,
    out: str | Path,
    cfg: TurnGenConfig,
    game_config: Any = None,
    device: torch.device | str = "cuda",
    resume: bool = False,
    river_net: str | None = None,
    blueprint_path: str | None = None,
    log: Callable[[str], Any] | None = print,
) -> dict[str, Any]:
    """Generate ``cfg.samples`` turn-end samples with bootstrapped targets into
    shards under ``out`` (see the module docstring). ``game_config`` defaults to
    the blueprint's game (``bp.game``), or 100bb at 50/100 without a blueprint.
    Without ``resume`` an existing ``meta.json`` is an error; with it, the
    settings must match and existing shards are skipped. Returns the meta dict."""
    log = log or (lambda *_a, **_k: None)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if game_config is None:
        game_config = bp.game if bp is not None else DEFAULT_GAME
    game = vr._game(game_config)
    engine_cfg = vr.engine_game_config(game)
    spec_d = None
    if bp is not None:
        from ..blueprint.deepcfr.config import spec_to_dict

        spec_d = spec_to_dict(bp.spec)
    meta_path = out / "meta.json"
    if meta_path.exists():
        if not resume:
            raise FileExistsError(f"{meta_path} exists; pass resume=True (--resume) to continue")
        meta = json.loads(meta_path.read_text())
        old, new = meta.get("config", {}), cfg.to_dict()
        bad = [k for k in RESUME_KEYS if old.get(k) != new.get(k)]
        if meta.get("kind") != KIND:
            bad.append("kind")
        if bad:
            raise ValueError(f"cannot resume: settings differ from meta.json: {bad}")
        meta["config"]["samples"] = cfg.samples
    else:
        meta = {
            "format": FORMAT_VERSION,
            "kind": KIND,
            "river_net": river_net,
            "blueprint": blueprint_path,
            "spec": spec_d,
            "game": game,
            "config": cfg.to_dict(),
            "keys": {k: str(v).replace("torch.", "") for k, v in SHARD_DTYPES.items()},
            "boards": "[n, 4] turn boards (river card not dealt)",
            "sources": {"0": "self-play", "1": "perturbed self-play", "2": "random"},
            "targets": "river-net chance average / opponent disjoint mass / (2c), (OOP, IP)",
            "shards": {},
        }
    write_atomic(meta_path, meta, as_json=True)
    dev = torch.device(device)
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
        )
        t1 = time.time()
        rows = label_states(river_predictor, states, dev, cfg.chunk, cfg.batch)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t2 = time.time()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(seeds[2]))
        rows = {key: v[perm].contiguous() for key, v in rows.items()}
        if not bool(torch.isfinite(rows["targets"].float()).all()):
            raise FloatingPointError(f"non-finite targets in shard {k}")
        write_atomic(path, rows)
        dt = time.time() - t0
        info = {
            "samples": n,
            "seeds": seeds,
            "seconds": dt,
            "states_s": t1 - t0,
            "targets_s": t2 - t1,
            "c_median": float(rows["c"].double().median()),
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
            f"({n / dt:.1f}/s; states {t1 - t0:.1f}s, targets {t2 - t1:.1f}s), "
            f"c median {info['c_median']:.0f}; run {done_now / max(total_s, 1e-9):.1f} samples/s"
        )
    return meta


__all__ = [
    "RiverAveragePredictor",
    "TurnGenConfig",
    "generate_turn_data",
    "label_states",
    "make_turn_states",
    "random_turn_states",
    "turn_checkdown_samples",
    "turn_strength",
    "turn_targets",
]
