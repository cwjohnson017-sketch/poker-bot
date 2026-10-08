"""Training data, training loop and held-out report of the river value net.

* **Data.** Shard files ``shard_*.pt`` (one or more directories), each a
  ``torch.save`` of :data:`SHARD_DTYPES`: ``boards`` [n, 5], ``c``, ``stack``,
  ``ranges`` and ``targets`` [n, 2, 1326] (OOP, IP; ranges sum to 1, targets
  ``ev_p(c)`` in pot units, see :mod:`.value_net`), ``exploit`` (pot units) and
  ``source`` (0 blueprint self-play, 1 perturbed, 2 random ranges).
  :func:`save_shard` writes one; :func:`load_shards` concatenates them.
  Turn-end shards (:mod:`.turn_data`) have the same keys with ``boards``
  [n, 4] (the river card is not dealt).
* **Features.** :class:`ValueData` keeps the samples (on the GPU when they fit)
  and the ``rank2`` table of every distinct board once
  (:class:`~.value_net.BoardFeatureCache`, int16); the buckets of a batch are
  one gather and an integer division. Another feature cache with the same
  interface can be passed in (``cache`` / ``cache_factory``; the turn-end net
  uses :class:`~.turn_net.TurnFeatureCache`, see :func:`.turn_net.train_turn_net`).
* **Loss.** Huber (or MSE) on ``ev`` of both players after the zero-sum layer,
  over valid combos whose opponent disjoint mass exceeds ``MASS_EPS``.
* **Report** (pot units) on held-out samples: MAE / RMSE per combo, overall, per
  pot-size bin of ``c`` and per source; the range-weighted MAE (weights
  ``w_p = r_p * m_{-p}``) and the error of each player's range-weighted game
  value; and two references, the zero prediction and the *bucket oracle*
  (each target replaced by the mean target of its strength bucket in its own
  sample: the floor of a bucket-level output at this ``K``).

:func:`checkdown_samples` makes synthetic samples with exact targets (both
players check the river down) for tests and smoke runs.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .showdown import ShowdownTables
from .value_net import (
    MASS_EPS,
    BoardFeatureCache,
    RiverValueNet,
    ValueNetConfig,
    normalise_ranges,
    opponent_mass,
    river_board_features,
    save_value_net,
)

C = 1326
SHARD_DTYPES: dict[str, torch.dtype] = {
    "boards": torch.uint8,
    "c": torch.int32,
    "stack": torch.int32,
    "ranges": torch.float16,
    "targets": torch.float16,
    "exploit": torch.float32,
    "source": torch.uint8,
}
SOURCE_NAMES = {0: "blueprint", 1: "perturbed", 2: "random", 3: "on-policy"}
POT_BINS = (100, 250, 500, 1000, 2000, 4000)  # lower edges of the c bins; last is [4000, 10000]


# --------------------------------------------------------------------------- shards


def save_shard(path: str | Path, data: dict[str, Any]) -> Path:
    """Write one shard (atomically: a temp file renamed into place). ``data``
    holds the :data:`SHARD_DTYPES` keys (cast to those dtypes, moved to CPU);
    ``boards`` is ``[n, 5]`` (river) or ``[n, 4]`` (turn-end)."""
    missing = [k for k in SHARD_DTYPES if k not in data]
    if missing:
        raise KeyError(f"shard is missing {missing}")
    # clone: torch.save writes a view's whole storage (a slice would save its parent)
    out = {
        k: torch.as_tensor(data[k])
        .detach()
        .to("cpu", dt)
        .clone(memory_format=torch.contiguous_format)
        for k, dt in SHARD_DTYPES.items()
    }
    n = out["boards"].shape[0]
    nb = out["boards"].shape[1] if out["boards"].dim() == 2 else 5
    shapes = {"boards": (n, nb if nb in (4, 5) else 5), "c": (n,), "stack": (n,)}
    shapes["ranges"] = (n, 2, C)
    shapes.update(targets=(n, 2, C), exploit=(n,), source=(n,))
    for k, shp in shapes.items():
        if tuple(out[k].shape) != shp:
            raise ValueError(f"shard {k} has shape {tuple(out[k].shape)}, expected {shp}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(out, tmp)
    os.replace(tmp, path)
    return path


def shard_files(paths: str | Path | Iterable[str | Path]) -> list[Path]:
    """``shard_*.pt`` files of the given directories (sorted), or the files themselves."""
    if isinstance(paths, str | Path):
        paths = [paths]
    files: list[Path] = []
    for p in map(Path, paths):
        files += sorted(p.glob("shard_*.pt")) if p.is_dir() else [p]
    return files


def load_shards(paths: str | Path | Iterable[str | Path]) -> dict[str, torch.Tensor]:
    """All shards of one or more directories concatenated into one dict (CPU)."""
    files = shard_files(paths)
    if not files:
        raise FileNotFoundError(f"no shard_*.pt files in {paths}")
    parts = [torch.load(f, map_location="cpu", weights_only=True) for f in files]
    return {k: torch.cat([p[k].to(dt) for p in parts]) for k, dt in SHARD_DTYPES.items()}


# --------------------------------------------------------------------------- synthetic data


@torch.no_grad()
def checkdown_samples(
    n: int, seed: int = 0, device: torch.device | str = "cpu", total: int = 10000
) -> dict[str, torch.Tensor]:
    """``n`` synthetic samples in the shard format whose river is checked down:
    ``ev_p(c) = 0.5 * showdown(r_{-p})(c) / m_{-p}(c)``. Random boards,
    log-uniform ``c`` in ``[100, total / 2]``; per sample a range family
    (``source``): 0 uniform, 1 strength-correlated (``exp(beta * pct)`` tilts
    and percentile bands, with noise), 2 random (sparse powers of uniforms)."""
    dev = torch.device(device)
    g = torch.Generator(device=dev).manual_seed(seed)

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, generator=g, device=dev)

    boards = rand(n, 52).argsort(1)[:, :5]
    f = river_board_features(boards, 2)
    valid, pct = f["valid"], f["pct"]
    fam = torch.randint(0, 3, (n,), generator=g, device=dev)
    v = valid[:, None, :].float().expand(n, 2, C)
    # strength-correlated: tilt exp(beta * pct) or a band [lo, lo + width], with noise
    beta = (rand(n, 2, 1) * 2 - 1) * 12
    tilt = torch.exp(beta * (pct[:, None, :] - 0.5))
    lo = rand(n, 2, 1) * 0.8
    band = ((pct[:, None, :] >= lo) & (pct[:, None, :] <= lo + 0.05 + rand(n, 2, 1) * 0.5)).float()
    strength = torch.where(rand(n, 2, 1) < 0.5, tilt, band + 0.01)
    strength = strength * torch.exp(0.5 * torch.randn(n, 2, C, generator=g, device=dev))
    # random: sparse powers of uniforms
    power = 1 + 5 * rand(n, 2, 1)
    keep = rand(n, 2, C) < (0.05 + 0.95 * rand(n, 2, 1))
    random = rand(n, 2, C).pow(power) * keep
    r = torch.where(
        (fam == 0)[:, None, None], v, torch.where((fam == 1)[:, None, None], strength, random)
    )
    ranges = normalise_ranges(r * v, valid)
    m = opponent_mass(ranges, valid)
    targets = torch.zeros(n, 2, C, device=dev)
    for s in range(0, n, 1024):
        sl = slice(s, s + 1024)
        tables = ShowdownTables(boards[sl].tolist(), dev)
        k = tables.num_boards
        bid = torch.arange(k, device=dev).repeat_interleave(2)
        opp = ranges[sl].flip(1).reshape(2 * k, C)
        targets[sl] = 0.5 * tables.showdown(opp, bid).view(k, 2, C)
    ok = valid[:, None, :] & (m > 1e-9)
    targets = torch.where(ok, targets / m.clamp(min=1e-12), torch.zeros_like(targets))
    c = torch.exp(math.log(100) + rand(n) * math.log(total / 200)).round().long()
    return {
        "boards": boards,
        "c": c,
        "stack": total - c,
        "ranges": ranges,
        "targets": targets,
        "exploit": torch.zeros(n),
        "source": fam,
    }


# --------------------------------------------------------------------------- dataset


@dataclass
class ValueTrainConfig:
    steps: int = 20_000
    batch: int = 2048  # samples per Adam step
    lr: float = 1e-3
    lr_final: float = 1e-5  # cosine decay to this
    warmup: int = 200  # linear warm-up steps
    weight_decay: float = 0.0  # AdamW decoupled weight decay
    loss: str = "huber"  # huber | mse
    huber_delta: float = 1.0  # pot units
    holdout: float = 0.03  # held-out share when no separate held-out data is given
    holdout_by: str = "board"  # board (no board in both sets) | sample
    max_exploit: float = 0.0  # drop samples whose solve exploitability exceeds this (0 = keep)
    grad_clip: float = 1.0
    eval_every: int = 0  # 0 = steps // 10
    eval_samples: int = 8192  # held-out samples for the periodic log line
    eval_chunk: int = 1024
    data_device: str = "auto"  # auto | cpu | cuda: where the samples live
    seed: int = 0
    # relative sampling weights per source, e.g. "3:4" draws on-policy samples (source 3)
    # four times as often as the others (weight 1); "" samples uniformly
    source_weights: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> ValueTrainConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


class ValueData:
    """Samples on ``device`` plus one feature row per distinct board: ``rank2``
    of river boards by default, or the given ``cache`` (an empty feature cache
    with ``max_boards=None``, e.g. a turn-end one). The cache may live on another
    device than the samples (``cache_device`` for the default one): with the
    samples in host memory, board features are still built and gathered on the
    GPU."""

    def __init__(
        self,
        data: dict[str, torch.Tensor],
        device: torch.device | str = "cpu",
        max_exploit: float = 0.0,
        cache: Any = None,
        cache_device: torch.device | str | None = None,
    ):
        dev = self.device = torch.device(device)
        keep = torch.ones(data["c"].shape[0], dtype=torch.bool)
        if max_exploit > 0:
            keep &= data["exploit"].float() <= max_exploit
        self.dropped = int((~keep).sum())
        sel = keep.nonzero().squeeze(1)
        if cache is None:
            if data["boards"].shape[1] != 5:
                raise ValueError(
                    f"{data['boards'].shape[1]}-card boards need their own feature cache "
                    "(turn-end shards: pokerbot.search.turn_net.train_turn_net)"
                )
            cdev = torch.device(cache_device) if cache_device is not None else dev
            cache = BoardFeatureCache(cdev, max_boards=None, tables=False)
        self.cache = cache
        self.board_id = self.cache.ids(data["boards"][sel])
        self.ranges = data["ranges"][sel].to(dev, torch.float16)
        self.targets = data["targets"][sel].to(dev, torch.float16)
        self.c = data["c"][sel].to(dev).long()
        self.stack = data["stack"][sel].to(dev).long()
        self.exploit = data["exploit"][sel].float()
        self.source = data["source"][sel].long()

    def __len__(self) -> int:
        return int(self.c.shape[0])

    def batch(self, idx: torch.Tensor, buckets: int, device: torch.device) -> dict:
        """Model inputs, targets, opponent masses and the loss mask of rows ``idx``."""
        idx = idx.to(self.device)
        f = self.cache.features(self.board_id[idx.to(self.board_id.device)], buckets)
        b = {
            "ranges": self.ranges[idx],
            "targets": self.targets[idx],
            "c": self.c[idx],
            "stack": self.stack[idx],
            **f,
        }
        b = {k: v.to(device, non_blocking=True) for k, v in b.items()}
        b["ranges"] = normalise_ranges(b["ranges"], b["valid"])
        b["targets"] = b["targets"].float()
        b["m"] = opponent_mass(b["ranges"], b["valid"])
        b["mask"] = b["valid"][:, None, :] & (b["m"] > MASS_EPS)
        return b


def split_holdout(
    ds: ValueData, frac: float, by: str, seed: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(train, held-out)`` sample indices (CPU); ``by="board"`` keeps every
    board on one side, so perturbed copies of a state never straddle the split."""
    g = torch.Generator().manual_seed(seed)
    n = len(ds)
    if by == "board":
        bid = ds.board_id.cpu()
        u = torch.rand(int(bid.max()) + 1 if n else 0, generator=g)
        held = u[bid] < frac
    elif by == "sample":
        held = torch.rand(n, generator=g) < frac
    else:
        raise ValueError(f"holdout_by must be board or sample, got {by!r}")
    if n > 1 and not bool(held.any()):
        held[torch.randint(0, n, (1,), generator=g)] = True
    return (~held).nonzero().squeeze(1), held.nonzero().squeeze(1)


def parse_source_weights(spec: str) -> dict[int, float]:
    """``"3:4"`` or ``"2:0.5,3:4"`` -> ``{3: 4.0}`` / ``{2: 0.5, 3: 4.0}``; ``""`` -> ``{}``."""
    out: dict[int, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        src, sep, w = part.partition(":")
        if not sep or not src.strip().isdigit():
            raise ValueError(f"bad source weight {part!r}: expected <source>:<weight>")
        if float(w) < 0:
            raise ValueError(f"source weight {part!r} is negative")
        out[int(src)] = float(w)
    return out


def sampling_cdf(source: torch.Tensor, weights: dict[int, float]) -> torch.Tensor | None:
    """Cumulative sampling probabilities (float64) of samples with sources ``source``
    under per-source ``weights`` (1 for unlisted sources); ``None`` without weights."""
    if not weights:
        return None
    w = torch.ones(source.shape[0], dtype=torch.float64, device=source.device)
    for s, x in weights.items():
        w[source == s] = x
    cdf = w.cumsum(0)
    if not float(cdf[-1]) > 0:
        raise ValueError("source weights leave no sample to draw")
    return cdf / cdf[-1]


def value_loss(pred: torch.Tensor, b: dict, kind: str = "huber", delta: float = 1.0):
    mask = b["mask"].float()
    if kind == "huber":
        loss = F.huber_loss(pred, b["targets"], reduction="none", delta=delta)
    elif kind == "mse":
        loss = (pred - b["targets"]).square()
    else:
        raise ValueError(f"unknown loss {kind!r}")
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _forward(net: RiverValueNet, b: dict, amp: bool) -> torch.Tensor:
    return net(
        b["ranges"], b["bucket"], b["onehot"], b["c"], b["stack"], pct=b["pct"], m=b["m"], amp=amp
    )


# --------------------------------------------------------------------------- held-out report


@torch.no_grad()
def sample_stats(
    net: RiverValueNet,
    ds: ValueData,
    idx: torch.Tensor,
    device: torch.device,
    amp: bool,
    chunk: int = 1024,
) -> dict[str, torch.Tensor]:
    """Per-sample error sums (CPU tensors) of the net, the zero prediction and
    the bucket oracle on samples ``idx``."""
    net.eval()
    K = net.cfg.buckets
    out: dict[str, list[torch.Tensor]] = {}
    for lo in range(0, len(idx), chunk):
        b = ds.batch(idx[lo : lo + chunk], K, device)
        pred = _forward(net, b, amp)
        M = b["mask"].float()
        tgt = b["targets"] * M
        err = (pred - tgt) * M
        B = M.shape[0]
        bi = b["bucket"][:, None, :].expand(B, 2, C)
        tot = M.new_zeros(B, 2, K + 1).scatter_add_(2, bi, tgt)
        num = M.new_zeros(B, 2, K + 1).scatter_add_(2, bi, M)
        oracle = (tot / num.clamp(min=1.0)).gather(2, bi) * M
        oerr = oracle - tgt
        w = b["ranges"] * b["m"] * M
        z2 = w.sum((1, 2))  # 2 Z (pair mass counted for both players)
        zok = z2 > 1e-8
        zs = z2.clamp(min=1e-8)
        zp = (z2 / 2).clamp(min=1e-8)[:, None]

        def gv(x: torch.Tensor, w=w, zp=zp) -> torch.Tensor:  # [B, 2] range-weighted values
            return (w * x).sum(-1) / zp

        gv_t = gv(tgt)
        row = {
            "cnt": M.sum((1, 2)),
            "abs": err.abs().sum((1, 2)),
            "sq": err.square().sum((1, 2)),
            "zabs": tgt.abs().sum((1, 2)),
            "zsq": tgt.square().sum((1, 2)),
            "oabs": oerr.abs().sum((1, 2)),
            "osq": oerr.square().sum((1, 2)),
            "zok": zok.float(),
            "wabs": (w * err.abs()).sum((1, 2)) / zs,
            "wzabs": (w * tgt.abs()).sum((1, 2)) / zs,
            "woabs": (w * oerr.abs()).sum((1, 2)) / zs,
            "gv": (gv(pred) - gv_t).abs(),
            "gvz": gv_t.abs(),
            "gvo": (gv(oracle) - gv_t).abs(),
            "gv_target_sum": gv_t.sum(1),
        }
        for k, v in row.items():
            out.setdefault(k, []).append(v.float().cpu())
    net.train()
    return {k: torch.cat(v) for k, v in out.items()}


def summarize(st: dict[str, torch.Tensor], sel: torch.Tensor | None = None) -> dict[str, Any]:
    """Pot-unit metrics of the samples ``sel`` (bool mask; all when ``None``)."""
    if sel is None:
        sel = torch.ones_like(st["cnt"], dtype=torch.bool)
    n = int(sel.sum())
    cnt = float(st["cnt"][sel].sum())
    if n == 0 or cnt == 0:
        return {"samples": n}
    wsel = sel & (st["zok"] > 0)

    def part(a: str, s: str, wa: str, g: str) -> dict[str, Any]:
        return {
            "mae": float(st[a][sel].sum()) / cnt,
            "rmse": math.sqrt(float(st[s][sel].sum()) / cnt),
            "wmae": float(st[wa][wsel].mean()) if bool(wsel.any()) else float("nan"),
            "gv_mae": [float(x) for x in st[g][wsel].mean(0)] if bool(wsel.any()) else [],
        }

    return {
        "samples": n,
        "combos": int(cnt),
        **part("abs", "sq", "wabs", "gv"),
        "zero": part("zabs", "zsq", "wzabs", "gvz"),
        "oracle": part("oabs", "osq", "woabs", "gvo"),
    }


def heldout_report(st: dict[str, torch.Tensor], c: torch.Tensor, source: torch.Tensor) -> dict:
    """Overall, per pot-size bin of ``c`` and per source."""
    c = c.cpu()
    source = source.cpu()
    edges = torch.tensor(POT_BINS[1:])
    bins = torch.bucketize(c, edges, right=True)
    names = [f"[{lo},{hi})" for lo, hi in zip(POT_BINS[:-1], POT_BINS[1:], strict=True)]
    names.append(f"[{POT_BINS[-1]},10000]")
    rep: dict[str, Any] = {"overall": summarize(st)}
    rep["by_pot"] = {nm: summarize(st, bins == i) for i, nm in enumerate(names)}
    rep["by_source"] = {
        SOURCE_NAMES.get(int(s), str(int(s))): summarize(st, source == s)
        for s in torch.unique(source).tolist()
    }
    rep["gv_target_sum_mean"] = float(st["gv_target_sum"].mean())
    return rep


def format_report(rep: dict[str, Any]) -> str:
    """A text table of :func:`heldout_report` (pot units)."""
    head = (
        f"{'':>16} {'samples':>8} {'MAE':>8} {'RMSE':>8} {'wMAE':>8} {'GV0':>8} {'GV1':>8}"
        f" | {'zeroMAE':>8} {'zeroWMAE':>8} | {'orclMAE':>8} {'orclRMSE':>8} {'orclWMAE':>8}"
    )
    lines = [head]

    def line(name: str, s: dict[str, Any]) -> str:
        if "mae" not in s:
            return f"{name:>16} {s['samples']:>8}"
        gv = s["gv_mae"] + [float("nan")] * (2 - len(s["gv_mae"]))
        return (
            f"{name:>16} {s['samples']:>8} {s['mae']:8.4f} {s['rmse']:8.4f} {s['wmae']:8.4f}"
            f" {gv[0]:8.4f} {gv[1]:8.4f} | {s['zero']['mae']:8.4f} {s['zero']['wmae']:8.4f}"
            f" | {s['oracle']['mae']:8.4f} {s['oracle']['rmse']:8.4f} {s['oracle']['wmae']:8.4f}"
        )

    lines.append(line("overall", rep["overall"]))
    for nm, s in rep["by_pot"].items():
        lines.append(line(f"c {nm}", s))
    for nm, s in rep["by_source"].items():
        lines.append(line(nm, s))
    return "\n".join(lines)


# --------------------------------------------------------------------------- training


def _data_device(spec: str, device: torch.device, n: int) -> torch.device:
    if spec != "auto":
        return torch.device(spec)
    if device.type != "cuda":
        return device
    need = n * (2 * (2 * C * 2) + 2 * C + 64)  # fp16 ranges + targets, int16 rank table
    free, _ = torch.cuda.mem_get_info(device)
    return device if need + (3 << 30) < free else torch.device("cpu")


def train_value_net(
    data: str | Path | Sequence[str | Path],
    out: str | Path | None = None,
    net_cfg: ValueNetConfig | None = None,
    cfg: ValueTrainConfig | None = None,
    heldout: str | Path | Sequence[str | Path] | None = None,
    device: str | torch.device = "cuda",
    log: Any = print,
    raw: dict[str, torch.Tensor] | None = None,
    raw_heldout: dict[str, torch.Tensor] | None = None,
    cache_factory: Any = None,
    meta: dict[str, Any] | None = None,
) -> tuple[RiverValueNet, dict[str, Any]]:
    """Train on the shards of ``data`` (or the loaded dict ``raw``); evaluate on
    ``heldout`` when given, else on a seeded held-out split. Writes ``out`` (the
    checkpoint) and ``out.with_suffix('.json')`` (the report) when ``out`` is set.
    ``cache_factory(device)`` makes the board-feature cache of each dataset
    (default: river :class:`~.value_net.BoardFeatureCache`); ``meta`` is merged
    into the checkpoint's meta (e.g. ``kind: turn_end``) and set as ``net.meta``."""
    net_cfg = net_cfg or ValueNetConfig()
    cfg = cfg or ValueTrainConfig()
    log = log or (lambda *_a, **_k: None)
    dev = torch.device(device)
    t0 = time.time()
    raw = raw if raw is not None else load_shards(data)
    ddev = _data_device(cfg.data_device, dev, int(raw["c"].shape[0]))

    def make_cache() -> Any:  # board features on the compute device
        return cache_factory(dev) if cache_factory is not None else None

    ds = ValueData(raw, ddev, cfg.max_exploit, make_cache(), cache_device=dev)
    del raw
    if heldout is not None or raw_heldout is not None:
        raw_h = raw_heldout if raw_heldout is not None else load_shards(heldout)
        val_ds = ValueData(raw_h, ddev, cfg.max_exploit, make_cache(), cache_device=dev)
        del raw_h
        tr, va = torch.arange(len(ds)), torch.arange(len(val_ds))
    else:
        val_ds = ds
        tr, va = split_holdout(ds, cfg.holdout, cfg.holdout_by, cfg.seed)
    if dev.type == "cuda":
        # building the board features of ~1M boards leaves GBs of cached temporaries;
        # without releasing them the residual head's activations spill out of VRAM
        torch.cuda.empty_cache()
    log(
        f"# value net: {len(tr):,} train / {len(va):,} held-out samples on {ddev} "
        f"({len(ds.cache):,} boards, {ds.dropped} dropped by max_exploit), "
        f"{time.time() - t0:.0f}s to load; {net_cfg}"
    )
    torch.manual_seed(cfg.seed)
    net = RiverValueNet(net_cfg).to(dev).train()
    log(f"# parameters: {sum(p.numel() for p in net.parameters()):,}")
    opt = torch.optim.AdamW(
        net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, fused=dev.type == "cuda"
    )
    amp = dev.type == "cuda"
    g = torch.Generator(device=ddev).manual_seed(cfg.seed + 1)
    tr_dev = tr.to(ddev)
    tr_src = ds.source[tr].to(ddev)
    cdf = sampling_cdf(tr_src, parse_source_weights(cfg.source_weights))
    if cdf is not None:
        p = cdf.diff(prepend=cdf.new_zeros(1))
        share = {
            SOURCE_NAMES.get(s, str(s)): float(p[tr_src == s].sum())
            for s in torch.unique(tr_src).tolist()
        }
        log(f"# sampling shares by source: {', '.join(f'{k} {v:.3f}' for k, v in share.items())}")
    every = cfg.eval_every or max(1, cfg.steps // 10)
    va_log = va[: cfg.eval_samples]
    history: list[dict[str, float]] = []
    ema = None
    t1 = time.time()
    for step in range(cfg.steps):
        frac = step / max(1, cfg.steps - 1)
        lr = cfg.lr_final + 0.5 * (cfg.lr - cfg.lr_final) * (1 + math.cos(math.pi * frac))
        if cfg.warmup > 0 and step < cfg.warmup:
            lr *= (step + 1) / cfg.warmup
        for pg in opt.param_groups:
            pg["lr"] = lr
        if cdf is None:
            pick = torch.randint(0, len(tr_dev), (cfg.batch,), generator=g, device=ddev)
        else:
            u = torch.rand(cfg.batch, generator=g, device=ddev, dtype=torch.float64)
            pick = torch.searchsorted(cdf, u).clamp_(max=len(tr_dev) - 1)
        idx = tr_dev[pick]
        b = ds.batch(idx, net_cfg.buckets, dev)
        loss = value_loss(_forward(net, b, amp), b, cfg.loss, cfg.huber_delta)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        opt.step()
        lv = float(loss.detach())
        ema = lv if ema is None else 0.98 * ema + 0.02 * lv
        if (step + 1) % every == 0 or step + 1 == cfg.steps:
            st = sample_stats(net, val_ds, va_log, dev, amp, cfg.eval_chunk)
            s = summarize(st)
            rec = {"step": step + 1, "loss": ema, "mae": s.get("mae"), "wmae": s.get("wmae")}
            history.append(rec)
            log(
                f"# step {step + 1}: loss {ema:.5f}, held-out MAE {s.get('mae', 0):.4f} "
                f"(zero {s.get('zero', {}).get('mae', 0):.4f}, oracle "
                f"{s.get('oracle', {}).get('mae', 0):.4f}), wMAE {s.get('wmae', 0):.4f}, "
                f"lr {lr:.2e}, {time.time() - t1:.0f}s"
            )
    train_time = time.time() - t1
    st = sample_stats(net, val_ds, va, dev, amp, cfg.eval_chunk)
    rep = heldout_report(st, val_ds.c[va.to(val_ds.device)], val_ds.source[va])
    rep.update(
        train_samples=len(tr),
        heldout_samples=len(va),
        train_seconds=train_time,
        history=history,
        net=net_cfg.to_dict(),
        train=cfg.to_dict(),
        data=[str(p) for p in shard_files(data)] if data is not None else [],
        heldout_data=[str(p) for p in shard_files(heldout)] if heldout is not None else [],
    )
    if meta:
        rep["meta"] = dict(meta)
    log(format_report(rep))
    net.meta = dict(meta or {})
    if out is not None:
        out = Path(out)
        ck_meta = {k: rep[k] for k in ("train", "data", "heldout_data", "train_samples")}
        ck_meta["heldout"] = {"overall": rep["overall"]}
        ck_meta.update(meta or {})
        save_value_net(out, net, ck_meta)
        out.with_suffix(".json").write_text(json.dumps(rep, indent=1))
        log(f"# wrote {out} and {out.with_suffix('.json')}")
    return net.eval(), rep
