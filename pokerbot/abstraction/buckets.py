"""Card-bucket tables for the tabular blueprint (DESIGN.md section 5.3), on torch.

For each postflop street every canonical (suit-isomorphic) hand class gets an
equity feature, the classes are clustered with k-means, and the result is a
table ``table[canonical_index] = bucket`` that
``poker_engine.CardAbstraction(tables=[None, flop, turn, river])`` (or the
MCCFR config's ``cards.tables``) loads directly.

Features (per canonical index, computed on its representative hand):

* **river**: exact equity against all 990 opponent hands
  (``pokerbot.env.equity.equity_river``), one number;
* **turn**: histogram (``bins`` equal-width bins over [0, 1]) of the exact
  river equity over all 46 river cards (``runouts: 0``), or over that many
  sampled rivers;
* **flop**: the same histogram over ``runouts`` sampled (turn, river) runouts
  with exact river equity each (``pokerbot.env.equity.equity_histogram``);
  ``runouts: 0`` enumerates all 1,081 runouts instead.

Clustering (weighted by :func:`~.isomorphism.orbit_sizes`, i.e. by how many
raw hands each class stands for):

* **river**: 1-D k-means on equity (Lloyd's algorithm from weighted-quantile
  centres). Buckets are sorted by equity and assigned by nearest centre, so
  bucket ids are monotone in equity.
* **flop / turn**: k-means under the earth mover's distance between
  histograms. For 1-D histograms on equally spaced bins, EMD is exactly the
  L1 distance between cumulative sums, so points live in CDF space
  (``bins - 1`` coordinates) and distances are ``torch.cdist(p=1)``. k-means++
  seeding (D^2 sampling under EMD), then ``iters`` Lloyd iterations; the
  centre update is the weighted mean histogram (the usual choice for
  EMD k-means; the exact L1 minimiser would be the coordinate-wise median).
  Buckets are sorted by the centre's mean equity.

Everything is chunked: features are computed ``chunk`` indices at a time
into host arrays (optionally memory-mapped under ``<out_dir>/features``),
centres are fitted on all indices or a random ``sample`` of them, and the
assignment pass streams over all indices. ``limit`` computes only a seeded
random subset of indices (smoke runs, tests); the written table is still
full length, as the engine requires, with every other entry padded with
``index % buckets`` (a plumbing filler, not an abstraction; the sidecar
records ``padded``).

Outputs per street in ``out_dir``: ``<street>.npy`` (1-D ``uint16``, or
``uint32`` above 65,535 buckets, length ``canonical_size(street)``) and
``<street>.json`` (config, feature and bucket statistics, timings, git hash).
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..config import REPO_ROOT, git_hash
from ..env.equity import equity_histogram, equity_river
from .isomorphism import (
    BOARD_LEN,
    build_canonical_cache,
    canonical_size,
    num_cards,
    orbit_sizes,
    representatives,
)

STREET_NAMES = ("preflop", "flop", "turn", "river")
STREET_IDS = {name: i for i, name in enumerate(STREET_NAMES)}


# ----------------------------------------------------------------------------- config


@dataclass
class StreetConfig:
    """Per-street settings (see ``configs/buckets_*.yaml``)."""

    buckets: int = 1000
    bins: int = 10  # histogram bins (flop / turn)
    runouts: int = 0  # 0 = enumerate every runout exactly; > 0 = sample this many
    sample: int | None = None  # fit centres on this many random indices (None = all)
    iters: int = 25  # Lloyd iterations
    chunk: int = 2048  # canonical indices per feature batch
    fit_chunk: int = 65536  # rows per distance batch in k-means


@dataclass
class BuildConfig:
    streets: dict[str, StreetConfig] = field(default_factory=dict)
    out_dir: str = "data/abstraction/buckets"
    seed: int = 0
    device: str = "cpu"
    max_rows: int = 1 << 21  # evaluated 7-card hands per equity kernel call
    weighted: bool = True  # weight classes by the raw hands they stand for
    canonical_cache: str | None = None  # dir of canonical_<street>.npy (None = unindex on the fly)
    feature_cache: bool = True  # keep features under <out_dir>/features and reuse them
    limit: int | None = None  # only this many (random) indices per street: smoke runs

    @staticmethod
    def from_dict(d: dict[str, Any]) -> BuildConfig:
        d = dict(d.get("buckets", d))
        streets_raw = d.pop("streets", {}) or {}
        known = {f.name for f in fields(StreetConfig)}
        streets = {}
        for name, sd in streets_raw.items():
            if name not in ("flop", "turn", "river"):
                raise ValueError(f"unknown street {name!r} (flop, turn or river)")
            bad = set(sd or {}) - known
            if bad:
                raise ValueError(f"unknown {name} options {sorted(bad)}")
            streets[name] = StreetConfig(**(sd or {}))
        bknown = {f.name for f in fields(BuildConfig)} - {"streets"}
        bad = set(d) - bknown
        if bad:
            raise ValueError(f"unknown buckets options {sorted(bad)}")
        return BuildConfig(streets=streets, **d)

    def resolved_out_dir(self) -> Path:
        p = Path(self.out_dir)
        return p if p.is_absolute() else REPO_ROOT / p

    def resolved_cache_dir(self) -> Path | None:
        if not self.canonical_cache:
            return None
        p = Path(self.canonical_cache)
        return p if p.is_absolute() else REPO_ROOT / p


# --------------------------------------------------------------------------- features


def _completions(hole: torch.Tensor, board: torch.Tensor) -> torch.Tensor:
    """Every completion of ``board`` to five cards, ``[N, C, 5]`` long."""
    n, k = board.shape
    known = torch.cat([hole, board], 1)
    used = torch.zeros(n, 52, dtype=torch.long, device=hole.device).scatter_(1, known, 1)
    rest = torch.argsort(used, dim=1, stable=True)[:, : 52 - 2 - k]  # unused cards, ascending
    combos = torch.combinations(torch.arange(52 - 2 - k, device=hole.device), 5 - k)  # [C, 5-k]
    run = rest[:, combos]  # [N, C, 5-k]
    return torch.cat([board[:, None, :].expand(n, combos.shape[0], k), run], 2)


@torch.no_grad()
def street_features(
    street: int,
    cards: torch.Tensor,
    bins: int = 10,
    runouts: int = 0,
    generator: torch.Generator | None = None,
    max_rows: int = 1 << 21,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``(features [N, D] float32, equity [N] float32)`` of ``[N, 2 + board]``
    hands on ``street``: river ``D = 1`` (the exact equity), flop/turn the
    ``bins``-bin river-equity histogram over runouts (``D = bins``); ``equity``
    is the mean equity."""
    cards = cards.long()
    hole, board = cards[:, :2], cards[:, 2:]
    if street == 3:
        eq = equity_river(hole, board, max_rows=max_rows)
        return eq[:, None], eq
    if street not in (1, 2):
        raise ValueError("features exist for streets 1..3")
    n = cards.shape[0]
    if runouts > 0:
        hist, eq = equity_histogram(
            hole, board, runouts, bins, 0, generator=generator, return_equity=True
        )
        return hist.float(), eq.float()
    full = _completions(hole, board)  # [N, C, 5]
    c = full.shape[1]
    req = equity_river(hole.repeat_interleave(c, 0), full.reshape(-1, 5), max_rows=max_rows)
    idx = (req * bins).long().clamp(0, bins - 1)
    hist = torch.zeros(n * c, bins, device=cards.device).scatter_(1, idx[:, None], 1.0)
    return hist.view(n, c, bins).mean(1), req.view(n, c).mean(1)


# ---------------------------------------------------------------------------- k-means


def to_cdf(hist: torch.Tensor) -> torch.Tensor:
    """Histogram rows -> cumulative sums without the final 1 (EMD space)."""
    return hist.cumsum(1)[:, :-1]


def _dist(x: torch.Tensor, c: torch.Tensor, metric: str) -> torch.Tensor:
    if metric == "emd":  # x, c already in CDF space
        return torch.cdist(x, c, p=1)
    d = x[:, None, :] - c[None, :, :]
    return (d * d).sum(-1)


def assign(
    x: torch.Tensor, centres: torch.Tensor, metric: str, chunk: int = 65536
) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest centre (lowest index on ties) and its distance, chunked."""
    labels = torch.empty(x.shape[0], dtype=torch.long, device=x.device)
    dmin = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    for s in range(0, x.shape[0], chunk):
        d = _dist(x[s : s + chunk], centres, metric)
        dmin[s : s + chunk], labels[s : s + chunk] = d.min(1)
    return labels, dmin


def _weighted_choice(p: torch.Tensor, generator: torch.Generator | None) -> torch.Tensor:
    cum = p.double().cumsum(0)
    r = torch.rand(1, generator=generator, device=p.device, dtype=torch.float64) * cum[-1]
    return torch.searchsorted(cum, r).clamp_max(p.shape[0] - 1)


def kmeanspp(
    x: torch.Tensor,
    w: torch.Tensor,
    k: int,
    metric: str,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """k-means++ seeding (D^2 sampling, weighted); ``[k', D]`` with ``k' <= k``
    (fewer when the data has fewer distinct points)."""
    first = _weighted_choice(w, generator)
    centres = [x[first]]
    dmin = _dist(x, centres[0], metric)[:, 0]
    for _ in range(1, k):
        p = w * dmin * dmin
        if not bool(p.sum() > 0):  # every point coincides with a centre
            break
        i = _weighted_choice(p, generator)
        centres.append(x[i])
        dmin = torch.minimum(dmin, _dist(x, centres[-1], metric)[:, 0])
    return torch.cat(centres, 0)


def quantile_init(x: torch.Tensor, w: torch.Tensor, k: int) -> torch.Tensor:
    """1-D: centres at the weighted ``(j + 0.5) / k`` quantiles, de-duplicated."""
    v, order = x[:, 0].sort()
    cw = w[order].double().cumsum(0)
    q = (torch.arange(k, device=x.device, dtype=torch.float64) + 0.5) / k * cw[-1]
    pos = torch.searchsorted(cw, q).clamp_max(v.shape[0] - 1)
    return v[pos].unique()[:, None]


@torch.no_grad()
def kmeans(
    x: torch.Tensor,
    k: int,
    weights: torch.Tensor | None = None,
    metric: str = "emd",
    iters: int = 25,
    generator: torch.Generator | None = None,
    chunk: int = 65536,
    init: str | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Weighted Lloyd k-means. ``metric="emd"`` expects histogram rows and
    works in CDF space (returns centres as CDFs); ``"l2"`` is squared
    Euclidean. ``init`` is ``"kmeans++"`` (default for EMD) or ``"quantile"``
    (default for 1-D ``l2``). Returns ``(centres [k', D], objective per
    iteration)`` with ``k' <= k``; empty clusters are re-seeded at the points
    farthest from their centres."""
    if metric == "emd":
        x = to_cdf(x.float())
    x = x.float()
    n = x.shape[0]
    w = torch.ones(n, device=x.device) if weights is None else weights.float().to(x.device)
    init = init or ("quantile" if metric == "l2" and x.shape[1] == 1 else "kmeans++")
    if init == "quantile":
        c = quantile_init(x, w, k)
    else:
        c = kmeanspp(x, w, min(k, n), metric, generator)
    kk = c.shape[0]
    history: list[float] = []
    prev = None
    for _ in range(iters):
        labels, dmin = assign(x, c, metric, chunk)
        history.append(float((w * dmin).sum() / w.sum()))
        if prev is not None and torch.equal(labels, prev):
            break
        prev = labels
        sums = torch.zeros(kk, x.shape[1], device=x.device).index_add_(0, labels, x * w[:, None])
        cnt = torch.zeros(kk, device=x.device).index_add_(0, labels, w)
        empty = cnt <= 0
        c = torch.where(empty[:, None], c, sums / cnt.clamp_min(1e-12)[:, None])
        ne = int(empty.sum())
        if ne:
            far = torch.topk(dmin * w, min(ne, n)).indices
            c[empty.nonzero()[: far.shape[0], 0]] = x[far]
    return c, history


def centre_equity(centres: torch.Tensor, metric: str) -> torch.Tensor:
    """Mean equity of each centre (bin centres for histograms in CDF space)."""
    if metric == "l2":
        return centres[:, 0]
    k, b1 = centres.shape
    ones = torch.ones(k, 1, device=centres.device)
    zeros = torch.zeros(k, 1, device=centres.device)
    hist = torch.cat([centres, ones], 1) - torch.cat([zeros, centres], 1)
    mid = (torch.arange(b1 + 1, device=centres.device, dtype=torch.float32) + 0.5) / (b1 + 1)
    return (hist * mid).sum(1)


# ------------------------------------------------------------------------- the build


def _sample_indices(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """``k`` distinct sorted indices from ``range(n)``."""
    if k >= n:
        return np.arange(n, dtype=np.int64)
    if k > n // 8:
        return np.sort(rng.permutation(n)[:k]).astype(np.int64)
    out = np.unique(rng.integers(0, n, size=k + k // 8 + 16))
    while out.shape[0] < k:
        out = np.unique(np.concatenate([out, rng.integers(0, n, size=k)]))
    return np.sort(rng.permutation(out)[:k]).astype(np.int64)


def _fmt_time(s: float) -> str:
    if s < 120:
        return f"{s:.1f}s"
    if s < 7200:
        return f"{s / 60:.1f}min"
    return f"{s / 3600:.2f}h"


@dataclass
class StreetResult:
    street: int
    table_path: Path
    info: dict[str, Any]
    indices: np.ndarray | None  # computed indices (None = all)
    labels: np.ndarray  # bucket of each computed index
    features: np.ndarray  # [n, D] float32 (may be a memmap)
    equity: np.ndarray  # [n] float32
    weights: np.ndarray  # [n] orbit sizes


def _feature_meta(street: int, sc: StreetConfig, bc: BuildConfig, n: int) -> dict[str, Any]:
    return {
        "street": street,
        "bins": sc.bins if street < 3 else 1,
        "runouts": sc.runouts if street < 3 else 0,
        "seed": bc.seed,
        "limit": bc.limit,
        "chunk": sc.chunk if (street < 3 and sc.runouts > 0) else None,
        "n": n,
    }


def compute_features(
    street: int,
    sc: StreetConfig,
    bc: BuildConfig,
    indices: np.ndarray | None,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Features, mean equity and orbit weights of ``indices`` (None = all)."""
    dev = torch.device(bc.device)
    n = canonical_size(street) if indices is None else int(indices.shape[0])
    d = 1 if street == 3 else sc.bins
    meta = _feature_meta(street, sc, bc, n)
    fdir = bc.resolved_out_dir() / "features"
    paths = {k: fdir / f"{STREET_NAMES[street]}_{k}.npy" for k in ("features", "equity", "weights")}
    meta_path = fdir / f"{STREET_NAMES[street]}_features.json"
    if bc.feature_cache and meta_path.exists():
        old = json.loads(meta_path.read_text())
        if old.get("meta") == meta and old.get("complete"):
            log(f"# {STREET_NAMES[street]}: reusing cached features in {fdir}")
            arrs = [np.load(paths[k], mmap_mode="r") for k in ("features", "equity", "weights")]
            return arrs[0], arrs[1], arrs[2], old["stats"]
    if bc.feature_cache:
        fdir.mkdir(parents=True, exist_ok=True)
        meta_path.unlink(missing_ok=True)
        feats = np.lib.format.open_memmap(paths["features"], "w+", np.float32, (n, d))
        eqs = np.lib.format.open_memmap(paths["equity"], "w+", np.float32, (n,))
        wts = np.lib.format.open_memmap(paths["weights"], "w+", np.uint8, (n,))
    else:
        feats = np.empty((n, d), np.float32)
        eqs = np.empty(n, np.float32)
        wts = np.empty(n, np.uint8)
    gen = torch.Generator(device=dev)
    cache_dir = bc.resolved_cache_dir()
    t0 = time.time()
    last = t0
    chunk = max(1, sc.chunk)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        idx = np.arange(s, e, dtype=np.int64) if indices is None else indices[s:e]
        cards = torch.from_numpy(
            representatives(street, idx, cache_dir, use_cache=cache_dir is not None)
        ).to(dev)
        gen.manual_seed(bc.seed * 1_000_003 + street * 7_919 + s)
        f, eq = street_features(street, cards, sc.bins, sc.runouts, gen, bc.max_rows)
        feats[s:e] = f.cpu().numpy()
        eqs[s:e] = eq.cpu().numpy()
        wts[s:e] = orbit_sizes(street, cards).cpu().numpy().astype(np.uint8)
        now = time.time()
        if now - last > 30 or e == n:
            rate = e / max(now - t0, 1e-9)
            log(
                f"# {STREET_NAMES[street]} features {e:,d}/{n:,d} | {rate:,.0f} idx/s "
                f"| eta {_fmt_time((n - e) / max(rate, 1e-9))}"
            )
            last = now
    secs = time.time() - t0
    w64 = wts.astype(np.float64)
    eq64 = np.asarray(eqs, dtype=np.float64)
    wsum = max(w64.sum(), 1.0)
    mean = float((w64 * eq64).sum() / wsum)
    stats = {
        "seconds": secs,
        "indices_per_second": n / max(secs, 1e-9),
        "count": n,
        "raw_hands": int(w64.sum()),
        "equity_mean_weighted": mean,
        "equity_std_weighted": float(math.sqrt(max((w64 * (eq64 - mean) ** 2).sum() / wsum, 0))),
        "equity_min": float(eq64.min()) if n else None,
        "equity_max": float(eq64.max()) if n else None,
    }
    if street < 3:
        stats["mean_histogram_weighted"] = [
            float(v) for v in (np.asarray(feats, np.float64) * w64[:, None]).sum(0) / wsum
        ]
    if bc.feature_cache:
        for a in (feats, eqs, wts):
            a.flush()
        meta_path.write_text(json.dumps({"meta": meta, "complete": True, "stats": stats}))
    return feats, eqs, wts, stats


def build_street(
    street: int | str, bc: BuildConfig, log: Callable[[str], None] = print
) -> StreetResult:
    """Features -> k-means -> table + sidecar for one street (see module docs)."""
    street = STREET_IDS[street] if isinstance(street, str) else int(street)
    name = STREET_NAMES[street]
    sc = bc.streets.get(name) or StreetConfig()
    dev = torch.device(bc.device)
    size = canonical_size(street)
    rng = np.random.default_rng([bc.seed, street])
    indices = None
    if bc.limit is not None and bc.limit < size:
        indices = _sample_indices(size, int(bc.limit), rng)
    n = size if indices is None else int(indices.shape[0])
    log(f"# {name}: {n:,d} of {size:,d} canonical indices, {sc.buckets} buckets on {dev}")
    cache_dir = bc.resolved_cache_dir()
    if cache_dir is not None and indices is None:
        build_canonical_cache(street, cache_dir, log)

    feats, eqs, wts, fstats = compute_features(street, sc, bc, indices, log)

    # Fit.
    t0 = time.time()
    metric = "l2" if street == 3 else "emd"
    fit_rows = None
    if sc.sample is not None and sc.sample < n:
        fit_rows = _sample_indices(n, int(sc.sample), rng)
    xf = torch.from_numpy(np.array(feats if fit_rows is None else feats[fit_rows])).to(dev)
    wf = torch.from_numpy(np.array(wts if fit_rows is None else wts[fit_rows])).to(dev)
    wf = wf.float() if bc.weighted else torch.ones_like(wf, dtype=torch.float32)
    gen = torch.Generator(device=dev)
    gen.manual_seed(bc.seed * 1_000_003 + 17 * street + 1)
    centres, history = kmeans(xf, sc.buckets, wf, metric, sc.iters, gen, sc.fit_chunk)
    order = centre_equity(centres, metric).argsort()
    centres = centres[order]
    del xf, wf
    fit_secs = time.time() - t0
    k_eff = int(centres.shape[0])
    log(
        f"# {name}: fitted {k_eff} centres on {n if fit_rows is None else len(fit_rows):,d} "
        f"indices in {_fmt_time(fit_secs)} (objective {history[0]:.5f} -> {history[-1]:.5f})"
    )

    # Assign every computed index.
    t0 = time.time()
    labels = np.empty(n, np.uint32)
    objective = 0.0
    step = max(sc.fit_chunk, 1) * 16
    for s in range(0, n, step):
        x = torch.from_numpy(np.array(feats[s : s + step])).to(dev).float()
        if metric == "emd":
            x = to_cdf(x)
        lab, dmin = assign(x, centres, metric, sc.fit_chunk)
        w = torch.from_numpy(np.array(wts[s : s + step])).to(dev).float()
        objective += float((w * dmin).sum())
        labels[s : s + step] = lab.cpu().numpy().astype(np.uint32)
    assign_secs = time.time() - t0

    # Write the full-length table.
    out_dir = bc.resolved_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    dtype = np.uint16 if sc.buckets <= 65535 else np.uint32
    path = out_dir / f"{name}.npy"
    tmp = out_dir / f".{name}.tmp{os.getpid()}.npy"
    table = np.lib.format.open_memmap(tmp, "w+", dtype, (size,))
    if indices is None:
        table[:] = labels.astype(dtype)
    else:
        blk = 1 << 24
        for s in range(0, size, blk):
            e = min(size, s + blk)
            table[s:e] = (np.arange(s, e, dtype=np.int64) % k_eff).astype(dtype)
        table[indices] = labels.astype(dtype)
    table.flush()
    del table
    os.replace(tmp, path)

    w64 = np.asarray(wts, dtype=np.float64)
    per_bucket = np.bincount(labels, weights=w64, minlength=k_eff)
    eq_bucket = np.bincount(labels, weights=w64 * np.asarray(eqs, np.float64), minlength=k_eff)
    nz = per_bucket > 0
    info = {
        "street": name,
        "canonical_size": size,
        "computed": n,
        "padded": size - n,
        "buckets": sc.buckets,
        "buckets_fitted": k_eff,
        "buckets_nonempty": int(nz.sum()),
        "dtype": np.dtype(dtype).name,
        "table": str(path),
        "metric": "emd (L1 of CDFs)" if metric == "emd" else "squared L2 on equity",
        "config": {"street": asdict(sc), **{k: v for k, v in asdict(bc).items() if k != "streets"}},
        "features": fstats,
        "fit": {
            "rows": n if fit_rows is None else len(fit_rows),
            "objective_per_iteration": history,
            "seconds": fit_secs,
        },
        "assign": {"objective": objective / max(w64.sum(), 1.0), "seconds": assign_secs},
        "bucket_raw_hands": {
            "min": float(per_bucket[nz].min()) if nz.any() else 0.0,
            "max": float(per_bucket.max()) if k_eff else 0.0,
            "mean": float(per_bucket[nz].mean()) if nz.any() else 0.0,
        },
        "bucket_mean_equity": [
            float(v) for v in np.where(nz, eq_bucket / np.maximum(per_bucket, 1e-12), np.nan)
        ],
        "centres": centres.cpu().tolist(),
        "git": git_hash(),
        "torch": torch.__version__,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (out_dir / f"{name}.json").write_text(json.dumps(info, indent=1, default=str))
    log(
        f"# {name}: wrote {path} ({size:,d} x {np.dtype(dtype).name}, "
        f"{int(nz.sum())} non-empty buckets) in {_fmt_time(assign_secs)} assign"
    )
    return StreetResult(street, path, info, indices, labels, feats, eqs, wts)


def build(
    bc: BuildConfig, streets: list[str] | None = None, log: Callable[[str], None] = print
) -> dict[str, StreetResult]:
    """Build the tables of ``streets`` (default: those in the config)."""
    names = streets or [s for s in ("flop", "turn", "river") if s in bc.streets]
    return {s: build_street(s, bc, log) for s in names}


def table_paths(out_dir: str | Path) -> list[str | None]:
    """``[None, flop, turn, river]`` table paths in ``out_dir`` (None when
    missing), ready for ``CardAbstraction(tables=...)`` / ``cards.tables``."""
    out = Path(out_dir)
    return [None] + [
        str(out / f"{s}.npy") if (out / f"{s}.npy").exists() else None
        for s in ("flop", "turn", "river")
    ]


def num_buckets(out_dir: str | Path) -> list[int | None]:
    """Bucket counts ``[None, flop, turn, river]`` from the JSON sidecars."""
    out = Path(out_dir)
    res: list[int | None] = [None]
    for s in ("flop", "turn", "river"):
        p = out / f"{s}.json"
        res.append(int(json.loads(p.read_text())["buckets"]) if p.exists() else None)
    return res


__all__ = [
    "BOARD_LEN",
    "BuildConfig",
    "StreetConfig",
    "StreetResult",
    "assign",
    "build",
    "build_street",
    "centre_equity",
    "compute_features",
    "kmeans",
    "kmeanspp",
    "load_config",
    "main",
    "num_buckets",
    "num_cards",
    "street_features",
    "table_paths",
    "to_cdf",
]


# ------------------------------------------------------------------------------- CLI


def load_config(path: str | Path) -> BuildConfig:
    """A :class:`BuildConfig` from a YAML file with a ``buckets:`` section
    (a top-level ``seed:`` is used when the section has none)."""
    from ..config import load_yaml

    raw = load_yaml(path)
    section = dict(raw.get("buckets") or {})
    if "seed" not in section and "seed" in raw:
        section["seed"] = raw["seed"]
    return BuildConfig.from_dict(section)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Build postflop card-bucket tables (DESIGN.md 5.3) for the MCCFR blueprint."
    )
    ap.add_argument("--config", required=True, help="YAML config (configs/buckets_*.yaml)")
    ap.add_argument("--streets", help="comma-separated subset of flop,turn,river")
    ap.add_argument("--limit", type=int, help="only this many random indices per street (smoke)")
    ap.add_argument("--sample", type=int, help="fit centres on this many indices (every street)")
    ap.add_argument("--device", help="torch device, e.g. cuda or cpu (overrides the config)")
    ap.add_argument("--out-dir", help="output directory (overrides the config)")
    ap.add_argument("--seed", type=int, help="override the seed")
    ap.add_argument("--no-feature-cache", action="store_true", help="keep features in RAM only")
    args = ap.parse_args(argv)

    bc = load_config(args.config)
    if args.limit is not None:
        bc.limit = args.limit
    if args.sample is not None:
        for sc in bc.streets.values():
            sc.sample = args.sample
    if args.device:
        bc.device = args.device
    if args.out_dir:
        bc.out_dir = args.out_dir
    if args.seed is not None:
        bc.seed = args.seed
    if args.no_feature_cache:
        bc.feature_cache = False
    streets = [s.strip() for s in args.streets.split(",")] if args.streets else None
    for s in streets or []:
        if s not in ("flop", "turn", "river"):
            ap.error(f"unknown street {s!r}")
        bc.streets.setdefault(s, StreetConfig())
    print(f"# git {git_hash()} | config {args.config} | device {bc.device} | out {bc.out_dir}")
    t0 = time.time()
    res = build(bc, streets)
    print(f"# done in {_fmt_time(time.time() - t0)}")
    out = bc.resolved_out_dir()
    buckets = [169] + [
        res[s].info["buckets"] if s in res else bc.streets.get(s, StreetConfig()).buckets
        for s in ("flop", "turn", "river")
    ]
    paths = {s: str(out / f"{s}.npy") for s in ("flop", "turn", "river")}
    print("# MCCFR config (mccfr.cards):")
    print(f"#   buckets: {buckets}")
    print(
        f"#   tables: {{preflop: null, flop: {paths['flop']}, turn: {paths['turn']}, "
        f"river: {paths['river']}}}"
    )
    return 0
