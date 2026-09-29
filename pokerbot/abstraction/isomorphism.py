"""Torch/numpy helpers around the Rust suit-isomorphic canonical hand index.

The index itself lives in ``engine/src/isomorphism.rs`` (see its module
docs for the exact definition): ``poker_engine.canonical_index(street, hole,
board)`` maps a hand to a dense id in ``0..canonical_size(street)`` (169 /
1,286,792 / 13,960,050 / 123,156,254), equal for all hands related by a suit
permutation or by reordering the hole cards or the board, and
``canonical_unindex`` returns the class representative (suits dealt as
c, d, h, s in canonical order, each round sorted ascending). Bucket tables
are indexed by it.

This module adds:

* :func:`canonical_index_batch` - rows of a tensor/array through the Rust
  batch function;
* :func:`representatives` - representative cards of arbitrary indices,
  from the on-disk cache when present, else via ``canonical_unindex``;
* :func:`enumerate_canonical` - representatives of every index of a street,
  built once and cached under ``data/abstraction/`` (river: 862 MB);
* :func:`orbit_sizes` - how many raw ``(hole set, board set)`` hands each
  canonical class stands for (24 divided by the order of the suit
  permutations fixing the representative). Summed over a street they give
  ``C(52, 2) * C(50, k)``; the bucket k-means weights classes by it.
"""

from __future__ import annotations

import itertools
import os
from pathlib import Path

import numpy as np
import torch

from ..config import REPO_ROOT

BOARD_LEN = (0, 3, 4, 5)
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "abstraction"
_UNINDEX_CHUNK = 1 << 18


def _pe():
    import poker_engine as pe

    return pe


def canonical_size(street: int) -> int:
    return int(_pe().canonical_size(street))


def num_cards(street: int) -> int:
    return 2 + BOARD_LEN[street]


def canonical_index_batch(street: int, cards: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Canonical indices of ``[N, 2 + board_len]`` card rows (hole cards
    first), as an int64 tensor on the input's device (CPU for numpy input).
    Rows must hold distinct cards in ``0..52`` (the Rust side validates)."""
    device = cards.device if isinstance(cards, torch.Tensor) else torch.device("cpu")
    arr = cards.detach().cpu().numpy() if isinstance(cards, torch.Tensor) else np.asarray(cards)
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    if arr.ndim != 2 or arr.shape[1] != num_cards(street):
        raise ValueError(f"street {street} needs shape (N, {num_cards(street)}), got {arr.shape}")
    idx = _pe().canonical_index_batch(street, arr)
    return torch.from_numpy(idx.astype(np.int64)).to(device)


def unindex_batch(street: int, indices: np.ndarray | torch.Tensor) -> np.ndarray:
    """Representatives of ``indices`` via ``canonical_unindex`` (about 1M/s),
    ``uint8 [N, 2 + board_len]``: hole cards then board, each sorted."""
    unindex = _pe().canonical_unindex
    idx = np.asarray(indices.cpu() if isinstance(indices, torch.Tensor) else indices)
    flat: list[int] = []
    ext = flat.extend
    for i in idx.tolist():
        h, b = unindex(street, i)
        ext(h)
        ext(b)
    return np.asarray(flat, dtype=np.uint8).reshape(len(idx), num_cards(street))


def cache_path(street: int, cache_dir: str | Path | None = None) -> Path:
    return Path(cache_dir or DEFAULT_CACHE_DIR) / f"canonical_{street}.npy"


def _open_cache(street: int, cache_dir: str | Path | None) -> np.ndarray | None:
    p = cache_path(street, cache_dir)
    if not p.exists():
        return None
    arr = np.load(p, mmap_mode="r")
    if arr.shape != (canonical_size(street), num_cards(street)) or arr.dtype != np.uint8:
        return None
    return arr


def build_canonical_cache(street: int, cache_dir: str | Path | None = None, log=None) -> np.ndarray:
    """Write (if missing) and memory-map the representatives of every index."""
    arr = _open_cache(street, cache_dir)
    if arr is not None:
        return arr
    p = cache_path(street, cache_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}.npy")
    n, w = canonical_size(street), num_cards(street)
    out = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(n, w))
    for s in range(0, n, _UNINDEX_CHUNK):
        e = min(n, s + _UNINDEX_CHUNK)
        out[s:e] = unindex_batch(street, np.arange(s, e))
        if log is not None and (s // _UNINDEX_CHUNK) % 32 == 0:
            log(f"# canonical cache street {street}: {e:,d}/{n:,d}")
    out.flush()
    del out
    os.replace(tmp, p)
    return np.load(p, mmap_mode="r")


def enumerate_canonical(
    street: int, cache_dir: str | Path | None = None, device: torch.device | str = "cpu"
) -> torch.Tensor:
    """Representative cards of every canonical index of ``street``,
    ``uint8 [canonical_size, 2 + board_len]`` (row ``i`` is
    ``canonical_unindex(street, i)``), built once and cached on disk as
    ``<cache_dir>/canonical_<street>.npy`` (default ``data/abstraction``)."""
    arr = build_canonical_cache(street, cache_dir)
    return torch.from_numpy(np.array(arr)).to(device)


def representatives(
    street: int,
    indices: np.ndarray | torch.Tensor,
    cache_dir: str | Path | None = None,
    use_cache: bool = True,
) -> np.ndarray:
    """``uint8 [N, 2 + board_len]`` representatives of ``indices``: gathered
    from the on-disk cache when it exists (``use_cache``), else computed."""
    idx = np.asarray(indices.cpu() if isinstance(indices, torch.Tensor) else indices)
    if use_cache:
        arr = _open_cache(street, cache_dir)
        if arr is not None:
            return np.asarray(arr[idx])
    return unindex_batch(street, idx)


_PERMS = torch.tensor(list(itertools.permutations(range(4))), dtype=torch.long)  # [24, 4]


@torch.no_grad()
def orbit_sizes(street: int, cards: torch.Tensor) -> torch.Tensor:
    """Raw hands per canonical class, ``[N]`` int64, from representative rows
    ``[N, 2 + board_len]`` (any hand of the class works): the number of
    distinct ``(hole set, board set)`` images under the 24 suit permutations."""
    c = cards.long()
    perms = _PERMS.to(c.device)
    rank, suit = c // 4, c % 4
    img = rank[None] * 4 + perms[:, suit]  # [24, N, W]
    one = torch.ones((), dtype=torch.long, device=c.device)

    def sets(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:  # 52-bit card-set masks
        bits = one << x
        return bits[..., :2].sum(-1), bits[..., 2:].sum(-1)

    h0, b0 = sets(c)
    h, b = sets(img)
    same = (h == h0[None]) & (b == b0[None])  # [24, N]
    return 24 // same.sum(0)
