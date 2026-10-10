"""Hand-strength network inputs by table lookup (per canonical hand class).

The card-bucket build (:mod:`pokerbot.abstraction.buckets`) stores, for every
suit-isomorphic (hole, board) class of each postflop street, its equity
against a random hand (``features/<street>_equity.npy``) and, on the flop and
turn, a 10-bin histogram of its river equity over the runouts
(``features/<street>_features.npy``). Looking these up gives the networks hand
strength for the price of a canonical index (the Rust indexer, CPU) and a
gather, instead of Monte Carlo equity per decision.

Columns appended to the scalars (``NUM_STRENGTH = 11``): the acting player's
equity, then the 10-bin histogram (zeros preflop and on the river). Preflop
equity comes from a 169-class table computed once per process on the CPU
with a fixed seed, so training and play see identical numbers.

``features.strength_tables`` names the bucket-build directory (absolute, or
relative to the repository root). The traversal computes the columns once per
root deal for both seats and every street (:meth:`StrengthTables.root_features`)
and gathers them per slot; everything else appends them to a feature dict
with :func:`add_strength`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ...env.cards import NO_CARD
from ...env.obs import SCALAR_NAMES

NUM_STRENGTH = 11
BINS = 10
BOARD_LEN = (0, 3, 4, 5)
STREET_NAMES = ("preflop", "flop", "turn", "river")
_STREET_COL = SCALAR_NAMES.index("preflop")  # street one-hot in the base scalars
PREFLOP_SAMPLES = 20000
PREFLOP_SEED = 0

_CACHE: dict[str, StrengthTables] = {}
_PREFLOP: np.ndarray | None = None


def _canonical_index(street: int, cards: np.ndarray) -> np.ndarray:
    from ...abstraction.isomorphism import canonical_index_batch

    return canonical_index_batch(street, np.asarray(cards, dtype=np.int64)).numpy()


def preflop_equity() -> np.ndarray:
    """``[169]`` equity against a random hand per canonical preflop class
    (Monte Carlo on the CPU with a fixed seed; computed once per process)."""
    global _PREFLOP
    if _PREFLOP is None:
        from ...abstraction.isomorphism import unindex_batch
        from ...env.equity import equity_vs_random

        reps = torch.as_tensor(np.asarray(unindex_batch(0, np.arange(169)), dtype=np.int64))
        gen = torch.Generator().manual_seed(PREFLOP_SEED)
        eq = equity_vs_random(reps[:, :2], None, PREFLOP_SAMPLES, gen)
        _PREFLOP = eq.numpy().astype(np.float32)
    return _PREFLOP


class StrengthTables:
    """Per-class strength features of a bucket build (memory-mapped)."""

    def __init__(self, path: str | Path | None) -> None:
        self.path = Path(path) if path is not None else None
        self.equity: dict[int, np.ndarray] = {}
        self.hist: dict[int, np.ndarray] = {}
        if self.path is not None:
            feat = self.path / "features"
            for s in (1, 2, 3):
                name = STREET_NAMES[s]
                self.equity[s] = np.load(feat / f"{name}_equity.npy", mmap_mode="r")
                if s < 3:
                    h = np.load(feat / f"{name}_features.npy", mmap_mode="r")
                    if h.ndim != 2 or h.shape[1] != BINS:
                        raise ValueError(f"{name} histogram features must be [n, {BINS}]")
                    self.hist[s] = h

    def preflop(self) -> np.ndarray:
        return preflop_equity()

    def street_rows(self, street: int, idx: np.ndarray) -> np.ndarray:
        """``[m, 11]`` features of canonical indices ``idx`` on a postflop street."""
        out = np.zeros((len(idx), NUM_STRENGTH), np.float32)
        order = np.argsort(idx)  # sorted reads from the memory-mapped files
        out[order, 0] = self.equity[street][idx[order]]
        if street < 3:
            out[order, 1:] = self.hist[street][idx[order]]
        return out

    def lookup(self, hole: np.ndarray, board: np.ndarray, street: np.ndarray) -> np.ndarray:
        """``[n, 11]`` for hands ``hole [n, 2]`` on ``board [n, 5]`` (``NO_CARD``
        padded) at ``street [n]`` (0 preflop ... 3 river)."""
        hole = np.asarray(hole, dtype=np.int64)
        board = np.asarray(board, dtype=np.int64)
        street = np.asarray(street, dtype=np.int64)
        out = np.zeros((len(hole), NUM_STRENGTH), np.float32)
        for s in range(4):
            rows = np.nonzero(street == s)[0]
            if rows.size == 0:
                continue
            k = BOARD_LEN[s]
            cards = np.concatenate([hole[rows], board[rows, :k]], 1)
            if (cards == NO_CARD).any():
                raise ValueError("a hand's board is not dealt as far as its street")
            idx = _canonical_index(s, cards)
            if s == 0:
                out[rows, 0] = self.preflop()[idx]
            else:
                out[rows] = self.street_rows(s, idx)
        return out

    def root_features(self, cards9: np.ndarray) -> np.ndarray:
        """``[K, 2, 4, 11]`` for dealt hands ``cards9 [K, 9]`` (seat 0's hole
        cards, seat 1's, then the board): both seats on every street."""
        cards9 = np.asarray(cards9, dtype=np.int64)
        K = cards9.shape[0]
        hole = np.concatenate([cards9[:, 0:2], cards9[:, 2:4]])  # [2K, 2] seat-major
        board = np.concatenate([cards9[:, 4:9], cards9[:, 4:9]])
        out = np.zeros((2 * K, 4, NUM_STRENGTH), np.float32)
        for s in range(4):
            b = board.copy()
            b[:, BOARD_LEN[s] :] = NO_CARD
            out[:, s] = self.lookup(hole, b, np.full(2 * K, s))
        return out.reshape(2, K, 4, NUM_STRENGTH).transpose(1, 0, 2, 3).copy()


def resolve(path: str | Path) -> Path:
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    from ...config import REPO_ROOT

    return REPO_ROOT / p


def load_strength(path: str) -> StrengthTables:
    """The tables named by ``features.strength_tables`` (cached per process).
    Tests may put stand-ins into ``_CACHE`` under any key."""
    key = str(path)
    if key not in _CACHE:
        _CACHE[key] = StrengthTables(resolve(path))
    return _CACHE[key]


def add_strength(feats: dict[str, torch.Tensor], tables: StrengthTables) -> dict[str, torch.Tensor]:
    """Append the strength columns of each row's own hand (``cards[:, :2]``)
    on its board (``cards[:, 2:]``) at its street (the scalars' one-hot)."""
    cards = feats["cards"].detach().cpu().numpy()
    street = feats["scalars"][:, _STREET_COL : _STREET_COL + 4].argmax(1).cpu().numpy()
    cols = tables.lookup(cards[:, :2], cards[:, 2:], street)
    sc = feats["scalars"]
    extra = torch.from_numpy(cols).to(sc.device, sc.dtype)
    return {**feats, "scalars": torch.cat([sc, extra], 1)}
