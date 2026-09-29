"""Slow, obviously-correct hand evaluators used only as test oracles.

Values are ``category * 13**5 + tiebreak`` where the tiebreak lists ranks
grouped by (count desc, rank desc); straights use their high card (5 for
the wheel). Independent of the bitmask algorithm in ``pokerbot.env``.
"""

from __future__ import annotations

import itertools

import numpy as np

B5 = 13**5
COMBOS_7_5 = np.array(list(itertools.combinations(range(7), 5)), dtype=np.int64)  # [21, 5]


def naive5(cards) -> int:
    ranks = sorted((c // 4 for c in cards), reverse=True)
    suits = [c % 4 for c in cards]
    flush = len(set(suits)) == 1
    distinct = len(set(ranks)) == 5
    straight_high = -1
    if distinct and ranks[0] - ranks[4] == 4:
        straight_high = ranks[0]
    elif distinct and ranks == [12, 3, 2, 1, 0]:
        straight_high = 3
    counts = {r: ranks.count(r) for r in ranks}
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    shape = [cnt for _, cnt in groups]
    ordered = []
    for r, cnt in groups:
        ordered += [r] * cnt
    tb = 0
    for r in ordered:
        tb = tb * 13 + r
    if straight_high >= 0 and flush:
        return 8 * B5 + straight_high * 13**4
    if shape[0] == 4:
        return 7 * B5 + tb
    if shape[:2] == [3, 2]:
        return 6 * B5 + tb
    if flush:
        return 5 * B5 + tb
    if straight_high >= 0:
        return 4 * B5 + straight_high * 13**4
    if shape[0] == 3:
        return 3 * B5 + tb
    if shape[:2] == [2, 2]:
        return 2 * B5 + tb
    if shape[0] == 2:
        return 1 * B5 + tb
    return tb


def naive7(cards) -> int:
    return max(naive5(c) for c in itertools.combinations(cards, 5))


def naive5_np(cards: np.ndarray) -> np.ndarray:
    """Vectorized version of :func:`naive5` over ``[M, 5]``."""
    cards = cards.astype(np.int64)
    r = -np.sort(-(cards // 4), axis=1)
    s = cards % 4
    flush = (s == s[:, :1]).all(1)
    distinct = (np.diff(r, axis=1) != 0).all(1)
    wheel = distinct & (r[:, 0] == 12) & (r[:, 1] == 3)
    straight = (distinct & (r[:, 0] - r[:, 4] == 4)) | wheel
    high = np.where(wheel, 3, r[:, 0])
    cnt = (r[:, :, None] == r[:, None, :]).sum(2)
    key = -np.sort(-(cnt * 16 + r), axis=1)
    ordered = key % 16
    shape = key // 16
    tb = np.zeros(len(cards), dtype=np.int64)
    for i in range(5):
        tb = tb * 13 + ordered[:, i]
    st_tb = high * 13**4
    val = tb.copy()
    val = np.where(shape[:, 0] == 2, 1 * B5 + tb, val)
    val = np.where((shape[:, 0] == 2) & (shape[:, 2] == 2), 2 * B5 + tb, val)
    val = np.where(shape[:, 0] == 3, 3 * B5 + tb, val)
    val = np.where(straight, 4 * B5 + st_tb, val)
    val = np.where(flush, 5 * B5 + tb, val)
    val = np.where((shape[:, 0] == 3) & (shape[:, 3] == 2), 6 * B5 + tb, val)
    val = np.where(shape[:, 0] == 4, 7 * B5 + tb, val)
    val = np.where(straight & flush, 8 * B5 + st_tb, val)
    return val


def naive7_np(cards: np.ndarray) -> np.ndarray:
    """Max of :func:`naive5_np` over the 21 five-card subsets of ``[M, 7]``."""
    subsets = cards[:, COMBOS_7_5]  # [M, 21, 5]
    vals = naive5_np(subsets.reshape(-1, 5)).reshape(len(cards), 21)
    return vals.max(1)
