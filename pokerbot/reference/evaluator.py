"""Reference hand evaluator.

A rank is ``category << 20 | k1 << 16 | k2 << 12 | k3 << 8 | k4 << 4 | k5``
where ``category`` is 0 (high card) .. 8 (straight flush) and ``k1..k5`` are
the tie-break ranks (0 = deuce .. 12 = ace) in significance order. Higher is
better. The evaluator works directly on 5, 6 or 7 cards from per-suit rank
bitmasks and per-rank counts, so it does not enumerate 5-card subsets; the
tests check it against a naive best-of-subsets evaluator.

The absolute values are specific to this engine (the contract only fixes the
ordering and :func:`hand_category`).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

HIGH_CARD, PAIR, TWO_PAIR, TRIPS, STRAIGHT, FLUSH, FULL_HOUSE, QUADS, STRAIGHT_FLUSH = range(9)
CATEGORY_NAMES = (
    "high card",
    "pair",
    "two pair",
    "trips",
    "straight",
    "flush",
    "full house",
    "quads",
    "straight flush",
)

_WHEEL = (1 << 12) | 0b1111  # A 2 3 4 5


def _build_tables() -> tuple[list[int], list[int]]:
    high_bit = [-1] * 8192
    straight_high = [-1] * 8192
    for m in range(1, 8192):
        high_bit[m] = m.bit_length() - 1
    for m in range(8192):
        for h in range(12, 3, -1):
            window = 0b11111 << (h - 4)
            if m & window == window:
                straight_high[m] = h
                break
        else:
            if m & _WHEEL == _WHEEL:
                straight_high[m] = 3
    return high_bit, straight_high


_HB, _SH = _build_tables()
_HB_NP = np.array(_HB, dtype=np.int32)
_SH_NP = np.array(_SH, dtype=np.int32)


def _pack(mask: int, k: int, shift: int) -> int:
    val = 0
    for _ in range(k):
        b = _HB[mask]
        val |= b << shift
        mask &= ~(1 << b)
        shift -= 4
    return val


def _evaluate(cards: Sequence[int]) -> int:
    s0 = s1 = s2 = s3 = 0
    counts = [0] * 13
    for c in cards:
        c = int(c)
        r = c >> 2
        bit = 1 << r
        u = c & 3
        if u == 0:
            s0 |= bit
        elif u == 1:
            s1 |= bit
        elif u == 2:
            s2 |= bit
        else:
            s3 |= bit
        counts[r] += 1

    flush_val = 0
    for m in (s0, s1, s2, s3):
        if m.bit_count() >= 5:
            sh = _SH[m]
            if sh >= 0:
                return (STRAIGHT_FLUSH << 20) | (sh << 16)
            flush_val = (FLUSH << 20) | _pack(m, 5, 16)
            break

    any_mask = s0 | s1 | s2 | s3
    m4 = m3 = m2 = 0
    for r in range(13):
        n = counts[r]
        if n == 4:
            m4 |= 1 << r
        elif n == 3:
            m3 |= 1 << r
        elif n == 2:
            m2 |= 1 << r

    if m4:
        q = _HB[m4]
        return (QUADS << 20) | (q << 16) | (_HB[any_mask & ~(1 << q)] << 12)
    if m3:
        t = _HB[m3]
        rest = (m3 & ~(1 << t)) | m2
        if rest:
            return (FULL_HOUSE << 20) | (t << 16) | (_HB[rest] << 12)
    if flush_val:
        return flush_val
    sh = _SH[any_mask]
    if sh >= 0:
        return (STRAIGHT << 20) | (sh << 16)
    if m3:
        t = _HB[m3]
        return (TRIPS << 20) | (t << 16) | _pack(any_mask & ~(1 << t), 2, 12)
    if m2:
        p1 = _HB[m2]
        rest = m2 & ~(1 << p1)
        if rest:
            p2 = _HB[rest]
            k = _HB[any_mask & ~((1 << p1) | (1 << p2))]
            return (TWO_PAIR << 20) | (p1 << 16) | (p2 << 12) | (k << 8)
        return (PAIR << 20) | (p1 << 16) | _pack(any_mask & ~(1 << p1), 3, 12)
    return (HIGH_CARD << 20) | _pack(any_mask, 5, 16)


def _check(cards: Sequence[int], n: int) -> None:
    if len(cards) != n:
        raise ValueError(f"expected {n} cards, got {len(cards)}")
    if len(set(int(c) for c in cards)) != n or any(not 0 <= int(c) < 52 for c in cards):
        raise ValueError(f"cards must be {n} distinct ints in 0..52: {list(cards)}")


def evaluate5(cards: Sequence[int]) -> int:
    _check(cards, 5)
    return _evaluate(cards)


def evaluate6(cards: Sequence[int]) -> int:
    _check(cards, 6)
    return _evaluate(cards)


def evaluate7(cards: Sequence[int]) -> int:
    _check(cards, 7)
    return _evaluate(cards)


def evaluate(cards: Sequence[int]) -> int:
    """Evaluate 5, 6 or 7 cards (no validation beyond the count)."""
    if not 5 <= len(cards) <= 7:
        raise ValueError("evaluate needs 5..7 cards")
    return _evaluate(cards)


def hand_category(rank: int) -> int:
    return int(rank) >> 20


# ----------------------------------------------------------------------------
# numpy batch evaluator (same encoding, vectorized)
# ----------------------------------------------------------------------------


def _np_pack(mask: np.ndarray, k: int, shift: int) -> np.ndarray:
    val = np.zeros(mask.shape, dtype=np.int32)
    mask = mask.copy()
    for _ in range(k):
        b = _HB_NP[mask]
        bc = np.maximum(b, 0)
        val |= np.where(b >= 0, bc << shift, 0)
        mask &= ~(np.int32(1) << bc)
        shift -= 4
    return val


def evaluate_batch(cards: np.ndarray) -> np.ndarray:
    """Evaluate an ``(N, 7)`` (or ``(N, 5)``/``(N, 6)``) array of cards -> ``int32[N]``."""
    cards = np.asarray(cards)
    if cards.ndim != 2 or not 5 <= cards.shape[1] <= 7:
        raise ValueError(f"expected shape (N, 5..7), got {cards.shape}")
    c = cards.astype(np.int32)
    n = c.shape[0]
    ranks = c >> 2
    suits = c & 3
    rbits = np.int32(1) << ranks  # (N, k)

    suit_masks = np.zeros((n, 4), dtype=np.int32)
    for u in range(4):
        suit_masks[:, u] = np.bitwise_or.reduce(np.where(suits == u, rbits, 0), axis=1)
    any_mask = np.bitwise_or.reduce(suit_masks, axis=1)

    counts = np.zeros((n, 13), dtype=np.int32)
    rows = np.repeat(np.arange(n), c.shape[1])
    np.add.at(counts, (rows, ranks.ravel()), 1)
    weights = np.int32(1) << np.arange(13, dtype=np.int32)
    m4 = ((counts == 4) * weights).sum(axis=1).astype(np.int32)
    m3 = ((counts == 3) * weights).sum(axis=1).astype(np.int32)
    m2 = ((counts == 2) * weights).sum(axis=1).astype(np.int32)

    suit_counts = np.zeros((n, 4), dtype=np.int32)
    for u in range(4):
        suit_counts[:, u] = (suits == u).sum(axis=1)
    has_flush = (suit_counts >= 5).any(axis=1)
    fsuit = np.argmax(suit_counts, axis=1)
    fmask = np.where(has_flush, suit_masks[np.arange(n), fsuit], 0).astype(np.int32)

    one = np.int32(1)

    sf_high = _SH_NP[fmask]
    v_sf = (STRAIGHT_FLUSH << 20) | (np.maximum(sf_high, 0) << 16)

    q = np.maximum(_HB_NP[m4], 0)
    v_quads = (QUADS << 20) | (q << 16) | (np.maximum(_HB_NP[any_mask & ~(one << q)], 0) << 12)

    t = np.maximum(_HB_NP[m3], 0)
    fh_rest = (m3 & ~(one << t)) | m2
    v_fh = (FULL_HOUSE << 20) | (t << 16) | (np.maximum(_HB_NP[fh_rest], 0) << 12)

    v_flush = (FLUSH << 20) | _np_pack(fmask, 5, 16)

    st_high = _SH_NP[any_mask]
    v_straight = (STRAIGHT << 20) | (np.maximum(st_high, 0) << 16)

    v_trips = (TRIPS << 20) | (t << 16) | _np_pack(any_mask & ~(one << t), 2, 12)

    p1 = np.maximum(_HB_NP[m2], 0)
    rest2 = m2 & ~(one << p1)
    p2 = np.maximum(_HB_NP[rest2], 0)
    k2 = np.maximum(_HB_NP[any_mask & ~((one << p1) | (one << p2))], 0)
    v_two = (TWO_PAIR << 20) | (p1 << 16) | (p2 << 12) | (k2 << 8)

    v_pair = (PAIR << 20) | (p1 << 16) | _np_pack(any_mask & ~(one << p1), 3, 12)
    v_high = (HIGH_CARD << 20) | _np_pack(any_mask, 5, 16)

    conds = [
        has_flush & (sf_high >= 0),
        m4 != 0,
        (m3 != 0) & (fh_rest != 0),
        has_flush,
        st_high >= 0,
        m3 != 0,
        rest2 != 0,
        m2 != 0,
    ]
    vals = [v_sf, v_quads, v_fh, v_flush, v_straight, v_trips, v_two, v_pair]
    return np.select(conds, vals, default=v_high).astype(np.int32)
