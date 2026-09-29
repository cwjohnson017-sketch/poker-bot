"""Batched 5/6/7-card hand evaluator in pure torch.

Algorithm (no large tables, only gathers into 8192-entry tables):

* rank counts are packed as 3-bit fields (``sum 8**rank``) and unpacked to a
  ``[N, 13]`` count matrix, giving bitmasks of ranks held >= 1, 2, 3, 4 times;
* per-suit 13-bit rank masks come from ``sum 1 << (13*suit + rank)``;
* straights and "keep the top k ranks of a mask" are 8192-entry lookups.

For sets of k distinct ranks, comparing their 13-bit masks as integers is
the same as comparing the sorted rank lists lexicographically, so every tie
breaker is a pair of masks ``(primary << 13) | kickers``. The final rank is
``category << 26 | tiebreak`` (fits in 30 bits); higher is better and
``hand_category(rank) = rank >> 26``.

Everything is device-agnostic; tables are built once per device and cached.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

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
CATEGORY_SHIFT = 26

_SPREAD5 = 0b1001001001001  # bit 0 of five consecutive 3-bit fields
_FIELD_LOW = sum(1 << (3 * i) for i in range(13))  # bit 0 of all 13 fields

_TABLES: dict[str, dict[str, torch.Tensor]] = {}


def _build_tables() -> dict[str, torch.Tensor]:
    arange13 = torch.arange(13, dtype=torch.long)
    bitval = 1 << arange13  # [13]
    masks = torch.arange(8192, dtype=torch.long)
    bits = (masks[:, None] >> arange13) & 1  # [8192, 13]
    pop = bits.sum(1)
    # number of set bits at positions >= i (counting from the ace down)
    from_top = bits.flip(1).cumsum(1).flip(1)
    top = torch.stack([(bits * (from_top <= k) * bitval).sum(1) for k in range(6)])  # [6, 8192]
    straight = torch.zeros(8192, dtype=torch.long)
    wheel = (1 << 12) | 0b1111
    straight = torch.where((masks & wheel) == wheel, torch.full_like(masks, 4), straight)
    for high in range(4, 13):
        pat = 0b11111 << (high - 4)
        straight = torch.where((masks & pat) == pat, torch.full_like(masks, high + 1), straight)
    cards = torch.arange(52, dtype=torch.long)
    r, s = cards // 4, cards % 4
    # compress 5 bits spread at positions 0,3,6,9,12 into 5 adjacent bits
    spread = torch.arange(_SPREAD5 + 1, dtype=torch.long)
    compress = sum(((spread >> (3 * j)) & 1) << j for j in range(5))
    return {
        "compress": compress,
        "pop": pop,
        "top": top,
        "straight": straight,
        "rank_pack": 8**r,  # 3-bit count field per rank
        "suit_bit": 1 << (13 * s + r),  # one bit per card in 4 13-bit suit fields
        "shift13": 13 * torch.arange(4, dtype=torch.long),
    }


def tables(device: torch.device | str) -> dict[str, torch.Tensor]:
    key = str(torch.device(device))
    t = _TABLES.get(key)
    if t is None:
        if "cpu" not in _TABLES:
            _TABLES["cpu"] = _build_tables()
        t = {k: v.to(device) for k, v in _TABLES["cpu"].items()}
        _TABLES[key] = t
    return t


@torch.no_grad()
def evaluate_batch(cards: torch.Tensor) -> torch.Tensor:
    """Rank ``[..., k]`` hands of 5 to 7 distinct cards; returns ``[...]`` long.

    Higher is better; equal ranks tie. Any leading shape is allowed.
    """
    lead = cards.shape[:-1]
    c = cards.reshape(-1, cards.shape[-1]).long()
    T = tables(c.device)
    top, pop, straight = T["top"], T["pop"], T["straight"]

    # 3-bit count per rank; counts are 0..4 so bit 2 set <=> count == 4
    packed = T["rank_pack"][c].sum(1)  # [N]
    b0 = packed & _FIELD_LOW
    b1 = (packed >> 1) & _FIELD_LOW
    b2 = (packed >> 2) & _FIELD_LOW
    comp = T["compress"]

    def squeeze(x: torch.Tensor) -> torch.Tensor:  # field bits -> 13-bit rank mask
        return (
            comp[x & _SPREAD5]
            | (comp[(x >> 15) & _SPREAD5] << 5)
            | (comp[(x >> 30) & _SPREAD5] << 10)
        )

    m1 = squeeze(b0 | b1 | b2)  # count >= 1
    m2 = squeeze(b1 | b2)  # count >= 2
    m3 = squeeze((b0 & b1) | b2)  # count >= 3
    m4 = squeeze(b2)  # count == 4

    sbits = T["suit_bit"][c].sum(1)
    smask = (sbits[:, None] >> T["shift13"]) & 8191  # [N, 4]
    fmask = (smask * (pop[smask] >= 5).long()).sum(1)  # at most one suit has 5+

    t1 = top[1]
    sf = straight[fmask]
    st = straight[m1]
    q = t1[m4]
    t = t1[m3]
    fh_pair = t1[m2 & ~t]
    top2p = top[2][m2]
    p1 = t1[m2]

    s = CATEGORY_SHIFT
    val = top[5][m1]  # high card (category 0)
    val = torch.where(m2 != 0, (PAIR << s) | (p1 << 13) | top[3][m1 & ~p1], val)
    val = torch.where(pop[m2] >= 2, (TWO_PAIR << s) | (top2p << 13) | t1[m1 & ~top2p], val)
    val = torch.where(m3 != 0, (TRIPS << s) | (t << 13) | top[2][m1 & ~t], val)
    val = torch.where(st > 0, (STRAIGHT << s) | st, val)
    val = torch.where(fmask != 0, (FLUSH << s) | top[5][fmask], val)
    val = torch.where((m3 != 0) & (fh_pair != 0), (FULL_HOUSE << s) | (t << 13) | fh_pair, val)
    val = torch.where(m4 != 0, (QUADS << s) | (q << 13) | t1[m1 & ~q], val)
    val = torch.where(sf > 0, (STRAIGHT_FLUSH << s) | sf, val)
    return val.reshape(lead)


def evaluate7_batch(cards: torch.Tensor) -> torch.Tensor:
    """``[N, 7]`` long -> ``[N]`` long ranks (higher is better)."""
    if cards.shape[-1] != 7:
        raise ValueError("evaluate7_batch expects 7 cards per hand")
    return evaluate_batch(cards)


def hand_category(ranks: torch.Tensor | int) -> torch.Tensor | int:
    """Map ranks to categories 0 (high card) .. 8 (straight flush)."""
    if isinstance(ranks, torch.Tensor):
        return ranks >> CATEGORY_SHIFT
    return int(ranks) >> CATEGORY_SHIFT


def _scalar(cards: Sequence[int], k: int) -> int:
    cards = list(cards)
    if len(cards) != k or len(set(cards)) != k or not all(0 <= c < 52 for c in cards):
        raise ValueError(f"expected {k} distinct cards in 0..51, got {cards}")
    return int(evaluate_batch(torch.tensor([cards], dtype=torch.long))[0])


def evaluate7(cards: Sequence[int]) -> int:
    return _scalar(cards, 7)


def evaluate6(cards: Sequence[int]) -> int:
    return _scalar(cards, 6)


def evaluate5(cards: Sequence[int]) -> int:
    return _scalar(cards, 5)
