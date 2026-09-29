"""Canonical order of the 1326 two-card hole combos and card-conflict tables.

Combo ``i`` is the ``i``-th pair ``(a, b)`` with ``a < b`` in lexicographic
order, i.e. ``itertools.combinations(range(52), 2)`` order:
``combo_index(a, b) = a * (103 - a) // 2 + (b - a - 1)``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import lru_cache

import torch

NUM_CARDS = 52
NUM_COMBOS = 1326

_CARD1: list[int] = []
_CARD2: list[int] = []
for _a in range(NUM_CARDS):
    for _b in range(_a + 1, NUM_CARDS):
        _CARD1.append(_a)
        _CARD2.append(_b)


def combo_index(c1: int, c2: int) -> int:
    """Index of the unordered hole combo ``{c1, c2}`` in ``0..1325``."""
    a, b = (int(c1), int(c2)) if c1 < c2 else (int(c2), int(c1))
    if a == b or not (0 <= a < NUM_CARDS and 0 <= b < NUM_CARDS):
        raise ValueError(f"bad combo ({c1}, {c2})")
    return a * (103 - a) // 2 + (b - a - 1)


def combo_cards(index: int) -> tuple[int, int]:
    """Reverse of :func:`combo_index`: ``(low card, high card)``."""
    return _CARD1[index], _CARD2[index]


@lru_cache(maxsize=8)
def _tables(device: str) -> dict[str, torch.Tensor]:
    d = torch.device(device)
    cards = torch.tensor([_CARD1, _CARD2], dtype=torch.long).t().contiguous()  # [C, 2]
    inc = torch.zeros(NUM_COMBOS, NUM_CARDS, dtype=torch.bool)
    inc[torch.arange(NUM_COMBOS), cards[:, 0]] = True
    inc[torch.arange(NUM_COMBOS), cards[:, 1]] = True
    return {
        "cards": cards.to(d),
        "incidence": inc.to(d),  # [C, 52] bool
        "incidence_f": inc.float().to(d),
        "no_card": (~inc).t().contiguous().to(d),  # [52, C] combo avoids card
    }


def combo_table(device: torch.device | str = "cpu") -> torch.Tensor:
    """``[1326, 2]`` long: the two cards of every combo."""
    return _tables(str(torch.device(device)))["cards"]


def incidence(device: torch.device | str = "cpu", dtype: torch.dtype | None = None) -> torch.Tensor:
    """``[1326, 52]``: combo contains card (bool, or ``dtype`` when given)."""
    t = _tables(str(torch.device(device)))
    return t["incidence"] if dtype is None else t["incidence_f"].to(dtype)


def avoids_card(device: torch.device | str = "cpu") -> torch.Tensor:
    """``[52, 1326]`` bool: combo does not contain the card."""
    return _tables(str(torch.device(device)))["no_card"]


def conflict_matrix(device: torch.device | str = "cpu") -> torch.Tensor:
    """``[1326, 1326]`` bool: the two combos share a card."""
    return _conflict(str(torch.device(device)))


@lru_cache(maxsize=4)
def _conflict(device: str) -> torch.Tensor:
    inc = incidence(device, torch.float32)
    return (inc @ inc.t()) > 0


def valid_mask(board: Iterable[int], device: torch.device | str = "cpu") -> torch.Tensor:
    """``[1326]`` bool: combos disjoint from ``board`` (any number of cards)."""
    board = [int(c) for c in board]
    m = torch.ones(NUM_COMBOS, dtype=torch.bool, device=device)
    if board:
        m &= ~incidence(device)[:, board].any(1)
    return m


def valid_masks(
    boards: Sequence[Sequence[int]], device: torch.device | str = "cpu"
) -> torch.Tensor:
    """``[len(boards), 1326]`` bool, one :func:`valid_mask` per board."""
    onehot = torch.zeros(len(boards), NUM_CARDS, device=device)
    for i, b in enumerate(boards):
        if len(b):
            onehot[i, list(b)] = 1.0
    hit = onehot @ incidence(device, torch.float32).t()
    return hit == 0


def blocked_sum(reach: torch.Tensor) -> torch.Tensor:
    """For every combo ``c``: sum of ``reach`` over combos disjoint from ``c``.

    ``reach`` is ``[..., 1326]``. Inclusion-exclusion over the two cards of
    ``c``: total - (combos holding card 1) - (combos holding card 2) + ``c``.
    """
    inc = incidence(reach.device, reach.dtype)
    per_card = reach @ inc  # [..., 52]
    cards = combo_table(reach.device)
    total = reach.sum(-1, keepdim=True)
    return total - per_card[..., cards[:, 0]] - per_card[..., cards[:, 1]] + reach
