"""Card encoding and tensor dealing.

A card is an int in ``0..52``: ``card = rank * 4 + suit`` with ranks
``0 = 2 ... 12 = A`` and suits ``0 = c, 1 = d, 2 = h, 3 = s``.

A deck is a permutation of ``0..52`` dealt in this order (seat index, not
button): player 0 hole (2), player 1 hole (2), flop (3), turn (1), river (1).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
NUM_CARDS = 52
NO_CARD = 52  # padding index for undealt / hidden card slots

# Positions inside a dealt deck (the first 9 cards are all a heads-up hand uses).
HOLE_SLICES = (slice(0, 2), slice(2, 4))
FLOP_SLICE = slice(4, 7)
TURN_INDEX = 7
RIVER_INDEX = 8
BOARD_SLICE = slice(4, 9)
DEALT_CARDS = 9
# Number of visible board cards on each street (0 preflop, 1 flop, 2 turn, 3 river).
BOARD_LEN_BY_STREET = (0, 3, 4, 5)


def card_from_str(s: str) -> int:
    if len(s) != 2 or s[0] not in RANK_CHARS or s[1] not in SUIT_CHARS:
        raise ValueError(f"bad card string {s!r}")
    return RANK_CHARS.index(s[0]) * 4 + SUIT_CHARS.index(s[1])


def card_to_str(c: int) -> str:
    c = int(c)
    if not 0 <= c < NUM_CARDS:
        raise ValueError(f"bad card {c}")
    return RANK_CHARS[c // 4] + SUIT_CHARS[c % 4]


def cards_from_str(s: str) -> list[int]:
    """Parse a concatenation like ``"AsKd"`` or a space-separated ``"As Kd"``."""
    s = s.replace(" ", "")
    if len(s) % 2:
        raise ValueError(f"bad cards string {s!r}")
    return [card_from_str(s[i : i + 2]) for i in range(0, len(s), 2)]


def cards_to_str(cards: Sequence[int]) -> str:
    return " ".join(card_to_str(c) for c in cards)


def card_rank(cards: torch.Tensor) -> torch.Tensor:
    return torch.div(cards, 4, rounding_mode="floor")


def card_suit(cards: torch.Tensor) -> torch.Tensor:
    return cards % 4


def make_generator(seed: int, device: torch.device | str | None = None) -> torch.Generator:
    g = torch.Generator(device=torch.device(device) if device is not None else "cpu")
    g.manual_seed(int(seed))
    return g


def shuffled_decks(
    n: int,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """``[n, 52]`` long tensor; each row an independent uniform permutation of 0..51.

    Built by a per-row argsort of random 62-bit integer keys (ties are
    astronomically unlikely, and the sort is stable anyway).
    """
    if generator is not None:
        device = generator.device
    keys = torch.randint(
        0, 2**62, (n, NUM_CARDS), generator=generator, device=device, dtype=torch.long
    )
    return torch.argsort(keys, dim=1, stable=True)


def deck_from_hands(
    hole0: Sequence[int], hole1: Sequence[int], board: Sequence[int] = ()
) -> list[int]:
    """Build a full 52-card deck that deals the given cards (test helper).

    Unspecified board cards and the rest of the deck are filled with the
    remaining cards in ascending order.
    """
    head = list(hole0) + list(hole1) + list(board)
    if len(set(head)) != len(head):
        raise ValueError("duplicate cards")
    rest = [c for c in range(NUM_CARDS) if c not in set(head)]
    return head + rest
