"""Card encoding shared by every engine: ``card = rank * 4 + suit``."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

RANK_CHARS = "23456789TJQKA"
SUIT_CHARS = "cdhs"
NUM_CARDS = 52


def card_from_str(s: str) -> int:
    """``"As" -> 51``. Accepts upper- or lower-case suit, upper- or lower-case rank."""
    if len(s) != 2:
        raise ValueError(f"bad card string {s!r}")
    r = RANK_CHARS.find(s[0].upper())
    u = SUIT_CHARS.find(s[1].lower())
    if r < 0 or u < 0:
        raise ValueError(f"bad card string {s!r}")
    return r * 4 + u


def card_to_str(c: int) -> str:
    c = int(c)
    if not 0 <= c < NUM_CARDS:
        raise ValueError(f"card out of range: {c}")
    return RANK_CHARS[c >> 2] + SUIT_CHARS[c & 3]


def cards_from_str(s: str) -> list[int]:
    """Parse a run of cards with optional whitespace: ``"AsKd"`` or ``"As Kd"``."""
    s = "".join(s.split())
    if len(s) % 2:
        raise ValueError(f"bad card run {s!r}")
    return [card_from_str(s[i : i + 2]) for i in range(0, len(s), 2)]


def cards_to_str(cards: Iterable[int], sep: str = "") -> str:
    return sep.join(card_to_str(c) for c in cards)


def rank_of(c: int) -> int:
    return c >> 2


def suit_of(c: int) -> int:
    return c & 3


def check_distinct(cards: Sequence[int]) -> None:
    seen = 0
    for c in cards:
        c = int(c)
        if not 0 <= c < NUM_CARDS:
            raise ValueError(f"card out of range: {c}")
        bit = 1 << c
        if seen & bit:
            raise ValueError(f"duplicate card {card_to_str(c)}")
        seen |= bit
