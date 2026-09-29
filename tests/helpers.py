"""Shared test helpers."""

from __future__ import annotations

from collections.abc import Sequence

from pokerbot.reference import cards_from_str


def make_deck(holes: Sequence[str], board: str = "") -> list[int]:
    """Deck dealing ``holes[i]`` to seat ``i`` and ``board`` as flop/turn/river;
    the remaining cards follow in increasing order."""
    deck: list[int] = []
    for h in holes:
        cards = cards_from_str(h)
        assert len(cards) == 2
        deck.extend(cards)
    deck.extend(cards_from_str(board))
    used = set(deck)
    assert len(used) == len(deck), "duplicate card in scenario"
    return deck + [c for c in range(52) if c not in used]


def naive_rank5(cards: Sequence[int]) -> tuple:
    """Textbook 5-card ranking as a comparable tuple (category, tiebreaks...)."""
    ranks = sorted((c >> 2 for c in cards), reverse=True)
    suits = [c & 3 for c in cards]
    counts: dict[int, int] = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len(set(suits)) == 1
    uniq = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            straight_high = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            straight_high = 3
    if straight_high is not None and flush:
        return (8, straight_high)
    if groups[0][1] == 4:
        return (7, groups[0][0], groups[1][0])
    if groups[0][1] == 3 and groups[1][1] == 2:
        return (6, groups[0][0], groups[1][0])
    if flush:
        return (5, *ranks)
    if straight_high is not None:
        return (4, straight_high)
    if groups[0][1] == 3:
        return (3, groups[0][0], *[g[0] for g in groups[1:]])
    if groups[0][1] == 2 and groups[1][1] == 2:
        return (2, groups[0][0], groups[1][0], groups[2][0])
    if groups[0][1] == 2:
        return (1, groups[0][0], *[g[0] for g in groups[1:]])
    return (0, *ranks)


def naive_best(cards: Sequence[int]) -> tuple:
    from itertools import combinations

    return max(naive_rank5(c) for c in combinations(cards, 5))
