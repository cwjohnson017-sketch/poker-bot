"""Heads-up hands in PokerStars' hand-history text format, for tracking software.

One hand per call of :func:`format_pokerstars`, written from one player's
point of view (``hero``): the other player's cards appear only when they were
shown down. Play-money style header (``Hold'em No Limit (50/100)``), no rake.
Timestamps are US Eastern time (computed from UTC with the US daylight-saving
rule, since Windows Python has no time-zone database by default).

The layout follows PokerStars' own files::

    PokerStars Hand #2026102114050900001:  Hold'em No Limit (50/100) - 2026/10/21 10:05:09 ET
    Table 'pokerbot' 2-max (Play Money) Seat #1 is the button
    Seat 1: You (10000 in chips)
    Seat 2: Bot (10000 in chips)
    You: posts small blind 50
    Bot: posts big blind 100
    *** HOLE CARDS ***
    Dealt to You [Ah Kd]
    You: raises 150 to 250
    Bot: folds
    Uncalled bet (150) returned to You
    You collected 200 from pot
    *** SUMMARY ***
    Total pot 200 | Rake 0
    Seat 1: You (button) (small blind) collected (200)
    Seat 2: Bot (big blind) folded before Flop

Whether a given tracker imports these files has not been checked.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from ..eval.match import showdown_seats
from ..reference.cards import card_to_str
from ..reference.evaluator import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    QUADS,
    STRAIGHT,
    STRAIGHT_FLUSH,
    TRIPS,
    TWO_PAIR,
    evaluate,
    hand_category,
)

RANK_NAMES = (
    "Deuce",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Jack",
    "Queen",
    "King",
    "Ace",
)
RANK_PLURALS = tuple(n + ("es" if n == "Six" else "s") for n in RANK_NAMES)
STREET_TITLES = ("Flop", "Turn", "River")


def describe_hand(cards: Sequence[int]) -> str:
    """PokerStars' description of the best five of 5-7 cards, e.g. ``"a pair of
    Kings"``, ``"two pair, Aces and Fours"``, ``"a straight, Ace to Five"``."""
    rank = evaluate([int(c) for c in cards])
    cat = hand_category(rank)
    k1, k2 = (rank >> 16) & 15, (rank >> 12) & 15
    if cat == STRAIGHT_FLUSH:
        if k1 == 12:
            return "a Royal Flush"
        low = "Ace" if k1 == 3 else RANK_NAMES[k1 - 4]
        return f"a straight flush, {low} to {RANK_NAMES[k1]}"
    if cat == QUADS:
        return f"four of a kind, {RANK_PLURALS[k1]}"
    if cat == FULL_HOUSE:
        return f"a full house, {RANK_PLURALS[k1]} full of {RANK_PLURALS[k2]}"
    if cat == FLUSH:
        return f"a flush, {RANK_NAMES[k1]} high"
    if cat == STRAIGHT:
        low = "Ace" if k1 == 3 else RANK_NAMES[k1 - 4]
        return f"a straight, {low} to {RANK_NAMES[k1]}"
    if cat == TRIPS:
        return f"three of a kind, {RANK_PLURALS[k1]}"
    if cat == TWO_PAIR:
        return f"two pair, {RANK_PLURALS[k1]} and {RANK_PLURALS[k2]}"
    if cat == PAIR:
        return f"a pair of {RANK_PLURALS[k1]}"
    assert cat == HIGH_CARD
    return f"high card {RANK_NAMES[k1]}"


def us_eastern(when: datetime) -> datetime:
    """``when`` (aware) as naive US Eastern time: UTC-4 from the second Sunday of
    March 2:00 to the first Sunday of November 2:00 (local), UTC-5 otherwise."""
    utc = when.astimezone(UTC).replace(tzinfo=None)
    year = utc.year
    march = datetime(year, 3, 8)  # the second Sunday is in 8..14
    start = march + timedelta(days=(6 - march.weekday()) % 7, hours=2 + 5)  # 2:00 EST in UTC
    nov = datetime(year, 11, 1)
    end = nov + timedelta(days=(6 - nov.weekday()) % 7, hours=2 + 4)  # 2:00 EDT in UTC
    return utc - timedelta(hours=4 if start <= utc < end else 5)


def _cards(cards: Sequence[int]) -> str:
    return " ".join(card_to_str(c) for c in cards)


def _deck(state: Any) -> list[int]:
    deck: list[int] = []
    for p in range(state.num_players):
        deck.extend(int(c) for c in state.hole_cards(p))
    deck.extend(int(c) for c in state.board)
    used = set(deck)
    deck.extend(c for c in range(52) if c not in used)
    return deck


def format_pokerstars(
    state: Any,
    config: Any,
    hand_id: int,
    names: Sequence[str],
    hero: int,
    when: datetime | None = None,
    table: str = "pokerbot",
) -> str:
    """One finished heads-up hand (a terminal, unmasked ``GameState``) as
    PokerStars text from seat ``hero``'s point of view."""
    if state.num_players != 2:
        raise ValueError("PokerStars export is heads-up only")
    when = when or datetime.now(UTC)
    button = int(state.button)
    sb_seat, bb_seat = button, 1 - button
    start = [int(s) for s in config.stacks]
    lines = [
        f"PokerStars Hand #{hand_id}:  Hold'em No Limit ({config.small_blind}/"
        f"{config.big_blind}) - {us_eastern(when):%Y/%m/%d %H:%M:%S} ET",
        f"Table '{table}' 2-max (Play Money) Seat #{button + 1} is the button",
    ]
    lines += [f"Seat {p + 1}: {names[p]} ({start[p]} in chips)" for p in (0, 1)]

    replay = type(state).new_hand(config, button, _deck(state))
    contrib = [start[p] - int(replay.stacks[p]) for p in (0, 1)]
    lines.append(f"{names[sb_seat]}: posts small blind {contrib[sb_seat]}")
    lines.append(f"{names[bb_seat]}: posts big blind {contrib[bb_seat]}")
    lines.append("*** HOLE CARDS ***")
    lines.append(f"Dealt to {names[hero]} [{_cards(state.hole_cards(hero))}]")

    board = [int(c) for c in state.board]
    street = 0
    folded_on: dict[int, int] = {}

    def header(s: int) -> str:
        title = ("FLOP", "TURN", "RIVER")[s - 1]
        n = (3, 4, 5)[s - 1]
        if s == 1:
            return f"*** FLOP *** [{_cards(board[:3])}]"
        return f"*** {title} *** [{_cards(board[: n - 1])}] [{card_to_str(board[n - 1])}]"

    for st, p, action in state.history:
        st, p = int(st), int(p)
        while street < st:
            street += 1
            lines.append(header(street))
        bets = [int(b) for b in replay.street_bets]
        cur, mine = max(bets), bets[p]
        behind = int(replay.stacks[p])
        legal = replay.legal_actions()
        kind = int(action.kind)
        if kind == 0:
            added, text = 0, "folds"
            folded_on[p] = st
        elif kind == 1 and legal.can_check:
            added, text = 0, "checks"
        elif kind == 1:
            added = min(cur - mine, behind)
            text = f"calls {added}"
        else:
            to = int(action.amount)
            added = to - mine
            text = f"bets {to}" if cur == 0 else f"raises {to - cur} to {to}"
        if kind and added and added == behind:
            text += " and is all-in"
        contrib[p] += added
        lines.append(f"{names[p]}: {text}")
        replay.apply(action)

    payoffs = [int(x) for x in state.payoffs()]
    shown = showdown_seats(state)
    hi = max((0, 1), key=lambda q: contrib[q])
    uncalled = contrib[hi] - contrib[1 - hi]
    if uncalled > 0:
        lines.append(f"Uncalled bet ({uncalled}) returned to {names[hi]}")
    matched = min(contrib)
    for s in range(street + 1, 4):  # an all-in runs the board out
        if len(board) >= (3, 4, 5)[s - 1]:
            lines.append(header(s))
    if shown:
        lines.append("*** SHOW DOWN ***")
        for p in shown:
            hole = [int(c) for c in state.hole_cards(p)]
            lines.append(f"{names[p]}: shows [{_cards(hole)}] ({describe_hand(hole + board)})")
    collected = [max(0, payoffs[p] + matched) for p in (0, 1)]
    for p in (0, 1):
        if collected[p]:
            lines.append(f"{names[p]} collected {collected[p]} from pot")

    lines.append("*** SUMMARY ***")
    lines.append(f"Total pot {2 * matched} | Rake 0")
    if board:
        lines.append(f"Board [{_cards(board)}]")
    for p in (0, 1):
        role = "(button) (small blind)" if p == sb_seat else "(big blind)"
        if p in folded_on:
            st = folded_on[p]
            what = "folded before Flop" if st == 0 else f"folded on the {STREET_TITLES[st - 1]}"
        elif p in shown:
            hole = [int(c) for c in state.hole_cards(p)]
            desc = describe_hand(hole + board)
            if collected[p]:
                what = f"showed [{_cards(hole)}] and won ({collected[p]}) with {desc}"
            else:
                what = f"showed [{_cards(hole)}] and lost with {desc}"
        else:
            what = f"collected ({collected[p]})"
        lines.append(f"Seat {p + 1}: {names[p]} {role} {what}")
    return "\n".join(lines) + "\n"
