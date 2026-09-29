"""Plain-text hand-history export.

One block per hand, blank line between hands::

    # hand 7 | button 1 | blinds 50/100 ante 0 | engine reference
    seat 0 always_call 20000 [Ah Kd]
    seat 1 equity 20000 [7c 7h] button
    preflop: 1:r300 0:c
    flop [2c 3d 4h]: 0:k 1:r450 0:c
    turn [2c 3d 4h 5s]: 0:k 1:k
    river [2c 3d 4h 5s 6d]: 0:k 1:k
    showdown: 0 1
    result: 0:-750 1:+750

Action tokens: ``f`` fold, ``k`` check, ``c`` call, ``r<amount>`` raise to
``amount`` chips on the street (the contract's raise-to).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TextIO

from ..reference.cards import cards_to_str

STREET_NAMES = ("preflop", "flop", "turn", "river")


def _replay_deck(state: Any) -> list[int]:
    n = state.num_players
    deck: list[int] = []
    for p in range(n):
        deck.extend(state.hole_cards(p))
    deck.extend(state.board)
    used = set(deck)
    deck.extend(c for c in range(52) if c not in used)
    return deck


def action_tokens(state: Any, config: Any) -> list[list[str]]:
    """Per-street lists of ``"<seat>:<token>"`` for a (fully visible) state."""
    replay = type(state).new_hand(config, state.button, _replay_deck(state))
    streets: list[list[str]] = [[] for _ in range(4)]
    for street, player, action in state.history:
        if action.kind == 0:
            tok = "f"
        elif action.kind == 1:
            tok = "k" if replay.legal_actions().can_check else "c"
        else:
            tok = f"r{action.amount}"
        streets[street].append(f"{player}:{tok}")
        replay.apply(action)
    return streets


def format_hand(
    state: Any,
    config: Any,
    hand_no: int = 0,
    names: Sequence[str] | None = None,
    engine: str = "",
) -> str:
    n = state.num_players
    names = list(names) if names else [f"p{i}" for i in range(n)]
    header = (
        f"# hand {hand_no} | button {state.button} | blinds {config.small_blind}/"
        f"{config.big_blind} ante {config.ante}"
    )
    if engine:
        header += f" | engine {engine}"
    lines = [header]
    for p in range(n):
        tag = " button" if p == state.button else ""
        cards = cards_to_str(state.hole_cards(p), " ")
        lines.append(f"seat {p} {names[p]} {config.stacks[p]} [{cards}]{tag}")
    board = list(state.board)
    streets = action_tokens(state, config)
    board_sizes = (0, 3, 4, 5)
    for s in range(4):
        if s and not streets[s]:
            continue
        shown = f" [{cards_to_str(board[: board_sizes[s]], ' ')}]" if s else ""
        lines.append(f"{STREET_NAMES[s]}{shown}: {' '.join(streets[s])}".rstrip())
    if len(board) > 0:
        lines.append(f"board: {cards_to_str(board, ' ')}")
    folded = state.folded
    live = [p for p in range(n) if not folded[p]]
    if len(live) > 1:
        lines.append("showdown: " + " ".join(str(p) for p in live))
    pay = state.payoffs()
    lines.append("result: " + " ".join(f"{p}:{pay[p]:+d}" for p in range(n)))
    return "\n".join(lines) + "\n"


class HandHistoryWriter:
    def __init__(self, fh: TextIO, config: Any, names: Sequence[str], engine: str = "") -> None:
        self.fh = fh
        self.config = config
        self.names = list(names)
        self.engine = engine

    def write(self, state: Any, hand_no: int, names: Sequence[str] | None = None) -> None:
        self.fh.write(
            format_hand(state, self.config, hand_no, names or self.names, self.engine) + "\n"
        )
