"""ACPC-style line protocol messages.

One message per line (``\\n`` terminated, ASCII). Messages:

``VERSION:2.0.0:<name>``
    client -> server, first line after connecting.
``GAMEDEF:<num_players>:<small_blind>:<big_blind>:<ante>:<stack0,stack1,...>``
    server -> client, reply to VERSION.
``MATCHSTATE:<seat>:<hand>:<button>:<betting>:<cards>``
    server -> client when it is the client's turn to act. The client answers
    with the same line followed by ``:<action>``.
``ENDHAND:<seat>:<hand>:<button>:<betting>:<cards>:<payoff0,payoff1,...>``
    server -> client when a hand ends; cards shown at showdown are included.
``BYE``
    server -> client at the end of the match.

``<betting>`` lists actions per street separated by ``/`` (one segment per
street reached): ``f`` fold, ``c`` check or call, ``r<amount>`` raise to
``amount``. Unlike ACPC, the raise amount is the contract's *street* raise-to
(chips committed on the current street after the raise), not the total for
the hand. ``<cards>`` is the hole cards of every seat separated by ``|``
(empty when hidden) followed by ``/flop``, ``/turn``, ``/river`` for the
streets dealt, e.g. ``AsKd|/2c3d4h/5s``. ``<action>`` uses the same tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import Any

import numpy as np

from ..engine_select import get_engine
from ..reference.cards import cards_from_str, cards_to_str

VERSION = "2.0.0"
BOARD_SIZES = (0, 3, 4, 5)


class ProtocolError(ValueError):
    pass


# -- actions ----------------------------------------------------------------


def encode_action(action: Any) -> str:
    kind = int(action.kind)
    if kind == 0:
        return "f"
    if kind == 1:
        return "c"
    if kind == 2:
        return f"r{int(action.amount)}"
    raise ProtocolError(f"unknown action kind {kind}")


def decode_action(token: str, engine: ModuleType | None = None) -> Any:
    engine = engine or get_engine()
    token = token.strip()
    if token == "f":
        return engine.Action.fold()
    if token in ("c", "k"):
        return engine.Action.check_call()
    if token.startswith("r") and token[1:].isdigit():
        return engine.Action.raise_to(int(token[1:]))
    raise ProtocolError(f"bad action token {token!r}")


# -- game definition ----------------------------------------------------------


def encode_hello(name: str) -> str:
    return f"VERSION:{VERSION}:{name.replace(':', '_')}"


def decode_hello(line: str) -> str:
    parts = line.strip().split(":", 2)
    if len(parts) < 2 or parts[0] != "VERSION":
        raise ProtocolError(f"expected VERSION line, got {line!r}")
    if parts[1].split(".")[0] != VERSION.split(".")[0]:
        raise ProtocolError(f"unsupported protocol version {parts[1]}")
    return parts[2] if len(parts) > 2 and parts[2] else "remote"


def encode_gamedef(config: Any) -> str:
    stacks = ",".join(str(int(s)) for s in config.stacks)
    return (
        f"GAMEDEF:{config.num_players}:{config.small_blind}:{config.big_blind}:"
        f"{config.ante}:{stacks}"
    )


def decode_gamedef(line: str, engine: ModuleType | None = None) -> Any:
    engine = engine or get_engine()
    parts = line.strip().split(":")
    if len(parts) != 6 or parts[0] != "GAMEDEF":
        raise ProtocolError(f"bad GAMEDEF line {line!r}")
    return engine.GameConfig(
        num_players=int(parts[1]),
        stacks=[int(s) for s in parts[5].split(",")],
        small_blind=int(parts[2]),
        big_blind=int(parts[3]),
        ante=int(parts[4]),
    )


# -- match states ---------------------------------------------------------------


@dataclass
class MatchState:
    seat: int
    hand_no: int
    button: int
    betting: list[list[str]]  # per street, action tokens
    holes: list[list[int]]  # per seat; [] when hidden
    board: list[int]
    payoffs: list[int] | None = None
    ended: bool = False
    raw: str = field(default="", compare=False)

    @property
    def num_players(self) -> int:
        return len(self.holes)

    @property
    def actions(self) -> list[str]:
        return [t for street in self.betting for t in street]


def _betting_string(state: Any) -> str:
    streets: list[list[str]] = [[] for _ in range(state.street + 1)]
    for street, _player, action in state.history:
        while len(streets) <= street:
            streets.append([])
        streets[street].append(encode_action(action))
    return "/".join("".join(s) for s in streets)


def _cards_string(state: Any) -> str:
    holes = "|".join(cards_to_str(state.hole_cards(p)) for p in range(state.num_players))
    board = list(state.board)
    parts = [holes]
    for s in (1, 2, 3):
        if len(board) >= BOARD_SIZES[s]:
            parts.append(cards_to_str(board[BOARD_SIZES[s - 1] : BOARD_SIZES[s]]))
    return "/".join(parts)


def encode_matchstate(state: Any, seat: int, hand_no: int) -> str:
    """Encode what ``seat`` may see. ``state`` should be a view already masked
    for ``seat`` (e.g. a ``MaskedState``); hidden hole cards encode as empty."""
    return (
        f"MATCHSTATE:{seat}:{hand_no}:{state.button}:{_betting_string(state)}:"
        f"{_cards_string(state)}"
    )


def encode_endhand(state: Any, seat: int, hand_no: int) -> str:
    body = encode_matchstate(state, seat, hand_no)[len("MATCHSTATE:") :]
    payoffs = ",".join(str(int(p)) for p in state.payoffs())
    return f"ENDHAND:{body}:{payoffs}"


def _parse_betting(s: str) -> list[list[str]]:
    streets = []
    for seg in s.split("/"):
        toks: list[str] = []
        i = 0
        while i < len(seg):
            ch = seg[i]
            if ch in "fck":
                toks.append("c" if ch == "k" else ch)
                i += 1
            elif ch == "r":
                j = i + 1
                while j < len(seg) and seg[j].isdigit():
                    j += 1
                if j == i + 1:
                    raise ProtocolError(f"raise without amount in {s!r}")
                toks.append(seg[i:j])
                i = j
            else:
                raise ProtocolError(f"bad betting char {ch!r} in {s!r}")
        streets.append(toks)
    return streets


def _parse_cards(s: str) -> tuple[list[list[int]], list[int]]:
    parts = s.split("/")
    holes = [cards_from_str(h) for h in parts[0].split("|")]
    board: list[int] = []
    for p in parts[1:]:
        board.extend(cards_from_str(p))
    return holes, board


def decode_matchstate(line: str) -> MatchState:
    line = line.strip()
    parts = line.split(":")
    if parts[0] == "MATCHSTATE" and len(parts) in (6, 7):
        ended, payoffs = False, None
    elif parts[0] == "ENDHAND" and len(parts) == 7:
        ended, payoffs = True, [int(x) for x in parts[6].split(",")]
    else:
        raise ProtocolError(f"bad match state {line!r}")
    try:
        seat, hand_no, button = int(parts[1]), int(parts[2]), int(parts[3])
        betting = _parse_betting(parts[4])
        holes, board = _parse_cards(parts[5])
    except ValueError as e:
        raise ProtocolError(f"bad match state {line!r}: {e}") from e
    return MatchState(seat, hand_no, button, betting, holes, board, payoffs, ended, line)


def split_response(sent: str, reply: str) -> str:
    """Return the action token of a client reply to ``sent``."""
    reply = reply.strip()
    if not reply.startswith(sent + ":"):
        raise ProtocolError(f"reply does not echo the match state: {reply!r}")
    return reply[len(sent) + 1 :]


def rebuild_state(
    ms: MatchState,
    config: Any,
    engine: ModuleType | None = None,
    rng: np.random.Generator | None = None,
) -> Any:
    """Reconstruct a ``GameState`` consistent with a match state. Cards the
    receiver cannot see are filled in at random from the unseen cards."""
    engine = engine or get_engine()
    rng = rng if rng is not None else np.random.default_rng()
    n = ms.num_players
    if n != config.num_players:
        raise ProtocolError("match state seat count does not match the game definition")
    deck: list[int | None] = [None] * 52
    for p, cards in enumerate(ms.holes):
        if cards:
            if len(cards) != 2:
                raise ProtocolError(f"seat {p} must have 0 or 2 hole cards")
            deck[2 * p], deck[2 * p + 1] = cards
    for j, c in enumerate(ms.board):
        deck[2 * n + j] = c
    known = {c for c in deck if c is not None}
    if len(known) != sum(c is not None for c in deck):
        raise ProtocolError("duplicate cards in match state")
    fill = iter(int(c) for c in rng.permutation([c for c in range(52) if c not in known]))
    full = [c if c is not None else next(fill) for c in deck]
    state = engine.GameState.new_hand(config, ms.button, full)
    for tok in ms.actions:
        try:
            state.apply(decode_action(tok, engine))
        except ValueError as e:
            raise ProtocolError(f"illegal action {tok!r} in {ms.raw!r}: {e}") from e
    return state


def visible_seats(ms: MatchState) -> list[int]:
    return [p for p, cards in enumerate(ms.holes) if cards]
