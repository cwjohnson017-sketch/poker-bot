"""Scalar (per ``GameState``) abstract actions, action mapping and state helpers.

These mirror the tensor code in :mod:`pokerbot.env.actions` exactly (same
integer formulas, same legality rules, same dedupe), but work on one scalar
engine ``GameState`` in plain Python, which is what the tree builder, the
rollouts and the agent need. ``tests/search/test_abstract.py`` checks the two
agree.

Also here:

* :func:`map_concrete` - pseudo-harmonic mapping (Ganzfried & Sandholm 2013)
  of a concrete action onto the abstract actions legal in a state, as a
  probability distribution over abstract indices;
* :class:`CardView` - a read-only view of a state with the board and one
  player's hole cards replaced, so a blueprint can be queried for any combo
  and any runout without re-dealing the engine state;
* :func:`make_state` - rebuild a scalar ``GameState`` from public
  information (button, board, action history) with dummy hole cards.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Any

from ..env.actions import ActionSpec

FOLD, CHECK_CALL, RAISE = 0, 1, 2
BOARD_LEN = (0, 3, 4, 5)


@dataclass(frozen=True)
class Option:
    """One legal abstract action in a state: its index and concrete action."""

    index: int
    kind: int  # FOLD / CHECK_CALL / RAISE
    amount: int  # raise-to for RAISE, else 0
    label: str  # "fold", "check_call", "raise", "raise_x", "allin"


def raises_this_street(state: Any) -> int:
    """Voluntary raises on the current street (blinds do not count)."""
    st = state.street
    return sum(1 for s, _p, a in state.history if s == st and int(a.kind) == RAISE)


def _milli(x: float) -> int:
    return int(round(float(x) * 1000))


def legal_options(
    state: Any, spec: ActionSpec, max_raises: int | None = None, street_actions=None
) -> list[Option]:
    """Legal abstract actions of the acting player, as in ``env.actions.legal_mask``.

    ``street_actions`` overrides ``spec.streets[state.street]`` (the tree
    builder uses it to drop bet sizes); ``max_raises`` overrides
    ``spec.max_raises``.
    """
    if state.is_terminal:
        return []
    street = state.street
    acts = spec.streets[street] if street_actions is None else street_actions
    cap = spec.max_raises if max_raises is None else max_raises
    p = state.current_player
    bets = list(state.street_bets)
    max_bet = max(bets)
    to_call = max_bet - bets[p]
    pot = int(state.pot)
    la = state.legal_actions()
    raise_ok = la.min_raise_to > 0
    lo, hi = int(la.min_raise_to), int(la.max_raise_to)
    can_raise = raise_ok and raises_this_street(state) < cap
    out: list[Option] = []
    seen: list[int] = []
    for i, a in enumerate(acts):
        name = a[0]
        if name == "fold":
            if to_call > 0:
                out.append(Option(i, FOLD, 0, name))
            continue
        if name == "check_call":
            out.append(Option(i, CHECK_CALL, 0, name))
            continue
        if not can_raise:
            continue
        if name == "allin":
            out.append(Option(i, RAISE, hi, name))
            continue
        m = _milli(a[1])
        if name == "raise":
            raw = max_bet + (m * (pot + to_call) + 500) // 1000
        else:  # raise_x
            raw = (m * max_bet + 500) // 1000
        target = min(max(raw, lo), hi)
        if target >= hi:
            continue  # would be all-in: left to the allin action
        if spec.dedupe and target in seen:
            continue
        seen.append(target)
        out.append(Option(i, RAISE, target, name))
    return out


def legal_mask(state: Any, spec: ActionSpec) -> list[bool]:
    mask = [False] * spec.num_actions
    for o in legal_options(state, spec):
        mask[o.index] = True
    return mask


def pot_fraction(state: Any, raise_to: int) -> float:
    """Size of a raise as a fraction of the pot after calling."""
    p = state.current_player
    bets = list(state.street_bets)
    max_bet = max(bets)
    to_call = max_bet - bets[p]
    return (int(raise_to) - max_bet) / max(1, int(state.pot) + to_call)


def map_concrete(
    state: Any, spec: ActionSpec, kind: int, amount: int = 0, options: list[Option] | None = None
) -> dict[int, float]:
    """Pseudo-harmonic mapping of a concrete action onto abstract indices.

    Fold and check/call map to their own index. A raise to ``amount`` equal
    to an abstract raise's amount maps to it; otherwise, with ``A < x < B``
    the neighbouring abstract sizes as pot fractions, it maps to ``A`` with
    probability ``(B - x)(1 + A) / ((B - A)(1 + x))`` and to ``B`` otherwise.
    Below the smallest (above the largest) abstract raise it maps to that raise.
    With no legal abstract raise it maps to check/call.
    """
    opts = legal_options(state, spec) if options is None else options
    kind = int(kind)
    if kind == FOLD:
        for o in opts:
            if o.kind == FOLD:
                return {o.index: 1.0}
        kind = CHECK_CALL
    if kind == CHECK_CALL:
        for o in opts:
            if o.kind == CHECK_CALL:
                return {o.index: 1.0}
        return {}
    raises = sorted((o for o in opts if o.kind == RAISE), key=lambda o: o.amount)
    if not raises:
        return map_concrete(state, spec, CHECK_CALL, 0, opts)
    amount = int(amount)
    for o in raises:
        if o.amount == amount:
            return {o.index: 1.0}
    below = [o for o in raises if o.amount < amount]
    above = [o for o in raises if o.amount > amount]
    if not below:
        return {above[0].index: 1.0}
    if not above:
        return {below[-1].index: 1.0}
    a_o, b_o = below[-1], above[0]
    fa = pot_fraction(state, a_o.amount)
    fb = pot_fraction(state, b_o.amount)
    fx = pot_fraction(state, amount)
    if fb <= fa:
        return {a_o.index: 1.0}
    pa = (fb - fx) * (1 + fa) / ((fb - fa) * (1 + fx))
    pa = min(1.0, max(0.0, pa))
    out = {a_o.index: pa}
    out[b_o.index] = out.get(b_o.index, 0.0) + 1.0 - pa
    return out


def action_key(action: Any) -> tuple[int, int]:
    kind = int(action.kind)
    return (kind, int(action.amount) if kind == RAISE else 0)


def to_action(engine: ModuleType, kind: int, amount: int = 0) -> Any:
    if kind == FOLD:
        return engine.Action.fold()
    if kind == CHECK_CALL:
        return engine.Action.check_call()
    return engine.Action.raise_to(int(amount))


def public_key_bytes(button: int, board: Sequence[int], history: Sequence) -> bytes:
    """Same byte layout as the engines' ``public_key()``."""
    out = bytearray([int(button), len(board)])
    out.extend(int(c) for c in board)
    for street, _player, a in history:
        kind = int(a.kind)
        out.append((int(street) << 4) | kind)
        if kind == RAISE:
            out.extend(int(a.amount).to_bytes(4, "little"))
    return bytes(out)


class CardView:
    """Read-only view of ``state`` with replaced cards.

    ``board`` (up to 5 cards) is shown truncated to the length the engine
    state has dealt, so a single engine state can stand for every runout.
    ``hole`` replaces ``player``'s hole cards; other players' cards are hidden.
    Everything else is delegated to the wrapped state.
    """

    __slots__ = ("_state", "_board", "_player", "_hole")

    def __init__(
        self,
        state: Any,
        board: Sequence[int] | None = None,
        player: int = -1,
        hole: Sequence[int] | None = None,
    ) -> None:
        self._state = state
        self._board = None if board is None else [int(c) for c in board]
        self._player = player
        self._hole = None if hole is None else [int(c) for c in hole]

    def with_hole(self, player: int, hole: Sequence[int]) -> CardView:
        return CardView(self._state, self._board, player, hole)

    @property
    def board(self) -> list[int]:
        real = list(self._state.board)
        if self._board is None:
            return real
        return self._board[: len(real)]

    def hole_cards(self, player: int) -> list[int]:
        if self._hole is not None:
            return list(self._hole) if player == self._player else []
        return list(self._state.hole_cards(player))

    def public_key(self) -> bytes:
        if self._board is None:
            return self._state.public_key()
        return public_key_bytes(self._state.button, self.board, self._state.history)

    def infoset_key(self, player: int) -> bytes:
        h = sorted(self.hole_cards(player))
        if len(h) != 2:
            raise ValueError(f"hole cards of seat {player} are not visible")
        return self.public_key() + bytes([player, h[0], h[1]])

    @property
    def state(self) -> Any:
        return self._state

    def __getattr__(self, name: str) -> Any:
        return getattr(self._state, name)


def contributions(state: Any, config: Any) -> list[int]:
    """Chips committed this hand per seat (antes included)."""
    c = getattr(state, "contributed", None)
    if c is None:
        c = getattr(state, "contributions", None)
    if c is not None:
        return [int(x) for x in c]
    return [int(s0) - int(s) for s0, s in zip(config.stacks, state.stacks, strict=True)]


def make_state(
    engine: ModuleType,
    config: Any,
    button: int,
    board: Sequence[int],
    history: Sequence,
    holes: Sequence[Sequence[int]] | None = None,
) -> Any:
    """A scalar ``GameState`` with the given board (up to 5 cards, later cards
    are dealt from it as the history needs) and the given action history.

    Hole cards are ``holes`` or the lowest cards not on ``board``. The rest of
    the deck is filled in increasing order.
    """
    board = [int(c) for c in board]
    used = set(board)
    if holes is None:
        free = [c for c in range(52) if c not in used]
        holes = [free[0:2], free[2:4]]
    head: list[int | None] = [*holes[0], *holes[1]]
    used.update(int(c) for h in holes for c in h)
    rest = iter(c for c in range(52) if c not in used)
    deck = [int(c) for c in head] + board
    deck += [next(rest) for _ in range(9 - len(deck))]
    deck += list(rest)
    state = engine.GameState.new_hand(config, int(button), deck)
    for _street, _player, a in history:
        state.apply(to_action(engine, int(a.kind), int(a.amount)))
    return state
