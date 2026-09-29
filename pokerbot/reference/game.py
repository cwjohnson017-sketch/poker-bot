"""Pure-Python reference implementation of the NLHE rules contract.

See ``docs/INTERFACES.md``. Rules decisions this implementation makes where
the contract is silent (the Rust engine must make the same ones):

These are the same choices the Rust engine documents in ``engine/README.md``
and the two engines are cross-checked on random play:

* Antes are posted by every seat before the blinds, count towards ``pot`` but
  not towards ``street_bets``. A blind or ante larger than the stack is posted
  for the whole stack and the seat is all-in.
* Short big blind: the preflop bet level is always the full big blind, even
  when the big blind was all-in for less; the excess comes back through the
  side-pot logic.
* ``min_raise_to = current_bet + last_raise`` where ``last_raise`` starts each
  street at the big blind and becomes the size of every full raise. An all-in
  raise for less does not change it.
* Re-opening (TDA rule): a player who has not acted on this street may always
  raise; a player who has acted may raise again only if the bet level rose by
  at least ``last_raise`` since their last action. Several short all-ins that
  add up to a full raise therefore re-open action.
* A player may not raise when no other player could respond (every other
  live player is all-in).
* A player who is the only one able to act only acts when facing a bet. When
  a round closes with fewer than two players able to act, the rest of the
  board is dealt in the same call, the hand is terminal, ``street == 3`` and
  ``street_bets`` are zero. A hand ending by folds keeps its street.
* Side pots are cut at the contribution levels of live players; chips folded
  players put in above the largest live contribution join the top pot. Within
  each pot odd chips go one at a time to the tied winners in clockwise order
  starting with the seat after the button.
* ``LegalActions.min_raise_to`` is the nominal minimum raise-to whenever any
  raise is legal, even if it exceeds ``max_raise_to`` (then only the all-in
  ``max_raise_to`` is legal). Folds and check/calls must carry amount 0.

Keys (compact bytes, identical to the Rust engine):

``public_key()`` = ``[button, len(board)] + board`` then for each action
``street << 4 | kind`` followed, for raises, by the raise-to amount as 4
little-endian bytes. Blinds, antes and stacks come from the config and are
not encoded.

``infoset_key(p)`` = ``public_key() + [p, low hole card, high hole card]``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .evaluator import evaluate7

FOLD, CHECK_CALL, RAISE = 0, 1, 2
_KIND_NAMES = {FOLD: "fold", CHECK_CALL: "check_call", RAISE: "raise"}


@dataclass
class GameConfig:
    num_players: int = 2
    stacks: list[int] = field(default_factory=list)
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0

    def __post_init__(self) -> None:
        if not 2 <= self.num_players <= 9:
            raise ValueError("num_players must be in 2..9")
        if not self.stacks:
            self.stacks = [200 * self.big_blind] * self.num_players
        self.stacks = [int(s) for s in self.stacks]
        if len(self.stacks) != self.num_players:
            raise ValueError("len(stacks) must equal num_players")
        if any(s <= 0 for s in self.stacks):
            raise ValueError("stacks must be positive")
        if self.big_blind <= 0 or not 0 <= self.small_blind <= self.big_blind:
            raise ValueError("need big_blind > 0 and 0 <= small_blind <= big_blind")
        if self.ante < 0:
            raise ValueError("ante must be non-negative")
        if sum(self.stacks) > 2**31 - 1:
            raise ValueError("total chips must fit in a signed 32-bit integer")


@dataclass(frozen=True, slots=True)
class Action:
    kind: int
    amount: int = 0

    @staticmethod
    def fold() -> Action:
        return Action(FOLD, 0)

    @staticmethod
    def check_call() -> Action:
        return Action(CHECK_CALL, 0)

    @staticmethod
    def raise_to(amount: int) -> Action:
        return Action(RAISE, int(amount))

    def __repr__(self) -> str:
        if self.kind == RAISE:
            return f"Action.raise_to({self.amount})"
        return f"Action.{_KIND_NAMES.get(self.kind, '?')}()"


@dataclass(frozen=True, slots=True)
class LegalActions:
    can_fold: bool
    can_check: bool
    call_amount: int
    min_raise_to: int
    max_raise_to: int

    @property
    def can_raise(self) -> bool:
        return self.min_raise_to > 0

    def is_legal(self, action: Action) -> bool:
        if action.kind == FOLD:
            return self.can_fold
        if action.kind == CHECK_CALL:
            return True
        if action.kind == RAISE:
            if self.min_raise_to <= 0:
                return False
            a = action.amount
            return self.min_raise_to <= a <= self.max_raise_to or a == self.max_raise_to
        return False


class GameState:
    __slots__ = (
        "_config",
        "_n",
        "_button",
        "_deck",
        "_holes",
        "_board",
        "_street",
        "_stacks",
        "_street_bets",
        "_contrib",
        "_folded",
        "_all_in",
        "_acted",
        "_acted_level",
        "_last_raise",
        "_current_bet",
        "_current",
        "_terminal",
        "_history",
        "_payoffs",
    )

    # -- construction -------------------------------------------------------

    @staticmethod
    def new_hand(config: GameConfig, button: int, deck: Sequence[int]) -> GameState:
        n = config.num_players
        if not 0 <= button < n:
            raise ValueError("button out of range")
        deck = tuple(int(c) for c in deck)
        if len(deck) < 2 * n + 5:
            raise ValueError(f"deck needs at least {2 * n + 5} cards")
        if len(set(deck)) != len(deck) or any(not 0 <= c < 52 for c in deck):
            raise ValueError("deck must contain distinct cards in 0..52")

        s = GameState.__new__(GameState)
        s._config = config
        s._n = n
        s._button = button
        s._deck = deck
        s._holes = [[deck[2 * i], deck[2 * i + 1]] for i in range(n)]
        s._board = []
        s._street = 0
        s._stacks = list(config.stacks)
        s._street_bets = [0] * n
        s._contrib = [0] * n
        s._folded = [False] * n
        s._all_in = [False] * n
        s._acted = [False] * n
        s._acted_level = [0] * n
        s._last_raise = config.big_blind
        s._current_bet = config.big_blind
        s._current = -1
        s._terminal = False
        s._history = []
        s._payoffs = None

        if config.ante:
            for i in range(n):
                s._post(i, config.ante, street=False)
        if n == 2:
            sb, bb = button, (button + 1) % n
        else:
            sb, bb = (button + 1) % n, (button + 2) % n
        s._post(sb, config.small_blind)
        s._post(bb, config.big_blind)

        nxt = s._next_to_act(bb)
        if nxt >= 0:
            s._current = nxt
        else:
            s._close_round()
        return s

    def _post(self, i: int, amount: int, street: bool = True) -> None:
        a = min(amount, self._stacks[i])
        self._stacks[i] -= a
        self._contrib[i] += a
        if street:
            self._street_bets[i] += a
        if self._stacks[i] == 0:
            self._all_in[i] = True

    # -- read-only properties ------------------------------------------------

    @property
    def config(self) -> GameConfig:
        return self._config

    @property
    def num_players(self) -> int:
        return self._n

    @property
    def button(self) -> int:
        return self._button

    @property
    def street(self) -> int:
        return self._street

    @property
    def board(self) -> list[int]:
        return list(self._board)

    def hole_cards(self, player: int) -> list[int]:
        return list(self._holes[player])

    @property
    def stacks(self) -> list[int]:
        return list(self._stacks)

    @property
    def street_bets(self) -> list[int]:
        return list(self._street_bets)

    @property
    def contributions(self) -> list[int]:
        """Chips committed by each seat over the whole hand (extension)."""
        return list(self._contrib)

    @property
    def pot(self) -> int:
        return sum(self._contrib)

    @property
    def current_player(self) -> int:
        return self._current

    @property
    def is_terminal(self) -> bool:
        return self._terminal

    @property
    def folded(self) -> list[bool]:
        return list(self._folded)

    @property
    def all_in(self) -> list[bool]:
        return list(self._all_in)

    @property
    def history(self) -> list[tuple[int, int, Action]]:
        return list(self._history)

    # -- helpers ---------------------------------------------------------------

    def _can_act(self, i: int) -> bool:
        return not self._folded[i] and not self._all_in[i]

    def _num_can_act(self) -> int:
        return sum(1 for i in range(self._n) if self._can_act(i))

    def _needs_action(self, i: int, num_can_act: int) -> bool:
        if not self._can_act(i):
            return False
        if self._street_bets[i] < self._current_bet:
            return True
        return not self._acted[i] and num_can_act >= 2

    def _next_to_act(self, start: int) -> int:
        k = self._num_can_act()
        for d in range(1, self._n + 1):
            i = (start + d) % self._n
            if self._needs_action(i, k):
                return i
        return -1

    def _close_round(self) -> None:
        """Betting round is over: deal the next street, run out, or finish."""
        n = self._n
        while True:
            if self._num_can_act() <= 1:
                if self._street < 3:
                    self._deal_to(3)
                    self._street_bets = [0] * n
                self._finish()
                return
            if self._street == 3:
                self._finish()
                return
            self._deal_to(self._street + 1)
            self._street_bets = [0] * n
            self._acted = [False] * n
            self._acted_level = [0] * n
            self._current_bet = 0
            self._last_raise = self._config.big_blind
            nxt = self._next_to_act(self._button)
            if nxt >= 0:
                self._current = nxt
                return

    def _deal_to(self, street: int) -> None:
        base = 2 * self._n
        target = {0: 0, 1: 3, 2: 4, 3: 5}[street]
        self._board = list(self._deck[base : base + target])
        self._street = street

    def _finish(self) -> None:
        self._terminal = True
        self._current = -1

    # -- actions ---------------------------------------------------------------

    def legal_actions(self) -> LegalActions:
        if self._terminal:
            return LegalActions(False, False, 0, 0, 0)
        p = self._current
        to_call = self._current_bet - self._street_bets[p]
        can_check = to_call <= 0
        call_amount = 0 if can_check else min(to_call, self._stacks[p])
        max_to = self._street_bets[p] + self._stacks[p]
        others = any(i != p and self._can_act(i) for i in range(self._n))
        reopened = (
            not self._acted[p] or self._current_bet - self._acted_level[p] >= self._last_raise
        )
        raise_ok = max_to > self._current_bet and others and reopened
        if raise_ok:
            min_to = self._current_bet + self._last_raise
            return LegalActions(not can_check, can_check, call_amount, min_to, max_to)
        return LegalActions(not can_check, can_check, call_amount, 0, 0)

    def apply(self, action: Action) -> None:
        if self._terminal:
            raise ValueError("hand is over")
        kind = int(action.kind)
        amount = int(getattr(action, "amount", 0))
        if kind in (FOLD, CHECK_CALL) and amount != 0:
            raise ValueError("fold and check/call take no amount")
        legal = self.legal_actions()
        p = self._current
        if kind == FOLD:
            if not legal.can_fold:
                raise ValueError("cannot fold when checking is possible")
            self._folded[p] = True
            action = Action.fold()
        elif kind == CHECK_CALL:
            if legal.call_amount:
                self._post(p, legal.call_amount)
            action = Action.check_call()
        elif kind == RAISE:
            if not legal.can_raise:
                raise ValueError("raising is not legal here")
            if not (
                legal.min_raise_to <= amount <= legal.max_raise_to or amount == legal.max_raise_to
            ):
                raise ValueError(
                    f"raise to {amount} outside [{legal.min_raise_to}, {legal.max_raise_to}]"
                )
            increment = amount - self._current_bet
            if increment >= self._last_raise:
                self._last_raise = increment
            self._post(p, amount - self._street_bets[p])
            self._current_bet = amount
            action = Action.raise_to(amount)
        else:
            raise ValueError(f"unknown action kind {kind}")

        self._acted[p] = True
        self._acted_level[p] = self._current_bet
        self._history.append((self._street, p, action))

        live = sum(1 for f in self._folded if not f)
        if live == 1:
            self._finish()
            return
        nxt = self._next_to_act(p)
        if nxt >= 0:
            self._current = nxt
        else:
            self._close_round()

    def child(self, action: Action) -> GameState:
        c = self.clone()
        c.apply(action)
        return c

    def clone(self) -> GameState:
        c = GameState.__new__(GameState)
        c._config = self._config
        c._n = self._n
        c._button = self._button
        c._deck = self._deck
        c._holes = self._holes  # never mutated
        c._board = list(self._board)
        c._street = self._street
        c._stacks = list(self._stacks)
        c._street_bets = list(self._street_bets)
        c._contrib = list(self._contrib)
        c._folded = list(self._folded)
        c._all_in = list(self._all_in)
        c._acted = list(self._acted)
        c._acted_level = list(self._acted_level)
        c._last_raise = self._last_raise
        c._current_bet = self._current_bet
        c._current = self._current
        c._terminal = self._terminal
        c._history = list(self._history)
        c._payoffs = self._payoffs
        return c

    # -- outcome -------------------------------------------------------------

    def payoffs(self) -> list[int]:
        if not self._terminal:
            raise ValueError("payoffs are only defined for terminal states")
        if self._payoffs is None:
            self._payoffs = self._settle()
        return list(self._payoffs)

    def _settle(self) -> list[int]:
        n = self._n
        contrib = self._contrib
        won = [0] * n
        live = [i for i in range(n) if not self._folded[i]]
        if len(live) == 1:
            won[live[0]] = sum(contrib)
            return [won[i] - contrib[i] for i in range(n)]

        ranks = {i: evaluate7(self._holes[i] + self._board) for i in live}
        levels = sorted({contrib[i] for i in live})
        top_all = max(contrib)
        prev = 0
        for idx, level in enumerate(levels):
            top = level if idx < len(levels) - 1 else max(level, top_all)
            amount = sum(min(c, top) - min(c, prev) for c in contrib)
            prev = top
            if amount == 0:
                continue
            eligible = [i for i in live if contrib[i] >= level]
            best = max(ranks[i] for i in eligible)
            winners = [i for i in eligible if ranks[i] == best]
            share, rem = divmod(amount, len(winners))
            for w in winners:
                won[w] += share
            winners.sort(key=lambda i: (i - self._button - 1) % n)
            for w in winners[:rem]:
                won[w] += 1
        return [won[i] - contrib[i] for i in range(n)]

    # -- keys -----------------------------------------------------------------

    def public_key(self) -> bytes:
        out = bytearray((self._button, len(self._board)))
        out.extend(self._board)
        for street, _player, a in self._history:
            out.append((street << 4) | a.kind)
            if a.kind == RAISE:
                out.extend(int(a.amount).to_bytes(4, "little"))
        return bytes(out)

    def infoset_key(self, player: int) -> bytes:
        if not 0 <= player < self._n:
            raise ValueError(f"no seat {player}")
        return self.public_key() + bytes((player, *sorted(self._holes[player])))

    def __repr__(self) -> str:
        from .cards import cards_to_str

        return (
            f"GameState(street={self._street}, board={cards_to_str(self._board)!r}, "
            f"pot={self.pot}, stacks={self._stacks}, bets={self._street_bets}, "
            f"current={self._current}, terminal={self._terminal})"
        )
