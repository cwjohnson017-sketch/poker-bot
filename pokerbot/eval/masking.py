"""Hole-card masking for agents.

``MaskedState`` wraps a live ``GameState`` for one seat. It exposes the public
part of the contract API and that seat's own hole cards; other seats'
``hole_cards`` return ``[]`` unless they were shown at showdown. It never
exposes the deck. ``apply`` is not available on the view; ``clone`` and
``child`` return a full, unmasked ``GameState`` in which every card the seat
cannot see (other players' hole cards and the undealt board) has been
re-sampled from the unseen cards, so agents can search without leaking
hidden information.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np


class HiddenInformationError(ValueError):
    pass


class MaskedState:
    __slots__ = ("__state", "__seat", "__config", "__rng", "__revealed", "__det")

    def __init__(
        self,
        state: Any,
        seat: int,
        config: Any,
        rng: np.random.Generator | None = None,
        revealed: Iterable[int] = (),
    ) -> None:
        self.__state = state
        self.__seat = seat
        self.__config = config
        self.__rng = rng
        self.__revealed = frozenset(revealed)
        self.__det = None

    # -- public information ---------------------------------------------------

    @property
    def seat(self) -> int:
        return self.__seat

    @property
    def config(self) -> Any:
        return self.__config

    @property
    def num_players(self) -> int:
        return self.__state.num_players

    @property
    def button(self) -> int:
        return self.__state.button

    @property
    def street(self) -> int:
        return self.__state.street

    @property
    def board(self) -> list[int]:
        return list(self.__state.board)

    @property
    def stacks(self) -> list[int]:
        return list(self.__state.stacks)

    @property
    def street_bets(self) -> list[int]:
        return list(self.__state.street_bets)

    @property
    def pot(self) -> int:
        return self.__state.pot

    @property
    def current_player(self) -> int:
        return self.__state.current_player

    @property
    def is_terminal(self) -> bool:
        return self.__state.is_terminal

    @property
    def folded(self) -> list[bool]:
        return list(self.__state.folded)

    @property
    def all_in(self) -> list[bool]:
        return list(self.__state.all_in)

    @property
    def history(self) -> list:
        return list(self.__state.history)

    def _visible(self, player: int) -> bool:
        return player == self.__seat or player in self.__revealed

    def hole_cards(self, player: int) -> list[int]:
        if self._visible(player):
            return list(self.__state.hole_cards(player))
        return []

    def legal_actions(self) -> Any:
        return self.__state.legal_actions()

    def payoffs(self) -> list[int]:
        return list(self.__state.payoffs())

    def public_key(self) -> bytes:
        return self.__state.public_key()

    def infoset_key(self, player: int) -> bytes:
        if not self._visible(player):
            raise HiddenInformationError(f"seat {self.__seat} cannot see seat {player}'s cards")
        return self.__state.infoset_key(player)

    # -- simulation -----------------------------------------------------------

    def apply(self, action: Any) -> None:
        raise TypeError("MaskedState is read-only; use clone() or child() to simulate")

    def clone(self) -> Any:
        return self._determinized().clone()

    def child(self, action: Any) -> Any:
        return self._determinized().child(action)

    def _determinized(self) -> Any:
        if self.__det is None:
            self.__det = determinize(
                self.__state,
                self.__config,
                visible=[p for p in range(self.num_players) if self._visible(p)],
                rng=self.__rng if self.__rng is not None else np.random.default_rng(),
            )
        return self.__det

    def __repr__(self) -> str:
        return (
            f"MaskedState(seat={self.__seat}, street={self.street}, board={self.board}, "
            f"pot={self.pot}, current={self.current_player})"
        )


def determinize(state: Any, config: Any, visible: Iterable[int], rng: np.random.Generator) -> Any:
    """Rebuild ``state`` from a deck that keeps the visible hole cards and the
    dealt board and re-samples every other card, then replay the history."""
    n = state.num_players
    deck: list[int | None] = [None] * 52
    known: set[int] = set()
    for p in visible:
        cards = list(state.hole_cards(p))
        deck[2 * p], deck[2 * p + 1] = cards
        known.update(cards)
    board = list(state.board)
    for j, c in enumerate(board):
        deck[2 * n + j] = c
        known.add(c)
    unseen = [c for c in range(52) if c not in known]
    fill = iter(int(c) for c in rng.permutation(unseen))
    full = [c if c is not None else next(fill) for c in deck]
    new = type(state).new_hand(config, state.button, full)
    for _street, _player, action in state.history:
        new.apply(action)
    return new
