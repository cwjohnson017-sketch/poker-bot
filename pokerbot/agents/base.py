"""Agent protocol and small helpers shared by agent implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np

from ..engine_select import get_engine

FOLD, CHECK_CALL, RAISE = 0, 1, 2


@runtime_checkable
class Agent(Protocol):
    name: str

    def new_hand(self, seat: int, config: Any) -> None: ...

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any: ...

    def observe_end(self, state: Any) -> None: ...


class BaseAgent:
    """Convenience base class with no-op hooks. Subclasses implement ``act``."""

    name = "base"

    def __init__(self, name: str | None = None) -> None:
        if name is not None:
            self.name = name
        self.seat = -1
        self.config = None
        self._engine = get_engine()

    def new_hand(self, seat: int, config: Any) -> None:
        self.seat = seat
        self.config = config

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:  # pragma: no cover
        raise NotImplementedError

    def observe_end(self, state: Any) -> None:
        pass

    # Action constructors from the active engine. The match runner converts
    # actions between engines, so agents may use either.
    def fold(self) -> Any:
        return self._engine.Action.fold()

    def check_call(self) -> Any:
        return self._engine.Action.check_call()

    def raise_to(self, amount: int) -> Any:
        return self._engine.Action.raise_to(int(amount))


def current_bet(state: Any) -> int:
    return max(state.street_bets)


def pot_raise_to(state: Any, seat: int, fraction: float = 1.0) -> int:
    """Raise-to amount for a raise of ``fraction`` times the pot after calling."""
    bets = state.street_bets
    cur = max(bets)
    to_call = cur - bets[seat]
    return int(cur + fraction * (state.pot + to_call))


def clamp_raise(legal: Any, raise_to: int) -> int | None:
    """Clamp a desired raise-to into the legal range; ``None`` if no raise is legal."""
    if legal.min_raise_to <= 0:
        return None
    if legal.max_raise_to <= legal.min_raise_to:
        return legal.max_raise_to
    return max(legal.min_raise_to, min(legal.max_raise_to, int(raise_to)))
