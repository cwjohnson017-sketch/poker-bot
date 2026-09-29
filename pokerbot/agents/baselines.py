"""Rule-based baseline agents."""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseAgent, clamp_raise, pot_raise_to


class AlwaysCallAgent(BaseAgent):
    """Checks or calls every decision; never folds, never raises."""

    name = "always_call"

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        return self.check_call()


class AlwaysRaiseAgent(BaseAgent):
    """Raises the pot every decision (all-in when the pot raise exceeds the
    stack); calls when no raise is legal."""

    name = "always_raise"

    def __init__(self, fraction: float = 1.0, name: str | None = None) -> None:
        super().__init__(name)
        self.fraction = fraction

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        amt = clamp_raise(legal, pot_raise_to(state, seat, self.fraction))
        if amt is None:
            return self.check_call()
        return self.raise_to(amt)


class RandomAgent(BaseAgent):
    """Picks uniformly among fold / check-call / raise (whichever are legal),
    with raise sizes uniform over the legal range. Never folds when it can
    check."""

    name = "random"

    def __init__(
        self,
        fold_weight: float = 1.0,
        call_weight: float = 1.0,
        raise_weight: float = 1.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.weights = (fold_weight, call_weight, raise_weight)

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        w = np.array(
            [
                self.weights[0] if legal.can_fold else 0.0,
                self.weights[1],
                self.weights[2] if legal.min_raise_to > 0 else 0.0,
            ]
        )
        choice = int(rng.choice(3, p=w / w.sum()))
        if choice == 0:
            return self.fold()
        if choice == 1:
            return self.check_call()
        lo, hi = legal.min_raise_to, legal.max_raise_to
        if hi <= lo:
            return self.raise_to(hi)
        return self.raise_to(int(rng.integers(lo, hi + 1)))
