"""Monte Carlo equity and the equity-threshold baseline agent."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np

from ..engine_select import get_engine
from .base import BaseAgent, clamp_raise, pot_raise_to


def monte_carlo_equity(
    hole: Sequence[int],
    board: Sequence[int],
    num_opponents: int,
    samples: int,
    rng: np.random.Generator,
    evaluate_batch: Callable[[np.ndarray], np.ndarray] | None = None,
) -> float:
    """Equity (win + split share) of ``hole`` against ``num_opponents`` uniformly
    random hands, completing the board at random. ``samples`` Monte Carlo draws."""
    if evaluate_batch is None:
        evaluate_batch = get_engine().evaluate_batch
    num_opponents = max(1, int(num_opponents))
    known = set(int(c) for c in hole) | set(int(c) for c in board)
    unseen = np.array([c for c in range(52) if c not in known], dtype=np.uint8)
    n_board = 5 - len(board)
    need = 2 * num_opponents + n_board
    draws = rng.permuted(np.tile(unseen, (samples, 1)), axis=1)[:, :need]
    runout = draws[:, 2 * num_opponents :]
    full_board = np.concatenate(
        [np.tile(np.asarray(board, dtype=np.uint8), (samples, 1)), runout], axis=1
    )
    hero_cards = np.concatenate(
        [np.tile(np.asarray(hole, dtype=np.uint8), (samples, 1)), full_board], axis=1
    )
    hero = np.asarray(evaluate_batch(hero_cards), dtype=np.int64)
    opp = np.stack(
        [
            np.asarray(
                evaluate_batch(np.concatenate([draws[:, 2 * j : 2 * j + 2], full_board], axis=1)),
                dtype=np.int64,
            )
            for j in range(num_opponents)
        ],
        axis=1,
    )
    best_opp = opp.max(axis=1)
    ties = (opp == hero[:, None]).sum(axis=1)
    share = np.where(hero > best_opp, 1.0, np.where(hero == best_opp, 1.0 / (1 + ties), 0.0))
    return float(share.mean())


class EquityThresholdAgent(BaseAgent):
    """Raise when equity vs random hands is at least ``raise_threshold``, call
    (or check) when it is at least ``call_threshold``, otherwise check if free
    or fold."""

    name = "equity"

    def __init__(
        self,
        raise_threshold: float = 0.70,
        call_threshold: float = 0.45,
        samples: int = 200,
        raise_fraction: float = 0.75,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        self.raise_threshold = raise_threshold
        self.call_threshold = call_threshold
        self.samples = samples
        self.raise_fraction = raise_fraction
        self.last_equity = float("nan")

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        folded = state.folded
        opponents = sum(1 for i in range(state.num_players) if i != seat and not folded[i])
        eq = monte_carlo_equity(
            state.hole_cards(seat),
            state.board,
            opponents,
            self.samples,
            rng,
            self._engine.evaluate_batch,
        )
        self.last_equity = eq
        if eq >= self.raise_threshold:
            amt = clamp_raise(legal, pot_raise_to(state, seat, self.raise_fraction))
            if amt is not None:
                return self.raise_to(amt)
            return self.check_call()
        if eq >= self.call_threshold or legal.can_check:
            return self.check_call()
        return self.fold()
