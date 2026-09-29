"""Match runner: single hands, plain matches and duplicate matches.

Agents only ever see a :class:`~pokerbot.eval.masking.MaskedState` for their
own seat. Results are reported from the first agent's point of view in
mbb/hand with a bootstrap confidence interval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, TextIO

import numpy as np

from ..engine_select import engine_name, get_engine, to_engine_action
from .history import HandHistoryWriter
from .masking import MaskedState
from .stats import WinRate, win_rate


class IllegalActionError(ValueError):
    pass


def showdown_seats(state: Any) -> list[int]:
    """Seats whose cards are shown at the end of a hand (none if won by folds)."""
    live = [p for p, f in enumerate(state.folded) if not f]
    return live if len(live) > 1 else []


def play_hand(
    config: Any,
    agents: Sequence[Any],
    button: int,
    deck: Sequence[int],
    rng: np.random.Generator,
    engine: ModuleType | None = None,
    on_illegal: str = "raise",
) -> Any:
    """Play one hand and return the terminal (unmasked) ``GameState``.

    ``on_illegal``: ``"raise"`` raises :class:`IllegalActionError`; ``"fold"``
    replaces an illegal action by fold (or check when checking is free).
    """
    engine = engine or get_engine()
    if len(agents) != config.num_players:
        raise ValueError("need one agent per seat")
    state = engine.GameState.new_hand(config, button, [int(c) for c in deck])
    for seat, agent in enumerate(agents):
        agent.new_hand(seat, config)
    redeal_rng = np.random.default_rng(int(rng.integers(2**63)))
    while not state.is_terminal:
        p = state.current_player
        view = MaskedState(state, p, config, redeal_rng)
        raw = agents[p].act(view, p, rng)
        try:
            action = to_engine_action(engine, raw)
            state.apply(action)
        except (ValueError, AttributeError, TypeError) as e:
            if on_illegal != "fold":
                name = getattr(agents[p], "name", type(agents[p]).__name__)
                raise IllegalActionError(f"agent {name} (seat {p}) sent {raw!r}: {e}") from e
            legal = state.legal_actions()
            state.apply(engine.Action.check_call() if legal.can_check else engine.Action.fold())
    shown = showdown_seats(state)
    for seat, agent in enumerate(agents):
        agent.observe_end(MaskedState(state, seat, config, redeal_rng, revealed=shown))
    return state


def _names(agents: Sequence[Any]) -> list[str]:
    return [str(getattr(a, "name", type(a).__name__)) for a in agents]


@dataclass
class MatchResult:
    """Outcome of a match from agent A's (``names[0]``) point of view.

    ``samples[i]`` is A's chip result for sample unit ``i``: one hand in a
    plain match, one deal (two hands, both seatings) in a duplicate match.
    ``b_samples`` holds the chip results of everyone else in the same units.
    """

    names: list[str]
    big_blind: int
    samples: np.ndarray
    b_samples: np.ndarray
    hands_per_sample: int
    duplicate: bool
    confidence: float = 0.95
    n_boot: int = 2000
    seed: int = 0
    seat_payoffs: np.ndarray | None = None
    stats: WinRate = field(init=False)

    def __post_init__(self) -> None:
        self.stats = win_rate(
            self.samples,
            self.big_blind,
            self.hands_per_sample,
            self.confidence,
            self.n_boot,
            rng=self.seed,
        )

    @property
    def hands(self) -> int:
        return int(len(self.samples) * self.hands_per_sample)

    @property
    def mbb_per_hand(self) -> float:
        return self.stats.mbb_per_hand

    @property
    def ci(self) -> tuple[float, float]:
        return self.stats.ci_low, self.stats.ci_high

    @property
    def total_a(self) -> int:
        return int(self.samples.sum())

    @property
    def total_b(self) -> int:
        return int(self.b_samples.sum())

    def summary(self) -> str:
        kind = "duplicate" if self.duplicate else "plain"
        return f"{self.names[0]} vs {' / '.join(self.names[1:])} ({kind}): {self.stats}"


def run_match(
    agents: Sequence[Any],
    config: Any,
    num_hands: int,
    seed: int = 0,
    engine: ModuleType | None = None,
    history: TextIO | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    on_illegal: str = "raise",
) -> MatchResult:
    """Agents keep their seats; the button rotates every hand; each hand gets a
    fresh shuffled deck from ``seed``. Works for 2..9 seats."""
    engine = engine or get_engine()
    n = config.num_players
    names = _names(agents)
    writer = HandHistoryWriter(history, config, names, engine_name(engine)) if history else None
    deck_rng = np.random.default_rng(seed)
    pay = np.zeros((num_hands, n), dtype=np.int64)
    for h in range(num_hands):
        deck = deck_rng.permutation(52)
        rng = np.random.default_rng((seed, h))
        state = play_hand(config, agents, h % n, deck, rng, engine, on_illegal)
        pay[h] = state.payoffs()
        if writer:
            writer.write(state, h)
    return MatchResult(
        names=names,
        big_blind=config.big_blind,
        samples=pay[:, 0].copy(),
        b_samples=pay[:, 1:].sum(axis=1),
        hands_per_sample=1,
        duplicate=False,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
        seat_payoffs=pay,
    )


def run_duplicate_match(
    agent_a: Any,
    agent_b: Any,
    config: Any,
    num_deals: int,
    seed: int = 0,
    engine: ModuleType | None = None,
    history: TextIO | None = None,
    confidence: float = 0.95,
    n_boot: int = 2000,
    on_illegal: str = "raise",
) -> MatchResult:
    """Heads-up duplicate match: every deal is played twice with the same deck
    and button, once with A in seat 0 and once with A in seat 1, so each agent
    holds each hand in each position. ``num_deals`` deals = ``2 * num_deals``
    hands. Both plays of a deal use the same agent RNG seed."""
    if config.num_players != 2:
        raise ValueError("duplicate matches are heads-up only")
    engine = engine or get_engine()
    names = _names([agent_a, agent_b])
    writer = HandHistoryWriter(history, config, names, engine_name(engine)) if history else None
    deck_rng = np.random.default_rng(seed)
    a_res = np.zeros(num_deals, dtype=np.int64)
    b_res = np.zeros(num_deals, dtype=np.int64)
    seat_pay = np.zeros((num_deals, 2, 2), dtype=np.int64)
    for d in range(num_deals):
        deck = deck_rng.permutation(52)
        button = d % 2
        for g, seating in enumerate(((agent_a, agent_b), (agent_b, agent_a))):
            rng = np.random.default_rng((seed, d))
            state = play_hand(config, seating, button, deck, rng, engine, on_illegal)
            p = state.payoffs()
            seat_pay[d, g] = p
            a_seat = 0 if g == 0 else 1
            a_res[d] += p[a_seat]
            b_res[d] += p[1 - a_seat]
            if writer:
                seat_names = names if g == 0 else names[::-1]
                writer.write(state, 2 * d + g, seat_names)
    return MatchResult(
        names=names,
        big_blind=config.big_blind,
        samples=a_res,
        b_samples=b_res,
        hands_per_sample=2,
        duplicate=True,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
        seat_payoffs=seat_pay,
    )
