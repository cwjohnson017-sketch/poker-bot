"""Agent interface and implementations."""

from __future__ import annotations

from typing import Any

from .base import Agent, BaseAgent, clamp_raise, current_bet, pot_raise_to
from .baselines import AlwaysCallAgent, AlwaysRaiseAgent, RandomAgent
from .equity import EquityThresholdAgent, monte_carlo_equity
from .human import HumanCLIAgent

AGENTS: dict[str, type] = {
    "always_call": AlwaysCallAgent,
    "call": AlwaysCallAgent,
    "always_raise": AlwaysRaiseAgent,
    "raise": AlwaysRaiseAgent,
    "random": RandomAgent,
    "equity": EquityThresholdAgent,
    "human": HumanCLIAgent,
}


def make_agent(name: str, **kwargs: Any) -> Agent:
    """Instantiate an agent by registry name (``always_call``, ``equity``, ...)."""
    try:
        cls = AGENTS[name]
    except KeyError:
        raise ValueError(f"unknown agent {name!r}; choose from {sorted(AGENTS)}") from None
    return cls(**kwargs)


__all__ = [
    "AGENTS",
    "Agent",
    "AlwaysCallAgent",
    "AlwaysRaiseAgent",
    "BaseAgent",
    "EquityThresholdAgent",
    "HumanCLIAgent",
    "RandomAgent",
    "clamp_raise",
    "current_bet",
    "make_agent",
    "monte_carlo_equity",
    "pot_raise_to",
]
