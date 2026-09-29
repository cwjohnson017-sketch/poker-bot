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
    if name.startswith("search:"):  # real-time search over a blueprint spec
        from ..search.agent import make_search_agent

        return make_search_agent(name.split(":", 1)[1], **kwargs)
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
