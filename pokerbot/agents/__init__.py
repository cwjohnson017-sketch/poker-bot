"""Agent interface and implementations."""

from __future__ import annotations

from typing import Any

from .base import Agent, BaseAgent, clamp_raise, current_bet, pot_raise_to
from .baselines import AlwaysCallAgent, AlwaysRaiseAgent, RandomAgent
from .equity import EquityThresholdAgent, monte_carlo_equity
from .human import HumanCLIAgent
from .policy import (
    FixedPolicyAgent,
    PolicyAgent,
    SampledPolicyAgent,
    UniformPolicyAgent,
    as_policy_agent,
)
from .registry import available_agents, parse_spec, register

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
    """Instantiate an agent from a name or spec string (``always_call``,
    ``equity:samples=100``, ``blueprint:<checkpoint>``); see ``registry``."""
    from .registry import make_agent as _make

    return _make(name, **kwargs)


__all__ = [
    "AGENTS",
    "Agent",
    "AlwaysCallAgent",
    "AlwaysRaiseAgent",
    "BaseAgent",
    "EquityThresholdAgent",
    "FixedPolicyAgent",
    "HumanCLIAgent",
    "PolicyAgent",
    "RandomAgent",
    "SampledPolicyAgent",
    "UniformPolicyAgent",
    "as_policy_agent",
    "available_agents",
    "clamp_raise",
    "current_bet",
    "make_agent",
    "monte_carlo_equity",
    "parse_spec",
    "pot_raise_to",
    "register",
]
