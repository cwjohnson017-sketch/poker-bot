"""Neural blueprint: Deep CFR with the single-deep-CFR average (DESIGN.md 5.5).

See ``README.md`` in this directory for the algorithm as implemented.
Imports are lazy-friendly: the agent and trainer pull in their own
dependencies, so ``pokerbot.agents`` can register ``neural:<dir>`` cheaply.
"""

from .config import DeepCFRConfig
from .features import FeatureConfig, features_from_obs
from .memory import ReservoirMemory
from .networks import AdvantageNet, NetConfig, StrategyHead, regret_matching
from .policy import SDCFRPolicy
from .traversal import (
    FrontierTraverser,
    NetPolicy,
    TraversalConfig,
    TraversalResult,
    actor_probs,
    rollout,
    uniform_policy,
)

__all__ = [
    "AdvantageNet",
    "DeepCFRConfig",
    "FeatureConfig",
    "FrontierTraverser",
    "NetConfig",
    "NetPolicy",
    "ReservoirMemory",
    "SDCFRPolicy",
    "StrategyHead",
    "TraversalConfig",
    "TraversalResult",
    "actor_probs",
    "features_from_obs",
    "regret_matching",
    "rollout",
    "uniform_policy",
]
