"""Real-time depth-limited search: public subgame trees, range-vs-range CFR,
blueprint rollouts at depth-limit leaves, and the safe resolving gadget.

See ``pokerbot/search/README.md``.
"""

from .agent import SearchAgent, make_search_agent
from .blueprint import (
    Blueprint,
    TabularBlueprintFromCallable,
    UniformBlueprint,
    make_blueprint,
    policy_matrix,
    range_reach,
    register_blueprint,
)
from .combos import NUM_COMBOS, combo_cards, combo_index
from .config import SearchConfig, search_config
from .gadget import ContinualCache, Gadget
from .leaf import LeafConfig
from .showdown import ShowdownTables, naive_showdown
from .solver import RangeSolver, SolverConfig
from .tree import SubgameTree, TreeBuilder, TreeConfig, build_tree

__all__ = [
    "NUM_COMBOS",
    "Blueprint",
    "ContinualCache",
    "Gadget",
    "LeafConfig",
    "RangeSolver",
    "SearchAgent",
    "SearchConfig",
    "ShowdownTables",
    "SolverConfig",
    "SubgameTree",
    "TabularBlueprintFromCallable",
    "TreeBuilder",
    "TreeConfig",
    "UniformBlueprint",
    "build_tree",
    "combo_cards",
    "combo_index",
    "make_blueprint",
    "make_search_agent",
    "naive_showdown",
    "policy_matrix",
    "range_reach",
    "register_blueprint",
    "search_config",
]
