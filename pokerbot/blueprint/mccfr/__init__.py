"""Tabular MCCFR blueprint (Phase 1, DESIGN.md section 5.4).

The solver core is Rust (``poker_engine.Trainer``, ``engine/src/mccfr.rs``);
this package holds the training driver, the strategy-file reader, the
play-time ``BlueprintAgent`` and a sampled best-response check. See
``README.md`` next to this file.
"""

from .export import StrategyFile, decode_key, export_strategy, make_key, read_strategy
from .interfaces import DEFAULT_ACTIONS, normalize_action_spec, write_bucket_table

__all__ = [
    "DEFAULT_ACTIONS",
    "BlueprintAgent",
    "StrategyFile",
    "TabularPolicy",
    "decode_key",
    "export_strategy",
    "make_key",
    "normalize_action_spec",
    "read_strategy",
    "write_bucket_table",
]


def __getattr__(name: str):
    # Lazy: the agent needs the compiled poker_engine extension.
    if name == "BlueprintAgent":
        from .agent import BlueprintAgent

        return BlueprintAgent
    if name == "TabularPolicy":
        from .policy import TabularPolicy

        return TabularPolicy
    raise AttributeError(name)
