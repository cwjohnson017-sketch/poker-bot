"""Abstractions shared by the blueprint, search and agents (DESIGN.md 5.3).

* :mod:`.actions` - the action abstraction (one Python spec, its Rust twin,
  pseudo-harmonic off-tree mapping);
* :mod:`.isomorphism` - torch/numpy helpers around the Rust canonical index;
* :mod:`.buckets` - equity features, EMD k-means and bucket tables for the
  tabular MCCFR solver.
"""

from .actions import (
    DEFAULT_SPEC,
    ActionSpec,
    as_spec,
    from_rust,
    legal_actions,
    map_offtree,
    pseudo_harmonic,
    to_rust,
)

__all__ = [
    "DEFAULT_SPEC",
    "ActionSpec",
    "as_spec",
    "from_rust",
    "legal_actions",
    "map_offtree",
    "pseudo_harmonic",
    "to_rust",
]
