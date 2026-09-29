"""What the MCCFR blueprint needs from the (future) ``pokerbot.abstraction``
package, as small interfaces with working defaults.

**Action abstraction.** Per-street lists in the contract format of
``docs/INTERFACES.md``: ``("fold",)``, ``("check_call",)``,
``("raise", pot_fraction)``, ``("allin",)``. ``normalize_action_spec`` accepts
that format plus YAML-friendly shorthands (``fold``, ``call``, ``allin``,
``[raise, 0.75]``). The Rust ``poker_engine.ActionAbstraction`` implements the
mapping to concrete actions and the pseudo-harmonic reverse mapping.

**Card abstraction.** A bucket function of ``(street, hole, board)``. The
solver cannot call Python per node, so buckets are either the Rust default
(169 preflop classes, quantized hand strength after the flop) or per-street
tables: a 1-D integer ``.npy`` array of length
``poker_engine.canonical_size(street)`` whose entry ``i`` is the bucket of
every hand with ``poker_engine.canonical_index(street, hole, board) == i``.
``poker_engine.canonical_unindex(street, i)`` gives a representative hand of
index ``i``, so an abstraction module can compute features per index and
write a table with ``write_bucket_table``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

STREETS = ("preflop", "flop", "turn", "river")

# DESIGN.md section 5.3. Heads-up preflop, "2.5x" and "3x" raises are 0.75 and
# 1.0 pot fractions (raise to x times the current bet = (x - 1) / 2 pot).
DEFAULT_ACTIONS: dict[str, list[tuple]] = {
    "preflop": [("fold",), ("check_call",), ("raise", 0.75), ("raise", 1.0), ("allin",)],
    "flop": [
        ("fold",),
        ("check_call",),
        ("raise", 0.33),
        ("raise", 0.75),
        ("raise", 1.5),
        ("allin",),
    ],
    "turn": [("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",)],
    "river": [
        ("fold",),
        ("check_call",),
        ("raise", 0.5),
        ("raise", 1.0),
        ("raise", 2.0),
        ("allin",),
    ],
}

_NAMES = {
    "fold": "fold",
    "f": "fold",
    "check_call": "check_call",
    "call": "check_call",
    "check": "check_call",
    "c": "check_call",
    "allin": "allin",
    "all_in": "allin",
    "all-in": "allin",
    "a": "allin",
    "raise": "raise",
    "r": "raise",
}


def normalize_action(a: Any) -> tuple:
    """One abstract action in contract form."""
    if isinstance(a, str):
        name, args = a, []
    else:
        items = list(a)
        if not items:
            raise ValueError("empty abstract action")
        name, args = str(items[0]), items[1:]
    kind = _NAMES.get(name.strip().lower())
    if kind is None:
        raise ValueError(f"unknown abstract action {a!r}")
    if kind == "raise":
        if len(args) != 1 or float(args[0]) <= 0:
            raise ValueError(f"raise needs one positive pot fraction: {a!r}")
        return ("raise", float(args[0]))
    if args:
        raise ValueError(f"{kind} takes no argument: {a!r}")
    return (kind,)


def normalize_action_spec(spec: Any = None) -> list[list[tuple]]:
    """Four per-street lists (preflop..river) in contract form. ``spec`` is a
    dict keyed by street name (missing streets use the default) or a list of
    four lists; ``None`` gives ``DEFAULT_ACTIONS``."""
    if spec is None:
        spec = {}
    if isinstance(spec, dict):
        return [[normalize_action(a) for a in spec.get(s, DEFAULT_ACTIONS[s])] for s in STREETS]
    lists = list(spec)
    if len(lists) != 4:
        raise ValueError("need 4 per-street action lists")
    return [[normalize_action(a) for a in lst] for lst in lists]


@runtime_checkable
class CardBucketer(Protocol):
    """Card abstraction as seen by play-time code (``poker_engine.CardAbstraction``,
    ``BlueprintStrategy`` and ``Trainer`` all provide it)."""

    def bucket(self, street: int, hole: Sequence[int], board: Sequence[int]) -> int: ...


def write_bucket_table(path: str | Path, street: int, buckets: np.ndarray) -> Path:
    """Validate and save a bucket table for ``street`` (entry ``i`` = bucket of
    canonical index ``i``) as ``.npy``; use the path in ``cards.tables``."""
    import poker_engine as pe

    arr = np.asarray(buckets)
    if arr.ndim != 1 or arr.shape[0] != pe.canonical_size(street):
        raise ValueError(
            f"street {street} table must have shape ({pe.canonical_size(street)},), got {arr.shape}"
        )
    if arr.size and (arr.min() < 0 or arr.max() >= 65000):
        raise ValueError("buckets must be in 0..65000")
    dtype = np.uint16 if arr.max(initial=0) < 65536 else np.uint32
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr.astype(dtype))
    return path
