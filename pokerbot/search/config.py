"""Search configuration: dataclasses and YAML loading (``configs/search_default.yaml``).

A YAML file may start from another with ``base: <path>`` (relative to the
repository root): its sections are merged key by key over the base's. That is
how ``configs/search_value_net_fixed.yaml`` is production with fixed iteration
counts, without a copy that could drift.

``street_iterations`` caps the DCFR iterations per street (``{flop: 68, ...}``;
streets without an entry use ``solver.iterations``). With a generous
``time_budget`` the caps decide, so a search plays the same on any GPU, only
slower or faster. Runs on cloud GPUs use this (``cloud/aws/README.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch

from ..config import REPO_ROOT, load_yaml
from ..env.actions import DEFAULT_SPEC, ActionSpec, spec_from_lists
from .leaf import LeafConfig
from .solver import SolverConfig
from .tree import LEAF_MODES, TreeConfig

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "search_default.yaml"


GADGET_TERMINATE = ("rollouts", "blueprint", "blueprint_br", "unsafe")


@dataclass
class GadgetConfig:
    safe: bool = True
    prior_mix: float = 0.05  # uniform mixed into the opponent prior at the gadget
    rollouts: int = 256  # blueprint rollouts for terminate values when nothing is cached
    # terminate values when no earlier solve is cached (the first decision of a street):
    # "rollouts": blueprint-vs-blueprint rollouts (``rollouts`` of them); "blueprint": the
    # opponent's values with both players on the blueprint, in the search tree with its own
    # leaves; "blueprint_br": the opponent's best response to the blueprint there;
    # "unsafe": no gadget at such decisions (the root ranges are the blueprint's own)
    terminate: str = "rollouts"


@dataclass
class SearchConfig:
    device: str = "auto"
    seed: int = 0
    time_budget: dict[int, float] = field(default_factory=lambda: {1: 2.0, 2: 1.0, 3: 1.0})
    min_iterations: int = 10
    # per-street caps on DCFR iterations; streets missing here use solver.iterations
    street_iterations: dict[int, int] = field(default_factory=dict)
    remove_own_blockers: bool = True
    fallback_on_error: bool = True
    tree: TreeConfig = field(default_factory=TreeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    leaf: LeafConfig = field(default_factory=LeafConfig)
    gadget: GadgetConfig = field(default_factory=GadgetConfig)

    def torch_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)

    def budget(self, street: int) -> float:
        return float(self.time_budget.get(street, self.time_budget.get(3, 1.0)))

    def max_iterations(self, street: int) -> int:
        return int(self.street_iterations.get(street, self.solver.iterations))


_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


def _by_street(data: Any, cast: type) -> dict[int, Any]:
    """``{flop: x, ...}`` or ``{1: x, ...}`` (or one number for every street) as
    ``{street index: cast(x)}``."""
    if isinstance(data, int | float):
        return {1: cast(data), 2: cast(data), 3: cast(data)}
    return {_STREETS.get(k, k) if isinstance(k, str) else int(k): cast(v) for k, v in data.items()}


def _merge(base: dict, over: dict) -> dict:
    """``over`` merged into ``base``, recursively for nested mappings."""
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_search_yaml(path: str | Path) -> dict:
    """A search YAML with its ``base:`` chain resolved (paths relative to the
    repository root, or absolute)."""
    data = load_yaml(path)
    base = data.pop("base", None)
    if base is None:
        return data
    base_path = Path(base) if Path(base).is_absolute() else REPO_ROOT / base
    return _merge(load_search_yaml(base_path), data)


def _sub(cls: type, data: dict | None, base: Any = None) -> Any:
    base = base if base is not None else cls()
    if not data:
        return base
    names = {f.name for f in fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return replace(base, **data)


def _spec(data: Any, max_raises: int | None) -> ActionSpec | None:
    if data == "blueprint":
        return None  # the blueprint's own abstraction, filled in by SearchAgent
    if data in (None, "default"):
        spec = DEFAULT_SPEC
    else:
        spec = spec_from_lists([[tuple(a) for a in st] for st in data])
    if max_raises is not None:
        spec = ActionSpec(spec.streets, int(max_raises), spec.dedupe)
    return spec


def search_config(data: dict | str | Path | None = None, **overrides: Any) -> SearchConfig:
    """Build a :class:`SearchConfig` from a mapping, a YAML path (the ``search:``
    section, or the whole file) or ``None`` (the default YAML if present).
    ``overrides`` are applied on top, with nested sections as dicts."""
    if data is None:
        data = load_search_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    elif isinstance(data, str | Path):
        data = load_search_yaml(data)
    elif "base" in data:
        base = data["base"]
        base_path = Path(base) if Path(base).is_absolute() else REPO_ROOT / base
        data = _merge(load_search_yaml(base_path), {k: v for k, v in data.items() if k != "base"})
    data = dict(data.get("search", data))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = {**data[k], **v}
        else:
            data[k] = v
    tree = dict(data.pop("tree", None) or {})
    spec = _spec(tree.pop("actions", None), tree.pop("max_raises", None))
    leaf = _sub(LeafConfig, data.pop("leaf", None))
    if leaf.mode not in LEAF_MODES:
        raise ValueError(f"unknown leaf.mode {leaf.mode!r} (expected one of {LEAF_MODES})")
    if tree.setdefault("leaf_mode", leaf.mode) != leaf.mode:
        raise ValueError("tree.leaf_mode must match leaf.mode (set leaf.mode only)")
    tree.setdefault("num_continuations", len(leaf.strategies))
    tcfg = _sub(TreeConfig, tree, TreeConfig(spec=spec))
    scfg = _sub(SolverConfig, data.pop("solver", None))
    gcfg = _sub(GadgetConfig, data.pop("gadget", None))
    if gcfg.terminate not in GADGET_TERMINATE:
        raise ValueError(
            f"unknown gadget.terminate {gcfg.terminate!r} (expected one of {GADGET_TERMINATE})"
        )
    tb = data.pop("time_budget", None)
    si = data.pop("street_iterations", None)
    cfg = _sub(SearchConfig, data)
    if tb is not None:
        cfg.time_budget = _by_street(tb, float)
    if si is not None:
        cfg.street_iterations = _by_street(si, int)
    cfg.tree, cfg.solver, cfg.leaf, cfg.gadget = tcfg, scfg, leaf, gcfg
    return cfg
