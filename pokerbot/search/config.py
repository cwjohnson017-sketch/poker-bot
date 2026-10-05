"""Search configuration: dataclasses and YAML loading (``configs/search_default.yaml``)."""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import torch

from ..config import REPO_ROOT, load_yaml
from ..env.actions import DEFAULT_SPEC, ActionSpec, spec_from_lists
from .leaf import LeafConfig
from .solver import SolverConfig
from .tree import TreeConfig

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "search_default.yaml"


@dataclass
class GadgetConfig:
    safe: bool = True
    prior_mix: float = 0.05  # uniform mixed into the opponent prior at the gadget
    rollouts: int = 256  # blueprint rollouts for terminate values when nothing is cached


@dataclass
class SearchConfig:
    device: str = "auto"
    seed: int = 0
    time_budget: dict[int, float] = field(default_factory=lambda: {1: 2.0, 2: 1.0, 3: 1.0})
    min_iterations: int = 10
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


_STREETS = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}


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
        data = load_yaml(DEFAULT_CONFIG_PATH) if DEFAULT_CONFIG_PATH.exists() else {}
    elif isinstance(data, str | Path):
        data = load_yaml(data)
    data = dict(data.get("search", data))
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(data.get(k), dict):
            data[k] = {**data[k], **v}
        else:
            data[k] = v
    tree = dict(data.pop("tree", None) or {})
    spec = _spec(tree.pop("actions", None), tree.pop("max_raises", None))
    leaf = _sub(LeafConfig, data.pop("leaf", None))
    tree.setdefault("num_continuations", len(leaf.strategies))
    tcfg = _sub(TreeConfig, tree, TreeConfig(spec=spec))
    scfg = _sub(SolverConfig, data.pop("solver", None))
    gcfg = _sub(GadgetConfig, data.pop("gadget", None))
    tb = data.pop("time_budget", None)
    cfg = _sub(SearchConfig, data)
    if tb is not None:
        if isinstance(tb, int | float):
            tb = {1: float(tb), 2: float(tb), 3: float(tb)}
        cfg.time_budget = {
            _STREETS.get(k, k) if isinstance(k, str) else int(k): float(v) for k, v in tb.items()
        }
    cfg.tree, cfg.solver, cfg.leaf, cfg.gadget = tcfg, scfg, leaf, gcfg
    return cfg
