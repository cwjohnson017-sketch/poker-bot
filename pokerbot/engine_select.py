"""Pick the game engine: the Rust ``poker_engine`` module when built, else the
pure-Python reference.

``POKERBOT_ENGINE=reference`` forces the reference engine,
``POKERBOT_ENGINE=rust`` requires the Rust module (ImportError if missing),
unset or ``auto`` prefers Rust and falls back to the reference.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

ENV_VAR = "POKERBOT_ENGINE"


def get_engine(name: str | None = None) -> ModuleType:
    choice = (name or os.environ.get(ENV_VAR, "auto")).strip().lower()
    if choice in ("reference", "ref", "python", "py"):
        return importlib.import_module("pokerbot.reference")
    if choice in ("rust", "poker_engine"):
        return importlib.import_module("poker_engine")
    if choice != "auto":
        raise ValueError(f"{ENV_VAR} must be 'reference', 'rust' or 'auto', got {choice!r}")
    try:
        return importlib.import_module("poker_engine")
    except ImportError:
        return importlib.import_module("pokerbot.reference")


def engine_name(engine: ModuleType) -> str:
    return "reference" if engine.__name__ == "pokerbot.reference" else engine.__name__


def make_config(engine: ModuleType | None = None, **kwargs) -> object:
    """Build ``engine.GameConfig`` from keyword arguments of the contract."""
    engine = engine or get_engine()
    return engine.GameConfig(**kwargs)


def to_engine_action(engine: ModuleType, action) -> object:
    """Convert any object with ``kind``/``amount`` into ``engine.Action``."""
    kind = int(action.kind)
    if kind == 0:
        return engine.Action.fold()
    if kind == 1:
        return engine.Action.check_call()
    if kind == 2:
        return engine.Action.raise_to(int(action.amount))
    raise ValueError(f"unknown action kind {kind}")
