"""Run configuration helpers: YAML loading, game configs, run provenance.

Every run is a config file plus the git commit it ran at; ``run_info``
collects both so scripts can print or store them next to their outputs.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from .engine_select import engine_name, get_engine

REPO_ROOT = Path(__file__).resolve().parent.parent

GAME_DEFAULTS: dict[str, Any] = {
    "num_players": 2,
    "stacks": [20000, 20000],
    "small_blind": 50,
    "big_blind": 100,
    "ante": 0,
}


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    return data


def game_config(section: dict[str, Any] | None = None, engine: ModuleType | None = None) -> Any:
    """Build ``engine.GameConfig`` from a ``game:`` section (defaults: HU 50/100, 200bb)."""
    engine = engine or get_engine()
    cfg = dict(GAME_DEFAULTS)
    cfg.update(section or {})
    n = int(cfg["num_players"])
    stacks = cfg["stacks"]
    if isinstance(stacks, int):
        stacks = [stacks] * n
    elif len(stacks) != n:
        stacks = [int(stacks[0])] * n
    return engine.GameConfig(
        num_players=n,
        stacks=[int(s) for s in stacks],
        small_blind=int(cfg["small_blind"]),
        big_blind=int(cfg["big_blind"]),
        ante=int(cfg["ante"]),
    )


def git_hash(short: bool = False) -> str:
    """Current commit hash, with ``-dirty`` when the tree has local changes."""
    try:
        args = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
        h = subprocess.run(
            args, cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return h + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run_info(config_path: str | Path | None = None, engine: ModuleType | None = None) -> dict:
    engine = engine or get_engine()
    return {
        "git": git_hash(),
        "config": str(config_path) if config_path else None,
        "engine": engine_name(engine),
    }
