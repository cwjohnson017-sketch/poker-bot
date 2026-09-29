"""Fixtures for the abstraction package tests. They need the compiled
``poker_engine`` extension (canonical index, ActionAbstraction, Trainer)."""

from __future__ import annotations

import pytest

try:
    import poker_engine as pe

    HAVE_ENGINE = hasattr(pe, "canonical_index_batch") and hasattr(pe, "Trainer")
except ImportError:  # pragma: no cover - depends on the build
    HAVE_ENGINE = False

if not HAVE_ENGINE:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session")
def tiny_tables(tmp_path_factory):
    """Bucket tables from ``configs/buckets_tiny.yaml`` (256 random classes per
    street, 8 buckets), built once into a temporary directory."""
    import torch

    from pokerbot.abstraction.buckets import build, load_config
    from pokerbot.config import REPO_ROOT

    torch.manual_seed(0)
    bc = load_config(REPO_ROOT / "configs" / "buckets_tiny.yaml")
    bc.out_dir = str(tmp_path_factory.mktemp("buckets_tiny"))
    results = build(bc, log=lambda _msg: None)
    return {"config": bc, "results": results, "out_dir": bc.out_dir}
