"""Shared fixtures for the MCCFR blueprint tests. They need the compiled
``poker_engine`` extension with the solver and are not collected otherwise
(CI runs the pure-Python suite)."""

from __future__ import annotations

import pytest

try:
    import poker_engine as pe

    HAVE_SOLVER = hasattr(pe, "Trainer")
except ImportError:  # pragma: no cover - depends on the build
    pe = None
    HAVE_SOLVER = False

if not HAVE_SOLVER:
    collect_ignore_glob = ["test_*.py"]


@pytest.fixture(scope="session")
def tiny_blueprint(tmp_path_factory):
    """A tiny-game blueprint trained for 60k iterations (single thread, so
    deterministic), exported to a temporary strategy file."""
    from mccfr_helpers import tiny_config

    trainer = pe.Trainer(tiny_config())
    trainer.run(60_000, 1)
    path = tmp_path_factory.mktemp("blueprint") / "tiny_strategy.bin"
    trainer.export_strategy(str(path))
    return {"trainer": trainer, "path": path}
