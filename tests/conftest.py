import pytest


@pytest.fixture
def reference_engine(monkeypatch):
    """Force the pure-Python engine so tests do not depend on the Rust build."""
    monkeypatch.setenv("POKERBOT_ENGINE", "reference")
