import importlib.util
from pathlib import Path

import pytest

import pokerbot.reference as reference
from pokerbot.config import game_config, git_hash, load_yaml
from pokerbot.engine_select import get_engine, to_engine_action

HAVE_RUST = importlib.util.find_spec("poker_engine") is not None


def test_env_override_reference(monkeypatch):
    monkeypatch.setenv("POKERBOT_ENGINE", "reference")
    assert get_engine() is reference


def test_auto_falls_back(monkeypatch):
    monkeypatch.delenv("POKERBOT_ENGINE", raising=False)
    eng = get_engine()
    if HAVE_RUST:
        assert eng.__name__ == "poker_engine"
    else:
        assert eng is reference


def test_rust_required(monkeypatch):
    monkeypatch.setenv("POKERBOT_ENGINE", "rust")
    if HAVE_RUST:
        assert get_engine().__name__ == "poker_engine"
    else:
        with pytest.raises(ImportError):
            get_engine()


def test_bad_choice(monkeypatch):
    monkeypatch.setenv("POKERBOT_ENGINE", "java")
    with pytest.raises(ValueError):
        get_engine()


def test_reference_exports_contract():
    for name in (
        "GameConfig",
        "Action",
        "LegalActions",
        "GameState",
        "evaluate5",
        "evaluate6",
        "evaluate7",
        "evaluate_batch",
        "hand_category",
        "card_from_str",
        "card_to_str",
        "FOLD",
        "CHECK_CALL",
        "RAISE",
    ):
        assert hasattr(reference, name), name
    assert (reference.FOLD, reference.CHECK_CALL, reference.RAISE) == (0, 1, 2)


def test_to_engine_action():
    class Foreign:
        kind, amount = 2, 500

    assert to_engine_action(reference, Foreign()) == reference.Action.raise_to(500)


def test_match_config_file_loads():
    cfg = load_yaml(Path(__file__).parent.parent / "configs" / "match.yaml")
    gc = game_config(cfg["game"], reference)
    assert gc.num_players == 2 and gc.big_blind == 100 and gc.stacks == [10000, 10000]
    assert git_hash() != ""
