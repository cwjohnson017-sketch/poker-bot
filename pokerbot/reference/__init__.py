"""Pure-Python reference engine implementing ``docs/INTERFACES.md``.

Slow but simple; used for tests, cross-checks against the Rust engine, and as
the fallback engine when ``poker_engine`` is not built.
"""

from .cards import card_from_str, card_to_str, cards_from_str, cards_to_str
from .evaluator import (
    CATEGORY_NAMES,
    evaluate,
    evaluate5,
    evaluate6,
    evaluate7,
    evaluate_batch,
    hand_category,
)
from .game import CHECK_CALL, FOLD, RAISE, Action, GameConfig, GameState, LegalActions

ENGINE_NAME = "reference"

__all__ = [
    "CATEGORY_NAMES",
    "CHECK_CALL",
    "ENGINE_NAME",
    "FOLD",
    "RAISE",
    "Action",
    "GameConfig",
    "GameState",
    "LegalActions",
    "card_from_str",
    "card_to_str",
    "cards_from_str",
    "cards_to_str",
    "evaluate",
    "evaluate5",
    "evaluate6",
    "evaluate7",
    "evaluate_batch",
    "hand_category",
]
