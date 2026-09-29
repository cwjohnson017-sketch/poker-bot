"""Evaluation harness: match runner, duplicate matches, statistics, histories,
checkpoint ladder, local best response (LBR), approximate best response (ABR)."""

from typing import Any

from .history import HandHistoryWriter, format_hand
from .ladder import Ladder, compute_ratings, schedule_pairs
from .logging import RunLogger
from .masking import HiddenInformationError, MaskedState, determinize
from .match import (
    IllegalActionError,
    MatchResult,
    play_hand,
    run_duplicate_match,
    run_match,
    showdown_seats,
)
from .stats import WinRate, bootstrap_ci, mbb_per_hand, win_rate

# torch-backed tools are imported on first use so the match runner stays light
_LAZY = {
    "LBRAgent": ".lbr",
    "LBRResult": ".lbr",
    "run_lbr": ".lbr",
    "range_equity": ".lbr",
    "ABRConfig": ".abr",
    "train_abr": ".abr",
    "evaluate_br": ".abr",
    "make_vec_policy": ".abr",
    "UniformRandomVecPolicy": ".abr",
    "ScalarVecPolicy": ".abr",
    "VecPolicy": ".abr",
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ABRConfig",
    "HandHistoryWriter",
    "HiddenInformationError",
    "IllegalActionError",
    "LBRAgent",
    "LBRResult",
    "Ladder",
    "MaskedState",
    "MatchResult",
    "RunLogger",
    "ScalarVecPolicy",
    "UniformRandomVecPolicy",
    "VecPolicy",
    "WinRate",
    "bootstrap_ci",
    "compute_ratings",
    "determinize",
    "evaluate_br",
    "format_hand",
    "make_vec_policy",
    "mbb_per_hand",
    "play_hand",
    "range_equity",
    "run_duplicate_match",
    "run_lbr",
    "run_match",
    "schedule_pairs",
    "showdown_seats",
    "train_abr",
    "win_rate",
]
