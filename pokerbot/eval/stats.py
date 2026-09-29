"""Win-rate statistics: mbb/hand and bootstrap confidence intervals."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


def to_mbb(chips_per_hand: float, big_blind: int) -> float:
    return 1000.0 * chips_per_hand / big_blind


def mbb_per_hand(samples: Sequence[float], big_blind: int, hands_per_sample: int = 1) -> float:
    """Mean winnings in milli-big-blinds per hand. Each sample is the chip
    result of ``hands_per_sample`` hands (2 for a duplicate pair)."""
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    return to_mbb(x.mean() / hands_per_sample, big_blind)


def bootstrap_ci(
    samples: Sequence[float],
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval for the mean of ``samples``."""
    x = np.asarray(samples, dtype=np.float64)
    n = x.size
    if n == 0:
        return float("nan"), float("nan")
    if n == 1:
        return float(x[0]), float(x[0])
    rng = np.random.default_rng(rng)
    chunk = max(1, min(n_boot, 4_000_000 // n))
    means = np.empty(n_boot)
    for start in range(0, n_boot, chunk):
        stop = min(n_boot, start + chunk)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = x[idx].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    lo, hi = np.quantile(means, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


@dataclass(frozen=True)
class WinRate:
    mbb_per_hand: float
    ci_low: float
    ci_high: float
    hands: int
    confidence: float = 0.95

    @property
    def half_width(self) -> float:
        return (self.ci_high - self.ci_low) / 2.0

    def significant(self) -> bool:
        """True when the confidence interval excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def __str__(self) -> str:
        pct = round(self.confidence * 100)
        return (
            f"{self.mbb_per_hand:+.1f} mbb/h ± {self.half_width:.1f} "
            f"({pct}% CI [{self.ci_low:+.1f}, {self.ci_high:+.1f}], {self.hands} hands)"
        )


def win_rate(
    samples: Sequence[float],
    big_blind: int,
    hands_per_sample: int = 1,
    confidence: float = 0.95,
    n_boot: int = 2000,
    rng: np.random.Generator | int | None = 0,
) -> WinRate:
    x = np.asarray(samples, dtype=np.float64)
    lo, hi = bootstrap_ci(x, confidence, n_boot, rng)
    scale = 1000.0 / (big_blind * hands_per_sample)
    return WinRate(
        mbb_per_hand=mbb_per_hand(x, big_blind, hands_per_sample),
        ci_low=lo * scale,
        ci_high=hi * scale,
        hands=int(x.size * hands_per_sample),
        confidence=confidence,
    )
