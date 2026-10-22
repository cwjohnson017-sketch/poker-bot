"""Running results of a play session, from the person's point of view.

Raw chips, the luck-adjusted chips of :mod:`pokerbot.eval.luck` (AIVAT's
chance terms: all-in EV and street equity changes, both hands known) and AIVAT
(:mod:`pokerbot.eval.aivat`: the chance terms plus the bot's action terms),
each as mbb/hand with a percentile-bootstrap CI
(:func:`pokerbot.eval.stats.win_rate`). The bot's numbers are the negatives.
"""

from __future__ import annotations

from ..eval.stats import win_rate


class SessionStats:
    def __init__(self, big_blind: int, confidence: float = 0.95, n_boot: int = 2000) -> None:
        self.big_blind = int(big_blind)
        self.confidence = confidence
        self.n_boot = n_boot
        self.raw: list[float] = []
        self.adjusted: list[float] = []
        self.aivat: list[float] = []

    @property
    def hands(self) -> int:
        return len(self.raw)

    def add(self, raw: float, adjusted: float, aivat: float | None = None) -> None:
        """One hand. ``aivat`` defaults to ``adjusted`` (no action terms)."""
        self.raw.append(float(raw))
        self.adjusted.append(float(adjusted))
        self.aivat.append(float(adjusted if aivat is None else aivat))

    def _rate(self, samples: list[float]) -> dict:
        wr = win_rate(samples, self.big_blind, 1, self.confidence, self.n_boot, rng=0)
        return {"mbb": wr.mbb_per_hand, "lo": wr.ci_low, "hi": wr.ci_high}

    def summary(self) -> dict:
        out: dict = {"hands": self.hands, "net": 0.0, "net_bb": 0.0}
        if not self.raw:
            return out
        net = sum(self.raw)
        out.update(
            net=net,
            net_bb=net / self.big_blind,
            raw=self._rate(self.raw),
            adjusted=self._rate(self.adjusted),
            aivat=self._rate(self.aivat),
            confidence=self.confidence,
        )
        return out
