"""AIVAT for a bot whose strategy is known: chance terms plus the bot's action terms.

AIVAT (Burch et al., AAAI 2018) scores a hand as its result plus a sum of
corrections, one per chance event and one per decision of a player whose
strategy is known. Each correction has the form ``E_x[v(h x)] - v(h x_taken)``:
the expected value of whatever happens next minus the value of what did
happen, under some value function ``v`` and the true distribution of ``x``
given the history ``h`` so far. Here ``h`` is the whole history, both hands
included. Whatever ``v`` is, such a term has zero mean given everything that
happened before it, so the corrections are martingale differences over the
hand and the total (finitely many terms, bounded values) has exactly the mean
of the raw result. A good ``v`` only lowers the variance; a poor one cannot
add bias. Different terms may use different value functions.

This module combines two families:

* **Chance terms** (:mod:`pokerbot.eval.luck`): the all-in EV and the street
  equity changes, valued by the check-down value of seat 0's equity with both
  hands known. A board card is dealt uniformly from the cards not yet seen,
  whatever the players did, so ``E_x`` there is the exact average over the
  undealt cards.
* **Action terms** for the bot's own decisions. At a decision the bot samples
  its action from ``sigma(.|c)``, the strategy of its own combo ``c``
  (:class:`~pokerbot.search.agent.SearchAgent` draws from exactly the
  recorded row). The opponent's cards do not enter that draw, so given the
  whole history the action is distributed as ``sigma(.|c)`` and the term
  ``sum_a sigma(a|c) v(a) - v(a_taken)`` has zero mean for any ``v(a)``
  fixed before the draw. The bot's seat gains the term and the other seat
  loses it, so rows stay zero-sum.

The opponent's actions get no term: a person's strategy is unknown, so
``E_a`` cannot be taken over it.

**Which values.** The evaluator knows both hands, so ``v(a)`` may depend on
the opponent's combo ``o`` too. ``values="pair"`` (the default) uses the
record's ``values_vs[o, a]``: the bot's expected chips holding ``c`` after
``a`` against ``o``, under the solved strategies
(:func:`~pokerbot.search.agent.pair_values`). ``values="range"`` uses
``values[c, a]``, the same averaged over the modelled opponent range. The
range version is nearly useless by construction: a converged solve makes the
bot indifferent between the actions it mixes, so ``sum_a sigma v - v(a_taken)``
is about zero where it mixes and exactly zero where it does not. The pair
version is not: holding ``c`` against the actual ``o`` the actions differ in
value.

Every term is decided from what is known before its own random event, which is
what the zero-mean argument needs: a decision whose values are missing or not
finite for the actions the bot could take (``values`` is NaN where the
modelled opponent range is empty, ``values_vs`` on combos sharing a card with
``c``) falls back from pair to range values, or gets no term at all. That
choice depends on the record and on ``o``, both fixed before the action is
drawn. Actions with zero probability never enter the expectation.

**Preflop decisions have no action terms.** The production preflop comes from
the solved preflop table (:class:`~pokerbot.search.preflop_solve.PreflopTable`),
which stores strategies but not values, and a blueprint preflop has no values
either. Their records carry ``values = None`` and score zero, which keeps the
estimate unbiased.

**What the value function costs.** Unbiasedness does not depend on ``v``; the
variance does. The chance terms and the action terms use different value
functions (check-down equity, the solve's values), so they do not telescope
the way a single value function would, and the solve's values assume the
opponent plays the solved strategy: against an opponent who does not, they
predict the hand's outcome less well. ``pokerbot/eval/README.md`` has the
measured standard deviations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

from ..search.combos import combo_index
from .luck import HandRecord, luck_adjusted
from .stats import WinRate, win_rate


def _term(p: np.ndarray, v: np.ndarray | None, taken: int) -> float | None:
    """``sum_a p(a) v(a) - v(taken)``, or None when a value it needs is not finite."""
    if v is None:
        return None
    v = np.asarray(v, dtype=np.float64)
    used = p > 0
    if not np.isfinite(v[used]).all() or not np.isfinite(v[taken]):
        return None
    return float((p[used] * v[used]).sum() - v[taken])


def decision_correction(
    decision: dict, combo: int, opp_combo: int | None = None, values: str = "pair"
) -> float:
    """The action term ``sum_a sigma(a) v(a) - v(a_taken)`` of one bot decision
    for the bot's own ``combo``; zero when the record has no usable values.

    ``values="pair"`` values the actions against the opponent's actual combo
    ``opp_combo`` (the record's ``values_vs`` row), falling back to the range
    values when that row is missing or not finite; ``values="range"`` always
    uses the range values (``values[combo]``). ``sigma`` is the strategy row
    normalised the way the agent normalises it before sampling (uniform when
    the row is unusable)."""
    if values not in ("pair", "range"):
        raise ValueError(f"values must be 'pair' or 'range', got {values!r}")
    p = np.asarray(decision["strategy"][combo], dtype=np.float64)
    if not np.isfinite(p).all() or p.sum() <= 0:
        p = np.ones(len(p))
    p = p / p.sum()
    taken = int(decision["taken"])
    term = None
    vs = decision.get("values_vs")
    if values == "pair" and vs is not None and opp_combo is not None:
        term = _term(p, vs[opp_combo], taken)
    if term is None and decision.get("values") is not None:
        term = _term(p, decision["values"][combo], taken)
    return 0.0 if term is None else term


def action_correction(
    decisions: Sequence[dict],
    hole: Sequence[int],
    opp_hole: Sequence[int] | None = None,
    values: str = "pair",
) -> float:
    """Sum of the action terms of one hand's bot ``decisions``; ``hole`` and
    ``opp_hole`` are the bot's and the opponent's hole cards."""
    c = combo_index(int(hole[0]), int(hole[1]))
    o = None if opp_hole is None else combo_index(int(opp_hole[0]), int(opp_hole[1]))
    return float(sum(decision_correction(d, c, o, values) for d in decisions))


@torch.no_grad()
def aivat_adjusted(
    hands: Sequence[HandRecord],
    decisions: Sequence[Sequence[dict]],
    bot_seats: int | Sequence[int],
    config: Any,
    engine: ModuleType | None = None,
    preflop_samples: int = 2048,
    seed: int = 0,
    device: torch.device | str = "cpu",
    values: str = "pair",
) -> tuple[np.ndarray, np.ndarray]:
    """AIVAT results of heads-up ``hands`` played by a recording bot.

    ``decisions[i]`` are the bot's decision records of hand ``i``
    (``SearchAgent.decisions`` after that hand) and ``bot_seats`` its seat
    (one for all hands, or one per hand). Returns ``(chance, aivat)``, both
    ``[n, 2]`` zero-sum chip results: ``chance`` is
    :func:`~pokerbot.eval.luck.luck_adjusted`, ``aivat`` adds the bot's
    action terms (``values`` as in :func:`decision_correction`)."""
    n = len(hands)
    if len(decisions) != n:
        raise ValueError(f"{n} hands but {len(decisions)} decision lists")
    seats = [int(bot_seats)] * n if np.isscalar(bot_seats) else [int(s) for s in bot_seats]
    chance = luck_adjusted(hands, config, engine, preflop_samples, seed, device)
    out = chance.copy()
    for i, (hand, recs, seat) in enumerate(zip(hands, decisions, seats, strict=True)):
        state = hand[2]
        corr = action_correction(recs, state.hole_cards(seat), state.hole_cards(1 - seat), values)
        out[i, seat] += corr
        out[i, 1 - seat] -= corr
    return chance, out


# -- play sessions -------------------------------------------------------------


@dataclass
class SessionScore:
    """Per-hand results of a play session from the person's seat, in chips:
    ``raw``, ``chance`` (luck-adjusted: the chance terms) and ``aivat`` (chance
    plus the bot's action terms). ``action_terms`` counts the bot decisions
    that had values."""

    session: str
    big_blind: int
    hands: list[int] = field(default_factory=list)
    raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    chance: np.ndarray = field(default_factory=lambda: np.zeros(0))
    aivat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    decisions: int = 0
    action_terms: int = 0

    def rates(self, confidence: float = 0.95, n_boot: int = 2000) -> dict[str, WinRate]:
        """The person's win rates (mbb/hand with bootstrap CIs) by estimator."""
        return {
            k: win_rate(getattr(self, k), self.big_blind, 1, confidence, n_boot, rng=0)
            for k in ("raw", "chance", "aivat")
        }

    def summary(self) -> str:
        n = len(self.raw)
        lines = [
            f"session {self.session}: {n} hands, {self.decisions} bot decisions, "
            f"{self.action_terms} with action terms"
        ]
        if n == 0:
            return lines[0]
        bb = self.big_blind
        for k, wr in self.rates().items():
            x = getattr(self, k)
            sd = x.std(ddof=1) / bb if n > 1 else float("nan")
            lines.append(
                f"  {k:>6}: person {wr.mbb_per_hand:+8.0f} mbb/hand, 95% CI "
                f"[{wr.ci_low:+.0f}, {wr.ci_high:+.0f}]; bot {-wr.mbb_per_hand:+.0f}; "
                f"SD {sd:.2f} bb/hand"
            )
        return "\n".join(lines)


def logged_decisions(record: dict, arrays: dict[str, np.ndarray]) -> list[dict]:
    """A finished hand's bot decisions rebuilt from its ``hands.jsonl`` line and
    its ``solves/hand_<n>.npz`` arrays (:mod:`pokerbot.play.log`)."""
    out = []
    for k, d in enumerate(record.get("bot_decisions") or []):
        out.append(
            {
                "taken": int(d["taken"]),
                "strategy": arrays[f"d{k}_strategy"],
                "values": arrays.get(f"d{k}_values"),
                "values_vs": arrays.get(f"d{k}_values_vs"),
            }
        )
    return out


def score_session(directory: str | Path, values: str = "pair") -> SessionScore:
    """AIVAT of a play session (``runs/play/<session>``) from its logs.

    The chance terms are the luck-adjusted results logged with each hand; the
    action terms are recomputed from the bot's logged strategies and values,
    so sessions logged before the play app scored AIVAT are scored too."""
    from ..play.log import SessionLog

    log = SessionLog(directory)
    big_blind = int((log.meta.get("game") or {}).get("big_blind", 100))
    score = SessionScore(str(log.meta.get("session", Path(directory).name)), big_blind)
    raw, chance, aivat = [], [], []
    for rec in log.hands():
        hs = int(rec["human_seat"])
        bot = 1 - hs
        arrays = log.solves(int(rec["hand"])) if rec.get("solves") else {}
        decisions = logged_decisions(rec, arrays)
        holes = rec["hole_cards"]
        corr = action_correction(decisions, holes[bot], holes[hs], values) if decisions else 0.0
        score.hands.append(int(rec["hand"]))
        raw.append(float(rec["human_net"]))
        chance.append(float(rec["human_adjusted"]))
        aivat.append(float(rec["human_adjusted"]) - corr)
        score.decisions += len(decisions)
        score.action_terms += sum(d["values"] is not None for d in decisions)
    score.raw, score.chance, score.aivat = (
        np.array(x, dtype=np.float64) for x in (raw, chance, aivat)
    )
    return score
