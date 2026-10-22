"""Scripted archetype opponents: stand-ins for "the average player".

Every archetype is a :class:`ProfileAgent` with its own frequencies, plus a
translation-stress agent built on :class:`~.baselines.RandomAgent`. They
exist to measure the bot against the kinds of mistakes people make
(roadmap item 1.3), so they are simple, fast on CPU and always legal: a
desired raise size is clamped into the legal range, and the agent checks
instead of folding whenever checking is free.

**Hand strength is a percentile.** Preflop, a hand's percentile is its rank
among the 1,326 combos by equity against a random hand (0 = aces, 1 = the
worst hand); "play the top 35%" means ``pct < 0.35``. Postflop, the hand's
Monte Carlo equity against a random hand (:func:`~.equity.monte_carlo_equity`,
the same helper as the ``equity`` baseline) is turned into a percentile with a
per-street table of the equities of hands from the agent's own preflop range
(its top ``vpip``) on random boards, so ``p`` is about uniform over the hands
it brings to the flop. A frequency target such as "fold 50% to c-bets" then
becomes "fold when ``p < 0.5``". Ranges narrow further as the hand goes on
and boards are not random, so measured frequencies drift from their targets; each agent counts
its own HUD statistics (:class:`Hud`), and the measured numbers are in
``pokerbot/eval/README.md`` (Archetype suite).

**Population defaults.** ``population`` is the average (losing) online player
and ``regular`` a competent one. Sources:

* VPIP/PFR: regulars about 19/17 to 25/23, losing players 30-40 VPIP with low
  PFR (Poker Copilot statistics guide, quoted in ``docs/roadmap.md`` 3.5).
* 3-bet: "a good 3-bet percentage will be around 5%-9%" at low stakes; fold
  to 3-bet "somewhere near 55%" balanced, higher at low stakes
  (pokercopilot.com/poker-statistics/three-betting).
* C-bet: "many players will have a continuation bet frequency of 40%-60%";
  fold to c-bet for good opponents "around 42%-57% at the lower stakes"
  (pokercopilot.com/poker-statistics/continuation-bet).
* Aggression: post-flop aggression frequency, (bets + raises) / (bets +
  raises + calls + folds), of 50-60% is called ideal for a 15-20% VPIP
  player (Poker Copilot user guide, HUD statistics); losing players sit
  below it.

These are six-max numbers and the game here is heads-up, where good players
play far more hands. They are kept as-is on purpose: the archetypes model
the population that would sit down against the bot, not heads-up experts.

Spec examples::

    population
    population:vpip=0.30,pfr=0.10,agg=0.3
    calling_station
    maniac:bet_size=1.0
    translation_stress:limp=0.5
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..engine_select import get_engine
from .base import CHECK_CALL, FOLD, RAISE, BaseAgent, clamp_raise, pot_raise_to
from .baselines import RandomAgent
from .equity import monte_carlo_equity

# --------------------------------------------------------------------------
# Hand strength percentiles


_PREFLOP_PCT: np.ndarray | None = None
_POSTFLOP_CDF: dict[tuple[int, float], np.ndarray] = {}


def _hand_class(c1: int, c2: int) -> tuple[int, int, bool]:
    """(high rank, low rank, suited) of two cards encoded ``rank * 4 + suit``."""
    r1, r2 = c1 // 4, c2 // 4
    hi, lo = max(r1, r2), min(r1, r2)
    return hi, lo, (c1 % 4 == c2 % 4) and hi != lo


def preflop_percentiles(samples: int = 6000, seed: int = 1) -> np.ndarray:
    """``(52, 52)`` table of preflop percentiles (0 best, 1 worst), from each
    of the 169 hand classes' equity against a random hand. A class sits at
    the midpoint of the combos it covers. Computed once per process (about a
    second with the Rust evaluator)."""
    global _PREFLOP_PCT
    if _PREFLOP_PCT is not None:
        return _PREFLOP_PCT
    rng = np.random.default_rng(seed)
    evaluate = get_engine().evaluate_batch
    classes: list[tuple[float, int, tuple[int, int, bool]]] = []
    for hi in range(13):
        for lo in range(hi + 1):
            for suited in (False, True) if hi != lo else (False,):
                cards = [hi * 4, lo * 4 + (0 if suited else 1)]
                eq = monte_carlo_equity(cards, [], 1, samples, rng, evaluate)
                combos = 6 if hi == lo else (4 if suited else 12)
                classes.append((eq, combos, (hi, lo, suited)))
    classes.sort(key=lambda c: -c[0])
    mid: dict[tuple[int, int, bool], float] = {}
    above = 0
    for _, combos, key in classes:
        mid[key] = (above + combos / 2) / 1326.0
        above += combos
    table = np.ones((52, 52))
    for a in range(52):
        for b in range(52):
            if a != b:
                table[a, b] = mid[_hand_class(a, b)]
    _PREFLOP_PCT = table
    return table


def postflop_cdf(
    street: int, top: float = 1.0, deals: int = 2000, samples: int = 200, seed: int = 2
) -> np.ndarray:
    """Sorted equities against a random hand of ``deals`` hands on random
    boards of ``street`` (1 flop, 2 turn, 3 river). The hands are drawn from
    the top ``top`` of preflop hands (1.0: any two cards), so a percentile in
    this table ranks a hand within a range that played to this street.
    Cached per (street, top rounded to 0.01)."""
    top = round(float(np.clip(top, 0.02, 1.0)), 2)
    key = (street, top)
    if key in _POSTFLOP_CDF:
        return _POSTFLOP_CDF[key]
    rng = np.random.default_rng((seed, street, int(top * 100)))
    evaluate = get_engine().evaluate_batch
    pre = preflop_percentiles()
    n_board = 2 + street
    out = np.empty(deals)
    for i in range(deals):
        while True:
            cards = rng.permutation(52)[: 2 + n_board]
            if pre[cards[0], cards[1]] < top:
                break
        out[i] = monte_carlo_equity(cards[:2], cards[2:], 1, samples, rng, evaluate)
    out.sort()
    _POSTFLOP_CDF[key] = out
    return out


# --------------------------------------------------------------------------
# Reading the betting from the public history


def _is_raise(action: Any) -> bool:
    return int(action.kind) == RAISE


@dataclass
class Spot:
    """What a scripted agent needs to know about the current decision."""

    street: int
    to_call: int
    pot: int
    raises: int  # raises (bets count) on this street so far
    first: bool  # no one has acted on this street yet
    acted: bool  # this seat already acted on this street
    limped: bool  # this seat (the button) completed the small blind preflop
    preflop_aggressor: int  # seat of the last preflop raiser, -1 if none

    @property
    def facing_fraction(self) -> float:
        """The bet faced, as a fraction of the pot before it."""
        return self.to_call / max(1, self.pot - self.to_call)


def read_spot(state: Any, seat: int) -> Spot:
    street = state.street
    raises = actions = 0
    acted = limped = False
    agg = -1
    pre_raises = 0
    for st, player, action in state.history:
        raised = _is_raise(action)
        if st == 0:
            if player == seat and player == state.button and pre_raises == 0 and not raised:
                limped = True
            if raised:
                agg = player
                pre_raises += 1
        if st == street:
            actions += 1
            raises += raised
            acted = acted or player == seat
    bets = state.street_bets
    return Spot(
        street=street,
        to_call=max(bets) - bets[seat],
        pot=state.pot,
        raises=raises,
        first=actions == 0,
        acted=acted,
        limped=limped,
        preflop_aggressor=agg,
    )


# --------------------------------------------------------------------------
# HUD statistics


@dataclass
class Hud:
    """Counts of the standard HUD statistics for one agent, filled in from
    the public history at the end of each hand (heads-up):

    * VPIP / PFR: hands with a voluntary call or raise / a raise preflop;
    * 3-bet: raised when facing exactly one preflop raise;
    * c-bet: bet the flop as the preflop raiser when no one had bet yet;
    * fold to c-bet: folded to the preflop raiser's first flop bet;
    * Agg: postflop (bets + raises) / (bets + raises + calls + folds).

    ``opp_actions[street]`` counts the opponent's decisions per street (to
    estimate the cost of a match against an expensive agent).
    """

    hands: int = 0
    vpip: int = 0
    pfr: int = 0
    three_bet_opp: int = 0
    three_bet: int = 0
    cbet_opp: int = 0
    cbet: int = 0
    fold_cbet_opp: int = 0
    fold_cbet: int = 0
    post_aggr: int = 0
    post_passive: int = 0
    opp_actions: list[int] = field(default_factory=lambda: [0, 0, 0, 0])

    def record(self, history: list, seat: int, button: int) -> None:
        self.hands += 1
        vp = pf = False
        street = -1
        facing_seat: int | None = None  # the seat facing a bet, if any
        street_raises = 0
        pre_agg = -1
        for st, player, action in history:
            if st != street:
                street, street_raises = st, 0
                # preflop the button (small blind) faces the big blind
                facing_seat = button if st == 0 else None
            kind = int(action.kind)
            raised, folded = kind == RAISE, kind == FOLD
            facing = facing_seat == player
            if player != seat:
                self.opp_actions[st] += 1
            if player == seat:
                if st == 0:
                    vp = vp or raised or (facing and kind == CHECK_CALL)
                    pf = pf or raised
                    if street_raises == 1:
                        self.three_bet_opp += 1
                        self.three_bet += raised
                else:
                    if raised:
                        self.post_aggr += 1
                    elif folded or facing:
                        self.post_passive += 1
                    if st == 1 and street_raises == 0 and pre_agg == seat:
                        self.cbet_opp += 1
                        self.cbet += raised
                    if st == 1 and street_raises == 1 and facing and pre_agg not in (-1, seat):
                        self.fold_cbet_opp += 1
                        self.fold_cbet += folded
            if raised:
                street_raises += 1
                if st == 0:
                    pre_agg = player
                facing_seat = 1 - player
            elif facing:
                facing_seat = None
        self.vpip += vp
        self.pfr += pf

    def summary(self) -> dict[str, float]:
        def rate(a: int, b: int) -> float:
            return round(a / b, 3) if b else float("nan")

        return {
            "hands": self.hands,
            "vpip": rate(self.vpip, self.hands),
            "pfr": rate(self.pfr, self.hands),
            "three_bet": rate(self.three_bet, self.three_bet_opp),
            "cbet": rate(self.cbet, self.cbet_opp),
            "fold_to_cbet": rate(self.fold_cbet, self.fold_cbet_opp),
            "agg": rate(self.post_aggr, self.post_aggr + self.post_passive),
        }

    def merge(self, counts: dict[str, Any]) -> None:
        """Add counts saved by :meth:`counts` (e.g. from an earlier chunk)."""
        for k, v in counts.items():
            if k == "opp_actions":
                self.opp_actions = [a + b for a, b in zip(self.opp_actions, v, strict=True)]
            else:
                setattr(self, k, getattr(self, k) + v)

    def counts(self) -> dict[str, Any]:
        return {k: (list(v) if isinstance(v, list) else v) for k, v in vars(self).items()}


# --------------------------------------------------------------------------
# The profile agent


class ProfileAgent(BaseAgent):
    """A frequency-driven scripted player.

    Preflop (heads-up; the button posts the small blind and acts first):

    * small blind first in: raise the top ``pfr``, limp the rest of the top
      ``vpip``, fold the others;
    * big blind after a limp: raise the top ``pfr`` (isolation), check the rest;
    * facing an open (or an isolation raise after limping): reraise the top
      ``three_bet``, call the rest of the top ``vpip``, fold;
    * facing a reraise of its own raise: continue with ``1 - fold_to_3bet``
      of the range it raised with, the top ``four_bet`` of all hands
      reraising again (all-in from the fourth raise on).

    Postflop, with ``p`` the hand's strength percentile on this street:

    * no bet yet: the preflop raiser bets the flop with frequency ``cbet``,
      otherwise the agent bets with frequency ``agg`` (scaled by ``donk`` when
      leading into the preflop raiser). A share ``bluff_share`` of those bets
      are bluffs: random hands, or the weakest hands when ``polar``. The
      preflop raiser ranks its hand within the range it raised with, the
      caller within its ``vpip``. Thresholds are absolute within that range,
      so a hand that checks and then faces a bet comes from a range without
      its strongest hands (keep ``donk`` low for realistic fold rates);
    * facing a bet: fold the weakest ``fold_to_cbet`` (a flop c-bet) or
      ``fold_to_bet`` (any other bet), halving the continuing share for each
      extra raise on the street. Bigger bets fold more: the continuing share
      scales with ``(0.66 / bet fraction) ** size_sensitivity``. The top
      ``raise_freq`` of continuing hands raise.

    Sizes: preflop raise-to ``open_size`` big blinds, reraises
    ``three_bet_x`` times the raise faced; postflop bets and raises are
    ``bet_size`` pot fractions. ``sizing="min"`` makes every raise a
    min-raise; ``sizing="overbet"`` draws postflop sizes uniformly from
    ``overbet_range`` pot fractions.
    """

    name = "profile"

    def __init__(
        self,
        vpip: float = 0.35,
        pfr: float = 0.12,
        three_bet: float = 0.04,
        four_bet: float = 0.015,
        fold_to_3bet: float = 0.60,
        cbet: float = 0.55,
        fold_to_cbet: float = 0.50,
        fold_to_bet: float = 0.50,
        agg: float = 0.35,
        raise_freq: float = 0.10,
        bluff_share: float = 0.35,
        polar: bool = False,
        donk: float = 0.3,
        size_sensitivity: float = 0.5,
        open_size: float = 2.5,
        three_bet_x: float = 3.0,
        bet_size: float = 0.66,
        sizing: str = "fixed",
        overbet_range: tuple[float, float] = (1.5, 2.5),
        samples: int = 128,
        name: str | None = None,
    ) -> None:
        super().__init__(name)
        if sizing not in ("fixed", "min", "overbet"):
            raise ValueError(f"sizing must be fixed, min or overbet, not {sizing!r}")
        self.vpip = vpip
        self.pfr = min(pfr, vpip)
        self.three_bet = three_bet
        self.four_bet = four_bet
        self.fold_to_3bet = fold_to_3bet
        self.cbet = cbet
        self.fold_to_cbet = fold_to_cbet
        self.fold_to_bet = fold_to_bet
        self.agg = agg
        self.raise_freq = raise_freq
        self.bluff_share = bluff_share
        self.polar = polar
        self.donk = donk
        self.size_sensitivity = size_sensitivity
        self.open_size = open_size
        self.three_bet_x = three_bet_x
        self.bet_size = bet_size
        self.sizing = sizing
        self.overbet_range = tuple(overbet_range)
        self.samples = samples
        self.hud = Hud()
        self._range = 1.0  # the top fraction of hands this agent last raised with
        self._pre = preflop_percentiles()

    def new_hand(self, seat: int, config: Any) -> None:
        super().new_hand(seat, config)
        self._range = 1.0

    def observe_end(self, state: Any) -> None:
        self.hud.record(list(state.history), self.seat, state.button)

    # --- strength ---

    def preflop_pct(self, state: Any, seat: int) -> float:
        a, b = state.hole_cards(seat)
        return float(self._pre[int(a), int(b)])

    def strength(
        self, state: Any, seat: int, rng: np.random.Generator, top: float | None = None
    ) -> float:
        """Postflop strength percentile (0 weakest, 1 strongest) within the
        top ``top`` of preflop hands (default: the hands it plays, ``vpip``)."""
        eq = monte_carlo_equity(
            state.hole_cards(seat), state.board, 1, self.samples, rng, self._engine.evaluate_batch
        )
        cdf = postflop_cdf(state.street, self.vpip if top is None else top)
        return float(np.searchsorted(cdf, eq, side="right")) / len(cdf)

    # --- sizes ---

    def _raise(self, state: Any, seat: int, legal: Any, target: int) -> Any:
        if self.sizing == "min":
            target = legal.min_raise_to
        amt = clamp_raise(legal, target)
        return self.check_call() if amt is None else self.raise_to(amt)

    def _post_size(self, state: Any, seat: int, rng: np.random.Generator) -> int:
        frac = self.bet_size
        if self.sizing == "overbet":
            frac = float(rng.uniform(*self.overbet_range))
        return pot_raise_to(state, seat, frac)

    def _fold_or_check(self, legal: Any) -> Any:
        return self.check_call() if legal.can_check else self.fold()

    # --- decisions ---

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        spot = read_spot(state, seat)
        if spot.street == 0:
            return self._preflop(state, seat, legal, spot)
        return self._postflop(state, seat, legal, spot, rng)

    def _preflop(self, state: Any, seat: int, legal: Any, spot: Spot) -> Any:
        pct = self.preflop_pct(state, seat)
        bb = self.config.big_blind if self.config is not None else 100
        cur = max(state.street_bets)
        if spot.raises == 0:
            if pct < self.pfr:
                self._range = self.pfr
                size = self.open_size + (1.0 if legal.can_check else 0.0)  # isolate a limp
                return self._raise(state, seat, legal, int(round(size * bb)))
            if legal.can_check or pct < self.vpip:
                return self.check_call()
            return self.fold()
        if spot.raises == 1 and not (spot.acted and not spot.limped):
            # facing an open, or an isolation raise after limping
            if pct < self.three_bet:
                self._range = self.three_bet
                return self._raise(state, seat, legal, int(self.three_bet_x * cur))
            limit = self.vpip if not spot.limped else self.vpip * (1.0 - self.fold_to_3bet / 2)
            return self.check_call() if pct < limit else self._fold_or_check(legal)
        # facing a reraise of our raise (or a later raise)
        if pct < self.four_bet:
            self._range = self.four_bet
            target = int(self.three_bet_x * 0.8 * cur) if spot.raises < 3 else legal.max_raise_to
            return self._raise(state, seat, legal, target)
        keep = self._range * (1.0 - self.fold_to_3bet)
        return self.check_call() if pct < keep else self._fold_or_check(legal)

    def _postflop(
        self, state: Any, seat: int, legal: Any, spot: Spot, rng: np.random.Generator
    ) -> Any:
        # the preflop raiser ranks its hand within the range it raised with
        top = self._range if spot.preflop_aggressor == seat else self.vpip
        p = self.strength(state, seat, rng, top)
        if spot.to_call == 0:
            if spot.street == 1 and spot.raises == 0 and spot.preflop_aggressor == seat:
                freq = self.cbet
            else:
                freq = self.agg
                if spot.first and spot.preflop_aggressor not in (-1, seat):
                    freq *= self.donk  # leading into the preflop raiser
            if self._bets(p, freq, rng):
                return self._raise(state, seat, legal, self._post_size(state, seat, rng))
            return self.check_call()
        cbet = spot.street == 1 and spot.raises == 1 and spot.preflop_aggressor not in (-1, seat)
        fold = self.fold_to_cbet if cbet else self.fold_to_bet
        cont = (1.0 - fold) * 0.5 ** max(0, spot.raises - 1)
        frac = max(spot.facing_fraction, 1e-3)
        cont = float(np.clip(cont * (0.66 / frac) ** self.size_sensitivity, 0.02, 1.0))
        if p >= 1.0 - cont:
            if legal.min_raise_to > 0 and p >= 1.0 - cont * self.raise_freq:
                return self._raise(state, seat, legal, self._post_size(state, seat, rng))
            return self.check_call()
        bluff = self.raise_freq * self.bluff_share * 0.5
        if legal.min_raise_to > 0 and self._bluffs(p, bluff, 1.0 - cont, rng):
            return self._raise(state, seat, legal, self._post_size(state, seat, rng))
        return self._fold_or_check(legal)

    def _bets(self, p: float, freq: float, rng: np.random.Generator) -> bool:
        """Bet with overall frequency ``freq``: the top ``freq * (1 -
        bluff_share)`` for value, plus ``freq * bluff_share`` of bluffs."""
        value = freq * (1.0 - self.bluff_share)
        if p >= 1.0 - value:
            return True
        return self._bluffs(p, freq * self.bluff_share, 1.0 - value, rng)

    def _bluffs(self, p: float, share: float, below: float, rng: np.random.Generator) -> bool:
        """Bluff with ``share`` of all hands, chosen among hands under ``below``:
        the weakest ones when ``polar``, otherwise at random."""
        if share <= 0 or below <= 0:
            return False
        if self.polar:
            return p < share
        return bool(rng.random() < share / below)


# --------------------------------------------------------------------------
# Named archetypes


def _preset(defaults: dict[str, Any], name: str) -> Callable[..., ProfileAgent]:
    def factory(**kwargs: Any) -> ProfileAgent:
        kwargs.setdefault("name", name)
        return ProfileAgent(**{**defaults, **kwargs})

    factory.__name__ = name
    factory.__doc__ = f"``{name}`` archetype: ProfileAgent with {defaults}."
    return factory


ARCHETYPE_DEFAULTS: dict[str, dict[str, Any]] = {
    # The average (losing) online player: loose-passive preflop, middling
    # postflop; see the module docstring for the sources.
    "population": dict(
        vpip=0.35,
        pfr=0.12,
        three_bet=0.04,
        fold_to_3bet=0.62,
        cbet=0.55,
        fold_to_cbet=0.50,
        fold_to_bet=0.50,
        agg=0.35,
    ),
    # A competent six-max regular.
    "regular": dict(
        vpip=0.23,
        pfr=0.20,
        three_bet=0.07,
        four_bet=0.025,
        fold_to_3bet=0.55,
        cbet=0.60,
        fold_to_cbet=0.48,
        fold_to_bet=0.48,
        agg=0.50,
        raise_freq=0.15,
    ),
    # Calls almost everything, rarely raises, bets only strong hands.
    "calling_station": dict(
        vpip=0.65,
        pfr=0.05,
        three_bet=0.01,
        fold_to_3bet=0.15,
        cbet=0.30,
        fold_to_cbet=0.10,
        fold_to_bet=0.15,
        agg=0.12,
        raise_freq=0.05,
        bluff_share=0.0,
        donk=0.5,
        size_sensitivity=0.0,
    ),
    # Plays only strong hands and gives up without them.
    "nit": dict(
        vpip=0.14,
        pfr=0.12,
        three_bet=0.03,
        four_bet=0.012,
        fold_to_3bet=0.65,
        cbet=0.50,
        fold_to_cbet=0.62,
        fold_to_bet=0.65,
        agg=0.30,
        bluff_share=0.05,
        size_sensitivity=0.7,
    ),
    # Raises and bets nearly everything, big.
    "maniac": dict(
        vpip=0.85,
        pfr=0.70,
        three_bet=0.35,
        four_bet=0.15,
        fold_to_3bet=0.25,
        cbet=0.90,
        fold_to_cbet=0.20,
        fold_to_bet=0.25,
        agg=0.75,
        raise_freq=0.40,
        bluff_share=0.70,
        donk=1.0,
        size_sensitivity=0.1,
        open_size=3.5,
        three_bet_x=3.5,
        bet_size=1.0,
    ),
    # Limps a wide range, calls raises, rarely bets.
    "loose_passive": dict(
        vpip=0.55,
        pfr=0.03,
        three_bet=0.01,
        fold_to_3bet=0.40,
        cbet=0.30,
        fold_to_cbet=0.40,
        fold_to_bet=0.40,
        agg=0.15,
        raise_freq=0.05,
        bluff_share=0.1,
        donk=0.5,
        size_sensitivity=0.3,
    ),
    # Plays a normal range preflop, then bets or continues only with a fit;
    # never bluffs.
    "fit_or_fold": dict(
        vpip=0.40,
        pfr=0.15,
        three_bet=0.04,
        fold_to_3bet=0.55,
        cbet=0.40,
        fold_to_cbet=0.65,
        fold_to_bet=0.65,
        agg=0.35,
        raise_freq=0.15,
        bluff_share=0.0,
        donk=0.3,
        size_sensitivity=0.0,
    ),
    # Every raise is a min-raise, preflop and postflop.
    "min_raiser": dict(
        vpip=0.45,
        pfr=0.30,
        three_bet=0.10,
        four_bet=0.04,
        fold_to_3bet=0.45,
        cbet=0.70,
        fold_to_cbet=0.45,
        fold_to_bet=0.45,
        agg=0.50,
        raise_freq=0.25,
        bluff_share=0.4,
        sizing="min",
    ),
    # Polarised overbets: the strongest hands and the weakest bluff 1.5-2.5x
    # pot; the middle checks and calls.
    "overbet_bluffer": dict(
        vpip=0.40,
        pfr=0.25,
        three_bet=0.06,
        fold_to_3bet=0.50,
        cbet=0.60,
        fold_to_cbet=0.45,
        fold_to_bet=0.45,
        agg=0.45,
        raise_freq=0.15,
        bluff_share=0.5,
        polar=True,
        sizing="overbet",
    ),
}

ARCHETYPES: dict[str, Callable[..., Any]] = {
    name: _preset(defaults, name) for name, defaults in ARCHETYPE_DEFAULTS.items()
}


class TranslationStressAgent(RandomAgent):
    """Random legal play shaped to hit the bot's action-translation and
    off-tree code: :class:`RandomAgent` with structure.

    * Sizes: every raise is drawn from a mixture of the min-raise, all-in,
      uniform over the legal range (``RandomAgent``'s draw) and log-uniform
      between the min-raise and all-in, so small, odd and huge sizes all come
      up on every street.
    * Preflop small blind: limps with probability ``limp``, otherwise opens to
      an odd size (uniform 2-12 big blinds, a shove with probability
      ``shove``), and limp-reraises with probability ``limp_reraise`` when
      raised after limping.
    * Postflop: leads (donk bets) into the preflop raiser with probability
      ``donk``.
    * Facing a bet it folds the weakest ``fold_below`` of hands (strength
      percentile, as in :class:`ProfileAgent`) so that hands reach later
      streets; other decisions are ``RandomAgent``'s weighted fold / call /
      raise with call and raise weights ``call_weight``, ``raise_weight``.
    """

    name = "translation_stress"

    def __init__(
        self,
        limp: float = 0.35,
        shove: float = 0.05,
        limp_reraise: float = 0.5,
        donk: float = 0.5,
        fold_below: float = 0.25,
        call_weight: float = 1.0,
        raise_weight: float = 1.0,
        samples: int = 64,
        name: str | None = None,
    ) -> None:
        super().__init__(0.0, call_weight, raise_weight, name)
        self.limp = limp
        self.shove = shove
        self.limp_reraise = limp_reraise
        self.donk = donk
        self.fold_below = fold_below
        self.samples = samples
        self.hud = Hud()

    def observe_end(self, state: Any) -> None:
        self.hud.record(list(state.history), self.seat, state.button)

    def stress_size(self, legal: Any, rng: np.random.Generator) -> int:
        lo, hi = legal.min_raise_to, legal.max_raise_to
        if hi <= lo:
            return hi
        u = rng.random()
        if u < 0.2:
            return lo
        if u < 0.3:
            return hi
        if u < 0.65:
            return int(rng.integers(lo, hi + 1))
        return int(np.clip(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))), lo, hi))

    def act(self, state: Any, seat: int, rng: np.random.Generator) -> Any:
        legal = state.legal_actions()
        spot = read_spot(state, seat)
        can_raise = legal.min_raise_to > 0
        bb = self.config.big_blind if self.config is not None else 100
        if spot.street == 0 and seat == state.button and can_raise:
            if spot.raises == 0 and not spot.acted:
                u = rng.random()
                if u < self.limp:
                    return self.check_call()
                if u < self.limp + self.shove:
                    return self.raise_to(legal.max_raise_to)
                amt = clamp_raise(legal, int(rng.uniform(2.0, 12.0) * bb))
                return self.raise_to(amt) if amt is not None else self.check_call()
            if spot.limped and spot.raises == 1 and rng.random() < self.limp_reraise:
                return self.raise_to(self.stress_size(legal, rng))
        if spot.street > 0:
            if (
                spot.first
                and spot.preflop_aggressor not in (-1, seat)
                and can_raise
                and rng.random() < self.donk
            ):
                return self.raise_to(self.stress_size(legal, rng))
            if spot.to_call > 0 and self.fold_below > 0:
                eq = monte_carlo_equity(
                    state.hole_cards(seat),
                    state.board,
                    1,
                    self.samples,
                    rng,
                    self._engine.evaluate_batch,
                )
                cdf = postflop_cdf(state.street)
                if np.searchsorted(cdf, eq) / len(cdf) < self.fold_below:
                    return self.fold()
        action = super().act(state, seat, rng)
        if int(action.kind) == RAISE and rng.random() < 0.5:
            return self.raise_to(self.stress_size(legal, rng))
        return action


ARCHETYPES["translation_stress"] = TranslationStressAgent

# The suite's default list: the named archetypes, the translation-stress
# agent and the existing ``equity`` baseline.
SUITE: list[str] = [
    "population",
    "regular",
    "calling_station",
    "nit",
    "maniac",
    "loose_passive",
    "fit_or_fold",
    "min_raiser",
    "overbet_bluffer",
    "translation_stress",
    "equity",
]
