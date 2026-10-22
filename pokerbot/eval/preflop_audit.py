"""Preflop off-tree audit: how a bot answers every preflop size, on its tree or not.

The search plays preflop from its blueprint (for production, the solved table
``preflop_r15.pt`` over the Deep CFR blueprint). The table covers 31 flop starts:
opens of 2, 2.25, 2.5, 3 and 4bb and all-in, 3-bets and 4-bets of 3x, limp-raises
of the same sizes. Any other size is played by the base blueprint, and the
search later rebuilds both ranges through pseudo-harmonic translation
(:func:`~pokerbot.search.blueprint.range_reach`). So odd preflop sizes are where
a person would look for a hole first (roadmap 1.5).

The audit puts the bot in front of a grid of sizes (:func:`audit_spots`):

* ``open``: the big blind facing every open from a min-raise to all-in;
* ``3bet``: the button facing 3-bets after each of its own on-tree opens;
* ``4bet``: the big blind facing 4-bets after its own on-tree 3-bet;
* ``limp_raise``: the button facing a raise after it limped;
* ``limp_reraise``: the big blind facing a re-raise after it raised a limp.

At each spot it asks the agent for one decision with ``record=True``
(:class:`~pokerbot.search.agent.SearchAgent` keeps the strategy of all 1326 combos
and both players' reach) and reports the fold, call and raise frequencies over
the bot's range there (the strategy weighted by its own reach).

**Reference: minimum defence.** A raise that adds ``bet`` chips to a pot of
``pot`` (before the raise) wins at once when the bot folds; it profits with any
two cards if the bot folds more than ``bet / (pot + bet)``. That is the most the
bot may fold (``max_fold``). It ignores the raiser's equity when called, so it is
a loose bar: folding above it is a clear leak, folding below it need not be
right. Two more flags catch translation artifacts in a series of sizes: folding
much *less* against a bigger raise than against the next smaller one
(``non_monotone``), and a jump of more than ``JUMP`` in the fold frequency between
neighbouring sizes at most ``JUMP_RATIO`` apart (``jump``).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType
from typing import Any

import numpy as np

from ..engine_select import get_engine
from ..search.abstract import CHECK_CALL, FOLD, RAISE, legal_options, map_concrete, to_action
from .masking import MaskedState

# raise-to sizes in big blinds (the button's open, the big blind's raise of a limp)
OPEN_BB = (2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
LIMP_RAISE_BB = (2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0)
# the bot's own on-tree first raises the re-raise series start from
TABLE_OPENS_BB = (2.0, 2.25, 2.5, 3.0, 4.0)
TABLE_LIMP_RAISES_BB = (2.0, 3.0, 4.0)
# re-raises as multiples of the raise they face ("min": a min-raise); the tree has 3x
RERAISE_X: tuple = ("min", 2.5, 3.0, 3.5, 4.0, 5.0)
FOURBET_X: tuple = ("min", 2.0, 2.5, 3.0, 4.0)
ON_TREE_RERAISE_X = 3.0
ALLIN = "allin"

OVERFOLD_TOL = 0.005  # fold above max_fold by more than this: "overfold"
NON_MONOTONE = 0.10  # fold this much less against a bigger raise: "non_monotone"
JUMP = 0.25  # fold frequency change between neighbouring sizes: "jump"
JUMP_RATIO = 1.5  # ... at most this far apart (an all-in may well be folded to far more)


@dataclass(frozen=True)
class Spot:
    """One audited decision. ``prefix`` is the preflop action sequence from the
    deal as ``(kind, raise-to)`` pairs (the button, seat 0, acts first); the bot
    decides after it, as the button (``bot_button``) or the big blind. ``series``
    groups the sizes of one line (one row of plots); ``size`` labels this one."""

    family: str
    series: str
    size: str
    prefix: tuple[tuple[int, int], ...]
    bot_button: bool

    @property
    def bot_seat(self) -> int:
        return 0 if self.bot_button else 1


def _bb(x: float, big_blind: int) -> int:
    return int(round(x * big_blind))


def _fmt_bb(chips: int, big_blind: int) -> str:
    return f"{chips / big_blind:g}bb"


def _sizes(base: int, mults: tuple, min_to: int, stack: int) -> list[tuple[str, int]]:
    """Raise-to amounts ``base * m`` (``"min"``: ``min_to``) below ``stack``, plus
    all-in, without duplicates, labelled ``"min"``, ``"2.5x"``, ..., ``"allin"``."""
    out: list[tuple[str, int]] = []
    seen: set[int] = set()
    for m in mults:
        amt = min_to if m == "min" else max(min_to, int(round(base * m)))
        if amt >= stack or amt in seen:
            continue
        seen.add(amt)
        out.append(("min" if m == "min" else f"{m:g}x", amt))
    out.append((ALLIN, stack))
    return out


def audit_spots(config: Any) -> list[Spot]:
    """The audit's grid for a heads-up game with equal stacks (see the module doc)."""
    bb = int(config.big_blind)
    stack = int(config.stacks[0])
    R = RAISE
    spots: list[Spot] = []

    def add(family: str, series: str, size: str, prefix: list, bot_button: bool) -> None:
        spots.append(Spot(family, series, size, tuple(prefix), bot_button))

    # the big blind facing an open
    seen: set[int] = set()
    for x in OPEN_BB:
        amt = _bb(x, bb)
        if amt < stack and amt not in seen:
            seen.add(amt)
            add("open", "vs open", f"{x:g}bb", [(R, amt)], False)
    add("open", "vs open", ALLIN, [(R, stack)], False)
    # the button facing a 3-bet after its own open
    for o in TABLE_OPENS_BB:
        op = _bb(o, bb)
        for label, amt in _sizes(op, RERAISE_X, 2 * op - bb, stack):
            add("3bet", f"open {o:g}bb", label, [(R, op), (R, amt)], True)
    # the big blind facing a 4-bet after its own 3-bet
    for o in TABLE_OPENS_BB:
        op = _bb(o, bb)
        tb = int(round(op * ON_TREE_RERAISE_X))
        if tb >= stack:
            continue
        for label, amt in _sizes(tb, FOURBET_X, 2 * tb - op, stack):
            add(
                "4bet",
                f"open {o:g}bb, 3-bet {tb / bb:g}bb",
                label,
                [(R, op), (R, tb), (R, amt)],
                False,
            )
    # the button facing a raise after its limp
    seen = set()
    for x in LIMP_RAISE_BB:
        amt = _bb(x, bb)
        if amt < stack and amt not in seen:
            seen.add(amt)
            add("limp_raise", "limp", f"{x:g}bb", [(CHECK_CALL, 0), (R, amt)], True)
    add("limp_raise", "limp", ALLIN, [(CHECK_CALL, 0), (R, stack)], True)
    # the big blind facing a re-raise after it raised a limp
    for r in TABLE_LIMP_RAISES_BB:
        ra = _bb(r, bb)
        for label, amt in _sizes(ra, RERAISE_X, 2 * ra - bb, stack):
            add(
                "limp_reraise",
                f"limp, raise {r:g}bb",
                label,
                [(CHECK_CALL, 0), (R, ra), (R, amt)],
                False,
            )
    return spots


def _replay(engine: ModuleType, config: Any, prefix: Any) -> Any:
    """The state after ``prefix`` from a deal with the button in seat 0. The cards
    never matter: the record covers every combo."""
    state = engine.GameState.new_hand(config, 0, list(range(52)))
    for kind, amount in prefix:
        state.apply(to_action(engine, kind, amount))
    return state


def _on_table(blueprint: Any, state: Any) -> bool | None:
    fn = getattr(blueprint, "on_table", None)
    return None if fn is None else bool(fn(state))


def audit_decision(
    agent: Any,
    config: Any,
    spot: Spot,
    engine: ModuleType | None = None,
    seed: int = 0,
) -> dict:
    """The agent's decision at ``spot``: its frequencies over its range there
    (reach-weighted), the minimum-defence bar and where the translation puts the
    raise it faces. ``agent`` must keep ``decisions`` with ``record`` on."""
    engine = engine or get_engine()
    before = _replay(engine, config, spot.prefix[:-1])
    state = _replay(engine, config, spot.prefix)
    pot_before = int(before.pot)
    if state.is_terminal or int(state.street) != 0 or state.current_player != spot.bot_seat:
        raise ValueError(f"spot {spot} does not end at a preflop decision of the bot")
    bet = int(state.pot) - pot_before
    seat = spot.bot_seat
    rng = np.random.default_rng(seed)
    agent.new_hand(seat, config)
    view = MaskedState(state, seat, config, rng)
    agent.act(view, seat, rng)
    rec = agent.decisions[-1]
    strat = np.asarray(rec["strategy"], dtype=np.float64)  # [C, A]
    reach = np.asarray(rec["reach"], dtype=np.float64)[seat]  # [C]
    w = reach / max(reach.sum(), 1e-30)
    freq = w @ strat  # [A]
    acts = rec["actions"]
    allin_to = int(state.legal_actions().max_raise_to)
    fold = call = raise_ = allin = 0.0
    raises: dict[str, float] = {}
    for f, (k, a) in zip(freq.tolist(), acts, strict=True):
        if k == FOLD:
            fold += f
        elif k == CHECK_CALL:
            call += f
        else:
            raise_ += f
            allin += f if a == allin_to else 0.0
            if f >= 0.005:
                raises[_fmt_bb(int(a), config.big_blind)] = round(f, 4)
    bp = getattr(agent, "blueprint", None)
    translation = None
    if bp is not None:  # where the blueprint's abstraction puts the raise faced
        kind, amount = spot.prefix[-1]
        opts = {o.index: o for o in legal_options(before, bp.spec)}
        m = map_concrete(before, bp.spec, kind, amount, list(opts.values()))
        translation = {
            _fmt_bb(opts[i].amount, config.big_blind): round(p, 3)
            for i, p in sorted(m.items(), key=lambda kv: opts[kv[0]].amount)
        }
    on_table = _on_table(bp, state) if bp is not None else None
    return {
        "family": spot.family,
        "series": spot.series,
        "size": spot.size,
        "raise_to": int(spot.prefix[-1][1]),
        "prefix": [list(a) for a in spot.prefix],
        "bot": "button" if spot.bot_button else "big blind",
        "source": "table" if on_table else ("blueprint" if on_table is False else rec["source"]),
        "translation": translation,
        "range": float(reach.sum() / len(reach)),  # share of all deals' hands that get here
        "fold": fold,
        "call": call,
        "raise": raise_,
        "allin": allin,
        "raises": raises,
        "bet": bet,
        "pot_before": pot_before,
        "max_fold": bet / (pot_before + bet),
    }


def flag_rows(rows: list[dict]) -> list[dict]:
    """Add ``flags`` to every row (in place, also returned): ``overfold`` against
    the minimum-defence bar; ``non_monotone`` and ``jump`` against the previous
    (smaller) size of the same series."""
    prev: dict[tuple, dict] = {}
    for r in rows:
        flags = []
        if r["fold"] > r["max_fold"] + OVERFOLD_TOL:
            flags.append(f"overfold +{100 * (r['fold'] - r['max_fold']):.0f}")
        key = (r["family"], r["series"])
        p = prev.get(key)
        if p is not None:
            d = r["fold"] - p["fold"]
            if d < -NON_MONOTONE:
                flags.append(f"non_monotone {100 * d:+.0f} vs {p['size']}")
            if abs(d) > JUMP and r["raise_to"] <= JUMP_RATIO * p["raise_to"]:
                flags.append(f"jump {100 * d:+.0f} vs {p['size']}")
        prev[key] = r
        r["flags"] = flags
    return rows


def run_audit(
    agent: Any,
    config: Any,
    spots: list[Spot] | None = None,
    engine: ModuleType | None = None,
    seed: int = 0,
) -> list[dict]:
    """Audit every spot (default: :func:`audit_spots`) and flag the rows. Turns on
    the agent's ``record`` for the run."""
    engine = engine or get_engine()
    spots = audit_spots(config) if spots is None else spots
    old = getattr(agent, "record", False)
    agent.record = True
    try:
        rows = [audit_decision(agent, config, s, engine, seed) for s in spots]
    finally:
        agent.record = old
    return flag_rows(rows)


FAMILY_TITLES = {
    "open": "Big blind facing an open",
    "3bet": "Button facing a 3-bet after its open",
    "4bet": "Big blind facing a 4-bet after its 3-bet",
    "limp_raise": "Button facing a raise after its limp",
    "limp_reraise": "Big blind facing a re-raise after it raised a limp",
}


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def audit_markdown(rows: list[dict], big_blind: int = 100) -> str:
    """Markdown tables, one per family, and the flagged rows."""
    out: list[str] = []
    for fam, title in FAMILY_TITLES.items():
        fr = [r for r in rows if r["family"] == fam]
        if not fr:
            continue
        out.append(f"### {title}\n")
        out.append(
            "| line | raise to | source | range | fold | max fold | call | raise (all-in) | flags |"
        )
        out.append("|---|---:|---|---:|---:|---:|---:|---:|---|")
        for r in fr:
            size = r["size"]
            to = _fmt_bb(r["raise_to"], big_blind)
            label = to if size == to else f"{size} ({to})"
            src = r["source"]
            if src != "table" and r["translation"] and len(r["translation"]) > 1:
                src += (
                    " (as "
                    + " / ".join(f"{k} {_pct(v)}" for k, v in r["translation"].items())
                    + ")"
                )
            out.append(
                f"| {r['series']} | {label} | {src} | {_pct(r['range'])} | "
                f"{_pct(r['fold'])} | {_pct(r['max_fold'])} | {_pct(r['call'])} | "
                f"{_pct(r['raise'])} ({_pct(r['allin'])}) | {'; '.join(r['flags'])} |"
            )
        out.append("")
    flagged = [r for r in rows if r["flags"]]
    out.append(f"**{len(flagged)} of {len(rows)} spots flagged.**\n")
    return "\n".join(out)


__all__ = [
    "Spot",
    "audit_decision",
    "audit_markdown",
    "audit_spots",
    "flag_rows",
    "run_audit",
]
