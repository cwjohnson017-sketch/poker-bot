"""The action abstraction: one Python spec, its Rust twin, and off-tree mapping.

``docs/INTERFACES.md`` puts the action abstraction here. There is exactly one
Python spec type, :class:`ActionSpec` (re-exported from
``pokerbot.env.actions``, which the torch env uses), with per-street lists of
``("fold",)``, ``("check_call",)``, ``("raise", pot_fraction)``,
``("raise_x", multiple_of_the_bet)`` and ``("allin",)``. The Rust
``poker_engine.ActionAbstraction`` (tabular MCCFR, blueprint play) implements
the same idea with pot fractions only; :func:`to_rust` and :func:`from_rust`
convert between the two so that both sides give the **same abstract indices,
the same legal sets and the same raise-to amounts** in every state
(``tests/abstraction/test_actions.py`` checks this on random states).

How the conversion keeps the two sides identical
------------------------------------------------

* **Rounding.** The env computes
  ``raise_to = max_bet + (f_milli * (pot + to_call) + 500) // 1000``
  (round half up, sizes in thousandths); Rust computes
  ``current_bet + round(f * (pot + to_call))`` in ``f64``. :func:`to_rust`
  passes ``f = f_milli / 1000 + EPS`` with ``EPS = 1e-10``: the exact value
  ``f_milli * X / 1000`` is either a half-integer (then ``EPS * X`` beats the
  ``~1e-16`` relative float error and rounds it up, like the env) or at least
  ``0.001`` away from one (then ``EPS * X`` cannot cross it while
  ``X = pot + to_call < 10**7`` chips). So amounts agree exactly for any pot
  below ten million chips; the project's games have ~10^4.
* **Multiples.** ``("raise_x", m)`` (raise to ``m`` times the bet faced) has
  no Rust counterpart. Heads-up preflop without antes the pot after calling
  is always twice the current bet, so ``raise_x m`` equals ``raise`` with
  pot fraction ``(m - 1) / 2`` exactly (2.5x = 0.75 pot, 3x = 1.0 pot).
  :func:`to_rust` converts preflop multiples that way and refuses them on
  later streets (and, when given a config, outside heads-up no-ante games).
* **Duplicates.** ``DEFAULT_SPEC`` has both ``3x`` and ``pot`` preflop,
  which are the same raise heads-up; Rust rejects duplicate entries, so a
  repeated fraction gets one more ``EPS`` per repeat. The amounts stay equal,
  and both sides then drop the later entry as a duplicate amount, so the
  index layout is unchanged.
* **All-in.** The env masks a sized raise that would put the player all-in
  (``allin`` covers it); Rust clamps it and merges it into the all-in entry.
  Same legal set when the street has an ``allin`` entry (every spec used in
  this project); without one the env drops the size and Rust keeps it.
* ``dedupe=False`` has no Rust equivalent (Rust always de-duplicates);
  :func:`to_rust` rejects it.

:func:`from_rust` returns pot fractions rounded to thousandths, so
``from_rust(to_rust(spec))`` equals ``spec`` except that preflop multiples come
back as the equivalent pot fractions.

Off-tree mapping
----------------

:func:`map_offtree` maps a concrete (possibly off-tree) opponent action to an
abstract index with the pseudo-harmonic mapping of Ganzfried & Sandholm
(2013): between neighbouring abstract sizes ``a < x < b`` (pot fractions) the
bet maps to ``a`` with probability
``f(x) = (b - x)(1 + a) / ((b - a)(1 + x))``. For a ``poker_engine.GameState``
it delegates to ``ActionAbstraction.translate``; for any other contract state
(the reference engine, the match runner's masked view) a line-by-line Python
mirror of that method is used. ``mode="deterministic"`` uses ``u = 0.5``,
i.e. the nearest size under the pseudo-harmonic metric.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from typing import Any

import numpy as np
import torch

from ..env.actions import (
    CHECK_CALL,
    DEFAULT_SPEC,
    FOLD,
    RAISE,
    ActionSpec,
    legal_mask,
    raise_targets,
    spec_from_lists,
)

__all__ = [
    "CHECK_CALL",
    "DEFAULT_SPEC",
    "EPS",
    "FOLD",
    "RAISE",
    "ActionSpec",
    "as_spec",
    "decision_info",
    "from_rust",
    "legal_actions",
    "legal_actions_batch",
    "map_offtree",
    "pot_fraction",
    "pseudo_harmonic",
    "rust_abstraction",
    "to_concrete",
    "to_rust",
]

EPS = 1e-10  # added to pot fractions handed to Rust (see the module docstring)
STREET_NAMES = ("preflop", "flop", "turn", "river")
_ALIASES = {
    "fold": "fold",
    "f": "fold",
    "check_call": "check_call",
    "call": "check_call",
    "check": "check_call",
    "c": "check_call",
    "allin": "allin",
    "all_in": "allin",
    "all-in": "allin",
    "a": "allin",
    "raise": "raise",
    "r": "raise",
    "raise_pot": "raise",
    "raise_x": "raise_x",
    "x": "raise_x",
}


# ----------------------------------------------------------------------------- specs


def _norm_action(a: Any) -> tuple:
    if isinstance(a, str):
        name, args = a, []
    else:
        items = list(a)
        if not items:
            raise ValueError("empty abstract action")
        name, args = str(items[0]), items[1:]
    kind = _ALIASES.get(name.strip().lower())
    if kind is None:
        raise ValueError(f"unknown abstract action {a!r}")
    if kind in ("raise", "raise_x"):
        if len(args) != 1:
            raise ValueError(f"{kind} needs one size: {a!r}")
        return (kind, float(args[0]))
    if args:
        raise ValueError(f"{kind} takes no argument: {a!r}")
    return (kind,)


def as_spec(obj: Any = None, max_raises: int | None = None, dedupe: bool = True) -> ActionSpec:
    """An :class:`ActionSpec` from a spec, a ``poker_engine.ActionAbstraction``,
    four per-street lists, or a dict keyed by street name (YAML shorthands such
    as ``call`` and ``[raise, 0.75]`` are accepted). ``None`` is ``DEFAULT_SPEC``."""
    if obj is None:
        spec = DEFAULT_SPEC
    elif isinstance(obj, ActionSpec):
        spec = obj
    elif type(obj).__name__ == "ActionAbstraction":
        spec = from_rust(obj)
    else:
        if isinstance(obj, dict):
            mr = obj.get("max_raises", obj.get("max_raises_per_street"))
            if mr is not None and max_raises is None:
                max_raises = int(mr)
            missing = [s for s in STREET_NAMES if s not in obj]
            if missing:
                raise ValueError(f"action spec is missing streets {missing}")
            lists = [obj[s] for s in STREET_NAMES]
        else:
            lists = list(obj)
        if len(lists) != 4:
            raise ValueError("need 4 per-street action lists")
        spec = spec_from_lists(
            [[_norm_action(a) for a in lst] for lst in lists],
            max_raises=4 if max_raises is None else max_raises,
            dedupe=dedupe,
        )
        return spec
    if max_raises is not None and max_raises != spec.max_raises:
        spec = ActionSpec(spec.streets, max_raises, spec.dedupe)
    return spec


def _milli(x: float) -> int:
    return int(round(float(x) * 1000))


def to_rust(spec: ActionSpec | Any, config: Any = None) -> Any:
    """The ``poker_engine.ActionAbstraction`` equivalent to ``spec`` (same
    indices, legal sets and amounts; see the module docstring). ``config``
    (optional) is checked for the heads-up no-ante condition under which
    preflop ``raise_x`` entries convert exactly."""
    import poker_engine as pe

    spec = as_spec(spec)
    if not spec.dedupe:
        raise ValueError("the Rust abstraction always de-duplicates; use dedupe=True")
    streets = []
    for s, st in enumerate(spec.streets):
        seen: dict[int, int] = {}
        out: list[tuple] = []
        for a in st:
            kind = a[0]
            if kind in ("fold", "check_call", "allin"):
                out.append((kind,))
                continue
            if kind == "raise":
                num, den = _milli(a[1]), 1000
            else:  # raise_x
                if s != 0:
                    raise ValueError(
                        f"street {s}: raise_x has no pot-fraction equivalent after preflop"
                    )
                if config is not None and (
                    int(config.num_players) != 2 or int(getattr(config, "ante", 0)) != 0
                ):
                    raise ValueError("raise_x converts exactly only heads-up without antes")
                m = _milli(a[1])
                if m <= 1000:
                    raise ValueError(f"raise_x multiple must exceed 1, got {a[1]}")
                num, den = m - 1000, 2000  # (m - 1) / 2
            # Key on the exact rational num/den (in 1/2000 units).
            key = num * (2000 // den)
            rep = seen.get(key, 0)
            seen[key] = rep + 1
            out.append(("raise", num / den + EPS * (1 + rep)))
        streets.append(out)
    return pe.ActionAbstraction(streets, spec.max_raises)


def from_rust(abstraction: Any) -> ActionSpec:
    """The :class:`ActionSpec` of a ``poker_engine.ActionAbstraction`` (pot
    fractions rounded to thousandths, ``dedupe=True``)."""
    lists = []
    for st in abstraction.streets:
        out = []
        for a in st:
            if a[0] == "raise":
                out.append(("raise", _milli(a[1]) / 1000))
            else:
                out.append((a[0],))
        lists.append(out)
    return spec_from_lists(lists, int(abstraction.max_raises), True)


@lru_cache(maxsize=64)
def rust_abstraction(spec: ActionSpec) -> Any:
    """Cached :func:`to_rust` (specs are frozen and hashable)."""
    return to_rust(spec)


# ------------------------------------------------------------------- state quantities


def _is_rust_state(state: Any) -> bool:
    return type(state).__module__ == "poker_engine" and type(state).__name__ == "GameState"


def decision_info(state: Any) -> dict[str, int]:
    """Public quantities of a decision point, as the env and Rust define them:
    ``street, player, pot, max_bet, to_call, raise_ok, min_raise_to,
    max_raise_to, n_raises`` (voluntary raises, bets included, this street)."""
    p = int(state.current_player)
    street = int(state.street)
    bets = [int(b) for b in state.street_bets]
    cur = getattr(state, "current_bet", None)
    if cur is None:
        cur = max(bets)
        if street == 0:
            cur = max(cur, int(state.config.big_blind))
    cur = int(cur)
    n_raises = getattr(state, "num_raises_this_street", None)
    if n_raises is None:
        n_raises = sum(1 for s, _, a in state.history if int(s) == street and int(a.kind) == RAISE)
    la = state.legal_actions()
    raise_ok = int(la.min_raise_to) > 0
    return {
        "street": street,
        "player": p,
        "pot": int(state.pot),
        "max_bet": cur,
        "to_call": max(cur - bets[p], 0),
        "raise_ok": raise_ok,
        "min_raise_to": int(la.min_raise_to) if raise_ok else 0,
        "max_raise_to": int(la.max_raise_to) if raise_ok else 0,
        "n_raises": int(n_raises),
    }


def legal_actions_batch(
    spec: ActionSpec, states: Sequence[Any]
) -> list[list[tuple[int, int, int]]]:
    """Python-side legal abstract actions of many live states, computed with
    the env's tensor functions ``raise_targets`` / ``legal_mask``: per state a
    list of ``(abstract_index, kind, amount)`` in index order (``amount`` is
    the raise-to for raises, 0 otherwise)."""
    spec = as_spec(spec)
    if not states:
        return []
    tab = spec.tables("cpu")
    infos = [decision_info(s) for s in states]

    def col(name: str) -> torch.Tensor:
        return torch.tensor([int(i[name]) for i in infos], dtype=torch.long)

    street = col("street")
    targets = raise_targets(
        tab,
        street,
        col("pot"),
        col("max_bet"),
        col("to_call"),
        col("min_raise_to"),
        col("max_raise_to"),
    )
    mask = legal_mask(
        tab,
        street,
        torch.ones(len(states), dtype=torch.bool),
        col("to_call"),
        col("raise_ok").bool(),
        col("n_raises"),
        targets,
        col("max_raise_to"),
    )
    out = []
    for r, info in enumerate(infos):
        row = []
        for i, a in enumerate(spec.streets[info["street"]]):
            if not bool(mask[r, i]):
                continue
            if a[0] == "fold":
                row.append((i, FOLD, 0))
            elif a[0] == "check_call":
                row.append((i, CHECK_CALL, 0))
            else:
                row.append((i, RAISE, int(targets[r, i])))
        out.append(row)
    return out


def legal_actions(spec: ActionSpec, state: Any) -> list[tuple[int, int, int]]:
    """Legal ``(abstract_index, kind, amount)`` of one live state (Python side)."""
    return legal_actions_batch(spec, [state])[0]


def to_concrete(spec: ActionSpec, state: Any, index: int) -> tuple[int, int] | None:
    """``(kind, amount)`` of abstract ``index`` in ``state``, or None if not legal."""
    for i, k, amt in legal_actions(spec, state):
        if i == index:
            return k, amt
    return None


# --------------------------------------------------------------------- off-tree map


def pseudo_harmonic(a: float, b: float, x: float) -> float:
    """Probability of mapping pot fraction ``x`` (``a <= x <= b``) to ``a``:
    ``f(x) = (b - x)(1 + a) / ((b - a)(1 + x))`` (Ganzfried & Sandholm 2013)."""
    if b <= a:
        return 1.0
    x = min(max(x, a), b)
    return ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x))


def pot_fraction(info: dict[str, int], raise_to: int) -> float:
    """Pot fraction of ``raise_to`` at a decision (Rust ``raise_fraction``)."""
    return (int(raise_to) - info["max_bet"]) / max(info["pot"] + info["to_call"], 1)


def _translate_py(
    spec: ActionSpec, abs_state: Any, real_state: Any, kind: int, amount: int, u: float
) -> int | None:
    """Python mirror of ``ActionAbstraction::translate`` (engine/src/abstraction.rs)."""
    if abs_state.is_terminal:
        return None
    legal = legal_actions(spec, abs_state)
    names = spec.streets[int(abs_state.street)]

    def find(name: str) -> int | None:
        return next((i for i, _, _ in legal if names[i][0] == name), None)

    call = find("check_call")
    if kind == FOLD:
        f = find("fold")
        return f if f is not None else call
    if kind == CHECK_CALL:
        return call
    ai = decision_info(abs_state)
    raises = [
        (pot_fraction(ai, amt), i, amt == ai["max_raise_to"]) for i, k, amt in legal if k == RAISE
    ]
    if not raises:
        return call
    ri = decision_info(real_state)
    if ri["raise_ok"] and amount >= ri["max_raise_to"]:
        for _, i, is_allin in raises:
            if is_allin:
                return i
    raises.sort(key=lambda r: r[0])  # stable, like Rust's sort_by
    x = pot_fraction(ri, amount)
    if x <= raises[0][0]:
        return raises[0][1]
    if x >= raises[-1][0]:
        return raises[-1][1]
    k = next(j for j, r in enumerate(raises) if r[0] >= x)
    a, b = raises[k - 1], raises[k]
    return a[1] if u < pseudo_harmonic(a[0], b[0], x) else b[1]


def map_offtree(
    spec: ActionSpec | Any,
    state: Any,
    action: Any,
    rng: np.random.Generator | None = None,
    mode: str = "randomized",
    abs_state: Any = None,
) -> int | None:
    """Abstract index for a concrete ``action`` taken at ``state``.

    Fold maps to fold (check/call when folding is not an abstract option),
    check/call to check/call; a raise maps to the abstract all-in when it is
    all-in, otherwise by the pseudo-harmonic mapping between the neighbouring
    abstract sizes (below the smallest / above the largest: that size; no
    abstract raise available: check/call). ``abs_state`` is the matching
    decision point of the abstract game when its pot differs from ``state``
    (defaults to ``state``). ``mode="randomized"`` draws ``u ~ U[0, 1)`` from
    ``rng``; ``"deterministic"`` uses ``u = 0.5``. Returns None at a terminal
    state.
    """
    spec = as_spec(spec)
    if mode == "deterministic":
        u = 0.5
    elif mode == "randomized":
        if rng is None:
            raise ValueError("randomized mapping needs an rng")
        u = float(rng.random())
    else:
        raise ValueError(f"mode must be 'randomized' or 'deterministic', got {mode!r}")
    abs_state = state if abs_state is None else abs_state
    kind, amount = int(action.kind), int(getattr(action, "amount", 0) or 0)
    if _is_rust_state(state) and _is_rust_state(abs_state):
        import poker_engine as pe

        act = (
            pe.Action.raise_to(amount)
            if kind == RAISE
            else (pe.Action.fold() if kind == FOLD else pe.Action.check_call())
        )
        return rust_abstraction(spec).translate(abs_state, state, act, u)
    return _translate_py(spec, abs_state, state, kind, amount, u)
