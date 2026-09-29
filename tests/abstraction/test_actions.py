"""Action abstraction: Python spec vs Rust ActionAbstraction, pseudo-harmonic mapping."""

from __future__ import annotations

import numpy as np
import poker_engine as pe
import pytest

import pokerbot.reference as ref
from pokerbot.abstraction import actions as A
from pokerbot.abstraction.actions import (
    CHECK_CALL,
    DEFAULT_SPEC,
    FOLD,
    RAISE,
    as_spec,
    from_rust,
    legal_actions,
    legal_actions_batch,
    map_offtree,
    pseudo_harmonic,
    to_rust,
)
from pokerbot.blueprint.mccfr.interfaces import DEFAULT_ACTIONS

MCCFR_SMALL = as_spec(
    {
        "preflop": ["fold", "check_call", ["raise", 0.75], ["raise", 1.0], "allin"],
        "flop": ["fold", "check_call", ["raise", 0.5], ["raise", 1.0], "allin"],
        "turn": ["fold", "check_call", ["raise", 0.75], "allin"],
        "river": ["fold", "check_call", ["raise", 0.75], "allin"],
        "max_raises": 3,
    }
)
# Fractions whose float products land on the wrong side of .5 without the
# conversion's epsilon (0.35 * 90 = 31.499999...).
ODD = as_spec(
    [["fold", "call", ("raise", 0.07), ("raise", 0.35), ("raise", 0.7), ("raise", 1.005), "allin"]]
    * 4,
    max_raises=3,
)


def random_states(n: int, seed: int, blinds: bool = True, with_actions: bool = False):
    """Decision states from random play with random stacks and blinds."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        stacks = [int(rng.integers(1, 400)) * int(rng.choice([1, 7, 50])) for _ in range(2)]
        bb = int(rng.integers(2, 300)) if blinds else 100
        cfg = pe.GameConfig(num_players=2, stacks=stacks, small_blind=bb // 2, big_blind=bb)
        deck = rng.permutation(52).tolist()
        button = int(rng.integers(2))
        s = pe.GameState.new_hand(cfg, button, deck)
        while not s.is_terminal:
            la = s.legal_actions()
            r = rng.random()
            if la.can_raise and r < 0.45:
                lo, hi = la.min_raise_to, la.max_raise_to
                a = pe.Action.raise_to(int(rng.integers(lo, hi + 1)) if lo <= hi else hi)
            elif la.can_fold and r < 0.55:
                a = pe.Action.fold()
            else:
                a = pe.Action.check_call()
            out.append((s.clone(), a, cfg, deck, button) if with_actions else s.clone())
            s.apply(a)
    return out[:n]


@pytest.mark.parametrize(
    ("spec", "seed"),
    [(DEFAULT_SPEC, 1), (MCCFR_SMALL, 2), (ODD, 3)],
    ids=["default", "small", "odd"],
)
def test_python_and_rust_give_same_legal_sets_and_amounts(spec, seed):
    states = random_states(400, seed=seed)
    rust = to_rust(spec)
    py = legal_actions_batch(spec, states)
    for s, p in zip(states, py, strict=True):
        r = [(i, a.kind, a.amount) for i, a in rust.legal(s)]
        assert r == p, (s.street, s.pot, s.street_bets, s.stacks)
        for i, kind, amount in p:
            a = rust.to_concrete(s, i)
            assert (a.kind, a.amount) == (kind, amount)
    # Every street and several raise levels were exercised.
    assert {s.street for s in states} == {0, 1, 2, 3}
    assert max(s.num_raises_this_street for s in states) >= 2


def test_rounding_matches_only_with_the_epsilon():
    # SB 25 / BB 50: SB faces the BB, pot + to_call = 100, 1.005 pot = 100.5 chips.
    cfg = pe.GameConfig(num_players=2, stacks=[5000, 5000], small_blind=25, big_blind=50)
    s = pe.GameState.new_hand(cfg, 0, list(range(52)))
    spec = as_spec([["fold", "call", ("raise", 1.005), "allin"]] + [["fold", "call"]] * 3)
    assert legal_actions(spec, s)[2] == (2, RAISE, 50 + 101)  # env: round half up
    assert to_rust(spec).to_concrete(s, 2).amount == 151
    raw = pe.ActionAbstraction([[("fold",), ("check_call",), ("raise", 1.005), ("allin",)]] * 4)
    assert raw.to_concrete(s, 2).amount == 150  # plain f64 product 100.49999...
    # Exhaustive check of the arithmetic for pots up to 2M chips.
    x = np.arange(1, 2_000_000, dtype=np.int64)
    for milli in (5, 70, 330, 350, 700, 750, 1005, 1500, 2000):
        p = (milli / 1000 + A.EPS) * x.astype(np.float64)
        rust_round = np.floor(p) + ((p - np.floor(p)) >= 0.5)  # f64::round, positive values
        assert np.array_equal(rust_round.astype(np.int64), (milli * x + 500) // 1000), milli


def test_conversions():
    rust_default = from_rust(pe.ActionAbstraction())
    assert [list(s) for s in rust_default.streets] == [
        DEFAULT_ACTIONS[k] for k in ("preflop", "flop", "turn", "river")
    ]
    assert from_rust(to_rust(MCCFR_SMALL)) == MCCFR_SMALL
    assert from_rust(to_rust(ODD)) == ODD
    # Preflop multiples come back as the equivalent pot fractions (2.5x = 0.75, 3x = 1.0).
    back = from_rust(to_rust(DEFAULT_SPEC))
    assert back.streets[0][2:5] == (("raise", 0.75), ("raise", 1.0), ("raise", 1.0))
    assert back.streets[1:] == DEFAULT_SPEC.streets[1:]
    assert len(to_rust(DEFAULT_SPEC).streets[0]) == len(DEFAULT_SPEC.streets[0])
    with pytest.raises(ValueError):
        to_rust(as_spec([["fold", "call"]] + [["fold", "call", ("raise_x", 2.0)]] * 3))
    with pytest.raises(ValueError):
        to_rust(A.ActionSpec(DEFAULT_SPEC.streets, 4, dedupe=False))
    with pytest.raises(ValueError):
        to_rust(DEFAULT_SPEC, pe.GameConfig(num_players=2, stacks=[1000, 1000], ante=10))
    assert as_spec(pe.ActionAbstraction()) == rust_default


def test_pseudo_harmonic_values():
    # Sizes 0.5 and 1.0 pot, a bet of 0.75 pot: f = (1 - .75)(1 + .5) / ((1 - .5)(1 + .75)).
    assert pseudo_harmonic(0.5, 1.0, 0.75) == pytest.approx(0.375 / 0.875)
    assert pseudo_harmonic(0.5, 1.0, 0.5) == 1.0
    assert pseudo_harmonic(0.5, 1.0, 1.0) == 0.0
    # f = 1/2 at the pseudo-harmonic midpoint (a + b + 2ab) / (2 + a + b).
    a, b = 0.33, 1.5
    assert pseudo_harmonic(a, b, (a + b + 2 * a * b) / (2 + a + b)) == pytest.approx(0.5)
    rng = np.random.default_rng(0)
    for _ in range(200):
        a, b = sorted(rng.uniform(0.05, 4.0, 2))
        x = rng.uniform(a, b)
        assert pseudo_harmonic(a, b, x) == pytest.approx(pe.pseudo_harmonic(a, b, x), abs=1e-12)


def flop_state():
    """Flop after SB limp / BB check: pot 200, the BB (seat 1) to act."""
    cfg = pe.GameConfig(num_players=2, stacks=[10_000, 10_000], small_blind=50, big_blind=100)
    s = pe.GameState.new_hand(cfg, 0, list(range(52)))
    s.apply(pe.Action.check_call())
    s.apply(pe.Action.check_call())
    assert s.street == 1 and s.pot == 200 and s.current_player == 1
    return s


HALF_POT = as_spec(
    [["fold", "call", ("raise", 1.0), "allin"]]
    + [["fold", "call", ("raise", 0.5), ("raise", 1.0), "allin"]] * 3
)


def test_map_offtree_hand_computed_cases():
    s = flop_state()
    bet = pe.Action.raise_to(150)  # 0.75 pot, between 0.5 (index 2) and 1.0 (index 3)
    p_small = 0.375 / 0.875  # ~0.4286
    assert map_offtree(HALF_POT, s, bet, mode="deterministic") == 3  # u = 0.5 >= f
    rng = np.random.default_rng(1)
    picks = [map_offtree(HALF_POT, s, bet, rng) for _ in range(4000)]
    assert set(picks) == {2, 3}
    freq = picks.count(2) / len(picks)
    assert abs(freq - p_small) < 0.03  # 4.5 sigma
    # Just below the midpoint maps to the smaller size deterministically.
    mid = (0.5 + 1.0 + 2 * 0.5 * 1.0) / (2 + 0.5 + 1.0)  # 0.714 pot
    assert (
        map_offtree(HALF_POT, s, pe.Action.raise_to(int(mid * 200) - 1), mode="deterministic") == 2
    )
    # Exact sizes, below the smallest, above the largest, all-in, fold, call.
    assert map_offtree(HALF_POT, s, pe.Action.raise_to(100), rng) == 2
    assert map_offtree(HALF_POT, s, pe.Action.raise_to(200), rng) == 3
    assert map_offtree(HALF_POT, s, pe.Action.raise_to(100), mode="deterministic") == 2
    assert map_offtree(HALF_POT, s, pe.Action.raise_to(1000), mode="deterministic") == 4
    assert map_offtree(HALF_POT, s, pe.Action.raise_to(9900), mode="deterministic") == 4
    assert map_offtree(HALF_POT, s, pe.Action.check_call(), mode="deterministic") == 1
    # Folding when checking is possible maps to check/call.
    assert map_offtree(HALF_POT, s, pe.Action.fold(), mode="deterministic") == 1
    with pytest.raises(ValueError):
        map_offtree(HALF_POT, s, bet)  # randomized needs an rng
    # The pure-Python mirror agrees on the same cases.
    for amount in (100, 120, 150, 180, 200, 1000):
        for u in (0.0, 0.3, 0.5, 0.9):
            assert A._translate_py(HALF_POT, s, s, RAISE, amount, u) == to_rust(HALF_POT).translate(
                s, s, pe.Action.raise_to(amount), u
            )


def _reference_state(cfg, deck, button, history):
    rcfg = ref.GameConfig(
        num_players=2,
        stacks=list(cfg.stacks),
        small_blind=cfg.small_blind,
        big_blind=cfg.big_blind,
    )
    s = ref.GameState.new_hand(rcfg, button, deck)
    for _, _, a in history:
        s.apply(ref.Action(int(a.kind), int(a.amount)))
    return s


@pytest.mark.parametrize("spec", [DEFAULT_SPEC, ODD], ids=["default", "odd"])
def test_python_mirror_matches_rust_translate(spec):
    rng = np.random.default_rng(5)
    rust = to_rust(spec)
    rows = random_states(250, seed=11, with_actions=True)
    checked_ref = 0
    for s, a, cfg, deck, button in rows:
        u = float(rng.random())
        want = rust.translate(s, s, a, u)
        kind, amount = int(a.kind), int(a.amount)
        assert A._translate_py(spec, s, s, kind, amount, u) == want
        if len(s.history) < 6 and checked_ref < 60:
            r = _reference_state(cfg, deck, button, s.history)
            got = A._translate_py(spec, r, r, kind, amount, u)
            assert got == want
            assert legal_actions(spec, r) == legal_actions(spec, s)
            checked_ref += 1
        mode = "deterministic" if u < 0.5 else "randomized"
        got = map_offtree(spec, s, a, np.random.default_rng(0), mode=mode)
        assert got in {i for i, _, _ in legal_actions(spec, s)}
        if kind == FOLD and s.legal_actions().can_fold:
            assert spec.streets[s.street][got] == ("fold",)
        if kind == CHECK_CALL:
            assert spec.streets[s.street][got] == ("check_call",)
    assert checked_ref >= 30
