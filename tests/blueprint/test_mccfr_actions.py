"""Action abstraction: sizes, raise cap, de-duplication, pseudo-harmonic mapping."""

import poker_engine as pe
import pytest
from poker_engine import Action, GameConfig, GameState

from pokerbot.blueprint.mccfr import DEFAULT_ACTIONS, normalize_action_spec

DECK = list(range(52))


def new_hand(stack=20000):
    return GameState.new_hand(GameConfig(stacks=[stack, stack]), 0, DECK)


# --- pseudo-harmonic mapping, hand-computed values ---------------------------
# f(x) = (B - x)(1 + A) / ((B - A)(1 + x)) is the probability of mapping x to A.


@pytest.mark.parametrize(
    "a,b,x,want",
    [
        (0.5, 1.0, 0.75, 0.375 / 0.875),  # 0.428571...
        (0.33, 0.75, 0.5, (0.25 * 1.33) / (0.42 * 1.5)),  # 0.527778
        (1.0, 2.0, 1.5, 0.4),  # (0.5 * 2) / (1 * 2.5)
        (0.5, 1.0, 0.5, 1.0),
        (0.5, 1.0, 1.0, 0.0),
        (0.0, 1.0, 0.5, 1.0 / 3.0),  # (0.5 * 1) / (1 * 1.5)
    ],
)
def test_pseudo_harmonic_values(a, b, x, want):
    assert pe.pseudo_harmonic(a, b, x) == pytest.approx(want, abs=1e-12)


def test_pseudo_harmonic_is_monotone():
    xs = [0.5 + 0.05 * i for i in range(11)]
    fs = [pe.pseudo_harmonic(0.5, 1.0, x) for x in xs]
    assert all(f1 > f2 for f1, f2 in zip(fs, fs[1:], strict=False))


def test_translate_uses_pseudo_harmonic_between_neighbours():
    aa = pe.ActionAbstraction()  # flop: 0.33, 0.75, 1.5 pot, all-in
    s = new_hand()
    s.apply(Action.raise_to(300))
    s.apply(Action.check_call())  # flop, pot 600, BB acts first
    assert s.street == 1 and s.pot == 600
    legal = dict(aa.legal(s))
    assert legal[2] == Action.raise_to(198) and legal[3] == Action.raise_to(450)
    # A bet of 450 (0.75 pot) is on the tree.
    assert aa.translate(s, s, Action.raise_to(450), 0.99) == 3
    # A bet of 300 (0.5 pot) lies between 198 (0.33) and 450 (0.75).
    p = (0.75 - 0.5) * 1.33 / ((0.75 - 0.33) * 1.5)
    assert p == pytest.approx(0.527778, abs=1e-6)
    assert aa.translate(s, s, Action.raise_to(300), p - 1e-6) == 2
    assert aa.translate(s, s, Action.raise_to(300), p + 1e-6) == 3
    # Below the smallest size maps to it; checks map to check/call.
    assert aa.translate(s, s, Action.raise_to(100), 0.99) == 2
    assert aa.translate(s, s, Action.check_call(), 0.5) == 1
    # An all-in maps to the abstract all-in.
    assert aa.translate(s, s, Action.raise_to(19700), 0.99) == 5


def test_translate_between_different_pots():
    aa = pe.ActionAbstraction()
    real, abs_ = new_hand(), new_hand()
    # Off-tree open to 400 = 1.5 pot, between pot (1.0) and all-in (99.5 pot).
    p = pe.pseudo_harmonic(1.0, 99.5, 1.5)
    assert p == pytest.approx((99.5 - 1.5) * 2.0 / (98.5 * 2.5))
    assert aa.translate(abs_, real, Action.raise_to(400), p - 1e-6) == 3
    assert aa.translate(abs_, real, Action.raise_to(400), p + 1e-6) == 4
    abs_.apply(aa.to_concrete(abs_, 3))  # abstract game: open to 300
    real.apply(Action.raise_to(400))
    # A pot-sized 3-bet in the real game (to 1200) is the pot-sized 3-bet of the
    # abstract game (to 900), because sizes compare as pot fractions.
    assert aa.translate(abs_, real, Action.raise_to(1200), 0.5) == 3
    assert aa.to_concrete(abs_, 3) == Action.raise_to(900)


def test_default_table_and_sizes():
    aa = pe.ActionAbstraction()
    assert aa.max_raises == 4
    want = normalize_action_spec(None)
    got = [[tuple(a) for a in street] for street in aa.streets]
    assert [[a[0] for a in s] for s in got] == [[a[0] for a in s] for s in want]
    assert got[1][2][1] == pytest.approx(0.33)
    assert list(DEFAULT_ACTIONS) == ["preflop", "flop", "turn", "river"]
    s = new_hand()
    # SB: fold, call, 2.5x = 250 (0.75 pot), 3x = 300 (pot), all-in.
    assert aa.legal(s) == [
        (0, Action.fold()),
        (1, Action.check_call()),
        (2, Action.raise_to(250)),
        (3, Action.raise_to(300)),
        (4, Action.raise_to(20000)),
    ]


def test_raise_cap_and_dedup():
    aa = pe.ActionAbstraction(max_raises=2)
    s = new_hand()
    s.apply(Action.raise_to(300))
    s.apply(Action.raise_to(900))
    assert [i for i, _ in aa.legal(s)] == [0, 1]  # cap reached: fold / call only
    # Short stacks: pot-sized and 0.75-pot raises exceed the stack and collapse
    # into the all-in, which keeps its own index.
    s = new_hand(stack=600)
    s.apply(Action.raise_to(300))
    assert aa.legal(s) == [(0, Action.fold()), (1, Action.check_call()), (4, Action.raise_to(600))]
    assert aa.to_concrete(s, 3) is None
    assert aa.to_concrete(s, 4) == Action.raise_to(600)


def test_sequence_replay():
    aa = pe.ActionAbstraction()
    s = new_hand()
    for a in (Action.raise_to(250), Action.raise_to(750), Action.check_call()):
        s.apply(a)
    assert aa.sequence(s) == [2, 3, 1]
    s.apply(Action.raise_to(333))  # not an abstract size
    with pytest.raises(ValueError):
        aa.sequence(s)


def test_custom_spec_and_validation():
    aa = pe.ActionAbstraction(
        {
            "preflop": ["fold", "call", ["raise", 1.0]],
            "flop": ["call"],
            "turn": ["call"],
            "river": ["call", "allin"],
        },
        max_raises=1,
    )
    assert aa.streets[0] == [("fold",), ("check_call",), ("raise", 1.0)]
    with pytest.raises(ValueError):
        pe.ActionAbstraction([["fold"]] * 4)  # no check_call
    with pytest.raises(ValueError):
        pe.ActionAbstraction(None, max_raises=9)
    with pytest.raises(ValueError):
        normalize_action_spec({"flop": [["raise", -1]]})
