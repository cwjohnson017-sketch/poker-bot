"""Preflop best response with flop-root leaves (pokerbot.search.preflop_eval)."""

from __future__ import annotations

import math

import torch

from pokerbot.engine_select import get_engine
from pokerbot.search.blueprint import UniformBlueprint
from pokerbot.search.combos import NUM_COMBOS as C
from pokerbot.search.combos import valid_mask
from pokerbot.search.preflop_eval import (
    FLOP_SCALE,
    FlopValues,
    Node,
    evaluate,
    preflop_tree,
    sample_flops,
    suit_perms,
)


def _config(engine):
    return engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)


def test_suit_perms_are_permutations():
    P = suit_perms()
    assert P.shape == (24, C)
    assert torch.equal(P[0], torch.arange(C))  # the identity comes first
    for k in range(24):
        assert torch.equal(P[k].sort().values, torch.arange(C))


def test_sample_flops_distinct_and_sorted():
    fl = sample_flops(50, seed=3)
    assert len(set(fl)) == 50
    assert all(list(f) == sorted(f) and len(set(f)) == 3 for f in fl)
    assert sample_flops(50, seed=3) == fl


def test_flop_values_mean_of_constant():
    """A value of 1 for every valid hand averages to FLOP_SCALE for every hand."""
    fv = FlopValues(suit_perms())
    for f in sample_flops(3, seed=0):
        fv.add(0, f, torch.ones(2, C) * valid_mask(list(f)))
    m = fv.mean((0,))
    assert torch.allclose(m, torch.full((2, C), FLOP_SCALE, dtype=m.dtype))
    assert math.isclose(FLOP_SCALE, math.comb(50, 3) / math.comb(48, 3))


def test_preflop_tree_uniform_blueprint():
    engine = get_engine()
    nodes = preflop_tree(UniformBlueprint(), _config(engine), engine)
    assert nodes[0].kind == "dec" and nodes[0].player == 0
    kinds = {n.kind for n in nodes}
    assert kinds == {"dec", "fold", "allin", "flop"}
    for n in nodes:
        if n.kind == "dec":
            assert torch.allclose(n.policy.sum(1), torch.ones(C))
            assert len(n.children) == n.policy.shape[1]
    # limp, check: both put in the big blind
    limp_check = [n for n in nodes if n.kind == "flop" and n.line() == "0:c 1:c"]
    assert len(limp_check) == 1 and limp_check[0].contrib == (100, 100)
    ids = sorted(n.flop_id for n in nodes if n.kind == "flop")
    assert ids == list(range(len(ids)))


def _toy():
    """Seat 0 folds (loses 50) or shoves; over a shove seat 1 folds (loses 100)."""
    ones = torch.ones(2, C)
    half = torch.full((C, 2), 0.5)
    shove = ones.clone()
    shove[0] *= 0.5
    n0 = Node("dec", [], ones, (50, 100), player=0, policy=half, children=[1, 2])
    n1 = Node("fold", [], shove, (50, 100), player=0)
    n2 = Node("fold", [], shove, (10000, 100), player=1)
    return [n0, n1, n2]


def test_evaluate_toy_tree():
    nodes = _toy()
    res = evaluate(nodes, {}, None, big_blind=100)
    s = res["seats"]
    # blueprint: half the time -50, half +100 -> +25 chips = +250 mbb/hand for seat 0
    assert math.isclose(s[0]["blueprint"], 250.0, rel_tol=1e-9)
    assert math.isclose(res["zero_sum_check"], 0.0, abs_tol=1e-9)
    # best response: always shove, +100 chips
    assert math.isclose(s[0]["br_in_sample"], 1000.0, rel_tol=1e-9)
    assert math.isclose(s[0]["br_cross_fit"], 1000.0, rel_tol=1e-9)
    # seat 1 has no decision: its best response is the blueprint value
    assert math.isclose(s[1]["br_in_sample"], -250.0, rel_tol=1e-9)
    assert math.isclose(res["exploitability_in_sample"], 375.0, rel_tol=1e-9)
    top = s[0]["nodes"][0]
    assert top["best_freq"] == [0.0, 1.0]
    assert math.isclose(top["gain"], 750.0, rel_tol=1e-9)
