"""Preflop solving against flop-root leaves (pokerbot.search.preflop_solve)."""

from __future__ import annotations

import math

import torch

from pokerbot.engine_select import get_engine
from pokerbot.search.abstract import make_state, to_action
from pokerbot.search.blueprint import UniformBlueprint, policy_matrix
from pokerbot.search.combos import NUM_COMBOS as C
from pokerbot.search.combos import combo_index
from pokerbot.search.preflop_eval import Node, preflop_allin_matrix, preflop_tree
from pokerbot.search.preflop_solve import (
    K,
    PreflopGame,
    PreflopTable,
    class_matrix,
    exploitability,
    hand_classes,
    node_policies,
    solve_preflop,
    table_from_strategy,
    with_strategy,
)


def test_hand_classes():
    cls = hand_classes()
    counts = torch.bincount(cls, minlength=K)
    assert len(counts) == K
    assert (counts[:13] == 6).all()  # pairs
    assert (counts[13 : 13 + 78] == 4).all()  # suited
    assert (counts[13 + 78 :] == 12).all()  # offsuit
    aa = combo_index(48, 49)  # Ac Ad
    aks = combo_index(48, 44)  # Ac Kc
    ako = combo_index(48, 45)  # Ac Kd
    assert cls[aa] == 12
    assert cls[aks] != cls[ako]
    assert class_matrix().sum() == C


def _toy():
    """Seat 0 folds (loses 50) or shoves; over a shove seat 1 folds (loses 100)."""
    ones = torch.ones(2, C)
    half = torch.full((C, 2), 0.5)
    n0 = Node("dec", [], ones, (50, 100), player=0, policy=half, children=[1, 2])
    n1 = Node("fold", [], ones, (50, 100), player=0)
    n2 = Node("fold", [], ones, (10000, 100), player=1)
    return [n0, n1, n2]


def test_cfr_toy_finds_the_shove():
    nodes = _toy()
    game = PreflopGame(nodes, {}, torch.zeros(C, C))
    sigma = node_policies(nodes)
    ex0 = exploitability(game, sigma)
    assert math.isclose(ex0["values"][0], 250.0, rel_tol=1e-9)
    sol, _ = solve_preflop(game, 50)
    assert float(sol[0][:, 1].min()) > 0.99
    ex = exploitability(game, sol)
    assert ex["exploitability"] < 1.0


def test_with_strategy_recomputes_reach():
    nodes = _toy()
    shove = {0: torch.tensor([[0.0, 1.0]]).expand(C, 2).double()}
    out = with_strategy(nodes, shove)
    assert torch.allclose(out[2].reach[0], torch.ones(C))
    assert torch.allclose(out[1].reach[0], torch.zeros(C))
    assert torch.equal(nodes[0].policy, torch.full((C, 2), 0.5))  # original untouched


def test_preflop_table_replaces_the_tree_and_maps_the_button():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    bp = UniformBlueprint()
    nodes = preflop_tree(bp, cfg, engine)
    # a strategy that always takes the node's last action
    sigma = {}
    for i, n in enumerate(nodes):
        if n.kind == "dec":
            s = torch.zeros(C, len(n.children), dtype=torch.float64)
            s[:, -1] = 1
            sigma[i] = s
    table = table_from_strategy(nodes, sigma, bp.spec.num_actions)
    tb = PreflopTable(bp, table)
    for button in (0, 1):
        st = make_state(engine, cfg, button, [], [])
        p = int(st.current_player)
        P = policy_matrix(tb, st, p)
        assert torch.allclose(P[:, nodes[0].indices[-1]], torch.ones(C))
        row = tb.policy(st, p)
        assert row[nodes[0].indices[-1]] == 1.0
    # after the button limps, the big blind's node
    st = make_state(engine, cfg, 1, [], [])
    st.apply(to_action(engine, 1, 0))
    limp = [n for n in nodes if n.kind == "dec" and n.line() == "0:c"][0]
    P = policy_matrix(tb, st, int(st.current_player))
    assert torch.allclose(P[:, limp.indices[-1]], torch.ones(C))
    # off the table (the flop): the base blueprint
    hist = [(0, 0, to_action(engine, 1, 0)), (0, 1, to_action(engine, 1, 0))]
    st2 = make_state(engine, cfg, 0, [0, 1, 2], hist)
    assert int(st2.street) == 1 and tb._key(st2) is None


def test_allin_matrix_antisymmetric():
    E = preflop_allin_matrix(200)
    assert torch.allclose(E, -E.t())


def test_frozen_leaves_zero_sum_for_any_reach():
    from pokerbot.search.preflop_solve import FrozenLeaves

    g = torch.Generator().manual_seed(0)
    L = 3
    fl = FrozenLeaves(
        torch.arange(L),
        torch.randn(2, L, C, generator=g, dtype=torch.float64),
        torch.randn(L, generator=g, dtype=torch.float64),
    )
    for _ in range(3):
        reach = torch.rand(2, L, C, generator=g, dtype=torch.float64)
        total = sum((reach[p] * fl.values(p, reach)).sum() for p in (0, 1))
        scale = (reach[0] * fl.values(0, reach)).abs().sum()
        assert float(total.abs()) < 1e-9 * float(scale)
