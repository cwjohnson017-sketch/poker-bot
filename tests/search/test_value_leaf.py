"""Value-net leaves (``leaf_mode="value_net"``): exactness against a solved
river, card removal, cached net values, flop trees, the node budget and the
agent."""

from __future__ import annotations

import pytest
import torch

from pokerbot.agents import RandomAgent
from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, UniformBlueprint, search_config
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, blocked_sum, valid_mask
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import (
    CONTINUATION,
    DECISION,
    LEAF,
    VALUE,
    TreeConfig,
    build_tree,
)
from pokerbot.search.value_leaf import FixedLeafValues, ShowdownOracle, ValueLeafEvaluator

BIG = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
PASSIVE_STREET = (("fold",), ("check_call",))
SPEC = ActionSpec(streets=(BIG, BIG, BIG, PASSIVE_STREET), max_raises=2)
BOARD = [4, 9, 14, 19, 24]


def _turn(stacks: int = 2000):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, BOARD, [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    assert s.street == 2
    return cfg, s


def _flop(engine, cfg):
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    return s


def _ranges(board, seed=1, dtype=torch.float64):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(board)).to(dtype)


def _value_tree(cfg, s, **kw):
    tc = TreeConfig(spec=SPEC, depth_streets=0, leaf_mode="value_net", **kw)
    return build_tree(cfg, s.button, s.board, s.history, tc)


def _key(tree, n):
    return tree.histories[n], tree.boards[int(tree.board_id[n])]


def test_value_tree_layout():
    cfg, s = _turn()
    tree = _value_tree(cfg, s)
    ids = (tree.kind == VALUE).nonzero().flatten()
    assert len(ids) > 0
    assert tree.count(LEAF) == 0 and tree.count(CONTINUATION) == 0
    assert bool((tree.num_children[ids] == 0).all())
    assert bool((tree.actor[ids] == -1).all())
    assert bool((tree.street[ids] == 3).all())
    assert all(len(tree.boards[int(b)]) == 4 for b in tree.board_id[ids])
    assert all(tree.states[int(n)] is not None for n in ids)
    assert f"value={len(ids)}" in tree.summary()
    with pytest.raises(ValueError, match="leaf_mode"):
        build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SPEC, leaf_mode="nope"))


def test_value_leaves_match_a_checked_down_river():
    """Turn root: VALUE leaves valued by the exact check-down oracle give the
    same solve as the tree that runs to showdown with a checked-down river."""
    cfg, s = _turn()
    tree_a = _value_tree(cfg, s)
    tree_b = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SPEC, depth_streets=1))
    river = [
        n for n in (tree_b.kind == DECISION).nonzero().flatten().tolist() if tree_b.street[n] == 3
    ]
    assert river and all(int(tree_b.num_children[n]) == 1 for n in river)  # river is check-down
    r = _ranges(s.board)
    sc = SolverConfig(dtype="float64")
    sa = RangeSolver(tree_a, r, sc, value_leaves=ValueLeafEvaluator(tree_a, ShowdownOracle()))
    sb = RangeSolver(tree_b, r, sc)
    sa.solve(25)
    sb.solve(25)
    pot = int(s.pot)
    ea, eb = sa.exploitability(), sb.exploitability()
    for k in ("br", "ev"):
        for p in (0, 1):
            assert abs(ea[k][p] - eb[k][p]) < 1e-4 * pot, (k, ea, eb)
    assert abs(ea["exploitability"] - eb["exploitability"]) < 1e-4 * max(1.0, eb["exploitability"])
    for p in (0, 1):
        for br in (False, True):
            va, _ = sa.values(p, best_response=br)
            vb, _ = sb.values(p, best_response=br)
            scale = float(vb[0].abs().max())
            assert float((va[0] - vb[0]).abs().max()) < 1e-4 * scale, (p, br)
    turn_b = {
        _key(tree_b, n): n
        for n in (tree_b.kind == DECISION).nonzero().flatten().tolist()
        if tree_b.street[n] == 2
    }
    dec_a = (tree_a.kind == DECISION).nonzero().flatten().tolist()
    assert len(dec_a) == len(turn_b)
    for n in dec_a:
        a = sa.node_strategy(n)
        b = sb.node_strategy(turn_b[_key(tree_a, n)])
        assert float((a - b).abs().max()) < 1e-4


def test_value_leaf_values_are_exact_with_card_removal():
    """The chance average over river cards equals enumerating the river with
    the dense showdown, and combos that hit the leaf board get zero."""
    cfg, s = _turn()
    tree = _value_tree(cfg, s)
    ev = ValueLeafEvaluator(tree, ShowdownOracle(), chunk=48 * 2)  # several chunks
    L = ev.num_leaves
    g = torch.Generator().manual_seed(5)
    reach = torch.rand(2, L, NUM_COMBOS, generator=g, dtype=torch.float64)  # also on the board
    board = list(tree.boards[int(tree.board_id[ev.ids[0]])])
    valid = valid_mask(board)
    for p in (0, 1):
        v = ev.values(p, reach)
        assert v.shape == (L, NUM_COMBOS)
        assert bool((v[:, ~valid] == 0).all())
        for j in (0, L - 1):
            c = int(tree.contrib[ev.ids[j], 0])
            opp = reach[1 - p, j] * valid
            ref = torch.zeros(NUM_COMBOS, dtype=torch.float64)
            for x in range(52):
                if x in board:
                    continue
                ok = valid_mask([x]).double()
                ref += naive_showdown((opp * ok)[None], board + [x])[0] * ok
            ref *= c / 44
            assert torch.allclose(v[j], ref, atol=1e-9 * float(ref.abs().max()))
    # the per-card identity for the masked opponent masses
    opp = reach[0] * ev.valid
    mx = ev.opponent_mass(opp, ev.cards)
    for x_slot in (0, 17, 47):
        x = ev.cards[:, x_slot]
        avoid = valid_mask([]).expand(L, -1).clone()
        for j in range(L):
            avoid[j] = valid_mask([int(x[j])])
        direct = blocked_sum(opp * avoid)
        assert torch.allclose(mx[:, x_slot] * avoid, direct * avoid, atol=1e-10)


def test_fixed_leaf_values_and_missing_provider():
    cfg, s = _turn()
    tree = _value_tree(cfg, s)
    with pytest.raises(ValueError, match="leaf-value provider"):
        RangeSolver(tree, _ranges(s.board))
    L = tree.count(VALUE)
    fixed = torch.zeros(2, L, NUM_COMBOS, dtype=torch.float64)
    solver = RangeSolver(
        tree, _ranges(s.board), SolverConfig(dtype="float64"), value_leaves=FixedLeafValues(fixed)
    )
    solver.solve(3)
    v, _ = solver.values(0)
    ids = (tree.kind == VALUE).nonzero().flatten()
    assert bool((v[ids] == 0).all())


class _Counting:
    def __init__(self, inner):
        self.inner = inner
        self.calls = 0

    def predict(self, *args):
        self.calls += 1
        return self.inner.predict(*args)


def test_net_every_reuses_cached_values():
    cfg, s = _turn()
    tree = _value_tree(cfg, s)
    r = _ranges(s.board)
    sc = SolverConfig(dtype="float64")
    out = {}
    for every in (1, 3):
        pred = _Counting(ShowdownOracle())
        vl = ValueLeafEvaluator(tree, pred, every=every)
        solver = RangeSolver(tree, r, sc, value_leaves=vl)
        solver.solve(30)
        calls = pred.calls
        e = solver.exploitability()  # 4 fresh evaluations, the cache is left alone
        assert pred.calls == calls + 4
        out[every] = (calls, e)
    calls1, e1 = out[1]
    calls3, e3 = out[3]
    assert calls1 == 60 and calls3 == 20
    pot = int(s.pot)
    assert abs(e3["ev"][0] - e1["ev"][0]) < 0.01 * pot, (e1, e3)
    assert e3["exploitability"] < 2 * e1["exploitability"] < 0.03 * pot, (e1, e3)


def test_flop_value_net_tree_solves():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = _flop(engine, cfg)
    tc = TreeConfig(
        spec=SPEC, depth_streets=1, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
    )
    tree = build_tree(cfg, s.button, s.board, s.history, tc)
    ids = (tree.kind == VALUE).nonzero().flatten()
    assert len(ids) > 0 and tree.count(LEAF) == 0
    assert all(len(tree.boards[int(b)]) == 4 for b in tree.board_id[ids])
    vl = ValueLeafEvaluator(tree, ShowdownOracle(), every=2, chunk=48 * 7)  # several chunks
    assert len(vl.boards4) == 3 and vl.num_rows == 48 * len(ids)
    solver = RangeSolver(tree, _ranges(s.board, 3, torch.float32), value_leaves=vl)
    solver.solve(4)
    v, reach = solver.values(0)
    assert bool(torch.isfinite(v).all())
    # the solver's values at VALUE nodes on each turn board: enumerate the river
    bid = tree.board_id[vl.ids].tolist()  # evaluation order: grouped by turn board
    assert sum(bid[j] != bid[j - 1] for j in range(1, len(bid))) == 2
    firsts = [int(vl.ids[j]) for j in range(len(bid)) if j == 0 or bid[j] != bid[j - 1]]
    firsts += [int(vl.ids[-1])]
    assert len(firsts) >= 4
    for n in firsts:
        board = list(tree.boards[int(tree.board_id[n])])
        opp = reach[1, n] * valid_mask(board)
        ref = torch.zeros(NUM_COMBOS)
        for x in range(52):
            if x not in board:
                ok = valid_mask([x]).float()
                ref += naive_showdown((opp * ok)[None], board + [x])[0] * ok
        ref *= int(tree.contrib[n, 0]) / 44
        assert torch.allclose(v[n], ref, atol=1e-5 * float(ref.abs().max())), n
    assert bool(torch.isfinite(solver.average_strategy()).all())
    assert torch.isfinite(torch.tensor(solver.exploitability()["exploitability"]))
    # a flop solve with depth_streets 0 would need a turn net
    tc0 = TreeConfig(spec=SPEC, depth_streets=0, max_nodes=3000, leaf_mode="value_net")
    tree0 = build_tree(cfg, s.button, s.board, s.history, tc0)
    with pytest.raises(ValueError, match="4-card"):
        ValueLeafEvaluator(tree0, ShowdownOracle())


def test_leaf_budget_cost_reproduces_the_rollout_abstraction():
    engine = get_engine()
    cfg = engine.GameConfig()
    s = _flop(engine, cfg)
    budget = 3000

    def tree(**kw):
        return build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=budget, **kw))

    k = TreeConfig().num_continuations
    ro = tree()
    vn = tree(leaf_mode="value_net", leaf_budget_cost=1 + k)
    loose = tree(leaf_mode="value_net")
    full = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=10**7))
    assert ro.street_actions != full.street_actions  # the budget binds
    assert vn.street_actions == ro.street_actions
    assert vn.raise_caps == ro.raise_caps
    assert vn.chance_cards_used == ro.chance_cards_used
    dec = {_key(ro, n) for n in (ro.kind == DECISION).nonzero().flatten().tolist()}
    assert dec == {_key(vn, n) for n in (vn.kind == DECISION).nonzero().flatten().tolist()}
    leaves = {_key(ro, n) for n in (ro.kind == LEAF).nonzero().flatten().tolist()}
    assert leaves == {_key(vn, n) for n in (vn.kind == VALUE).nonzero().flatten().tolist()}
    assert vn.num_nodes == ro.num_nodes - ro.count(CONTINUATION)
    # the default cost (1 per leaf) keeps at least as many actions
    assert sum(map(len, loose.street_actions)) >= sum(map(len, ro.street_actions))
    assert loose.num_nodes <= budget


def test_search_config_leaf_mode():
    cfg = search_config({"leaf": {"mode": "value_net", "net": "x.pt", "net_every": 2}})
    assert cfg.tree.leaf_mode == "value_net" and cfg.leaf.net_every == 2
    assert search_config({}).tree.leaf_mode == "rollouts"
    with pytest.raises(ValueError, match="leaf.mode"):
        search_config({"leaf": {"mode": "magic"}})
    with pytest.raises(ValueError, match="leaf.net"):
        SearchAgent(UniformBlueprint(), {"device": "cpu", "leaf": {"mode": "value_net"}})


TINY_VN = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"mode": "value_net", "net_every": 1},
    "gadget": {"rollouts": 8},
}


def test_search_agent_with_value_net_leaves_plays_legal_hands():
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    agent = SearchAgent(UniformBlueprint(), TINY_VN, value_predictor=ShowdownOracle())
    res = run_match([agent, RandomAgent()], config, num_hands=10, seed=3)
    assert res.hands == 10
    assert int(res.seat_payoffs.sum()) == 0
    postflop = [s for s in agent.stats if "nodes" in s]
    assert postflop, "the agent never searched"
    assert not any(s.get("fallback") for s in agent.stats)
    flop = [s for s in postflop if s["street"] == 1]
    assert flop and all(s["value_leaves"] > 0 for s in flop)
    assert all(s["value_leaves"] == 0 for s in postflop if s["street"] > 1)
