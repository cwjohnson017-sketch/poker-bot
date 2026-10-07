"""(c) Tree builder: node budget, off-tree sizes, flat layout invariants."""

from __future__ import annotations

import torch

from pokerbot.engine_select import get_engine
from pokerbot.search.abstract import make_state
from pokerbot.search.tree import (
    CHANCE,
    CONTINUATION,
    DECISION,
    LEAF,
    TreeConfig,
    build_tree,
)


def _flop(engine, cfg, extra=()):
    s = make_state(engine, cfg, 0, [4, 9, 14, 19, 24], [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    for a in extra:
        s.apply(a)
    return s


def _check_layout(tree):
    n = tree.num_nodes
    assert int(tree.parent[0]) == -1
    ids = torch.arange(1, n)
    par = tree.parent[1:]
    assert bool((par < ids).all())
    assert bool((tree.depth[1:] == tree.depth[par] + 1).all())
    assert bool((tree.children[par, tree.slot[1:]] == ids).all())
    kids = tree.num_children
    assert bool(((tree.kind == DECISION) <= (kids > 0)).all())
    assert bool(((tree.kind == CHANCE) <= (kids > 0)).all())
    assert bool(((tree.kind == LEAF) <= (kids > 0)).all())


def test_node_budget_is_respected_and_reported():
    engine = get_engine()
    cfg = engine.GameConfig()
    s = _flop(engine, cfg)
    big = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=60000))
    for budget in (20000, 3000, 800):
        tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=budget))
        assert tree.num_nodes <= budget, (budget, tree.summary())
        assert tree.num_nodes < big.num_nodes
        assert str(tree.num_nodes) in tree.summary()
        _check_layout(tree)
    # sizes were dropped deepest street first: the turn lost sizes before the flop
    mid = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=20000))
    sized = lambda st: [a for a in mid.street_actions[st] if a[0] == "raise"]  # noqa: E731
    assert len(sized(2)) <= len(sized(1))
    # depth limit: leaves sit at the start of the river with k continuations each
    leaves = (mid.kind == LEAF).nonzero().flatten()
    assert len(leaves) > 0 and bool((mid.street[leaves] == 3).all())
    assert bool((mid.num_children[leaves] == 4).all())
    assert mid.count(CONTINUATION) == 4 * len(leaves)


def test_off_tree_opponent_size_is_added_where_it_happened():
    engine = get_engine()
    cfg = engine.GameConfig()
    # flop: BB (seat 1) bets 117 into 500, which is no abstract size
    s = _flop(engine, cfg, [engine.Action.raise_to(117)])
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=5000), searcher=0)
    assert int(tree.actor[tree.current_node]) == 0
    root_actions = tree.child_actions(0)
    assert (2, 117) in root_actions
    slot = root_actions.index((2, 117))
    child = int(tree.children[0, slot])
    assert int(tree.action_abstract[child]) == -1
    assert child == tree.current_node
    assert tree.path_nodes == [(0, slot)]
    # the regular abstract sizes are still there next to it
    assert sum(1 for k, _ in root_actions if k == 2) >= 2
    _check_layout(tree)


def test_turn_tree_runs_to_the_end_with_every_river_card():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, [4, 9, 14, 19, 24], [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(max_nodes=100000))
    assert tree.count(LEAF) == 0
    chance = (tree.kind == CHANCE).nonzero().flatten()
    assert len(chance) > 0
    assert bool((tree.num_children[chance] == 48).all())
    kids = tree.children[chance[0], :48]
    assert sorted(tree.deal_card[kids].tolist()) == [c for c in range(52) if c not in s.board]
    assert abs(float(tree.chance_weight[kids[0]]) - 1 / 44) < 1e-7


def test_keep_open_drops_the_reraise_on_a_tie():
    """A turn with a pot-sized open and a pot-sized re-raise: the budget keeps one
    sized action, the re-raise by default and the open with ``keep_open``."""
    from pokerbot.env.actions import ActionSpec

    engine = get_engine()
    cfg = engine.GameConfig()
    s = _flop(engine, cfg)
    street = (
        ("fold",),
        ("check_call",),
        ("raise", 1.0, "open"),
        ("raise", 1.0, "reraise"),
        ("allin",),
    )
    pre = (("fold",), ("check_call",), ("raise_x", 2.5), ("allin",))
    spec = ActionSpec(streets=(pre, street, street, street), max_raises=2)
    full = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=spec, max_nodes=10**6))
    budget = full.num_nodes - 1  # forces exactly one drop on the turn
    kept = {}
    for keep_open in (False, True):
        tc = TreeConfig(spec=spec, max_nodes=budget, keep_open=keep_open)
        tree = build_tree(cfg, s.button, s.board, s.history, tc)
        sized = [tuple(a) for a in tree.street_actions[2] if a[0] == "raise"]
        kept[keep_open] = sized
        assert tree.num_nodes <= budget
    assert kept[False] == [("raise", 1.0, "reraise")]
    assert kept[True] == [("raise", 1.0, "open")]
