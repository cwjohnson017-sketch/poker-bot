"""Continual resolving below turn-end value leaves: a depth-0 turn solve
(``tree.depth_streets_turn: 0``) with a river net averaged over the river cards
stores the river roots below its ``VALUE`` leaves, exactly as a turn solve to
showdown stores its chance children, and the river search starts from them."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, blocked_sum, valid_mask
from pokerbot.search.exact_eval import map_sigma
from pokerbot.search.gadget import (
    ContinualCache,
    Gadget,
    card_value_provider,
    history_key,
    mixed_prior,
    normalise_entry,
)
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import CHANCE, DECISION, VALUE, TreeConfig, build_tree
from pokerbot.search.turn_data import RiverAveragePredictor
from pokerbot.search.value_leaf import (
    FixedLeafValues,
    ShowdownOracle,
    TurnEndLeafEvaluator,
    ValueLeafEvaluator,
)

C = NUM_COMBOS
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
    return engine, cfg, s


def _ranges(board, seed=1):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(2, C, generator=g, dtype=torch.float64) ** 2) * valid_mask(board)


def _chance_children(tree) -> list[int]:
    kids = (tree.parent >= 0) & (tree.kind[tree.parent.clamp(min=0)] == CHANCE)
    return kids.nonzero().flatten().tolist()


class _ToyRiver:
    """A nonlinear river 'net' of both ranges, the board, ``c`` and ``stack`` (0 on
    combos that hit the board)."""

    def predict(self, boards, ranges, c, stack):
        valid = vrg.board_valid(boards.to(ranges.device))[:, None, :]
        r = ranges / ranges.sum(-1, keepdim=True).clamp(min=1e-30)
        z = 300.0 * r - 200.0 * r.flip(1) + (boards.sum(1).to(r.dtype) / 200.0)[:, None, None]
        z = z + (c.to(r.dtype) / 1000.0 - stack.to(r.dtype) / 20000.0)[:, None, None]
        z = z + torch.tensor([0.1, -0.2], dtype=r.dtype, device=r.device)[None, :, None]
        return torch.tanh(z) * valid


# --------------------------------------------------------------------------- per-card values


@pytest.mark.parametrize("river", [ShowdownOracle, _ToyRiver])
def test_card_values_are_the_terms_of_the_river_average(river):
    """``card_values`` gives the per-river-card values whose chance average (weight
    1/44, the solver's ``1 / (52 - 4 - 4)``) is ``values()``, for an exact and a
    nonlinear predictor, leaves on several turn boards, a subset in any order."""
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    tc = TreeConfig(
        spec=SPEC, depth_streets=1, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
    )
    tree = build_tree(cfg, s.button, s.board, s.history, tc)  # flop tree: 3 turn boards
    vl = ValueLeafEvaluator(tree, river(), chunk=48 * 3)  # several chunks
    L = vl.num_leaves
    assert len(vl.boards4) == 3 and L > 6
    g = torch.Generator().manual_seed(4)
    reach = torch.rand(2, L, C, generator=g, dtype=torch.float64) ** 2  # also on the board
    reach[:, 0] = 0.0  # an unreached leaf
    leaves = torch.cat([torch.tensor([0]), 1 + torch.randperm(L - 1, generator=g)[: L // 2]])
    assert vl.cards_per_pair == 44
    rows = vl.net_rows
    for p in (0, 1):
        want = vl.values(p, reach)
        cards, v = vl.card_values(p, reach, leaves)
        assert v.shape == (len(leaves), 48, C) and v.dtype == torch.float64
        assert torch.equal(cards, vl.cards[leaves])
        scale = float(want.abs().max())
        torch.testing.assert_close(v.sum(1) / 44, want[leaves], rtol=1e-9, atol=1e-9 * scale)
        assert bool((v[0] == 0).all())  # unreached leaf
        # two terms from the definition
        for i, j in ((1, 0), (len(leaves) - 1, 31)):
            leaf, x = int(leaves[i]), int(cards[i, j])
            b4 = list(vl.boards4[int(vl.leaf_board[leaf])])
            assert x not in b4
            ok = valid_mask([x]).double() * valid_mask(b4)
            r = reach[:, leaf] * ok  # both reaches at the river root b4 + x
            o = vl.oop_seat
            ev = river().predict(
                torch.tensor([[*b4, x]]),
                r[[o, 1 - o]][None],
                vl.c[leaf : leaf + 1].double(),
                vl.stack[leaf : leaf + 1].double(),
            )[0, 0 if p == o else 1]
            ref = ok * blocked_sum(r[1 - p]) * (2 * int(vl.c[leaf])) * ev
            torch.testing.assert_close(v[i, j], ref, rtol=1e-9, atol=1e-9 * scale)
    assert vl.net_rows - rows == 2 * (L + len(leaves)) * 48  # values() + the subset's rows
    cards, v = vl.card_values(1, reach)  # every leaf by default
    assert v.shape == (L, 48, C) and torch.equal(cards, vl.cards)
    assert vl.card_values(0, reach, [])[1].shape == (0, 48, C)


# --------------------------------------------------------------------------- the stored entries


def _store_all(solver, tree, prefixes):
    out = {}
    for prefix, agent in prefixes:
        cache = ContinualCache()
        n = cache.store(solver, tree, agent, prefix)
        assert n == len(cache.entries)
        out[prefix] = cache
    return out


@pytest.mark.parametrize("safe", [False, True])
def test_river_roots_below_turn_end_leaves_match_the_showdown_tree(safe):
    """With a checked-down river, the depth-0 turn tree valued by
    ``ShowdownOracle`` and the turn tree solved to showdown are the same game.
    Given the same turn strategy (the showdown solve's average, mapped by
    history) and root reach, the entries stored after every turn action agree
    on keys, reaches and the opponent's values."""
    _, cfg, s = _turn()
    full = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SPEC, depth_streets=1))
    tc0 = TreeConfig(spec=SPEC, depth_streets=1, depth_streets_turn=0, leaf_mode="value_net")
    depth0 = build_tree(cfg, s.button, s.board, s.history, tc0)
    assert depth0.count(CHANCE) == 0 and depth0.count(VALUE) > 0 and full.count(VALUE) == 0
    assert full.street_actions[2] == depth0.street_actions[2]
    river = [n for n in (full.kind == DECISION).nonzero().flatten().tolist() if full.street[n] == 3]
    assert river and all(int(full.num_children[n]) == 1 for n in river)  # river is check-down
    # a chance child's history is its parent's betting history: the leaf's history is the key
    kids = _chance_children(full)
    assert kids and all(full.histories[n] == full.histories[int(full.parent[n])] for n in kids)
    # the solver's chance weight of a river card is the 1/44 of the leaf average
    assert float((full.chance_weight[kids] - 1 / 44).abs().max()) < 1e-7
    r = _ranges(s.board)
    sc = SolverConfig(dtype="float64")
    searcher = int(s.current_player)
    gadget = None
    if safe:
        opp = 1 - searcher
        prior = mixed_prior(r[opp], valid_mask(s.board), 0.05)
        term = torch.linspace(-50.0, 50.0, C, dtype=torch.float64) * valid_mask(s.board)
        gadget = Gadget(opp, prior, term)
    sol_full = RangeSolver(full, r, sc, gadget=gadget)
    sol_full.solve(30)
    sol_0 = RangeSolver(
        depth0, r, sc, gadget=gadget, value_leaves=ValueLeafEvaluator(depth0, ShowdownOracle())
    )
    assert card_value_provider(sol_0, depth0) is sol_0.terminals.value_leaves
    assert card_value_provider(sol_full, full) is None
    sigma, missing = map_sigma(sol_full, sol_0)
    assert missing == 0
    sol_0.strat_sum.copy_(sigma)
    sol_0.sigma.copy_(sigma)
    torch.testing.assert_close(sol_0.average_strategy(), sigma, rtol=0, atol=1e-12)
    if safe:
        sol_0.g_enter = sol_full.g_enter.clone()
        assert float((sol_0.g_enter - 0.5).abs().max()) > 0.1  # the gadget moved
    torch.testing.assert_close(sol_0.root_reach(), sol_full.root_reach(), rtol=0, atol=0)
    # every action of every turn decision, stored for the player who takes it
    prefixes = []
    for n in (depth0.kind == DECISION).nonzero().flatten().tolist():
        a = int(depth0.actor[n])
        for kind, amount in depth0.child_actions(n):
            prefixes.append(((*depth0.histories[n], (2, a, kind, amount)), a))
    got_full = _store_all(sol_full, full, prefixes)
    got_0 = _store_all(sol_0, depth0, prefixes)
    scale = max(
        float(e.opp_values.abs().max()) for c in got_full.values() for e in c.entries.values()
    )
    base = history_key(full.root_history)  # preflop and flop
    assert len(base) == 4 and base == history_key(depth0.root_history)
    root_total = 0
    for prefix, _ in prefixes:
        ef, e0 = got_full[prefix].entries, got_0[prefix].entries
        assert ef.keys() == e0.keys(), prefix
        assert got_0[prefix].skipped_leaves == 0
        if len(prefix) == 1:
            root_total += len(ef)
        for key, a in ef.items():
            b = e0[key]
            assert len(key[1]) == 5 and key[0][: len(base) + len(prefix)] == base + prefix
            torch.testing.assert_close(b.agent_reach, a.agent_reach, rtol=1e-12, atol=1e-15)
            torch.testing.assert_close(b.opp_reach, a.opp_reach, rtol=1e-12, atol=1e-15)
            torch.testing.assert_close(b.opp_values, a.opp_values, rtol=1e-9, atol=1e-9 * scale)
    assert root_total == 48 * depth0.count(VALUE)  # the root's actions cover every leaf


def _under(tree, prefix) -> list[int]:
    ids = (tree.kind == VALUE).nonzero().flatten().tolist()
    return [n for n in ids if tree.histories[n][: len(prefix)] == prefix]


def test_turn_end_net_and_flop_end_leaves_store_nothing():
    """A turn-end net predicts only the river average: its leaves store nothing,
    and the cache counts the leaves it skipped. So do flop-end leaves; a flop
    tree's turn-end leaves (below its chance nodes) are not this solve's next
    street, and its chance children are stored as before."""
    _, cfg, s = _turn()
    tc0 = TreeConfig(spec=SPEC, depth_streets=1, depth_streets_turn=0, leaf_mode="value_net")
    depth0 = build_tree(cfg, s.button, s.board, s.history, tc0)
    vl = TurnEndLeafEvaluator(depth0, RiverAveragePredictor(ShowdownOracle()))
    sol = RangeSolver(depth0, _ranges(s.board), SolverConfig(dtype="float64"), value_leaves=vl)
    sol.solve(3)
    assert card_value_provider(sol, depth0) is None
    a = int(s.current_player)
    cache = ContinualCache()
    for kind, amount in depth0.child_actions(0):
        prefix = ((2, a, kind, amount),)
        assert cache.store(sol, depth0, a, prefix) == 0
        assert cache.skipped_leaves == len(_under(depth0, prefix))
    assert not cache.entries
    # flop trees
    engine = get_engine()
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.check_call())
    s.apply(engine.Action.check_call())
    a = int(s.current_player)
    for depth in (0, 1):
        tc = TreeConfig(
            spec=SPEC, depth_streets=depth, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
        )
        tree = build_tree(cfg, s.button, s.board, s.history, tc)
        L = tree.count(VALUE)
        vl = (
            FixedLeafValues(torch.zeros(2, L, C, dtype=torch.float64))
            if depth == 0
            else ValueLeafEvaluator(tree, ShowdownOracle())
        )
        sol = RangeSolver(tree, _ranges(s.board), SolverConfig(dtype="float64"), value_leaves=vl)
        sol.solve(2)
        assert card_value_provider(sol, tree) is None
        prefix = ((1, a, *tree.child_actions(0)[0]),)
        kids = [n for n in _chance_children(tree) if tree.histories[n][:1] == prefix]
        cache = ContinualCache()
        n = cache.store(sol, tree, a, prefix)
        assert n == len(kids) and all(len(k[1]) == 4 for k in cache.entries)
        if depth == 0:
            assert n == 0 and cache.skipped_leaves == len(_under(tree, prefix)) > 0
        else:
            assert n > 0 and cache.skipped_leaves == 0 and len(_under(tree, prefix)) > 0


# --------------------------------------------------------------------------- the agent

# the turn has no all-in, so a turn bet and call leaves a river decision
TURN_NO_ALLIN = [["fold"], ["check_call"], ["raise", 1.0]]
ACTIONS = [[list(a) for a in BIG]] * 2 + [TURN_NO_ALLIN, [list(a) for a in BIG]]
TINY = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {
        "actions": ACTIONS,
        "max_raises": 1,
        "depth_streets": 1,
        "depth_streets_turn": 0,
        "max_nodes": 1500,
    },
    "solver": {"iterations": 4},
    "leaf": {"mode": "value_net", "net_every": 1},
    "gadget": {"terminate": "unsafe"},
}


def _turn_then_river(agent, seed):
    """The agent (out of position) acts on the turn, the opponent checks behind
    or calls, and the agent acts on the river of the same hand."""
    engine, cfg, s = _turn()
    seat = int(s.current_player)
    rng = np.random.default_rng(seed)
    agent.new_hand(seat, cfg)
    s.apply(agent.act(s, seat, rng))
    turn = agent.last_stats
    s.apply(engine.Action.check_call())
    assert s.street == 3 and int(s.current_player) == seat
    key = (history_key([h for h in s.history if int(h[0]) < 3]), tuple(s.board))
    entry = agent.cache.get(key)
    agent.act(s, seat, rng)
    return turn, agent.last_stats, entry, agent._roots[key], seat


@pytest.mark.parametrize("seed", [0, 3])  # a turn bet, a turn check
@pytest.mark.parametrize("how", ["value_predictor", "turn_value_predictor"])
def test_agent_river_search_starts_from_the_turn_end_leaves(how, seed):
    kw = {"value_predictor": ShowdownOracle()}
    if how == "turn_value_predictor":
        kw["turn_value_predictor"] = ShowdownOracle()
    agent = SearchAgent(UniformBlueprint(), TINY, **kw)
    turn, river, entry, root, seat = _turn_then_river(agent, seed)
    assert turn["street"] == 2 and turn["value_provider"] == "ValueLeafEvaluator"
    assert not turn["cached_root"] and turn["gadget"] == "unsafe"
    assert turn["cache_entries"] > 0 and turn["cache_entries"] % 48 == 0
    assert turn["cache_net_rows"] == turn["cache_entries"] and turn["cache_skipped_leaves"] == 0
    assert river["street"] == 3 and river["cached_root"] and river["gadget"] == "cache"
    assert river["value_leaves"] == 0 and river["cache_entries"] == 0
    # our range and the terminate values are the cached ones
    a, _, t = normalise_entry(entry)
    torch.testing.assert_close(root["ranges"][seat], a / a.sum().clamp(min=1e-30))
    torch.testing.assert_close(root["terminate"], t)
    assert float(entry.agent_reach.sum()) > 0


def test_agent_with_a_turn_end_net_falls_back_to_the_blueprint_on_the_river():
    turn_end = RiverAveragePredictor(ShowdownOracle())
    agent = SearchAgent(
        UniformBlueprint(), TINY, value_predictor=ShowdownOracle(), turn_value_predictor=turn_end
    )
    turn, river, entry, root, _ = _turn_then_river(agent, 0)
    assert turn["value_provider"] == "TurnEndLeafEvaluator"
    assert turn["cache_entries"] == 0 and turn["cache_skipped_leaves"] > 0
    assert entry is None and not river["cached_root"] and river["gadget"] == "unsafe"
    assert root["terminate"] is None
