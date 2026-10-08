"""The blueprint's strategy on a search tree (``tree_policy``) and the gadget's
terminate-value modes (``gadget.terminate``).

The batched path (one ``VecNLHE`` slot per node) must equal the per-node
``policy_matrix`` path on every combo that can reach the node; terminate
values computed in the search tree must be consistent with the solver's own
values, the best response must dominate the on-policy values, and a safe
re-solve with them must keep every opponent combo below its terminate value.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.eval.masking import MaskedState
from pokerbot.search import SearchAgent, UniformBlueprint, search_config
from pokerbot.search import vec_rollouts as vr
from pokerbot.search.abstract import RAISE, CardView, legal_options
from pokerbot.search.blueprint import make_blueprint, policy_matrix
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.gadget import gadget_entry_values, gadget_violation, tree_terminate_values
from pokerbot.search.solver import RangeSolver
from pokerbot.search.spot_eval import capture_solvers
from pokerbot.search.tree import DECISION, LEAF, TreeConfig, build_tree
from pokerbot.search.tree_policy import blueprint_profile
from pokerbot.search.value_leaf import ShowdownOracle, ValueLeafEvaluator

C = NUM_COMBOS


def reference_profile(solver, bp, own_board=True):
    """The per-node implementation ``spot_eval.blueprint_profile`` used to have,
    with each node queried on its own board (``own_board=False``: on the stored
    state's board, the old behaviour)."""
    tree = solver.tree
    sigma = solver.uniform.clone()
    dropped = torch.zeros(solver.Dn, C, dtype=solver.dtype)
    for d, node in enumerate(solver.dec_nodes.tolist()):
        kind = int(tree.kind[node])
        if kind == LEAF:
            sigma[d] = 0
            sigma[d, 0] = 1
            continue
        if kind != DECISION:
            continue
        n = int(tree.num_children[node])
        st = tree.states[node]
        if own_board:
            st = CardView(st, tree.boards[int(tree.board_id[node])])
        P = policy_matrix(bp, st, int(tree.actor[node])).to(solver.dtype).expand(C, -1)
        col = {(o.kind, o.amount): o.index for o in legal_options(st, bp.spec)}
        S = torch.zeros(C, n, dtype=solver.dtype)
        for j, (k, amt) in enumerate(tree.child_actions(node)):
            i = col.get((int(k), int(amt) if int(k) == RAISE else 0))
            if i is not None:
                S[:, j] = P[:, i]
        tot = S.sum(1, keepdim=True)
        dropped[d] = (1 - tot[:, 0]).clamp(min=0)
        S = torch.where(tot > 0, S / tot.clamp(min=1e-30), torch.full_like(S, 1.0 / n))
        sigma[d] = 0
        sigma[d, :n] = S.t()
    return sigma, dropped


def _game(bp):
    engine = get_engine()
    g = bp.game
    cfg = engine.GameConfig(
        num_players=2, stacks=g["stacks"], small_blind=g["small_blind"], big_blind=g["big_blind"]
    )
    return engine, cfg


def _flop_state(engine, cfg, seed=4):
    deck = np.random.default_rng(seed).permutation(52).tolist()
    s = engine.GameState.new_hand(cfg, 0, deck)
    s.apply(engine.Action.check_call())  # the button limps
    s.apply(engine.Action.check_call())  # the big blind checks
    assert int(s.street) == 1
    return s


def _flop_solver(bp, engine, cfg, s, max_nodes=900):
    tc = TreeConfig(
        spec=bp.spec, depth_streets=1, leaf_mode="value_net", max_nodes=max_nodes, chance_cards=3
    )
    tree = build_tree(cfg, s.button, s.board, s.history, tc, searcher=int(s.current_player))
    g = torch.Generator().manual_seed(3)
    ranges = torch.rand(2, C, generator=g) * valid_mask(s.board)
    ranges = ranges / ranges.sum(1, keepdim=True)
    return RangeSolver(tree, ranges, value_leaves=ValueLeafEvaluator(tree, ShowdownOracle()))


def _assert_same_on_reachable(solver, got, want, atol):
    """Rows of combos valid on each decision node's board (the others have no reach)."""
    tree = solver.tree
    for d, node in enumerate(solver.dec_nodes.tolist()):
        ok = valid_mask(tree.boards[int(tree.board_id[node])])
        torch.testing.assert_close(got[d][:, ok], want[d][:, ok], rtol=0, atol=atol)


@pytest.fixture(scope="module")
def distilled(tiny_distilled_run):
    bp = make_blueprint(f"neural:{tiny_distilled_run}")
    assert vr.supports(bp)
    return bp


def test_batched_profile_matches_per_node_policies(distilled):
    engine, cfg = _game(distilled)
    s = _flop_state(engine, cfg)
    solver = _flop_solver(distilled, engine, cfg, s)
    n_dec = int((solver.tree.kind == DECISION).sum())
    assert n_dec > 20  # flop and turn decisions on several turn cards
    sigma, dropped = blueprint_profile(solver, distilled)
    ref_sigma, ref_dropped = reference_profile(solver, distilled)
    _assert_same_on_reachable(solver, sigma, ref_sigma, atol=2e-5)
    _assert_same_on_reachable(solver, dropped[:, None], ref_dropped[:, None], atol=2e-5)
    assert torch.allclose(sigma.sum(1), torch.ones(solver.Dn, C, dtype=sigma.dtype), atol=1e-5)


def test_turn_nodes_are_queried_on_their_own_board(distilled):
    """Stored states below a chance node carry the template's turn card, so a
    profile built from them (the old evaluation) differs on the turn."""
    engine, cfg = _game(distilled)
    s = _flop_state(engine, cfg)
    solver = _flop_solver(distilled, engine, cfg, s)
    tree = solver.tree
    turn = [
        d
        for d, n in enumerate(solver.dec_nodes.tolist())
        if len(tree.boards[int(tree.board_id[n])]) == 4
    ]
    stale = [
        d
        for d in turn
        if tuple(tree.states[int(solver.dec_nodes[d])].board)
        != tuple(tree.boards[int(tree.board_id[int(solver.dec_nodes[d])])])
    ]
    assert turn and stale, "expected turn nodes that share a template state"
    sigma, _ = blueprint_profile(solver, distilled)
    old, _ = reference_profile(solver, distilled, own_board=False)
    gap = max(float((sigma[d] - old[d]).abs().max()) for d in stale)
    assert gap > 1e-3  # the template's card gives another strategy


def test_profile_falls_back_to_policy_matrix():
    bp = UniformBlueprint()
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[2000, 2000], small_blind=50, big_blind=100)
    s = _flop_state(engine, cfg)
    solver = _flop_solver(bp, engine, cfg, s, max_nodes=600)
    assert not vr.supports(bp)
    sigma, dropped = blueprint_profile(solver, bp)
    ref_sigma, ref_dropped = reference_profile(solver, bp)
    torch.testing.assert_close(sigma, ref_sigma)
    torch.testing.assert_close(dropped, ref_dropped)


def test_tree_terminate_values_are_the_solver_values(distilled):
    engine, cfg = _game(distilled)
    s = _flop_state(engine, cfg)
    solver = _flop_solver(distilled, engine, cfg, s)
    sigma = blueprint_profile(solver, distilled)[0]
    opp = 1 - int(s.current_player)
    t_br = tree_terminate_values(solver, sigma, opp, best_response=True)
    t_ev = tree_terminate_values(solver, sigma, opp, best_response=False)
    ok = valid_mask(s.board)
    assert bool((t_br[ok] >= t_ev[ok] - 1e-4).all())  # a best response dominates per combo
    assert float((t_br - t_ev)[ok].max()) > 0  # and gains somewhere against this blueprint
    v, _ = solver.values(opp, sigma, best_response=False, root=solver.ranges)
    torch.testing.assert_close(t_ev, v[0])


SEARCH = {
    "device": "cpu",
    "time_budget": 1e6,
    "fallback_on_error": False,
    "tree": {"actions": "blueprint", "max_nodes": 900, "chance_cards": 3},
    "solver": {"max_runouts": 6},
    "leaf": {"mode": "value_net", "net_every": 1},
    "gadget": {"rollouts": 8},
}


def _search(bp, s, cfg, terminate, iters):
    over = {**SEARCH, "min_iterations": iters}
    over["solver"] = {**SEARCH["solver"], "iterations": iters}
    over["gadget"] = {**SEARCH["gadget"], "terminate": terminate}
    agent = SearchAgent(bp, search_config(None, **over), value_predictor=ShowdownOracle())
    seat = int(s.current_player)
    rng = np.random.default_rng(0)
    with capture_solvers() as got:
        agent.new_hand(seat, cfg)
        agent.act(MaskedState(s, seat, cfg, rng), seat, rng)
    return agent, got[-1]


@pytest.mark.parametrize("terminate", ["rollouts", "blueprint", "blueprint_br", "unsafe"])
def test_agent_terminate_modes(distilled, terminate):
    engine, cfg = _game(distilled)
    s = _flop_state(engine, cfg)
    agent, solver = _search(distilled, s, cfg, terminate, iters=4)
    st = agent.last_stats
    assert st["gadget"] == terminate
    assert (solver.gadget is None) == (terminate == "unsafe")
    if terminate in ("blueprint", "blueprint_br"):
        # T is the opponent's value against the blueprint in this very tree
        plain = RangeSolver(
            solver.tree, solver.ranges, solver.cfg, value_leaves=solver.terminals.value_leaves
        )
        sigma = blueprint_profile(plain, distilled)[0]
        opp = 1 - int(s.current_player)
        want = tree_terminate_values(plain, sigma, opp, terminate == "blueprint_br")
        torch.testing.assert_close(solver.g_term, want.to(solver.g_term), rtol=1e-4, atol=1e-3)
        assert st["terminate_seconds"] > 0


def test_blueprint_br_gadget_is_safe(distilled):
    """Against best-response terminate values the re-solve gives no opponent
    combo more than it could get against the blueprint (Burch et al.)."""
    engine, cfg = _game(distilled)
    s = _flop_state(engine, cfg)
    _, solver = _search(distilled, s, cfg, "blueprint_br", iters=300)
    pot = int(s.pot)
    viol = gadget_violation(solver)
    assert viol < 0.01 * pot, viol
    entry = gadget_entry_values(solver)
    worst = float(((entry - solver.g_term) * solver.g_prior).max())
    assert worst < 0.01 * pot
