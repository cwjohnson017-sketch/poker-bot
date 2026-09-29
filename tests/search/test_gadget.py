"""(d) Safe resolving: no opponent combo gains more by entering the re-solved
subgame than its terminate value; and depth-limit leaf rollouts are unbiased."""

from __future__ import annotations

import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.blueprint import TabularBlueprintFromCallable
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.gadget import Gadget, gadget_entry_values, gadget_violation, mixed_prior
from pokerbot.search.leaf import LeafConfig, build_leaf_rollouts
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import LEAF, TreeConfig, build_tree

SMALL = ActionSpec(
    streets=((("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",)),) * 4,
    max_raises=2,
)


def _river(stacks=2000):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 1, [4, 9, 14, 19, 24], [])
    for _ in range(6):
        s.apply(engine.Action.check_call())
    return engine, cfg, s


def test_safe_gadget_never_lets_the_opponent_gain_by_entering():
    engine, cfg, s = _river()
    agent = int(s.current_player)
    opp = 1 - agent
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SMALL), searcher=agent)
    g = torch.Generator().manual_seed(7)
    ranges = torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(s.board)
    ranges = ranges / ranges.sum(1, keepdim=True)
    # "blueprint": a lightly solved strategy; T = opponent best-response values against it
    bp = RangeSolver(tree, ranges, SolverConfig())
    bp.solve(8)
    sigma_bp = bp.average_strategy()
    v_bp, _ = bp.values(opp, sigma_bp, best_response=True)
    terminate = v_bp[0]
    prior = mixed_prior(ranges[opp], valid_mask(s.board), 0.05)
    safe = RangeSolver(tree, ranges, SolverConfig(), gadget=Gadget(opp, prior, terminate))
    pot = int(s.pot)
    viol = []
    for n in (5, 300):
        safe.solve(n)
        viol.append(gadget_violation(safe))
    assert viol[-1] < 0.002 * pot, viol
    assert viol[-1] <= viol[0] + 1e-6
    entry = gadget_entry_values(safe)
    per_combo = (entry - terminate).clamp(min=0) * prior
    assert float(per_combo.max()) < 0.002 * pot
    # the blueprint itself is feasible, so the re-solve does at least as well against a BR
    ev_bp = float((prior * v_bp[0]).sum())
    ev_new = float((prior * torch.maximum(entry, terminate)).sum())
    assert ev_new <= ev_bp + 0.002 * pot


def test_leaf_rollouts_estimate_the_continuation_value():
    """Turn root, depth limit at the river, a blueprint that always checks: the
    rollout estimate of the leaf matches the exact check-down over all rivers."""
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[2000, 2000], small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, [4, 9, 14, 19, 24], [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    passive = ActionSpec(streets=((("fold",), ("check_call",)),) * 4, max_raises=0)
    tree = build_tree(
        cfg,
        s.button,
        s.board,
        s.history,
        TreeConfig(spec=passive, depth_streets=0, num_continuations=1),
    )
    assert tree.count(LEAF) == 1
    bp = TabularBlueprintFromCallable(lambda st, p: {1: 1.0}, card_independent=True)
    lc = LeafConfig(strategies=["blueprint"], rollouts=3000, max_total_rollouts=10**6, explore=0)
    rs = build_leaf_rollouts(tree, bp, cfg, lc, engine, torch.device("cpu"))
    g = torch.Generator().manual_seed(3)
    r = torch.rand(2, NUM_COMBOS, generator=g, dtype=torch.float64) * valid_mask(s.board)
    solver = RangeSolver(tree, r, SolverConfig(dtype="float64"), rollouts=rs)
    v, _ = solver.values(0)
    board = list(s.board)
    ref = torch.zeros(NUM_COMBOS, dtype=torch.float64)
    for x in (c for c in range(52) if c not in board):
        ok = valid_mask([x]).double()
        ref += naive_showdown((r[1] * ok)[None], board + [x])[0] * ok
    ref *= (s.pot // 2) / 44
    pot = float(s.pot)
    ev = float((r[0] * v[0]).sum() / r[0].sum())
    ev_ref = float((r[0] * ref).sum() / r[0].sum())
    assert abs(ev - ev_ref) < 0.02 * abs(ev_ref), (ev, ev_ref)
    err = (v[0] - ref).abs().mean() / (pot / 2 * float(r[1].sum()))
    assert float(err) < 0.02, float(err)  # mean error per combo, as a fraction of the stake
