"""CUDA-graph replay of DCFR iterations (RangeSolver.solve): the same solve as eager
iterations, and eager whenever a part of the iteration cannot be captured."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.gadget import Gadget, mixed_prior
from pokerbot.search.leaf_recorder import LeafStateStore, RecordingLeafProvider
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import TreeConfig, build_tree
from pokerbot.search.value_leaf import ShowdownOracle, ValueLeafEvaluator
from pokerbot.search.value_net import RiverValueNet, ValueNetConfig, ValueNetPredictor

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

SMALL = ActionSpec(
    streets=((("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",)),) * 4,
    max_raises=2,
)
BOARD = [4, 9, 14, 19, 24]
DEV = torch.device("cuda")


def _state(street: int):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[2000] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 1, BOARD, [])
    for _ in range(2 * street):
        s.apply(engine.Action.check_call())
    return cfg, s


def _ranges(board, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(board)
    return (r / r.sum(1, keepdim=True)).to(DEV, dtype)


def _pair(tree, ranges, dtype="float32", **kw):
    """The same solver eager and with CUDA graphs (``kw`` builds fresh providers)."""
    out = []
    for graph in (False, True):
        extra = {k: v() for k, v in kw.items()}
        cfg = SolverConfig(cuda_graph=graph, dtype=dtype)
        out.append(RangeSolver(tree, ranges, cfg, **extra))
    return out


def _max_rel_diff(a: RangeSolver, b: RangeSolver) -> float:
    """Largest difference of regrets, strategy sums, strategies and gadget state,
    relative to each tensor's largest entry."""
    worst = 0.0
    for k in ("regret", "strat_sum", "sigma", "g_regret", "g_enter"):
        x, y = getattr(a, k, None), getattr(b, k, None)
        if x is not None:
            worst = max(worst, float((x - y).abs().max()) / max(1e-30, float(x.abs().max())))
    return worst


def _noisy_river_predictor() -> ValueNetPredictor:
    torch.manual_seed(0)
    net = RiverValueNet(ValueNetConfig(buckets=32, width=64, layers=2, residual_head=True))
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return ValueNetPredictor(net.eval(), DEV)


def test_graph_replay_is_the_eager_iteration():
    """In float64 a replay matches an eager iteration to rounding (eager-vs-eager
    differences are about 1e-12 too: CUDA atomics accumulate in any order)."""
    cfg, s = _state(3)
    agent = int(s.current_player)
    opp = 1 - agent
    tc = TreeConfig(spec=SMALL)
    tree = build_tree(cfg, s.button, s.board, s.history, tc, searcher=agent, device=DEV)
    ranges = _ranges(s.board, 7, torch.float64)
    bp = RangeSolver(tree, ranges, SolverConfig(cuda_graph=False, dtype="float64"))
    bp.solve(8)
    terminate = bp.values(opp, bp.average_strategy(), best_response=True)[0][0]
    prior = mixed_prior(ranges[opp].cpu(), valid_mask(s.board), 0.05)
    gadget = lambda: Gadget(opp, prior, terminate)  # noqa: E731
    for kw in ({}, {"gadget": gadget}):
        eager, graphed = _pair(tree, ranges, "float64", **kw)
        assert graphed.graph_ok()
        for solver in (eager, graphed):
            solver.solve(3)  # too short to capture (as the agent's first call, min_iterations)
            solver.solve(6)  # captures at its first iteration (t = 4), then replays 5
        assert eager.graph_iterations == 0 and graphed.graph_iterations == 5
        assert _max_rel_diff(eager, graphed) < 1e-9
        for solver in (eager, graphed):
            info = solver.solve(40)
        assert info["graph_iterations"] == 39 and graphed.iterations_done == 49
        assert _max_rel_diff(eager, graphed) < 1e-9
        # the solver stays usable eagerly after the graph is freed
        graphed.solve(3)
        assert graphed.graph_iterations == 44 and graphed.iterations_done == 52


def test_graph_replay_with_value_net_leaves():
    """The river net (fused Triton head) inside the graph. Its outputs are not
    bit-reproducible on CUDA (atomic bucket sums ahead of a bf16 trunk), and
    regret matching amplifies that wherever regrets nearly tie, so two eager solves
    differ too: compare exploitability, which does not."""
    cfg, s = _state(2)  # depth-0 turn tree: value leaves at the end of the turn
    tc = TreeConfig(spec=SMALL, depth_streets=0, max_nodes=3000, leaf_mode="value_net")
    tree = build_tree(cfg, s.button, s.board, s.history, tc, device=DEV)
    ranges = _ranges(s.board, 3)
    pred = _noisy_river_predictor()
    eager, graphed = _pair(tree, ranges, value_leaves=lambda: ValueLeafEvaluator(tree, pred))
    assert graphed.terminals.value_leaves.graph_safe and graphed.graph_ok()
    for solver in (eager, graphed):
        solver.solve(100)
    assert graphed.graph_iterations == 97  # t = 1, 2 eager, 3 eager while capturing
    pot = int(s.pot)
    xe, xg = (sv.exploitability()["exploitability"] for sv in (eager, graphed))
    assert abs(xe - xg) < 0.01 * pot, (xe, xg)
    ve, vg = eager.terminals.value_leaves, graphed.terminals.value_leaves
    assert (ve.net_calls, ve.net_rows) == (vg.net_calls, vg.net_rows) and ve.net_calls > 0
    # a time budget: the loop syncs and checks the clock between replays
    info = graphed.solve(iterations=1000, time_budget=0.3)
    assert 0 < info["graph_iterations"] < info["iterations"] < 1000


def test_eager_when_part_of_the_iteration_cannot_be_captured():
    cfg, s = _state(2)
    tc = TreeConfig(spec=SMALL, depth_streets=0, max_nodes=3000, leaf_mode="value_net")
    tree = build_tree(cfg, s.button, s.board, s.history, tc, device=DEV)
    ranges = _ranges(s.board, 4)
    pred = _noisy_river_predictor()
    unsafe = {
        "oracle predictor (host syncs)": ValueLeafEvaluator(tree, ShowdownOracle()),
        "net every 2 updates": ValueLeafEvaluator(tree, pred, every=2),
        "recording provider": RecordingLeafProvider(
            ValueLeafEvaluator(tree, pred), tree, LeafStateStore()
        ),
    }
    for name, provider in unsafe.items():
        solver = RangeSolver(tree, ranges, SolverConfig(), value_leaves=provider)
        assert not solver.graph_ok(), name
        solver.solve(6)
        assert solver.graph_iterations == 0, name
    cfr_plus = RangeSolver(
        tree, ranges, SolverConfig(algorithm="cfr+"), value_leaves=ValueLeafEvaluator(tree, pred)
    )
    assert not cfr_plus.graph_ok()
