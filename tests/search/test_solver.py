"""(b) Range CFR on small subgames: exploitability falls to a small value;
chance nodes and all-in run-outs are exact."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig, TerminalEvaluator
from pokerbot.search.tree import SHOWDOWN, TreeConfig, build_tree

SMALL = ActionSpec(
    streets=((("fold",), ("check_call",), ("raise", 1.0), ("allin",)),) * 4, max_raises=2
)
PASSIVE = ActionSpec(streets=((("fold",), ("check_call",)),) * 4, max_raises=0)


def _state(street: int, stacks: int = 2000):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, [4, 9, 14, 19, 24], [])
    for _ in range(2 * street):
        s.apply(engine.Action.check_call())
    assert s.street == street
    return cfg, s


def _ranges(board, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(board)


@pytest.mark.parametrize("algorithm", ["dcfr", "cfr+"])
def test_river_exploitability_decreases(algorithm):
    cfg, s = _state(3)
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SMALL))
    solver = RangeSolver(tree, _ranges(s.board), SolverConfig(algorithm=algorithm))
    pot = int(s.pot)
    expl = []
    done = 0
    for n in (5, 30, 150):
        solver.solve(n - done)
        done = n
        expl.append(solver.exploitability()["exploitability"])
    assert expl[0] > expl[1] > expl[2] >= -1e-3
    assert expl[2] < 0.01 * pot, expl
    e = solver.exploitability()
    assert abs(e["ev"][0] + e["ev"][1]) < 1e-3 * pot  # zero-sum


def test_chance_node_values_are_exact():
    """Check-down from the turn: the solver's root value equals enumerating the
    river with the naive showdown product."""
    cfg, s = _state(2)
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=PASSIVE))
    r = _ranges(s.board, 2).double()
    solver = RangeSolver(tree, r, SolverConfig(dtype="float64"))
    v, _ = solver.values(0)
    board = list(s.board)
    ref = torch.zeros(NUM_COMBOS, dtype=torch.float64)
    rivers = [c for c in range(52) if c not in board]
    for x in rivers:
        ok = valid_mask([x]).double()
        ref += naive_showdown((r[1] * ok)[None], board + [x])[0] * ok
    ref *= (s.pot // 2) / 44
    assert torch.allclose(v[0], ref, atol=1e-6)


def test_allin_dense_and_runouts_agree():
    cfg, s = _state(2, stacks=600)
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=SMALL))
    assert bool(((tree.kind == SHOWDOWN) & (tree.street == 2)).any())
    r = _ranges(s.board, 3).double()
    outs = []
    for mode in ("dense", "runouts"):
        sc = SolverConfig(dtype="float64", allin_mode=mode, max_runouts=48)
        te = TerminalEvaluator(tree, sc, None)
        v = torch.zeros(tree.num_nodes, NUM_COMBOS, dtype=torch.float64)
        te.evaluate(0, r[1][None].expand(tree.num_nodes, -1).contiguous(), v)
        outs.append(v)
    assert torch.allclose(outs[0], outs[1], atol=1e-6)


def test_turn_exploitability_with_chance_decreases():
    cfg, s = _state(2, stacks=800)
    spec = ActionSpec(streets=((("fold",), ("check_call",), ("allin",)),) * 4, max_raises=1)
    tree = build_tree(cfg, s.button, s.board, s.history, TreeConfig(spec=spec))
    solver = RangeSolver(tree, _ranges(s.board, 4), SolverConfig())
    solver.solve(3)
    e0 = solver.exploitability()["exploitability"]
    solver.solve(60)
    e1 = solver.exploitability()["exploitability"]
    assert e1 < e0 and e1 < 0.02 * int(s.pot), (e0, e1)
