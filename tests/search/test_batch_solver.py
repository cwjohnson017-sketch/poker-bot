"""BatchRiverSolver: B river subgames on one shared betting tree match one
RangeSolver per instance, converge, and keep card removal per instance."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import contributions, make_state
from pokerbot.search.batch_solver import BatchRiverSolver, river_tree
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import TreeConfig, build_tree

SPEC = ActionSpec(
    streets=((("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",)),) * 4,
    max_raises=2,
)
C_CHIPS = 250
BOARDS = [[2, 7, 19, 33, 48], [0, 1, 2, 3, 4], [51, 40, 30, 20, 10]]  # [0..4]: tie-heavy
ITERS = 40


def _config():
    engine = get_engine()
    return engine, engine.GameConfig(
        num_players=2, stacks=[10000] * 2, small_blind=50, big_blind=100
    )


def _ranges(boards, seed: int, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(len(boards), 2, NUM_COMBOS, generator=g, dtype=torch.float64)
    return (r * torch.stack([valid_mask(b) for b in boards])[:, None]).to(dtype)


def _range_tree(board, c: int = C_CHIPS):
    """The same river subgame built from a real hand on ``board``."""
    engine, cfg = _config()
    s = make_state(engine, cfg, 1, board, [])
    s.apply(engine.Action.raise_to(c))
    s.apply(engine.Action.check_call())
    for _ in range(4):
        s.apply(engine.Action.check_call())
    return build_tree(cfg, 1, board, s.history, TreeConfig(spec=SPEC, max_nodes=10**7))


def _edge_strategy(rs: RangeSolver) -> torch.Tensor:
    """RangeSolver's average strategy as ``[N, 1326]`` child rows (root row 1)."""
    tree = rs.tree
    avg = rs.average_strategy()
    out = torch.ones(tree.num_nodes, NUM_COMBOS, dtype=avg.dtype)
    for d, node in enumerate(rs.dec_nodes.tolist()):
        first, n = int(tree.first_child[node]), int(tree.num_children[node])
        out[first : first + n] = avg[d, :n]
    return out


def _close(a, b, rel: float = 1e-4) -> bool:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return bool((a - b).abs().max() <= rel * max(1.0, float(b.abs().max())))


@pytest.fixture(scope="module", params=["dcfr", "cfr+"])
def solved(request):
    """A batch of 3 instances and one RangeSolver per instance, 40 iterations each."""
    _, cfg = _config()
    sc = SolverConfig(algorithm=request.param, dtype="float64")
    tree = river_tree(cfg, C_CHIPS, SPEC)
    ranges = _ranges(BOARDS, 1)
    batch = BatchRiverSolver(tree, BOARDS, ranges, sc)
    batch.solve(ITERS)
    refs = []
    for b, board in enumerate(BOARDS):
        t = _range_tree(board)
        assert torch.equal(t.kind, tree.kind) and torch.equal(t.parent, tree.parent)
        rs = RangeSolver(t, ranges[b], sc)
        rs.solve(ITERS)
        refs.append(rs)
    return batch, refs, ranges


def test_average_strategy_matches_range_solver(solved):
    batch, refs, _ = solved
    avg = batch.average_strategy()
    assert avg.shape == (batch.N, len(BOARDS), NUM_COMBOS)
    for b, rs in enumerate(refs):
        assert _close(avg[:, b], _edge_strategy(rs)), b


def test_values_and_exploitability_match_range_solver(solved):
    batch, refs, _ = solved
    ex = batch.exploitability()
    assert ex["pot"] == 2 * C_CHIPS
    for b, rs in enumerate(refs):
        ref = rs.exploitability()
        assert _close(ex["br"][b], ref["br"]) and _close(ex["ev"][b], ref["ev"]), b
        assert _close(ex["exploitability"][b], ref["exploitability"])
        assert _close(ex["nashconv"][b], ref["nashconv"])
    for p in (0, 1):
        for br in (False, True):
            v = batch.root_values(p, best_response=br)
            for b, rs in enumerate(refs):
                ref, _ = rs.values(p, best_response=br)
                assert _close(v[b], ref[0]), (p, br, b)


def test_root_values_with_other_ranges(solved):
    """``ranges`` overrides the root reach, as ``RangeSolver.values(root=...)``."""
    batch, refs, _ = solved
    other = _ranges(BOARDS, 7)
    sigma = batch.average_strategy()
    for p in (0, 1):
        for br in (False, True):
            v = batch.root_values(p, sigma, best_response=br, ranges=other)
            for b, rs in enumerate(refs):
                ref, _ = rs.values(p, best_response=br, root=other[b])
                assert _close(v[b], ref[0]), (p, br, b)


def test_card_removal_and_independent_instances(solved):
    batch, _, ranges = solved
    invalid = ~torch.stack([valid_mask(b) for b in BOARDS])
    for p in (0, 1):
        for br in (False, True):
            assert batch.root_values(p, best_response=br)[invalid].abs().max() == 0
    b = 1
    alone = BatchRiverSolver(batch.tree, [BOARDS[b]], ranges[b : b + 1], batch.cfg)
    alone.solve(ITERS)
    assert _close(alone.average_strategy()[:, 0], batch.average_strategy()[:, b], 1e-9)
    for p in (0, 1):
        v = alone.root_values(p, best_response=True)[0]
        assert _close(v, batch.root_values(p, best_response=True)[b], 1e-9)


def test_check_down_values_are_exact_showdowns():
    """Ties and card removal: a check-down river against the dense reference."""
    passive = ActionSpec(streets=((("fold",), ("check_call",)),) * 4, max_raises=0)
    _, cfg = _config()
    tree = river_tree(cfg, 300, passive)
    boards = [[0, 1, 2, 3, 4], [2, 6, 10, 14, 18], [2, 7, 19, 33, 48]]  # quads, straight flush
    ranges = _ranges(boards, 5)
    solver = BatchRiverSolver(tree, boards, ranges, SolverConfig(dtype="float64"))
    for p in (0, 1):
        v = solver.root_values(p)
        for b, board in enumerate(boards):
            ref = 300 * naive_showdown(ranges[b, 1 - p][None], board)[0]
            assert torch.allclose(v[b], ref, atol=1e-9), (p, b)


def test_exploitability_falls_below_one_percent_of_pot():
    _, cfg = _config()
    tree = river_tree(cfg, C_CHIPS, SPEC)
    boards = [[5, 17, 22, 38, 44], [12, 25, 38, 51, 3], [0, 4, 8, 13, 26], [9, 10, 11, 30, 31]]
    solver = BatchRiverSolver(tree, boards, _ranges(boards, 3, torch.float32))
    expl, done = [], 0
    for n in (5, 30, 300):
        solver.solve(n - done)
        done = n
        expl.append(solver.exploitability()["exploitability"])
    pot = 2 * C_CHIPS
    assert bool((expl[0] > expl[1]).all() and (expl[1] > expl[2]).all()), expl
    assert float(expl[2].max()) < 0.01 * pot and float(expl[2].min()) > -1e-3, expl
    ev = solver.exploitability()["ev"]
    assert float((ev[:, 0] + ev[:, 1]).abs().max()) < 1e-3 * pot  # zero-sum


@pytest.mark.parametrize("c", [100, 250, 9000])
def test_river_tree(c):
    _, cfg = _config()
    tree = river_tree(cfg, c, SPEC)
    assert tree.root_street == 3
    assert tree.contrib[0].tolist() == [c, c]
    assert int(tree.actor[0]) == 0


def test_river_tree_rejects_impossible_pots():
    _, cfg = _config()
    with pytest.raises(ValueError):
        river_tree(cfg, 150, SPEC)  # below a minimum raise
    with pytest.raises(ValueError):
        river_tree(cfg, 10000, SPEC)  # all-in preflop: no river decision


def test_rejects_trees_with_chance_nodes():
    engine, cfg = _config()
    s = make_state(engine, cfg, 1, [2, 7, 19, 33], [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    assert s.street == 2 and contributions(s, cfg) == [100, 100]
    turn = build_tree(cfg, 1, [2, 7, 19, 33], s.history, TreeConfig(spec=SPEC))
    with pytest.raises(ValueError, match="river tree"):
        BatchRiverSolver(turn, [[2, 7, 19, 33, 48]], _ranges([[2, 7, 19, 33, 48]], 0))
