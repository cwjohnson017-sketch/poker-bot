"""BatchTurnSolver: B turn subgames on one shared betting tree, with turn-end
leaves, match one RangeSolver per instance (TurnEndLeafEvaluator leaves, dense
all-in run-outs), the all-in matrices match a dense enumeration of the river,
and exploitability falls."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.batch_solver import river_tree
from pokerbot.search.batch_turn_solver import BatchTurnSolver, allin_turn_matrix, turn_tree
from pokerbot.search.combos import NUM_COMBOS, valid_mask, valid_masks
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig, allin_matrix
from pokerbot.search.tree import SHOWDOWN, VALUE, TreeConfig, build_tree
from pokerbot.search.turn_data import RiverAveragePredictor
from pokerbot.search.value_leaf import ShowdownOracle, TurnEndLeafEvaluator

C = NUM_COMBOS
STREET = (("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",))
SPEC = ActionSpec(streets=(STREET,) * 4, max_raises=2)
C_CHIPS = 250
STACK = 3000
BOARDS = [[2, 7, 19, 33], [0, 1, 2, 3], [51, 40, 30, 20]]  # [0..3]: tie-heavy
ITERS = 12


def _config(stack: int = STACK):
    engine = get_engine()
    return engine, engine.GameConfig(
        num_players=2, stacks=[stack] * 2, small_blind=50, big_blind=100
    )


def _ranges(boards, seed: int, dtype=torch.float64) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(len(boards), 2, C, generator=g, dtype=torch.float64) ** 2
    return (r * torch.stack([valid_mask(b) for b in boards])[:, None]).to(dtype)


def _range_tree(board, c: int = C_CHIPS):
    """The same turn subgame built from a real hand on ``board``."""
    engine, cfg = _config()
    s = make_state(engine, cfg, 1, board, [])
    s.apply(engine.Action.raise_to(c))
    for _ in range(3):
        s.apply(engine.Action.check_call())
    tc = TreeConfig(spec=SPEC, depth_streets=0, max_nodes=10**7, leaf_mode="value_net")
    return build_tree(cfg, 1, board, s.history, tc)


def _edge_strategy(rs: RangeSolver) -> torch.Tensor:
    """RangeSolver's average strategy as ``[N, 1326]`` child rows (root row 1)."""
    tree = rs.tree
    avg = rs.average_strategy()
    out = torch.ones(tree.num_nodes, C, dtype=avg.dtype)
    for d, node in enumerate(rs.dec_nodes.tolist()):
        first, n = int(tree.first_child[node]), int(tree.num_children[node])
        out[first : first + n] = avg[d, :n]
    return out


def _close(a, b, rel: float = 1e-6) -> bool:
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    return bool((a - b).abs().max() <= rel * max(1.0, float(b.abs().max())))


class _Counting:
    def __init__(self, inner):
        self.inner = inner
        self.kind = getattr(inner, "kind", "river")
        self.calls = 0

    def predict(self, *args):
        self.calls += 1
        return self.inner.predict(*args)


@pytest.fixture(scope="module")
def solved():
    """A batch of 3 instances and one RangeSolver per instance (turn-end leaves
    from the exact check-down river, dense all-in run-outs), ITERS iterations."""
    _, cfg = _config()
    sc = SolverConfig(dtype="float64", allin_mode="dense")
    tree = turn_tree(cfg, C_CHIPS, SPEC)
    ranges = _ranges(BOARDS, 1)
    pred = RiverAveragePredictor(ShowdownOracle())
    batch = BatchTurnSolver(tree, BOARDS, ranges, pred, sc)
    batch.solve(ITERS)
    refs = []
    for b, board in enumerate(BOARDS):
        t = _range_tree(board)
        assert torch.equal(t.kind, tree.kind) and torch.equal(t.parent, tree.parent)
        rs = RangeSolver(t, ranges[b], sc, value_leaves=TurnEndLeafEvaluator(t, pred))
        rs.solve(ITERS)
        refs.append(rs)
    return batch, refs, ranges


def test_turn_tree_layout():
    _, cfg = _config(10000)
    for c in (100, 250, 9000):
        tree = turn_tree(cfg, c, SPEC)
        assert tree.root_street == 2 and tree.contrib[0].tolist() == [c, c]
        assert int(tree.actor[0]) == 0
        kinds = set(tree.kind.tolist())
        assert VALUE in kinds and kinds <= {0, 2, 3, 6}  # decision, fold, showdown, value
        assert all(len(b) == 4 for b in tree.boards)
    with pytest.raises(ValueError):
        turn_tree(cfg, 150, SPEC)  # below a minimum raise
    with pytest.raises(ValueError, match="turn tree"):
        BatchTurnSolver(
            river_tree(cfg, 250, SPEC), [[0, 1, 2, 3]], _ranges([[0, 1, 2, 3]], 0), ShowdownOracle()
        )
    with pytest.raises(ValueError, match="boards"):
        BatchTurnSolver(
            turn_tree(cfg, 250, SPEC),
            [[0, 1, 2, 3, 4]],
            _ranges([[0, 1, 2, 3, 4]], 0),
            ShowdownOracle(),
        )


def test_average_strategy_matches_range_solver(solved):
    batch, refs, _ = solved
    assert batch.num_leaves == batch.tree.count(VALUE) > 0
    avg = batch.average_strategy()
    assert avg.shape == (batch.N, len(BOARDS), C)
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
    for p in (0, 1):
        for br in (False, True):
            v = batch.root_values(p, best_response=br)
            for b, rs in enumerate(refs):
                ref, _ = rs.values(p, best_response=br)
                assert _close(v[b], ref[0]), (p, br, b)
    other = _ranges(BOARDS, 7)
    sigma = batch.average_strategy()
    for p in (0, 1):
        v = batch.root_values(p, sigma, best_response=True, ranges=other)
        for b, rs in enumerate(refs):
            ref, _ = rs.values(p, best_response=True, root=other[b])
            assert _close(v[b], ref[0]), (p, b)


def test_card_removal_and_independent_instances(solved):
    batch, _, ranges = solved
    invalid = ~valid_masks(BOARDS)
    for p in (0, 1):
        assert batch.root_values(p, best_response=True)[invalid].abs().max() == 0
    b = 1
    pred = RiverAveragePredictor(ShowdownOracle())
    alone = BatchTurnSolver(batch.tree, [BOARDS[b]], ranges[b : b + 1], pred, batch.cfg)
    alone.solve(ITERS)
    assert _close(alone.average_strategy()[:, 0], batch.average_strategy()[:, b], 1e-9)


def test_allin_matrix_is_the_river_average():
    """The compact all-in matrices equal RangeSolver's dense matrix over the 48
    run-outs and a dense enumeration of the river with naive showdowns."""
    boards = torch.tensor([[2, 7, 19, 33], [0, 1, 2, 3]])
    valid = valid_masks(boards)
    cmap = valid.nonzero()[:, 1].view(2, -1)
    E = allin_turn_matrix(boards, cmap, torch.float64, inst_step=1, river_step=5)
    assert E.shape == (2, 1128, 1128)
    g = torch.Generator().manual_seed(3)
    for b in range(2):
        board = boards[b].tolist()
        dense = allin_matrix(tuple(board), 48, torch.device("cpu"), torch.float64, 0)  # [c, c']
        sub = dense[cmap[b][:, None], cmap[b][None, :]]
        assert torch.allclose(E[b], sub.t(), atol=1e-12)
        opp = torch.rand(C, generator=g, dtype=torch.float64) * valid[b]
        ref = torch.zeros(C, dtype=torch.float64)
        for x in range(52):
            if x not in board:
                ok = valid_mask([x]).double()
                ref += naive_showdown((opp * ok)[None], board + [x])[0] * ok
        got = torch.zeros(C, dtype=torch.float64)
        got[cmap[b]] = opp[cmap[b]] @ E[b]
        assert torch.allclose(got, ref / 44, atol=1e-12)


def test_allin_values_match_dense_enumeration():
    """Root values of a tree whose only action is all-in (call or fold):
    the all-in node's value equals the enumerated river showdown."""
    shove = ActionSpec(streets=((("fold",), ("check_call",), ("allin",)),) * 4, max_raises=1)
    _, cfg = _config()
    tree = turn_tree(cfg, 400, shove)
    assert tree.count(SHOWDOWN) >= 1
    boards = [[5, 17, 22, 38], [0, 1, 2, 3]]
    ranges = _ranges(boards, 2)
    batch = BatchTurnSolver(tree, boards, ranges, ShowdownOracle(), SolverConfig(dtype="float64"))
    batch.solve(1)
    v = batch.root_values(0, best_response=True)  # check / shove, uniform at the first iteration
    sd = next(int(n) for n in range(tree.num_nodes) if int(tree.kind[n]) == SHOWDOWN)
    stake = int(tree.contrib[sd].min())
    for b, board in enumerate(boards):
        reach = batch.reach  # after root_values: both players' reach under sigma
        row = batch._src[1][sd]
        opp = torch.zeros(C, dtype=torch.float64)
        opp[batch.cmap[b]] = reach[row, b]
        ref = torch.zeros(C, dtype=torch.float64)
        for x in range(52):
            if x not in board:
                ok = valid_mask([x]).double()
                ref += naive_showdown((opp * ok)[None], board + [x])[0] * ok
        got = torch.zeros(C, dtype=torch.float64)
        got[batch.cmap[b]] = batch.v[batch._inv[sd], b]
        assert torch.allclose(got, stake * ref / 44, atol=1e-9), b
    assert bool(torch.isfinite(v).all())


def test_exploitability_falls():
    _, cfg = _config()
    tree = turn_tree(cfg, C_CHIPS, SPEC)
    boards = [[5, 17, 22, 38], [12, 25, 38, 51]]
    pred = RiverAveragePredictor(ShowdownOracle())
    solver = BatchTurnSolver(tree, boards, _ranges(boards, 3, torch.float32), pred)
    expl, done = [], 0
    for n in (3, 10, 40):  # about 43%, 12% and 0.9% of the pot
        solver.solve(n - done)
        done = n
        expl.append(solver.exploitability()["exploitability"])
    pot = 2 * C_CHIPS
    assert bool((expl[0] > expl[1]).all() and (expl[1] > expl[2]).all()), expl
    assert float(expl[2].max()) < 0.02 * pot and float(expl[2].min()) > -1e-3 * pot, expl
    ev = solver.exploitability()["ev"]
    assert float((ev[:, 0] + ev[:, 1]).abs().max()) < 1e-3 * pot  # zero-sum


def test_river_predictor_is_averaged_and_leaf_every_caches():
    _, cfg = _config()
    tree = turn_tree(cfg, C_CHIPS, SPEC)
    boards = BOARDS[:2]
    ranges = _ranges(boards, 4)
    ranges[1, 0] = 0.0  # an empty range: finite values
    sc = SolverConfig(dtype="float64")
    a = BatchTurnSolver(tree, boards, ranges, ShowdownOracle(), sc)
    b = BatchTurnSolver(tree, boards, ranges, RiverAveragePredictor(ShowdownOracle()), sc)
    a.solve(3)
    b.solve(3)
    assert torch.equal(a.average_strategy(), b.average_strategy())
    assert bool(torch.isfinite(a.root_values(1, best_response=True)).all())
    for every, want in ((1, 20), (3, 8)):
        pred = _Counting(RiverAveragePredictor(ShowdownOracle()))
        s = BatchTurnSolver(tree, boards, ranges, pred, sc, leaf_every=every, leaf_chunk=10**6)
        s.solve(10)
        assert pred.calls == want and s.net_calls == want, (every, pred.calls)
        s.exploitability()  # 4 fresh evaluations
        assert pred.calls == want + 4
    with pytest.raises(ValueError, match="turn-end"):
        bad = _Counting(ShowdownOracle())
        bad.kind = "turn_start"
        BatchTurnSolver(tree, boards, ranges, bad, sc)
