"""On-policy leaf recording: the recorder leaves the solve unchanged, stores the
states the leaf provider was queried with, and turns them into river states."""

from __future__ import annotations

import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.leaf_recorder import LeafStateStore, RecordingLeafProvider, river_states
from pokerbot.search.solver import RangeSolver
from pokerbot.search.tree import VALUE, TreeConfig, build_tree
from pokerbot.search.value_leaf import ShowdownOracle, ValueLeafEvaluator

BIG = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
SPEC = ActionSpec(streets=(BIG,) * 4, max_raises=2)
BOARD = [4, 9, 14, 19, 24]


def _flop_tree():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    tc = TreeConfig(
        spec=SPEC, depth_streets=1, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
    )
    tree = build_tree(cfg, s.button, s.board, s.history, tc)
    g = torch.Generator().manual_seed(2)
    ranges = torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(s.board)
    return tree, ranges / ranges.sum(1, keepdim=True)


def test_recording_does_not_change_the_solve_and_stores_leaf_states():
    tree, ranges = _flop_tree()
    plain = RangeSolver(tree, ranges, value_leaves=ValueLeafEvaluator(tree, ShowdownOracle()))
    store = LeafStateStore()
    rec = RecordingLeafProvider(
        ValueLeafEvaluator(tree, ShowdownOracle()), tree, store, every=3, per_call=5
    )
    recorded = RangeSolver(tree, ranges, value_leaves=rec)
    plain.solve(12)
    recorded.solve(12)
    assert torch.allclose(plain.average_strategy(), recorded.average_strategy())
    # 2 regret updates per iteration: every 3rd of 24 calls records up to 5 leaves
    assert 0 < len(store) <= 8 * 5 and rec.recorded == len(store)
    assert rec.num_leaves == int((tree.kind == VALUE).sum())  # passes through
    t = store.tensors()
    assert t["boards"].shape[1] == 4 and t["ranges"].shape[1:] == (2, NUM_COMBOS)
    sums = t["ranges"].float().sum(-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-2)
    leaf_boards = {tuple(tree.boards[int(b)]) for b in tree.board_id[tree.kind == VALUE]}
    leaf_c = set(tree.contrib[tree.kind == VALUE, 0].tolist())
    for b, c in zip(t["boards"].tolist(), t["c"].tolist(), strict=True):
        assert tuple(b) in leaf_boards and c in leaf_c
    assert (t["stack"] == 1000 - t["c"]).all()
    # ranges never hold combos that hit the leaf board
    for b, r in zip(t["boards"].tolist(), t["ranges"].float(), strict=True):
        assert float((r * ~valid_mask(b)).abs().sum()) == 0


def test_recorded_ranges_are_the_leaf_reaches_in_oop_ip_order():
    tree, ranges = _flop_tree()
    store = LeafStateStore()
    rec = RecordingLeafProvider(
        ValueLeafEvaluator(tree, ShowdownOracle()), tree, store, every=1, per_call=10**6
    )
    solver = RangeSolver(tree, ranges, value_leaves=rec)
    # the reaches of the first update: uniform strategies
    oop = rec.oop
    r0 = solver.forward(solver.sigma.clone(), solver.root_reach())[[oop, 1 - oop]][:, rec.ids]
    mass = r0.sum(-1)
    ok = (mass > rec.min_mass).all(0)
    r0 = (r0 / mass[..., None]).transpose(0, 1)[ok]  # [K, 2, C]
    solver.solve(1)  # first call records every leaf with both reaches non-negligible
    first = store.tensors()["ranges"][: len(r0)].float()
    assert len(first) == len(r0) > 0
    d = (first[:, None] - r0[None]).abs().amax((-1, -2))  # [K, K]
    assert float(d.min(1).values.max()) < 2e-3  # each record is one of the leaf pairs


def test_river_states_mask_and_renormalise():
    g = torch.Generator().manual_seed(0)
    b4 = torch.tensor([[4, 9, 14, 19], [0, 1, 2, 3]])
    r = torch.rand(2, 2, NUM_COMBOS, generator=g)
    for i in range(2):
        r[i] *= valid_mask(b4[i].tolist())
    turn = {"boards": b4, "c": torch.tensor([250, 750]), "stack": torch.tensor([9750, 9250])}
    turn["ranges"] = r / r.sum(-1, keepdim=True)
    out = river_states(turn, rivers_per_state=3, generator=g)
    assert out["boards"].shape == (6, 5)
    for i, b in enumerate(out["boards"].tolist()):
        assert len(set(b)) == 5 and b[:4] == b4[i // 3].tolist()
        assert float((out["ranges"][i] * ~valid_mask(b)).abs().sum()) == 0
    assert torch.allclose(out["ranges"].sum(-1), torch.ones(6, 2), atol=1e-5)
    assert out["c"].tolist() == [250] * 3 + [750] * 3
    # three distinct river cards per state
    assert len({b[4] for b in out["boards"][:3].tolist()}) == 3
