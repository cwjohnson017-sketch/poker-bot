"""Exact trunk exploitability (``exact_eval``) against a solve of the full tree.

The decomposition (trunk with ``VALUE`` leaves, every river subgame solved by
``BatchRiverSolver``, best responses backed up) must equal the exploitability
of the same profile computed by ``RangeSolver`` on the full turn-to-showdown
tree, with the river strategies copied in."""

from __future__ import annotations

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search.abstract import make_state
from pokerbot.search.combos import NUM_COMBOS, valid_mask
from pokerbot.search.exact_eval import map_sigma, node_key, trunk_exploitability
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import DECISION, VALUE, TreeConfig, build_tree
from pokerbot.search.value_leaf import FixedLeafValues

pytest.importorskip("pokerbot.search.batch_solver")

STREET = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
SPEC = ActionSpec(streets=(STREET,) * 4, max_raises=1)


def _turn_state(stacks: int = 1500):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, [4, 9, 14, 19, 24], [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    assert s.street == 2
    return cfg, s


def _ranges(board, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(2, NUM_COMBOS, generator=g) * valid_mask(board)


def _random_sigma(solver: RangeSolver, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    s = torch.rand(solver.regret.shape, generator=g) * solver.legal[:, :, None]
    return s / s.sum(1, keepdim=True)


def _trees(cfg, s, chance_cards=None):
    tv = build_tree(
        cfg,
        s.button,
        s.board,
        s.history,
        TreeConfig(spec=SPEC, depth_streets=0, leaf_mode="value_net", chance_cards=chance_cards),
    )
    tf = build_tree(
        cfg,
        s.button,
        s.board,
        s.history,
        TreeConfig(spec=SPEC, depth_streets=1, chance_cards=chance_cards),
    )
    return tv, tf


def _value_solver(tree, ranges):
    L = int((tree.kind == VALUE).sum())
    zeros = FixedLeafValues(torch.zeros(2, L, NUM_COMBOS))
    return RangeSolver(tree, ranges, SolverConfig(), value_leaves=zeros)


def _full_sigma(sv, sf, sigma_t, river):
    """Profile on the full tree: trunk from ``sigma_t``, river from the batch solves."""
    out, missing = map_sigma(sv, sf, sigma_t, strict=False)
    tf = sf.tree
    trunk_keys = {node_key(sv.tree, n) for n in sv.dec_nodes.tolist()}
    river_nodes = [
        (d, n)
        for d, n in enumerate(sf.dec_nodes.tolist())
        if int(tf.kind[n]) == DECISION and node_key(tf, n) not in trunk_keys
    ]
    assert missing == len(river_nodes)
    # river strategies of each instance, keyed by (leaf history + river history, board)
    table = {}
    for leaf_hist, board, rt, sig_r in river:
        for n in range(rt.num_nodes):
            p = int(rt.parent[n])
            if p < 0:
                continue
            key = (leaf_hist + rt.histories[p], board)
            table.setdefault(key, {})[rt.child_actions(p)[int(rt.slot[n])]] = sig_r[n]
    for d, n in river_nodes:
        probs = table[node_key(tf, n)]
        out[d] = 0
        for j, a in enumerate(tf.child_actions(n)):
            out[d, j] = probs[a].to(out)
    return out


@pytest.mark.parametrize("iterations", [0, 30])
def test_trunk_exploitability_matches_full_tree(iterations):
    cfg, s = _turn_state()
    tv, tf = _trees(cfg, s)
    ranges = _ranges(s.board)
    sv = _value_solver(tv, ranges)
    sf = RangeSolver(tf, ranges, SolverConfig())
    sigma_t = _random_sigma(sv, 1)
    river = []
    res = trunk_exploitability(
        sv, sigma_t, SPEC, cfg, iterations=iterations, mix=0.05, min_mass=0.0, keep_river=river
    )
    sigma_f = _full_sigma(sv, sf, sigma_t, river)
    root = sf.ranges
    Z = sf.pair_mass(root)
    br = []
    for p in (0, 1):
        vb, _ = sf.values(p, sigma_f, True, root)
        br.append(float((root[p] * vb[0]).sum() / Z))
    assert res["br"] == pytest.approx(br, rel=1e-4, abs=1e-3)
    assert res["exploitability"] > 0


def test_map_sigma_round_trip():
    cfg, s = _turn_state()
    tv, _ = _trees(cfg, s)
    ranges = _ranges(s.board)
    a = _value_solver(tv, ranges)
    b = _value_solver(tv, ranges)
    sig = _random_sigma(a, 2)
    out, missing = map_sigma(a, b, sig)
    assert missing == 0
    assert torch.allclose(out * a.legal[:, :, None], sig * a.legal[:, :, None])
