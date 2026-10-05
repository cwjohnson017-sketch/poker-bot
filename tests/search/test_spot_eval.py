"""Search exploitability with an exact river (``spot_eval``): one tiny spot end to
end on the CPU (uniform blueprint, ``ShowdownOracle`` leaves), the blueprint
profile, strict strategy mapping, the trunk check, and resuming a run."""

from __future__ import annotations

import json

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import UniformBlueprint
from pokerbot.search import spot_eval as se
from pokerbot.search.abstract import legal_options
from pokerbot.search.blueprint import policy_matrix
from pokerbot.search.combos import NUM_COMBOS
from pokerbot.search.exact_eval import map_sigma, node_key
from pokerbot.search.solver import RangeSolver
from pokerbot.search.tree import DECISION, VALUE, TreeConfig, build_tree
from pokerbot.search.value_leaf import FixedLeafValues, ShowdownOracle

pytest.importorskip("pokerbot.search.batch_solver")

PRE = (("fold",), ("check_call",), ("raise_x", 2.5), ("allin",))
STREET = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
RIVER = (("fold",), ("check_call",), ("allin",))
SPEC = ActionSpec(streets=(PRE, STREET, STREET, RIVER), max_raises=1)
RICH_STREET = (("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",))
RICH = ActionSpec(streets=(PRE, RICH_STREET, RICH_STREET, RIVER), max_raises=2)

TINY = se.EvalSettings(
    iters=4,
    max_nodes=300,
    river_iters=20,
    river_batch=128,
    device="cpu",
    eval_max_runouts=None,  # the search's 6 sampled run-outs: cheap on the CPU
    search={
        "tree": {"chance_cards": 2},
        "solver": {"max_runouts": 6},
        "gadget": {"rollouts": 8},
    },
)


def _config():
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100)
    return engine, cfg


@pytest.fixture(scope="module")
def spot_result():
    torch.manual_seed(0)
    engine, cfg = _config()
    (spot,) = se.exploit_spots(engine, cfg, 1, 5, ["btn_vs_lead"])
    keep: dict = {}
    res = se.evaluate_spot(
        spot, UniformBlueprint(SPEC), cfg, TINY, ShowdownOracle, keep=keep, log=lambda m: None
    )
    return res, keep


def test_exploit_spots_match_search_noise():
    engine, cfg = _config()
    spots = se.exploit_spots(engine, cfg, 2, 5)
    assert [s.label for s in spots] == [
        f"board{b} {name}" for b in range(2) for name in se.SPOT_NAMES.values()
    ]
    assert all(int(s.state.street) == 1 for s in spots)
    assert [int(s.state.pot) for s in spots[:3]] == [500, 500, 750]
    assert int(spots[0].state.current_player) == 1  # BB first
    assert int(spots[1].state.current_player) == 0  # BTN vs check
    lead = list(spots[2].state.history)[-1][2]
    assert (int(lead.kind), int(lead.amount)) == (2, 250)
    more = se.exploit_spots(engine, cfg, 3, 5, ["bb_first"])
    assert [list(s.state.board) for s in more[:2]] == [
        list(s.state.board) for s in spots if s.spot_type == "bb_first"
    ]
    with pytest.raises(ValueError, match="spot types"):
        se.exploit_spots(engine, cfg, 1, 5, ["river"])


def test_search_overrides_share_the_trunk():
    ro = se.search_overrides(TINY)
    vn = se.search_overrides(TINY, "value_net", "net.pt", 2, None, 4)
    assert ro["leaf"] == {"seed": TINY.leaf_seed}
    assert vn["leaf"] == {"mode": "value_net", "net": "net.pt", "net_every": 2}
    assert vn["tree"]["leaf_budget_cost"] == 5 and "leaf_budget_cost" not in ro["tree"]
    assert vn["tree"]["chance_cards"] == ro["tree"]["chance_cards"] == 2  # extra overrides
    assert ro["solver"] == {"iterations": TINY.iters, "max_runouts": 6}
    budget = se.search_overrides(TINY, "value_net", None, 1, 3.0, 4)
    assert budget["time_budget"]["flop"] == 3.0
    assert "iterations" not in budget["solver"] and "min_iterations" not in budget
    assert se.parse_variant("value_net_every3") == {"net_every": 3, "budget": False, "turn": False}
    assert se.parse_variant("value_net_budget")["budget"]
    assert se.parse_variant("value_net_turn") == {"net_every": 1, "budget": False, "turn": True}
    assert se.parse_variant("value_net_turn_every2")["turn"]
    assert se.parse_variant("value_net_turn_budget")["budget"]
    with pytest.raises(ValueError):
        se.parse_variant("value_net_every0")


def test_spot_end_to_end(spot_result):
    res, keep = spot_result
    assert res["trunk_check"]["ok"]
    assert res["trunk_check"]["ranges_max_diff"] == 0.0
    assert res["leaves"] > 0 and res["leaves"] == res["trunk_check"]["leaves"]
    for name in se.BASE_PROFILES:
        p = res["profiles"][name]
        assert torch.isfinite(torch.tensor([p["chips"], *p["br"]])).all(), name
        assert p["chips"] >= -1e-6, (name, p)
        assert p["mbb"] == pytest.approx(10 * p["chips"])
        assert p["instances"] + p["skipped"] == 48 * res["leaves"]
    assert res["reference_sanity"]["chips"] >= -1e-6
    for name in ("value_net", "rollout"):
        st = res["searches"][name]
        assert st["iterations"] == TINY.iters and st["total_seconds"] > 0
    assert res["searches"]["value_net"]["value_leaves"] == res["leaves"]
    assert res["nodes"]["rollout"] > res["nodes"]["value_net"]  # + continuations
    json.dumps(res)  # JSON-ready
    md = se.render_markdown({"settings": TINY.comparable(), "meta": {}, "spots": [res]})
    assert "| **mean** |" in md and "board0 BTN vs 1/2 lead" in md


def test_blueprint_sigma_rows_sum_to_one(spot_result):
    _, keep = spot_result
    solver = keep["runs"]["value_net"].solver
    sig = keep["sigmas"]["blueprint"]
    tree = solver.tree
    off_tree = 0
    for d, n in enumerate(solver.dec_nodes.tolist()):
        assert int(tree.kind[n]) == DECISION
        k = int(tree.num_children[n])
        assert torch.allclose(sig[d, :k].sum(0), torch.ones(NUM_COMBOS), atol=1e-5)
        assert float(sig[d, k:].abs().sum()) == 0
        # the uniform blueprint: uniform over its own options, 0 on the off-tree 1/2 lead
        st = tree.states[n]
        opts = {(o.kind, o.amount) for o in legal_options(st, SPEC)}
        acts = [(kd, amt if kd == 2 else 0) for kd, amt in tree.child_actions(n)]
        want = torch.tensor([float(a in opts) for a in acts])
        want = want / want.sum()
        assert torch.allclose(sig[d, :k], want[:, None].expand(k, NUM_COMBOS), atol=1e-6)
        off_tree += int(len(opts) < k)
    assert off_tree == 1  # the BB's root, where the lead is forced in
    assert float(keep["dropped"].abs().max()) < 1e-6


def test_rollout_strategy_mapping_is_strict_and_complete(spot_result):
    _, keep = spot_result
    src = keep["runs"]["rollout"].solver
    dst = keep["runs"]["value_net"].solver
    mapped, missing = map_sigma(src, dst, strict=True)
    assert missing == 0
    assert torch.equal(mapped, keep["sigmas"]["rollout"])
    avg = src.average_strategy()
    index = {node_key(src.tree, n): d for d, n in enumerate(src.dec_nodes.tolist())}
    checked = 0
    for d, n in enumerate(dst.dec_nodes.tolist()):
        ds = index[node_key(dst.tree, n)]  # every value-net decision node has a source
        s_acts = src.tree.child_actions(int(src.dec_nodes[ds]))
        for j, a in enumerate(dst.tree.child_actions(n)):
            assert torch.equal(mapped[d, j], avg[ds, s_acts.index(a)])
            checked += 1
    assert checked == int(dst.legal.sum())


class _RandomBlueprint:
    """Card-dependent random policy over ``RICH`` (seeded by the public history)."""

    def __init__(self, spec):
        self.spec = spec

    def policy(self, state, player):  # pragma: no cover - policy_combos is used
        raise NotImplementedError

    def policy_combos(self, state, player):
        g = torch.Generator().manual_seed(hash(str(list(state.history))) & 0xFFFF)
        return torch.rand(NUM_COMBOS, self.spec.num_actions, generator=g)


def test_blueprint_profile_renormalises_dropped_sizes():
    engine, cfg = _config()
    (spot,) = se.exploit_spots(engine, cfg, 1, 5, ["bb_first"])
    s = spot.state
    tc = TreeConfig(spec=RICH, max_nodes=400, chance_cards=2, leaf_mode="value_net")
    tree = build_tree(cfg, s.button, s.board, s.history, tc)
    assert tree.street_actions[2] != list(RICH.streets[2])  # the budget dropped turn sizes
    L = int((tree.kind == VALUE).sum())
    solver = RangeSolver(
        tree, torch.ones(2, NUM_COMBOS), value_leaves=FixedLeafValues(torch.zeros(2, L, 1326))
    )
    bp = _RandomBlueprint(RICH)
    sig, dropped = se.blueprint_profile(solver, bp)
    valid = solver.ranges[0] > 0
    full = 0
    for d, n in enumerate(solver.dec_nodes.tolist()):
        k = int(tree.num_children[n])
        assert torch.allclose(sig[d, :k].sum(0), torch.ones(NUM_COMBOS), atol=1e-5)
        st = tree.states[n]
        n_bp = len(legal_options(st, RICH))
        if n_bp == k:  # nothing dropped here: exactly the blueprint
            full += 1
            assert float(dropped[d].abs().max()) < 1e-6
            P = policy_matrix(bp, st, int(tree.actor[n]))
            col = {(o.kind, o.amount): o.index for o in legal_options(st, RICH)}
            for j, (kd, amt) in enumerate(tree.child_actions(n)):
                i = col[(kd, amt if kd == 2 else 0)]
                assert torch.allclose(sig[d, j][valid], P[valid, i], atol=1e-6)
        else:
            assert float(dropped[d][valid].min()) > 0
    assert 0 < full < len(solver.dec_nodes)
    assert se.offtree_mass(solver, sig, dropped) > 0


def test_trunk_check_catches_a_different_abstraction():
    engine, cfg = _config()
    (spot,) = se.exploit_spots(engine, cfg, 1, 5, ["bb_first"])
    s = spot.state

    def tree(**kw):
        tc = TreeConfig(spec=RICH, max_nodes=1000, chance_cards=2, **kw)
        return build_tree(cfg, s.button, s.board, s.history, tc)

    ro = tree()
    same = tree(leaf_mode="value_net", leaf_budget_cost=5)
    loose = tree(leaf_mode="value_net")  # a leaf costs 1 node: more sizes survive
    assert se.trunk_differences(ro, same) == []
    assert loose.street_actions != ro.street_actions
    assert se.trunk_differences(ro, loose)


def test_run_evaluation_resumes(tmp_path, monkeypatch, spot_result):
    res, _ = spot_result
    calls = []

    def fake(spot, *a, **k):
        calls.append(spot.label)
        return {**res, "label": spot.label}

    monkeypatch.setattr(se, "evaluate_spot", fake)
    spots = [se.Spot(f"s{i}", 0, "btn_vs_lead", None) for i in range(3)]
    meta = {"blueprint": "uniform", "leaf_model": "oracle:showdown"}
    out, md = tmp_path / "e.json", tmp_path / "e.md"
    se.run_evaluation(spots[:2], None, None, TINY, dict(meta), out, md, log=lambda m: None)
    assert calls == ["s0", "s1"]
    data = se.run_evaluation(
        spots, None, None, TINY, dict(meta), out, md, resume=True, log=lambda m: None
    )
    assert calls == ["s0", "s1", "s2"]
    assert [s["label"] for s in data["spots"]] == ["s0", "s1", "s2"]
    assert json.loads(out.read_text())["spots"][2]["label"] == "s2"
    assert "| s2 |" in md.read_text(encoding="utf-8")
    other = se.EvalSettings(**{**TINY.__dict__, "iters": 5})
    with pytest.raises(ValueError, match="other settings"):
        se.run_evaluation(
            spots, None, None, other, dict(meta), out, md, resume=True, log=lambda m: None
        )
