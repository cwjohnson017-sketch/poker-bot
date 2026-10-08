"""Tree size under a fixed budget (``size_eval``): the search configs, spot picks,
one tiny spot end to end on the CPU (uniform blueprint, ``ShowdownOracle``
leaves, two tree sizes plus fixed-iteration references, a re-search at the
opponent's off-tree bet), re-searches with locks on earlier decisions, and
resuming."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import UniformBlueprint
from pokerbot.search import size_eval as sz
from pokerbot.search import spot_eval as se
from pokerbot.search.combos import NUM_COMBOS
from pokerbot.search.config import search_config
from pokerbot.search.tree import DECISION, TreeConfig, build_tree
from pokerbot.search.tree_map import (
    StrategySnapshot,
    match_nodes,
    offtree_edges,
    path_nodes,
    subtree_nodes,
    translate_sigma,
)
from pokerbot.search.value_leaf import ShowdownOracle

pytest.importorskip("pokerbot.search.batch_solver")

PRE = (("fold",), ("check_call",), ("raise_x", 2.5), ("allin",))
FLOP = (("fold",), ("check_call",), ("raise", 0.5), ("raise", 1.0), ("allin",))
LATE = (("fold",), ("check_call",), ("allin",))
SPEC = ActionSpec(streets=(PRE, FLOP, LATE, LATE), max_raises=1)

# 100 nodes drops the flop's 0.5-pot bet (69 nodes, 6 leaves); 200 keeps it (111, 10)
TINY = sz.SizeSettings(
    sizes=(200, 100),
    budget=0.01,  # below the config's min_iterations: 10 iterations each
    iters=3,
    river_iters=10,
    river_batch=512,
    eval_max_runouts=None,  # the search's 6 sampled run-outs: cheap on the CPU
    device="cpu",
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
def size_result():
    torch.manual_seed(0)
    engine, cfg = _config()
    (spot,) = sz.select_spots(engine, cfg, picks=["0:btn_vs_check"])
    keep: dict = {}
    res = sz.evaluate_sizes(
        spot, UniformBlueprint(SPEC), cfg, TINY, ShowdownOracle, keep=keep, log=lambda m: None
    )
    return res, keep


def test_settings_and_search_configs():
    assert TINY.sizes == (100, 200)  # sorted
    assert sz.search_plan(TINY) == [
        ("100", 100, None),
        ("200", 200, None),
        ("100_it3", 100, 3),
        ("200_it3", 200, 3),
    ]
    assert sz.display_name("100_it3") == "100, 3 it"
    prod = search_config(sz.config_path(sz.DEFAULT_CONFIG))
    over = {**TINY.search, "tree": {"max_nodes": 1, "chance_cards": 2}, "time_budget": 9.0}
    s = sz.SizeSettings(**{**TINY.__dict__, "search": over})
    cfg = sz.size_config(s, 100, "net.pt")
    assert cfg.tree.max_nodes == 100 and cfg.tree.chance_cards == 2  # max_nodes is the run's
    assert cfg.budget(1) == s.budget and cfg.leaf.mode == "value_net" and cfg.leaf.net == "net.pt"
    assert cfg.min_iterations == prod.min_iterations and cfg.solver.iterations == 1000
    assert cfg.tree.leaf_budget_cost is None  # production: a value-net leaf counts 1 node
    assert cfg.gadget.rollouts == 8 and cfg.device == "cpu" and not cfg.fallback_on_error
    fixed = sz.size_config(TINY, 100, iters=7)
    assert fixed.min_iterations == fixed.solver.iterations == 7 and fixed.budget(1) > 1e5
    assert sz.flop_sizes([(), FLOP]) == "0.5/1"
    rich = [(), (("raise", 0.75, "open"), ("raise", 1.0, "open"), ("raise", 1.0, "reraise"))]
    assert sz.flop_sizes(rich) == "open 0.75/1; rr 1"


def test_select_spots():
    engine, cfg = _config()
    spots = sz.select_spots(engine, cfg, picks=["0:bb_first", "0:btn_vs_check", "1:bb_first"])
    assert [s.label for s in spots] == ["board0 BB first", "board0 BTN vs check", "board1 BB first"]
    every = se.exploit_spots(engine, cfg, 2, 5)
    assert list(spots[2].state.board) == list(every[3].state.board)
    assert [s.label for s in sz.select_spots(engine, cfg, 1, 5, ["bb_first"])] == [
        "board0 BB first"
    ]
    with pytest.raises(ValueError, match="bad spot"):
        sz.select_spots(engine, cfg, picks=["0:river"])


def test_sizes_end_to_end(size_result):
    res, keep = size_result
    seat = res["seat"]
    assert res["trunk"] == "200" and res["leaves"] == 10
    names = ["100", "200", "100_it3", "200_it3"]
    assert list(res["searches"]) == names
    assert list(res["profiles"]) == [
        "100",
        "100_research",
        "200",
        "200_research",
        "100_it3",
        "200_it3",
        "blueprint",
    ]
    assert sz.profile_order(TINY) == list(res["profiles"])
    for name, p in res["profiles"].items():
        vals = torch.tensor([p["chips"], *p["br"], p["one_sided_mbb"]])
        assert torch.isfinite(vals).all(), name
        assert p["chips"] >= -1e-6 and p["mbb"] == pytest.approx(10 * p["chips"])
        assert p["one_sided_mbb"] == pytest.approx(10 * p["br"][1 - seat])
        assert p["instances"] + p["skipped"] == 48 * res["leaves"]
    sr = res["searches"]
    assert sr["100"]["leaves"] == 6 and sr["200"]["leaves"] == 10
    assert sr["100"]["iterations"] == sr["200"]["iterations"] == 10  # min_iterations
    assert sr["100_it3"]["iterations"] == sr["200_it3"]["iterations"] == 3
    assert all(x["subset_of_trunk"] and x["ranges_max_diff"] == 0 for x in sr.values())
    assert sz.flop_sizes(sr["100"]["street_actions"]) == "1"
    assert sz.flop_sizes(sr["200"]["street_actions"]) == "0.5/1"
    for name in names:
        rep = sr[name]["translate"]
        assert sum(rep["unmatched_nodes"].values()) == 0
        assert rep["lost_mass"]["searcher"]["nodes"] == 0
        assert rep["lost_mass"]["opponent"]["nodes"] == 0
        assert sr[name]["self_exploitability"]["chips"] >= -1e-6
    # the trunk's own searches map exactly; the small tree's through translations
    assert sr["200"]["translate"]["translated_nodes"] == {"searcher": 0, "opponent": 0}
    assert sr["100"]["translate"]["translated_nodes"]["opponent"] > 0
    assert torch.equal(keep["sigmas"]["200"], keep["snaps"]["200"].sigma)
    # JSON round trip and the report
    back = json.loads(json.dumps(res))
    assert back == res
    data = {"settings": TINY.comparable(), "meta": {}, "spots": [back]}
    md = sz.render_markdown(data)
    assert "| board0 BTN vs check |" in md and "| **mean** |" in md
    assert "200 - 100" in md and "100, 3 it" in md
    assert "re-searched (as played)" in md and "translated only" in md
    assert "| board0 BTN vs check | 100 | 1 | 9 |" in md  # one re-search, 9 nodes below


def test_research_replaces_the_subtree_below_the_offtree_bet(size_result):
    res, keep = size_result
    seat = res["seat"]  # the button; the big blind's 0.5-pot bet is off the 100 tree
    sr = res["searches"]
    r100 = sr["100"]["research"]
    assert r100["count"] == 1 and r100["all_paths_exact"] and r100["seeded_locks"] == 0
    (edge,) = r100["edges"]
    assert edge["history"] == [[1, 1, 2, 250]]  # the big blind bets 250 into 500
    assert sr["200"]["research"]["count"] == 0  # the trunk itself: nothing is off-tree
    assert res["profiles"]["200_research"]["same_as"] == "200"
    assert "100_it3_research" not in res["profiles"]  # fixed-iteration runs: no re-search
    (r,) = keep["researches"]["100"]
    solver = keep["eval_solver"]
    trunk = solver.tree
    assert r.edge == edge["node"] and trunk.histories[r.edge] == ((1, 1, 2, 250),)
    assert r.snapshot.tree.histories[r.snapshot.tree.current_node] == trunk.histories[r.edge]
    # below the edge: the re-search's strategy; everywhere else: the first search's
    sub = set(subtree_nodes(trunk, r.edge))
    want, _ = translate_sigma(r.snapshot, solver, seat, strict=True)
    comp, base = keep["sigmas"]["100_research"], keep["sigmas"]["100"]
    below = 0
    for d, n in enumerate(solver.dec_nodes.tolist()):
        if n in sub:
            assert torch.equal(comp[d], want[d].cpu()), n
            below += 1
        else:
            assert torch.equal(comp[d], base[d]), n
    assert below == r100["subtree_decision_nodes"] == 9
    ids = [d for d, n in enumerate(solver.dec_nodes.tolist()) if n in sub]
    assert not torch.equal(comp[ids], base[ids])  # the re-search did change something
    # the first search's tree lacks the bet (translated); it is the only off-tree edge
    m = keep["matches"]["100"]
    assert m.translated[r.edge] and offtree_edges(m, trunk, 1 - seat) == [r.edge]


SPEC3 = ActionSpec(streets=(PRE, FLOP, LATE, LATE), max_raises=3)
# BB first at 300 nodes keeps only the pot bet (165 nodes); the full tree has 651
SMALL = sz.SizeSettings(
    sizes=(300,), budget=0.01, device="cpu", search=dict(TINY.search), spot_warmup=False
)


def test_research_locks_the_searchers_earlier_decisions():
    engine, cfg = _config()
    (spot,) = sz.select_spots(engine, cfg, picks=["0:bb_first"])
    s = spot.state
    seat = int(s.current_player)  # the big blind, first to act on the flop
    bp = UniformBlueprint(SPEC3)
    conf = sz.size_config(SMALL, 300)
    first = se.run_search("300", bp, s, cfg, conf, ShowdownOracle())
    snap = StrategySnapshot.of(first.solver)
    tc = TreeConfig(spec=SPEC3, max_nodes=10**6, chance_cards=2, leaf_mode="value_net")
    trunk = build_tree(cfg, s.button, s.board, s.history, tc)
    # identical trees: nothing to re-search
    m_self = match_nodes(snap.tree, snap.tree)
    assert offtree_edges(m_self, snap.tree, 1 - seat) == []
    same = build_tree(cfg, s.button, s.board, s.history, replace(tc, max_nodes=300))
    assert sz.research_offtree("300", first.agent, snap, same, s, cfg) == []
    # the button's 0.5-pot sizes: a bet after the check, a raise over the BB's pot bet,
    # and a re-raise over the BB's check-raise (two BB decisions on that path)
    rs = sz.research_offtree("300", first.agent, snap, trunk, s, cfg, log=lambda m: None)
    assert [r.history for r in rs] == [
        ((1, 1, 1, 0), (1, 0, 2, 250)),
        ((1, 1, 2, 500), (1, 0, 2, 1250)),
        ((1, 1, 1, 0), (1, 0, 2, 500), (1, 1, 2, 2000), (1, 0, 2, 4250)),
    ]
    assert [r.seeded for r in rs] == [0, 0, 1]
    first_index = {snap.tree.histories[n]: (d, n) for d, n in enumerate(snap.dec_nodes.tolist())}
    for r in rs:
        t = r.snapshot.tree
        cur = t.current_node
        assert t.histories[cur] == r.history and int(t.actor[cur]) == seat
        idx = {n: d for d, n in enumerate(r.snapshot.dec_nodes.tolist())}
        locked = 0
        for q in path_nodes(t, cur):
            if int(t.kind[q]) != DECISION or int(t.actor[q]) != seat:
                continue
            # locked to the first search's average strategy there, renormalised over
            # the re-search tree's actions
            d0, n0 = first_index[t.histories[q]]
            acts0 = snap.tree.child_actions(n0)
            acts = t.child_actions(q)
            zero = torch.zeros(NUM_COMBOS)
            want = torch.stack(
                [snap.sigma[d0, acts0.index(a)] if a in acts0 else zero for a in acts]
            )
            want = want / want.sum(0, keepdim=True)
            got = r.snapshot.sigma[idx[q], : len(acts)]
            assert torch.allclose(got, want, atol=1e-5), (r.history, t.histories[q])
            locked += 1
        assert locked == len([h for h in r.history if h[1] == seat])


def test_replay_from_the_street_root():
    engine, cfg = _config()
    (spot,) = sz.select_spots(engine, cfg, picks=["0:btn_vs_check"])
    s = spot.state
    root = sz.street_root_state(engine, cfg, s)
    assert int(root.street) == 1 and not [h for h in root.history if int(h[0]) == 1]
    assert [list(root.hole_cards(p)) for p in (0, 1)] == [list(s.hole_cards(p)) for p in (0, 1)]
    again = sz.replay(engine, cfg, s, [(1, 1, 1, 0)])  # the big blind checks
    assert again.public_key() == s.public_key() and int(again.current_player) == 0
    assert sz.played_key(again) == sz.played_key(s)
    with pytest.raises(ValueError, match="does not fit"):
        sz.replay(engine, cfg, s, [(1, 0, 1, 0)])  # the button is not first to act


def test_run_size_evaluation_resumes(tmp_path, monkeypatch, size_result):
    res, _ = size_result
    calls = []

    def fake(spot, *a, **k):
        calls.append(spot.label)
        return {**res, "label": spot.label}

    monkeypatch.setattr(sz, "evaluate_sizes", fake)
    spots = [se.Spot(f"s{i}", 0, "btn_vs_check", None) for i in range(3)]
    meta = {"blueprint": "uniform", "leaf_model": "oracle:showdown"}
    out, md = tmp_path / "e.json", tmp_path / "e.md"
    sz.run_size_evaluation(spots[:2], None, None, TINY, dict(meta), out, md, log=lambda m: None)
    assert calls == ["s0", "s1"]
    data = sz.run_size_evaluation(
        spots, None, None, TINY, dict(meta), out, md, resume=True, log=lambda m: None
    )
    assert calls == ["s0", "s1", "s2"]
    assert [s["label"] for s in data["spots"]] == ["s0", "s1", "s2"]
    assert json.loads(out.read_text())["spots"][2]["label"] == "s2"
    assert "| s2 |" in md.read_text(encoding="utf-8")
    other = sz.SizeSettings(**{**TINY.__dict__, "budget": 1.0})
    with pytest.raises(ValueError, match="other settings"):
        sz.run_size_evaluation(
            spots, None, None, other, dict(meta), out, md, resume=True, log=lambda m: None
        )
