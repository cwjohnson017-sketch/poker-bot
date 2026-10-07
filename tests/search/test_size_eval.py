"""Tree size under a fixed budget (``size_eval``): the search configs, spot picks,
one tiny spot end to end on the CPU (uniform blueprint, ``ShowdownOracle``
leaves, two tree sizes plus fixed-iteration references), and resuming."""

from __future__ import annotations

import json

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import ActionSpec
from pokerbot.search import UniformBlueprint
from pokerbot.search import size_eval as sz
from pokerbot.search import spot_eval as se
from pokerbot.search.config import search_config
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
    assert list(res["profiles"]) == [*names, "blueprint"]
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
