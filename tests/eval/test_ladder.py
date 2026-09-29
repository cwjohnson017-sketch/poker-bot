import json

import pytest

from pokerbot.eval.ladder import Ladder, compute_ratings, main, pair_key, schedule_pairs

pytestmark = pytest.mark.usefixtures("reference_engine")

BASELINES = ["always_call", "always_raise", "equity:samples=20"]


def test_schedule_pairs():
    s = ["c1", "c2", "c3", "c4"]
    assert len(schedule_pairs(s, "all")) == 6
    assert schedule_pairs(s, "newest", k=2, anchors=["bp"]) == [
        ("c4", "c2"),
        ("c4", "c3"),
        ("c4", "bp"),
    ]
    prev = schedule_pairs(s, "previous", k=1)
    assert prev == [("c2", "c1"), ("c3", "c2"), ("c4", "c3")]
    assert len({pair_key(a, b) for a, b in schedule_pairs(s + ["c1"], "all", anchors=s)}) == 6


def test_ratings_reproduce_transitive_results():
    # true strengths +100, 0, -100 mbb/h
    true = {"a": 100.0, "b": 0.0, "c": -100.0}
    ms = []
    for x, y in (("a", "b"), ("a", "c"), ("b", "c")):
        d = true[x] - true[y]
        wins = 60 if d > 0 else 40
        ms.append(
            dict(
                a=x,
                b=y,
                mbb=d,
                ci_low=d - 20,
                ci_high=d + 20,
                hands=1000,
                wins=wins,
                losses=100 - wins,
                ties=0,
            )
        )
    r = {row["agent"]: row for row in compute_ratings(["a", "b", "c"], ms)}
    for k, v in true.items():
        assert r[k]["rating"] == pytest.approx(v, abs=1e-6)
    assert r["a"]["elo"] > r["b"]["elo"] > r["c"]["elo"]
    assert sum(row["elo"] for row in r.values()) / 3 == pytest.approx(1500.0)


def test_ladder_round_robin_is_incremental(tmp_path):
    out = tmp_path / "ladder.json"
    lad = Ladder(out, game={"stacks": [20000, 20000]})
    pairs = schedule_pairs(BASELINES, "all")
    new = lad.run(pairs, hands=60, workers=2, progress=None)
    assert len(new) == 3
    data = json.loads(out.read_text())
    assert len(data["matches"]) == 3
    assert {r["agent"] for r in data["ratings"]} == set(BASELINES)
    md = out.with_suffix(".md").read_text()
    assert "| # | agent | rating (mbb/h) | Elo |" in md
    for b in BASELINES:
        assert f"`{b}`" in md
    # results are antisymmetric when read from the other side
    m1, m2 = lad.get("always_call", "always_raise"), lad.get("always_raise", "always_call")
    assert m1["mbb"] == -m2["mbb"] and m1["ci_low"] == -m2["ci_high"]

    # a second run skips everything already played
    lad2 = Ladder(out, game={"stacks": [20000, 20000]})
    assert lad2.run(pairs, hands=60, progress=None) == []
    # adding an agent plays only the new pairs; more hands replays old ones
    specs = BASELINES + ["random"]
    new = lad2.run(schedule_pairs(specs, "all"), hands=60, progress=None)
    assert sorted(r["b"] if r["a"] == "random" else r["a"] for r in new) == sorted(BASELINES)
    assert len(json.loads(out.read_text())["matches"]) == 6
    assert len(lad2.run([("random", "always_call")], hands=120, progress=None)) == 1
    # a different game refuses to mix results
    with pytest.raises(ValueError, match="built for game"):
        Ladder(out, game={"stacks": [1000, 1000]})


def test_ladder_cli_with_chain_and_regressions(tmp_path, capsys):
    out = tmp_path / "chain.json"
    # a "checkpoint" sequence where the last one is much worse than its predecessor
    argv = [
        "--agents",
        "always_call",
        "equity:samples=20",
        "random",
        "--schedule",
        "previous",
        "-k",
        "1",
        "--hands",
        "200",
        "--out",
        str(out),
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    assert main(argv) == 0
    data = json.loads(out.read_text())
    assert len(data["matches"]) == 2
    assert data["chain"] == ["always_call", "equity:samples=20", "random"]
    lad = Ladder(out)
    regs = lad.regressions()
    assert [(m["a"], m["b"]) for m in regs] == [("random", "equity:samples=20")]
    assert "## Regressions" in out.with_suffix(".md").read_text()
    assert (tmp_path / "logs" / "scalars.csv").exists()
