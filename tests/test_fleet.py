"""cloud/aws/fleet.py: job blocks, launch requests, the queue and the merges (no AWS)."""

from __future__ import annotations

import base64
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # dataclasses look their module up while it executes
    spec.loader.exec_module(mod)
    return mod


fleet = _load("fleet", ROOT / "cloud" / "aws" / "fleet.py")


def args(*argv: str):
    a = fleet.build_parser().parse_args(["--job", "calibrate", *argv])
    a.dir = a.dir or f"runs/fleet/{a.job}"
    return a


def test_blocks_of_every_job_are_disjoint_and_play_the_fixed_search():
    a = fleet.build_parser().parse_args(["--job", "archetypes", "--archetypes", "nit,maniac"])
    a.archetype_deals, a.block_deals, a.chunk = 1000, 400, 100
    blocks = fleet.archetype_blocks(a, "runs/x")
    assert [b.name for b in blocks] == [f"{n}_0{j}" for n in ("nit", "maniac") for j in range(3)]
    deals = [int(b.commands[0][b.commands[0].index("--deals") + 1]) for b in blocks]
    assert deals == [400, 400, 200] * 2
    # chunk k of a block plays seed + k: no two blocks share a chunk seed
    seeds = []
    for b in blocks:
        cmd = b.commands[0]
        seed = int(cmd[cmd.index("--seed") + 1])
        chunks = -(-int(cmd[cmd.index("--deals") + 1]) // a.chunk)
        seeds += [seed + k for k in range(chunks)]
    assert len(seeds) == len(set(seeds))
    for job in ("archetypes", "lbr", "soak"):
        b = fleet.JOBS[job](fleet.build_parser().parse_args(["--job", job]), "runs/x")
        assert len({x.out for x in b}) == len(b) == len({x.name for x in b})
        assert all(fleet.FIXED in x.commands[0] for x in b)
        assert all(x.type is None for x in b)
    lbr = fleet.lbr_blocks(fleet.build_parser().parse_args(["--job", "lbr"]), "runs/x")
    assert len(lbr) == 6 * 8 and {x.group for x in lbr} == {
        "fcpa/call",
        "fcpa/fc",
        "fcpa/full",
        "spec/call",
        "spec/fc",
        "spec/full",
    }


def test_calibration_has_one_typed_block_per_type():
    a = args("--types", "g5.xlarge", "g6.xlarge", "--local")
    blocks = fleet.calibrate_blocks(a, "runs/c")
    assert [b.type for b in blocks] == ["g5.xlarge", "g6.xlarge", "local"]
    cfgs = [[c[c.index("--search-config") + 1] for c in b.commands] for b in blocks]
    assert cfgs == [[fleet.BUDGETED, fleet.FIXED]] * 3


def test_launch_request_shapes():
    data = fleet.launch_template_data("ami-1", "key", "sg-1", "/dev/sda1", 150, 1.5, "lbr")
    assert base64.b64decode(data["UserData"]).decode().endswith("shutdown -h +90\n")
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert data["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
    cfg = fleet.fleet_configs("t", "3", ["g5.xlarge", "g6.xlarge"], [("s1", "a"), ("s2", "b")], 0.7)
    over = cfg[0]["Overrides"]
    assert len(over) == 4 and {o["MaxPrice"] for o in over} == {"0.7000"}
    assert cfg[0]["LaunchTemplateSpecification"] == {"LaunchTemplateName": "t", "Version": "3"}


def _fleet(tmp_path, blocks, **kw):
    a = args("--budget", "5", *kw.pop("argv", []))
    a.dir = str(tmp_path / "state")
    return fleet.Fleet(a, blocks)


def test_launch_types_count_instances_already_coming(tmp_path):
    blocks = [
        fleet.Block("g5", [], str(tmp_path / "b5"), type="g5.xlarge"),
        fleet.Block("g6", [], str(tmp_path / "b6"), type="g6.xlarge"),
        fleet.Block("any", [], str(tmp_path / "b0")),
    ]
    f = _fleet(tmp_path, blocks)
    assert f.launch_types() == ["g5.xlarge"]
    f.workers["aws-1"] = fleet.Worker(
        "aws-1", "aws", "us-east-1", "i-1", type="g5.xlarge", state="booting"
    )
    assert f.launch_types() == ["g6.xlarge"]  # the booting g5 covers the g5 block
    f.workers["aws-2"] = fleet.Worker(
        "aws-2", "aws", "us-east-1", "i-2", type="g6.xlarge", state="setup"
    )
    assert f.launch_types() == ["g5.xlarge", "g6.xlarge"]  # only the untyped block is left
    f.workers["aws-3"] = fleet.Worker(
        "aws-3", "aws", "us-east-2", "i-3", type="g6.xlarge", state="ready"
    )
    assert f.launch_types() is None
    f.st.blocks = {"g5": "done", "g6": "done", "any": "running"}
    assert f.launch_types() is None


def test_state_survives_a_restart_and_done_blocks_stay_done(tmp_path):
    blocks = [fleet.Block(f"b{i}", [], str(tmp_path / f"b{i}")) for i in range(3)]
    f = _fleet(tmp_path, blocks)
    w = fleet.Worker(
        "aws-1",
        "aws",
        "us-east-1",
        "i-1",
        type="g5.xlarge",
        state="running",
        price=0.5,
        launched=1.0,
    )
    f.workers["aws-1"] = w
    f.assign(w, blocks[1])
    (tmp_path / "b2").mkdir()
    (tmp_path / "b2" / ".done").write_text("x")
    f.save()
    g = _fleet(tmp_path, blocks)
    assert g.st.blocks == {"b0": "todo", "b1": "running", "b2": "done"}
    assert g.workers["aws-1"].block == "b1" and g.st.meta["b1"]["type"] == "g5.xlarge"
    assert g.next_block("g5.xlarge").name == "b0"


def test_run_block_runs_in_order_and_marks_done(tmp_path):
    out = tmp_path / "blk"
    out.mkdir()
    script = tmp_path / "w.py"
    script.write_text("import sys, pathlib\npathlib.Path(sys.argv[1]).write_text(sys.argv[2])\n")
    spec = {
        "out": str(out),
        "commands": [[str(script), str(out / "a"), "1"], [str(script), str(out / "b"), "2"]],
    }
    (out / "block.json").write_text(json.dumps(spec))
    rb = ROOT / "cloud" / "aws" / "run_block.py"
    rc = subprocess.call([sys.executable, str(rb), str(out / "block.json")])
    assert rc == 0 and (out / ".done").exists()
    assert (out / "a").read_text() == "1" and (out / "b").read_text() == "2"
    spec["commands"].insert(1, [str(tmp_path / "missing.py")])
    (out / ".done").unlink()
    (out / "block.json").write_text(json.dumps(spec))
    assert subprocess.call([sys.executable, str(rb), str(out / "block.json")]) != 0
    assert not (out / ".done").exists()


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))


def test_merge_calibrate_costs_per_thousand_hands(tmp_path):
    b = fleet.Block("g5_xlarge", [], "c/g5", type="g5.xlarge")
    streets = {
        "flop": {"iterations_mean": 68.0, "seconds_mean": 4.1},
        "turn": {"iterations_mean": 280.0, "seconds_mean": 2.0},
    }
    common = {"machine": {"gpu": "NVIDIA A10G"}, "streets": streets}
    _write(tmp_path / "c/g5/budget.json", {**common, "seconds_per_hand": 3.0})
    _write(tmp_path / "c/g5/fixed.json", {**common, "seconds_per_hand": 3.6})
    res = fleet.merge_calibrate(tmp_path, [b], {"g5_xlarge": {"price": 0.5}})
    row = res["rows"][0]
    assert row["dollars_per_1000_hands"] == pytest.approx(0.5 * 3.6 * 1000 / 3600)
    assert "NVIDIA A10G" in res["lines"][-1]


def test_merge_archetypes_keeps_chunks_of_every_block(tmp_path):
    rng = np.random.default_rng(0)

    def result(seed):
        chunks = [
            {
                "k": k,
                "seed": seed + k,
                "deals": 4,
                "a": rng.normal(0, 100, 4).tolist(),
                "adjusted": rng.normal(0, 50, 4).tolist(),
                "hud": None,
                "seconds": 1.0,
            }
            for k in range(2)
        ]
        return {
            "agent": "x",
            "reference": "y",
            "config": None,
            "game": {"big_blind": 100},
            "deals": 8,
            "chunk": 4,
            "seed": seed,
            "luck_adjust": True,
            "archetypes": {"nit": {"agent": {"chunks": chunks}, "reference": {"chunks": chunks}}},
        }

    blocks = []
    for j in range(2):
        _write(tmp_path / f"a/nit_{j}/result.json", result(1000 * j))
        blocks.append(fleet.Block(f"nit_{j}", [], f"a/nit_{j}", group="nit", index=j))
    res = fleet.merge_archetypes(tmp_path, blocks, {})
    chunks = res["merged"]["archetypes"]["nit"]["agent"]["chunks"]
    assert sorted(c["k"] for c in chunks) == [0, 1, 1000, 1001]
    assert res["merged"]["deals"] == 16
    assert any("nit" in line for line in res["lines"])


def test_merge_lbr_pools_the_hands_of_a_configuration(tmp_path):
    blocks = []
    for j, n in enumerate((3, 5)):
        d = {
            "opponent": "search",
            "seconds": 10.0,
            "range_resets": 0,
            "by_street": [{"street": "flop", "lbr_actions": {"pot": 1}}],
            "opponent_cost": {"real/flop": [2, 8.0]},
            "samples": {"payoffs": [100.0] * n, "adjusted": [50.0] * n, "end_street": [1] * n},
        }
        _write(tmp_path / f"l/b{j}/lbr.json", d)
        blocks.append(fleet.Block(f"b{j}", [], f"l/b{j}", group="fcpa/call", index=j))
    res = fleet.merge_lbr(tmp_path, blocks, {})
    out = res["configs"]["fcpa/call"]
    assert out["hands"] == 8
    assert out["adjusted"]["mbb"] == pytest.approx(500.0)
    assert out["opponent_cost"]["real/flop"] == [4, 16.0]


def test_merge_soak_pools_chunk_files(tmp_path):
    from pokerbot.agents import AlwaysCallAgent, RandomAgent
    from pokerbot.engine_select import get_engine
    from pokerbot.eval.soak import run_soak

    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    blocks = []
    for j in range(2):
        out = tmp_path / f"s/soak_{j}"
        run_soak(
            RandomAgent(), [AlwaysCallAgent()], config, 20, out, chunk=10, seed=j, engine=engine
        )
        blocks.append(fleet.Block(f"soak_{j}", [], f"s/soak_{j}", group="soak", index=j))
    res = fleet.merge_soak(tmp_path, blocks, {}, dest=tmp_path / "merged")
    assert res["chunks"] == 4
    assert res["summary"]["hands"] == 40
