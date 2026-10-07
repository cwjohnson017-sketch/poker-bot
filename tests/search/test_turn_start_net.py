"""Turn-start value net (``docs/turn_start_net.md``): data from batched turn
solves with turn-end leaves, the shard format, learning check-down turn-start
values, checkpoint kinds, the flop-end leaf evaluator and the agent options."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import CHECK_CALL, DEFAULT_SPEC, FOLD, ActionSpec
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import legal_options
from pokerbot.search.batch_turn_solver import BatchTurnSolver, turn_tree
from pokerbot.search.blueprint import TabularBlueprintFromCallable
from pokerbot.search.combos import NUM_COMBOS
from pokerbot.search.tree import DECISION, VALUE
from pokerbot.search.turn_data import RiverAveragePredictor, turn_checkdown_samples, turn_targets
from pokerbot.search.turn_net import (
    TurnEndPredictor,
    TurnStartPredictor,
    checkpoint_kind,
    load_leaf_predictor,
    save_turn_net,
    train_turn_net,
    turn_meta,
)
from pokerbot.search.turn_start_data import (
    TurnStartGenConfig,
    generate_turn_start_data,
    solve_turn_batch,
    solve_turn_states,
)
from pokerbot.search.value_data import SHARD_DTYPES
from pokerbot.search.value_leaf import ShowdownOracle
from pokerbot.search.value_net import RiverValueNet, ValueNetConfig, opponent_mass
from pokerbot.search.value_train import ValueTrainConfig, load_shards

C = NUM_COMBOS
BIG = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
SMALL = ActionSpec(streets=(BIG,) * 4, max_raises=2)
PASSIVE = ActionSpec(streets=((("fold",), ("check_call",)),) * 4, max_raises=0)
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _engine_config(stacks: int = 10000):
    return get_engine().GameConfig(
        num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100
    )


def _passive_blueprint():
    def fn(state, player):
        a, b = state.hole_cards(player)
        x = ((int(a) * 7 + int(b) * 3) % 11) / 10
        w = {}
        for o in legal_options(state, DEFAULT_SPEC):
            w[o.index] = {FOLD: 0.2, CHECK_CALL: 3.0}.get(o.kind, 0.2 * (0.5 + x))
        return w

    bp = TabularBlueprintFromCallable(fn, DEFAULT_SPEC)
    bp.game = {"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100}
    return bp


def _noisy_net(cfg: ValueNetConfig, seed: int = 0) -> RiverValueNet:
    torch.manual_seed(seed)
    net = RiverValueNet(cfg)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return net.eval()


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------- data


def test_generate_turn_start_data_shards_and_clis(tmp_path):
    """Self-play (turn root), perturbed and random states solved with check-down
    turn-end leaves: the shard format, the stored targets equal a direct solve,
    resume, and the data and training CLIs."""
    bp = _passive_blueprint()
    pred = RiverAveragePredictor(ShowdownOracle())
    cfg = TurnStartGenConfig(samples=12, shard_size=8, seed=2, batch=5, iterations=6)
    meta = generate_turn_start_data(
        pred, bp, tmp_path / "sp", cfg, device="cpu", spec=SMALL, log=None, turn_net="checkdown"
    )
    assert meta["kind"] == "turn_start" and len(meta["shards"]) == 2
    meta_file = json.loads((tmp_path / "sp" / "meta.json").read_text())
    assert meta_file["kind"] == "turn_start" and meta_file["spec"]["max_raises"] == 2
    for name in meta["shards"]:
        shard = torch.load(tmp_path / "sp" / name, weights_only=True)
        assert {k: v.dtype for k, v in shard.items()} == SHARD_DTYPES
    d = load_shards(tmp_path / "sp")
    assert d["boards"].shape == (12, 4) and d["ranges"].shape == (12, 2, C)
    assert d["targets"].shape == (12, 2, C)
    assert sorted(set(d["source"].tolist())) == [0, 1, 2]
    assert bool((d["c"] + d["stack"] == 10000).all())
    valid = vrg.board_valid(d["boards"].long())
    r = d["ranges"].float()
    torch.testing.assert_close(r.sum(-1), torch.ones(12, 2), atol=2e-3, rtol=0)
    assert bool((r[~valid[:, None].expand_as(r)] == 0).all())
    assert bool((d["targets"][~valid[:, None].expand_as(r)] == 0).all())
    ex = d["exploit"]
    assert bool(torch.isfinite(ex).all()) and float(ex.max()) > 0 and float(ex.min()) > -1e-3
    # a stored row equals solving its state alone at its c (instances are independent)
    engine_cfg = _engine_config()
    for i in (0, 7):
        c = int(d["c"][i])
        rows = solve_turn_batch(
            {"boards": d["boards"][i : i + 1].long(), "ranges": r[i : i + 1]},
            c,
            SMALL,
            engine_cfg,
            6,
            pred,
            "cpu",
        )
        # (re-quantising the stored fp16 ranges moves them by about 1e-3 relative)
        got, want = rows["targets"].float(), d["targets"][i : i + 1].float()
        torch.testing.assert_close(got, want, rtol=5e-3, atol=1e-3)
        torch.testing.assert_close(rows["exploit"], d["exploit"][i : i + 1], rtol=0.05, atol=1e-4)
    # resume: settings are checked, existing shards are kept
    with pytest.raises(FileExistsError):
        generate_turn_start_data(pred, bp, tmp_path / "sp", cfg, device="cpu", spec=SMALL)
    bad = TurnStartGenConfig(samples=12, shard_size=8, seed=2, batch=5, iterations=7)
    with pytest.raises(ValueError, match="iterations"):
        generate_turn_start_data(
            pred, bp, tmp_path / "sp", bad, device="cpu", spec=SMALL, resume=True
        )
    # the CLI: random states only (no blueprint: DEFAULT_SPEC turn trees), resume
    gen = _load_script("gen_turn_start_data")
    out = tmp_path / "rand"
    argv = ["--turn-net", "checkdown", "--out", str(out), "--samples", "6", "--mix", "0,0,1"]
    argv += ["--shard-size", "3", "--iterations", "3", "--batch", "3", "--device", "cpu"]
    assert gen.main(argv) == 0
    first = torch.load(out / "shard_00001.pt", weights_only=True)
    (out / "shard_00001.pt").unlink()
    assert gen.main([*argv, "--resume"]) == 0
    again = torch.load(out / "shard_00001.pt", weights_only=True)
    for k, v in first.items():
        assert torch.equal(v, again[k]), k
    with pytest.raises(SystemExit):
        gen.main(["--turn-net", "checkdown", "--out", str(tmp_path / "x"), "--samples", "4"])
    # train a turn-start net on it with the CLI; a turn-end run rejects the data
    train = _load_script("train_turn_net")
    ck = tmp_path / "vn" / "ts.pt"
    argv = ["--data", str(out), "--heldout-data", str(tmp_path / "sp"), "--out", str(ck)]
    argv += ["--device", "cpu", "--steps", "5", "--batch", "4", "--buckets", "16"]
    argv += ["--width", "32", "--layers", "2", "--spread-buckets", "4", "--quiet"]
    with pytest.raises(ValueError, match="turn_start shards"):
        train.main(argv)  # --kind defaults to turn_end
    assert train.main([*argv, "--kind", "turn_start"]) == 0
    assert checkpoint_kind(ck) == "turn_start"
    rep = json.loads(ck.with_suffix(".json").read_text())
    assert rep["train_samples"] == 6 and rep["heldout_samples"] == 12
    p = load_leaf_predictor(ck)
    assert isinstance(p, TurnStartPredictor) and p.kind == "turn_start"


def test_checkdown_turn_start_targets_and_learning():
    """With a check-down turn and river, the turn-start targets of the solve
    pipeline are the exact check-down values (turn_targets of ShowdownOracle);
    a small net learns them."""
    torch.manual_seed(0)
    data = turn_checkdown_samples(480, seed=3, boards=160)
    tree = turn_tree(_engine_config(), 300, PASSIVE)
    kinds = tree.kind.tolist()
    assert kinds == [DECISION, DECISION, VALUE]  # check, check, turn end
    sub = {k: v[:40] for k, v in data.items()}
    rows = solve_turn_states(sub, PASSIVE, _engine_config(), 1, 16, ShowdownOracle(), "cpu")
    want = turn_targets(
        ShowdownOracle(), rows["boards"].long(), rows["ranges"].float(), rows["c"], rows["stack"]
    )
    torch.testing.assert_close(rows["targets"].float(), want, rtol=2e-3, atol=2e-4)
    torch.testing.assert_close(
        rows["targets"].float(), sub["targets"].float(), rtol=1e-2, atol=2e-3
    )
    assert float(rows["exploit"].abs().max()) < 1e-6  # nothing to decide
    net_cfg = ValueNetConfig(buckets=64, width=192, layers=3)
    cfg = ValueTrainConfig(steps=250, batch=64, warmup=20, holdout=0.15, seed=1)
    net, rep = train_turn_net(
        None,
        None,
        net_cfg,
        cfg,
        spread_buckets=4,
        device="cpu",
        log=None,
        raw=data,
        kind="turn_start",
    )
    o = rep["overall"]
    assert rep["meta"]["kind"] == "turn_start" and net.meta["kind"] == "turn_start"
    assert o["samples"] > 30 and o["mae"] < 0.35 * o["zero"]["mae"], o
    assert o["wmae"] < 0.35 * o["zero"]["wmae"], o


def test_turn_start_checkpoint_kind_round_trip(tmp_path):
    net = _noisy_net(ValueNetConfig(buckets=32, width=48, layers=2))
    ts, te = tmp_path / "ts.pt", tmp_path / "te.pt"
    save_turn_net(ts, net, spread_buckets=4, meta={"note": "x"}, kind="turn_start")
    save_turn_net(te, net, spread_buckets=4)
    assert checkpoint_kind(ts) == "turn_start" and checkpoint_kind(te) == "turn_end"
    p = load_leaf_predictor(ts)
    assert type(p) is TurnStartPredictor and p.kind == "turn_start"
    assert p.spread_buckets == 4 and p.net.meta["note"] == "x"
    assert type(load_leaf_predictor(te)) is TurnEndPredictor
    assert type(TurnStartPredictor.from_path(ts)) is TurnStartPredictor
    with pytest.raises(ValueError, match="not a turn-end"):
        TurnEndPredictor.from_path(ts)
    with pytest.raises(ValueError, match="not a turn-start"):
        TurnStartPredictor.from_path(te)
    with pytest.raises(ValueError, match="kind"):
        turn_meta(4, kind="flop")
    # the same net and features as the turn-end predictor
    b4 = torch.tensor([[2, 7, 19, 33], [0, 1, 2, 3]])
    valid = vrg.board_valid(b4)
    g = torch.Generator().manual_seed(8)
    r = torch.rand(2, 2, C, generator=g) ** 3 * valid[:, None]
    c, stack = torch.tensor([300, 2000]), torch.tensor([9700, 8000])
    ev = p.predict(b4, r, c, stack)
    assert torch.equal(ev, load_leaf_predictor(te).predict(b4, r, c, stack))
    rn = r / r.sum(-1, keepdim=True)
    assert float((rn * opponent_mass(rn, valid) * ev).sum((1, 2)).abs().max()) < 1e-5


def test_batch_turn_solver_with_a_turn_end_net():
    """A (random) turn-end net as the leaf predictor: finite values, and the
    leaf rows go through the predictor's board cache."""
    net = _noisy_net(ValueNetConfig(buckets=32, width=48, layers=2))
    pred = TurnEndPredictor(net, "cpu", spread_buckets=4)
    boards = [[2, 7, 19, 33], [0, 1, 2, 3], [51, 40, 30, 20]]
    g = torch.Generator().manual_seed(1)
    ranges = torch.rand(3, 2, C, generator=g) * vrg.board_valid(torch.tensor(boards))[:, None]
    tree = turn_tree(_engine_config(), 400, SMALL)
    s = BatchTurnSolver(tree, boards, ranges, pred, leaf_chunk=6)  # 2 leaves x 3 boards a call
    s.solve(5)
    L = s.num_leaves
    assert L > 2 and s.net_rows == 10 * L * 3 and s.net_calls == 10 * math.ceil(L / 2)
    ex = s.exploitability()
    assert bool(torch.isfinite(ex["br"]).all()) and len(pred.cache) == 3
