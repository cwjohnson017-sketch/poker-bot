"""Turn-start value net (``docs/turn_start_net.md``): data from batched turn
solves with turn-end leaves, the shard format, learning check-down turn-start
values, checkpoint kinds, the flop-end leaf evaluator and the agent options."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from pokerbot.agents import RandomAgent
from pokerbot.engine_select import get_engine
from pokerbot.env.actions import CHECK_CALL, DEFAULT_SPEC, FOLD, ActionSpec
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import legal_options, make_state
from pokerbot.search.batch_turn_solver import BatchTurnSolver, turn_tree
from pokerbot.search.blueprint import TabularBlueprintFromCallable
from pokerbot.search.combos import NUM_COMBOS, blocked_sum, valid_mask
from pokerbot.search.solver import RangeSolver, SolverConfig, allin_matrix
from pokerbot.search.tree import DECISION, VALUE, TreeConfig, build_tree
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
from pokerbot.search.value_leaf import (
    FlopEndLeafEvaluator,
    ShowdownOracle,
    TurnEndLeafEvaluator,
    make_leaf_evaluator,
)
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


# --------------------------------------------------------------------------- flop-end leaves


class _TurnStartOracle(RiverAveragePredictor):
    """Exact turn-start values when the turn and the river are checked down:
    the check-down turn-end values of the same ranges (kind turn_start)."""

    kind = "turn_start"


PASS = (("fold",), ("check_call",))
FLOP_BOARD = [4, 9, 14, 19, 24]  # the flop is the first three cards


def _flop_state(stacks: int = 2000):
    engine = get_engine()
    cfg = _engine_config(stacks)
    s = make_state(engine, cfg, 0, FLOP_BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    return cfg, s


def _flop_tree(cfg, s, spec=SMALL, depth=0):
    tc = TreeConfig(spec=spec, depth_streets=depth, max_nodes=10**7, leaf_mode="value_net")
    return build_tree(cfg, s.button, s.board, s.history, tc)


def test_flop_end_leaves_are_the_exact_checkdown_runout():
    """FlopEndLeafEvaluator with the check-down turn-start oracle equals the
    dense all-in enumeration over all 1176 turn and river run-outs, card
    removal included; the per-card opponent-mass identity holds on 3-card
    boards; make_leaf_evaluator picks it for a turn-start predictor."""
    cfg, s = _flop_state()
    tree = _flop_tree(cfg, s)
    oracle = _TurnStartOracle(ShowdownOracle(max_boards=1 << 13))
    ev = FlopEndLeafEvaluator(tree, oracle, chunk=49 * 2)  # several chunks
    assert type(make_leaf_evaluator(tree, oracle)) is FlopEndLeafEvaluator
    L = ev.num_leaves
    assert L == tree.count(VALUE) > 1 and ev.num_rows == 49 * L and ev.cards_per_pair == 45
    board = list(s.board)
    valid = valid_mask(board)
    g = torch.Generator().manual_seed(5)
    reach = torch.rand(2, L, C, generator=g, dtype=torch.float64) ** 2  # also on the board
    reach[1, 0] = 0.0  # an unreached leaf
    E = allin_matrix(tuple(board), 1176, torch.device("cpu"), torch.float64, 0)  # every run-out
    both = ev.values_both(reach)
    for p in (0, 1):
        v = ev.values(p, reach)
        assert v.shape == (L, C) and bool((v[:, ~valid] == 0).all())
        torch.testing.assert_close(both[p], v, rtol=1e-12, atol=1e-12)
        for j in range(L):
            c = int(tree.contrib[ev.ids[j], 0])
            ref = c * ((reach[1 - p, j] * valid) @ E.t()) * valid
            assert torch.allclose(v[j], ref, atol=1e-9 * max(1.0, float(ref.abs().max()))), (p, j)
    opp = reach[0] * ev.valid
    mx = ev.opponent_mass(opp, ev.cards)
    for slot in (0, 30, 48):
        x = ev.cards[:, slot]
        avoid = torch.stack([valid_mask([int(t)]) for t in x])
        assert torch.allclose(mx[:, slot] * avoid, blocked_sum(opp * avoid) * avoid, atol=1e-10)
    # a turn-start net cannot value turn-end leaves, nor a turn-end net flop-end ones
    with pytest.raises(ValueError, match="3-card"):
        FlopEndLeafEvaluator(_flop_tree(cfg, s, depth=1), oracle)
    with pytest.raises(ValueError, match="turn-start net"):
        TurnEndLeafEvaluator(tree, RiverAveragePredictor(ShowdownOracle()))


class _ToyTurn:
    """A nonlinear turn 'net' that depends on both ranges, the 4-card board,
    ``c`` and ``stack`` (0 on combos that hit the board), to check the plumbing."""

    kind = "turn_end"

    def predict(self, boards, ranges, c, stack):
        valid = vrg.board_valid(boards.to(ranges.device))[:, None, :]
        r = ranges / ranges.sum(-1, keepdim=True).clamp(min=1e-30)
        z = 300.0 * r - 200.0 * r.flip(1) + (boards.sum(1).to(r.dtype) / 200.0)[:, None, None]
        z = z + (c.to(r.dtype) / 1000.0 - stack.to(r.dtype) / 20000.0)[:, None, None]
        z = z + torch.tensor([0.1, -0.2], dtype=r.dtype, device=r.device)[None, :, None]
        return torch.tanh(z) * valid


class _ToyTurnStart(_ToyTurn):
    kind = "turn_start"


def test_flop_solve_with_flop_end_leaves_matches_the_two_street_solve():
    """A depth_streets 0 flop solve whose flop-end leaves use a turn-start
    predictor ``P`` equals a flop + turn solve with a checked-down turn (all 49
    turn cards dealt by chance nodes) whose turn-end leaves use the same ``P``
    as a turn-end predictor: values, best responses, exploitability and every
    flop strategy. ``P`` is nonlinear in both ranges (the check-down oracle is
    checked against the dense enumeration above)."""
    cfg, s = _flop_state()
    tree_a = _flop_tree(cfg, s)
    tree_b = _flop_tree(cfg, s, ActionSpec(streets=(BIG, BIG, PASS, PASS), max_raises=2), 1)
    turn_dec = [n for n in range(tree_b.num_nodes) if int(tree_b.street[n]) == 2]
    assert turn_dec and all(int(tree_b.num_children[n]) == 1 for n in turn_dec)
    # the tree stores chance weights in float32 (1/45 to 6e-8): use the exact float64 one
    dealt = tree_b.deal_card >= 0
    assert int(dealt.sum()) == 49 * tree_a.count(VALUE)
    tree_b.chance_weight = torch.where(dealt, 1.0 / 45, tree_b.chance_weight.double())
    g = torch.Generator().manual_seed(1)
    r = torch.rand(2, C, generator=g, dtype=torch.float64) * valid_mask(s.board)
    sc = SolverConfig(dtype="float64")
    sa = RangeSolver(tree_a, r, sc, value_leaves=FlopEndLeafEvaluator(tree_a, _ToyTurnStart()))
    sb = RangeSolver(tree_b, r, sc, value_leaves=TurnEndLeafEvaluator(tree_b, _ToyTurn()))
    sa.solve(10)
    sb.solve(10)
    pot = int(s.pot)
    ea, eb = sa.exploitability(), sb.exploitability()
    for k in ("br", "ev"):
        for p in (0, 1):
            assert abs(ea[k][p] - eb[k][p]) < 1e-10 * pot, (k, ea, eb)
    for p in (0, 1):
        for br in (False, True):
            va, _ = sa.values(p, best_response=br)
            vb, _ = sb.values(p, best_response=br)
            assert float((va[0] - vb[0]).abs().max()) < 1e-10 * float(vb[0].abs().max()), (p, br)
    flop_b = {
        tree_b.histories[n]: n
        for n in range(tree_b.num_nodes)
        if int(tree_b.street[n]) == 1 and int(tree_b.kind[n]) == DECISION
    }
    dec_a = [n for n in range(tree_a.num_nodes) if int(tree_a.kind[n]) == DECISION]
    assert len(dec_a) == len(flop_b)
    for n in dec_a:
        diff = sa.node_strategy(n) - sb.node_strategy(flop_b[tree_a.histories[n]])
        assert float(diff.abs().max()) < 1e-9


# --------------------------------------------------------------------------- agent

SMALL_ACTIONS = [[list(a) for a in BIG]] * 4
TINY_TS = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"actions": SMALL_ACTIONS, "max_raises": 1, "depth_streets": 0, "max_nodes": 1500},
    "solver": {"iterations": 3, "max_runouts": 6},
    "leaf": {"mode": "value_net", "net_every": 1},
    "gadget": {"rollouts": 8},
}


def _play(agent, hands=6, seed=0):  # seed 0: decisions on every street
    config = get_engine().GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    res = run_match([agent, RandomAgent()], config, num_hands=hands, seed=seed)
    assert res.hands == hands and int(res.seat_payoffs.sum()) == 0
    assert not any(st.get("fallback") for st in agent.stats)
    return {k: [st for st in agent.stats if st.get("street") == k] for k in (1, 2, 3)}


def _with(base, **sections):
    out = dict(base)
    for k, v in sections.items():
        out[k] = {**base.get(k, {}), **v}
    return out


def test_agent_with_flop_end_and_turn_end_leaves_plays_legal_hands(tmp_path):
    oracle = _TurnStartOracle(ShowdownOracle(max_boards=4096))
    turn_end = RiverAveragePredictor(ShowdownOracle(max_boards=4096))
    # 1. flop depth 0 (turn-start oracle), the turn solved to showdown
    cfg1 = _with(TINY_TS, tree={"depth_streets_turn": 1})
    a = SearchAgent(UniformBlueprint(), cfg1, value_predictor=oracle)
    by = _play(a)
    assert by[1] and all(st["value_provider"] == "FlopEndLeafEvaluator" for st in by[1])
    assert all(st["value_net_rows"] % 49 == 0 and st["value_leaves"] > 0 for st in by[1])
    assert by[2] and all(st["value_leaves"] == 0 for st in by[2])
    # 2. the turn depth-limited too: turn-end leaves from the turn predictor
    cfg2 = _with(TINY_TS, tree={"depth_streets_turn": 0})
    b = SearchAgent(UniformBlueprint(), cfg2, value_predictor=oracle, turn_value_predictor=turn_end)
    by = _play(b)
    assert by[1] and all(st["value_provider"] == "FlopEndLeafEvaluator" for st in by[1])
    assert by[2] and all(st["value_provider"] == "TurnEndLeafEvaluator" for st in by[2])
    assert all(st["value_leaves"] > 0 for st in by[2])
    # 3. both nets from checkpoints (leaf.net turn-start, leaf.turn_net turn-end)
    net = _noisy_net(ValueNetConfig(buckets=16, width=32, layers=2))
    ts, te = tmp_path / "ts.pt", tmp_path / "te.pt"
    save_turn_net(ts, net, spread_buckets=2, kind="turn_start")
    save_turn_net(te, net, spread_buckets=2)
    cfg3 = _with(cfg2, leaf={"net": str(ts), "turn_net": str(te)})
    c = SearchAgent(UniformBlueprint(), cfg3)
    assert type(c.get_value_predictor()) is TurnStartPredictor
    assert type(c.get_turn_value_predictor()) is TurnEndPredictor
    by = _play(c, hands=4, seed=4)
    assert by[1] and all(st["value_provider"] == "FlopEndLeafEvaluator" for st in by[1])
    assert all(st["value_provider"] == "TurnEndLeafEvaluator" for st in by[2])


def test_depth_streets_turn_sets_the_depth_of_turn_trees():
    engine = get_engine()
    cfg = _engine_config(2000)
    s = make_state(engine, cfg, 0, FLOP_BOARD[:4], [])
    for _ in range(4):
        s.apply(engine.Action.check_call())
    assert int(s.street) == 2

    def tree(**kw):
        tc = TreeConfig(spec=SMALL, max_nodes=10**6, leaf_mode="value_net", **kw)
        return build_tree(cfg, s.button, s.board, s.history, tc)

    assert tree(depth_streets=1).count(VALUE) == 0  # today: the turn runs to showdown
    limited = tree(depth_streets=1, depth_streets_turn=0)
    assert limited.count(VALUE) > 0 and limited.last_street == 2
    assert torch.equal(limited.kind, tree(depth_streets=0).kind)  # None: depth_streets
    assert tree(depth_streets=0, depth_streets_turn=1).count(VALUE) == 0
    flop_cfg, flop = _flop_state()
    t = build_tree(
        flop_cfg,
        flop.button,
        flop.board,
        flop.history,
        TreeConfig(spec=SMALL, depth_streets=0, depth_streets_turn=1, leaf_mode="value_net"),
    )
    assert t.last_street == 1 and all(len(t.boards[int(b)]) == 3 for b in t.board_id)


def test_agent_rejects_inconsistent_turn_start_configs():
    oracle = _TurnStartOracle(ShowdownOracle())
    engine = get_engine()
    config = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = make_state(engine, config, 0, FLOP_BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    rng = np.random.default_rng(0)

    def act(cfg, **kw):
        agent = SearchAgent(UniformBlueprint(), cfg, value_predictor=oracle, **kw)
        agent.new_hand(int(s.current_player), config)
        return agent.act(s, int(s.current_player), rng)

    with pytest.raises(ValueError, match="depth_streets: 0"):
        act(_with(TINY_TS, tree={"depth_streets": 1}))
    with pytest.raises(ValueError, match="leaf.turn_net"):
        act(TINY_TS)  # depth_streets_turn defaults to depth_streets (0) without a turn net
    with pytest.raises(ValueError, match="turn-end or river"):
        act(_with(TINY_TS, tree={"depth_streets_turn": 0}), turn_value_predictor=oracle)
    assert act(_with(TINY_TS, tree={"depth_streets_turn": 1})) is not None
