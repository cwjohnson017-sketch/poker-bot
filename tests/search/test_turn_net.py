"""Turn-end value net (DeepStack's auxiliary net): the factored river-card
average, bootstrapped check-down targets, turn-end features and checkpoints,
learning, the one-row-per-leaf provider, the agent's provider choice and the
data / training CLIs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

from pokerbot.agents import RandomAgent
from pokerbot.engine_select import get_engine
from pokerbot.env.actions import CHECK_CALL, DEFAULT_SPEC, FOLD, ActionSpec
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import legal_options, make_state
from pokerbot.search.batch_solver import BatchRiverSolver, river_tree
from pokerbot.search.blueprint import TabularBlueprintFromCallable
from pokerbot.search.combos import NUM_COMBOS, avoids_card, blocked_sum, valid_mask
from pokerbot.search.showdown import naive_showdown
from pokerbot.search.solver import RangeSolver, SolverConfig
from pokerbot.search.tree import TreeConfig, build_tree
from pokerbot.search.turn_data import (
    RiverAveragePredictor,
    TurnGenConfig,
    generate_turn_data,
    turn_checkdown_samples,
    turn_targets,
)
from pokerbot.search.turn_net import (
    TurnEndPredictor,
    TurnFeatureCache,
    checkpoint_kind,
    load_leaf_predictor,
    save_turn_net,
    train_turn_net,
    turn_board_features,
    turn_buckets,
    turn_river_sums,
)
from pokerbot.search.value_leaf import (
    ShowdownOracle,
    TurnEndLeafEvaluator,
    ValueLeafEvaluator,
    make_leaf_evaluator,
    river_average,
)
from pokerbot.search.value_net import (
    RiverValueNet,
    ValueNetConfig,
    ValueNetPredictor,
    opponent_mass,
    save_value_net,
    strength_rank2,
)
from pokerbot.search.value_train import ValueTrainConfig, load_shards

C = NUM_COMBOS
BIG = (("fold",), ("check_call",), ("raise", 1.0), ("allin",))
PASSIVE_STREET = (("fold",), ("check_call",))
SPEC = ActionSpec(streets=(BIG, BIG, BIG, PASSIVE_STREET), max_raises=2)
BOARD = [4, 9, 14, 19, 24]
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _flop_tree(stacks: int = 1000):
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[stacks] * 2, small_blind=50, big_blind=100)
    s = make_state(engine, cfg, 0, BOARD, [])
    s.apply(engine.Action.raise_to(250))
    s.apply(engine.Action.check_call())
    tc = TreeConfig(
        spec=SPEC, depth_streets=1, max_nodes=3000, chance_cards=3, leaf_mode="value_net"
    )
    return build_tree(cfg, s.button, s.board, s.history, tc), s


def _boards4(n: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(n, 52, generator=g).argsort(1)[:, :4]


class _ToyRiver:
    """A nonlinear river 'net' that depends on both ranges, the board, ``c`` and
    ``stack`` (0 on combos that hit the board), to check the plumbing."""

    def predict(self, boards, ranges, c, stack):
        valid = vrg.board_valid(boards.to(ranges.device))[:, None, :]
        r = ranges / ranges.sum(-1, keepdim=True).clamp(min=1e-30)
        z = 300.0 * r - 200.0 * r.flip(1) + (boards.sum(1).to(r.dtype) / 200.0)[:, None, None]
        z = z + (c.to(r.dtype) / 1000.0 - stack.to(r.dtype) / 20000.0)[:, None, None]
        z = z + torch.tensor([0.1, -0.2], dtype=r.dtype, device=r.device)[None, :, None]
        return torch.tanh(z) * valid


def _noisy_net(cfg: ValueNetConfig, seed: int = 0) -> RiverValueNet:
    torch.manual_seed(seed)
    net = RiverValueNet(cfg)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(0.05 * torch.randn_like(p))
    return net.eval()


# --------------------------------------------------------------------------- river average


@pytest.mark.parametrize("river", [ShowdownOracle, _ToyRiver])
def test_river_average_equals_value_leaf_evaluator(river):
    tree, _ = _flop_tree()
    vl = ValueLeafEvaluator(tree, river(), chunk=48 * 7)
    L = vl.num_leaves
    g = torch.Generator().manual_seed(4)
    reach = torch.rand(2, L, C, generator=g, dtype=torch.float64) ** 2
    reach[:, 0] = 0.0  # an unreached leaf
    want = torch.stack([vl.values(p, reach) for p in (0, 1)])
    b4 = torch.tensor([tree.boards[int(tree.board_id[n])] for n in vl.ids.tolist()])
    o = vl.oop_seat
    # (OOP, IP) order, and seat order; leaves shuffled across boards, small chunks
    got = river_average(river(), b4, vl.c, vl.stack, reach[[o, 1 - o]], True, chunk=48 * 4)
    assert torch.allclose(got, want[[o, 1 - o]], rtol=1e-9, atol=1e-9 * float(want.abs().max()))
    perm = torch.randperm(L, generator=g)
    got = river_average(river(), b4[perm], vl.c[perm], vl.stack[perm], reach[:, perm], o == 0)
    assert torch.allclose(got, want[:, perm], rtol=1e-9, atol=1e-9 * float(want.abs().max()))
    assert river_average(river(), b4[:0], vl.c[:0], vl.stack[:0], reach[:, :0]).shape == (2, 0, C)


def test_turn_targets_are_the_exact_checkdown_values():
    """Bootstrapped from ShowdownOracle, the targets are the exact turn-end
    values of a checked-down river: against a dense enumeration of the river
    cards and against BatchRiverSolver on a check-down river tree."""
    b4 = _boards4(2, 1)
    valid = vrg.board_valid(b4)
    g = torch.Generator().manual_seed(2)
    ranges = (torch.rand(2, 2, C, generator=g, dtype=torch.float64) ** 3) * valid[:, None]
    ranges[1, 0] *= 7.0  # any scale
    c, stack = torch.tensor([300, 1200]), torch.tensor([9700, 8800])
    ev = turn_targets(ShowdownOracle(), b4, ranges, c, stack)
    assert ev.shape == (2, 2, C) and ev.dtype == torch.float64
    for i in range(2):
        board = b4[i].tolist()
        r = ranges[i] / ranges[i].sum(-1, keepdim=True)
        for p in (0, 1):
            opp = r[1 - p]
            sd = torch.zeros(C, dtype=torch.float64)
            for x in range(52):
                if x not in board:
                    ok = valid_mask([x]).double()
                    sd += naive_showdown((opp * ok)[None], board + [x])[0] * ok
            m = blocked_sum(opp) * valid[i]
            # v = c * sd / 44 chips; ev = v / (m * 2c)
            ref = torch.where(m > 1e-9, sd / 44 / (2 * m.clamp(min=1e-12)), 0.0)
            assert torch.allclose(ev[i, p], ref, rtol=0, atol=1e-12)
            assert bool((ev[i, p][~valid[i]] == 0).all())
    # the same chance average from the exact river solver on a check-down river
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[10000] * 2, small_blind=50, big_blind=100)
    board = b4[0].tolist()
    rivers = torch.tensor([x for x in range(52) if x not in board])
    r = ranges[0] / ranges[0].sum(-1, keepdim=True)
    masked = r[None] * avoids_card()[rivers][:, None, :].double()
    boards5 = torch.cat([b4[0].expand(48, 4), rivers[:, None]], 1)
    bs = BatchRiverSolver(river_tree(cfg, 300, SPEC, button=1), boards5, masked, None, "cpu")
    bs.solve(1)
    m = opponent_mass(r[None], valid[:1])[0].double()
    for p in (0, 1):
        v = bs.root_values(p, best_response=True).double().sum(0) / 44  # chips
        want = torch.where(m[p] > 1e-9, v / (m[p].clamp(min=1e-12) * 600), 0.0)
        assert torch.allclose(ev[0, p], want, rtol=0, atol=1e-5)


# --------------------------------------------------------------------------- features


def test_turn_features_match_their_definition():
    b4 = _boards4(3, 5)
    st = turn_river_sums(b4)
    f = turn_board_features(b4, 64, 1)
    valid = vrg.board_valid(b4)
    assert torch.equal(f["valid"], valid) and int(valid[0].sum()) == 1128
    assert bool((st["count"][valid] == 46).all()) and bool((st["count"][~valid] == 0).all())
    # direct: rank2 on each of the 48 river boards
    board = b4[0].tolist()
    rivers = [x for x in range(52) if x not in board]
    r2 = strength_rank2(torch.tensor([board + [x] for x in rivers]))  # [48, C]
    ok = r2 >= 0
    S = (r2.clamp(min=0) * ok).sum(0)
    assert torch.equal(st["S"][0], S)
    eq = S.double() / (46 * 2160)
    assert torch.allclose(f["equity"][0].double(), torch.where(valid[0], eq, 0.0), atol=1e-6)
    # 1-D buckets: percentile of the mean river strength, monotone in it, nuts on top
    b = f["bucket"]
    assert bool((b[~valid] == 64).all()) and int(b[valid].max()) == 63
    for i in range(3):
        e, bi = f["equity"][i][valid[i]], b[i][valid[i]]
        order = e.argsort()
        assert bool((bi[order].diff() >= 0).all())
        counts = torch.bincount(bi, minlength=64)  # about 1128 / 64 each, up to tie groups
        assert int((counts > 0).sum()) >= 48 and int(counts.max()) < 5 * 1128 / 64
    # 2-D: 16 mean buckets x 4 spread quantiles, sub-buckets ordered by spread
    b2 = turn_buckets(f["mrank2"], f["vrank2"], 64, 4)
    assert bool((b2[~valid] == 64).all())
    assert torch.equal(b2[valid] // 4, turn_buckets(f["mrank2"], f["vrank2"], 16, 1)[valid])
    for i in range(3):
        bi, sp = b2[i][valid[i]], f["spread"][i][valid[i]]
        counts = torch.bincount(bi, minlength=64).view(16, 4)
        # equal-count sub-buckets up to tie groups (suit isomorphs share S and V)
        assert int(counts.sum()) == 1128 and int((counts == 0).sum()) <= 8
        for mb in range(16):
            sel = (bi // 4) == mb
            sub, s = bi[sel] % 4, sp[sel]
            assert bool((sub[s.argsort()].diff() >= 0).all())
    with pytest.raises(ValueError, match="multiple"):
        turn_buckets(f["mrank2"], f["vrank2"], 64, 3)


def test_turn_feature_cache():
    b4 = _boards4(4, 6)
    rows = b4[torch.tensor([0, 1, 2, 3, 2, 0])][:, torch.randperm(4)]  # card order is irrelevant
    for tables in (True, False):
        for sb in (1, 4):
            cache = TurnFeatureCache("cpu", sb, tables=tables)
            ids = cache.ids(rows)
            assert len(cache) == 4 and ids[0] == ids[5] and ids[2] == ids[4]
            for K in (32, 64):
                f = cache.features(ids, K)
                ref = turn_board_features(rows, K, sb)
                assert f["bucket"].dtype == torch.int32
                for k in ("valid", "pct", "bucket", "onehot"):
                    assert torch.equal(f[k].to(ref[k].dtype), ref[k]), (tables, sb, K, k)
    small = TurnFeatureCache("cpu", max_boards=3)
    small.ids(b4[:3])
    gen = small.generation
    assert small.ids(b4[3:]).tolist() == [0] and small.generation == gen + 1
    with pytest.raises(ValueError, match="4 cards"):
        small.ids(torch.tensor([[0, 1, 2, 3, 4]]))


# --------------------------------------------------------------------------- predictor, checkpoints


def test_turn_end_predictor_and_checkpoints(tmp_path):
    cfg = ValueNetConfig(buckets=32, width=48, layers=2)
    net = _noisy_net(cfg)
    path = tmp_path / "turn.pt"
    save_turn_net(path, net, spread_buckets=4, meta={"note": "x"})
    assert checkpoint_kind(path) == "turn_end"
    pred = load_leaf_predictor(path)
    assert isinstance(pred, TurnEndPredictor) and pred.kind == "turn_end"
    assert pred.spread_buckets == 4 and pred.net.meta["note"] == "x"
    b4 = _boards4(3, 7)
    valid = vrg.board_valid(b4)
    g = torch.Generator().manual_seed(8)
    r = torch.rand(3, 2, C, generator=g) ** 3 * valid[:, None]
    r[0, 1] = 0.0  # an empty range
    c, stack = torch.tensor([100, 700, 3000]), torch.tensor([9900, 9300, 7000])
    ev = pred.predict(b4, r, c, stack)
    assert ev.shape == (3, 2, C) and bool(torch.isfinite(ev).all())
    assert bool((ev[~valid[:, None].expand(3, 2, C)] == 0).all())
    for sb in (4, 1):  # the same net with 1-D buckets reads its inputs differently
        ev_sb = TurnEndPredictor(net, "cpu", spread_buckets=sb).predict(b4, r, c, stack)
        assert torch.equal(ev, ev_sb) == (sb == 4)
    # zero-sum on the 4-card board's ranges
    rn = r[1:] / r[1:].sum(-1, keepdim=True)
    gv = (rn * opponent_mass(rn, valid[1:]) * ev[1:]).sum((1, 2))
    assert float(gv.abs().max()) < 1e-5
    # a river checkpoint loads as a river predictor; kinds are checked
    rpath = tmp_path / "river.pt"
    save_value_net(rpath, net, {})
    assert checkpoint_kind(rpath) == "river"
    rp = load_leaf_predictor(rpath)
    assert type(rp) is ValueNetPredictor and rp.kind == "river"
    with pytest.raises(ValueError, match="not a turn-end"):
        TurnEndPredictor.from_path(rpath)
    odd = _noisy_net(ValueNetConfig(buckets=30, width=16, layers=1))
    with pytest.raises(ValueError, match="multiple"):
        TurnEndPredictor(odd, "cpu", spread_buckets=4)
    with pytest.raises(ValueError, match="spread_buckets"):
        TurnEndPredictor(net, "cpu")  # a bare net: the bucket layout is unknown


# --------------------------------------------------------------------------- learning


def test_learns_turn_checkdown_values():
    torch.manual_seed(0)
    data = turn_checkdown_samples(480, seed=3, boards=160)
    assert data["boards"].shape == (480, 4)
    # zero-sum targets: the range-weighted check-down values cancel
    r, t = data["ranges"], data["targets"]
    m = opponent_mass(r, vrg.board_valid(data["boards"]))
    assert float((r * m * t).sum((1, 2)).abs().max()) < 1e-5
    net_cfg = ValueNetConfig(buckets=64, width=192, layers=3)
    cfg = ValueTrainConfig(steps=250, batch=64, warmup=20, holdout=0.15, seed=1)
    net, rep = train_turn_net(
        None, None, net_cfg, cfg, spread_buckets=4, device="cpu", log=None, raw=data
    )
    o = rep["overall"]
    assert o["samples"] > 30 and rep["meta"]["kind"] == "turn_end"
    assert o["mae"] < 0.35 * o["zero"]["mae"], o
    assert o["wmae"] < 0.35 * o["zero"]["wmae"], o
    assert abs(rep["gv_target_sum_mean"]) < 1e-3
    assert net.meta["turn_features"]["spread_buckets"] == 4


# --------------------------------------------------------------------------- provider


class _Counting:
    def __init__(self, inner):
        self.inner = inner
        self.kind = getattr(inner, "kind", "river")
        self.calls = 0

    def predict(self, *args):
        self.calls += 1
        return self.inner.predict(*args)


def test_turn_end_leaf_evaluator_reproduces_the_river_average():
    """An exact turn-end predictor (river_average of ShowdownOracle) gives the
    same leaf values and the same solve as the river net averaged per leaf."""
    tree, s = _flop_tree()
    vl = ValueLeafEvaluator(tree, ShowdownOracle())
    exact = RiverAveragePredictor(ShowdownOracle())
    te = TurnEndLeafEvaluator(tree, exact)
    assert torch.equal(te.ids, vl.ids) and te.num_rows == te.num_leaves == vl.num_leaves
    assert isinstance(make_leaf_evaluator(tree, exact), TurnEndLeafEvaluator)
    assert type(make_leaf_evaluator(tree, ShowdownOracle())) is ValueLeafEvaluator
    bad = _Counting(ShowdownOracle())
    bad.kind = "flop"
    with pytest.raises(ValueError, match="kind"):
        make_leaf_evaluator(tree, bad)
    g = torch.Generator().manual_seed(9)
    reach = torch.rand(2, te.num_leaves, C, generator=g, dtype=torch.float64)
    for p in (0, 1):
        a, b = te.values(p, reach), vl.values(p, reach)
        assert torch.allclose(a, b, rtol=1e-9, atol=1e-9 * float(b.abs().max()))
    g = torch.Generator().manual_seed(1)
    ranges = torch.rand(2, C, generator=g, dtype=torch.float64) * valid_mask(s.board)
    sc = SolverConfig(dtype="float64")
    sa = RangeSolver(tree, ranges, sc, value_leaves=vl)
    sb = RangeSolver(tree, ranges, sc, value_leaves=te)
    sa.solve(6)
    sb.solve(6)
    ea, eb = sa.exploitability(), sb.exploitability()
    for k in ("br", "ev"):
        for p in (0, 1):
            assert abs(ea[k][p] - eb[k][p]) < 1e-7 * int(s.pot), (k, ea, eb)
    assert torch.allclose(sa.average_strategy(), sb.average_strategy(), atol=1e-7)
    # net_every: the turn-end net runs every n-th regret update per player
    for every, want in ((1, 20), (3, 8)):
        pred = _Counting(exact)
        solver = RangeSolver(tree, ranges, sc, value_leaves=TurnEndLeafEvaluator(tree, pred, every))
        solver.solve(10)
        assert pred.calls == want, (every, pred.calls)
        solver.exploitability()  # 4 fresh evaluations
        assert pred.calls == want + 4


TINY_VN = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"mode": "value_net", "net_every": 1},
    "gadget": {"rollouts": 8},
}


def test_agent_picks_the_provider_from_the_checkpoint_kind(tmp_path):
    net = _noisy_net(ValueNetConfig(buckets=16, width=32, layers=2))
    tpath, rpath = tmp_path / "turn.pt", tmp_path / "river.pt"
    save_turn_net(tpath, net, spread_buckets=2)
    save_value_net(rpath, net, {})
    tree, _ = _flop_tree()

    def agent(path):
        cfg = {**TINY_VN, "leaf": {**TINY_VN["leaf"], "net": str(path)}}
        return SearchAgent(UniformBlueprint(), cfg)

    a = agent(tpath)
    assert isinstance(a.get_value_predictor(), TurnEndPredictor)
    assert isinstance(a.value_leaf_provider(tree), TurnEndLeafEvaluator)
    r = agent(rpath)
    assert type(r.get_value_predictor()) is ValueNetPredictor
    assert type(r.value_leaf_provider(tree)) is ValueLeafEvaluator
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    res = run_match([a, RandomAgent()], config, num_hands=6, seed=3)
    assert res.hands == 6 and int(res.seat_payoffs.sum()) == 0
    flop = [st for st in a.stats if st.get("street") == 1]
    assert flop and not any(st.get("fallback") for st in a.stats)
    for st in flop:
        assert st["value_provider"] == "TurnEndLeafEvaluator"
        assert st["value_net_rows"] > 0 and st["value_net_rows"] % st["value_leaves"] == 0


# --------------------------------------------------------------------------- self-play states


def _passive_blueprint():
    def fn(state, player):
        a, b = state.hole_cards(player)
        x = ((int(a) * 7 + int(b) * 3) % 11) / 10
        w = {}
        for o in legal_options(state, DEFAULT_SPEC):
            w[o.index] = {FOLD: 0.2, CHECK_CALL: 3.0}.get(o.kind, 0.2 * (0.5 + x))
        return w

    return TabularBlueprintFromCallable(fn, DEFAULT_SPEC)


def test_selfplay_turn_end_states_are_river_states_without_the_river():
    bp = _passive_blueprint()
    config = get_engine().GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100
    )
    river = vrg.selfplay_river_states(bp, config, 3, seed=5, explore=0.2)
    turn = vrg.selfplay_river_states(bp, config, 3, seed=5, explore=0.2, turn_end=True)
    assert turn["boards"].shape == (3, 4)
    assert torch.equal(turn["boards"], river["boards"][:, :4])
    assert torch.equal(turn["c"], river["c"]) and torch.equal(turn["holes"], river["holes"])
    valid4, valid5 = vrg.board_valid(turn["boards"]), vrg.board_valid(river["boards"])
    r = turn["ranges"].double()
    assert bool((r[~valid4[:, None].expand_as(r)] == 0).all())
    assert bool((r[(valid4 & ~valid5)[:, None].expand_as(r)] > 0).all())  # hit the river card
    masked = r * valid5[:, None]
    masked = masked / masked.sum(-1, keepdim=True)
    torch.testing.assert_close(masked, river["ranges"].double(), rtol=1e-5, atol=1e-7)


# --------------------------------------------------------------------------- data and CLIs


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_generate_turn_data_and_clis(tmp_path):
    # self-play + perturbed + random states with check-down targets
    bp = _passive_blueprint()
    bp.game = {"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100}
    cfg = TurnGenConfig(samples=12, shard_size=8, seed=2, mix=(0.5, 0.25, 0.25), batch=5)
    meta = generate_turn_data(ShowdownOracle(), bp, tmp_path / "sp", cfg, device="cpu", log=None)
    assert meta["kind"] == "turn_end" and len(meta["shards"]) == 2
    d = load_shards(tmp_path / "sp")
    assert d["boards"].shape == (12, 4) and d["ranges"].shape == (12, 2, C)
    assert sorted(set(d["source"].tolist())) == [0, 1, 2]
    assert bool((d["exploit"] == 0).all())
    want = turn_targets(ShowdownOracle(), d["boards"], d["ranges"].float(), d["c"], d["stack"])
    torch.testing.assert_close(d["targets"].float(), want, rtol=2e-3, atol=2e-4)
    # the CLI: random states only (no blueprint), resume
    gen = _load_script("gen_turn_data")
    out = tmp_path / "rand"
    argv = ["--river-net", "checkdown", "--out", str(out), "--samples", "20", "--mix", "0,0,1"]
    argv += ["--shard-size", "10", "--device", "cpu", "--seed", "4"]
    assert gen.main(argv) == 0
    first = torch.load(out / "shard_00001.pt", weights_only=True)
    (out / "shard_00001.pt").unlink()
    assert gen.main([*argv, "--resume"]) == 0
    again = torch.load(out / "shard_00001.pt", weights_only=True)
    for k, v in first.items():
        assert torch.equal(v, again[k]), k
    assert json.loads((out / "meta.json").read_text())["kind"] == "turn_end"
    with pytest.raises(SystemExit):
        gen.main(["--river-net", "checkdown", "--out", str(tmp_path / "x"), "--samples", "4"])
    # train on it with the CLI: a turn-end checkpoint the agent can load
    train = _load_script("train_turn_net")
    ck = tmp_path / "vn" / "turn.pt"
    argv = ["--data", str(out), "--heldout-data", str(tmp_path / "sp"), "--out", str(ck)]
    argv += ["--device", "cpu", "--steps", "10", "--batch", "8", "--buckets", "16"]
    argv += ["--width", "32", "--layers", "2", "--spread-buckets", "4", "--quiet"]
    assert train.main(argv) == 0
    rep = json.loads(ck.with_suffix(".json").read_text())
    assert rep["train_samples"] == 20 and rep["heldout_samples"] == 12
    pred = load_leaf_predictor(ck)
    assert isinstance(pred, TurnEndPredictor) and pred.spread_buckets == 4
    ev = pred.predict(d["boards"][:3], d["ranges"][:3].float(), d["c"][:3], d["stack"][:3])
    assert bool(torch.isfinite(ev).all())
