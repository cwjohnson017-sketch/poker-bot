"""River-root state sampler and range generators for the leaf value net."""

import pytest
import torch

from pokerbot.engine_select import get_engine
from pokerbot.env.actions import CHECK_CALL, DEFAULT_SPEC, FOLD
from pokerbot.search import value_ranges as vrg
from pokerbot.search.abstract import contributions, legal_options, make_state, to_action
from pokerbot.search.blueprint import (
    TabularBlueprintFromCallable,
    UniformBlueprint,
    make_blueprint,
    range_reach,
)
from pokerbot.search.combos import NUM_COMBOS, combo_index, valid_masks
from pokerbot.search.showdown import combo_strengths


@pytest.fixture
def one_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


@pytest.fixture(scope="module")
def tiny_distilled_run(tiny_neural_run, tmp_path_factory):
    """A single-net softmax blueprint (distilled from the tiny SD-CFR run), so
    the lockstep path applies."""
    from pokerbot.blueprint.deepcfr.distill import DistillConfig, distill

    out = tmp_path_factory.mktemp("distilled_vr")
    threads = torch.get_num_threads()
    try:
        cfg = DistillConfig(rows=3000, n_envs=64, steps=40, batch=256, holdout=0.1)
        distill(tiny_neural_run, out, cfg, device="cpu", log=None)
    finally:
        torch.set_num_threads(threads)
    return out


def _config(bp):
    g = bp.game
    return get_engine().GameConfig(
        num_players=2, stacks=g["stacks"], small_blind=g["small_blind"], big_blind=g["big_blind"]
    )


def _check_against_range_reach(bp, config, states, atol, street=3):
    """Every sample's ranges equal range_reach replayed on its history, and the
    replayed state is the river root (the root of ``street``) with the sample's
    c and stack."""
    engine = get_engine()
    m = states["ranges"].shape[0]
    for i in range(m):
        board = states["boards"][i].tolist()
        button = int(states["button"][i])
        state = make_state(engine, config, button, board, [])
        for j in range(int(states["hist_len"][i])):
            kind = int(states["hist_kind"][i, j])
            state.apply(to_action(engine, kind, int(states["hist_amount"][i, j])))
        assert not state.is_terminal and int(state.street) == street
        assert all(int(s) < street for s, _p, _a in state.history)
        contrib = contributions(state, config)
        assert contrib[0] == contrib[1] == int(states["c"][i])
        assert min(int(s) for s in state.stacks) == int(states["stack"][i]) > 0
        want = range_reach(bp, config, button, board, state.history).double()
        want = want / want.sum(1, keepdim=True)
        oop = 1 - button
        want = torch.stack([want[oop], want[1 - oop]])
        torch.testing.assert_close(states["ranges"][i].double(), want, rtol=1e-3, atol=atol)


def _check_shapes(states, m, start_stack):
    r = states["ranges"]
    assert r.shape == (m, 2, NUM_COMBOS)
    torch.testing.assert_close(r.sum(-1).double(), torch.ones(m, 2, dtype=torch.float64))
    valid = valid_masks(states["boards"].tolist())
    assert bool((r[~valid[:, None].expand_as(r)] == 0).all())
    # public-belief ranges: positive on both players' actual hands (the
    # opponent's cards are never removed)
    for i in range(m):
        for p in range(2):
            own = combo_index(*states["holes"][i, p].tolist())
            assert float(r[i, p, own]) > 0
            assert float(r[i, 1 - p, own]) > 0
    assert bool((states["c"] + states["stack"] == start_stack).all())


def test_lockstep_ranges_match_range_reach(tiny_distilled_run, one_thread):
    bp = make_blueprint(f"neural:{tiny_distilled_run}")
    assert vrg._lockstep_ok(bp)
    config = _config(bp)
    stats = {}
    states = vrg.selfplay_river_states(
        bp, config, 10, device="cpu", seed=3, explore=0.1, n_envs=256, replay_chunk=4, stats=stats
    )
    _check_shapes(states, 10, 2000)  # the tiny run is 20bb
    assert 10 <= stats["kept"] <= stats["hands"]
    assert stats["hands"] == stats["kept"] + stats["folded"] + stats["allin"]
    assert int(states["hist_len"].min()) >= 2  # at least the preflop call and checks
    _check_against_range_reach(bp, config, states, atol=1e-6)


def test_lockstep_turn_start_states(tiny_distilled_run, one_thread):
    """turn_start=True: hands stopped at the turn root, 4-card boards."""
    bp = make_blueprint(f"neural:{tiny_distilled_run}")
    config = _config(bp)
    stats = {}
    states = vrg.selfplay_river_states(
        bp,
        config,
        8,
        device="cpu",
        seed=4,
        explore=0.1,
        n_envs=256,
        replay_chunk=4,
        stats=stats,
        turn_start=True,
    )
    assert states["boards"].shape == (8, 4)
    _check_shapes(states, 8, 2000)
    assert stats["hands"] == stats["kept"] + stats["folded"] + stats["allin"]
    _check_against_range_reach(bp, config, states, atol=1e-6, street=2)
    with pytest.raises(ValueError, match="exclusive"):
        vrg.selfplay_river_states(bp, config, 1, turn_end=True, turn_start=True)


def _passive_blueprint():
    """A card-dependent blueprint without the lockstep path that mostly checks
    and calls (so most hands reach the river) and never gives an action zero
    probability."""

    def fn(state, player):
        a, b = state.hole_cards(player)
        x = ((int(a) * 7 + int(b) * 3) % 11) / 10
        w = {}
        for o in legal_options(state, DEFAULT_SPEC):
            w[o.index] = {FOLD: 0.2, CHECK_CALL: 3.0}.get(o.kind, 0.2 * (0.5 + x))
        return w

    return TabularBlueprintFromCallable(fn, DEFAULT_SPEC)


def test_scalar_path_matches_range_reach():
    bp = _passive_blueprint()
    assert not vrg._lockstep_ok(bp)
    config = get_engine().GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100
    )
    stats = {}
    states = vrg.selfplay_river_states(bp, config, 4, seed=5, explore=0.2, stats=stats)
    _check_shapes(states, 4, 10000)
    assert stats["hands"] == stats["kept"] + stats["folded"] + stats["allin"]
    _check_against_range_reach(bp, config, states, atol=1e-7)


def test_scalar_turn_start_states():
    bp = _passive_blueprint()
    config = get_engine().GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100
    )
    stats = {}
    states = vrg.selfplay_river_states(
        bp, config, 4, seed=6, explore=0.2, stats=stats, turn_start=True
    )
    assert states["boards"].shape == (4, 4)
    _check_shapes(states, 4, 10000)
    _check_against_range_reach(bp, config, states, atol=1e-7, street=2)


def test_uniform_blueprint_gives_uniform_ranges():
    bp = UniformBlueprint()
    config = get_engine().GameConfig(
        num_players=2, stacks=[10000, 10000], small_blind=50, big_blind=100
    )
    states = vrg.selfplay_river_states(bp, config, 4, seed=1)
    valid = valid_masks(states["boards"].tolist()).float()
    want = (valid / valid.sum(1, keepdim=True))[:, None].expand(-1, 2, -1)
    torch.testing.assert_close(states["ranges"], want)
    assert bool((states["c"] + states["stack"] == 10000).all())


def _boards(n, seed):
    g = torch.Generator().manual_seed(seed)
    return vrg.random_boards(n, g)


def test_strength_pct_matches_naive_count():
    boards = _boards(3, 0)
    pct = vrg.strength_pct(boards)
    s = combo_strengths(boards)
    for i in range(3):
        v = s[i] >= 0
        sv = s[i][v]
        V = int(v.sum())
        weaker = (sv[:, None] > sv[None, :]).sum(1).double()
        ties = (sv[:, None] == sv[None, :]).sum(1).double()
        want = (weaker + (ties - 1) / 2) / (V - 1)
        torch.testing.assert_close(pct[i][v].double(), want, rtol=0, atol=1e-6)
        assert bool((pct[i][~v] == 0).all())
        assert abs(float(pct[i][v].mean()) - 0.5) < 1e-6  # midpoints of ties: exact mean


def test_random_ranges_recursive_splits():
    n = 300
    boards = _boards(n, 1)
    g = torch.Generator().manual_seed(2)
    r = vrg.random_ranges(boards, g)
    valid = valid_masks(boards.tolist())
    assert r.shape == (n, NUM_COMBOS)
    torch.testing.assert_close(r.sum(1).double(), torch.ones(n, dtype=torch.float64))
    assert bool((r[~valid] == 0).all()) and bool((r[valid] > 0).all())
    # the weaker half of the strength order gets a U(0, 1) share of the mass:
    # mean 1/2, variance 1/12
    s = combo_strengths(boards).double()
    order = s.argsort(dim=1, stable=True)[:, NUM_COMBOS - 1081 :]
    weak = r.gather(1, order[:, :540]).sum(1).double()  # first half: floor(1081 / 2)
    assert abs(float(weak.mean()) - 0.5) < 0.06
    assert abs(float(weak.var()) - 1 / 12) < 0.025
    # tied combos are spread in random order: the split level is not tied to
    # the combo index
    r2 = vrg.random_ranges(boards[:1].expand(2, -1).contiguous(), g)
    assert not torch.equal(r2[0], r2[1])


def test_perturb_ranges_keeps_ranges_valid():
    n = 64
    boards = _boards(n, 3)
    g = torch.Generator().manual_seed(4)
    base = torch.stack([vrg.random_ranges(boards, g), vrg.random_ranges(boards, g)], 1)
    # one range on a single combo: removing support must never empty it
    one = torch.zeros(NUM_COMBOS)
    vm = valid_masks(boards[:1].tolist())[0]
    one[vm.nonzero()[0, 0]] = 1.0
    base[0, 0] = one
    out = vrg.perturb_ranges(base, boards, g, p_zero=1.0)
    assert out.shape == base.shape
    valid = valid_masks(boards.tolist())[:, None].expand_as(out)
    torch.testing.assert_close(out.sum(-1).double(), torch.ones(n, 2, dtype=torch.float64))
    assert bool((out[~valid] == 0).all())
    assert bool((out.sum(-1) > 0).all())
    changed = (out - base).abs().sum(-1) > 0
    assert bool(changed.view(-1)[1:].all())  # all but the one-combo range changed
    flat = vrg.perturb_ranges(base[:, 0], boards, g)
    assert flat.shape == (n, NUM_COMBOS)
    # zeroing removes support (no uniform mixing, no noise)
    zeroed = vrg.perturb_ranges(
        base[:, 1], boards, g, p_noise=0.0, p_tilt=0.0, p_mix=0.0, p_zero=1.0, max_zero=0.5
    )
    lost = ((base[:, 1] > 0) & (zeroed == 0)).sum(1).float() / (base[:, 1] > 0).sum(1)
    assert 0.1 < float(lost.mean()) < 0.4


def test_random_c_and_states():
    g = torch.Generator().manual_seed(5)
    c = vrg.random_c(20000, g)
    assert int(c.min()) >= 100 and int(c.max()) <= 9900
    assert not bool(((c > 100) & (c < 200)).any())  # no river root commits 1-2 big blinds
    assert 0.05 < float((c == 100).double().mean()) < 0.1  # log(sqrt 2) / log 99 = 7.5%
    snapped = vrg.reachable_c(torch.tensor([50, 100, 141, 142, 199, 200, 777]))
    assert snapped.tolist() == [100, 100, 100, 200, 200, 200, 777]
    med = float(c.double().median())
    assert 900 < med < 1100  # log-uniform: median sqrt(100 * 9900) ~ 995
    st = vrg.random_states(8, g)
    assert st["ranges"].shape == (8, 2, NUM_COMBOS)
    assert bool((st["c"] + st["stack"] == 10000).all())


def test_board_valid_matches_valid_masks():
    boards = _boards(16, 6)
    assert torch.equal(vrg.board_valid(boards), valid_masks(boards.tolist()))
    assert torch.equal(vrg.board_valid(boards[:, :3]), valid_masks(boards[:, :3].tolist()))


# -- value_data: batching, solving, shards -------------------------------------


class _FakeTree:
    def __init__(self, c):
        self.c = c


def _fake_solver_module(calls):
    """Stand-in for pokerbot.search.batch_solver: best-response values
    ``(p + 1) * 0.1 * c * m_{-p}`` (so the targets are ``0.05 * (p + 1)``) and
    exploitability ``0.02 * c`` chips (0.01 pot)."""
    import types

    from pokerbot.search.combos import blocked_sum

    mod = types.ModuleType("pokerbot.search.batch_solver")

    def river_tree(game_config, c, spec, button=1, device="cpu"):
        assert button == 1
        calls["trees"].append(int(c))
        return _FakeTree(int(c))

    class BatchRiverSolver:
        def __init__(self, tree, boards, ranges, cfg=None, device=None):
            assert boards.shape[0] == ranges.shape[0] and ranges.shape[1:] == (2, NUM_COMBOS)
            self.tree, self.boards, self.ranges = tree, boards, ranges
            calls["solves"].append((tree.c, boards.shape[0]))

        def solve(self, iterations):
            calls["iterations"].append(iterations)
            return {}

        def root_values(self, player, sigma=None, best_response=False, ranges=None):
            assert best_response
            m = blocked_sum(self.ranges[:, 1 - player])
            return m * (player + 1) * 0.1 * self.tree.c

        def exploitability(self, ranges=None):
            B = self.boards.shape[0]
            c = float(self.tree.c)
            return {"exploitability": torch.full((B,), 0.02 * c), "pot": torch.full((B,), 2 * c)}

    mod.river_tree = river_tree
    mod.BatchRiverSolver = BatchRiverSolver
    return mod


@pytest.fixture
def fake_solver(monkeypatch):
    import sys

    calls = {"trees": [], "solves": [], "iterations": []}
    monkeypatch.setitem(sys.modules, "pokerbot.search.batch_solver", _fake_solver_module(calls))
    return calls


def test_make_batches_sorts_by_c_and_uses_the_median():
    from pokerbot.search.value_data import make_batches, mix_counts

    c = torch.tensor([900, 50, 300, 120, 7000, 301, 302, 5000, 60, 150, 170, 180])
    batches = make_batches({"c": c}, 4)
    assert [b[0].numel() for b in batches] == [4, 4, 4]
    flat = torch.cat([b[0] for b in batches])
    assert sorted(flat.tolist()) == list(range(12))
    assert c[flat].tolist() == sorted(c.tolist())
    # medians 135 (-> 100: one to two big blinds is unreachable), 240.5 (-> 240), 2950
    assert [b[1] for b in batches] == [100, 240, 2950]
    assert make_batches({"c": torch.tensor([150, 160, 170])}, 4)[0][1] == 200
    assert make_batches({"c": c}, 12, max_c=200)[0][1] == 200
    assert mix_counts(4096, (0.5, 0.25, 0.25)) == (2048, 1024, 1024)
    assert sum(mix_counts(7, (0.5, 0.3, 0.2))) == 7


def test_solve_batch_targets_and_format(fake_solver):
    from pokerbot.search.value_data import SHARD_DTYPES, solve_batch

    g = torch.Generator().manual_seed(7)
    st = vrg.random_states(6, g)
    st["source"] = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.uint8)
    config = vrg.engine_game_config({"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100})
    rows = solve_batch(st, 1234, DEFAULT_SPEC, config, 17, device="cpu")
    assert fake_solver["trees"] == [1234] and fake_solver["iterations"] == [17]
    assert {k: v.dtype for k, v in rows.items()} == SHARD_DTYPES
    assert rows["ranges"].shape == rows["targets"].shape == (6, 2, NUM_COMBOS)
    assert bool((rows["c"] == 1234).all()) and bool((rows["stack"] == 10000 - 1234).all())
    assert torch.equal(rows["boards"].long(), st["boards"])
    assert torch.equal(rows["source"], st["source"])
    torch.testing.assert_close(rows["exploit"], torch.full((6,), 0.01))
    valid = valid_masks(st["boards"].tolist())
    for p in range(2):
        t = rows["targets"][:, p].float()
        assert bool((t[~valid] == 0).all())
        want = torch.full_like(t[valid], 0.05 * (p + 1))
        torch.testing.assert_close(t[valid], want, rtol=1e-3, atol=0)  # fp16 storage
    # the stored fp16 ranges are the solved ranges: normalised up to fp16 rounding
    torch.testing.assert_close(rows["ranges"].float().sum(-1), torch.ones(6, 2), rtol=0, atol=2e-3)


def test_generate_shards_and_resume(fake_solver, tmp_path):
    from pokerbot.search.value_data import GenConfig, generate

    bp = _passive_blueprint()
    bp.game = {"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100}
    cfg = GenConfig(samples=20, shard_size=8, batch=4, iterations=9, seed=3, n_envs=16)
    meta = generate(bp, tmp_path, cfg, device="cpu", log=None)
    names = sorted(p.name for p in tmp_path.glob("shard_*.pt"))
    assert names == ["shard_00000.pt", "shard_00001.pt", "shard_00002.pt"]
    shards = [torch.load(tmp_path / n, weights_only=True) for n in names]
    assert [int(s["c"].shape[0]) for s in shards] == [8, 8, 4]
    s0 = shards[0]
    assert sorted(s0["source"].tolist()) == [0, 0, 0, 0, 1, 1, 2, 2]
    assert bool((s0["c"] + s0["stack"] == 10000).all())
    assert set(fake_solver["iterations"]) == {9}
    assert meta["config"]["samples"] == 20 and len(meta["shards"]) == 3
    assert meta["timing"]["samples_written"] == 20
    assert "exploit_p90" in meta["shards"]["shard_00001.pt"]
    # resume: only the missing shard is regenerated, identically
    (tmp_path / "shard_00001.pt").unlink()
    n_solves = len(fake_solver["solves"])
    generate(bp, tmp_path, cfg, device="cpu", resume=True, log=None)
    assert len(fake_solver["solves"]) - n_solves == 2  # 8 samples in batches of 4
    again = torch.load(tmp_path / "shard_00001.pt", weights_only=True)
    for k, v in shards[1].items():
        assert torch.equal(v, again[k]), k
    with pytest.raises(FileExistsError):
        generate(bp, tmp_path, cfg, device="cpu", log=None)
    cfg2 = GenConfig(samples=20, shard_size=8, batch=4, iterations=10, seed=3, n_envs=16)
    with pytest.raises(ValueError, match="iterations"):
        generate(bp, tmp_path, cfg2, device="cpu", resume=True, log=None)


def test_solve_batch_with_the_real_solver():
    pytest.importorskip("pokerbot.search.batch_solver")
    from pokerbot.search.combos import blocked_sum
    from pokerbot.search.value_data import solve_batch

    g = torch.Generator().manual_seed(8)
    st = vrg.random_states(3, g)
    config = vrg.engine_game_config({"stacks": [10000, 10000], "small_blind": 50, "big_blind": 100})
    rows = solve_batch(st, 600, DEFAULT_SPEC, config, 50, device="cpu")
    t = rows["targets"].float()
    assert bool(torch.isfinite(t).all()) and bool((rows["exploit"] > 0).all())
    # sum_c r_p(c) m_{-p}(c) ev_p(c) is player p's best-response value per unit
    # of pair mass Z, in pots; the two add up to 2 Z exploit
    r = rows["ranges"].float()
    terms = [r[:, p] * blocked_sum(r[:, 1 - p]) * t[:, p] for p in range(2)]
    gv = sum(x.sum(-1) for x in terms)
    Z = (r[:, 0] * blocked_sum(r[:, 1])).sum(-1)
    scale = sum(x.abs().sum(-1) for x in terms)
    torch.testing.assert_close(gv, 2 * Z * rows["exploit"], rtol=0, atol=float(2e-3 * scale.max()))
