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


def _check_against_range_reach(bp, config, states, atol):
    """Every sample's ranges equal range_reach replayed on its history, and the
    replayed state is the river root with the sample's c and stack."""
    engine = get_engine()
    m = states["ranges"].shape[0]
    for i in range(m):
        board = states["boards"][i].tolist()
        button = int(states["button"][i])
        state = make_state(engine, config, button, board, [])
        for j in range(int(states["hist_len"][i])):
            kind = int(states["hist_kind"][i, j])
            state.apply(to_action(engine, kind, int(states["hist_amount"][i, j])))
        assert not state.is_terminal and int(state.street) == 3
        assert all(int(s) < 3 for s, _p, _a in state.history)
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
    med = float(c.double().median())
    assert 900 < med < 1100  # log-uniform: median sqrt(100 * 9900) ~ 995
    st = vrg.random_states(8, g)
    assert st["ranges"].shape == (8, 2, NUM_COMBOS)
    assert bool((st["c"] + st["stack"] == 10000).all())


def test_board_valid_matches_valid_masks():
    boards = _boards(16, 6)
    assert torch.equal(vrg.board_valid(boards), valid_masks(boards.tolist()))
    assert torch.equal(vrg.board_valid(boards[:, :3]), valid_masks(boards[:, :3].tolist()))
