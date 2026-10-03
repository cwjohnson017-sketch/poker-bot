"""ABR learner with its own (richer) action set: off-tree translation and stepping."""

from types import SimpleNamespace

import pytest
import torch

from pokerbot.abstraction.actions import map_offtree
from pokerbot.agents import UniformPolicyAgent
from pokerbot.env import GameConfig, VecNLHE
from pokerbot.env.actions import (
    CHECK_CALL,
    DEFAULT_SPEC,
    FOLD,
    RAISE,
    harmonic_abstract,
    nearest_abstract,
)
from pokerbot.eval.abr import (
    ABRConfig,
    LearnerView,
    ScalarVecPolicy,
    UniformRandomVecPolicy,
    evaluate_br,
    greedy_policy,
    learner_spec_from,
    slot_state,
    train_abr,
)

GAME = GameConfig()  # 100bb: deep enough for many raise sizes
RICH = {
    "streets": [
        [
            ["fold"],
            ["check_call"],
            ["raise_x", 2.0],
            ["raise_x", 2.5],
            ["raise_x", 3.0],
            ["raise_x", 4.0],
            ["raise", 1.0],
            ["allin"],
        ],
        [
            ["fold"],
            ["check_call"],
            ["raise", 0.25],
            ["raise", 0.33],
            ["raise", 0.5],
            ["raise", 0.75],
            ["raise", 1.0],
            ["raise", 1.5],
            ["raise", 2.0],
            ["allin"],
        ],
        [
            ["fold"],
            ["check_call"],
            ["raise", 0.33],
            ["raise", 0.5],
            ["raise", 0.75],
            ["raise", 1.0],
            ["raise", 1.5],
            ["raise", 2.0],
            ["allin"],
        ],
        [
            ["fold"],
            ["check_call"],
            ["raise", 0.33],
            ["raise", 0.5],
            ["raise", 0.75],
            ["raise", 1.0],
            ["raise", 1.5],
            ["raise", 2.0],
            ["raise", 3.0],
            ["allin"],
        ],
    ],
    "max_raises": 6,
}


def test_harmonic_abstract_matches_scalar_translation():
    """At on-tree states, the vectorized pseudo-harmonic mapping of random
    concrete actions equals map_offtree on the scalar engine, for random u."""
    torch.manual_seed(0)
    env = VecNLHE(64, GAME, "cpu", seed=3)
    pol = UniformRandomVecPolicy("cpu", 1)
    checked = raises = interior = 0
    for _ in range(25):
        live = ~env.done
        info = env.legal_info()
        te = env.action_amounts()
        me = env.legal_mask()
        n = env.n
        r = torch.rand(n)
        span = (info.max_raise_to - info.min_raise_to).clamp(min=0)
        amount = info.min_raise_to + (torch.rand(n) * (span + 1).float()).long().clamp(max=span)
        amount = torch.where(r > 0.85, info.max_raise_to, amount)  # some all-ins
        kind = torch.where(
            info.raise_ok & (r > 0.3),
            RAISE,
            torch.where(info.can_fold & (r < 0.15), FOLD, CHECK_CALL),
        )
        u = torch.rand(n, dtype=torch.float64)
        idx = harmonic_abstract(
            env.tab,
            info.street,
            kind,
            amount,
            te,
            me,
            info.pot,
            info.max_bet,
            info.to_call,
            info.max_raise_to,
            u,
        )
        near = nearest_abstract(
            env.tab, info.street, kind, amount, te, legal=me, max_raise_to=info.max_raise_to
        )
        for i in live.nonzero().squeeze(1).tolist():
            if int(env.hist_len[i]) >= env.history_len:
                continue
            st = slot_state(env, i)
            act = SimpleNamespace(kind=int(kind[i]), amount=int(amount[i]))
            fixed_u = SimpleNamespace(random=lambda ui=float(u[i]): ui)
            want = map_offtree(DEFAULT_SPEC, st, act, rng=fixed_u)
            assert int(idx[i]) == want, (i, int(kind[i]), int(amount[i]), int(idx[i]), want)
            checked += 1
            raises += int(kind[i]) == RAISE
            interior += int(idx[i]) != int(near[i])
        _, done = env.step(pol.act(env, live))
        env.reset(done)
    assert checked > 1000 and raises > 400
    assert interior > 50  # genuine between-sizes translations, not only exact or end sizes


def test_learner_view_plays_rich_actions_legally():
    spec = learner_spec_from(RICH)
    env = VecNLHE(48, GAME, "cpu", seed=5, validate=True)  # step_concrete validates legality
    view = LearnerView(env.spec, spec, "harmonic", "cpu", seed=0)
    opp = UniformRandomVecPolicy("cpu", 2)
    br_seat = torch.arange(env.n) % 2
    gen = torch.Generator().manual_seed(4)
    learner_raises = offtree = 0
    for _ in range(200):
        obs = view.obs(env, env.obs())
        assert obs["legal"].shape == (env.n, spec.num_actions)
        live = ~env.done
        br_turn = live & (env.actor == br_seat)
        legal = obs["legal"].float()
        a_br = torch.multinomial(legal, 1, generator=gen).squeeze(1)
        before = env.hist_len.clone()
        _, done = view.step(env, br_turn, a_br, opp.act(env, live & ~br_turn))
        # the learner's own history records its real action (spec index)
        wrote = br_turn & (view.len > 0)
        last = (view.len - 1).clamp(min=0)
        tok = view.tok.gather(1, last[:, None]).squeeze(1)
        rec = (tok - 1) % spec.num_actions
        assert bool((rec[wrote] == a_br[wrote]).all())
        is_r = br_turn & (
            spec.tables("cpu").concrete[env.street.clamp(0, 3)].gather(1, a_br[:, None]).squeeze(1)
            == RAISE
        )
        learner_raises += int(is_r.sum())
        # the env token the opponent saw is in its own (smaller) abstraction
        env_tok = env.hist_tok.gather(1, before.clamp(max=env.history_len - 1)[:, None]).squeeze(1)
        offtree += int((is_r & (env_tok > 0)).sum())
        env.reset(done)
    assert learner_raises > 100 and offtree > 0


def test_default_view_is_the_old_behaviour():
    env = VecNLHE(16, GAME, "cpu", seed=1)
    view = LearnerView(env.spec)
    obs = env.obs()
    assert view.obs(env, obs) is obs and view.num_actions == env.num_actions
    a = evaluate_br(
        lambda o: o["legal"].float().argmax(1),
        UniformRandomVecPolicy("cpu", 0),
        GAME,
        hands=64,
        n_envs=32,
        seed=9,
    )
    b = evaluate_br(
        lambda o: o["legal"].float().argmax(1),
        UniformRandomVecPolicy("cpu", 0),
        GAME,
        hands=64,
        n_envs=32,
        seed=9,
        learner_spec=None,
    )
    assert a.stats.mbb_per_hand == b.stats.mbb_per_hand


def test_abr_trains_with_rich_learner_actions():
    cfg = ABRConfig(
        n_envs=64,
        train_steps=40,
        buffer_size=5000,
        batch_size=64,
        learning_starts=200,
        eval_hands=256,
        eval_envs=128,
        equity_samples=0,
        learner_actions=RICH,
        eval_initial=False,
        log_every=0,
    )
    net, res = train_abr(cfg, UniformRandomVecPolicy("cpu", 0), GAME, "cpu")
    assert net.num_actions == learner_spec_from(RICH).num_actions
    assert res.final.hands == 256
    # the trained learner's greedy policy plays through the view
    ev = evaluate_br(
        greedy_policy(net),
        UniformRandomVecPolicy("cpu", 0),
        GAME,
        hands=128,
        n_envs=64,
        learner_spec=learner_spec_from(RICH),
    )
    assert ev.hands == 128


def test_rich_learner_rejects_scalar_opponents():
    cfg = ABRConfig(n_envs=8, train_steps=1, learner_actions=RICH)
    with pytest.raises(ValueError, match="vectorized opponent"):
        train_abr(cfg, ScalarVecPolicy(UniformPolicyAgent(), seed=0), GAME, "cpu")
