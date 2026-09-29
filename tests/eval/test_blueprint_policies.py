"""The trained blueprints as ``PolicyAgent``s (LBR, ABR) and the neural ``vec_policy``."""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
import yaml

from pokerbot.agents import PolicyAgent, RandomAgent, as_policy_agent, make_agent
from pokerbot.agents.policy import hypothetical_state, legal_vector
from pokerbot.config import REPO_ROOT
from pokerbot.engine_select import get_engine
from pokerbot.env import GameConfig, VecNLHE
from pokerbot.eval.abr import main as abr_main
from pokerbot.eval.abr import make_vec_policy
from pokerbot.eval.lbr import COMBOS, blocked_mask, run_lbr
from pokerbot.eval.masking import MaskedState
from pokerbot.eval.match import run_match


def engine_config(engine, game):
    return engine.GameConfig(
        num_players=2,
        stacks=list(game["stacks"]),
        small_blind=int(game["small_blind"]),
        big_blind=int(game["big_blind"]),
        ante=int(game.get("ante", 0)),
    )


class Recorder:
    """Wraps an agent and keeps the masked view at each of its decisions."""

    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name
        self.views = []

    def new_hand(self, seat, config):
        self.agent.new_hand(seat, config)

    def observe_end(self, state):
        self.agent.observe_end(state)

    def act(self, state, seat, rng):
        # a snapshot: the view wraps the live state, which moves on
        snap = MaskedState(state.clone(), seat, state.config)
        a = self.agent.act(state, seat, rng)
        self.views.append((snap, seat, getattr(self.agent, "last_probs", None)))
        return a


def check_policy_batch(agent, config, hands, seed, per_view, atol):
    engine = get_engine()
    rec = Recorder(agent)
    run_match([rec, RandomAgent()], config, num_hands=hands, seed=seed, engine=engine)
    rng = np.random.default_rng(seed)
    checked = 0
    for view, seat, _ in rec.views:
        board = list(view.board)
        live = np.nonzero(~blocked_mask(board))[0]
        holes = COMBOS[rng.choice(live, per_view, replace=False)]
        batch = np.asarray(agent.policy_batch(view, seat, holes))
        legal = legal_vector(view, agent.spec)
        for k, h in enumerate(holes):
            st = hypothetical_state(view, seat, h, config, engine=engine)
            one = np.asarray(agent.policy(MaskedState(st, seat, config), seat))
            np.testing.assert_allclose(one, batch[k], atol=atol)
            assert abs(one.sum() - 1) < 1e-6 and not one[~legal].any()
            checked += 1
    return rec, checked


def test_tabular_blueprint_policy_batch_matches_policy(small_strategy):
    agent = make_agent(f"blueprint:{small_strategy}")
    assert isinstance(agent, PolicyAgent) and hasattr(agent, "policy_batch")
    assert as_policy_agent(agent) is agent
    engine = get_engine()
    config = engine_config(engine, agent.default_game)
    _, checked = check_policy_batch(agent, config, 12, seed=1, per_view=12, atol=1e-6)
    assert checked > 200


def test_neural_blueprint_policy_batch_matches_policy_and_act(tiny_neural_run):
    agent = make_agent(f"neural:{tiny_neural_run}")
    assert isinstance(agent, PolicyAgent) and hasattr(agent, "policy_batch")
    engine = get_engine()
    config = engine_config(engine, agent.trained_game)
    rec, checked = check_policy_batch(agent, config, 8, seed=2, per_view=4, atol=1e-5)
    assert checked > 30
    # the stateless policy equals what the agent sampled from when it acted
    # (its own reach tracked through the hand by SDCFRPolicy.observe)
    for view, seat, probs in rec.views:
        np.testing.assert_allclose(agent.policy(view, seat), probs, atol=1e-5)


def test_lbr_against_tabular_blueprint_uses_exact_policy(small_strategy):
    agent = make_agent(f"blueprint:{small_strategy}")
    engine = get_engine()
    config = engine_config(engine, agent.default_game)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_lbr(agent, config, hands=30, seed=0, engine=engine, max_runouts=16)
    assert res.hands == 30
    assert not [w for w in caught if "exposes no policy" in str(w.message)]
    assert np.isfinite(res.raw.mbb_per_hand)


def test_neural_vec_policy_plays_legal_actions(tiny_neural_run):
    agent = make_agent(f"neural:{tiny_neural_run}")
    pol = make_vec_policy(f"neural:{tiny_neural_run}", "cpu")
    assert type(pol).__name__ == "NeuralVecPolicy"
    env = VecNLHE(64, GameConfig(stacks=[2000, 2000]), "cpu", seed=1, spec=agent.spec)
    for _ in range(60):
        a = pol.act(env, ~env.done)
        legal = env.legal_mask()
        assert bool(legal.gather(1, a[:, None]).all())
        env.step(a, validate=True)
        env.reset(env.done)
    with pytest.raises(ValueError):
        pol.act(VecNLHE(4, GameConfig(), "cpu", seed=0), None)  # default spec: mismatch


def test_neural_vec_policy_matches_scalar_policy(tiny_neural_run):
    """The per-slot average of the vec policy (own reach tracked across the
    hand) equals the stateless scalar policy on the replayed slot state."""
    from pokerbot.eval.abr import slot_state

    agent = make_agent(f"neural:{tiny_neural_run}")
    pol = agent.vec_policy("cpu", seed=3)
    engine = get_engine()
    cfg = engine_config(engine, agent.trained_game)
    env = VecNLHE(16, GameConfig(stacks=[2000, 2000]), "cpu", seed=4, spec=agent.spec)
    from pokerbot.blueprint.deepcfr.features import features_from_obs, index_features

    checked = 0
    for _ in range(12):
        live = (~env.done).nonzero().squeeze(1).tolist()
        # expected: the scalar policy of each live slot before acting
        want = {}
        for i in live[:6]:
            st = slot_state(env, i, engine, cfg)
            want[i] = agent.policy(st, st.current_player)
        # the vec policy's average for those slots (same computation as act)
        feats = features_from_obs(env.obs())
        for i, w in want.items():
            seat = int(env.actor[i])
            p = pol.policies[seat]
            pol._sync(env)
            lr = pol._log_reach[seat][:, [i]]
            avg, _ = p.average(index_features(feats, torch.tensor([i])), lr)
            np.testing.assert_allclose(avg[0].double().numpy(), w, atol=1e-5)
            checked += 1
        a = pol.act(env, ~env.done)
        env.step(a, validate=True)
        env.reset(env.done)
    assert checked > 30


def test_abr_tiny_config_against_neural_vec_policy(tiny_neural_run, tmp_path, capsys):
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "abr_tiny.yaml").read_text())
    cfg["device"] = "cpu"
    cfg["opponent"] = f"neural:{tiny_neural_run}"
    cfg["game"]["stacks"] = [2000, 2000]
    cfg["abr"].update(
        n_envs=16,
        learning_starts=32,
        batch_size=32,
        eval_hands=64,
        eval_envs=32,
        log_every=10,
        equity_samples=0,
    )
    path = tmp_path / "abr.yaml"
    path.write_text(yaml.safe_dump(cfg))
    threads = torch.get_num_threads()
    try:
        assert abr_main(["--config", str(path), "--steps", "20"]) == 0
    finally:
        torch.set_num_threads(threads)
    assert "ABR vs" in capsys.readouterr().out
