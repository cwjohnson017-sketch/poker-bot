import numpy as np
import pytest
import torch

from pokerbot.agents import (
    AlwaysCallAgent,
    FixedPolicyAgent,
    RandomAgent,
    SampledPolicyAgent,
    UniformPolicyAgent,
    as_policy_agent,
)
from pokerbot.agents.policy import (
    PolicyAgentBase,
    abstract_actions,
    legal_vector,
    nearest_abstract_index,
    policy_vector,
    range_policy,
)
from pokerbot.config import game_config
from pokerbot.engine_select import get_engine
from pokerbot.env import GameConfig, VecNLHE
from pokerbot.eval import run_duplicate_match
from pokerbot.eval.abr import slot_state


def test_scalar_abstract_actions_match_vec_env():
    """Replaying every slot's abstract history through the scalar engine gives
    the same legal mask, raise amounts and pot as the torch env."""
    env = VecNLHE(24, GameConfig(), "cpu", seed=3)
    g = torch.Generator().manual_seed(0)
    checked = 0
    for _ in range(40):
        legal = env.legal_mask()
        amounts = env.action_amounts()
        for i in range(env.n):
            if bool(env.done[i]):
                continue
            st = slot_state(env, i)
            assert st.current_player == int(env.actor[i])
            assert st.pot == int(env.pot[i])
            assert legal_vector(st).tolist() == legal[i].tolist()
            for c in abstract_actions(st):
                if c.kind == 2:
                    assert c.amount == int(amounts[i, c.index])
                assert nearest_abstract_index(st, c.action()) == c.index
            checked += 1
        a = torch.multinomial(legal.float(), 1, generator=g).squeeze(1)
        _, done = env.step(a)
        env.reset(done)
    assert checked > 300


def test_policy_vector_normalizes_and_masks():
    legal = np.array([False, True, True, False])
    assert policy_vector({1: 2.0, 0: 5.0}, legal).tolist() == [0.0, 1.0, 0.0, 0.0]
    got = policy_vector(torch.tensor([1.0, 1.0, 3.0, 1.0]), legal).tolist()
    assert got == [0.0, 0.25, 0.75, 0.0]
    assert policy_vector({0: 1.0}, legal).tolist() == [0.0, 0.5, 0.5, 0.0]


def test_policy_agents_play_legal_matches():
    cfg = game_config({})
    res = run_duplicate_match(UniformPolicyAgent(), FixedPolicyAgent("check_call"), cfg, 30)
    assert res.hands == 60


class PlainUniform(PolicyAgentBase):
    """Uniform policy without ``policy_batch`` (exercises the per-hand path)."""

    name = "plain_uniform"

    def policy(self, state, seat):
        return legal_vector(state, self.spec).astype(float)


def test_as_policy_agent_and_range_policy():
    cfg = game_config({})
    assert isinstance(as_policy_agent(AlwaysCallAgent()), FixedPolicyAgent)
    u = UniformPolicyAgent()
    assert as_policy_agent(u) is u
    with pytest.warns(UserWarning, match="estimating"):
        sampled = as_policy_agent(RandomAgent(), samples=4)
    assert isinstance(sampled, SampledPolicyAgent)

    eng = get_engine()
    st = eng.GameState.new_hand(cfg, 0, list(range(52)))
    holes = np.array([[10, 20], [30, 40], [12, 13]])
    p1 = range_policy(u, st, 0, holes, cfg, known={1: [2, 3]})
    p2 = range_policy(PlainUniform(), st, 0, holes, cfg, known={1: [2, 3]})
    np.testing.assert_allclose(p1, p2)
    np.testing.assert_allclose(p1.sum(1), 1.0)
    p3 = range_policy(sampled, st, 0, holes, cfg, known={1: [2, 3]})
    np.testing.assert_allclose(p3.sum(1), 1.0)
