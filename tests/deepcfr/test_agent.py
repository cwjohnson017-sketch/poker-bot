import numpy as np
import pytest
import torch
from torch import nn

from pokerbot.agents import RandomAgent, make_agent
from pokerbot.blueprint.deepcfr.agent import NeuralBlueprintAgent
from pokerbot.blueprint.deepcfr.checkpoint import list_checkpoints, save_net, write_meta
from pokerbot.blueprint.deepcfr.config import spec_to_dict
from pokerbot.blueprint.deepcfr.features import FeatureConfig
from pokerbot.blueprint.deepcfr.networks import AdvantageNet, NetConfig
from pokerbot.blueprint.deepcfr.policy import SDCFRPolicy
from pokerbot.config import game_config
from pokerbot.engine_select import get_engine
from pokerbot.env import DEFAULT_SPEC
from pokerbot.eval.match import run_duplicate_match, run_match


def make_run(tmp_path, iters=3):
    root = tmp_path / "checkpoints"
    cfg = NetConfig(card_dim=8, card_hidden=16, hist_dim=8, hist_hidden=16, width=32)
    meta = {
        "spec": spec_to_dict(DEFAULT_SPEC),
        "features": FeatureConfig().to_dict(),
        "game": {"stacks": [20000, 20000], "small_blind": 50, "big_blind": 100, "ante": 0},
        "net_config": cfg.to_dict(),
        "fallback": "uniform",
    }
    write_meta(root, meta)
    for p in (0, 1):
        for t in range(1, iters + 1):
            torch.manual_seed(100 * p + t)
            net = AdvantageNet(cfg)
            with torch.no_grad():
                net.head.weight.mul_(10)
            save_net(root, p, t, net, meta)
    return tmp_path


def test_neural_agent_plays_200_legal_hands_vs_random(tmp_path):
    run = make_run(tmp_path)
    assert [t for t, _ in list_checkpoints(run, 0)] == [1, 2, 3]
    engine = get_engine()
    config = game_config({}, engine)
    agent = make_agent(f"neural:{run}")
    assert isinstance(agent, NeuralBlueprintAgent) and len(agent.policies[0]) == 3
    res = run_match([agent, RandomAgent()], config, 200, seed=3, engine=engine)
    assert res.hands == 200  # on_illegal="raise": any illegal action would have raised
    p = agent.last_probs
    assert p is not None and abs(p.sum() - 1) < 1e-9
    # duplicate, both seats, restricted to the last net
    last = NeuralBlueprintAgent.from_dir(run, last_n=1, name="last")
    res = run_duplicate_match(agent, last, config, 10, seed=1, engine=engine)
    assert res.hands == 20


class FixedNet(nn.Module):
    def __init__(self, adv):
        super().__init__()
        self.adv = torch.tensor(adv)

    def forward(self, feats):
        return self.adv.expand(feats["legal"].shape[0], -1)


def test_sdcfr_average_is_reach_weighted():
    pol = SDCFRPolicy([(1, FixedNet([1.0, -1.0])), (2, FixedNet([1.0, 1.0]))])
    feats = {"legal": torch.ones(1, 2, dtype=torch.bool)}
    # root: (1 * [1, 0] + 2 * [.5, .5]) / 3
    assert torch.allclose(pol.act_probs(feats)[0], torch.tensor([2 / 3, 1 / 3]))
    pol.observe(0)  # reach: net1 1, net2 .5 -> weights 1:1
    assert torch.allclose(pol.act_probs(feats)[0], torch.tensor([0.75, 0.25]))
    pol.observe(1)  # net1 never plays action 1: only net2 remains
    assert torch.allclose(pol.act_probs(feats)[0], torch.tensor([0.5, 0.5]))
    pol.new_hand()
    assert torch.allclose(pol.act_probs(feats)[0], torch.tensor([2 / 3, 1 / 3]))
    flat = SDCFRPolicy(
        [(1, FixedNet([1.0, -1.0])), (2, FixedNet([1.0, 1.0]))], reach_weighted=False
    )
    flat.act_probs(feats)
    flat.observe(0)
    assert torch.allclose(flat.act_probs(feats)[0], torch.tensor([2 / 3, 1 / 3]))
    last = SDCFRPolicy([(1, FixedNet([1.0, -1.0])), (2, FixedNet([1.0, 1.0]))], last_n=1)
    assert len(last) == 1


def test_sdcfr_sampled_net_mixture_is_the_average():
    pol = SDCFRPolicy([(1, FixedNet([1.0, -1.0])), (3, FixedNet([1.0, 1.0]))])
    rng = np.random.default_rng(0)
    draws = np.array([pol.sample_index(rng) for _ in range(4000)])
    assert abs(draws.mean() - 0.75) < 0.03  # net of iteration 3 has weight 3 / 4
    feats = {"legal": torch.ones(1, 2, dtype=torch.bool)}
    assert torch.allclose(pol.net_probs(feats, 0)[0], torch.tensor([1.0, 0.0]))
    assert torch.allclose(pol.net_probs(feats, 1)[0], torch.tensor([0.5, 0.5]))
    # at the root, drawing a net per hand plays exactly the average policy
    w = torch.softmax(pol.log_w, 0).float()
    mix = w[0] * pol.net_probs(feats, 0)[0] + w[1] * pol.net_probs(feats, 1)[0]
    assert torch.allclose(mix, pol.act_probs(feats)[0])


def test_neural_agent_sample_net_keeps_one_net_per_hand(tmp_path):
    run = make_run(tmp_path)
    engine = get_engine()
    config = game_config({}, engine)
    agent = make_agent(f"neural:{run},sample_net=true")
    assert isinstance(agent, NeuralBlueprintAgent) and agent.sample_net
    drawn = []
    orig = agent.policies[0].sample_index

    def spy(rng):
        drawn.append(orig(rng))
        return drawn[-1]

    agent.policies[0].sample_index = spy
    res = run_match([agent, RandomAgent()], config, 60, seed=4, engine=engine)
    assert res.hands == 60  # on_illegal="raise": all actions legal
    # seat 0 draws at most once per hand, whatever the number of decisions
    assert 0 < len(drawn) <= 60 and set(drawn) <= {0, 1, 2}
    assert abs(agent.last_probs.sum() - 1) < 1e-9


def test_agent_greedy_is_deterministic(tmp_path):
    run = make_run(tmp_path, iters=1)
    engine = get_engine()
    config = game_config({}, engine)
    a = NeuralBlueprintAgent.from_dir(run, greedy=True, name="g")
    r1 = run_match([a, a], config, 20, seed=5, engine=engine)
    r2 = run_match([a, a], config, 20, seed=5, engine=engine)
    assert np.array_equal(r1.seat_payoffs, r2.seat_payoffs)


def test_unknown_agent_still_errors():
    with pytest.raises(ValueError):
        make_agent("nope")
