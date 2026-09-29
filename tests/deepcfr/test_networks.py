import pytest
import torch

from pokerbot.blueprint.deepcfr.features import FeatureConfig, features_from_obs
from pokerbot.blueprint.deepcfr.networks import (
    AdvantageNet,
    NetConfig,
    StrategyHead,
    num_params,
    regret_matching,
)
from pokerbot.env import GameConfig, VecNLHE


def _feats(n=32, seed=0, steps=3, **obs_kwargs):
    env = VecNLHE(n, GameConfig(), "cpu", seed=seed, validate=False)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):  # get some history and boards
        a = torch.multinomial(env.legal_mask().float(), 1, generator=g).squeeze(1)
        a = torch.where(a == 0, 1, a)  # avoid folds so most slots stay live
        env.step(a)
    return features_from_obs(env.obs(**obs_kwargs))


@pytest.mark.parametrize("hist_type", ["gru", "transformer"])
def test_forward_shapes_small(hist_type):
    f = _feats()
    net = AdvantageNet(NetConfig(hist_type=hist_type, card_hidden=32, hist_dim=16, hist_hidden=32))
    out = net(f)
    assert out.shape == (32, 6) and torch.isfinite(out).all()


def test_default_size_in_design_range():
    for t in ("gru", "transformer"):
        n = num_params(AdvantageNet(NetConfig(hist_type=t)))
        assert 2_000_000 <= n <= 4_000_000, (t, n)


def test_equity_features_widen_scalars():
    fc = FeatureConfig(equity_samples=16, hist_runouts=2, hist_opp_samples=16)
    f = _feats(n=8, **fc.obs_kwargs())
    assert f["scalars"].shape == (8, fc.num_scalars) == (8, 25)
    net = AdvantageNet(NetConfig(num_scalars=fc.num_scalars, card_hidden=16, hist_hidden=16))
    assert net(f).shape == (8, 6)


def test_masking_and_padding_are_respected():
    torch.manual_seed(0)
    net = AdvantageNet(NetConfig(card_hidden=32, hist_dim=16, hist_hidden=32)).eval()
    f = _feats(n=16)
    base = net(f)
    # undealt board cards: their card value must not matter
    g = {k: v.clone() for k, v in f.items()}
    hidden = ~g["card_mask"]
    assert hidden.any()
    g["cards"][hidden] = 7
    assert torch.allclose(net(g), base, atol=1e-5)
    # padded history positions: their amounts must not matter
    g = {k: v.clone() for k, v in f.items()}
    pad = g["hist"] == 0
    g["hist_amt"][pad] = 5.0
    assert torch.allclose(net(g), base, atol=1e-5)
    # a batch row does not depend on the other rows (packed GRU by length)
    one = net({k: v[3:4] for k, v in f.items()})
    assert torch.allclose(one, base[3:4], atol=1e-5)


def test_regret_matching():
    adv = torch.tensor([[1.0, -2.0, 3.0, 5.0], [-1.0, -2.0, -3.0, 4.0], [0.0, 2.0, 0.0, 0.0]])
    legal = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 0], [1, 0, 1, 1]], dtype=torch.bool)
    p = regret_matching(adv, legal)
    assert torch.allclose(p[0], torch.tensor([0.25, 0.0, 0.75, 0.0]))
    # no positive legal advantage (the positive one is illegal): uniform over legal
    assert torch.allclose(p[1], torch.tensor([1 / 3, 1 / 3, 1 / 3, 0.0]))
    assert torch.allclose(p[2], torch.tensor([1 / 3, 0.0, 1 / 3, 1 / 3]))
    pa = StrategyHead("argmax")(adv, legal)
    assert torch.equal(pa[1], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    assert torch.allclose(StrategyHead()(adv, legal), p)
