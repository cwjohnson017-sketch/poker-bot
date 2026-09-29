"""Pseudo-harmonic mapping of off-tree opponent raises in the Deep CFR agent."""

from __future__ import annotations

import importlib

import numpy as np
import pytest
import torch

from pokerbot.abstraction.actions import map_offtree, pseudo_harmonic
from pokerbot.agents import AlwaysRaiseAgent
from pokerbot.blueprint.deepcfr.agent import NeuralBlueprintAgent
from pokerbot.blueprint.deepcfr.networks import AdvantageNet, NetConfig
from pokerbot.blueprint.deepcfr.policy import SDCFRPolicy
from pokerbot.blueprint.deepcfr.scalar import ScalarSpec, decision_info, encode_state, engine_config
from pokerbot.config import game_config
from pokerbot.engine_select import get_engine
from pokerbot.env import DEFAULT_SPEC, GameConfig
from pokerbot.eval.match import run_match

ENGINES = ["pokerbot.reference"]
try:
    importlib.import_module("poker_engine")
    ENGINES.append("poker_engine")
except ImportError:  # pragma: no cover
    pass

A = DEFAULT_SPEC.num_actions
BET = 220  # flop, pot 200: between the 0.75 (150) and 1.5 (300) sizes


@pytest.fixture(autouse=True)
def single_torch_thread():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(threads)


def flop_bet(engine_name, bet=BET):
    """Limped pot, flop (pot 200); the first player bets ``bet``. Returns the
    config, the state before the bet and the state after it."""
    engine = importlib.import_module(engine_name)
    cfg = engine_config(engine, GameConfig())
    s = engine.GameState.new_hand(cfg, 0, list(range(52)))
    s.apply(engine.Action.check_call())
    s.apply(engine.Action.check_call())
    pre = engine.GameState.new_hand(cfg, 0, list(range(52)))
    for _, _, a in s.history:
        pre.apply(a)
    info = decision_info(pre, ScalarSpec.build(DEFAULT_SPEC), 0)
    assert info.targets[2:5] == [100, 150, 300] and all(info.legal[2:])
    s.apply(engine.Action.raise_to(bet))
    return cfg, pre, s


def bet_index(feats):
    return (int(feats["hist"][0, 2]) - 1) % A


def harmonic_index(s, seat, cfg, **kw):
    return bet_index(encode_state(s, seat, cfg, DEFAULT_SPEC, offtree="harmonic", **kw)[0])


@pytest.mark.parametrize("engine_name", ENGINES)
def test_harmonic_randomized_frequency_matches_map_offtree(engine_name):
    cfg, pre, s = flop_bet(engine_name)
    seat = s.current_player
    action = s.history[-1][2]
    n = 1500
    rng = np.random.default_rng(7)
    got = [harmonic_index(s, seat, cfg, rng=rng) for _ in range(n)]
    rng2 = np.random.default_rng(7)
    want = [map_offtree(DEFAULT_SPEC, pre, action, rng2) for _ in range(n)]
    assert got == want
    assert set(got) == {3, 4}
    # x = 220 / 200 = 1.1 pot between 0.75 and 1.5: lower size w.p. 0.444
    p = pseudo_harmonic(0.75, 1.5, BET / 200)
    assert p == pytest.approx(0.7 / 1.575)
    freq = got.count(3) / n
    assert abs(freq - p) < 4 * np.sqrt(p * (1 - p) / n)


@pytest.mark.parametrize("engine_name", ENGINES)
def test_harmonic_deterministic_and_nearest(engine_name):
    cfg, pre, s = flop_bet(engine_name)
    seat = s.current_player
    action = s.history[-1][2]
    det = map_offtree(DEFAULT_SPEC, pre, action, mode="deterministic")
    assert det == 4  # u = 0.5 >= 0.444: the 1.5-pot size
    assert harmonic_index(s, seat, cfg) == det
    f, _ = encode_state(s, seat, cfg, DEFAULT_SPEC, offtree="nearest")
    assert bet_index(f) == 3  # closest amount: 150
    with pytest.raises(ValueError):
        encode_state(s, seat, cfg, DEFAULT_SPEC, offtree="nope")


@pytest.mark.parametrize("engine_name", ENGINES)
def test_memo_keeps_the_draw_and_own_actions_stay_nearest(engine_name):
    engine = importlib.import_module(engine_name)
    cfg, _, s = flop_bet(engine_name)
    bettor = s.history[-1][1]
    rng = np.random.default_rng(3)
    memo: dict = {}
    first = harmonic_index(s, 1 - bettor, cfg, rng=rng, memo=memo)
    again = {harmonic_index(s, 1 - bettor, cfg, rng=rng, memo=memo) for _ in range(30)}
    assert again == {first} and len(memo) == 1
    # the bet is the bettor's own action: nearest size whatever the rng
    s.apply(engine.Action.check_call())  # called; turn, the bettor acts first
    assert s.current_player == bettor
    assert {harmonic_index(s, bettor, cfg, rng=rng) for _ in range(30)} == {3}


def make_agent(**kw):
    cfg = NetConfig(card_dim=8, card_hidden=16, hist_dim=8, hist_hidden=16, width=32)
    pols = []
    for p in (0, 1):
        nets = []
        for t in (1, 2):
            torch.manual_seed(100 * p + t)
            net = AdvantageNet(cfg).eval()
            with torch.no_grad():
                net.head.weight.mul_(10)
            nets.append((t, net))
        pols.append(SDCFRPolicy(nets))
    return NeuralBlueprintAgent(pols, DEFAULT_SPEC, **kw)


def test_range_policy_uses_the_deterministic_mapping():
    engine_name = ENGINES[-1]
    cfg, pre, s = flop_bet(engine_name)
    seat = s.current_player
    agent = make_agent()
    assert agent.offtree == "harmonic" and agent.range_policy.offtree == "harmonic"
    rp = agent.range_policy
    f, _ = rp._encode(s, seat, cfg)
    assert bet_index(f) == map_offtree(DEFAULT_SPEC, pre, s.history[-1][2], mode="deterministic")
    p1 = agent.policy(s, seat)
    hole = np.array([list(s.hole_cards(seat))])
    pb = agent.policy_batch(s, seat, hole)[0]
    assert np.allclose(p1, pb, atol=1e-5)
    near = make_agent(offtree="nearest")
    assert bet_index(near.range_policy._encode(s, seat, cfg)[0]) == 3
    with pytest.raises(ValueError):
        make_agent(offtree="nope")


def test_agent_plays_100_legal_hands_vs_always_raise():
    engine = get_engine()
    config = game_config({}, engine)
    agent = make_agent()
    res = run_match([agent, AlwaysRaiseAgent()], config, 100, seed=11, engine=engine)
    assert res.hands == 100  # on_illegal="raise": an illegal action would have raised
    assert agent.offtree_mapped > 0  # pot-sized raises off the spec's sizes
