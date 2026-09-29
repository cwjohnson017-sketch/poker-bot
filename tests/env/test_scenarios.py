"""Hand-written hands through ``replay_check`` and the abstract sizing rules."""

import pytest
import torch

from pokerbot.env import CHECK_CALL, FOLD, RAISE, GameConfig, VecNLHE
from pokerbot.env.cards import cards_from_str, deck_from_hands

F = (FOLD, 0)
C = (CHECK_CALL, 0)


def R(x):
    return (RAISE, x)


def deck(h0, h1, board=""):
    return deck_from_hands(cards_from_str(h0), cards_from_str(h1), cards_from_str(board))


@pytest.fixture
def env():
    return VecNLHE(1, GameConfig(), "cpu", seed=0)


def test_fold_preflop(env):
    out = env.replay_check(deck("AsAh", "7c2d"), [F], button=0)
    assert out["payoffs"] == [-50, 50] and out["board"] == [] and out["terminal"]
    first = out["legal"][0]
    assert first["current_player"] == 0 and first["can_fold"] and not first["can_check"]
    assert (first["call_amount"], first["min_raise_to"], first["max_raise_to"]) == (50, 200, 20000)
    assert first["pot"] == 150 and first["street_bets"] == [50, 100]
    out = env.replay_check(deck("AsAh", "7c2d"), [F], button=1)
    assert out["payoffs"] == [50, -50]
    assert out["legal"][0]["current_player"] == 1


def test_bb_folds_to_open(env):
    out = env.replay_check(deck("AsAh", "7c2d"), [R(250), F], button=0)
    assert out["payoffs"] == [100, -100]
    assert out["legal"][1]["min_raise_to"] == 400  # 250 + (250 - 100)


def test_allin_preflop_runs_out_board(env):
    d = deck("AsAh", "KsKh", "2c7d9h3c4d")
    out = env.replay_check(d, [R(20000), C], button=1)
    assert out["terminal"] and out["payoffs"] == [20000, -20000]
    assert out["board"] == cards_from_str("2c7d9h3c4d")
    facing = out["legal"][1]
    assert facing["current_player"] == 0 and facing["call_amount"] == 19900
    assert facing["min_raise_to"] == 0 and facing["max_raise_to"] == 0  # opponent is all-in
    assert out["stacks"] == [40000, 0]


def test_check_down_split_pot(env):
    d = deck("2c3d", "4h5h", "AsKsQsJsTs")
    out = env.replay_check(d, [C, C] + [C, C] * 3, button=0)
    assert out["payoffs"] == [0, 0] and out["street"] == 3 and len(out["board"]) == 5
    actors = [s["current_player"] for s in out["legal"]]
    assert actors == [0, 1, 1, 0, 1, 0, 1, 0]  # button first preflop, other seat first after
    streets = [s["street"] for s in out["legal"]]
    assert streets == [0, 0, 1, 1, 2, 2, 3, 3]
    bb_option = out["legal"][1]
    assert bb_option["can_check"] and not bb_option["can_fold"] and bb_option["min_raise_to"] == 200
    flop_first = out["legal"][2]
    assert flop_first["min_raise_to"] == 100 and flop_first["max_raise_to"] == 19900


def test_showdown_winner_takes_matched_pot(env):
    d = deck("QcQd", "JcJd", "2c5d9hKc3s")
    acts = [R(300), C, R(200), R(1000), C, C, C, R(2500), C]
    out = env.replay_check(d, acts, button=0)
    # contributions: 300 preflop + 1000 flop + 0 turn + 2500 river each
    assert out["payoffs"] == [3800, -3800]


def test_min_raise_tracking(env):
    d = deck("AsAh", "KsKh", "2c7d9h3c4d")
    out = env.replay_check(d, [R(300), R(1000), C, R(100), R(300), C, C, C, C, C], button=0)
    lg = out["legal"]
    assert lg[1]["min_raise_to"] == 500  # 300 + 200
    assert lg[2]["min_raise_to"] == 1700  # 1000 + 700
    assert lg[3]["min_raise_to"] == 100 and lg[3]["current_player"] == 1  # flop bet >= big blind
    assert lg[4]["min_raise_to"] == 200
    assert lg[5]["min_raise_to"] == 500
    assert out["payoffs"] == [1300, -1300]


def test_postflop_allin_runs_out(env):
    d = deck("QcQd", "JcJd", "2c5d9hKc3s")
    out = env.replay_check(d, [C, C, R(19900), C], button=0)
    assert out["terminal"] and out["board"] == cards_from_str("2c5d9hKc3s")
    assert out["payoffs"] == [20000, -20000]


def test_short_stack_calls_allin_for_less():
    env = VecNLHE(1, GameConfig(stacks=[1000, 20000]), "cpu", seed=0)
    d = deck("AsAh", "7c2d", "3c8d9hKc4s")
    out = env.replay_check(d, [R(5000), C], button=1)
    assert out["legal"][1]["call_amount"] == 900
    assert out["payoffs"] == [1000, -1000]
    # short stack shoves; the big stack folds its big blind
    out = env.replay_check(d, [R(1000), F], button=0)
    assert out["legal"][1]["call_amount"] == 900 and out["legal"][1]["min_raise_to"] == 0
    assert out["payoffs"] == [100, -100]


def test_incomplete_allin_raise():
    env = VecNLHE(1, GameConfig(stacks=[20000, 450]), "cpu", seed=0)
    d = deck("7c2d", "AsAh", "3c8d9hKc4s")
    out = env.replay_check(d, [R(300), R(450), C], button=0)
    bb = out["legal"][1]
    assert bb["min_raise_to"] == 500 and bb["max_raise_to"] == 450  # all-in for less is allowed
    assert out["legal"][2]["call_amount"] == 150 and out["legal"][2]["min_raise_to"] == 0
    assert out["payoffs"] == [-450, 450]


def test_illegal_replays_raise(env):
    d = deck("AsAh", "7c2d")
    with pytest.raises(ValueError):
        env.replay_check(d, [C, F])  # BB folds when it can check
    with pytest.raises(ValueError):
        env.replay_check(d, [R(150)])  # below the minimum raise
    with pytest.raises(ValueError):
        env.replay_check(d, [R(20001)])  # more than the stack
    with pytest.raises(ValueError):
        env.replay_check(d, [F, C])  # action after the hand ended
    with pytest.raises(ValueError):
        env.replay_check(d[:-1] + [0], [F])  # not a permutation


def _abstract_env(button=0):
    env = VecNLHE(1, GameConfig(), "cpu", seed=0)
    env.reset(button=button)
    return env


def test_preflop_abstract_sizes_and_dedupe():
    env = _abstract_env()
    # fold, call, 2.5x, 3x, pot, all-in
    assert env.action_amounts()[0, 2:].tolist() == [250, 300, 300, 20000]
    assert env.legal_mask()[0].tolist() == [True, True, True, True, False, True]  # pot == 3x here
    env.step(torch.tensor([2]))
    assert env.street_bets[0].tolist() == [250, 100]
    assert env.action_amounts()[0, 2:].tolist() == [625, 750, 750, 20000]
    assert env.legal_mask()[0].tolist() == [True, True, True, True, False, True]
    env.step(torch.tensor([1]))  # call: flop, pot 500
    assert int(env.street[0]) == 1 and int(env.actor[0]) == 1
    # 0.33, 0.75, 1.5 pot bets
    assert env.action_amounts()[0, 2:].tolist() == [165, 375, 750, 19750]
    assert env.legal_mask()[0].tolist() == [False, True, True, True, True, True]
    env.step(torch.tensor([2]))  # bet 165
    # raises: 165 + round(x * (665 + 165))
    assert env.action_amounts()[0, 2:].tolist() == [439, 788, 1410, 19750]


def test_rounding_half_up_and_clamping():
    env = _abstract_env()
    env.step_concrete(torch.tensor([CHECK_CALL]), torch.tensor([0]))
    env.step_concrete(torch.tensor([CHECK_CALL]), torch.tensor([0]))  # flop, pot 200
    env.step_concrete(torch.tensor([RAISE]), torch.tensor([125]))
    # pot + to_call = 325 + 125 = 450; 0.33 * 450 = 148.5 -> 149
    amounts = env.action_amounts()[0, 2:].tolist()
    assert amounts[0] == 125 + 149
    env = _abstract_env()
    env.step_concrete(torch.tensor([CHECK_CALL]), torch.tensor([0]))
    env.step_concrete(torch.tensor([CHECK_CALL]), torch.tensor([0]))
    env.step_concrete(torch.tensor([RAISE]), torch.tensor([1000]))
    # 0.33 pot raise would be 1000 + 726 < min raise 2000 -> clamped to the minimum
    assert env.action_amounts()[0, 2:].tolist() == [2000, 2650, 4300, 19900]


def test_sizes_above_stack_become_allin_only():
    env = VecNLHE(1, GameConfig(stacks=[1200, 20000]), "cpu", seed=0)
    env.reset(button=0)
    env.step(torch.tensor([1]))
    env.step(torch.tensor([1]))  # flop, pot 200, p0 has 1100 behind
    env.step(torch.tensor([4]))  # p1 bets 1.5 pot = 300
    # p0: 0.33 -> 564 clamped to the 600 min raise; 0.75 -> 900; 1.5 -> 1500 > stack -> all-in
    assert env.action_amounts()[0, 2:].tolist() == [600, 900, 1100, 1100]
    assert env.legal_mask()[0].tolist() == [True, True, True, True, False, True]
