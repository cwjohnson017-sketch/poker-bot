"""Luck-adjusted match results (all-in EV + street control variates)."""

import numpy as np
import pytest
import torch

from pokerbot.agents import make_agent
from pokerbot.config import game_config
from pokerbot.engine_select import get_engine
from pokerbot.env.cards import make_generator
from pokerbot.env.equity import street_equities
from pokerbot.eval.luck import luck_adjusted, replay_streets
from pokerbot.eval.match import play_hand, run_duplicate_match, run_match


def _play(a, b, n, stacks=10000, seed=0):
    engine = get_engine()
    config = game_config({"stacks": [stacks, stacks]}, engine)
    rng = np.random.default_rng(seed)
    records = []
    for h in range(n):
        deck = rng.permutation(52)
        state = play_hand(config, [a, b], h % 2, deck, np.random.default_rng(h), engine)
        records.append((h % 2, deck, state))
    return records, config, engine


def _preflop_equity(records, samples, seed=0):
    d = torch.tensor(np.array([r[1][:9] for r in records]), dtype=torch.long)
    E = street_equities(d[:, 0:2], d[:, 2:4], d[:, 4:9], samples, make_generator(seed))
    return E[:, 0].double().numpy()


def test_checked_down_hands_score_their_preflop_equity():
    # always_call vs always_call checks every hand down with one big blind in
    # per player: the street corrections telescope to 100 * (2 E_preflop - 1)
    records, config, engine = _play(make_agent("always_call"), make_agent("always_call"), 12)
    for r in records:
        transitions, allin, stake = replay_streets(r, config, engine)
        assert [(o, n) for o, n, _ in transitions] == [(0, 1), (1, 2), (2, 3)]
        assert all(c == 100 for _, _, c in transitions) and allin is None and stake == 100
    adj = luck_adjusted(records, config, engine, preflop_samples=256, seed=3)
    np.testing.assert_allclose(adj[:, 1], -adj[:, 0])
    expected = 100 * (2 * _preflop_equity(records, 256, 3) - 1)
    np.testing.assert_allclose(adj[:, 0], expected, atol=1e-6)


def test_preflop_all_ins_score_their_equity():
    records, config, engine = _play(make_agent("fixed:allin"), make_agent("always_call"), 40)
    adj = luck_adjusted(records, config, engine, preflop_samples=256, seed=1)
    raw = np.array([r[2].payoffs()[0] for r in records], dtype=float)
    assert set(np.abs(raw)) <= {0.0, 10000.0}
    for r in records:
        assert replay_streets(r, config, engine)[1:] == (0, 10000)
    np.testing.assert_allclose(adj[:, 0], 10000 * (2 * _preflop_equity(records, 256, 1) - 1))
    assert adj[:, 0].std() < 0.7 * raw.std()


def test_folds_before_the_flop_are_unchanged():
    records, config, engine = _play(make_agent("random"), make_agent("random"), 60, seed=2)
    adj = luck_adjusted(records, config, engine, preflop_samples=64)
    for i, r in enumerate(records):
        transitions, allin, _ = replay_streets(r, config, engine)
        if not transitions and allin is None:
            assert adj[i, 0] == r[2].payoffs()[0]


def test_duplicate_mirror_match_cancels_exactly_and_adjusted_is_reported():
    engine = get_engine()
    config = game_config({"stacks": [10000, 10000]}, engine)
    res = run_duplicate_match(
        make_agent("always_call"), make_agent("fixed:allin"), config, 30, seed=0, engine=engine
    )
    assert res.adjusted is None and res.adjusted_stats is None
    res = run_duplicate_match(
        make_agent("fixed:allin"),
        make_agent("always_call"),
        config,
        30,
        seed=0,
        engine=engine,
        luck_adjust=True,
    )
    # both seatings of a deal hold the same cards and play the same all-in:
    # raw and adjusted results cancel deal by deal
    assert np.all(res.samples == 0) and np.allclose(res.adjusted, 0.0)
    assert "luck-adjusted" in res.summary()


def test_adjusted_win_rate_agrees_with_raw_and_is_tighter():
    engine = get_engine()
    config = game_config({"stacks": [10000, 10000]}, engine)
    res = run_match(
        [make_agent("always_call"), make_agent("equity")],
        config,
        300,
        seed=5,
        engine=engine,
        luck_adjust=True,
    )
    raw, adj = res.samples.astype(float), res.adjusted
    se = np.sqrt((raw - adj).var() / len(raw))
    assert abs(raw.mean() - adj.mean()) < 4 * se + 1e-9  # unbiased
    assert adj.std() < raw.std()


def test_luck_adjust_is_heads_up_only():
    engine = get_engine()
    config = game_config({"num_players": 3, "stacks": [1000] * 3}, engine)
    with pytest.raises(ValueError):
        run_match([make_agent("always_call")] * 3, config, 2, luck_adjust=True, engine=engine)
