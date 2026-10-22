"""AIVAT's action terms for a recording bot (pokerbot/eval/aivat.py)."""

import numpy as np
import pytest

from pokerbot.agents import make_agent
from pokerbot.config import game_config
from pokerbot.engine_select import get_engine
from pokerbot.eval.aivat import action_correction, aivat_adjusted, decision_correction
from pokerbot.eval.luck import luck_adjusted
from pokerbot.eval.match import play_hand
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search.combos import combo_index

TINY = {
    "device": "cpu",
    "time_budget": 0.02,
    "min_iterations": 2,
    "fallback_on_error": False,
    "tree": {"max_nodes": 1500, "chance_cards": 3, "max_raises": 2},
    "solver": {"iterations": 4, "max_runouts": 6},
    "leaf": {"rollouts": 1, "max_total_rollouts": 64},
    "gadget": {"rollouts": 8},
}


def _decision(rng, actions=4, zero=None):
    strategy = rng.random((1326, actions))
    if zero is not None:
        strategy[:, zero] = 0
    strategy /= strategy.sum(1, keepdims=True)
    return {
        "taken": 0,
        "strategy": strategy.astype(np.float32),
        "values": rng.normal(0, 500, (1326, actions)).astype(np.float32),
        "values_vs": rng.normal(0, 2000, (1326, actions)).astype(np.float32),
    }


@pytest.mark.parametrize("values", ["pair", "range"])
def test_the_action_term_has_zero_mean_over_the_bots_draws(values):
    """At a fixed decision, drawing the action from the bot's strategy row many
    times averages the correction to zero."""
    rng = np.random.default_rng(0)
    d = _decision(rng)
    c, o = 17, 900
    p = d["strategy"][c].astype(np.float64)
    p /= p.sum()
    terms = []
    for a in rng.choice(len(p), size=20000, p=p):
        terms.append(decision_correction({**d, "taken": int(a)}, c, o, values))
    terms = np.array(terms)
    se = terms.std() / np.sqrt(len(terms))
    assert terms.std() > 10  # not trivially zero
    assert abs(terms.mean()) < 4 * se
    # exact: the expectation over the draw is zero
    exact = sum(p[a] * decision_correction({**d, "taken": a}, c, o, values) for a in range(4))
    assert abs(exact) < 1e-6 * terms.std()


def test_pair_values_are_used_and_fall_back_to_range_values():
    rng = np.random.default_rng(1)
    d = _decision(rng, actions=3)
    c, o = 5, 700
    p = d["strategy"][c].astype(np.float64)
    p /= p.sum()
    for row, key in ((o, "values_vs"), (c, "values")):
        v = d[key][row].astype(np.float64)
        want = float((p * v).sum() - v[0])
        got_o = o if key == "values_vs" else None  # no opponent hand: range values
        assert decision_correction(d, c, got_o) == pytest.approx(want, rel=1e-6)
    assert decision_correction(d, c, o, "range") == pytest.approx(
        float((p * d["values"][c]).sum() - d["values"][c][0]), rel=1e-6
    )
    nan_row = {**d, "values_vs": d["values_vs"].copy()}
    nan_row["values_vs"][o] = np.nan  # an opponent combo blocked by ours
    assert decision_correction(nan_row, c, o) == decision_correction(d, c, o, "range")
    no_values = {**d, "values": None, "values_vs": None}
    assert decision_correction(no_values, c, o) == 0.0
    with pytest.raises(ValueError):
        decision_correction(d, c, o, "both")


def test_zero_probability_actions_do_not_enter():
    rng = np.random.default_rng(2)
    d = _decision(rng, actions=3, zero=2)
    d["values"][:, 2] = np.nan
    d["values_vs"][:, 2] = np.nan
    c, o = 40, 1000
    p = d["strategy"][c].astype(np.float64)
    p /= p.sum()
    v = d["values_vs"][o].astype(np.float64)
    want = p[0] * v[0] + p[1] * v[1] - v[0]
    assert decision_correction(d, c, o) == pytest.approx(want, rel=1e-6)


def test_aivat_on_a_recording_search_bot():
    """Rows stay zero-sum, AIVAT is luck-adjusted plus the bot's action terms,
    and preflop (blueprint) decisions contribute nothing."""
    engine = get_engine()
    config = game_config({"stacks": [2000, 2000]}, engine)
    bot = SearchAgent(UniformBlueprint(), TINY, record=True)
    hands, decisions = [], []
    for h in range(8):
        deck = np.random.default_rng(h).permutation(52).tolist()
        state = play_hand(
            config, [make_agent("always_call"), bot], h % 2, deck, np.random.default_rng(h), engine
        )
        hands.append((h % 2, deck, state))
        decisions.append(bot.decisions)
    chance, aivat = aivat_adjusted(hands, decisions, 1, config, engine, preflop_samples=64)
    np.testing.assert_allclose(chance, luck_adjusted(hands, config, engine, 64))
    np.testing.assert_allclose(aivat[:, 0], -aivat[:, 1])
    searched = 0
    for (_, _, state), recs, ch, av in zip(hands, decisions, chance, aivat, strict=True):
        corr = action_correction(recs, state.hole_cards(1), state.hole_cards(0))
        assert av[1] == pytest.approx(ch[1] + corr)
        c = combo_index(*state.hole_cards(1))
        o = combo_index(*state.hole_cards(0))
        for d in recs:
            if d["street"] == 0:
                assert d["values"] is None and d["values_vs"] is None
                assert decision_correction(d, c, o) == 0.0
            else:
                searched += 1
                assert np.isfinite(d["values_vs"][o]).all()
    assert searched
