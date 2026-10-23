"""The search as a range policy (``SearchPolicyAgent``) and LBR against it."""

from __future__ import annotations

import numpy as np
import pytest

from pokerbot.agents.policy import as_policy_agent
from pokerbot.engine_select import get_engine
from pokerbot.eval.lbr import LBRAgent, run_lbr
from pokerbot.eval.masking import MaskedState
from pokerbot.eval.match import play_hand
from pokerbot.search import SearchAgent, UniformBlueprint
from pokerbot.search.combos import combo_index
from pokerbot.search.policy import SearchPolicyAgent, action_probs, public_key
from tests.search.test_agent import TINY

# a fixed iteration count with no time cut, so equal inputs give equal solves
DETERMINISTIC = dict(TINY, time_budget=60.0, min_iterations=4)


def _config(stack: int = 1000):
    engine = get_engine()
    return engine.GameConfig(
        num_players=2, stacks=[stack, stack], small_blind=50, big_blind=100, ante=0
    )


def test_search_agent_becomes_its_range_policy():
    agent = SearchAgent(UniformBlueprint(), TINY)
    pol = as_policy_agent(agent, hypothetical_budget=0.5)
    assert isinstance(pol, SearchPolicyAgent)
    assert pol.agent is agent and agent.record
    assert pol.spec is agent.blueprint.spec
    assert pol._hypo_cfg.budget(1) == pytest.approx(0.5 * agent.cfg.budget(1))


@pytest.mark.parametrize("raises,preflop", [("fcpa", "full"), ("spec", "fc"), ("fcpa", "call")])
def test_lbr_plays_legal_hands_against_the_search(raises, preflop):
    agent = SearchAgent(UniformBlueprint(), TINY)
    res = run_lbr(agent, _config(), hands=6, seed=1, raises=raises, preflop=preflop)
    assert res.hands == 6
    assert np.isfinite(res.adjusted).all()
    assert res.range_resets == 0
    real = sum(n for k, (n, _s) in res.opponent_cost.items() if k.startswith("real/"))
    assert real > 0
    if preflop == "full":
        # LBR's preflop raises need the bot's response: blueprint (table) lookups
        assert res.opponent_cost.get("hypothetical/preflop", [0])[0] > 0
    assert any(k.startswith("hypothetical/") for k in res.opponent_cost)


def test_range_strategy_is_what_the_agent_played():
    """At every real decision the adapter answers with the strategy the agent
    recorded there, and the bot's own hand gives the action it took positive
    probability; LBR's range never resets."""
    engine = get_engine()
    config = _config(1500)
    agent = SearchAgent(UniformBlueprint(), TINY)
    pol = SearchPolicyAgent(agent)
    lbr = LBRAgent(pol, raises="fcpa", preflop="fc", engine=engine)
    rng_deck = np.random.default_rng(0)
    checked = 0
    for h in range(8):
        deck = [int(c) for c in rng_deck.permutation(52)]
        state = play_hand(config, [lbr, pol], h % 2, deck, np.random.default_rng(h), engine)
        hole = list(state.hole_cards(1))
        row = combo_index(hole[0], hole[1])
        bot = [k for k, (_s, p, _a) in enumerate(state.history) if p == 1]
        assert len(bot) == len(agent.decisions)
        replay = engine.GameState.new_hand(config, h % 2, deck)
        for k, (_s, p, a) in enumerate(state.history):
            if p == 1:
                rec = agent.decisions[bot.index(k)]
                acts, strat = pol.range_strategy(replay, 1)
                assert acts == [tuple(x) for x in rec["actions"]]
                s = np.nan_to_num(rec["strategy"].astype(np.float64))
                ok = s.sum(1) > 0
                np.testing.assert_allclose(strat[ok], s[ok] / s[ok].sum(1, keepdims=True))
                assert action_probs(acts, strat, int(a.kind), int(a.amount))[row] > 0
                checked += 1
            replay.apply(a)
    assert checked > 8
    assert lbr.range_resets == 0
    assert agent.stats and not any(s.get("fallback") for s in agent.stats)


def _reach_turn(engine, config, pol, seed):
    """The bot (seat 1, button) limps and seat 0 checks; on the flop seat 0
    checks, the bot searches for real (filling its per-hand caches) and seat 0
    calls any bet. Returns the turn state with seat 0 to act, or ``None``."""
    deck = [int(c) for c in np.random.default_rng(seed).permutation(52)]
    state = engine.GameState.new_hand(config, 1, deck)
    pol.new_hand(1, config)
    pol._hole = [int(c) for c in state.hole_cards(1)]
    while state.street == 0:  # the bot's preflop would be the blueprint's
        state.apply(engine.Action.check_call())
    state.apply(engine.Action.check_call())
    state.apply(pol.act(MaskedState(state, 1, config), 1, np.random.default_rng(seed)))
    if not state.is_terminal and state.street == 1:
        state.apply(engine.Action.check_call())
    ok = not state.is_terminal and state.street == 2 and state.legal_actions().min_raise_to > 0
    return state if ok else None


def test_hypothetical_search_matches_the_real_one_and_leaves_caches_alone():
    engine = get_engine()
    config = _config(2000)
    agent = SearchAgent(UniformBlueprint(), DETERMINISTIC)
    pol = SearchPolicyAgent(agent)
    state = next(s for seed in range(20) if (s := _reach_turn(engine, config, pol, seed)))
    assert state.current_player == 0 and agent.decisions
    before = (
        set(agent.cache.entries),
        set(agent._roots),
        set(agent._played),
        len(agent.decisions),
        len(agent.stats),
    )
    raise_to = int(state.legal_actions().min_raise_to)
    child = state.child(engine.Action.raise_to(raise_to))
    acts, hypo = pol.range_strategy(child, 1)
    assert pol.hypo_stats and pol.hypo_stats[-1]["street"] == 2
    assert np.ptp(hypo[:, 0]) > 0  # a real solve, not a uniform fallback
    after = (
        set(agent.cache.entries),
        set(agent._roots),
        set(agent._played),
        len(agent.decisions),
        len(agent.stats),
    )
    assert before == after
    assert pol.range_strategy(child, 1)[1] is hypo  # cached by public state
    # LBR really raises: the bot's own search there solves the same game
    state.apply(engine.Action.raise_to(raise_to))
    pol.act(MaskedState(state, 1, config), 1, np.random.default_rng(1))
    real_acts, real = pol._real[public_key(state)]
    assert real_acts == acts
    np.testing.assert_allclose(real, hypo, atol=1e-6)
