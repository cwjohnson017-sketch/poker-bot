"""SearchAgent end to end: legal play through the match runner (masked states)."""

from __future__ import annotations

import numpy as np

from pokerbot.agents import RandomAgent, make_agent
from pokerbot.engine_select import get_engine
from pokerbot.eval.match import run_match
from pokerbot.search import SearchAgent, UniformBlueprint

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


def test_search_agent_plays_legal_hands_vs_random():
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100, ante=0
    )
    agent = SearchAgent(UniformBlueprint(), TINY)
    res = run_match([agent, RandomAgent()], config, num_hands=100, seed=3)
    assert res.hands == 100
    assert int(res.seat_payoffs.sum()) == 0
    postflop = [s for s in agent.stats if "nodes" in s]
    assert postflop, "the agent never searched"
    assert not any(s.get("fallback") for s in agent.stats)
    # a later decision in the same hand should have reused a cached street root
    assert all(s["nodes"] <= 1500 for s in postflop)


def test_search_agent_unsafe_and_cached_roots():
    engine = get_engine()
    config = engine.GameConfig(
        num_players=2, stacks=[1500, 1500], small_blind=50, big_blind=100, ante=0
    )
    cfg = dict(TINY, gadget={"safe": False, "rollouts": 8})
    agent = SearchAgent(UniformBlueprint(), cfg)
    from pokerbot.agents import AlwaysCallAgent

    run_match([agent, AlwaysCallAgent()], config, num_hands=20, seed=5)
    assert any(s.get("cached_root") for s in agent.stats if "nodes" in s)


def test_make_agent_registers_search_prefix():
    a = make_agent("search:uniform", config=TINY)
    assert isinstance(a, SearchAgent)
    assert isinstance(a.blueprint, UniformBlueprint)
    assert a.name == "search:uniform"
    rng = np.random.default_rng(0)
    engine = get_engine()
    cfg = engine.GameConfig(num_players=2, stacks=[1000, 1000], small_blind=50, big_blind=100)
    s = engine.GameState.new_hand(cfg, 0, list(range(52)))
    a.new_hand(0, cfg)
    act = a.act(s, 0, rng)  # preflop: blueprint
    assert s.legal_actions().is_legal(act)
