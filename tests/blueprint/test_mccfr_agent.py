"""BlueprintAgent: legal play through the match runner, off-tree mapping."""

import numpy as np
import poker_engine as pe
import pytest

import pokerbot.reference as ref
from pokerbot.agents import Agent, RandomAgent, make_agent
from pokerbot.blueprint.mccfr import BlueprintAgent
from pokerbot.eval import run_match
from pokerbot.eval.cli import main as play_match_main


def test_agent_protocol_and_registry(tiny_blueprint):
    a = make_agent(f"blueprint:{tiny_blueprint['path']}")
    assert isinstance(a, Agent) and isinstance(a, BlueprintAgent)
    assert a.name == "blueprint"
    assert a.default_game["stacks"] == [1000, 1000]


@pytest.mark.parametrize("stack", [1000, 20000])
def test_plays_legal_actions_vs_random(tiny_blueprint, stack):
    # 20,000-chip stacks are off the 10bb training game: every large bet is
    # mapped with the pseudo-harmonic mapping and sized against the real pot.
    agent = BlueprintAgent(tiny_blueprint["path"])
    config = pe.GameConfig(stacks=[stack, stack])
    res = run_match([agent, RandomAgent()], config, 500, seed=3, engine=pe, on_illegal="raise")
    assert res.hands == 500
    assert agent.counters["decisions"] > 500
    if stack == 20000:
        assert agent.counters["translated"] > 0


def test_plays_on_reference_engine(tiny_blueprint):
    agent = BlueprintAgent(tiny_blueprint["path"])
    config = ref.GameConfig(num_players=2, stacks=[1000, 1000])
    res = run_match([RandomAgent(), agent], config, 100, seed=4, engine=ref, on_illegal="raise")
    assert res.hands == 100


def test_follows_on_tree_actions(tiny_blueprint):
    # Against a copy of itself every action is on the tree: no fallbacks.
    a = BlueprintAgent(tiny_blueprint["path"], mapping="deterministic")
    b = BlueprintAgent(tiny_blueprint["path"], name="blueprint_b")
    run_match([a, b], pe.GameConfig(stacks=[1000, 1000]), 200, seed=5, engine=pe)
    for x in (a, b):
        assert x.counters["fallback"] == 0
        assert x.counters["off_tree"] == 0
        assert x.counters["translated"] == 0
        assert x.counters["missing_infoset"] < 0.05 * x.counters["decisions"]


def test_greedy_is_deterministic(tiny_blueprint):
    a = BlueprintAgent(tiny_blueprint["path"], greedy=True)
    state = pe.GameState.new_hand(pe.GameConfig(stacks=[1000, 1000]), 0, list(range(52)))
    acts = set()
    for i in range(5):
        a.new_hand(0, state.config)
        acts.add(a.act(state, 0, np.random.default_rng(i)))
    assert len(acts) == 1


def test_play_match_cli(tiny_blueprint, capsys):
    args = ["--b", "random", "--hands", "40", "--duplicate", "--seed", "0", "--engine", "rust"]
    rc = play_match_main(["--a", f"blueprint:{tiny_blueprint['path']}", *args])
    assert rc == 0
    out = capsys.readouterr().out
    assert "blueprint vs random (duplicate)" in out
