import numpy as np
import pytest

from pokerbot.agents import (
    Agent,
    AlwaysCallAgent,
    AlwaysRaiseAgent,
    EquityThresholdAgent,
    HumanCLIAgent,
    RandomAgent,
    make_agent,
    monte_carlo_equity,
)
from pokerbot.eval import MaskedState, play_hand
from pokerbot.reference import CHECK_CALL, FOLD, RAISE, GameConfig, GameState, cards_from_str
from tests.helpers import make_deck

pytestmark = pytest.mark.usefixtures("reference_engine")

CFG = GameConfig(num_players=2, stacks=[20000, 20000])


def view(state, seat):
    return MaskedState(state, seat, CFG, np.random.default_rng(0))


def test_registry_and_protocol():
    for name in ("always_call", "always_raise", "random", "equity", "human"):
        a = make_agent(name)
        assert isinstance(a, Agent)
    with pytest.raises(ValueError):
        make_agent("nope")


def test_always_raise_pot_sized_and_all_in_cap():
    s = GameState.new_hand(CFG, 0, make_deck(["AsKs", "2c7d"]))
    a = AlwaysRaiseAgent()
    act = a.act(view(s, 0), 0, np.random.default_rng(0))
    # pot 150, to call 50 -> raise to 100 + 200 = 300
    assert act.kind == RAISE and act.amount == 300
    s.apply(act)
    act = a.act(view(s, 1), 1, np.random.default_rng(0))
    # pot 400, to call 200 -> raise to 300 + 600 = 900
    assert act.amount == 900
    short = GameConfig(num_players=2, stacks=[20000, 700])
    s = GameState.new_hand(short, 0, make_deck(["AsKs", "2c7d"]))
    s.apply(act.__class__.raise_to(300))
    act = a.act(MaskedState(s, 1, short), 1, np.random.default_rng(0))
    assert act.amount == 700  # capped at all-in


def test_always_raise_calls_when_no_raise_possible():
    s = GameState.new_hand(CFG, 0, make_deck(["AsKs", "2c7d"]))
    s.apply(AlwaysRaiseAgent().raise_to(20000))
    act = AlwaysRaiseAgent().act(view(s, 1), 1, np.random.default_rng(0))
    assert act.kind == CHECK_CALL


def test_random_agent_always_legal():
    rng = np.random.default_rng(0)
    agents = [RandomAgent(), RandomAgent()]
    for h in range(100):
        s = play_hand(CFG, agents, h % 2, rng.permutation(52), rng)
        assert s.is_terminal


def test_monte_carlo_equity_sane():
    rng = np.random.default_rng(0)
    aa = monte_carlo_equity(cards_from_str("AsAh"), [], 1, 2000, rng)
    assert 0.80 < aa < 0.89  # ~0.85
    weak = monte_carlo_equity(cards_from_str("7c2d"), [], 1, 2000, rng)
    assert 0.28 < weak < 0.40  # ~0.35
    nuts = monte_carlo_equity(cards_from_str("AsKs"), cards_from_str("QsJsTs"), 1, 200, rng)
    assert nuts == 1.0
    multi = monte_carlo_equity(cards_from_str("AsAh"), [], 4, 2000, rng)
    assert 0.48 < multi < 0.62  # ~0.56


def test_equity_agent_decisions():
    rng = np.random.default_rng(0)
    agent = EquityThresholdAgent(samples=300)
    s = GameState.new_hand(CFG, 0, make_deck(["AsAh", "7c2d"]))
    assert agent.act(view(s, 0), 0, rng).kind == RAISE
    s = GameState.new_hand(CFG, 1, make_deck(["AsAh", "7c2d"]))
    assert agent.act(view(s, 1), 1, rng).kind == FOLD
    # can check for free with a weak hand: checks rather than folds
    s2 = GameState.new_hand(CFG, 0, make_deck(["AsAh", "7c2d"]))
    s2.apply(AlwaysCallAgent().check_call())
    assert agent.act(view(s2, 1), 1, rng).kind == CHECK_CALL


def test_human_cli_agent_scripted():
    inputs = iter(["?", "x", "r 50", "r 250"])
    out = []
    agent = HumanCLIAgent(input_fn=lambda prompt: next(inputs), output_fn=out.append)
    s = GameState.new_hand(CFG, 0, make_deck(["AsAh", "7c2d"]))
    agent.new_hand(0, CFG)
    act = agent.act(view(s, 0), 0, np.random.default_rng(0))
    assert act.kind == RAISE and act.amount == 250
    text = "\n".join(out)
    assert "As Ah" in text and "7c" not in text
    assert "illegal or unknown" in text

    inputs = iter(["f"])
    agent = HumanCLIAgent(input_fn=lambda prompt: next(inputs), output_fn=out.append)
    assert agent.act(view(s, 0), 0, np.random.default_rng(0)).kind == FOLD
    inputs = iter(["a"])
    agent = HumanCLIAgent(input_fn=lambda prompt: next(inputs), output_fn=out.append)
    assert agent.act(view(s, 0), 0, np.random.default_rng(0)).amount == 20000
