import io

import numpy as np
import pytest
from helpers import make_deck

from pokerbot.agents import (
    AlwaysCallAgent,
    AlwaysRaiseAgent,
    BaseAgent,
    EquityThresholdAgent,
    RandomAgent,
)
from pokerbot.eval import (
    HiddenInformationError,
    IllegalActionError,
    MaskedState,
    format_hand,
    play_hand,
    run_duplicate_match,
    run_match,
)
from pokerbot.reference import Action, GameConfig, GameState, cards_from_str

pytestmark = pytest.mark.usefixtures("reference_engine")


def hu_config():
    return GameConfig(num_players=2, stacks=[20000, 20000])


class PeekingAgent(BaseAgent):
    """Records everything it can see about the other seats."""

    name = "peeker"

    def __init__(self):
        super().__init__()
        self.seen_other = []
        self.seen_own = []
        self.end_views = []

    def act(self, state, seat, rng):
        for p in range(state.num_players):
            if p == seat:
                self.seen_own.append(state.hole_cards(p))
            else:
                self.seen_other.append(state.hole_cards(p))
        with pytest.raises(AttributeError):
            state._deck  # noqa: B018
        with pytest.raises(AttributeError):
            state.deck  # noqa: B018
        with pytest.raises(HiddenInformationError):
            state.infoset_key(1 - seat)
        with pytest.raises(TypeError):
            state.apply(Action.check_call())
        return Action.check_call()

    def observe_end(self, state):
        self.end_views.append([state.hole_cards(p) for p in range(state.num_players)])


def test_masking_hides_other_hole_cards():
    peeker = PeekingAgent()
    deck = make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")
    final = play_hand(hu_config(), [peeker, AlwaysCallAgent()], 0, deck, np.random.default_rng(0))
    assert final.is_terminal
    assert peeker.seen_other and all(c == [] for c in peeker.seen_other)
    assert all(c == cards_from_str("AsKs") for c in peeker.seen_own)
    # showdown reveals both hands at the end
    assert peeker.end_views == [[cards_from_str("AsKs"), cards_from_str("2c7d")]]


def test_masking_no_reveal_when_hand_folded():
    class Folder(BaseAgent):
        def act(self, state, seat, rng):
            return Action.fold()

    peeker = PeekingAgent()
    deck = make_deck(["AsKs", "2c7d"])
    play_hand(hu_config(), [Folder(), peeker], 0, deck, np.random.default_rng(0))
    assert peeker.end_views == [[[], cards_from_str("2c7d")]]


def test_masked_clone_resamples_hidden_cards():
    cfg = hu_config()
    deck = make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")
    s = GameState.new_hand(cfg, 0, deck)
    s.apply(Action.check_call())
    s.apply(Action.check_call())  # flop dealt
    seen_opp, seen_turn = set(), set()
    for seed in range(40):
        view = MaskedState(s, 1, cfg, np.random.default_rng(seed))
        c = view.clone()
        assert c.hole_cards(1) == cards_from_str("2c7d")
        assert c.board == cards_from_str("QhJhTh")
        assert c.history == s.history and c.pot == s.pot
        opp = c.hole_cards(0)
        assert not set(opp) & set(cards_from_str("2c7dQhJhTh"))
        seen_opp.add(tuple(opp))
        c.apply(Action.check_call())
        c.apply(Action.check_call())
        seen_turn.add(c.board[3])
    # hidden cards really are re-sampled, not copied from the live deck
    assert len(seen_opp) > 30 and len(seen_turn) > 10
    assert s.street == 1  # the live state is untouched


def test_duplicate_always_call_vs_always_raise_is_zero_sum():
    """Every hand goes all-in to showdown, so the result depends only on the
    cards and the duplicate totals cancel exactly."""
    res = run_duplicate_match(AlwaysCallAgent(), AlwaysRaiseAgent(), hu_config(), 40, seed=3)
    assert res.hands == 80
    assert res.total_a + res.total_b == 0
    assert res.total_a == 0
    assert (res.seat_payoffs.sum(axis=2) == 0).all()
    # pot-sized raise each street, always called: 300, 900, 2700, 8100 per player,
    # then showdown (or a split pot)
    assert set(np.unique(abs(res.seat_payoffs))) <= {0, 8100}
    assert abs(res.seat_payoffs).max() == 8100
    assert res.mbb_per_hand == 0.0


def test_duplicate_seats_swap_with_same_cards():
    hist = io.StringIO()
    run_duplicate_match(AlwaysCallAgent(), RandomAgent(), hu_config(), 3, seed=1, history=hist)
    blocks = [b for b in hist.getvalue().strip().split("\n\n") if b]
    assert len(blocks) == 6
    for d in range(3):
        a, b = blocks[2 * d].splitlines(), blocks[2 * d + 1].splitlines()
        # same cards per seat, swapped agent names
        assert a[1].split("[")[1] == b[1].split("[")[1]
        assert a[2].split("[")[1] == b[2].split("[")[1]
        assert "always_call" in a[1] and "always_call" in b[2]


def test_plain_match_zero_sum_and_button_rotation():
    res = run_match([RandomAgent(), AlwaysCallAgent()], hu_config(), 60, seed=5)
    assert res.hands == 60
    assert (res.seat_payoffs.sum(axis=1) == 0).all()
    assert res.total_a == -res.total_b
    lo, hi = res.ci
    assert lo <= res.mbb_per_hand <= hi


def test_multiway_match():
    cfg = GameConfig(num_players=4, stacks=[5000] * 4)
    agents = [
        RandomAgent(),
        AlwaysCallAgent(),
        AlwaysRaiseAgent(),
        EquityThresholdAgent(samples=50),
    ]
    res = run_match(agents, cfg, 12, seed=2)
    assert (res.seat_payoffs.sum(axis=1) == 0).all()


def test_match_is_reproducible():
    cfg = hu_config()
    r1 = run_match([RandomAgent(), EquityThresholdAgent(samples=50)], cfg, 30, seed=9)
    r2 = run_match([RandomAgent(), EquityThresholdAgent(samples=50)], cfg, 30, seed=9)
    assert (r1.samples == r2.samples).all()


def test_equity_agent_beats_always_call():
    res = run_duplicate_match(
        EquityThresholdAgent(samples=100), AlwaysCallAgent(), hu_config(), 150, seed=0
    )
    assert res.mbb_per_hand > 0
    assert res.stats.ci_low > 0


def test_illegal_action_policy():
    class Bad(BaseAgent):
        def act(self, state, seat, rng):
            return Action.raise_to(1)

    deck = make_deck(["AsKs", "2c7d"])
    with pytest.raises(IllegalActionError):
        play_hand(hu_config(), [Bad(), AlwaysCallAgent()], 0, deck, np.random.default_rng(0))
    s = play_hand(
        hu_config(),
        [Bad(), AlwaysCallAgent()],
        0,
        deck,
        np.random.default_rng(0),
        on_illegal="fold",
    )
    assert s.is_terminal and s.payoffs() == [-50, 50]


def test_hand_history_format():
    deck = make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")
    s = GameState.new_hand(hu_config(), 0, deck)
    for a in (Action.raise_to(300), Action.check_call(), Action.check_call(), Action.raise_to(400)):
        s.apply(a)
    s.apply(Action.fold())
    text = format_hand(s, hu_config(), 7, ["alice", "bob"])
    lines = text.splitlines()
    assert lines[0].startswith("# hand 7 | button 0")
    assert lines[1] == "seat 0 alice 20000 [As Ks] button"
    assert lines[2] == "seat 1 bob 20000 [2c 7d]"
    assert lines[3] == "preflop: 0:r300 1:c"
    assert lines[4] == "flop [Qh Jh Th]: 1:k 0:r400 1:f"
    assert lines[-1] == "result: 0:+300 1:-300"
