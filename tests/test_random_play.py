"""Property tests: random legal play conserves chips and respects invariants."""

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from pokerbot.reference import Action, GameConfig, GameState

BOARD_LEN = {0: 0, 1: 3, 2: 4, 3: 5}


def random_action(state, rng):
    la = state.legal_actions()
    opts = ["c", "c", "c"]
    if la.can_fold:
        opts.append("f")
    if la.min_raise_to > 0:
        opts += ["r", "r"] + (["a"] if rng.random() < 0.3 else [])
    o = opts[rng.integers(len(opts))]
    if o == "c":
        return Action.check_call()
    if o == "f":
        return Action.fold()
    if o == "a" or la.max_raise_to <= la.min_raise_to:
        return Action.raise_to(la.max_raise_to)
    # mostly small raises so hands last several streets
    hi = min(la.max_raise_to, la.min_raise_to * 3)
    return Action.raise_to(int(rng.integers(la.min_raise_to, hi + 1)))


def check_invariants(s, cfg):
    total = sum(cfg.stacks)
    assert sum(s.stacks) + s.pot == total
    assert all(x >= 0 for x in s.stacks)
    assert len(s.board) == BOARD_LEN[s.street]
    for i in range(cfg.num_players):
        assert s.all_in[i] == (s.stacks[i] == 0)
    if not s.is_terminal:
        p = s.current_player
        assert 0 <= p < cfg.num_players
        assert not s.folded[p] and not s.all_in[p]
        la = s.legal_actions()
        assert la.can_fold != la.can_check
        assert la.call_amount <= s.stacks[p]
        if la.min_raise_to:
            assert la.max_raise_to == s.street_bets[p] + s.stacks[p]
            assert la.max_raise_to > max(s.street_bets)
        else:
            assert la.max_raise_to == 0
    else:
        assert s.current_player == -1


def play_random_hand(cfg, button, deck, rng):
    s = GameState.new_hand(cfg, button, deck)
    check_invariants(s, cfg)
    steps = 0
    while not s.is_terminal:
        a = random_action(s, rng)
        before = s.clone()
        s.apply(a)
        # clone/child agree with apply
        assert before.child(a).public_key() == s.public_key()
        check_invariants(s, cfg)
        steps += 1
        assert steps < 500
    pay = s.payoffs()
    assert sum(pay) == 0
    for i in range(cfg.num_players):
        assert pay[i] >= -cfg.stacks[i]
        assert pay[i] <= sum(cfg.stacks) - cfg.stacks[i]
        if s.folded[i]:
            assert pay[i] <= 0
    live = [i for i in range(cfg.num_players) if not s.folded[i]]
    if len(live) > 1:
        assert len(s.board) == 5
    return s


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    n=st.integers(2, 9),
    seed=st.integers(0, 2**31 - 1),
    bb=st.sampled_from([2, 10, 100]),
    ante=st.integers(0, 3),
    deep=st.booleans(),
)
def test_random_play_conserves_chips(n, seed, bb, ante, deep):
    rng = np.random.default_rng(seed)
    hi = 300 * bb if deep else 8 * bb
    stacks = [int(x) for x in rng.integers(1, hi, size=n)]
    cfg = GameConfig(num_players=n, stacks=stacks, small_blind=bb // 2, big_blind=bb, ante=ante)
    for _ in range(5):
        play_random_hand(cfg, int(rng.integers(n)), rng.permutation(52), rng)


def test_many_heads_up_hands():
    rng = np.random.default_rng(123)
    cfg = GameConfig(num_players=2, stacks=[20000, 20000])
    streets = np.zeros(4, dtype=int)
    for h in range(1500):
        s = play_random_hand(cfg, h % 2, rng.permutation(52), rng)
        streets[s.street] += 1
    assert streets.min() > 0  # every street reached at least once
