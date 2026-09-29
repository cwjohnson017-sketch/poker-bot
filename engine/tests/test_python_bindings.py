"""End-to-end tests of the `poker_engine` Python bindings.

Build first:  cd engine && maturin develop --release
Run:          python -m pytest engine/tests/test_python_bindings.py
"""

import copy
import itertools
import pickle
from collections import Counter

import numpy as np
import pytest

import poker_engine as pe
from poker_engine import CHECK_CALL, FOLD, RAISE, Action, GameConfig, GameState

# ---------------------------------------------------------------------------
# Cards and evaluator
# ---------------------------------------------------------------------------


def cards(s):
    s = s.replace(" ", "")
    return [pe.card_from_str(s[i : i + 2]) for i in range(0, len(s), 2)]


def test_card_strings_roundtrip():
    for c in range(52):
        assert pe.card_from_str(pe.card_to_str(c)) == c
    assert pe.card_from_str("2c") == 0
    assert pe.card_from_str("As") == 51
    assert pe.card_from_str("Td") == 8 * 4 + 1
    assert pe.card_to_str(4 * 12 + 2) == "Ah"
    for bad in ["", "A", "1c", "Ax", "Asd"]:
        with pytest.raises(ValueError):
            pe.card_from_str(bad)
    for bad in [-1, 52]:
        with pytest.raises(ValueError):
            pe.card_to_str(bad)


def naive5(hand):
    """Slow independent 5-card ranking key (category, tiebreak tuple)."""
    ranks = sorted((c // 4 for c in hand), reverse=True)
    counts = Counter(ranks)
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    flush = len({c % 4 for c in hand}) == 1
    uniq = sorted(counts, reverse=True)
    top = None
    if len(uniq) == 5:
        if uniq[0] - uniq[4] == 4:
            top = uniq[0]
        elif uniq == [12, 3, 2, 1, 0]:
            top = 3
    pattern = [g[1] for g in groups]
    if top is not None and flush:
        return (8, (top,))
    if pattern == [4, 1]:
        cat = 7
    elif pattern == [3, 2]:
        cat = 6
    elif flush:
        cat = 5
    elif top is not None:
        return (4, (top,))
    elif pattern == [3, 1, 1]:
        cat = 3
    elif pattern == [2, 2, 1]:
        cat = 2
    elif pattern == [2, 1, 1, 1]:
        cat = 1
    else:
        cat = 0
    return (cat, tuple(g[0] for g in groups))


def naive7(hand):
    return max(naive5(h) for h in itertools.combinations(hand, 5))


def test_evaluator_orders_like_naive():
    rng = np.random.default_rng(1)
    hands = [rng.permutation(52)[:9].tolist() for _ in range(3000)]
    for d in hands:
        a = d[:2] + d[4:]
        b = d[2:4] + d[4:]
        ra, rb = pe.evaluate7(a), pe.evaluate7(b)
        na, nb = naive7(a), naive7(b)
        assert (ra > rb) == (na > nb) and (ra == rb) == (na == nb), (a, b)
        assert pe.hand_category(ra) == na[0]


def test_evaluate5_6_7_consistent():
    rng = np.random.default_rng(2)
    for _ in range(500):
        d = rng.permutation(52)[:7].tolist()
        r7 = pe.evaluate7(d)
        assert r7 == pe.evaluate(d)
        assert r7 == max(pe.evaluate6(list(h)) for h in itertools.combinations(d, 6))
        assert r7 == max(pe.evaluate5(list(h)) for h in itertools.combinations(d, 5))


def test_hand_categories():
    examples = {
        "Ah Kh Qh Jh Th 2c 3d": 8,
        "Ah Ad Ac As 2c 3d 5h": 7,
        "Ah Ad Ac Ks Kc 3d 5h": 6,
        "2h 5h 9h Jh Kh 3d 4c": 5,
        "Ah 2d 3c 4s 5h 9d Jc": 4,
        "7h 7d 7c 2s 9h Jd Kc": 3,
        "7h 7d 2c 2s 9h Jd Kc": 2,
        "7h 7d 3c 2s 9h Jd Kc": 1,
        "7h 5d 3c 2s 9h Jd Kc": 0,
    }
    for s, cat in examples.items():
        assert pe.hand_category(pe.evaluate7(cards(s))) == cat, s
    assert pe.HAND_CATEGORY_NAMES[8] == "straight flush"


def test_evaluate_batch():
    rng = np.random.default_rng(3)
    arr = np.stack([rng.permutation(52)[:7] for _ in range(2000)]).astype(np.uint8)
    out = pe.evaluate_batch(arr)
    assert out.dtype == np.int32 and out.shape == (2000,)
    assert out.tolist() == [pe.evaluate7(row.tolist()) for row in arr]
    # non-contiguous input
    wide = np.concatenate([arr, arr], axis=1)[:, ::2]
    assert pe.evaluate_batch(wide).tolist() == [pe.evaluate7(r.tolist()) for r in wide]
    assert pe.evaluate_batch(np.zeros((0, 7), np.uint8)).shape == (0,)
    with pytest.raises(ValueError):
        pe.evaluate_batch(arr[:, :6])
    with pytest.raises(ValueError):
        pe.evaluate_batch(arr.astype(np.int64))
    bad = arr.copy()
    bad[0, 1] = bad[0, 0]
    with pytest.raises(ValueError):
        pe.evaluate_batch(bad)


def test_evaluator_input_validation():
    with pytest.raises(ValueError):
        pe.evaluate7([0, 1, 2, 3, 4, 5])
    with pytest.raises(ValueError):
        pe.evaluate7([0, 1, 2, 3, 4, 5, 5])
    with pytest.raises(ValueError):
        pe.evaluate5([0, 1, 2, 3, 52])


# ---------------------------------------------------------------------------
# Actions and config
# ---------------------------------------------------------------------------


def test_action_constructors():
    assert (FOLD, CHECK_CALL, RAISE) == (0, 1, 2)
    assert Action.fold().kind == FOLD and Action.fold().amount == 0
    assert Action.check_call().kind == CHECK_CALL
    a = Action.raise_to(300)
    assert (a.kind, a.amount) == (RAISE, 300)
    assert a == Action(RAISE, 300) and hash(a) == hash(Action(RAISE, 300))
    assert a != Action.raise_to(301)
    assert pickle.loads(pickle.dumps(a)) == a
    with pytest.raises(ValueError):
        Action(CHECK_CALL, 5)
    with pytest.raises(ValueError):
        Action(7)
    with pytest.raises(ValueError):
        Action.raise_to(0)


def test_config():
    c = GameConfig()
    assert (c.num_players, c.stacks, c.small_blind, c.big_blind, c.ante) == (2, [20000, 20000], 50, 100, 0)
    c6 = GameConfig(num_players=6, stacks=[1000] * 6, small_blind=5, big_blind=10, ante=1)
    assert c6.num_players == 6
    assert GameConfig(stacks=[100, 200, 300]).num_players == 3
    assert pickle.loads(pickle.dumps(c6)) == c6
    with pytest.raises(ValueError):
        GameConfig(num_players=10)
    with pytest.raises(ValueError):
        GameConfig(num_players=3, stacks=[100, 100])
    c.stacks = [100, 0]
    with pytest.raises(ValueError):
        GameState.new_hand(c, 0, list(range(52)))


# ---------------------------------------------------------------------------
# Game rules through the bindings
# ---------------------------------------------------------------------------


def test_heads_up_start():
    s = GameState.new_hand(GameConfig(), 1, list(range(52)))
    assert s.num_players == 2 and s.button == 1 and s.street == 0
    assert s.board == [] and s.pot == 150
    assert s.stacks == [19900, 19950] and s.street_bets == [100, 50]
    assert s.current_player == 1 and not s.is_terminal
    assert s.hole_cards(0) == [0, 1] and s.hole_cards(1) == [2, 3]
    la = s.legal_actions()
    assert (la.can_fold, la.can_check, la.call_amount, la.min_raise_to, la.max_raise_to) == (
        True,
        False,
        50,
        200,
        20000,
    )
    s.apply(Action.check_call())
    s.apply(Action.check_call())
    assert s.street == 1 and s.board == [4, 5, 6] and s.current_player == 0
    assert s.history == [(0, 1, Action.check_call()), (0, 0, Action.check_call())]
    with pytest.raises(ValueError):
        s.payoffs()
    with pytest.raises(ValueError):
        s.hole_cards(2)


def test_side_pots_and_split():
    # Three all-ins of different sizes (see the Rust scenario tests).
    cfg = GameConfig(stacks=[1000, 2000, 3000, 10000])
    holes = cards("As Ac 7h 7s Kh Kd Qh Js")
    board = cards("Ah Ad 7c 7d 2s")
    deck = holes + board + [c for c in range(52) if c not in holes + board]
    s = GameState.new_hand(cfg, 3, deck)
    for a in [Action.raise_to(3000), Action.check_call(), Action.check_call(), Action.check_call()]:
        s.apply(a)
    assert s.is_terminal and len(s.board) == 5
    assert s.payoffs() == [3000, 1000, -1000, -3000]

    # Odd chip: 75 chopped between seats 0 and 2; seat 2 is closer to the button's left.
    cfg = GameConfig(stacks=[1000] * 3, small_blind=15, big_blind=30)
    holes = cards("2c 3d 6c 7d 4c 5d")
    board = cards("Ah Kh Qh Jh Th")
    deck = holes + board + [c for c in range(52) if c not in holes + board]
    s = GameState.new_hand(cfg, 0, deck)
    s.apply(Action.check_call())
    s.apply(Action.fold())
    while not s.is_terminal:
        s.apply(Action.check_call())
    assert s.payoffs() == [7, -15, 8]


def test_illegal_actions_rejected_without_side_effects():
    s = GameState.new_hand(GameConfig(), 0, list(range(52)))
    before = s.clone()
    for a in [Action.raise_to(199), Action.raise_to(20001)]:
        with pytest.raises(ValueError):
            s.apply(a)
        assert s == before
    s.apply(Action.check_call())
    with pytest.raises(ValueError):
        s.apply(Action.fold())  # checking is possible
    s.apply(Action.check_call())
    assert s.street == 1
    with pytest.raises(ValueError):
        s.apply(Action.fold())
    s.apply(Action.raise_to(19900))  # all-in bet on the flop
    la = s.legal_actions()
    assert (la.can_fold, la.call_amount, la.min_raise_to, la.max_raise_to) == (True, 19900, 0, 0)
    with pytest.raises(ValueError):
        s.apply(Action.raise_to(19900))  # no one left to raise against
    s.apply(Action.check_call())
    assert s.is_terminal
    with pytest.raises(ValueError):
        s.apply(Action.check_call())


def random_action(s, rng):
    la = s.legal_actions()
    u = rng.random()
    if la.can_fold and u < 0.1:
        return Action.fold()
    if la.min_raise_to > 0 and u > 0.75:
        if la.min_raise_to >= la.max_raise_to or rng.random() < 0.5:
            return Action.raise_to(la.max_raise_to)
        return Action.raise_to(int(rng.integers(la.min_raise_to, la.max_raise_to + 1)))
    return Action.check_call()


def candidate_actions(la, rng):
    out = [Action.fold(), Action.check_call()]
    for x in {la.min_raise_to - 1, la.min_raise_to, la.min_raise_to + 1, la.max_raise_to - 1, la.max_raise_to,
              la.max_raise_to + 1, 1, int(rng.integers(1, 2 * max(la.max_raise_to, 1) + 2))}:
        if x > 0:
            out.append(Action.raise_to(x))
    return out


def expected_legal(la, a):
    if a.kind == FOLD:
        return la.can_fold
    if a.kind == CHECK_CALL:
        return la.can_fold or la.can_check
    if la.min_raise_to <= 0:
        return False
    return a.amount == la.max_raise_to or la.min_raise_to <= a.amount <= la.max_raise_to


def random_config(rng):
    n = int(rng.integers(2, 10))
    bb = int(rng.choice([2, 10, 100]))
    sb = int(rng.choice([bb // 2, bb, max(1, bb // 3)]))
    ante = int(rng.integers(1, bb + 1)) if rng.random() < 0.25 else 0
    stacks = [int(rng.choice([rng.integers(1, 3 * bb + 1), 200 * bb, rng.integers(1, 100 * bb + 1)])) for _ in range(n)]
    return GameConfig(num_players=n, stacks=stacks, small_blind=sb, big_blind=bb, ante=ante)


def play_hand(cfg, button, deck, rng, check_legality):
    s = GameState.new_hand(cfg, button, deck)
    total = sum(cfg.stacks)
    actions = []
    while True:
        assert sum(s.stacks) + s.pot == total
        assert all(x >= 0 for x in s.stacks)
        assert s.contributed == [cfg.stacks[p] - s.stacks[p] for p in range(cfg.num_players)]
        assert len(s.board) == [0, 3, 4, 5][s.street]
        if s.is_terminal:
            assert s.current_player == -1
            break
        p = s.current_player
        assert 0 <= p < cfg.num_players and not s.folded[p] and not s.all_in[p]
        la = s.legal_actions()
        assert la.can_fold != la.can_check
        if check_legality:
            for a in candidate_actions(la, rng):
                ok = True
                try:
                    s.child(a)
                except ValueError:
                    ok = False
                assert ok == expected_legal(la, a) == la.is_legal(a), (a, la, s)
        a = random_action(s, rng)
        parent = s.clone()
        child = s.child(a)
        assert s == parent  # child() does not mutate
        s.apply(a)
        assert s == child
        actions.append(a)
    return s, actions


@pytest.mark.parametrize("seed", range(4))
def test_random_hands_conserve_chips_and_respect_legality(seed):
    rng = np.random.default_rng(seed)
    for _ in range(250):
        cfg = random_config(rng)
        button = int(rng.integers(cfg.num_players))
        deck = rng.permutation(52).tolist()
        s, _ = play_hand(cfg, button, deck, rng, check_legality=True)
        pay = s.payoffs()
        assert len(pay) == cfg.num_players
        assert sum(pay) == 0
        for p in range(cfg.num_players):
            assert pay[p] >= -s.contributed[p]
            assert cfg.stacks[p] + pay[p] >= 0
            if s.folded[p]:
                assert pay[p] == -s.contributed[p]
        if sum(not f for f in s.folded) >= 2:
            assert len(s.board) == 5


def test_heads_up_random_hands_bulk():
    rng = np.random.default_rng(99)
    cfg = GameConfig()
    for i in range(2000):
        s, _ = play_hand(cfg, i % 2, rng.permutation(52).tolist(), rng, check_legality=False)
        pay = s.payoffs()
        assert sum(pay) == 0 and abs(pay[0]) <= 20000


def test_determinism_given_same_deck():
    rng = np.random.default_rng(5)
    for _ in range(100):
        cfg = random_config(rng)
        button = int(rng.integers(cfg.num_players))
        deck = rng.permutation(52).tolist()
        seed = int(rng.integers(1 << 30))
        s1, a1 = play_hand(cfg, button, deck, np.random.default_rng(seed), check_legality=False)
        s2, a2 = play_hand(cfg, button, deck, np.random.default_rng(seed), check_legality=False)
        assert a1 == a2
        assert s1 == s2
        assert s1.history == s2.history
        assert s1.public_key() == s2.public_key()
        assert s1.payoffs() == s2.payoffs()
        assert s1.board == s2.board == deck[2 * cfg.num_players : 2 * cfg.num_players + len(s1.board)]
        # Replaying the recorded actions on a fresh state gives the same result.
        s3 = GameState.new_hand(cfg, button, deck)
        for a in a1:
            s3.apply(a)
        assert s3 == s1 and s3.payoffs() == s1.payoffs()
        for p in range(cfg.num_players):
            assert s1.hole_cards(p) == deck[2 * p : 2 * p + 2]


def test_keys_and_copies():
    s = GameState.new_hand(GameConfig(), 0, list(range(52)))
    pk = s.public_key()
    assert isinstance(pk, bytes)
    k0, k1 = s.infoset_key(0), s.infoset_key(1)
    assert k0.startswith(pk) and k1.startswith(pk) and k0 != k1
    c = copy.copy(s)
    d = copy.deepcopy(s)
    c.apply(Action.raise_to(250))
    assert s.public_key() == pk and d == s and c != s
    assert c.public_key() != pk
    assert c.config == GameConfig()


def test_masked_state():
    s = GameState.new_hand(GameConfig(), 0, list(range(52)))
    m = s.masked(1)
    assert m.is_masked
    assert m.hole_cards(0) == [] and m.hole_cards(1) == s.hole_cards(1)
    assert m.infoset_key(1) == s.infoset_key(1)
    with pytest.raises(ValueError):
        m.infoset_key(0)
    m.apply(Action.check_call())  # no card dealt yet
    with pytest.raises(ValueError):
        m.apply(Action.check_call())  # would deal the hidden flop
