"""Hand-written rule scenarios for the reference engine."""

import pytest
from helpers import make_deck

from pokerbot.reference import CHECK_CALL, RAISE, Action, GameConfig, GameState, cards_from_str

F, C, R = Action.fold, Action.check_call, Action.raise_to


def hu(stacks=(20000, 20000), **kw):
    return GameConfig(num_players=2, stacks=list(stacks), **kw)


# -- heads-up order -------------------------------------------------------------


def test_heads_up_blinds_and_order():
    deck = make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")
    for button in (0, 1):
        other = 1 - button
        s = GameState.new_hand(hu(), button, deck)
        assert s.street_bets[button] == 50 and s.street_bets[other] == 100
        assert s.pot == 150 and s.current_player == button
        la = s.legal_actions()
        assert la.can_fold and not la.can_check and la.call_amount == 50
        assert la.min_raise_to == 200 and la.max_raise_to == 20000
        s.apply(C())  # button completes
        assert s.street == 0 and s.current_player == other  # big blind option
        la = s.legal_actions()
        assert la.can_check and not la.can_fold and la.call_amount == 0
        s.apply(C())
        assert s.street == 1 and s.board == cards_from_str("QhJhTh")
        assert s.current_player == other  # non-button acts first postflop
        assert s.street_bets == [0, 0]
        s.apply(C())
        assert s.current_player == button


def test_hole_cards_dealt_by_seat_not_button():
    deck = make_deck(["AsKs", "2c7d"], "QhJhTh9h8h")
    s = GameState.new_hand(hu(), 1, deck)
    assert s.hole_cards(0) == cards_from_str("AsKs")
    assert s.hole_cards(1) == cards_from_str("2c7d")


def test_fold_preflop_payoffs():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    s.apply(F())
    assert s.is_terminal and s.current_player == -1
    assert s.payoffs() == [-50, 50]
    assert s.street == 0 and s.board == []


def test_cannot_fold_when_check_available_and_bad_raises():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    s.apply(C())
    with pytest.raises(ValueError):
        s.apply(F())
    with pytest.raises(ValueError):
        s.apply(R(150))  # below min raise
    with pytest.raises(ValueError):
        s.apply(R(20001))  # above stack
    s.apply(R(200))
    assert s.legal_actions().min_raise_to == 300


def test_min_raise_tracks_largest_raise():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    s.apply(R(350))  # raise by 250
    la = s.legal_actions()
    assert la.min_raise_to == 600 and la.call_amount == 250
    s.apply(R(1000))  # raise by 650
    assert s.legal_actions().min_raise_to == 1650
    s.apply(C())
    # new street: min bet is the big blind
    assert s.street == 1
    la = s.legal_actions()
    assert la.can_check and la.min_raise_to == 100


def test_all_in_preflop_runs_out_board():
    deck = make_deck(["AsAd", "KcKd"], "2h3h4h9sTs")
    s = GameState.new_hand(hu(), 0, deck)
    s.apply(R(20000))
    assert not s.is_terminal and s.current_player == 1
    la = s.legal_actions()
    assert la.min_raise_to == 0 and la.max_raise_to == 0  # nothing to raise against
    s.apply(C())
    assert s.is_terminal and s.street == 3
    assert s.board == cards_from_str("2h3h4h9sTs")
    assert s.all_in == [True, True]
    assert s.payoffs() == [20000, -20000]


def test_short_stack_call_all_in_and_uncalled_bet_returned():
    deck = make_deck(["2c2d", "AhKh"], "AsKd7c8d9s")
    s = GameState.new_hand(hu(stacks=(20000, 3000)), 0, deck)
    s.apply(R(10000))
    la = s.legal_actions()
    assert la.call_amount == 2900 and la.min_raise_to == 0
    s.apply(C())
    assert s.is_terminal and s.street == 3
    # seat 1 wins 3000 from seat 0; the rest of seat 0's raise is returned
    assert s.payoffs() == [-3000, 3000]


def test_split_pot_heads_up():
    deck = make_deck(["2c3d", "2d3c"], "AsKsQsJsTs")
    s = GameState.new_hand(hu(), 1, deck)
    s.apply(C())
    s.apply(C())
    for _ in range(3):
        s.apply(C())
        s.apply(C())
    assert s.is_terminal
    assert s.payoffs() == [0, 0]


def test_short_blind_is_all_in():
    s = GameState.new_hand(hu(stacks=(20000, 60)), 1, make_deck(["AsKs", "2c7d"]))
    # seat 1 is the button (small blind) and has 60 chips: posts 50, 10 behind
    assert s.street_bets == [100, 50]
    s2 = GameState.new_hand(hu(stacks=(20000, 40)), 1, make_deck(["AsKs", "2c7d"]))
    # small blind all-in for 40; big blind has nothing to call and no one to
    # raise against: the board runs out inside new_hand
    assert s2.all_in == [False, True] and s2.is_terminal and s2.street == 3
    assert s2.street_bets == [0, 0] and s2.pot == 140 and len(s2.board) == 5
    assert sum(s2.payoffs()) == 0 and abs(s2.payoffs()[0]) == 40


def test_short_big_blind_still_requires_full_call():
    deck = make_deck(["AsAh", "7c2d"], "KdQd3c4h9s")
    s = GameState.new_hand(hu(stacks=(20000, 30)), 0, deck)
    # seat 1 is the big blind with 30 chips: all-in for 30
    assert s.street_bets == [50, 30] and s.all_in == [False, True]
    la = s.legal_actions()
    assert la.call_amount == 50 and la.min_raise_to == 0  # to the full big blind
    s.apply(C())
    assert s.is_terminal and s.street_bets == [0, 0]
    assert s.payoffs() == [30, -30]  # the uncalled 70 comes back to seat 0


def test_antes_count_in_pot_not_street_bets():
    cfg = GameConfig(num_players=3, stacks=[1000, 1000, 1000], ante=10)
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h"]))
    assert s.pot == 30 + 150
    assert s.street_bets == [0, 50, 100]
    assert s.stacks == [990, 940, 890]


# -- multiway order -----------------------------------------------------------


def test_three_player_order():
    cfg = GameConfig(num_players=3, stacks=[10000] * 3)
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h"], "QhJhTh9h8h"))
    assert s.street_bets == [0, 50, 100]  # SB = button+1, BB = button+2
    assert s.current_player == 0  # UTG is the button three-handed
    s.apply(C())
    assert s.current_player == 1
    s.apply(C())
    assert s.current_player == 2
    s.apply(C())
    assert s.street == 1 and s.current_player == 1  # first seat after the button
    s.apply(C())
    assert s.current_player == 2
    s.apply(C())
    assert s.current_player == 0
    s.apply(C())
    assert s.street == 2 and s.current_player == 1


def test_multiway_order_skips_folded_and_all_in():
    cfg = GameConfig(num_players=6, stacks=[10000, 10000, 10000, 500, 10000, 10000])
    s = GameState.new_hand(cfg, 5, make_deck(["AsKs", "2c7d", "3h4h", "5c5d", "6h6s", "8c9c"]))
    # button 5, SB 0, BB 1, UTG 2
    assert s.street_bets[0] == 50 and s.street_bets[1] == 100
    assert s.current_player == 2
    s.apply(F())
    assert s.current_player == 3
    s.apply(R(500))  # all-in
    assert s.all_in[3]
    assert s.current_player == 4
    s.apply(C())
    s.apply(F())  # 5
    s.apply(C())  # 0 (SB)
    s.apply(C())  # 1 (BB)
    assert s.street == 1
    assert s.current_player == 0  # first live, non-all-in seat after the button
    s.apply(C())
    assert s.current_player == 1
    s.apply(C())
    assert s.current_player == 4


# -- incomplete raises -------------------------------------------------------------


def test_all_in_for_less_does_not_reopen_action():
    cfg = GameConfig(num_players=3, stacks=[10000, 10000, 350])
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h"]))
    s.apply(R(300))  # seat 0 raises by 200
    s.apply(C())  # seat 1 calls
    assert s.current_player == 2
    la = s.legal_actions()
    assert la.min_raise_to == 500 and la.max_raise_to == 350  # all-in for less only
    with pytest.raises(ValueError):
        s.apply(R(340))
    s.apply(R(350))  # all-in, raise of 50 < 200
    assert s.current_player == 0
    la = s.legal_actions()
    assert la.call_amount == 50 and la.min_raise_to == 0 and la.max_raise_to == 0
    with pytest.raises(ValueError):
        s.apply(R(10000))
    s.apply(C())
    assert s.current_player == 1
    assert s.legal_actions().min_raise_to == 0
    s.apply(C())
    assert s.street == 1


def test_short_all_ins_adding_up_to_full_raise_reopen_action():
    # button 0, SB 1, BB 2, UTG 3
    cfg = GameConfig(num_players=4, stacks=[400, 520, 10000, 10000])
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h", "5c6c"]))
    assert s.current_player == 3
    s.apply(R(300))  # full raise of 200
    s.apply(R(400))  # seat 0 all-in: +100, short
    s.apply(R(520))  # seat 1 all-in: +120, short
    s.apply(C())  # seat 2 (had not acted) calls
    assert s.current_player == 3
    la = s.legal_actions()
    # seat 3 last acted at 300; the bet rose by 220 >= 200: re-opened
    assert la.call_amount == 220 and la.min_raise_to == 720 and la.max_raise_to == 10000


def test_single_short_all_in_does_not_reopen_multiway():
    cfg = GameConfig(num_players=4, stacks=[400, 10000, 10000, 10000])
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h", "5c6c"]))
    s.apply(R(300))
    s.apply(R(400))  # seat 0 all-in: +100, short
    la = s.legal_actions()  # seat 1 has not acted: may raise
    assert la.min_raise_to == 600
    s.apply(C())
    s.apply(C())
    assert s.current_player == 3
    la = s.legal_actions()
    assert la.call_amount == 100 and la.min_raise_to == 0


def test_full_raise_all_in_reopens_action():
    cfg = GameConfig(num_players=3, stacks=[10000, 10000, 500])
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d", "3h4h"]))
    s.apply(R(300))
    s.apply(C())
    s.apply(R(500))  # all-in, raise of exactly 200 = full raise
    la = s.legal_actions()
    assert la.call_amount == 200 and la.min_raise_to == 700 and la.max_raise_to == 10000


def test_incomplete_raise_keeps_min_raise_increment_for_unacted_players():
    cfg = GameConfig(num_players=3, stacks=[10000, 150, 10000])
    s = GameState.new_hand(cfg, 2, make_deck(["AsKs", "2c7d", "3h4h"]))
    # button 2, SB 0, BB 1 (stack 150), first to act is 2
    s.apply(C())  # button limps 100
    s.apply(C())  # SB completes
    s.apply(R(150))  # BB all-in for 50 more: incomplete
    assert s.current_player == 2
    la = s.legal_actions()
    # button already acted and faced only an incomplete raise
    assert la.min_raise_to == 0 and la.call_amount == 50


def test_postflop_short_all_in_does_not_reopen_for_bettor():
    cfg = hu(stacks=(10000, 1200))
    s = GameState.new_hand(cfg, 0, make_deck(["AsKs", "2c7d"], "QhJhTh9h8h"))
    s.apply(C())
    s.apply(C())
    assert s.street == 1 and s.current_player == 1
    s.apply(C())
    s.apply(R(1000))
    la = s.legal_actions()
    assert la.min_raise_to == 2000 and la.max_raise_to == 1100
    s.apply(R(1100))
    la = s.legal_actions()
    assert la.min_raise_to == 0 and la.call_amount == 100
    s.apply(C())
    assert s.is_terminal and len(s.board) == 5


# -- pots -----------------------------------------------------------------------------


def test_side_pots_three_all_in_sizes():
    # best to worst: seat 0 > seat 1 > seat 2 > seat 3
    deck = make_deck(["AsAh", "KsKh", "QsQh", "7c2d"], "AdKdQd3c4h")
    cfg = GameConfig(num_players=4, stacks=[1000, 2000, 3000, 10000])
    s = GameState.new_hand(cfg, 3, deck)
    # button 3, SB 0, BB 1, UTG 2
    assert s.current_player == 2
    s.apply(R(3000))
    s.apply(R(10000))
    assert s.legal_actions().min_raise_to == 0  # cannot raise when all-in by calling
    s.apply(C())  # seat 0 all-in for 1000
    s.apply(C())  # seat 1 all-in for 2000
    assert s.is_terminal and s.street == 3
    assert s.pot == 1000 + 2000 + 3000 + 10000
    # main 4000 -> 0, side 3000 -> 1, side 2000 -> 2, 7000 back to 3
    assert s.payoffs() == [3000, 1000, -1000, -3000]


def test_side_pot_won_by_short_stack_and_rest_by_second():
    # seat 3 (short) best, seat 0 second, others worse
    deck = make_deck(["KsKh", "7c2d", "8c3d", "AsAh"], "AdKdQd3c4h")
    cfg = GameConfig(num_players=4, stacks=[5000, 5000, 5000, 1000])
    s = GameState.new_hand(cfg, 0, deck)
    # button 0, SB 1, BB 2, UTG 3
    s.apply(R(1000))  # seat 3 all-in
    s.apply(C())  # seat 0 calls 1000
    s.apply(C())  # seat 1
    s.apply(C())  # seat 2
    for _ in range(3):  # check it down on flop/turn/river
        for _ in range(3):
            s.apply(C())
    assert s.is_terminal
    assert s.payoffs() == [-1000, -1000, -1000, 3000]


def test_split_pot_odd_chip_goes_left_of_button():
    cfg = GameConfig(num_players=3, stacks=[1000] * 3, small_blind=5, big_blind=10)
    deck = make_deck(["2c3d", "4c5d", "2d3c"], "AsKsQsJsTs")
    s = GameState.new_hand(cfg, 0, deck)
    # button 0, SB 1 (5), BB 2 (10)
    s.apply(C())  # button calls 10
    s.apply(F())  # SB folds its 5
    s.apply(C())  # BB checks
    for _ in range(3):
        s.apply(C())
        s.apply(C())
    assert s.is_terminal and s.pot == 25
    # royal flush on board: seats 0 and 2 split 25; odd chip to seat 2
    # (first eligible seat clockwise from the seat after the button)
    assert s.payoffs() == [2, -5, 3]


def test_odd_chip_order_wraps_around_button():
    cfg = GameConfig(num_players=3, stacks=[1000] * 3, small_blind=5, big_blind=10)
    deck = make_deck(["2c3d", "2d3c", "4c5d"], "AsKsQsJsTs")
    s = GameState.new_hand(cfg, 1, deck)
    # button 1, SB 2, BB 0, UTG = button 1
    s.apply(C())  # seat 1 calls
    s.apply(F())  # seat 2 (SB) folds
    s.apply(C())  # seat 0 checks
    for _ in range(3):
        s.apply(C())
        s.apply(C())
    # seats 0 and 1 split 25; clockwise from seat 2: 2 (folded), 0 -> odd chip to 0
    assert s.payoffs() == [3, 2, -5]


def test_history_and_child_clone_independent():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    c = s.child(R(300))
    assert s.history == [] and len(c.history) == 1
    street, player, a = c.history[0]
    assert (street, player, a.kind, a.amount) == (0, 0, RAISE, 300)
    d = c.clone()
    d.apply(C())
    assert c.street == 0 and d.street == 1
    assert d.history[-1][2].kind == CHECK_CALL


def test_public_and_infoset_keys():
    a = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"], "QhJhTh9h8h"))
    b = GameState.new_hand(hu(), 0, make_deck(["3s4s", "5c6d"], "QhJhTh9h8h"))
    for s in (a, b):
        s.apply(R(300))
        s.apply(C())
    assert isinstance(a.public_key(), bytes)
    assert a.public_key() == b.public_key()
    assert a.infoset_key(0) != b.infoset_key(0)
    assert a.infoset_key(0) != a.infoset_key(1)
    assert a.infoset_key(0).startswith(a.public_key())
    c = a.child(C())
    assert c.public_key() != a.public_key()
    # different raise sizes give different keys
    x = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    assert x.child(R(300)).public_key() != x.child(R(400)).public_key()
    # hole-card order does not matter for the infoset key
    y = GameState.new_hand(hu(), 0, make_deck(["KsAs", "2c7d"]))
    z = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    assert y.infoset_key(0) == z.infoset_key(0)


def test_key_format():
    s = GameState.new_hand(hu(), 1, make_deck(["AsKs", "2c7d"], "QhJhTh9h8h"))
    assert s.public_key() == bytes([1, 0])
    s.apply(R(300))
    s.apply(C())
    s.apply(C())
    board = cards_from_str("QhJhTh")
    expected = bytes([1, 3, *board, 0x02, *(300).to_bytes(4, "little"), 0x01, 0x11])
    assert s.public_key() == expected
    ks = sorted(cards_from_str("AsKs"))
    assert s.infoset_key(0) == expected + bytes([0, *ks])


def test_amounts_on_fold_and_call_rejected():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    with pytest.raises(ValueError):
        s.apply(Action(CHECK_CALL, 50))
    with pytest.raises(ValueError):
        s.apply(Action(0, 5))
    assert s.history == []


def test_payoffs_only_at_terminal():
    s = GameState.new_hand(hu(), 0, make_deck(["AsKs", "2c7d"]))
    with pytest.raises(ValueError):
        s.payoffs()
    s.apply(F())
    with pytest.raises(ValueError):
        s.apply(C())


def test_bad_deck_rejected():
    with pytest.raises(ValueError):
        GameState.new_hand(hu(), 0, [0] * 52)
    with pytest.raises(ValueError):
        GameState.new_hand(hu(), 0, list(range(8)))
    with pytest.raises(ValueError):
        GameConfig(num_players=10)
