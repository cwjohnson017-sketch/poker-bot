#![allow(clippy::needless_range_loop)]
//! Hand-written betting scenarios plus randomized rule invariants.

use poker_engine::cards::{cards_from_str, Card};
use poker_engine::game::{Action, Chips, GameConfig, GameError, GameState, LegalActions, MAX_PLAYERS};
use poker_engine::sim::{fresh_deck, random_action, Rng};

// ----------------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------------

/// Deck with the given hole cards per seat and 5-card board; the rest of
/// the deck follows in ascending order.
fn deck(holes: &[&str], board: &str) -> Vec<Card> {
    let mut v: Vec<Card> = Vec::new();
    for h in holes {
        let c = cards_from_str(h).unwrap();
        assert_eq!(c.len(), 2);
        v.extend(c);
    }
    let b = cards_from_str(board).unwrap();
    assert_eq!(b.len(), 5);
    v.extend(b);
    for c in 0..52u8 {
        if !v.contains(&c) {
            v.push(c);
        }
    }
    v
}

fn cfg(stacks: &[Chips], sb: Chips, bb: Chips, ante: Chips) -> GameConfig {
    GameConfig { num_players: stacks.len(), stacks: stacks.to_vec(), small_blind: sb, big_blind: bb, ante }
}

fn c() -> Action {
    Action::check_call()
}
fn f() -> Action {
    Action::fold()
}
fn r(x: Chips) -> Action {
    Action::raise_to(x)
}

/// Apply actions, asserting the expected actor before each one.
fn play(s: &mut GameState, seq: &[(usize, Action)]) {
    for &(p, a) in seq {
        assert_eq!(s.current_player(), Some(p), "expected seat {p} to act before {a} in {s}");
        s.apply(a).unwrap_or_else(|e| panic!("{a} by seat {p} rejected: {e} in {s}"));
    }
}

fn la(s: &GameState) -> LegalActions {
    s.legal_actions()
}

fn legal(can_fold: bool, call: Chips, min: Chips, max: Chips) -> LegalActions {
    LegalActions { can_fold, can_check: !can_fold, call_amount: call, min_raise_to: min, max_raise_to: max }
}

fn assert_illegal(s: &mut GameState, a: Action) {
    let before = s.clone();
    assert!(matches!(s.apply(a), Err(GameError::IllegalAction(_))), "{a} should be illegal in {s}");
    assert_eq!(*s, before, "failed apply must not change the state");
}

// ----------------------------------------------------------------------
// Heads-up blinds and order
// ----------------------------------------------------------------------

#[test]
fn heads_up_preflop_order_and_blinds() {
    for button in 0..2 {
        let bb_seat = 1 - button;
        let mut s = GameState::new_hand(&GameConfig::default(), button, &fresh_deck()).unwrap();
        assert_eq!(s.street(), 0);
        assert_eq!(s.board().len(), 0);
        assert_eq!(s.pot(), 150);
        assert_eq!(s.street_bets()[button], 50);
        assert_eq!(s.street_bets()[bb_seat], 100);
        assert_eq!(s.stacks()[button], 19_950);
        assert_eq!(s.stacks()[bb_seat], 19_900);
        // Button (small blind) acts first preflop.
        assert_eq!(s.current_player(), Some(button));
        assert_eq!(la(&s), legal(true, 50, 200, 20_000));
        play(&mut s, &[(button, c())]);
        // Big blind has the option.
        assert_eq!(la(&s), legal(false, 0, 200, 20_000));
        play(&mut s, &[(bb_seat, c())]);
        // Non-button acts first on the flop; minimum bet is the big blind.
        assert_eq!(s.street(), 1);
        assert_eq!(s.board().len(), 3);
        assert_eq!(s.street_bets(), &[0, 0]);
        assert_eq!(s.pot(), 200);
        assert_eq!(s.current_player(), Some(bb_seat));
        assert_eq!(la(&s), legal(false, 0, 100, 19_900));
        play(&mut s, &[(bb_seat, c()), (button, c())]);
        assert_eq!(s.street(), 2);
        assert_eq!(s.current_player(), Some(bb_seat));
    }
}

#[test]
fn check_through_all_streets_to_showdown() {
    let d = deck(&["As Ad", "Ks Kd"], "2c 7h 9d Jc 3s");
    let mut s = GameState::new_hand(&GameConfig::default(), 0, &d).unwrap();
    play(&mut s, &[(0, c()), (1, c())]);
    for street in 1..=3 {
        assert_eq!(s.street(), street);
        assert_eq!(s.board().len(), [0, 3, 4, 5][street]);
        play(&mut s, &[(1, c()), (0, c())]);
    }
    assert!(s.is_terminal());
    assert_eq!(s.current_player(), None);
    assert_eq!(s.street(), 3);
    assert_eq!(s.board(), &cards_from_str("2c 7h 9d Jc 3s").unwrap()[..]);
    assert_eq!(s.payoffs().unwrap(), vec![100, -100]);
    let streets: Vec<u8> = s.history().iter().map(|h| h.street).collect();
    let players: Vec<u8> = s.history().iter().map(|h| h.player).collect();
    assert_eq!(streets, vec![0, 0, 1, 1, 2, 2, 3, 3]);
    assert_eq!(players, vec![0, 1, 1, 0, 1, 0, 1, 0]);
    assert_eq!(s.showdown_order(), vec![1, 0]);
    assert_illegal(&mut s, c());
}

#[test]
fn fold_preflop_and_uncalled_bets() {
    let mut s = GameState::new_hand(&GameConfig::default(), 0, &fresh_deck()).unwrap();
    assert!(matches!(s.payoffs(), Err(GameError::NotTerminal)));
    play(&mut s, &[(0, f())]);
    assert!(s.is_terminal());
    assert_eq!(s.board().len(), 0);
    assert_eq!(s.payoffs().unwrap(), vec![-50, 50]);

    // Shove, fold: the uncalled part of the shove comes back.
    let mut s = GameState::new_hand(&GameConfig::default(), 1, &fresh_deck()).unwrap();
    play(&mut s, &[(1, r(20_000)), (0, f())]);
    assert_eq!(s.payoffs().unwrap(), vec![-100, 100]);
    assert_eq!(s.pot(), 20_100);
}

// ----------------------------------------------------------------------
// Min-raise tracking
// ----------------------------------------------------------------------

#[test]
fn min_raise_tracking_heads_up() {
    let mut s = GameState::new_hand(&GameConfig::default(), 0, &fresh_deck()).unwrap();
    assert_illegal(&mut s, r(199));
    assert_illegal(&mut s, r(20_001));
    assert_illegal(&mut s, Action { amount: 5, ..c() });
    play(&mut s, &[(0, r(300))]); // raise of 200
    assert_eq!(la(&s), legal(true, 200, 500, 20_000));
    assert_illegal(&mut s, r(499));
    play(&mut s, &[(1, r(500))]); // min re-raise (200)
    assert_eq!(la(&s), legal(true, 200, 700, 20_000));
    play(&mut s, &[(0, r(1_000))]); // raise of 500
    assert_eq!(la(&s), legal(true, 500, 1_500, 20_000));
    assert_eq!(s.num_raises_this_street(), 3);
    play(&mut s, &[(1, c())]);
    // Flop: min bet resets to the big blind.
    assert_eq!(s.street(), 1);
    assert_eq!(s.num_raises_this_street(), 0);
    assert_eq!(la(&s), legal(false, 0, 100, 19_000));
    assert_illegal(&mut s, f());
    assert_illegal(&mut s, r(99));
    play(&mut s, &[(1, r(250))]);
    assert_eq!(la(&s), legal(true, 250, 500, 19_000));
    play(&mut s, &[(0, r(500))]);
    assert_eq!(la(&s), legal(true, 250, 750, 19_000));
}

#[test]
fn all_in_for_less_than_min_raise_does_not_reopen_action() {
    // Seat 0 button/UTG, seat 1 SB, seat 2 BB (short).
    let config = cfg(&[10_000, 10_000, 1_600], 50, 100, 0);
    let mut s = GameState::new_hand(&config, 0, &fresh_deck()).unwrap();
    assert_eq!(s.current_player(), Some(0)); // 3-handed: the button is UTG
    play(&mut s, &[(0, c()), (1, c()), (2, c())]);
    assert_eq!(s.street(), 1);
    // Flop: seat 1 bets 1000, seat 2 shoves 1500 (a raise of 500 < 1000).
    play(&mut s, &[(1, r(1_000))]);
    assert_eq!(la(&s), legal(true, 1_000, 2_000, 1_500));
    assert!(la(&s).is_legal(r(1_500)));
    assert!(!la(&s).is_legal(r(1_499)));
    play(&mut s, &[(2, r(1_500))]);
    assert!(s.all_in()[2]);
    // Seat 0 has not acted yet on this street: may raise; min is 1500 + 1000.
    assert_eq!(la(&s), legal(true, 1_500, 2_500, 9_900));
    play(&mut s, &[(0, c())]);
    // Seat 1 bet 1000 and faces only 500 more: call or fold, no raise.
    assert_eq!(la(&s), legal(true, 500, 0, 0));
    assert_illegal(&mut s, r(2_500));
    assert_illegal(&mut s, r(9_900));
    play(&mut s, &[(1, c())]);
    assert_eq!(s.street(), 2);
    assert_eq!(s.current_player(), Some(1));
    assert_eq!(s.pot(), 300 + 4_500);

    // Same spot, but seat 0 makes a full raise: action re-opens for seat 1.
    let mut s = GameState::new_hand(&config, 0, &fresh_deck()).unwrap();
    play(&mut s, &[(0, c()), (1, c()), (2, c()), (1, r(1_000)), (2, r(1_500)), (0, r(2_500))]);
    assert_eq!(la(&s), legal(true, 1_500, 3_500, 9_900));
}

#[test]
fn checker_facing_short_all_in_bet_cannot_raise() {
    // A flop bet below the minimum (all-in for less) does not re-open
    // action for a player who already checked; players yet to act may raise.
    let config = cfg(&[10_000, 150, 10_000], 50, 100, 0);
    let mut s = GameState::new_hand(&config, 2, &fresh_deck()).unwrap();
    // button 2, SB seat 0, BB seat 1, UTG seat 2.
    play(&mut s, &[(2, c()), (0, c()), (1, c())]);
    assert_eq!(s.street(), 1);
    play(&mut s, &[(0, c())]); // seat 0 checks
    assert_eq!(la(&s), legal(false, 0, 100, 50));
    play(&mut s, &[(1, r(50))]); // all-in bet of 50 < big blind
    // Seat 2 has not acted: may raise to at least 50 + 100.
    assert_eq!(la(&s), legal(true, 50, 150, 9_900));
    play(&mut s, &[(2, c())]);
    // Seat 0 checked before the short bet: call or fold only.
    assert_eq!(la(&s), legal(true, 50, 0, 0));
    play(&mut s, &[(0, c())]);
    assert_eq!(s.street(), 2);
    assert_eq!(s.current_player(), Some(0));
}

#[test]
fn cumulative_short_all_ins_reopen_when_they_add_up_to_a_full_raise() {
    // button 0, SB 1, BB 2, UTG 3.
    for (stack3, reopens) in [(2_200, true), (2_000, false)] {
        let config = cfg(&[10_000, 10_000, 1_600, stack3], 50, 100, 0);
        let mut s = GameState::new_hand(&config, 0, &fresh_deck()).unwrap();
        play(&mut s, &[(3, c()), (0, c()), (1, c()), (2, c())]);
        play(&mut s, &[(1, r(1_000)), (2, r(1_500))]);
        let all_in_to = stack3 - 100;
        play(&mut s, &[(3, r(all_in_to)), (0, c())]);
        let l = la(&s);
        assert_eq!(s.current_player(), Some(1));
        assert_eq!(l.call_amount, all_in_to - 1_000);
        if reopens {
            // Facing 1100 more after betting 1000: a full raise, re-opened.
            assert_eq!(l.min_raise_to, all_in_to + 1_000);
        } else {
            assert_eq!(l.min_raise_to, 0);
        }
    }
}

// ----------------------------------------------------------------------
// All-ins, run-outs, side pots, split pots
// ----------------------------------------------------------------------

#[test]
fn side_pots_with_three_all_ins_of_different_sizes() {
    // button 3, SB 0, BB 1, UTG 2.
    let config = cfg(&[1_000, 2_000, 3_000, 10_000], 50, 100, 0);
    let board = "Ah Ad 7c 7d 2s";
    // (holes, expected payoffs)
    let cases: [([&str; 4], [Chips; 4]); 3] = [
        // Shortest stack best, then the next: every pot goes to a different player.
        (["As Ac", "7h 7s", "Kh Kd", "Qh Js"], [3_000, 1_000, -1_000, -3_000]),
        // Middle stack best: main + first side pot; seat 2 beats seat 3 for the last.
        (["Qh Js", "As Ac", "Kh Kd", "3c 4c"], [-1_000, 5_000, -1_000, -3_000]),
        // Big stack best: wins everything that was matched.
        (["Kh Kd", "Qh Js", "3c 4c", "As Ac"], [-1_000, -2_000, -3_000, 6_000]),
    ];
    for (holes, expected) in cases {
        let mut s = GameState::new_hand(&config, 3, &deck(&holes, board)).unwrap();
        assert_eq!(s.current_player(), Some(2));
        play(&mut s, &[(2, r(3_000)), (3, c())]);
        assert_eq!(la(&s), legal(true, 950, 0, 0)); // seat 0: all-in call only
        play(&mut s, &[(0, c())]);
        assert!(s.all_in()[0]);
        play(&mut s, &[(1, c())]);
        // Only seat 3 can still act and has matched: board is run out.
        assert!(s.is_terminal());
        assert_eq!(s.street(), 3);
        assert_eq!(s.board().len(), 5);
        assert_eq!(s.street_bets(), &[0, 0, 0, 0]);
        assert_eq!(s.pot(), 9_000);
        assert_eq!(s.stacks(), &[0, 0, 0, 7_000]);
        assert_eq!(s.payoffs().unwrap(), expected.to_vec(), "{holes:?}");
    }
}

#[test]
fn split_pot_with_odd_chip() {
    // 3 players, blinds 15/30: seat 1 (SB) folds, the others chop a 75 pot.
    // Odd chip: first eligible seat clockwise after the button.
    let board = "Ah Kh Qh Jh Th"; // royal flush on board
    let holes = ["2c 3d", "6c 7d", "4c 5d"];
    for (button, expected) in [(0usize, vec![7, -15, 8]), (2, vec![-15, 8, 7])] {
        let config = cfg(&[1_000, 1_000, 1_000], 15, 30, 0);
        let mut s = GameState::new_hand(&config, button, &deck(&holes, board)).unwrap();
        let (sb, bb, utg) = ((button + 1) % 3, (button + 2) % 3, button);
        play(&mut s, &[(utg, c()), (sb, f()), (bb, c())]);
        while !s.is_terminal() {
            let p = s.current_player().unwrap();
            play(&mut s, &[(p, c())]);
        }
        assert_eq!(s.pot(), 75);
        assert_eq!(s.payoffs().unwrap(), expected, "button {button}");
    }

    // Three-way chop of 170 (56 each, two odd chips to seats 2 and 3).
    let config = cfg(&[1_000; 4], 20, 50, 0);
    let holes = ["2c 3d", "6c 7d", "4c 5d", "8c 9d"];
    let mut s = GameState::new_hand(&config, 0, &deck(&holes, board)).unwrap();
    play(&mut s, &[(3, c()), (0, c()), (1, f()), (2, c())]);
    while !s.is_terminal() {
        let p = s.current_player().unwrap();
        play(&mut s, &[(p, c())]);
    }
    assert_eq!(s.pot(), 170);
    assert_eq!(s.payoffs().unwrap(), vec![56 - 50, -20, 57 - 50, 57 - 50]);

    // Heads-up chop: even split.
    let mut s = GameState::new_hand(&GameConfig::default(), 0, &deck(&["2c 3d", "4c 5d"], board)).unwrap();
    play(&mut s, &[(0, r(300)), (1, c())]);
    while !s.is_terminal() {
        let p = s.current_player().unwrap();
        play(&mut s, &[(p, c())]);
    }
    assert_eq!(s.payoffs().unwrap(), vec![0, 0]);
}

#[test]
fn split_side_pot_while_main_pot_has_single_winner() {
    // Seat 0 (short) wins the main pot; seats 1 and 2 tie for the side pot.
    let config = cfg(&[500, 5_000, 5_000], 50, 100, 0);
    // Board gives seats 1 and 2 the same straight; seat 0 has a flush.
    let board = "9h Th Jc Qd 2h";
    let holes = ["3h 4h", "Kc 5s", "Kd 6s"];
    let mut s = GameState::new_hand(&config, 0, &deck(&holes, board)).unwrap();
    // button 0 = UTG; SB 1; BB 2.
    play(&mut s, &[(0, r(500)), (1, r(2_000)), (2, c())]);
    assert!(!s.is_terminal()); // seats 1 and 2 still have chips
    while !s.is_terminal() {
        let p = s.current_player().unwrap();
        play(&mut s, &[(p, c())]);
    }
    // main pot 1500 -> seat 0; side pot 3000 split 1500/1500.
    assert_eq!(s.payoffs().unwrap(), vec![1_000, -500, -500]);
}

#[test]
fn heads_up_all_in_preflop_runs_out_board() {
    let d = deck(&["As Ad", "Ks Kd"], "2c 7h 9d Jc Kh");
    let mut s = GameState::new_hand(&GameConfig::default(), 0, &d).unwrap();
    play(&mut s, &[(0, r(20_000))]);
    assert_eq!(la(&s), legal(true, 19_900, 0, 0));
    assert_illegal(&mut s, r(20_000));
    play(&mut s, &[(1, c())]);
    assert!(s.is_terminal());
    assert_eq!(s.board().len(), 5);
    assert_eq!(s.street(), 3);
    assert_eq!(s.payoffs().unwrap(), vec![-20_000, 20_000]);
    assert_eq!(s.showdown_order(), vec![0, 1]); // last aggressor first

    // Unequal stacks: the covered player can only lose his stack.
    let config = cfg(&[5_000, 20_000], 50, 100, 0);
    let mut s = GameState::new_hand(&config, 0, &d).unwrap();
    play(&mut s, &[(0, c()), (1, r(20_000))]);
    assert_eq!(la(&s), legal(true, 4_900, 0, 0));
    play(&mut s, &[(0, c())]);
    assert!(s.is_terminal());
    assert_eq!(s.payoffs().unwrap(), vec![-5_000, 5_000]);
}

#[test]
fn short_blinds() {
    // Big blind all-in for less than the blind: the small blind still has
    // to complete to the full big blind, with no raise possible.
    let d = deck(&["As Ad", "Ks Kd"], "2c 7h 9d Jc 3h");
    let config = cfg(&[1_000, 60], 50, 100, 0);
    let mut s = GameState::new_hand(&config, 0, &d).unwrap();
    assert!(s.all_in()[1]);
    assert_eq!(la(&s), legal(true, 50, 0, 0));
    play(&mut s, &[(0, c())]);
    assert!(s.is_terminal());
    assert_eq!(s.payoffs().unwrap(), vec![60, -60]);

    // Small blind all-in from posting: nobody acts at all.
    let config = cfg(&[30, 1_000], 50, 100, 0);
    let s = GameState::new_hand(&config, 0, &d).unwrap();
    assert!(s.is_terminal());
    assert!(s.history().is_empty());
    assert_eq!(s.board().len(), 5);
    assert_eq!(s.payoffs().unwrap(), vec![30, -30]);
}

#[test]
fn antes() {
    let config = cfg(&[1_000, 1_000, 1_000], 50, 100, 10);
    let s = GameState::new_hand(&config, 0, &fresh_deck()).unwrap();
    assert_eq!(s.pot(), 180);
    assert_eq!(s.street_bets(), &[0, 50, 100]);
    assert_eq!(s.contributed(), &[10, 60, 110]);
    assert_eq!(s.stacks(), &[990, 940, 890]);
    assert_eq!(la(&s), legal(true, 100, 200, 990));
    let mut s2 = s.clone();
    play(&mut s2, &[(0, f()), (1, f())]);
    assert_eq!(s2.payoffs().unwrap(), vec![-10, -60, 70]);
}

// ----------------------------------------------------------------------
// Multiway order, showdown order
// ----------------------------------------------------------------------

#[test]
fn multiway_positions() {
    let config = GameConfig::new(6, 10_000, 50, 100, 0);
    let mut s = GameState::new_hand(&config, 0, &fresh_deck()).unwrap();
    assert_eq!(s.street_bets(), &[0, 50, 100, 0, 0, 0]);
    assert_eq!(s.current_player(), Some(3));
    play(&mut s, &[(3, c()), (4, c()), (5, c()), (0, c()), (1, f()), (2, c())]);
    assert_eq!(s.street(), 1);
    // Seat 1 folded: the first live seat after the button acts first.
    assert_eq!(s.current_player(), Some(2));
    play(&mut s, &[(2, c()), (3, r(300)), (4, f()), (5, c()), (0, c()), (2, c())]);
    assert_eq!(s.street(), 2);
    assert_eq!(s.current_player(), Some(2));

    // Button wraps around.
    let s = GameState::new_hand(&config, 4, &fresh_deck()).unwrap();
    assert_eq!(s.street_bets(), &[100, 0, 0, 0, 0, 50]);
    assert_eq!(s.current_player(), Some(1));
}

#[test]
fn river_showdown_ordering_and_award() {
    let board = "2c 7h 9d Jc 3s";
    let holes = ["4c 5d", "Ah Ad", "Kh Kd"];
    let config = GameConfig::new(3, 10_000, 50, 100, 0);
    // Checked down: first live seat after the button shows first.
    let mut s = GameState::new_hand(&config, 0, &deck(&holes, board)).unwrap();
    play(&mut s, &[(0, c()), (1, c()), (2, c())]);
    while !s.is_terminal() {
        let p = s.current_player().unwrap();
        play(&mut s, &[(p, c())]);
    }
    assert_eq!(s.showdown_order(), vec![1, 2, 0]);
    assert_eq!(s.payoffs().unwrap(), vec![-100, 200, -100]);

    // River bet and raise: the last aggressor shows first.
    let mut s = GameState::new_hand(&config, 0, &deck(&holes, board)).unwrap();
    play(&mut s, &[(0, c()), (1, c()), (2, c())]);
    for _ in 0..2 {
        play(&mut s, &[(1, c()), (2, c()), (0, c())]);
    }
    assert_eq!(s.street(), 3);
    play(&mut s, &[(1, r(200)), (2, r(600)), (0, f()), (1, c())]);
    assert!(s.is_terminal());
    assert_eq!(s.showdown_order(), vec![2, 1]);
    assert_eq!(s.payoffs().unwrap(), vec![-100, 800, -700]);

    // Fold-terminal hands have no showdown.
    let mut s = GameState::new_hand(&config, 0, &deck(&holes, board)).unwrap();
    play(&mut s, &[(0, f()), (1, f())]);
    assert!(s.showdown_order().is_empty());
}

// ----------------------------------------------------------------------
// Validation, keys, masking
// ----------------------------------------------------------------------

#[test]
fn config_and_deck_validation() {
    let d = fresh_deck();
    assert!(GameState::new_hand(&GameConfig::new(1, 100, 1, 2, 0), 0, &d).is_err());
    assert!(GameState::new_hand(&GameConfig::new(10, 100, 1, 2, 0), 0, &d).is_err());
    assert!(GameState::new_hand(&GameConfig::new(2, 100, 1, 2, 0), 2, &d).is_err());
    assert!(GameState::new_hand(&GameConfig::new(2, 0, 1, 2, 0), 0, &d).is_err());
    assert!(GameState::new_hand(&GameConfig::new(2, 100, 3, 2, 0), 0, &d).is_err());
    assert!(GameState::new_hand(&GameConfig::new(2, 100, 1, 0, 0), 0, &d).is_err());
    let mut bad = d.to_vec();
    bad[3] = bad[0];
    assert!(matches!(GameState::new_hand(&GameConfig::default(), 0, &bad), Err(GameError::InvalidDeck(_))));
    assert!(GameState::new_hand(&GameConfig::default(), 0, &d[..8]).is_err());
    assert!(GameState::new_hand(&GameConfig::default(), 0, &d[..9]).is_ok());
    // All 9 seats deal 23 cards.
    let s = GameState::new_hand(&GameConfig::new(9, 1_000, 5, 10, 0), 8, &d).unwrap();
    assert_eq!(s.hole_cards(8), Some([16, 17]));
    assert_eq!(s.current_player(), Some(2));
}

#[test]
fn public_and_infoset_keys() {
    let d = deck(&["As Kd", "7c 2h"], "2c 7h 9d Jc 3s");
    let d_swapped = deck(&["Kd As", "7c 2h"], "2c 7h 9d Jc 3s");
    let d_other = deck(&["Qs Qd", "7c 2h"], "2c 7h 9d Jc 3s");
    let a = GameState::new_hand(&GameConfig::default(), 0, &d).unwrap();
    let b = GameState::new_hand(&GameConfig::default(), 0, &d_swapped).unwrap();
    let o = GameState::new_hand(&GameConfig::default(), 0, &d_other).unwrap();
    assert_eq!(a.public_key(), o.public_key());
    assert_eq!(a.infoset_key(0).unwrap(), b.infoset_key(0).unwrap());
    assert_ne!(a.infoset_key(0).unwrap(), o.infoset_key(0).unwrap());
    assert_eq!(a.infoset_key(1).unwrap(), o.infoset_key(1).unwrap());
    assert_ne!(a.infoset_key(0).unwrap(), a.infoset_key(1).unwrap());
    assert!(a.infoset_key(2).is_err());
    let pk = a.public_key();
    assert!(a.infoset_key(0).unwrap().starts_with(&pk));

    let x = a.child(r(300)).unwrap();
    let y = a.child(r(301)).unwrap();
    let z = a.child(c()).unwrap();
    assert_ne!(x.public_key(), y.public_key());
    assert_ne!(x.public_key(), z.public_key());
    let mut w = a.clone();
    w.apply(r(300)).unwrap();
    assert_eq!(w, x);
    assert_eq!(w.public_key(), x.public_key());
    // Flop dealt: the board shows up in the key.
    let flop = z.child(c()).unwrap();
    assert_eq!(flop.public_key()[1], 3);
    assert_eq!(&flop.public_key()[2..5], &cards_from_str("2c 7h 9d").unwrap()[..]);
    // Button is part of the public key.
    let a1 = GameState::new_hand(&GameConfig::default(), 1, &d).unwrap();
    assert_ne!(a.public_key(), a1.public_key());
}

#[test]
fn masked_state_hides_private_cards() {
    let d = deck(&["As Kd", "7c 2h"], "2c 7h 9d Jc 3s");
    let s = GameState::new_hand(&GameConfig::default(), 0, &d).unwrap();
    let m = s.masked(0);
    assert!(m.is_masked());
    assert_eq!(m.hole_cards(0), s.hole_cards(0));
    assert_eq!(m.hole_cards(1), None);
    assert_eq!(m.infoset_key(0).unwrap(), s.infoset_key(0).unwrap());
    assert!(m.infoset_key(1).is_err());
    assert_eq!(m.legal_actions(), s.legal_actions());
    // Betting that stays on the street works; dealing the flop does not.
    let mut m2 = m.child(r(300)).unwrap();
    assert!(matches!(m2.apply(c()), Err(GameError::HiddenCards(_))));
    assert_eq!(m2.street(), 0);
    // A masked state taken after the flop shows the flop.
    let flop = s.child(c()).unwrap().child(c()).unwrap();
    assert_eq!(flop.masked(1).board(), flop.board());
}

// ----------------------------------------------------------------------
// Randomized invariants
// ----------------------------------------------------------------------

const RANDOM_HANDS: usize = if cfg!(debug_assertions) { 3_000 } else { 60_000 };

fn random_config(rng: &mut Rng) -> GameConfig {
    let n = 2 + rng.below(8) as usize;
    let bb = [2, 10, 100][rng.below(3) as usize];
    let sb = [bb / 2, bb, 1.max(bb / 3)][rng.below(3) as usize];
    let ante = if rng.below(4) == 0 { 1 + rng.below(bb as u64) as Chips } else { 0 };
    let stacks = (0..n)
        .map(|_| match rng.below(3) {
            0 => 1 + rng.below(3 * bb as u64) as Chips, // very short
            1 => 200 * bb,
            _ => 1 + rng.below(100 * bb as u64) as Chips,
        })
        .collect();
    GameConfig { num_players: n, stacks, small_blind: sb, big_blind: bb, ante }
}

/// Candidate actions around the legal boundaries.
fn candidates(l: &LegalActions, rng: &mut Rng) -> Vec<Action> {
    let mut v = vec![f(), c(), Action { amount: 1, ..f() }, Action { amount: 1, ..c() }];
    for x in [l.min_raise_to, l.max_raise_to, l.min_raise_to - 1, l.max_raise_to + 1, l.min_raise_to + 1, 0, 1] {
        v.push(r(x));
    }
    if l.max_raise_to > 0 {
        v.push(r(rng.below(2 * l.max_raise_to as u64 + 2) as Chips));
    }
    v
}

#[test]
fn random_hands_invariants() {
    let mut rng = Rng::new(2024);
    for hand in 0..RANDOM_HANDS {
        let config = random_config(&mut rng);
        let n = config.num_players;
        let total: Chips = config.stacks.iter().sum();
        let button = rng.below(n as u64) as usize;
        let mut deck = fresh_deck();
        rng.partial_shuffle(&mut deck, 52);
        let mut s = GameState::new_hand(&config, button, &deck).unwrap();
        let mut actions = Vec::new();
        loop {
            // Chip conservation and sanity.
            assert_eq!(s.stacks().iter().sum::<Chips>() + s.pot(), total, "hand {hand}: {s}");
            assert!(s.stacks().iter().all(|&x| x >= 0));
            for p in 0..n {
                assert_eq!(s.all_in()[p], s.stacks()[p] == 0, "{s}");
            }
            assert_eq!(s.board().len(), [0, 3, 4, 5][s.street()]);
            if s.is_terminal() {
                break;
            }
            let p = s.current_player().unwrap();
            assert!(!s.folded()[p] && !s.all_in()[p]);
            let l = s.legal_actions();
            assert!(l.can_fold != l.can_check);
            assert_eq!(l.can_raise(), l.max_raise_to > 0);
            if l.can_raise() {
                assert_eq!(l.max_raise_to, s.street_bets()[p] + s.stacks()[p]);
                assert!(l.max_raise_to > s.current_bet());
            }
            // apply accepts exactly what legal_actions allows.
            for a in candidates(&l, &mut rng) {
                let ok = s.child(a).is_ok();
                assert_eq!(ok, l.is_legal(a), "hand {hand}: {a} legal={} in {s}", l.is_legal(a));
            }
            let a = random_action(&s, &mut rng);
            s.apply(a).unwrap();
            actions.push(a);
        }
        let pay = s.payoffs().unwrap();
        assert_eq!(pay.iter().sum::<Chips>(), 0, "{s}");
        for p in 0..n {
            assert!(pay[p] >= -s.contributed()[p]);
            if s.folded()[p] {
                assert_eq!(pay[p], -s.contributed()[p]);
            }
            assert!(config.stacks[p] + pay[p] >= 0);
        }
        if s.num_active() >= 2 {
            assert_eq!(s.board().len(), 5);
        }
        // Determinism: same deck and actions give the same state.
        let mut t = GameState::new_hand(&config, button, &deck).unwrap();
        for &a in &actions {
            t.apply(a).unwrap();
        }
        assert_eq!(t, s);
        assert_eq!(t.payoffs().unwrap(), pay);
        let mut buf = [0; MAX_PLAYERS];
        t.payoffs_into(&mut buf).unwrap();
        assert_eq!(&buf[..n], &pay[..]);
    }
}
