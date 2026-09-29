//! Evaluator verification against a naive enumerating evaluator and against
//! the known hand-category distributions.

use std::collections::HashSet;

use poker_engine::cards::{card_from_str, card_to_str, cards_from_str, Card};
use poker_engine::eval::{self, evaluate, evaluate5, evaluate6, evaluate7, evaluate_masks, hand_category, naive};
use poker_engine::sim::{fresh_deck, Rng};

/// Random hands checked against the naive evaluator: 1M+ in release mode.
const RANDOM_7: usize = if cfg!(debug_assertions) { 50_000 } else { 1_200_000 };
const RANDOM_56: usize = if cfg!(debug_assertions) { 20_000 } else { 200_000 };

#[test]
fn random_7card_hands_match_naive() {
    let mut rng = Rng::new(7);
    let mut deck = fresh_deck();
    for i in 0..RANDOM_7 {
        rng.partial_shuffle(&mut deck, 7);
        let h: [Card; 7] = deck[..7].try_into().unwrap();
        let fast = evaluate7(&h);
        let slow = naive::evaluate_best(&h);
        assert_eq!(fast, slow, "hand #{i}: {:?}", h.map(|c| card_to_str(c).unwrap()));
    }
}

#[test]
fn random_5_and_6_card_hands_match_naive() {
    let mut rng = Rng::new(56);
    let mut deck = fresh_deck();
    for _ in 0..RANDOM_56 {
        rng.partial_shuffle(&mut deck, 6);
        let h6: [Card; 6] = deck[..6].try_into().unwrap();
        let h5: [Card; 5] = deck[..5].try_into().unwrap();
        assert_eq!(evaluate6(&h6), naive::evaluate_best(&h6));
        assert_eq!(evaluate5(&h5), naive::evaluate5(&h5));
    }
}

#[test]
fn random_pairwise_orderings_match_naive() {
    // Stress the comparison semantics, not just equality of values.
    let mut rng = Rng::new(99);
    let mut deck = fresh_deck();
    for _ in 0..RANDOM_56 {
        rng.partial_shuffle(&mut deck, 9);
        let a: [Card; 7] = [deck[0], deck[1], deck[4], deck[5], deck[6], deck[7], deck[8]];
        let b: [Card; 7] = [deck[2], deck[3], deck[4], deck[5], deck[6], deck[7], deck[8]];
        assert_eq!(
            evaluate7(&a).cmp(&evaluate7(&b)),
            naive::evaluate_best(&a).cmp(&naive::evaluate_best(&b))
        );
    }
}

#[test]
fn all_5card_hands_category_distribution() {
    let mut counts = [0u64; 9];
    let mut royal = 0u64;
    let mut distinct = HashSet::new();
    for a in 0..52u8 {
        for b in a + 1..52 {
            for c in b + 1..52 {
                for d in c + 1..52 {
                    for e in d + 1..52 {
                        let h = [a, b, c, d, e];
                        let r = evaluate5(&h);
                        let cat = hand_category(r) as usize;
                        counts[cat] += 1;
                        distinct.insert(r);
                        if cat == 8 && (r >> 16) & 0xF == 12 {
                            royal += 1;
                        }
                    }
                }
            }
        }
    }
    let expected = [1_302_540, 1_098_240, 123_552, 54_912, 10_200, 5_108, 3_744, 624, 40];
    assert_eq!(counts, expected);
    assert_eq!(counts.iter().sum::<u64>(), 2_598_960);
    assert_eq!(royal, 4);
    // The number of distinct 5-card hand values.
    assert_eq!(distinct.len(), 7_462);
}

/// All C(52,7) = 133,784,560 seven-card hands (release mode only; a few
/// seconds). Suit masks are built incrementally.
#[test]
#[cfg_attr(debug_assertions, ignore)]
fn all_7card_hands_category_distribution() {
    let mut counts = [0u64; 9];
    // Bitset over all possible rank values to count distinct values.
    let mut seen = vec![0u64; ((9 << 20) / 64) + 1];
    let add = |m: [u32; 4], c: u8| {
        let mut m = m;
        m[(c & 3) as usize] |= 1 << (c >> 2);
        m
    };
    for a in 0..52u8 {
        let ma = add([0; 4], a);
        for b in a + 1..52 {
            let mb = add(ma, b);
            for c in b + 1..52 {
                let mc = add(mb, c);
                for d in c + 1..52 {
                    let md = add(mc, d);
                    for e in d + 1..52 {
                        let me = add(md, e);
                        for f in e + 1..52 {
                            let mf = add(me, f);
                            for g in f + 1..52 {
                                let r = evaluate_masks(add(mf, g));
                                counts[hand_category(r) as usize] += 1;
                                seen[(r >> 6) as usize] |= 1 << (r & 63);
                            }
                        }
                    }
                }
            }
        }
    }
    let expected = [
        23_294_460, 58_627_800, 31_433_400, 6_461_620, 6_180_020, 4_047_644, 3_473_184, 224_848, 41_584,
    ];
    assert_eq!(counts, expected);
    assert_eq!(counts.iter().sum::<u64>(), 133_784_560);
    // Distinct 7-card hand values (best five of seven).
    let distinct: u32 = seen.iter().map(|w| w.count_ones()).sum();
    assert_eq!(distinct, 4_824);
}

#[test]
fn batch_matches_single() {
    let mut rng = Rng::new(3);
    let mut deck = fresh_deck();
    let n = 1000;
    let mut flat = Vec::with_capacity(n * 7);
    for _ in 0..n {
        rng.partial_shuffle(&mut deck, 7);
        flat.extend_from_slice(&deck[..7]);
    }
    let mut out = vec![0i32; n];
    eval::evaluate_batch(&flat, &mut out);
    for i in 0..n {
        assert_eq!(out[i] as u32, evaluate(&flat[7 * i..7 * i + 7]));
    }
}

#[test]
fn hand_written_rankings() {
    let ev = |s: &str| evaluate(&cards_from_str(s).unwrap());
    // Strictly increasing list of 7-card hands.
    let ladder = [
        "7c 5d 4h 3s 2c 9d Jh",  // J high
        "Ac Kd Qh Js 9c 3d 2h",  // A high
        "2c 2d 5h 7s 9c Jd Kh",  // pair of 2s
        "Ac Ad 5h 7s 9c Jd Kh",  // pair of aces
        "3c 3d 2h 2s 9c Jd Kh",  // threes up
        "Ac Ad Kh Ks 2c 3d 5h",  // aces up
        "7c 7d 7h 2s 9c Jd Kh",  // trip 7s
        "Ac 2d 3h 4s 5c Jd Kh",  // wheel
        "Tc Jd Qh Ks Ac 2d 3h",  // broadway
        "2h 4h 6h 8h Th Jd Kc",  // T-high flush
        "2h 4h 6h 8h Ah Jd Kc",  // A-high flush
        "2c 2d 2h 3s 3c Jd Kh",  // deuces full
        "Ac Ad Ah Ks Kc 2d 3h",  // aces full
        "2c 2d 2h 2s 3c 4d 5h",  // quad 2s
        "Ac Ad Ah As Kc 2d 3h",  // quad aces
        "Ah 2h 3h 4h 5h Kd Kc",  // steel wheel
        "Ah Kh Qh Jh Th 2d 2c",  // royal
    ];
    for w in ladder.windows(2) {
        assert!(ev(w[0]) < ev(w[1]), "{} !< {}", w[0], w[1]);
    }
    for (i, s) in ladder.iter().enumerate() {
        let expected_cat = [0, 0, 1, 1, 2, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8][i];
        assert_eq!(hand_category(ev(s)), expected_cat, "{s}");
    }
    // Board plays: identical ranks.
    assert_eq!(ev("2c 3d Ah Kh Qh Jh Th"), ev("4c 5d Ah Kh Qh Jh Th"));
    // Counterfeited two pair: kicker decides.
    assert!(ev("Ac 2d 7h 7s 5c 5d Kh") > ev("Qc 2d 7h 7s 5c 5d Kh"));
    assert_eq!(card_from_str("As").unwrap(), 51);
}
