//! 5/6/7-card hand evaluator.
//!
//! The evaluator works on per-suit 13-bit rank masks, so a single code path
//! handles 5, 6 and 7 cards with no large lookup table (two 8 KiB-entry
//! tables built at compile time). With at most 7 cards a flush excludes
//! quads and full houses, so the flush test comes first.
//!
//! # Rank encoding
//!
//! `rank = category << 20 | tiebreak`, where `tiebreak` holds up to five
//! card ranks (0..=12) in 4-bit nibbles at bit offsets 16, 12, 8, 4, 0:
//!
//! | category | tiebreak nibbles (most significant first) |
//! |---|---|
//! | 8 straight flush | top card of the straight (wheel = 3) |
//! | 7 quads | quad rank, kicker |
//! | 6 full house | trips rank, pair rank |
//! | 5 flush | five flush ranks, descending |
//! | 4 straight | top card of the straight (wheel = 3) |
//! | 3 trips | trips rank, two kickers |
//! | 2 two pair | high pair, low pair, kicker |
//! | 1 pair | pair rank, three kickers |
//! | 0 high card | five ranks, descending |
//!
//! Only integer comparisons between ranks are meaningful across engines.

use crate::cards::{rank, suit, Card};

/// A hand rank; higher is better.
pub type HandRank = u32;

pub const HIGH_CARD: u32 = 0;
pub const ONE_PAIR: u32 = 1;
pub const TWO_PAIR: u32 = 2;
pub const TRIPS: u32 = 3;
pub const STRAIGHT: u32 = 4;
pub const FLUSH: u32 = 5;
pub const FULL_HOUSE: u32 = 6;
pub const QUADS: u32 = 7;
pub const STRAIGHT_FLUSH: u32 = 8;

/// Bit offset of the category in a [`HandRank`].
pub const CATEGORY_SHIFT: u32 = 20;

/// Names of the hand categories, indexed by category.
pub const CATEGORY_NAMES: [&str; 9] = [
    "high card",
    "pair",
    "two pair",
    "trips",
    "straight",
    "flush",
    "full house",
    "quads",
    "straight flush",
];

/// Category (`0..=8`) of a hand rank.
#[inline(always)]
pub const fn hand_category(r: HandRank) -> u32 {
    r >> CATEGORY_SHIFT
}

const WHEEL: u32 = 0x100F; // A,5,4,3,2

/// For each 13-bit rank mask: 1 + rank of the top card of the best straight
/// contained in the mask, or 0 if none.
static STRAIGHT_TOP: [u8; 8192] = build_straight_table();

/// For each 13-bit rank mask: the (up to) five highest ranks packed in
/// nibbles at offsets 16, 12, 8, 4, 0.
static TOP5: [u32; 8192] = build_top5_table();

const fn build_straight_table() -> [u8; 8192] {
    let mut t = [0u8; 8192];
    let mut m = 0usize;
    while m < 8192 {
        let mut top: i32 = 12;
        let mut found = 0u8;
        while top >= 4 {
            let run = 0x1Fu32 << (top - 4);
            if (m as u32) & run == run {
                found = top as u8 + 1;
                break;
            }
            top -= 1;
        }
        if found == 0 && (m as u32) & WHEEL == WHEEL {
            found = 3 + 1;
        }
        t[m] = found;
        m += 1;
    }
    t
}

const fn build_top5_table() -> [u32; 8192] {
    let mut t = [0u32; 8192];
    let mut m = 0usize;
    while m < 8192 {
        let mut packed = 0u32;
        let mut shift: i32 = 16;
        let mut r: i32 = 12;
        while r >= 0 && shift >= 0 {
            if (m >> r) & 1 == 1 {
                packed |= (r as u32) << shift;
                shift -= 4;
            }
            r -= 1;
        }
        t[m] = packed;
        m += 1;
    }
    t
}

#[inline(always)]
fn hi_bit(m: u32) -> u32 {
    31 - m.leading_zeros()
}

/// Evaluate a hand given its four per-suit rank masks (bit `r` of
/// `masks[s]` set when the card of rank `r` and suit `s` is present).
/// Valid for 5, 6 or 7 distinct cards.
#[inline]
pub fn evaluate_masks(masks: [u32; 4]) -> HandRank {
    let [a, b, c, d] = masks;
    // Flush (at most one suit can hold 5+ of 7 cards; a flush rules out
    // quads and full houses with 7 or fewer cards).
    for m in masks {
        if m.count_ones() >= 5 {
            let st = STRAIGHT_TOP[m as usize] as u32;
            if st != 0 {
                return (STRAIGHT_FLUSH << CATEGORY_SHIFT) | ((st - 1) << 16);
            }
            return (FLUSH << CATEGORY_SHIFT) | TOP5[m as usize];
        }
    }
    let all = a | b | c | d;
    let quads = a & b & c & d;
    if quads != 0 {
        let q = hi_bit(quads);
        let rest = all & !(1 << q);
        return (QUADS << CATEGORY_SHIFT) | (q << 16) | (hi_bit(rest) << 12);
    }
    let ge3 = (a & b & c) | (a & b & d) | (a & c & d) | (b & c & d);
    let ge2 = (a & b) | (a & c) | (a & d) | (b & c) | (b & d) | (c & d);
    if ge3 != 0 {
        let t = hi_bit(ge3);
        let pairs = ge2 & !(1 << t);
        if pairs != 0 {
            return (FULL_HOUSE << CATEGORY_SHIFT) | (t << 16) | (hi_bit(pairs) << 12);
        }
    }
    let st = STRAIGHT_TOP[all as usize] as u32;
    if st != 0 {
        return (STRAIGHT << CATEGORY_SHIFT) | ((st - 1) << 16);
    }
    if ge3 != 0 {
        let t = hi_bit(ge3);
        let kick = all & !(1 << t);
        return (TRIPS << CATEGORY_SHIFT) | (t << 16) | ((TOP5[kick as usize] & 0xFF000) >> 4);
    }
    if ge2 != 0 {
        let p1 = hi_bit(ge2);
        let rest_pairs = ge2 & !(1 << p1);
        if rest_pairs != 0 {
            let p2 = hi_bit(rest_pairs);
            let kick = all & !((1 << p1) | (1 << p2));
            return (TWO_PAIR << CATEGORY_SHIFT) | (p1 << 16) | (p2 << 12) | (hi_bit(kick) << 8);
        }
        let kick = all & !(1 << p1);
        return (ONE_PAIR << CATEGORY_SHIFT) | (p1 << 16) | ((TOP5[kick as usize] & 0xFFF00) >> 4);
    }
    (HIGH_CARD << CATEGORY_SHIFT) | TOP5[all as usize]
}

#[inline(always)]
fn masks_of(cards: &[Card]) -> [u32; 4] {
    let mut m = [0u32; 4];
    for &c in cards {
        m[suit(c) as usize] |= 1 << rank(c);
    }
    m
}

/// Evaluate 5 to 7 distinct cards (no validation; cards must be in `0..52`
/// and distinct).
#[inline]
pub fn evaluate(cards: &[Card]) -> HandRank {
    debug_assert!((5..=7).contains(&cards.len()));
    evaluate_masks(masks_of(cards))
}

/// Evaluate exactly 5 cards.
#[inline]
pub fn evaluate5(cards: &[Card; 5]) -> HandRank {
    evaluate(cards)
}

/// Evaluate exactly 6 cards (best 5 of 6).
#[inline]
pub fn evaluate6(cards: &[Card; 6]) -> HandRank {
    evaluate(cards)
}

/// Evaluate exactly 7 cards (best 5 of 7).
#[inline]
pub fn evaluate7(cards: &[Card; 7]) -> HandRank {
    evaluate(cards)
}

/// Evaluate `hole` (2 cards) plus a 5-card board.
#[inline]
pub fn evaluate_hole_board(hole: [Card; 2], board: &[Card]) -> HandRank {
    let mut m = masks_of(board);
    for c in hole {
        m[suit(c) as usize] |= 1 << rank(c);
    }
    evaluate_masks(m)
}

/// Evaluate a batch of 7-card hands stored row-major (`cards.len() == 7 * n`).
pub fn evaluate_batch(cards: &[Card], out: &mut [i32]) {
    assert_eq!(cards.len(), out.len() * 7);
    for (row, o) in cards.chunks_exact(7).zip(out.iter_mut()) {
        *o = evaluate(row) as i32;
    }
}

/// A deliberately simple (slow) evaluator used to verify the fast one.
/// It shares the rank encoding, so results compare for exact equality.
pub mod naive {
    use super::*;

    /// Evaluate exactly five cards by sorting and counting.
    pub fn evaluate5(cards: &[Card; 5]) -> HandRank {
        let mut counts = [0u8; 13];
        for &c in cards {
            counts[rank(c) as usize] += 1;
        }
        let flush = cards.iter().all(|&c| suit(c) == suit(cards[0]));
        // groups: (count, rank), sorted by count desc then rank desc
        let mut groups: Vec<(u8, u8)> = (0..13u8)
            .filter(|&r| counts[r as usize] > 0)
            .map(|r| (counts[r as usize], r))
            .collect();
        groups.sort_by(|x, y| y.cmp(x));
        let mut straight_top: Option<u32> = None;
        if groups.len() == 5 {
            let hi = groups[0].1 as u32;
            let lo = groups[4].1 as u32;
            if hi - lo == 4 {
                straight_top = Some(hi);
            } else if groups.iter().map(|g| g.1).collect::<Vec<_>>() == vec![12, 3, 2, 1, 0] {
                straight_top = Some(3);
            }
        }
        let pattern: Vec<u8> = groups.iter().map(|g| g.0).collect();
        let category = if straight_top.is_some() && flush {
            STRAIGHT_FLUSH
        } else if pattern == [4, 1] {
            QUADS
        } else if pattern == [3, 2] {
            FULL_HOUSE
        } else if flush {
            FLUSH
        } else if straight_top.is_some() {
            STRAIGHT
        } else if pattern == [3, 1, 1] {
            TRIPS
        } else if pattern == [2, 2, 1] {
            TWO_PAIR
        } else if pattern == [2, 1, 1, 1] {
            ONE_PAIR
        } else {
            HIGH_CARD
        };
        let tiebreak = match straight_top {
            Some(top) if category == STRAIGHT || category == STRAIGHT_FLUSH => top << 16,
            _ => groups
                .iter()
                .enumerate()
                .map(|(i, g)| (g.1 as u32) << (16 - 4 * i as u32))
                .fold(0, |acc, x| acc | x),
        };
        (category << CATEGORY_SHIFT) | tiebreak
    }

    /// Best five-card hand out of `cards` (5..=7 cards) by enumeration.
    pub fn evaluate_best(cards: &[Card]) -> HandRank {
        let n = cards.len();
        assert!((5..=7).contains(&n));
        let mut best = 0;
        for i0 in 0..n {
            for i1 in i0 + 1..n {
                for i2 in i1 + 1..n {
                    for i3 in i2 + 1..n {
                        for i4 in i3 + 1..n {
                            let h = [cards[i0], cards[i1], cards[i2], cards[i3], cards[i4]];
                            best = best.max(evaluate5(&h));
                        }
                    }
                }
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::cards_from_str;

    fn ev(s: &str) -> HandRank {
        evaluate(&cards_from_str(s).unwrap())
    }

    #[test]
    fn categories() {
        assert_eq!(hand_category(ev("As Ks Qs Js Ts")), STRAIGHT_FLUSH);
        assert_eq!(hand_category(ev("As 2s 3s 4s 5s")), STRAIGHT_FLUSH);
        assert_eq!(hand_category(ev("As Ad Ah Ac 2s")), QUADS);
        assert_eq!(hand_category(ev("As Ad Ah Kc Ks")), FULL_HOUSE);
        assert_eq!(hand_category(ev("As 9s 3s 4s 5s")), FLUSH);
        assert_eq!(hand_category(ev("Ad 2s 3s 4s 5s")), STRAIGHT);
        assert_eq!(hand_category(ev("Ad As 3s Ac 5s")), TRIPS);
        assert_eq!(hand_category(ev("Ad As 3s 3c 5s")), TWO_PAIR);
        assert_eq!(hand_category(ev("Ad As 3s 4c 5s")), ONE_PAIR);
        assert_eq!(hand_category(ev("Ad Ks 3s 4c 5s")), HIGH_CARD);
    }

    #[test]
    fn orderings() {
        // wheel is the lowest straight
        assert!(ev("Ad 2s 3s 4c 5s") < ev("2d 3s 4s 5c 6s"));
        assert!(ev("As 2s 3s 4s 5s") < ev("2h 3h 4h 5h 6h"));
        // 7-card: best five chosen
        assert_eq!(ev("As Ks Qs Js Ts 9s 8s"), ev("Ah Kh Qh Jh Th"));
        // two trips make a full house with the higher trips
        assert_eq!(hand_category(ev("Ks Kd Kh 7c 7d 7h 2c")), FULL_HOUSE);
        assert!(ev("Ks Kd Kh 7c 7d 7h 2c") > ev("Qs Qd Qh Ac Ad 3h 2c"));
        // three pairs: best two plus best kicker
        assert_eq!(ev("Ks Kd 7h 7c 5d 5h Ac"), ev("Ks Kd 7h 7c Ac"));
        assert_eq!(ev("Ks Kd 7h 7c 5d 5h 2c"), ev("Ks Kd 7h 7c 5d"));
        // kicker matters
        assert!(ev("As Ad Kh 4c 3d") > ev("Ah Ac Qh Jc 9d"));
        // flush beats straight in 7 cards
        assert_eq!(hand_category(ev("2s 7s 9s Js Ks Td 8h")), FLUSH);
        // straight flush inside a 7-card flush
        assert_eq!(hand_category(ev("2s 3s 4s 5s 6s Ks As")), STRAIGHT_FLUSH);
        assert_eq!(ev("2s 3s 4s 5s 6s Ks As"), ev("2h 3h 4h 5h 6h"));
        // six-card straight picks the highest run
        assert_eq!(ev("Ah 2s 3s 4d 5c 6c"), ev("2h 3s 4s 5d 6c"));
    }

    #[test]
    fn naive_agrees_on_examples() {
        for s in [
            "As Ks Qs Js Ts 9s 8s",
            "Ks Kd Kh 7c 7d 7h 2c",
            "Ks Kd 7h 7c 5d 5h Ac",
            "2s 3s 4s 5s 6s Ks As",
            "Ah 2s 3s 4d 5c 6c Kd",
            "Ah 2s 3s 4d 9c Tc Kd",
        ] {
            let c = cards_from_str(s).unwrap();
            assert_eq!(evaluate(&c), naive::evaluate_best(&c), "{s}");
        }
    }
}
