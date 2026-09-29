//! Suit-isomorphic canonical hand indexing.
//!
//! Two (hole cards, board) pairs are *isomorphic* when one becomes the other
//! by a permutation of the four suits (and by reordering cards within the
//! hole cards or within the board). They are strategically identical, so card
//! abstractions only need one entry per class. [`canonical_index`] maps a
//! hand to a dense index in `0..canonical_size(street)`, identical for all
//! isomorphic hands; [`canonical_unindex`] returns the canonical
//! representative of an index. Class counts:
//!
//! | street | cards | classes |
//! |---|---|---|
//! | 0 preflop | 2 | 169 |
//! | 1 flop | 2 + 3 | 1,286,792 |
//! | 2 turn | 2 + 3 + 1 | 13,960,050 |
//! | 3 river | 2 + 3 + 1 + 1 | 123,156,254 |
//!
//! # The indexing (so other code can build matching tables)
//!
//! Cards are grouped into two *rounds*: round 0 is the hole cards, round 1
//! the whole board (0, 3, 4 or 5 cards, unordered). The order in which board
//! cards arrived does not matter for a per-street bucket, and treating the
//! board as one set gives the standard class counts above (keeping flop,
//! turn and river as separate rounds would give 55,190,538 turn and
//! 2,428,287,420 river classes). For each suit `u` and round `r` let
//! `M[u][r]` be the 13-bit set of ranks of suit `u` dealt in round `r`, and
//! `c[u][r] = |M[u][r]|`.
//!
//! 1. **Suit index.** The ranks of one suit are indexed round by round in
//!    mixed radix, round 0 least significant. For round `r` the set
//!    `M[u][r]` is written relative to the ranks of that suit not used by
//!    earlier rounds (`avail = 13 - sum_{r'<r} c[u][r']` positions, in
//!    increasing rank order) and indexed in colexicographic order
//!    (`sum_i C(p_i, i)` over its compressed positions `p_1 < p_2 < ...`,
//!    `i` from 1). The radix of round `r` is `C(avail, c[u][r])`.
//! 2. **Suit order.** Each suit gets the pair `(v[u], idx[u])` where `v[u]`
//!    packs the counts `c[u][0..R]` as 4-bit digits with round 0 most
//!    significant. Suits are sorted by `(v, idx)` in *descending* order.
//! 3. **Configuration.** The sorted tuple `(v[0], v[1], v[2], v[3])` is the
//!    configuration. All configurations of a street are enumerated once and
//!    sorted by the packed 64-bit key `v0 << 48 | v1 << 32 | v2 << 16 | v3`
//!    (ascending); `offset[k]` is the number of hands in configurations
//!    before `k`.
//! 4. **Groups.** Within a configuration, consecutive suits with equal `v`
//!    form a group of `k` suits, each with a suit index in `0..n(v)`, where
//!    `n(v) = prod_r C(avail_r, c_r)`. The group's `k` indices, sorted
//!    descending `x_0 >= x_1 >= ... >= x_{k-1}`, form a multiset indexed as
//!    `sum_j C(x_j + k - 1 - j, k - j)` in `0..C(n(v) + k - 1, k)`.
//! 5. **Index.** Group indices are combined in mixed radix, first group
//!    least significant, and added to `offset[configuration]`.
//!
//! [`canonical_unindex`] inverts this and deals the sorted suits as clubs,
//! diamonds, hearts, spades in order, with the cards of each round sorted
//! ascending. The preflop lossless 169-class grid [`preflop_class`] is a
//! separate, human-readable bijection used for preflop buckets.

use std::collections::HashMap;
use std::sync::OnceLock;

use crate::cards::{make_card, rank, suit, Card};

/// Cards per round for each street: the hole cards, then the whole board.
pub const ROUNDS: [&[u8]; 4] = [&[2], &[2, 3], &[2, 4], &[2, 5]];

/// Number of canonical classes per street.
pub const CANONICAL_SIZES: [u64; 4] = [169, 1_286_792, 13_960_050, 123_156_254];

/// `C(n, k)` for `n, k <= 13`.
const fn binom13_table() -> [[u32; 14]; 14] {
    let mut t = [[0u32; 14]; 14];
    let mut n = 0;
    while n < 14 {
        t[n][0] = 1;
        let mut k = 1;
        while k <= n {
            t[n][k] = t[n - 1][k - 1] + if k < n { t[n - 1][k] } else { 0 };
            k += 1;
        }
        n += 1;
    }
    t
}

static BINOM13: [[u32; 14]; 14] = binom13_table();

#[inline]
fn c13(n: u32, k: u32) -> u64 {
    if k > n {
        0
    } else {
        BINOM13[n as usize][k as usize] as u64
    }
}

/// `C(n, k)` for small `k`; saturates on overflow (never hit for valid inputs).
fn binom(n: u64, k: u64) -> u64 {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut r: u128 = 1;
    for i in 0..k {
        r = r * (n - i) as u128 / (i + 1) as u128;
    }
    r.min(u64::MAX as u128) as u64
}

/// Colex index of a set of compressed positions given as a bitmask.
#[inline]
fn colex(mask: u32) -> u64 {
    let mut m = mask;
    let mut i = 1;
    let mut idx = 0;
    while m != 0 {
        let p = m.trailing_zeros();
        idx += c13(p, i);
        i += 1;
        m &= m - 1;
    }
    idx
}

/// Inverse of [`colex`] for a `k`-subset of `0..n` (n <= 13).
fn uncolex(mut idx: u64, k: u32, n: u32) -> u32 {
    let mut mask = 0u32;
    let mut hi = n;
    for i in (1..=k).rev() {
        // Largest p < hi with C(p, i) <= idx.
        let mut p = hi - 1;
        while c13(p, i) > idx {
            p -= 1;
        }
        idx -= c13(p, i);
        mask |= 1 << p;
        hi = p;
    }
    mask
}

/// Positions of the ranks in `mask` relative to the ranks not in `used`.
#[inline]
fn compress(mask: u32, used: u32) -> u32 {
    let mut out = 0u32;
    let mut m = mask;
    while m != 0 {
        let r = m.trailing_zeros();
        let below = (used & ((1u32 << r) - 1)).count_ones();
        out |= 1 << (r - below);
        m &= m - 1;
    }
    out
}

/// Inverse of [`compress`]: map compressed positions back to ranks.
fn expand(cmask: u32, used: u32) -> u32 {
    let mut out = 0u32;
    let mut pos = 0u32;
    for r in 0..13u32 {
        if used & (1 << r) != 0 {
            continue;
        }
        if cmask & (1 << pos) != 0 {
            out |= 1 << r;
        }
        pos += 1;
    }
    out
}

/// Count digit of round `r` in a packed count vector.
#[inline]
fn digit(v: u16, r: usize) -> u32 {
    ((v >> (4 * (3 - r))) & 0xF) as u32
}

/// Number of rank-set sequences of one suit with count vector `v`.
fn suit_size(v: u16, rounds: usize) -> u64 {
    let mut avail = 13;
    let mut n = 1u64;
    for r in 0..rounds {
        let c = digit(v, r);
        n *= c13(avail, c);
        avail -= c;
    }
    n
}

#[derive(Clone, Debug)]
struct Config {
    /// Sorted (descending) suit count vectors.
    v: [u16; 4],
    /// Groups of equal `v`: (start, len, suit_size).
    groups: Vec<(usize, usize, u64)>,
    /// Multiset size of each group.
    group_sizes: Vec<u64>,
}

/// Canonical indexer for one street.
pub struct Indexer {
    street: usize,
    rounds: usize,
    configs: Vec<Config>,
    offsets: Vec<u64>,
    lookup: HashMap<u64, usize>,
}

fn config_key(v: &[u16; 4]) -> u64 {
    ((v[0] as u64) << 48) | ((v[1] as u64) << 32) | ((v[2] as u64) << 16) | v[3] as u64
}

/// All ways to split `n` cards into 4 suits.
fn compositions(n: u8) -> Vec<[u8; 4]> {
    let mut out = Vec::new();
    for a in 0..=n {
        for b in 0..=n - a {
            for c in 0..=n - a - b {
                out.push([a, b, c, n - a - b - c]);
            }
        }
    }
    out
}

impl Indexer {
    pub fn new(street: usize) -> Indexer {
        assert!(street < 4);
        let rounds = ROUNDS[street].len();
        let per_round: Vec<Vec<[u8; 4]>> = ROUNDS[street].iter().map(|&n| compositions(n)).collect();
        let mut keys = std::collections::BTreeSet::new();
        // Cartesian product over rounds.
        let mut stack: Vec<([u16; 4], usize)> = vec![([0; 4], 0)];
        while let Some((v, r)) = stack.pop() {
            if r == rounds {
                let mut s = v;
                s.sort_unstable_by(|a, b| b.cmp(a));
                keys.insert(config_key(&s));
                continue;
            }
            for comp in &per_round[r] {
                let mut w = v;
                for u in 0..4 {
                    w[u] |= (comp[u] as u16) << (4 * (3 - r));
                }
                stack.push((w, r + 1));
            }
        }
        let mut configs = Vec::with_capacity(keys.len());
        let mut offsets = Vec::with_capacity(keys.len() + 1);
        let mut lookup = HashMap::new();
        let mut total = 0u64;
        for (i, key) in keys.iter().enumerate() {
            let v = [(key >> 48) as u16, (key >> 32) as u16, (key >> 16) as u16, *key as u16];
            let mut groups = Vec::new();
            let mut group_sizes = Vec::new();
            let mut start = 0;
            while start < 4 {
                let mut end = start + 1;
                while end < 4 && v[end] == v[start] {
                    end += 1;
                }
                let k = (end - start) as u64;
                let n = suit_size(v[start], rounds);
                groups.push((start, end - start, n));
                group_sizes.push(binom(n + k - 1, k));
                start = end;
            }
            let size: u64 = group_sizes.iter().product();
            offsets.push(total);
            total += size;
            lookup.insert(*key, i);
            configs.push(Config { v, groups, group_sizes });
        }
        offsets.push(total);
        debug_assert_eq!(total, CANONICAL_SIZES[street]);
        Indexer { street, rounds, configs, offsets, lookup }
    }

    /// Number of canonical classes.
    pub fn size(&self) -> u64 {
        *self.offsets.last().unwrap()
    }

    /// Canonical index of `cards`: the 2 hole cards, then the board. Cards must be
    /// distinct and in `0..52` (not validated).
    pub fn index(&self, cards: &[Card]) -> u64 {
        debug_assert_eq!(cards.len(), ROUNDS[self.street].iter().map(|&n| n as usize).sum::<usize>());
        let mut masks = [[0u32; 4]; 4]; // [suit][round]
        let mut pos = 0;
        for (r, &n) in ROUNDS[self.street].iter().enumerate() {
            for _ in 0..n {
                let c = cards[pos];
                masks[suit(c) as usize][r] |= 1 << rank(c);
                pos += 1;
            }
        }
        let mut suits = [(0u16, 0u64); 4];
        for u in 0..4 {
            let mut v = 0u16;
            let mut idx = 0u64;
            let mut mult = 1u64;
            let mut used = 0u32;
            for r in 0..self.rounds {
                let m = masks[u][r];
                let c = m.count_ones();
                v |= (c as u16) << (4 * (3 - r));
                let avail = 13 - used.count_ones();
                idx += colex(compress(m, used)) * mult;
                mult *= c13(avail, c);
                used |= m;
            }
            suits[u] = (v, idx);
        }
        suits.sort_unstable_by(|a, b| b.cmp(a));
        let v = [suits[0].0, suits[1].0, suits[2].0, suits[3].0];
        let ci = self.lookup[&config_key(&v)];
        let cfg = &self.configs[ci];
        let mut total = 0u64;
        let mut mult = 1u64;
        for (g, &(start, k, _n)) in cfg.groups.iter().enumerate() {
            // suits[start..start+k] are sorted descending by idx already.
            let mut gi = 0u64;
            for j in 0..k {
                let x = suits[start + j].1;
                gi += binom(x + (k - 1 - j) as u64, (k - j) as u64);
            }
            total += gi * mult;
            mult *= cfg.group_sizes[g];
        }
        self.offsets[ci] + total
    }

    /// Canonical representative of `index`: the 2 hole cards, then the
    /// board, each sorted ascending.
    pub fn unindex(&self, index: u64) -> Option<Vec<Card>> {
        if index >= self.size() {
            return None;
        }
        let ci = match self.offsets.binary_search(&index) {
            Ok(i) => i,
            Err(i) => i - 1,
        };
        let cfg = &self.configs[ci];
        let mut rem = index - self.offsets[ci];
        let mut suit_idx = [0u64; 4];
        for (g, &(start, k, n)) in cfg.groups.iter().enumerate() {
            let size = cfg.group_sizes[g];
            let mut gi = rem % size;
            rem /= size;
            let mut hi = n + k as u64 - 1; // y_0 <= n + k - 2, search below hi
            for j in 0..k {
                let m = (k - j) as u64;
                // Largest y < hi with C(y, m) <= gi.
                let mut lo_y = m - 1;
                let mut hi_y = hi - 1;
                while lo_y < hi_y {
                    let mid = (lo_y + hi_y).div_ceil(2);
                    if binom(mid, m) <= gi {
                        lo_y = mid;
                    } else {
                        hi_y = mid - 1;
                    }
                }
                gi -= binom(lo_y, m);
                suit_idx[start + j] = lo_y - (k - 1 - j) as u64;
                hi = lo_y;
            }
        }
        let mut by_round: Vec<Vec<Card>> = vec![Vec::new(); self.rounds];
        for u in 0..4 {
            let v = cfg.v[u];
            let mut idx = suit_idx[u];
            let mut used = 0u32;
            for r in 0..self.rounds {
                let c = digit(v, r);
                let avail = 13 - used.count_ones();
                let radix = c13(avail, c);
                let ri = idx % radix;
                idx /= radix;
                let ranks = expand(uncolex(ri, c, avail), used);
                used |= ranks;
                let mut m = ranks;
                while m != 0 {
                    let rk = m.trailing_zeros();
                    by_round[r].push(make_card(rk as u8, u as u8));
                    m &= m - 1;
                }
            }
        }
        let mut out = Vec::with_capacity(7);
        for mut cards in by_round {
            cards.sort_unstable();
            out.extend(cards);
        }
        Some(out)
    }
}

static INDEXERS: [OnceLock<Indexer>; 4] = [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()];

/// The shared indexer of `street` (built on first use).
pub fn indexer(street: usize) -> &'static Indexer {
    INDEXERS[street].get_or_init(|| Indexer::new(street))
}

/// Number of cards on the board for `street`.
pub const fn board_len(street: usize) -> usize {
    [0, 3, 4, 5][street]
}

/// Canonical (suit-isomorphic) index of `hole` + `board` on `street`, in
/// `0..CANONICAL_SIZES[street]`. `board` must hold exactly the cards of the
/// street (0, 3, 4 or 5).
pub fn canonical_index(street: usize, hole: [Card; 2], board: &[Card]) -> u32 {
    let mut cards = [0u8; 7];
    cards[0] = hole[0];
    cards[1] = hole[1];
    let nb = board_len(street);
    cards[2..2 + nb].copy_from_slice(&board[..nb]);
    indexer(street).index(&cards[..2 + nb]) as u32
}

/// Canonical representative `(hole, board)` of an index.
pub fn canonical_unindex(street: usize, index: u32) -> Option<([Card; 2], Vec<Card>)> {
    let cards = indexer(street).unindex(index as u64)?;
    Some(([cards[0], cards[1]], cards[2..].to_vec()))
}

/// Lossless preflop class in `0..169` on a 13x13 grid: pairs `r * 13 + r`,
/// suited `hi * 13 + lo`, offsuit `lo * 13 + hi` (`hi > lo` ranks, 0 = deuce).
/// So `AA = 168`, `AKs = 12 * 13 + 11 = 167`, `AKo = 11 * 13 + 12 = 155`,
/// `22 = 0`, `32o = 0 * 13 + 1 = 1`, `32s = 1 * 13 + 0 = 13`.
pub fn preflop_class(hole: [Card; 2]) -> u32 {
    let (r0, r1) = (rank(hole[0]) as u32, rank(hole[1]) as u32);
    let (hi, lo) = if r0 >= r1 { (r0, r1) } else { (r1, r0) };
    if hi == lo || suit(hole[0]) != suit(hole[1]) {
        lo * 13 + hi
    } else {
        hi * 13 + lo
    }
}

/// Number of hole-card combinations in a preflop class (6 pairs, 4 suited, 12 offsuit).
pub fn preflop_class_combos(class: u32) -> u32 {
    let (a, b) = (class / 13, class % 13);
    if a == b {
        6
    } else if a > b {
        4
    } else {
        12
    }
}

/// A representative hole-card pair for a preflop class.
pub fn preflop_class_hand(class: u32) -> [Card; 2] {
    let (a, b) = ((class / 13) as u8, (class % 13) as u8);
    if a == b {
        [make_card(a, 0), make_card(a, 1)]
    } else if a > b {
        [make_card(a, 0), make_card(b, 0)]
    } else {
        [make_card(b, 0), make_card(a, 1)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sim::{fresh_deck, Rng};

    #[test]
    fn sizes_match_known_counts() {
        for s in 0..4 {
            assert_eq!(indexer(s).size(), CANONICAL_SIZES[s], "street {s}");
        }
    }

    #[test]
    fn preflop_classes() {
        let mut seen = std::collections::HashSet::new();
        let mut combos = 0;
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                let c = preflop_class([a, b]);
                assert!(c < 169);
                assert_eq!(c, preflop_class([b, a]));
                seen.insert(c);
                combos += 1;
                let rep = preflop_class_hand(c);
                assert_eq!(preflop_class(rep), c);
            }
        }
        assert_eq!(seen.len(), 169);
        assert_eq!(combos, 1326);
        assert_eq!((0..169).map(preflop_class_combos).sum::<u32>(), 1326);
        let ac = crate::cards::cards_from_str("AsAh").unwrap();
        assert_eq!(preflop_class([ac[0], ac[1]]), 168);
        // Preflop canonical index is also a bijection onto 0..169.
        let mut idx = std::collections::HashSet::new();
        for a in 0..52u8 {
            for b in (a + 1)..52u8 {
                idx.insert(canonical_index(0, [a, b], &[]));
            }
        }
        assert_eq!(idx.len(), 169);
    }

    #[test]
    fn index_invariant_and_roundtrip() {
        let mut rng = Rng::new(7);
        let mut deck = fresh_deck();
        for street in 0..4 {
            let ix = indexer(street);
            let nb = board_len(street);
            for _ in 0..3000 {
                rng.partial_shuffle(&mut deck, 7);
                let hole = [deck[0], deck[1]];
                let board = &deck[2..2 + nb];
                let i = canonical_index(street, hole, board);
                assert!((i as u64) < ix.size());
                // Random suit permutation and in-round reordering.
                let mut perm = [0u8, 1, 2, 3];
                rng.partial_shuffle(&mut perm, 4);
                let map = |c: Card| make_card(rank(c), perm[suit(c) as usize]);
                let mut b2: Vec<Card> = board.iter().map(|&c| map(c)).collect();
                b2.reverse();
                let j = canonical_index(street, [map(hole[1]), map(hole[0])], &b2);
                assert_eq!(i, j);
                let (h, b) = canonical_unindex(street, i).unwrap();
                assert_eq!(canonical_index(street, h, &b), i);
            }
            // Random indices round-trip.
            for _ in 0..3000 {
                let i = rng.below(ix.size()) as u32;
                let (h, b) = canonical_unindex(street, i).unwrap();
                let mut all = vec![h[0], h[1]];
                all.extend(&b);
                crate::cards::validate_cards(&all).unwrap();
                assert_eq!(canonical_index(street, h, &b), i);
            }
        }
    }

    #[test]
    fn flop_index_is_a_bijection_on_a_prefix() {
        // Every one of the first 20,000 flop indices decodes to a distinct hand
        // that indexes back to itself.
        let ix = indexer(1);
        for i in 0..20_000u64 {
            let c = ix.unindex(i).unwrap();
            assert_eq!(ix.index(&c), i);
        }
        assert!(ix.unindex(ix.size()).is_none());
    }
}
