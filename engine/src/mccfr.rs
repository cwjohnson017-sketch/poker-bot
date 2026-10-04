//! Tabular external-sampling MCCFR with Linear CFR discounting and
//! Pluribus-style regret pruning.
//!
//! One *iteration* runs one traversal per player. A traversal samples a full
//! deal up front (every chance node, as external sampling prescribes), then
//! walks the abstract betting tree from the root with [`GameState`] from this
//! crate, so the solver plays exactly the engine's rules:
//!
//! * at the traverser's decisions every abstract action is explored, the
//!   node value is `v = sum_a sigma(a) v(a)` and the regret of each explored
//!   action grows by `v(a) - v`;
//! * at the other player's decisions one action is sampled from the current
//!   strategy `sigma` (regret matching), and `sigma` is added to that
//!   infoset's strategy sum. The sampled player's own reach is exactly
//!   accounted for by the sampling, so this is the unbiased
//!   "simple averaging" of external sampling (Lanctot et al. 2009);
//! * `sigma` is regret matching on the positive part of the regrets (uniform
//!   when none is positive). Negative regrets are kept (no CFR+ reset),
//!   bounded below by `regret_floor`.
//!
//! **Linear CFR.** Pluribus-style: every `lcfr_discount_every` iterations,
//! until `lcfr_stop`, all regrets and strategy sums are multiplied by
//! `k / (k + 1)` where `k` is the number of intervals so far. Iterations in
//! interval `j` then carry weight proportional to `j`, which is Linear CFR
//! at interval granularity while keeping the f32 tables bounded.
//!
//! **Pruning.** After `prune_start` iterations, a traversal is run with
//! pruning with probability `prune_prob`: at the traverser's nodes, actions
//! whose regret is below `prune_threshold` (and whose current probability is
//! zero) are not explored, except on the river and except actions that end
//! the hand. The other traversals explore everything, so pruned actions can
//! recover.
//!
//! **Infoset keys** are 128-bit: bits 126..127 street, bits 97..125 the
//! acting player's bucket on that street, bits 0..96 the abstract betting
//! sequence of the whole hand: a leading 1 bit followed by one 4-bit token
//! per action (the action's index in its street's abstract list), oldest
//! first. Seats are not stored: the sequence determines who acts, so keys
//! are relative to position (small blind / big blind).

use std::cmp::Ordering as CmpOrdering;
use std::collections::HashMap;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::{BufReader, BufWriter, Read, Write};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::abstraction::{
    AbstractAction, ActionAbstraction, ActionList, Bucketer, CardAbstraction, CardAbstractionSpec, When,
};
use crate::cards::Card;
use crate::eval::{evaluate_hole_board, HandRank};
use crate::game::{Chips, GameConfig, GameState, MAX_PLAYERS};
use crate::sim::{fresh_deck, Rng};

// ----------------------------------------------------------------------
// Keys
// ----------------------------------------------------------------------

/// Betting-sequence accumulator: a leading 1 bit then 4 bits per action.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Seq(pub u128);

/// Maximum actions per hand a key can hold.
pub const MAX_SEQ_LEN: usize = 24;
const SEQ_BITS: u32 = 97;
const BUCKET_SHIFT: u32 = 97;
const BUCKET_BITS: u32 = 29;
const STREET_SHIFT: u32 = 126;

impl Seq {
    pub const EMPTY: Seq = Seq(1);
    #[inline]
    pub fn push(self, idx: u8) -> Seq {
        debug_assert!(idx < 16);
        debug_assert!(self.0 < (1u128 << (SEQ_BITS - 4)), "betting sequence too long for the key");
        Seq((self.0 << 4) | idx as u128)
    }
    pub fn from_indices(idx: &[u8]) -> Result<Seq, String> {
        if idx.len() > MAX_SEQ_LEN {
            return Err(format!("betting sequence of {} actions exceeds {MAX_SEQ_LEN}", idx.len()));
        }
        let mut s = Seq::EMPTY;
        for &i in idx {
            if i >= 16 {
                return Err(format!("abstract index {i} >= 16"));
            }
            s = s.push(i);
        }
        Ok(s)
    }
    pub fn indices(self) -> Vec<u8> {
        let mut v = Vec::new();
        let mut x = self.0;
        while x > 1 {
            v.push((x & 0xF) as u8);
            x >>= 4;
        }
        v.reverse();
        v
    }
}

/// Infoset key: street, the acting player's bucket, betting sequence.
#[inline]
pub fn make_key(street: usize, bucket: u32, seq: Seq) -> u128 {
    debug_assert!(bucket < (1 << BUCKET_BITS));
    seq.0 | ((bucket as u128) << BUCKET_SHIFT) | ((street as u128) << STREET_SHIFT)
}

/// Inverse of [`make_key`].
pub fn decode_key(key: u128) -> (usize, u32, Seq) {
    let street = (key >> STREET_SHIFT) as usize;
    let bucket = ((key >> BUCKET_SHIFT) & ((1 << BUCKET_BITS) - 1)) as u32;
    let seq = Seq(key & ((1u128 << SEQ_BITS) - 1));
    (street, bucket, seq)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Key {
    hi: u64,
    lo: u64,
}

impl Key {
    #[inline]
    fn new(k: u128) -> Key {
        Key { hi: (k >> 64) as u64, lo: k as u64 }
    }
    #[inline]
    fn get(self) -> u128 {
        ((self.hi as u128) << 64) | self.lo as u128
    }
}

#[inline]
fn fmix64(mut k: u64) -> u64 {
    k ^= k >> 33;
    k = k.wrapping_mul(0xff51_afd7_ed55_8ccd);
    k ^= k >> 33;
    k = k.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    k ^ (k >> 33)
}

/// Hasher for [`Key`] inside a shard's map.
#[derive(Default)]
struct KeyHasher(u64);

impl Hasher for KeyHasher {
    #[inline]
    fn finish(&self) -> u64 {
        fmix64(self.0)
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = (self.0.rotate_left(29) ^ x).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
}

type KeyMap = HashMap<Key, Slot, BuildHasherDefault<KeyHasher>>;

// ----------------------------------------------------------------------
// Tables
// ----------------------------------------------------------------------

#[derive(Clone, Copy, Debug)]
struct Slot {
    off: u32,
    n: u32,
}

#[derive(Default)]
struct Shard {
    map: KeyMap,
    /// Per entry: `n` regrets followed by `n` strategy sums.
    data: Vec<f32>,
}

impl Shard {
    #[inline]
    fn entry(&mut self, key: Key, n: usize) -> &mut [f32] {
        use std::collections::hash_map::Entry;
        let off = match self.map.entry(key) {
            Entry::Occupied(e) => {
                debug_assert_eq!(e.get().n as usize, n, "infoset action count changed");
                e.get().off as usize
            }
            Entry::Vacant(v) => {
                let off = self.data.len();
                assert!(off + 2 * n <= u32::MAX as usize, "shard too large; use more shards");
                self.data.resize(off + 2 * n, 0.0);
                v.insert(Slot { off: off as u32, n: n as u32 });
                off
            }
        };
        &mut self.data[off..off + 2 * n]
    }
}

/// Regret and strategy-sum tables, sharded by key hash behind mutexes.
pub struct Tables {
    shards: Box<[Mutex<Shard>]>,
    shift: u32,
}

/// Regret matching on the positive part; uniform when nothing is positive.
#[inline]
pub fn regret_matching(regrets: &[f32], out: &mut [f32]) {
    let n = regrets.len();
    let mut sum = 0.0f32;
    for k in 0..n {
        let r = regrets[k].max(0.0);
        out[k] = r;
        sum += r;
    }
    if sum > 0.0 {
        let inv = 1.0 / sum;
        for x in &mut out[..n] {
            *x *= inv;
        }
    } else {
        let u = 1.0 / n as f32;
        for x in &mut out[..n] {
            *x = u;
        }
    }
}

impl Tables {
    pub fn new(shards: usize) -> Tables {
        let shards = shards.max(1).next_power_of_two();
        let v: Vec<Mutex<Shard>> = (0..shards).map(|_| Mutex::new(Shard::default())).collect();
        Tables { shards: v.into_boxed_slice(), shift: 64 - shards.trailing_zeros() }
    }

    #[inline]
    fn shard(&self, key: Key) -> &Mutex<Shard> {
        if self.shards.len() == 1 {
            return &self.shards[0];
        }
        let h = fmix64(key.lo.wrapping_mul(0xD6E8_FEB8_6659_FD93) ^ key.hi.wrapping_add(0xA076_1D64_78BD_642F));
        &self.shards[(h >> self.shift) as usize]
    }

    fn lock(&self, key: Key) -> std::sync::MutexGuard<'_, Shard> {
        self.shard(key).lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Copy the regrets of `key` (created if missing) into `out[..n]`.
    #[inline]
    pub fn read_regrets(&self, key: u128, n: usize, out: &mut [f32]) {
        let k = Key::new(key);
        let mut g = self.lock(k);
        let e = g.entry(k, n);
        out[..n].copy_from_slice(&e[..n]);
    }

    /// Current strategy of `key` into `sigma[..n]`, and add `weight * sigma`
    /// to its strategy sum.
    #[inline]
    pub fn strategy_accumulate(&self, key: u128, n: usize, weight: f32, sigma: &mut [f32]) {
        let k = Key::new(key);
        let mut g = self.lock(k);
        let e = g.entry(k, n);
        regret_matching(&e[..n], sigma);
        for a in 0..n {
            e[n + a] += weight * sigma[a];
        }
    }

    /// Add `delta[a]` to the regret of every action with `mask[a]`, then
    /// clamp at `floor`.
    #[inline]
    pub fn add_regrets(&self, key: u128, n: usize, delta: &[f32], mask: &[bool], floor: f32) {
        let k = Key::new(key);
        let mut g = self.lock(k);
        let e = g.entry(k, n);
        for a in 0..n {
            if mask[a] {
                e[a] = (e[a] + delta[a]).max(floor);
            }
        }
    }

    /// Regrets and strategy sums of `key`, if present.
    pub fn get(&self, key: u128) -> Option<(Vec<f32>, Vec<f32>)> {
        let k = Key::new(key);
        let g = self.lock(k);
        let slot = *g.map.get(&k)?;
        let (o, n) = (slot.off as usize, slot.n as usize);
        Some((g.data[o..o + n].to_vec(), g.data[o + n..o + 2 * n].to_vec()))
    }

    /// Insert or overwrite an entry.
    pub fn put(&self, key: u128, regrets: &[f32], sums: &[f32]) {
        let n = regrets.len();
        let k = Key::new(key);
        let mut g = self.lock(k);
        let e = g.entry(k, n);
        e[..n].copy_from_slice(regrets);
        e[n..].copy_from_slice(sums);
    }

    /// Multiply every regret and strategy sum by `d`.
    pub fn scale(&self, d: f32) {
        for s in self.shards.iter() {
            let mut g = s.lock().unwrap_or_else(|e| e.into_inner());
            for x in g.data.iter_mut() {
                *x *= d;
            }
        }
    }

    /// Number of infosets.
    pub fn len(&self) -> usize {
        self.shards.iter().map(|s| s.lock().unwrap_or_else(|e| e.into_inner()).map.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated heap bytes: map buckets (key + slot + control byte) plus the
    /// value arrays.
    pub fn memory_bytes(&self) -> u64 {
        let per_bucket = (std::mem::size_of::<(Key, Slot)>() + 1) as u64;
        self.shards
            .iter()
            .map(|s| {
                let g = s.lock().unwrap_or_else(|e| e.into_inner());
                g.map.capacity() as u64 * per_bucket + g.data.capacity() as u64 * 4
            })
            .sum()
    }

    /// Visit every entry (shard by shard, holding one lock at a time).
    pub fn for_each(&self, mut f: impl FnMut(u128, &[f32], &[f32])) {
        for s in self.shards.iter() {
            let g = s.lock().unwrap_or_else(|e| e.into_inner());
            for (k, slot) in g.map.iter() {
                let (o, n) = (slot.off as usize, slot.n as usize);
                f(k.get(), &g.data[o..o + n], &g.data[o + n..o + 2 * n]);
            }
        }
    }
}

// ----------------------------------------------------------------------
// Configuration
// ----------------------------------------------------------------------

/// Everything that defines a training run.
#[derive(Clone, Debug, PartialEq)]
pub struct SolverConfig {
    pub game: GameConfig,
    pub actions: ActionAbstraction,
    pub cards: CardAbstractionSpec,
    pub seed: u64,
    /// Discount interval in iterations (0 disables Linear CFR discounting).
    pub lcfr_discount_every: u64,
    /// No discounting after this many iterations.
    pub lcfr_stop: u64,
    /// Pruning starts after this many iterations (`u64::MAX` disables it).
    pub prune_start: u64,
    /// Probability that a traversal (after `prune_start`) prunes.
    pub prune_prob: f64,
    /// Regret (in big blinds) below which actions may be pruned.
    pub prune_threshold: f32,
    /// Lower bound on stored regrets (in big blinds).
    pub regret_floor: f32,
    pub checkpoint_path: Option<String>,
    /// Seconds between checkpoints written during `run` (0 disables).
    pub checkpoint_interval: f64,
    /// Number of table shards (rounded up to a power of two).
    pub shards: usize,
}

impl Default for SolverConfig {
    fn default() -> Self {
        SolverConfig {
            game: GameConfig::new(2, 10_000, 50, 100, 0),
            actions: ActionAbstraction::default(),
            cards: CardAbstractionSpec::default(),
            seed: 0,
            lcfr_discount_every: 100_000,
            lcfr_stop: 4_000_000,
            prune_start: 2_000_000,
            prune_prob: 0.95,
            prune_threshold: -300.0,
            regret_floor: -310.0,
            checkpoint_path: None,
            checkpoint_interval: 0.0,
            shards: 4096,
        }
    }
}

impl SolverConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.game.validate().map_err(|e| e.to_string())?;
        if self.game.num_players != 2 {
            // The traversal is written for n players, but the 97-bit sequence
            // field of the key is sized for heads-up (4 streets x (cap + 2)).
            return Err("only heads-up (num_players = 2) is supported by the key layout".into());
        }
        self.actions.validate()?;
        if !(0.0..=1.0).contains(&self.prune_prob) {
            return Err("prune_prob must be in [0, 1]".into());
        }
        if self.regret_floor > 0.0 || self.prune_threshold > 0.0 {
            return Err("regret_floor and prune_threshold must be <= 0".into());
        }
        for s in 0..4 {
            if self.cards.buckets[s] as u64 >= (1u64 << BUCKET_BITS) {
                return Err("too many buckets".into());
            }
        }
        Ok(())
    }

    /// JSON rendering (for strategy-file headers and logs).
    pub fn to_json(&self) -> String {
        let g = &self.game;
        let action = |a: &AbstractAction, w: When| match (a, w) {
            (AbstractAction::Fold, _) => "[\"fold\"]".to_string(),
            (AbstractAction::CheckCall, _) => "[\"check_call\"]".to_string(),
            (AbstractAction::RaisePot(f), When::Any) => format!("[\"raise\", {}]", f),
            (AbstractAction::RaisePot(f), When::Open) => format!("[\"raise\", {}, \"open\"]", f),
            (AbstractAction::RaisePot(f), When::Reraise) => format!("[\"raise\", {}, \"reraise\"]", f),
            (AbstractAction::AllIn, _) => "[\"allin\"]".to_string(),
        };
        let streets: Vec<String> = (0..4)
            .map(|s| {
                let l = &self.actions.streets[s];
                let items: Vec<String> =
                    l.iter().enumerate().map(|(i, a)| action(a, self.actions.when_at(s, i))).collect();
                format!("[{}]", items.join(", "))
            })
            .collect();
        let tables: Vec<String> = self
            .cards
            .tables
            .iter()
            .map(|t| match t {
                Some(p) => json_str(p),
                None => "null".into(),
            })
            .collect();
        format!(
            concat!(
                "{{\"game\": {{\"num_players\": {}, \"stacks\": {:?}, \"small_blind\": {}, \"big_blind\": {}, \"ante\": {}}}, ",
                "\"actions\": {{\"max_raises\": {}, \"streets\": [{}]}}, ",
                "\"cards\": {{\"buckets\": {:?}, \"hs_samples\": {}, \"tables\": [{}]}}, ",
                "\"seed\": {}, \"lcfr_discount_every\": {}, \"lcfr_stop\": {}, \"prune_start\": {}, ",
                "\"prune_prob\": {}, \"prune_threshold\": {}, \"regret_floor\": {}, ",
                "\"checkpoint_path\": {}, \"checkpoint_interval\": {}, \"shards\": {}}}"
            ),
            g.num_players,
            g.stacks,
            g.small_blind,
            g.big_blind,
            g.ante,
            self.actions.max_raises,
            streets.join(", "),
            self.cards.buckets,
            self.cards.hs_samples,
            tables.join(", "),
            self.seed,
            self.lcfr_discount_every,
            self.lcfr_stop,
            if self.prune_start == u64::MAX { "null".to_string() } else { self.prune_start.to_string() },
            self.prune_prob,
            self.prune_threshold as f64,
            self.regret_floor as f64,
            self.checkpoint_path.as_deref().map(json_str).unwrap_or_else(|| "null".into()),
            self.checkpoint_interval,
            self.shards,
        )
    }

    fn write_bin(&self, w: &mut Vec<u8>) {
        let g = &self.game;
        put_u32(w, g.num_players as u32);
        for &s in &g.stacks {
            put_i64(w, s);
        }
        put_i64(w, g.small_blind);
        put_i64(w, g.big_blind);
        put_i64(w, g.ante);
        w.push(self.actions.max_raises);
        for (s, l) in self.actions.streets.iter().enumerate() {
            w.push(l.len() as u8);
            for (i, a) in l.iter().enumerate() {
                match a {
                    AbstractAction::Fold => w.push(0),
                    AbstractAction::CheckCall => w.push(1),
                    AbstractAction::RaisePot(f) => {
                        // 2 = unconditioned (the original format), 4 = open, 5 = reraise
                        w.push(match self.actions.when_at(s, i) {
                            When::Any => 2,
                            When::Open => 4,
                            When::Reraise => 5,
                        });
                        w.extend_from_slice(&f.to_le_bytes());
                    }
                    AbstractAction::AllIn => w.push(3),
                }
            }
        }
        for &b in &self.cards.buckets {
            put_u32(w, b);
        }
        put_u32(w, self.cards.hs_samples);
        for t in &self.cards.tables {
            put_opt_str(w, t.as_deref());
        }
        put_u64(w, self.seed);
        put_u64(w, self.lcfr_discount_every);
        put_u64(w, self.lcfr_stop);
        put_u64(w, self.prune_start);
        w.extend_from_slice(&self.prune_prob.to_le_bytes());
        w.extend_from_slice(&self.prune_threshold.to_le_bytes());
        w.extend_from_slice(&self.regret_floor.to_le_bytes());
        put_opt_str(w, self.checkpoint_path.as_deref());
        w.extend_from_slice(&self.checkpoint_interval.to_le_bytes());
        put_u64(w, self.shards as u64);
    }

    fn read_bin(r: &mut ByteReader) -> Result<SolverConfig, String> {
        let n = r.u32()? as usize;
        if !(2..=MAX_PLAYERS).contains(&n) {
            return Err("bad num_players".into());
        }
        let mut stacks = Vec::with_capacity(n);
        for _ in 0..n {
            stacks.push(r.i64()?);
        }
        let game = GameConfig { num_players: n, stacks, small_blind: r.i64()?, big_blind: r.i64()?, ante: r.i64()? };
        let max_raises = r.u8()?;
        let mut streets: [Vec<AbstractAction>; 4] = Default::default();
        let mut when: [Vec<When>; 4] = Default::default();
        for (l, wl) in streets.iter_mut().zip(when.iter_mut()) {
            let len = r.u8()?;
            for _ in 0..len {
                let (a, w) = match r.u8()? {
                    0 => (AbstractAction::Fold, When::Any),
                    1 => (AbstractAction::CheckCall, When::Any),
                    2 => (AbstractAction::RaisePot(r.f64()?), When::Any),
                    3 => (AbstractAction::AllIn, When::Any),
                    4 => (AbstractAction::RaisePot(r.f64()?), When::Open),
                    5 => (AbstractAction::RaisePot(r.f64()?), When::Reraise),
                    k => return Err(format!("bad abstract action code {k}")),
                };
                l.push(a);
                wl.push(w);
            }
            if wl.iter().all(|&w| w == When::Any) {
                wl.clear();
            }
        }
        let mut buckets = [0u32; 4];
        for b in buckets.iter_mut() {
            *b = r.u32()?;
        }
        let hs_samples = r.u32()?;
        let mut tables: [Option<String>; 4] = Default::default();
        for t in tables.iter_mut() {
            *t = r.opt_str()?;
        }
        Ok(SolverConfig {
            game,
            actions: ActionAbstraction { streets, max_raises, when },
            cards: CardAbstractionSpec { buckets, hs_samples, tables },
            seed: r.u64()?,
            lcfr_discount_every: r.u64()?,
            lcfr_stop: r.u64()?,
            prune_start: r.u64()?,
            prune_prob: r.f64()?,
            prune_threshold: r.f32()?,
            regret_floor: r.f32()?,
            checkpoint_path: r.opt_str()?,
            checkpoint_interval: r.f64()?,
            shards: r.u64()? as usize,
        })
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

// ----------------------------------------------------------------------
// Binary helpers
// ----------------------------------------------------------------------

fn put_u32(w: &mut Vec<u8>, x: u32) {
    w.extend_from_slice(&x.to_le_bytes());
}
fn put_u64(w: &mut Vec<u8>, x: u64) {
    w.extend_from_slice(&x.to_le_bytes());
}
fn put_i64(w: &mut Vec<u8>, x: i64) {
    w.extend_from_slice(&x.to_le_bytes());
}
fn put_bytes(w: &mut Vec<u8>, b: &[u8]) {
    put_u32(w, b.len() as u32);
    w.extend_from_slice(b);
}
fn put_opt_str(w: &mut Vec<u8>, s: Option<&str>) {
    match s {
        None => w.push(0),
        Some(s) => {
            w.push(1);
            put_bytes(w, s.as_bytes());
        }
    }
}

struct ByteReader<'a> {
    b: &'a [u8],
    pos: usize,
}

impl<'a> ByteReader<'a> {
    fn new(b: &'a [u8]) -> Self {
        ByteReader { b, pos: 0 }
    }
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.pos..self.pos + n).ok_or("unexpected end of file")?;
        self.pos += n;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn i64(&mut self) -> Result<i64, String> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f32(&mut self) -> Result<f32, String> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> Result<f64, String> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn bytes(&mut self) -> Result<&'a [u8], String> {
        let n = self.u32()? as usize;
        self.take(n)
    }
    fn opt_str(&mut self) -> Result<Option<String>, String> {
        match self.u8()? {
            0 => Ok(None),
            _ => Ok(Some(String::from_utf8(self.bytes()?.to_vec()).map_err(|_| "bad utf-8")?)),
        }
    }
}

// ----------------------------------------------------------------------
// Trainer
// ----------------------------------------------------------------------

const CKPT_MAGIC: &[u8; 8] = b"PBMCCFR1";
const STRAT_MAGIC: &[u8; 8] = b"PBSTRAT1";
const FORMAT_VERSION: u32 = 1;

/// Per-traversal card information, computed lazily.
struct Deal<'a> {
    cards: &'a CardAbstraction,
    deck: [Card; 52],
    n: usize,
    buckets: [[std::cell::Cell<u32>; 4]; MAX_PLAYERS],
    ranks: [std::cell::Cell<HandRank>; MAX_PLAYERS],
}

impl<'a> Deal<'a> {
    fn new(cards: &'a CardAbstraction, deck: [Card; 52], n: usize) -> Self {
        Deal { cards, deck, n, buckets: Default::default(), ranks: Default::default() }
    }
    #[inline]
    fn hole(&self, p: usize) -> [Card; 2] {
        [self.deck[2 * p], self.deck[2 * p + 1]]
    }
    #[inline]
    fn board(&self) -> &[Card] {
        &self.deck[2 * self.n..2 * self.n + 5]
    }
    #[inline]
    fn bucket(&self, p: usize, street: usize) -> u32 {
        let c = &self.buckets[p][street];
        let v = c.get();
        if v != 0 {
            return v - 1;
        }
        let b = self.cards.bucket(street, self.hole(p), self.board());
        c.set(b + 1);
        b
    }
    #[inline]
    fn rank(&self, p: usize) -> HandRank {
        let c = &self.ranks[p];
        let v = c.get();
        if v != 0 {
            return v;
        }
        let r = evaluate_hole_board(self.hole(p), self.board());
        c.set(r);
        r
    }
}

/// Counters of one `run`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RunStats {
    pub iterations: u64,
    pub nodes: u64,
    pub seconds: f64,
}

/// The MCCFR solver.
pub struct Trainer {
    pub config: SolverConfig,
    pub cards: CardAbstraction,
    pub tables: Tables,
    iteration: AtomicU64,
    nodes: AtomicU64,
    /// Optional metadata (JSON) written into checkpoints and strategy files.
    pub meta: String,
    pub last_run: RunStats,
    last_checkpoint: Option<Instant>,
    /// Wall-clock seconds spent in `run` over the trainer's lifetime
    /// (including before a checkpoint was loaded).
    pub total_seconds: f64,
}

impl Trainer {
    pub fn new(config: SolverConfig) -> Result<Trainer, String> {
        config.validate()?;
        let cards = CardAbstraction::new(config.cards.clone())?;
        Self::with_cards(config, cards)
    }

    /// Use a prepared card abstraction (e.g. with in-memory tables).
    pub fn with_cards(config: SolverConfig, cards: CardAbstraction) -> Result<Trainer, String> {
        config.validate()?;
        let tables = Tables::new(config.shards);
        Ok(Trainer {
            config,
            cards,
            tables,
            iteration: AtomicU64::new(0),
            nodes: AtomicU64::new(0),
            meta: String::new(),
            last_run: RunStats::default(),
            last_checkpoint: None,
            total_seconds: 0.0,
        })
    }

    pub fn iterations(&self) -> u64 {
        self.iteration.load(Ordering::Relaxed)
    }

    pub fn nodes(&self) -> u64 {
        self.nodes.load(Ordering::Relaxed)
    }

    /// Terminal utility of player `i` in big blinds.
    #[inline]
    fn utility(&self, st: &GameState, deal: &Deal, i: usize) -> f32 {
        let chips: Chips = if st.num_players() == 2 {
            let c = st.contributed();
            let f = st.folded();
            let o = 1 - i;
            if f[i] {
                -c[i]
            } else if f[o] {
                c[o]
            } else {
                let m = c[0].min(c[1]);
                match deal.rank(i).cmp(&deal.rank(o)) {
                    CmpOrdering::Greater => m,
                    CmpOrdering::Less => -m,
                    CmpOrdering::Equal => 0,
                }
            }
        } else {
            let mut out = [0; MAX_PLAYERS];
            st.payoffs_into(&mut out).expect("terminal state");
            out[i]
        };
        chips as f32 / self.config.game.big_blind as f32
    }

    #[allow(clippy::too_many_arguments)]
    fn traverse(
        &self,
        st: GameState,
        seq: Seq,
        deal: &Deal,
        i: usize,
        prune: bool,
        rng: &mut Rng,
        nodes: &mut u64,
    ) -> f32 {
        *nodes += 1;
        if st.is_terminal() {
            return self.utility(&st, deal, i);
        }
        let p = st.current_player().expect("non-terminal");
        let street = st.street();
        let list: ActionList = self.config.actions.legal(&st);
        let n = list.len;
        let key = make_key(street, deal.bucket(p, street), seq);
        let mut sigma = [0f32; 16];
        if p == i {
            let mut regrets = [0f32; 16];
            self.tables.read_regrets(key, n, &mut regrets);
            regret_matching(&regrets[..n], &mut sigma);
            let can_prune = prune && street < 3;
            let threshold = self.config.prune_threshold;
            let mut vals = [0f32; 16];
            let mut explored = [false; 16];
            let mut v = 0.0f32;
            for k in 0..n {
                let child = st.child(list.action[k]).expect("abstract action is legal");
                if can_prune && regrets[k] < threshold && sigma[k] == 0.0 && !child.is_terminal() {
                    continue;
                }
                let vk = self.traverse(child, seq.push(list.index[k]), deal, i, prune, rng, nodes);
                vals[k] = vk;
                explored[k] = true;
                v += sigma[k] * vk;
            }
            let mut delta = [0f32; 16];
            for k in 0..n {
                delta[k] = vals[k] - v;
            }
            self.tables.add_regrets(key, n, &delta[..n], &explored[..n], self.config.regret_floor);
            v
        } else {
            self.tables.strategy_accumulate(key, n, 1.0, &mut sigma);
            let u = rng.uniform() as f32;
            let mut acc = 0.0f32;
            let mut k = n - 1;
            for a in 0..n {
                acc += sigma[a];
                if u < acc {
                    k = a;
                    break;
                }
            }
            let mut st = st;
            st.apply(list.action[k]).expect("abstract action is legal");
            self.traverse(st, seq.push(list.index[k]), deal, i, prune, rng, nodes)
        }
    }

    /// Run iteration number `t` (one traversal per player). Returns nodes visited.
    fn iterate(&self, t: u64) -> u64 {
        let n = self.config.game.num_players;
        let need = 2 * n + 5;
        let mut nodes = 0u64;
        for i in 0..n {
            let mut rng = Rng::new(fmix64(self.config.seed ^ fmix64(t.wrapping_mul(16).wrapping_add(i as u64 + 1))));
            let mut deck = fresh_deck();
            rng.partial_shuffle(&mut deck, need);
            let prune = t >= self.config.prune_start && rng.uniform() < self.config.prune_prob;
            let deal = Deal::new(&self.cards, deck, n);
            let root = GameState::new_hand(&self.config.game, 0, &deck[..need]).expect("valid config");
            self.traverse(root, Seq::EMPTY, &deal, i, prune, &mut rng, &mut nodes);
        }
        nodes
    }

    /// Run `iterations` more iterations on `threads` threads. `keep_going`
    /// is polled between batches (return false to stop early, e.g. on
    /// Ctrl-C). Discounting and periodic checkpoints happen between batches.
    pub fn run(
        &mut self,
        iterations: u64,
        threads: usize,
        mut keep_going: impl FnMut() -> bool,
    ) -> Result<RunStats, String> {
        let threads = threads.max(1);
        let start = Instant::now();
        if self.last_checkpoint.is_none() {
            self.last_checkpoint = Some(start);
        }
        let t0 = self.iterations();
        let target = t0 + iterations;
        let nodes0 = self.nodes();
        let mut batch = (32 * threads) as u64;
        let every = self.config.lcfr_discount_every;
        let mut t = t0;
        while t < target {
            let mut end = (t + batch).min(target);
            if every > 0 && t < self.config.lcfr_stop {
                let next = (t / every + 1) * every;
                end = end.min(next);
            }
            let bt = Instant::now();
            let next = AtomicU64::new(t);
            let stop = AtomicBool::new(false);
            let this: &Trainer = &*self;
            std::thread::scope(|scope| {
                for _ in 0..threads {
                    scope.spawn(|| {
                        let mut local = 0u64;
                        loop {
                            if stop.load(Ordering::Relaxed) {
                                break;
                            }
                            let it = next.fetch_add(1, Ordering::Relaxed);
                            if it >= end {
                                break;
                            }
                            local += this.iterate(it);
                        }
                        this.nodes.fetch_add(local, Ordering::Relaxed);
                    });
                }
            });
            t = end;
            self.iteration.store(t, Ordering::Relaxed);
            if every > 0 && t <= self.config.lcfr_stop && t.is_multiple_of(every) {
                let k = (t / every) as f32;
                self.tables.scale(k / (k + 1.0));
            }
            // Aim for batches of ~0.25 s.
            let dt = bt.elapsed().as_secs_f64();
            if dt < 0.1 {
                batch = (batch * 2).min(1 << 24);
            } else if dt > 0.5 && batch > threads as u64 {
                batch /= 2;
            }
            if self.config.checkpoint_interval > 0.0 {
                if let Some(path) = self.config.checkpoint_path.clone() {
                    let due =
                        self.last_checkpoint.map(|c| c.elapsed().as_secs_f64() >= self.config.checkpoint_interval);
                    if due.unwrap_or(false) {
                        self.save(&path)?;
                        self.last_checkpoint = Some(Instant::now());
                    }
                }
            }
            if !keep_going() {
                break;
            }
        }
        let stats =
            RunStats { iterations: t - t0, nodes: self.nodes() - nodes0, seconds: start.elapsed().as_secs_f64() };
        self.total_seconds += stats.seconds;
        self.last_run = stats;
        Ok(stats)
    }

    /// Average strategy at `state` for the player to act (whose hole cards
    /// must be visible in `state`), over `legal(state)`. The state's history
    /// must be on the abstract tree. Unvisited infosets give uniform.
    pub fn strategy(&self, state: &GameState, player: usize, average: bool) -> Result<(ActionList, Vec<f32>), String> {
        let (key, list) = self.key_of(state, player)?;
        let n = list.len;
        let probs = match self.tables.get(key) {
            Some((r, s)) if r.len() == n => {
                let mut out = vec![0f32; n];
                let sum: f32 = s.iter().sum();
                if average && sum > 0.0 {
                    for k in 0..n {
                        out[k] = s[k] / sum;
                    }
                } else {
                    regret_matching(&r, &mut out);
                }
                out
            }
            _ => vec![1.0 / n as f32; n],
        };
        Ok((list, probs))
    }

    /// Infoset key and abstract action list of `player` at `state`.
    pub fn key_of(&self, state: &GameState, player: usize) -> Result<(u128, ActionList), String> {
        infoset_key(&self.config.actions, &self.cards, state, player, None)
    }

    // -- persistence --------------------------------------------------

    /// Write a checkpoint: config, counters and every table entry.
    pub fn save(&self, path: &str) -> Result<(), String> {
        let tmp = format!("{path}.tmp");
        if let Some(dir) = std::path::Path::new(path).parent() {
            if !dir.as_os_str().is_empty() {
                std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
            }
        }
        {
            let f = File::create(&tmp).map_err(|e| format!("{tmp}: {e}"))?;
            let mut w = BufWriter::with_capacity(1 << 20, f);
            let mut head = Vec::new();
            head.extend_from_slice(CKPT_MAGIC);
            put_u32(&mut head, FORMAT_VERSION);
            let mut cfg = Vec::new();
            self.config.write_bin(&mut cfg);
            put_bytes(&mut head, &cfg);
            put_bytes(&mut head, self.meta.as_bytes());
            put_u64(&mut head, self.iterations());
            put_u64(&mut head, self.nodes());
            head.extend_from_slice(&self.total_seconds.to_le_bytes());
            put_u64(&mut head, self.tables.len() as u64);
            w.write_all(&head).map_err(|e| e.to_string())?;
            let mut err = None;
            let mut buf = Vec::with_capacity(256);
            self.tables.for_each(|k, r, s| {
                if err.is_some() {
                    return;
                }
                buf.clear();
                buf.extend_from_slice(&((k >> 64) as u64).to_le_bytes());
                buf.extend_from_slice(&(k as u64).to_le_bytes());
                buf.push(r.len() as u8);
                for x in r.iter().chain(s) {
                    buf.extend_from_slice(&x.to_le_bytes());
                }
                if let Err(e) = w.write_all(&buf) {
                    err = Some(e.to_string());
                }
            });
            if let Some(e) = err {
                return Err(e);
            }
            w.flush().map_err(|e| e.to_string())?;
        }
        std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))
    }

    /// Load a checkpoint written by [`Trainer::save`].
    pub fn load(path: &str) -> Result<Trainer, String> {
        let f = File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let mut r = BufReader::with_capacity(1 << 20, f);
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic).map_err(|e| e.to_string())?;
        if &magic != CKPT_MAGIC {
            return Err(format!("{path}: not an MCCFR checkpoint"));
        }
        let version = read_u32(&mut r)?;
        if version != FORMAT_VERSION {
            return Err(format!("{path}: unsupported checkpoint version {version}"));
        }
        let cfg_bytes = read_vec(&mut r)?;
        let config = SolverConfig::read_bin(&mut ByteReader::new(&cfg_bytes))?;
        let meta = String::from_utf8(read_vec(&mut r)?).map_err(|_| "bad meta")?;
        let iterations = read_u64(&mut r)?;
        let nodes = read_u64(&mut r)?;
        let mut secs = [0u8; 8];
        r.read_exact(&mut secs).map_err(|e| e.to_string())?;
        let count = read_u64(&mut r)?;
        let mut t = Trainer::new(config)?;
        t.meta = meta;
        t.iteration.store(iterations, Ordering::Relaxed);
        t.nodes.store(nodes, Ordering::Relaxed);
        t.total_seconds = f64::from_le_bytes(secs);
        let mut head = [0u8; 17];
        let mut vals = [0u8; 8 * 16];
        let mut fl = [0f32; 32];
        for _ in 0..count {
            r.read_exact(&mut head).map_err(|e| format!("{path}: truncated ({e})"))?;
            let hi = u64::from_le_bytes(head[0..8].try_into().unwrap());
            let lo = u64::from_le_bytes(head[8..16].try_into().unwrap());
            let n = head[16] as usize;
            if n == 0 || n > 16 {
                return Err(format!("{path}: corrupt entry"));
            }
            r.read_exact(&mut vals[..8 * n]).map_err(|e| format!("{path}: truncated ({e})"))?;
            for j in 0..2 * n {
                fl[j] = f32::from_le_bytes(vals[4 * j..4 * j + 4].try_into().unwrap());
            }
            t.tables.put(((hi as u128) << 64) | lo as u128, &fl[..n], &fl[n..2 * n]);
        }
        Ok(t)
    }

    /// Write the averaged strategy (see the module docs of
    /// [`BlueprintStrategy`] for the format). Infosets whose strategy sum is
    /// zero fall back to the current regret-matching strategy, or are left
    /// out when all their regrets are non-positive too. Returns the number of
    /// infosets written.
    pub fn export_strategy(&self, path: &str) -> Result<u64, String> {
        // Flat layout (about 24 + 2n bytes per infoset) so exporting a large
        // table does not need a heap allocation per row.
        let mut rows: Vec<Row> = Vec::with_capacity(self.tables.len());
        let mut probs: Vec<u16> = Vec::new();
        let mut sigma = [0f32; 16];
        let mut overflow = false;
        self.tables.for_each(|k, r, s| {
            let n = r.len();
            let sum: f32 = s.iter().sum();
            if sum > 0.0 {
                for a in 0..n {
                    sigma[a] = s[a] / sum;
                }
            } else if r.iter().any(|&x| x > 0.0) {
                regret_matching(r, &mut sigma);
            } else {
                return;
            }
            if probs.len() + n > u32::MAX as usize {
                overflow = true;
                return;
            }
            rows.push(Row { key: Key::new(k), off: probs.len() as u32, n: n as u8 });
            probs.extend(sigma[..n].iter().map(|&x| quantize(x)));
        });
        if overflow {
            return Err("strategy too large for 32-bit offsets".into());
        }
        rows.sort_unstable_by_key(|r| r.key);
        write_strategy(path, &self.config, &self.meta, &rows, &probs)?;
        Ok(rows.len() as u64)
    }
}

/// One exported infoset: key and its probabilities in the flat array.
struct Row {
    key: Key,
    off: u32,
    n: u8,
}

/// Quantize a probability to u16 (`round(p * 65535)`).
#[inline]
fn quantize(p: f32) -> u16 {
    (p.clamp(0.0, 1.0) * 65535.0).round() as u16
}

fn read_u32(r: &mut impl Read) -> Result<u32, String> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> Result<u64, String> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b).map_err(|e| e.to_string())?;
    Ok(u64::from_le_bytes(b))
}
fn read_vec(r: &mut impl Read) -> Result<Vec<u8>, String> {
    let n = read_u32(r)? as usize;
    let mut v = vec![0u8; n];
    r.read_exact(&mut v).map_err(|e| e.to_string())?;
    Ok(v)
}

/// Infoset key of `player` (who must be the player to act) at `state`, whose
/// history must be on the abstract tree. `cards` gives the player's hole
/// cards and the board; when `None` they are read from `state`.
pub fn infoset_key(
    actions: &ActionAbstraction,
    bucketer: &dyn Bucketer,
    state: &GameState,
    player: usize,
    cards: Option<([Card; 2], &[Card])>,
) -> Result<(u128, ActionList), String> {
    if state.current_player() != Some(player) {
        return Err(format!("player {player} is not to act"));
    }
    let seq = Seq::from_indices(&actions.sequence(state)?)?;
    let street = state.street();
    let (hole, board): ([Card; 2], Vec<Card>) = match cards {
        Some((h, b)) => (h, b.to_vec()),
        None => (state.hole_cards(player).ok_or("hole cards are hidden")?, state.board().to_vec()),
    };
    if board.len() != crate::isomorphism::board_len(street) {
        return Err(format!("board has {} cards on street {street}", board.len()));
    }
    let bucket = bucketer.bucket(street, hole, &board);
    Ok((make_key(street, bucket, seq), actions.legal(state)))
}

// ----------------------------------------------------------------------
// Strategy files
// ----------------------------------------------------------------------

fn write_strategy(path: &str, config: &SolverConfig, meta: &str, rows: &[Row], probs: &[u16]) -> Result<(), String> {
    if let Some(dir) = std::path::Path::new(path).parent() {
        if !dir.as_os_str().is_empty() {
            std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        }
    }
    let mut head = Vec::new();
    head.extend_from_slice(STRAT_MAGIC);
    put_u32(&mut head, FORMAT_VERSION);
    let mut cfg = Vec::new();
    config.write_bin(&mut cfg);
    put_bytes(&mut head, &cfg);
    put_bytes(&mut head, config.to_json().as_bytes());
    put_bytes(&mut head, meta.as_bytes());
    while head.len() % 8 != 0 {
        head.push(0);
    }
    let total = probs.len() as u64;
    if total > u32::MAX as u64 {
        return Err("strategy too large for 32-bit offsets".into());
    }
    put_u64(&mut head, rows.len() as u64);
    put_u64(&mut head, total);
    let tmp = format!("{path}.tmp");
    {
        let f = File::create(&tmp).map_err(|e| format!("{tmp}: {e}"))?;
        let mut w = BufWriter::with_capacity(1 << 20, f);
        let io = |e: std::io::Error| e.to_string();
        w.write_all(&head).map_err(io)?;
        for r in rows {
            w.write_all(&r.key.hi.to_le_bytes()).map_err(io)?;
            w.write_all(&r.key.lo.to_le_bytes()).map_err(io)?;
        }
        let mut off = 0u32;
        w.write_all(&off.to_le_bytes()).map_err(io)?;
        for r in rows {
            off += r.n as u32;
            w.write_all(&off.to_le_bytes()).map_err(io)?;
        }
        for r in rows {
            for q in &probs[r.off as usize..r.off as usize + r.n as usize] {
                w.write_all(&q.to_le_bytes()).map_err(io)?;
            }
        }
        w.flush().map_err(io)?;
    }
    std::fs::rename(&tmp, path).map_err(|e| format!("{path}: {e}"))
}

/// An exported averaged strategy, loaded for play.
///
/// File layout (all little-endian):
///
/// ```text
/// magic "PBSTRAT1" | u32 version (1)
/// u32 len + solver config (binary, see SolverConfig::write_bin)
/// u32 len + solver config as JSON (UTF-8)
/// u32 len + metadata JSON from the training script (UTF-8, may be empty)
/// zero padding to a multiple of 8 bytes
/// u64 N (infosets) | u64 P (total probabilities)
/// N x (u64 key_hi, u64 key_lo)      keys, sorted ascending as u128
/// (N + 1) x u32 offsets              row i is probs[offsets[i]..offsets[i+1]]
/// P x u16 probs                      probability * 65535, rounded
/// ```
///
/// Row `i` holds one probability per entry of `ActionAbstraction::legal`
/// at the infoset (abstract list order, after de-duplication).
pub struct BlueprintStrategy {
    pub config: SolverConfig,
    pub config_json: String,
    pub meta: String,
    pub cards: CardAbstraction,
    keys: Vec<Key>,
    offsets: Vec<u32>,
    probs: Vec<u16>,
}

impl BlueprintStrategy {
    pub fn load(path: &str) -> Result<BlueprintStrategy, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
        Self::from_bytes(&bytes).map_err(|e| format!("{path}: {e}"))
    }

    fn from_bytes(bytes: &[u8]) -> Result<BlueprintStrategy, String> {
        let mut r = ByteReader::new(bytes);
        if r.take(8)? != STRAT_MAGIC {
            return Err("not a strategy file".into());
        }
        let version = r.u32()?;
        if version != FORMAT_VERSION {
            return Err(format!("unsupported strategy version {version}"));
        }
        let config = SolverConfig::read_bin(&mut ByteReader::new(r.bytes()?))?;
        let config_json = String::from_utf8(r.bytes()?.to_vec()).map_err(|_| "bad json")?;
        let meta = String::from_utf8(r.bytes()?.to_vec()).map_err(|_| "bad meta")?;
        while !r.pos.is_multiple_of(8) {
            r.u8()?;
        }
        let n = r.u64()? as usize;
        let p = r.u64()? as usize;
        let mut keys = Vec::with_capacity(n);
        for _ in 0..n {
            let hi = r.u64()?;
            let lo = r.u64()?;
            keys.push(Key { hi, lo });
        }
        let mut offsets = Vec::with_capacity(n + 1);
        for _ in 0..=n {
            offsets.push(r.u32()?);
        }
        let raw = r.take(2 * p)?;
        let probs: Vec<u16> = raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        if offsets.last().copied() != Some(p as u32) {
            return Err("offsets do not match the probability count".into());
        }
        let cards = CardAbstraction::new(config.cards.clone())?;
        Ok(BlueprintStrategy { config, config_json, meta, cards, keys, offsets, probs })
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// Probabilities stored for `key` (normalized), if any.
    pub fn lookup(&self, key: u128) -> Option<Vec<f32>> {
        let k = Key::new(key);
        let i = self.keys.binary_search(&k).ok()?;
        let row = &self.probs[self.offsets[i] as usize..self.offsets[i + 1] as usize];
        let sum: f32 = row.iter().map(|&q| q as f32).sum();
        if sum <= 0.0 {
            return Some(vec![1.0 / row.len() as f32; row.len()]);
        }
        Some(row.iter().map(|&q| q as f32 / sum).collect())
    }

    /// The abstract actions at `abs_state` (whose history must be on the
    /// abstract tree) and their probabilities for the player to act holding
    /// `hole` with `board`. The flag is false when the infoset is missing
    /// from the file and the uniform fallback was used.
    pub fn action_probs(
        &self,
        abs_state: &GameState,
        hole: [Card; 2],
        board: &[Card],
    ) -> Result<(ActionList, Vec<f32>, bool), String> {
        let p = abs_state.current_player().ok_or("state is terminal")?;
        let (key, list) = infoset_key(&self.config.actions, &self.cards, abs_state, p, Some((hole, board)))?;
        match self.lookup(key) {
            Some(v) if v.len() == list.len => Ok((list, v, true)),
            _ => Ok((list, vec![1.0 / list.len as f32; list.len], false)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::abstraction::AbstractAction::*;

    fn tiny_config() -> SolverConfig {
        let street = vec![Fold, CheckCall, RaisePot(1.0), AllIn];
        SolverConfig {
            game: GameConfig::new(2, 1_000, 50, 100, 0),
            actions: ActionAbstraction {
                streets: [street.clone(), street.clone(), street.clone(), street],
                max_raises: 2,
                when: Default::default(),
            },
            cards: CardAbstractionSpec { buckets: [4, 4, 4, 4], hs_samples: 16, tables: Default::default() },
            seed: 1,
            lcfr_discount_every: 200,
            lcfr_stop: 2_000,
            prune_start: 1_000,
            prune_prob: 0.95,
            prune_threshold: -5.0,
            regret_floor: -6.0,
            checkpoint_path: None,
            checkpoint_interval: 0.0,
            shards: 16,
        }
    }

    #[test]
    fn key_roundtrip() {
        let seq = Seq::from_indices(&[2, 1, 1, 0, 15, 3]).unwrap();
        let k = make_key(3, 12345, seq);
        let (s, b, q) = decode_key(k);
        assert_eq!((s, b), (3, 12345));
        assert_eq!(q.indices(), vec![2, 1, 1, 0, 15, 3]);
        let long = Seq::from_indices(&[15; 24]).unwrap();
        let (s, b, q) = decode_key(make_key(2, (1 << 29) - 1, long));
        assert_eq!((s, b, q.indices()), (2, (1 << 29) - 1, vec![15; 24]));
        assert!(Seq::from_indices(&[0; 25]).is_err());
        // Keys for different sequences or buckets differ.
        assert_ne!(make_key(0, 1, Seq::EMPTY), make_key(0, 0, Seq::EMPTY));
        assert_ne!(make_key(0, 0, Seq::EMPTY.push(0)), make_key(0, 0, Seq::EMPTY));
    }

    #[test]
    fn fast_utility_matches_engine_payoffs() {
        let t = Trainer::new(tiny_config()).unwrap();
        let mut rng = Rng::new(3);
        for _ in 0..2000 {
            let mut deck = fresh_deck();
            rng.partial_shuffle(&mut deck, 9);
            let deal = Deal::new(&t.cards, deck, 2);
            let mut s = GameState::new_hand(&t.config.game, 0, &deck[..9]).unwrap();
            while !s.is_terminal() {
                let l = t.config.actions.legal(&s);
                let k = rng.below(l.len as u64) as usize;
                s.apply(l.action[k]).unwrap();
            }
            let pay = s.payoffs().unwrap();
            for i in 0..2 {
                assert_eq!(t.utility(&s, &deal, i), pay[i] as f32 / 100.0);
            }
        }
    }

    #[test]
    fn train_save_load_export() {
        let mut t = Trainer::new(tiny_config()).unwrap();
        let st = t.run(3_000, 2, || true).unwrap();
        assert_eq!(st.iterations, 3_000);
        assert!(st.nodes > 0);
        assert!(t.tables.len() > 10);
        let dir = std::env::temp_dir().join(format!("mccfr_test_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let ck = dir.join("c.bin").to_string_lossy().to_string();
        t.save(&ck).unwrap();
        let t2 = Trainer::load(&ck).unwrap();
        assert_eq!(t2.config, t.config);
        assert_eq!(t2.iterations(), 3_000);
        assert_eq!(t2.tables.len(), t.tables.len());
        t.tables.for_each(|k, r, s| {
            let (r2, s2) = t2.tables.get(k).unwrap();
            assert_eq!(r, &r2[..]);
            assert_eq!(s, &s2[..]);
        });
        let sp = dir.join("s.bin").to_string_lossy().to_string();
        let n = t.export_strategy(&sp).unwrap();
        let bp = BlueprintStrategy::load(&sp).unwrap();
        assert_eq!(bp.len() as u64, n);
        // Root strategy for some hand is a distribution.
        let deck: Vec<Card> = (0..52).collect();
        let root = GameState::new_hand(&t.config.game, 0, &deck).unwrap();
        let (list, p, found) = bp.action_probs(&root, [48, 49], &[]).unwrap();
        assert!(found);
        assert_eq!(list.len, p.len());
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-3);
        let (_, p2) = t.strategy(&root.clone(), 0, true).unwrap();
        assert_eq!(p2.len(), p.len());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn single_thread_is_deterministic() {
        let mut a = Trainer::new(tiny_config()).unwrap();
        let mut b = Trainer::new(tiny_config()).unwrap();
        a.run(500, 1, || true).unwrap();
        b.run(500, 1, || true).unwrap();
        a.tables.for_each(|k, r, s| {
            let (r2, s2) = b.tables.get(k).unwrap();
            assert_eq!(r, &r2[..]);
            assert_eq!(s, &s2[..]);
        });
    }
}
