//! Small deterministic RNG and random-play helpers used by tests, the
//! throughput example and (later) sampling-based solvers.

use crate::cards::Card;
use crate::game::{Action, Chips, GameConfig, GameError, GameState, MAX_PLAYERS};

/// xoshiro256** seeded through SplitMix64. Deterministic across platforms.
#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        let mut z = seed;
        let mut next = || {
            z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut x = z;
            x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^ (x >> 31)
        };
        Rng { s: [next(), next(), next(), next()] }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let result = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n` (`n > 0`), Lemire's multiply-shift (the
    /// tiny bias is irrelevant for simulation).
    #[inline]
    pub fn below(&mut self, n: u64) -> u64 {
        ((self.next_u64() as u128 * n as u128) >> 64) as u64
    }

    /// Uniform float in `[0, 1)`.
    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Shuffle so that the first `k` entries are a uniform random sample
    /// (partial Fisher-Yates).
    #[inline]
    pub fn partial_shuffle<T>(&mut self, v: &mut [T], k: usize) {
        let len = v.len();
        for i in 0..k.min(len) {
            let j = i + self.below((len - i) as u64) as usize;
            v.swap(i, j);
        }
    }
}

/// The ordered deck `0..52`.
pub fn fresh_deck() -> [Card; 52] {
    let mut d = [0u8; 52];
    for (i, c) in d.iter_mut().enumerate() {
        *c = i as Card;
    }
    d
}

/// A uniformly shuffled deck.
pub fn shuffled_deck(rng: &mut Rng) -> [Card; 52] {
    let mut d = fresh_deck();
    rng.partial_shuffle(&mut d, 52);
    d
}

/// A random legal action: fold 10% when facing a bet, raise 25% when legal
/// (half of those all-in, else uniform in the legal range), else check/call.
pub fn random_action(state: &GameState, rng: &mut Rng) -> Action {
    let la = state.legal_actions();
    let u = rng.uniform();
    if la.can_fold && u < 0.10 {
        return Action::fold();
    }
    if la.can_raise() && u > 0.75 {
        if la.min_raise_to >= la.max_raise_to || rng.below(2) == 0 {
            return Action::raise_to(la.max_raise_to);
        }
        let span = (la.max_raise_to - la.min_raise_to + 1) as u64;
        return Action::raise_to(la.min_raise_to + rng.below(span) as Chips);
    }
    Action::check_call()
}

/// Play one hand with [`random_action`] for every seat, dealing only the
/// cards needed. Returns the payoffs.
pub fn play_random_hand(
    config: &GameConfig,
    button: usize,
    rng: &mut Rng,
    deck: &mut [Card; 52],
) -> Result<[Chips; MAX_PLAYERS], GameError> {
    let need = 2 * config.num_players + 5;
    rng.partial_shuffle(deck, need);
    let mut s = GameState::new_hand(config, button, &deck[..need])?;
    while !s.is_terminal() {
        let a = random_action(&s, rng);
        s.apply(a)?;
    }
    let mut out = [0; MAX_PLAYERS];
    s.payoffs_into(&mut out)?;
    Ok(out)
}
