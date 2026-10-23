//! Action and card abstractions for the tabular solver and blueprint play.
//!
//! **Actions.** Each street has a list of abstract actions
//! ([`AbstractAction`]): fold, check/call, a raise of a fraction of the pot,
//! or all-in. [`ActionAbstraction::legal`] maps the list to concrete engine
//! actions in a state, dropping the ones that are not legal, capping raises
//! per street and de-duplicating sizes that map to the same amount. An
//! abstract action is identified by its index in the street's list; that
//! index is what infoset keys record. [`ActionAbstraction::translate`] maps an
//! arbitrary concrete opponent action back onto the abstraction with the
//! pseudo-harmonic mapping of Ganzfried & Sandholm (2013).
//!
//! **Cards.** A [`CardAbstraction`] maps `(street, hole, board)` to a bucket.
//! Preflop uses the 169 lossless classes (or fewer equity-ranked buckets);
//! postflop buckets come either from tables indexed by
//! [`crate::isomorphism::canonical_index`] or from the built-in default:
//! expected hand strength against a uniformly random hand, quantized.

use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::OnceLock;

use crate::cards::Card;
use crate::eval::evaluate_masks;
use crate::game::{Action, ActionKind, Chips, GameConfig, GameState, LegalActions};
use crate::isomorphism::{
    board_len, canonical_index, canonical_unindex, preflop_class, preflop_class_combos, preflop_class_hand,
    CANONICAL_SIZES,
};
use crate::sim::Rng;

// ----------------------------------------------------------------------
// Action abstraction
// ----------------------------------------------------------------------

/// Largest number of abstract actions per street (indices fit in 4 bits).
pub const MAX_ABSTRACT_ACTIONS: usize = 16;
/// Largest raise cap per street supported by the infoset key layout.
pub const MAX_RAISES_CAP: u8 = 4;

/// One abstract action.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AbstractAction {
    Fold,
    CheckCall,
    /// Raise by `fraction` of the pot after calling: raise-to =
    /// `current_bet + fraction * (pot + to_call)`.
    RaisePot(f64),
    AllIn,
}

impl AbstractAction {
    pub fn is_raise(&self) -> bool {
        matches!(self, AbstractAction::RaisePot(_) | AbstractAction::AllIn)
    }
}

/// When a sized raise (`RaisePot`) is offered: always, only as the street's
/// first voluntary raise (an opening bet or raise), or only facing one (a
/// re-raise).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum When {
    #[default]
    Any,
    Open,
    Reraise,
}

impl When {
    /// Whether an entry with this condition is offered after `n_raises`
    /// voluntary raises on the street.
    #[inline]
    pub fn allows(self, n_raises: usize) -> bool {
        match self {
            When::Any => true,
            When::Open => n_raises == 0,
            When::Reraise => n_raises >= 1,
        }
    }
}

/// Concrete actions available in a state under the abstraction, with the
/// index of each in the street's abstract list. Fixed capacity, no allocation.
#[derive(Clone, Copy, Debug)]
pub struct ActionList {
    pub len: usize,
    pub index: [u8; MAX_ABSTRACT_ACTIONS],
    pub action: [Action; MAX_ABSTRACT_ACTIONS],
}

impl ActionList {
    fn new() -> ActionList {
        ActionList { len: 0, index: [0; MAX_ABSTRACT_ACTIONS], action: [Action::fold(); MAX_ABSTRACT_ACTIONS] }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    #[inline]
    pub fn actions(&self) -> &[Action] {
        &self.action[..self.len]
    }
    #[inline]
    pub fn indices(&self) -> &[u8] {
        &self.index[..self.len]
    }
    /// Position of the entry with abstract index `idx`.
    pub fn position_of_index(&self, idx: u8) -> Option<usize> {
        self.indices().iter().position(|&i| i == idx)
    }
    /// Position of the entry equal to the concrete action `a`.
    pub fn position_of_action(&self, a: Action) -> Option<usize> {
        self.actions().iter().position(|&x| x == a)
    }
}

/// Per-street abstract action lists plus a raise cap per street. `when`
/// holds a condition per entry of `streets` (an empty list: all `Any`); only
/// `RaisePot` entries may carry one.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionAbstraction {
    pub streets: [Vec<AbstractAction>; 4],
    pub max_raises: u8,
    pub when: [Vec<When>; 4],
}

impl Default for ActionAbstraction {
    /// The table of DESIGN.md section 5.3. Preflop "2.5x open" and "3x
    /// 3-bet" are pot fractions 0.75 and 1.0: heads-up preflop with no antes
    /// a raise to `x` times the current bet is exactly `(x - 1) / 2` of the
    /// pot after calling, so 2.5x = 0.75 pot and 3x = pot.
    fn default() -> Self {
        use AbstractAction::*;
        ActionAbstraction {
            streets: [
                vec![Fold, CheckCall, RaisePot(0.75), RaisePot(1.0), AllIn],
                vec![Fold, CheckCall, RaisePot(0.33), RaisePot(0.75), RaisePot(1.5), AllIn],
                vec![Fold, CheckCall, RaisePot(0.5), RaisePot(1.0), AllIn],
                vec![Fold, CheckCall, RaisePot(0.5), RaisePot(1.0), RaisePot(2.0), AllIn],
            ],
            max_raises: 4,
            when: Default::default(),
        }
    }
}

/// Pot fraction of a raise-to amount for the player to act.
pub fn raise_fraction(state: &GameState, raise_to: Chips) -> f64 {
    let p = state.current_player().unwrap_or(0);
    let cur = state.current_bet();
    let to_call = (cur - state.street_bets()[p]).max(0);
    let base = (state.pot() + to_call).max(1) as f64;
    (raise_to - cur) as f64 / base
}

/// Raise-to for a pot-fraction raise, before clamping.
pub fn pot_raise_to(state: &GameState, fraction: f64) -> Chips {
    let p = state.current_player().unwrap_or(0);
    let cur = state.current_bet();
    let to_call = (cur - state.street_bets()[p]).max(0);
    cur + (fraction * (state.pot() + to_call) as f64).round() as Chips
}

/// Probability that the pseudo-harmonic mapping (Ganzfried & Sandholm 2013)
/// maps a bet of pot fraction `x` onto the smaller size `a` rather than the
/// larger size `b` (`a <= x <= b`, all as fractions of the pot):
/// `f(x) = (b - x)(1 + a) / ((b - a)(1 + x))`.
pub fn pseudo_harmonic(a: f64, b: f64, x: f64) -> f64 {
    if b <= a {
        return 1.0;
    }
    let x = x.clamp(a, b);
    ((b - x) * (1.0 + a)) / ((b - a) * (1.0 + x))
}

impl ActionAbstraction {
    /// Condition of entry `i` on street `s`.
    #[inline]
    pub fn when_at(&self, s: usize, i: usize) -> When {
        self.when[s].get(i).copied().unwrap_or_default()
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.max_raises == 0 || self.max_raises > MAX_RAISES_CAP {
            return Err(format!("max_raises must be in 1..={MAX_RAISES_CAP}, got {}", self.max_raises));
        }
        for (s, list) in self.streets.iter().enumerate() {
            if list.is_empty() || list.len() > MAX_ABSTRACT_ACTIONS {
                return Err(format!("street {s}: need 1..={MAX_ABSTRACT_ACTIONS} actions, got {}", list.len()));
            }
            if !list.contains(&AbstractAction::CheckCall) {
                return Err(format!("street {s}: check_call must be in the action list"));
            }
            if !self.when[s].is_empty() && self.when[s].len() != list.len() {
                return Err(format!("street {s}: {} conditions for {} actions", self.when[s].len(), list.len()));
            }
            for (i, a) in list.iter().enumerate() {
                if self.when_at(s, i) != When::Any && !matches!(a, AbstractAction::RaisePot(_)) {
                    return Err(format!("street {s}: only sized raises take an open/reraise condition, not {a:?}"));
                }
            }
            for a in list {
                if let AbstractAction::RaisePot(f) = a {
                    if !(f.is_finite() && *f > 0.0) {
                        return Err(format!("street {s}: pot fraction must be positive, got {f}"));
                    }
                }
            }
            for i in 0..list.len() {
                for j in 0..i {
                    if list[i] == list[j] && self.when_at(s, i) == self.when_at(s, j) {
                        return Err(format!("street {s}: duplicate action {:?}", list[i]));
                    }
                }
            }
        }
        Ok(())
    }

    /// Concrete actions available in `state` (non-terminal), in abstract
    /// list order, de-duplicated. Folding is only offered when facing a bet;
    /// raises only when a raise is legal and fewer than `max_raises` raises
    /// were made on this street. Raise sizes are rounded to whole chips and
    /// clamped to the legal range; a size at or above the stack is all-in.
    /// When two abstract raises give the same amount, the all-in entry (if
    /// it is one of them) or else the earliest one is kept.
    pub fn legal(&self, state: &GameState) -> ActionList {
        let mut out = ActionList::new();
        if state.is_terminal() {
            return out;
        }
        let la = state.legal_actions();
        self.legal_with(state, &la, &mut out);
        out
    }

    fn legal_with(&self, state: &GameState, la: &LegalActions, out: &mut ActionList) {
        let s = state.street();
        let list = &self.streets[s];
        let n_raises = state.num_raises_this_street();
        let raises_ok = la.can_raise() && n_raises < self.max_raises as usize;
        let mut is_allin = [false; MAX_ABSTRACT_ACTIONS];
        for (i, a) in list.iter().enumerate() {
            let concrete = match *a {
                AbstractAction::Fold => {
                    if !la.can_fold {
                        continue;
                    }
                    Action::fold()
                }
                AbstractAction::CheckCall => Action::check_call(),
                AbstractAction::RaisePot(f) => {
                    if !raises_ok || !self.when_at(s, i).allows(n_raises) {
                        continue;
                    }
                    Action::raise_to(la.clamp_raise_to(pot_raise_to(state, f)))
                }
                AbstractAction::AllIn => {
                    if !raises_ok {
                        continue;
                    }
                    Action::raise_to(la.max_raise_to)
                }
            };
            let allin = matches!(a, AbstractAction::AllIn);
            // De-duplicate against earlier entries.
            if let Some(j) = out.position_of_action(concrete) {
                if allin && !is_allin[j] {
                    out.index[j] = i as u8;
                    is_allin[j] = true;
                }
                continue;
            }
            let k = out.len;
            out.index[k] = i as u8;
            out.action[k] = concrete;
            is_allin[k] = allin;
            out.len += 1;
        }
    }

    /// Concrete action for abstract index `idx` in `state`, or `None` when
    /// that abstract action is not available there.
    pub fn to_concrete(&self, state: &GameState, idx: u8) -> Option<Action> {
        let l = self.legal(state);
        l.position_of_index(idx).map(|k| l.action[k])
    }

    /// Map a concrete action taken in `real` onto the abstract action list of
    /// `abs` (the same decision point in the abstract game, possibly with a
    /// different pot). Returns the chosen entry's abstract index, or `None`
    /// when `abs` is terminal.
    ///
    /// * fold -> fold (check/call if `abs` cannot fold), check/call -> check/call;
    /// * a raise: an all-in maps to the abstract all-in when there is one;
    ///   otherwise the raise's pot fraction `x` in `real` is compared with
    ///   the pot fractions of the abstract raises in `abs`. Below the
    ///   smallest or above the largest it maps to that size; between two
    ///   neighbours `a < x < b` it maps to `a` with the pseudo-harmonic
    ///   probability ([`pseudo_harmonic`]): to `a` iff `u < f(x)` for a
    ///   uniform `u` in `[0, 1)` (`u = 0.5` is the deterministic nearest
    ///   size under this metric). With no abstract raise left it maps to
    ///   check/call.
    pub fn translate(&self, abs: &GameState, real: &GameState, action: Action, u: f64) -> Option<u8> {
        if abs.is_terminal() {
            return None;
        }
        let l = self.legal(abs);
        let list = &self.streets[abs.street()];
        let find = |pred: &dyn Fn(&AbstractAction) -> bool| -> Option<u8> {
            l.indices().iter().copied().find(|&i| pred(&list[i as usize]))
        };
        let call = find(&|a| *a == AbstractAction::CheckCall);
        match action.kind {
            ActionKind::Fold => find(&|a| *a == AbstractAction::Fold).or(call),
            ActionKind::CheckCall => call,
            ActionKind::Raise => {
                // Abstract raises with their pot fractions in `abs`.
                let mut raises: Vec<(f64, u8, bool)> = Vec::new();
                let abs_la = abs.legal_actions();
                for k in 0..l.len {
                    let a = l.action[k];
                    if a.kind == ActionKind::Raise {
                        raises.push((raise_fraction(abs, a.amount), l.index[k], a.amount == abs_la.max_raise_to));
                    }
                }
                if raises.is_empty() {
                    return call;
                }
                let real_la = real.legal_actions();
                if real_la.can_raise() && action.amount >= real_la.max_raise_to {
                    if let Some(&(_, i, _)) = raises.iter().find(|r| r.2) {
                        return Some(i);
                    }
                }
                raises.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
                let x = raise_fraction(real, action.amount);
                if x <= raises[0].0 {
                    return Some(raises[0].1);
                }
                let last = raises[raises.len() - 1];
                if x >= last.0 {
                    return Some(last.1);
                }
                let k = raises.iter().position(|r| r.0 >= x).unwrap();
                let (a, b) = (raises[k - 1], raises[k]);
                let p = pseudo_harmonic(a.0, b.0, x);
                Some(if u < p { a.1 } else { b.1 })
            }
        }
    }

    /// Abstract indices of the actions in `state`'s history, replayed from
    /// the start of the hand. `Err` names the first action that is not in
    /// the abstraction.
    pub fn sequence(&self, state: &GameState) -> Result<Vec<u8>, String> {
        let mut s = root_like(state);
        let mut out = Vec::with_capacity(state.history().len());
        for h in state.history() {
            let l = self.legal(&s);
            let k = l
                .position_of_action(h.action)
                .ok_or_else(|| format!("action {} on street {} is not in the abstraction", h.action, h.street))?;
            out.push(l.index[k]);
            s.apply(h.action).map_err(|e| e.to_string())?;
        }
        Ok(out)
    }

    /// Count decision nodes of the abstract betting tree per street (cards
    /// do not affect betting). Stops counting (returns `None`) past `limit`
    /// total nodes.
    pub fn count_tree(&self, config: &GameConfig, limit: u64) -> Option<[u64; 4]> {
        let deck: Vec<Card> = (0..52).collect();
        let root = GameState::new_hand(config, 0, &deck).ok()?;
        let mut counts = [0u64; 4];
        let mut total = 0u64;
        let mut stack = vec![root];
        while let Some(s) = stack.pop() {
            if s.is_terminal() {
                continue;
            }
            counts[s.street()] += 1;
            total += 1;
            if total > limit {
                return None;
            }
            let l = self.legal(&s);
            for &a in l.actions() {
                stack.push(s.child(a).ok()?);
            }
        }
        Some(counts)
    }
}

/// The starting state of `state`'s hand (same config and button), dealt
/// from an ordered dummy deck; used to replay betting histories.
pub fn root_like(state: &GameState) -> GameState {
    let config = config_of(state);
    let deck: Vec<Card> = (0..52).collect();
    GameState::new_hand(&config, state.button(), &deck).expect("config of a valid state")
}

/// The configuration a state was started with.
pub fn config_of(state: &GameState) -> GameConfig {
    let n = state.num_players();
    GameConfig {
        num_players: n,
        stacks: (0..n).map(|p| state.stacks()[p] + state.contributed()[p]).collect(),
        small_blind: state.small_blind(),
        big_blind: state.big_blind(),
        ante: state.ante(),
    }
}

// ----------------------------------------------------------------------
// Card abstraction
// ----------------------------------------------------------------------

/// Maps `(street, hole, board)` to a bucket in `0..num_buckets(street)`.
pub trait Bucketer: Send + Sync {
    fn bucket(&self, street: usize, hole: [Card; 2], board: &[Card]) -> u32;
    fn num_buckets(&self, street: usize) -> u32;
}

impl<F: Fn(usize, [Card; 2], &[Card]) -> u32 + Send + Sync> Bucketer for (F, [u32; 4]) {
    fn bucket(&self, street: usize, hole: [Card; 2], board: &[Card]) -> u32 {
        (self.0)(street, hole, board)
    }
    fn num_buckets(&self, street: usize) -> u32 {
        self.1[street]
    }
}

/// Equity (win + half tie) of `hole` against one uniformly random hand.
/// River: exact over all opponent hands. Earlier streets: `samples` Monte
/// Carlo draws of (opponent hand, rest of the board) from `rng`.
pub fn hand_strength(hole: [Card; 2], board: &[Card], samples: u32, rng: &mut Rng) -> f64 {
    let mut used = 0u64;
    for &c in hole.iter().chain(board) {
        used |= 1 << c;
    }
    let mut rest: Vec<Card> = (0..52u8).filter(|c| used & (1 << c) == 0).collect();
    let mut bm = [0u32; 4];
    for &c in board {
        bm[(c & 3) as usize] |= 1 << (c >> 2);
    }
    let add = |m: &mut [u32; 4], c: Card| m[(c & 3) as usize] |= 1 << (c >> 2);
    if board.len() == 5 {
        let mut hm = bm;
        add(&mut hm, hole[0]);
        add(&mut hm, hole[1]);
        let hero = evaluate_masks(hm);
        let mut score = 0.0f64;
        let mut n = 0u32;
        for i in 0..rest.len() {
            for j in (i + 1)..rest.len() {
                let mut om = bm;
                add(&mut om, rest[i]);
                add(&mut om, rest[j]);
                let opp = evaluate_masks(om);
                score += if hero > opp {
                    1.0
                } else if hero == opp {
                    0.5
                } else {
                    0.0
                };
                n += 1;
            }
        }
        return score / n as f64;
    }
    let need = 2 + 5 - board.len();
    let mut score = 0.0f64;
    let samples = samples.max(1);
    for _ in 0..samples {
        rng.partial_shuffle(&mut rest, need);
        let mut full = bm;
        for &c in &rest[2..need] {
            add(&mut full, c);
        }
        let mut hm = full;
        add(&mut hm, hole[0]);
        add(&mut hm, hole[1]);
        let mut om = full;
        add(&mut om, rest[0]);
        add(&mut om, rest[1]);
        let (h, o) = (evaluate_masks(hm), evaluate_masks(om));
        score += if h > o {
            1.0
        } else if h == o {
            0.5
        } else {
            0.0
        };
    }
    score / samples as f64
}

/// Specification of a card abstraction.
#[derive(Clone, Debug, PartialEq)]
pub struct CardAbstractionSpec {
    /// Buckets per street. Preflop 169 means the lossless classes; fewer
    /// buckets group the 169 classes by equity against a random hand into
    /// equal-probability buckets.
    pub buckets: [u32; 4],
    /// Monte Carlo samples for the default flop/turn hand strength.
    pub hs_samples: u32,
    /// Optional `.npy` bucket tables per street (1-D, indexed by
    /// [`canonical_index`], integer dtype); `None` uses the default.
    pub tables: [Option<String>; 4],
}

impl Default for CardAbstractionSpec {
    fn default() -> Self {
        CardAbstractionSpec { buckets: [169, 1000, 1000, 1000], hs_samples: 256, tables: [None, None, None, None] }
    }
}

/// Card abstraction: preflop classes plus per-street tables or the default
/// quantized hand-strength buckets (computed lazily per canonical class and
/// cached, so every isomorphic hand gets the same deterministic bucket).
pub struct CardAbstraction {
    pub spec: CardAbstractionSpec,
    preflop: [u16; 169],
    tables: [Option<Vec<u32>>; 4],
    cache: [OnceLock<LazyTable>; 4],
}

/// Lazily filled `u16` table (0 = not computed, else bucket + 1), zeroed by
/// the allocator so untouched pages cost no memory.
struct LazyTable {
    data: Box<[AtomicU16]>,
}

impl LazyTable {
    fn new(n: usize) -> LazyTable {
        let layout = std::alloc::Layout::array::<AtomicU16>(n).expect("table size");
        // SAFETY: AtomicU16 has the same layout as u16 and all-zero bytes are
        // a valid value; the pointer comes from the global allocator with the
        // matching layout and is owned by the returned Box.
        let data = unsafe {
            let ptr = std::alloc::alloc_zeroed(layout) as *mut AtomicU16;
            if ptr.is_null() {
                std::alloc::handle_alloc_error(layout);
            }
            Box::from_raw(std::ptr::slice_from_raw_parts_mut(ptr, n))
        };
        LazyTable { data }
    }
}

impl CardAbstraction {
    /// Build from a spec; `.npy` tables named in the spec are loaded here.
    pub fn new(spec: CardAbstractionSpec) -> Result<CardAbstraction, String> {
        let mut tables: [Option<Vec<u32>>; 4] = [None, None, None, None];
        for s in 0..4 {
            if let Some(path) = &spec.tables[s] {
                tables[s] = Some(crate::npy::read_npy_u32(path)?);
            }
        }
        Self::with_tables(spec, tables)
    }

    /// Build from a spec with in-memory tables (the spec's paths are kept
    /// for the record but not read).
    pub fn with_tables(spec: CardAbstractionSpec, tables: [Option<Vec<u32>>; 4]) -> Result<CardAbstraction, String> {
        for s in 0..4 {
            let n = spec.buckets[s];
            if n == 0 || n > 65_000 {
                return Err(format!("street {s}: bucket count must be in 1..=65000, got {n}"));
            }
            if let Some(t) = &tables[s] {
                let want = CANONICAL_SIZES[s] as usize;
                if t.len() != want {
                    return Err(format!("street {s}: bucket table has {} entries, expected {want}", t.len()));
                }
                if let Some(&m) = t.iter().max() {
                    if m >= n {
                        return Err(format!("street {s}: bucket table holds bucket {m} >= {n} buckets"));
                    }
                }
            }
        }
        if tables[0].is_none() && spec.buckets[0] > 169 {
            return Err("preflop has at most 169 buckets without a table".into());
        }
        let mut preflop = [0u16; 169];
        let nb = spec.buckets[0];
        if nb == 169 {
            for (c, b) in preflop.iter_mut().enumerate() {
                *b = c as u16;
            }
        } else {
            // Rank the 169 classes by equity vs a random hand, then cut into
            // equal-probability buckets (weighted by combos).
            let mut eq: Vec<(f64, u32)> = (0..169u32)
                .map(|c| {
                    let mut rng = Rng::new(0x5EED_0000 + c as u64);
                    (hand_strength(preflop_class_hand(c), &[], 4000, &mut rng), c)
                })
                .collect();
            eq.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut cum = 0u32;
            for &(_, c) in &eq {
                let w = preflop_class_combos(c);
                let mid = cum as f64 + w as f64 / 2.0;
                preflop[c as usize] = ((mid / 1326.0 * nb as f64) as u32).min(nb - 1) as u16;
                cum += w;
            }
        }
        Ok(CardAbstraction {
            spec,
            preflop,
            tables,
            cache: [OnceLock::new(), OnceLock::new(), OnceLock::new(), OnceLock::new()],
        })
    }

    /// Default postflop bucket: quantized hand strength of the canonical
    /// representative (river exact, flop/turn Monte Carlo seeded by the
    /// canonical index), `floor(hs * buckets)`.
    fn default_bucket(&self, street: usize, idx: u32) -> u32 {
        let (h, b) = canonical_unindex(street, idx).expect("valid canonical index");
        let mut rng = Rng::new(((street as u64) << 32) ^ idx as u64 ^ 0x00B0_C4E7);
        let hs = hand_strength(h, &b, self.spec.hs_samples, &mut rng);
        let n = self.spec.buckets[street];
        ((hs * n as f64) as u32).min(n - 1)
    }

    /// Bucket of a canonical index (street > 0).
    pub fn bucket_of_index(&self, street: usize, idx: u32) -> u32 {
        if let Some(t) = &self.tables[street] {
            return t[idx as usize];
        }
        if street == 0 {
            let (h, _) = canonical_unindex(0, idx).expect("valid index");
            return self.preflop[preflop_class(h) as usize] as u32;
        }
        let cache = self.cache[street].get_or_init(|| LazyTable::new(CANONICAL_SIZES[street] as usize));
        let slot = &cache.data[idx as usize];
        let v = slot.load(Ordering::Relaxed);
        if v != 0 {
            return (v - 1) as u32;
        }
        let b = self.default_bucket(street, idx);
        slot.store((b + 1) as u16, Ordering::Relaxed);
        b
    }
}

impl Bucketer for CardAbstraction {
    fn bucket(&self, street: usize, hole: [Card; 2], board: &[Card]) -> u32 {
        if street == 0 && self.tables[0].is_none() {
            return self.preflop[preflop_class(hole) as usize] as u32;
        }
        let nb = board_len(street);
        self.bucket_of_index(street, canonical_index(street, hole, &board[..nb]))
    }

    fn num_buckets(&self, street: usize) -> u32 {
        self.spec.buckets[street]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cards::cards_from_str;

    fn cfg(stack: Chips) -> GameConfig {
        GameConfig::new(2, stack, 50, 100, 0)
    }

    fn deck() -> Vec<Card> {
        (0..52).collect()
    }

    #[test]
    fn default_preflop_sizes() {
        let a = ActionAbstraction::default();
        a.validate().unwrap();
        let s = GameState::new_hand(&cfg(20_000), 0, &deck()).unwrap();
        let l = a.legal(&s);
        // SB facing the big blind: fold, call, 2.5x (250), 3x (300), all-in.
        assert_eq!(
            l.actions(),
            &[
                Action::fold(),
                Action::check_call(),
                Action::raise_to(250),
                Action::raise_to(300),
                Action::raise_to(20_000)
            ]
        );
        assert_eq!(l.indices(), &[0, 1, 2, 3, 4]);
        // Facing a 250 open: 0.75 pot = raise to 250 + 0.75 * 500 = 625 (2.5x), pot = 750 (3x).
        let s2 = s.child(Action::raise_to(250)).unwrap();
        let l2 = a.legal(&s2);
        assert_eq!(l2.actions()[2], Action::raise_to(625));
        assert_eq!(l2.actions()[3], Action::raise_to(750));
    }

    #[test]
    fn dedup_and_all_in_clamp() {
        let a = ActionAbstraction::default();
        // 1,000-chip stacks: after an open to 300, 0.75 pot (300 + 450 = 750)
        // stays, pot (900) stays, all-in 1000.
        let s = GameState::new_hand(&cfg(1_000), 0, &deck()).unwrap().child(Action::raise_to(300)).unwrap();
        let l = a.legal(&s);
        assert_eq!(l.len, 5);
        // 600-chip stacks: both pot sizes exceed the stack and merge into all-in.
        let s = GameState::new_hand(&cfg(600), 0, &deck()).unwrap().child(Action::raise_to(300)).unwrap();
        let l = a.legal(&s);
        assert_eq!(l.actions(), &[Action::fold(), Action::check_call(), Action::raise_to(600)]);
        assert_eq!(l.indices(), &[0, 1, 4], "all-in keeps its own index");
        // Two sizes clamped up to the same min-raise keep the first.
        let tiny = ActionAbstraction {
            streets: [
                vec![AbstractAction::CheckCall, AbstractAction::RaisePot(0.01), AbstractAction::RaisePot(0.02)],
                vec![AbstractAction::CheckCall],
                vec![AbstractAction::CheckCall],
                vec![AbstractAction::CheckCall],
            ],
            max_raises: 4,
            when: Default::default(),
        };
        let s = GameState::new_hand(&cfg(20_000), 0, &deck()).unwrap();
        let l = tiny.legal(&s);
        assert_eq!(l.actions(), &[Action::check_call(), Action::raise_to(200)]);
        assert_eq!(l.indices(), &[0, 1]);
    }

    #[test]
    fn open_and_reraise_conditions() {
        use AbstractAction::*;
        let flop = vec![Fold, CheckCall, RaisePot(0.33), RaisePot(1.0), RaisePot(1.0), AllIn];
        let flop_when = vec![When::Any, When::Any, When::Open, When::Open, When::Reraise, When::Any];
        let a = ActionAbstraction {
            streets: [vec![Fold, CheckCall, RaisePot(1.0), AllIn], flop.clone(), flop.clone(), flop],
            max_raises: 4,
            when: [vec![], flop_when.clone(), flop_when.clone(), flop_when],
        };
        a.validate().unwrap();
        // Preflop limp and check -> flop, first to act: the open sizes, not the reraise one.
        let mut s = GameState::new_hand(&cfg(20_000), 0, &deck()).unwrap();
        s.apply(Action::check_call()).unwrap();
        s.apply(Action::check_call()).unwrap();
        assert_eq!(s.street(), 1);
        assert_eq!(a.legal(&s).indices(), &[1, 2, 3, 5]);
        // Facing a bet: only the reraise size (plus fold, call and all-in).
        s.apply(Action::raise_to(150)).unwrap();
        assert_eq!(a.legal(&s).indices(), &[0, 1, 4, 5]);
        // A condition on a non-raise entry is rejected; equal sizes need different conditions.
        let mut bad = a.clone();
        bad.when[1][5] = When::Open;
        assert!(bad.validate().is_err());
        let mut dup = a.clone();
        dup.when[1][4] = When::Open;
        assert!(dup.validate().is_err());
    }

    #[test]
    fn raise_cap() {
        let a = ActionAbstraction { max_raises: 2, ..Default::default() };
        let mut s = GameState::new_hand(&cfg(20_000), 0, &deck()).unwrap();
        s.apply(Action::raise_to(300)).unwrap();
        assert!(a.legal(&s).actions().iter().any(|x| x.kind == ActionKind::Raise));
        s.apply(Action::raise_to(900)).unwrap();
        let l = a.legal(&s);
        assert_eq!(l.actions(), &[Action::fold(), Action::check_call()]);
    }

    #[test]
    fn pseudo_harmonic_values() {
        assert!((pseudo_harmonic(0.5, 1.0, 0.75) - 0.375 / 0.875).abs() < 1e-12);
        assert_eq!(pseudo_harmonic(0.5, 1.0, 0.5), 1.0);
        assert_eq!(pseudo_harmonic(0.5, 1.0, 1.0), 0.0);
        // Midpoint in the harmonic sense: f = 1/2 at x = (a + b + 2ab) / (2 + a + b).
        let (a, b) = (0.33, 1.5);
        let x = (a + b + 2.0 * a * b) / (2.0 + a + b);
        assert!((pseudo_harmonic(a, b, x) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn translate_and_sequence() {
        let a = ActionAbstraction::default();
        let real = GameState::new_hand(&cfg(20_000), 0, &deck()).unwrap();
        let abs = real.clone();
        // Exact sizes map to themselves.
        assert_eq!(a.translate(&abs, &real, Action::raise_to(250), 0.99), Some(2));
        assert_eq!(a.translate(&abs, &real, Action::raise_to(300), 0.0), Some(3));
        // Below the smallest size -> smallest; huge non-all-in -> all-in (largest).
        assert_eq!(a.translate(&abs, &real, Action::raise_to(200), 0.0), Some(2));
        assert_eq!(a.translate(&abs, &real, Action::raise_to(19_000), 0.5), Some(4));
        assert_eq!(a.translate(&abs, &real, Action::raise_to(20_000), 0.0), Some(4));
        // 275 is x = 0.875 between 0.75 and 1.0.
        let p = pseudo_harmonic(0.75, 1.0, 0.875);
        assert_eq!(a.translate(&abs, &real, Action::raise_to(275), p - 1e-9), Some(2));
        assert_eq!(a.translate(&abs, &real, Action::raise_to(275), p + 1e-9), Some(3));
        assert_eq!(a.translate(&abs, &real, Action::fold(), 0.5), Some(0));
        assert_eq!(a.translate(&abs, &real, Action::check_call(), 0.5), Some(1));
        // Sequence replay.
        let mut s = real.clone();
        s.apply(Action::raise_to(250)).unwrap();
        s.apply(Action::check_call()).unwrap();
        s.apply(Action::check_call()).unwrap();
        assert_eq!(a.sequence(&s).unwrap(), vec![2, 1, 1]);
        let mut off = real.clone();
        off.apply(Action::raise_to(260)).unwrap();
        assert!(a.sequence(&off).is_err());
    }

    #[test]
    fn default_buckets_are_isomorphism_invariant_and_ordered() {
        let spec = CardAbstractionSpec { buckets: [8, 10, 10, 10], hs_samples: 64, tables: Default::default() };
        let ca = CardAbstraction::new(spec).unwrap();
        let c = |s: &str| cards_from_str(s).unwrap();
        let aa = c("AsAh");
        let low = c("7c2d");
        assert_eq!(ca.bucket(0, [aa[0], aa[1]], &[]), 7);
        assert_eq!(ca.bucket(0, [low[0], low[1]], &[]), 0);
        // River nuts vs air.
        let board = c("AdKdQd2s3c");
        let nuts = c("JdTd");
        let air = c("7h4c");
        assert_eq!(ca.bucket(3, [nuts[0], nuts[1]], &board), 9);
        assert!(ca.bucket(3, [air[0], air[1]], &board) <= 1);
        // Suit permutation gives the same flop bucket.
        let h1 = c("AhKh");
        let b1 = c("Qh7h2c");
        let h2 = c("AsKs");
        let b2 = c("2d7sQs");
        assert_eq!(ca.bucket(1, [h1[0], h1[1]], &b1), ca.bucket(1, [h2[0], h2[1]], &b2));
    }

    #[test]
    fn tree_counts_small() {
        let a = ActionAbstraction::default();
        let c = a.count_tree(&cfg(1_000), 10_000_000).unwrap();
        assert!(c[0] > 0 && c[1] > 0 && c[3] > 0);
    }
}
