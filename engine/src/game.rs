//! No-Limit Texas Hold'em rules for 2..=9 players.
//!
//! See `README.md` for the precise rule choices (min-raise tracking,
//! re-opening of action, short blinds, odd chips, public/infoset keys).

use std::fmt;

use crate::cards::{card_to_str, validate_cards, Card, NO_CARD};
use crate::eval::{evaluate_hole_board, HandRank};

/// Chip amounts. Totals are limited to `i32::MAX` by [`GameConfig::validate`].
pub type Chips = i64;

/// Largest supported table.
pub const MAX_PLAYERS: usize = 9;
/// Cards dealt for the largest table: two hole cards per seat plus the board.
pub const MAX_DEAL: usize = 2 * MAX_PLAYERS + 5;

/// Action kind codes (match the Python constants).
pub const FOLD: u8 = 0;
pub const CHECK_CALL: u8 = 1;
pub const RAISE: u8 = 2;

/// Number of board cards visible on each street.
pub const BOARD_LEN: [u8; 4] = [0, 3, 4, 5];

/// Error returned for invalid configurations, decks and illegal actions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GameError {
    InvalidConfig(String),
    InvalidDeck(String),
    IllegalAction(String),
    NotTerminal,
    HiddenCards(String),
    InvalidPlayer(usize),
}

impl fmt::Display for GameError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GameError::InvalidConfig(s) => write!(f, "invalid config: {s}"),
            GameError::InvalidDeck(s) => write!(f, "invalid deck: {s}"),
            GameError::IllegalAction(s) => write!(f, "illegal action: {s}"),
            GameError::NotTerminal => write!(f, "payoffs requested for a non-terminal state"),
            GameError::HiddenCards(s) => write!(f, "hidden cards: {s}"),
            GameError::InvalidPlayer(p) => write!(f, "invalid player index {p}"),
        }
    }
}

impl std::error::Error for GameError {}

/// Kind of a concrete action.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum ActionKind {
    Fold = FOLD,
    CheckCall = CHECK_CALL,
    Raise = RAISE,
}

impl ActionKind {
    pub fn from_u8(k: u8) -> Option<ActionKind> {
        match k {
            FOLD => Some(ActionKind::Fold),
            CHECK_CALL => Some(ActionKind::CheckCall),
            RAISE => Some(ActionKind::Raise),
            _ => None,
        }
    }
}

/// A concrete action. For raises `amount` is the total this player will have
/// committed on the current street after the action ("raise to"); it is 0
/// for folds and check/calls.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Action {
    pub kind: ActionKind,
    pub amount: Chips,
}

impl Action {
    pub const fn fold() -> Action {
        Action { kind: ActionKind::Fold, amount: 0 }
    }
    pub const fn check_call() -> Action {
        Action { kind: ActionKind::CheckCall, amount: 0 }
    }
    pub const fn raise_to(amount: Chips) -> Action {
        Action { kind: ActionKind::Raise, amount }
    }
}

impl fmt::Display for Action {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.kind {
            ActionKind::Fold => f.write_str("f"),
            ActionKind::CheckCall => f.write_str("c"),
            ActionKind::Raise => write!(f, "r{}", self.amount),
        }
    }
}

/// The legal actions of the player to act.
///
/// A raise is legal iff `min_raise_to > 0`; any raise-to in
/// `[min_raise_to, max_raise_to]` is legal, plus `max_raise_to` itself even
/// when it is below `min_raise_to` (all-in for less than a full raise).
/// In a terminal state every field is false/zero.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
pub struct LegalActions {
    /// False when checking is possible.
    pub can_fold: bool,
    pub can_check: bool,
    /// Chips added by a call (0 when checking; capped at the stack).
    pub call_amount: Chips,
    /// Smallest full raise-to; 0 when no raise is legal.
    pub min_raise_to: Chips,
    /// All-in raise-to; 0 when no raise is legal.
    pub max_raise_to: Chips,
}

impl LegalActions {
    /// Whether any raise is legal.
    #[inline]
    pub fn can_raise(&self) -> bool {
        self.min_raise_to > 0
    }

    /// Whether check/call is legal (always, unless terminal).
    #[inline]
    pub fn can_check_call(&self) -> bool {
        self.can_check || self.can_fold
    }

    /// Whether `a` would be accepted by [`GameState::apply`].
    pub fn is_legal(&self, a: Action) -> bool {
        match a.kind {
            ActionKind::Fold => self.can_fold && a.amount == 0,
            ActionKind::CheckCall => self.can_check_call() && a.amount == 0,
            ActionKind::Raise => {
                self.can_raise()
                    && (a.amount == self.max_raise_to
                        || (a.amount >= self.min_raise_to && a.amount <= self.max_raise_to))
            }
        }
    }

    /// The legal raise-to closest to `amount` (0 if no raise is legal).
    pub fn clamp_raise_to(&self, amount: Chips) -> Chips {
        if !self.can_raise() {
            return 0;
        }
        if self.min_raise_to >= self.max_raise_to || amount >= self.max_raise_to {
            return self.max_raise_to;
        }
        amount.max(self.min_raise_to)
    }
}

/// Table configuration.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GameConfig {
    pub num_players: usize,
    /// Starting stacks in chips, per seat.
    pub stacks: Vec<Chips>,
    pub small_blind: Chips,
    pub big_blind: Chips,
    pub ante: Chips,
}

impl Default for GameConfig {
    /// Heads-up, 20,000-chip stacks, 50/100 blinds, no ante.
    fn default() -> Self {
        GameConfig::new(2, 20_000, 50, 100, 0)
    }
}

impl GameConfig {
    /// A table of `num_players` with equal stacks.
    pub fn new(num_players: usize, stack: Chips, small_blind: Chips, big_blind: Chips, ante: Chips) -> Self {
        GameConfig { num_players, stacks: vec![stack; num_players], small_blind, big_blind, ante }
    }

    /// Check the configuration.
    pub fn validate(&self) -> Result<(), GameError> {
        let n = self.num_players;
        if !(2..=MAX_PLAYERS).contains(&n) {
            return Err(GameError::InvalidConfig(format!("num_players must be in 2..={MAX_PLAYERS}, got {n}")));
        }
        if self.stacks.len() != n {
            return Err(GameError::InvalidConfig(format!(
                "stacks has {} entries for {n} players",
                self.stacks.len()
            )));
        }
        if self.stacks.iter().any(|&s| s <= 0) {
            return Err(GameError::InvalidConfig("every stack must be positive".into()));
        }
        if self.big_blind <= 0 {
            return Err(GameError::InvalidConfig("big_blind must be positive".into()));
        }
        if self.small_blind < 0 || self.small_blind > self.big_blind {
            return Err(GameError::InvalidConfig("small_blind must be in 0..=big_blind".into()));
        }
        if self.ante < 0 {
            return Err(GameError::InvalidConfig("ante must be non-negative".into()));
        }
        let total: i128 = self.stacks.iter().map(|&s| s as i128).sum();
        if total > i32::MAX as i128 {
            return Err(GameError::InvalidConfig("total chips must fit in a signed 32-bit integer".into()));
        }
        Ok(())
    }
}

/// One entry of the betting history.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct HistoryEntry {
    pub street: u8,
    pub player: u8,
    pub action: Action,
}

/// Full state of one hand. Cheap to clone: fixed-size arrays plus the
/// betting-history vector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GameState {
    n: u8,
    button: u8,
    street: u8,
    board_len: u8,
    /// Seat to act, or -1 when terminal.
    current: i8,
    terminal: bool,
    /// True for a state produced by [`GameState::masked`].
    masked: bool,
    raises_this_street: u8,
    small_blind: Chips,
    big_blind: Chips,
    ante: Chips,
    /// Dealt cards: hole cards of seat p at `2p, 2p+1`, board at `2n..2n+5`.
    cards: [Card; MAX_DEAL],
    stacks: [Chips; MAX_PLAYERS],
    street_bets: [Chips; MAX_PLAYERS],
    contributed: [Chips; MAX_PLAYERS],
    /// Street bet level (`current_bet`) right after the player's last action
    /// on this street; used for re-opening of raises.
    acted_level: [Chips; MAX_PLAYERS],
    folded: [bool; MAX_PLAYERS],
    all_in: [bool; MAX_PLAYERS],
    needs_action: [bool; MAX_PLAYERS],
    acted: [bool; MAX_PLAYERS],
    /// Highest street bet the players must match.
    current_bet: Chips,
    /// Size of the last full bet/raise increment on this street (starts at
    /// the big blind); the minimum raise increment.
    last_raise: Chips,
    history: Vec<HistoryEntry>,
}

impl GameState {
    /// Start a hand: post antes and blinds and deal from `deck`.
    ///
    /// `deck` must hold at least `2 * num_players + 5` distinct cards in
    /// `0..52` (normally a full permutation); cards are dealt in order: two
    /// hole cards to seat 0, seat 1, ..., then flop, turn, river.
    pub fn new_hand(config: &GameConfig, button: usize, deck: &[Card]) -> Result<GameState, GameError> {
        config.validate()?;
        let n = config.num_players;
        if button >= n {
            return Err(GameError::InvalidConfig(format!("button {button} out of range for {n} players")));
        }
        let need = 2 * n + 5;
        if deck.len() < need || deck.len() > 52 {
            return Err(GameError::InvalidDeck(format!(
                "deck must hold between {need} and 52 cards, got {}",
                deck.len()
            )));
        }
        validate_cards(deck).map_err(|e| GameError::InvalidDeck(e.0))?;

        let mut s = GameState {
            n: n as u8,
            button: button as u8,
            street: 0,
            board_len: 0,
            current: -1,
            terminal: false,
            masked: false,
            raises_this_street: 0,
            small_blind: config.small_blind,
            big_blind: config.big_blind,
            ante: config.ante,
            cards: [NO_CARD; MAX_DEAL],
            stacks: [0; MAX_PLAYERS],
            street_bets: [0; MAX_PLAYERS],
            contributed: [0; MAX_PLAYERS],
            acted_level: [0; MAX_PLAYERS],
            folded: [false; MAX_PLAYERS],
            all_in: [false; MAX_PLAYERS],
            needs_action: [false; MAX_PLAYERS],
            acted: [false; MAX_PLAYERS],
            current_bet: config.big_blind,
            last_raise: config.big_blind,
            history: Vec::with_capacity(16),
        };
        s.cards[..need].copy_from_slice(&deck[..need]);
        s.stacks[..n].copy_from_slice(&config.stacks);

        // Antes are dead money: they go to the pot but not to street bets.
        if config.ante > 0 {
            for p in 0..n {
                let a = config.ante.min(s.stacks[p]);
                s.stacks[p] -= a;
                s.contributed[p] += a;
                if s.stacks[p] == 0 {
                    s.all_in[p] = true;
                }
            }
        }
        let (sb_seat, bb_seat, first) = if n == 2 {
            (button, (button + 1) % 2, button)
        } else {
            ((button + 1) % n, (button + 2) % n, (button + 3) % n)
        };
        let sb = config.small_blind.min(s.stacks[sb_seat]);
        s.put(sb_seat, sb);
        let bb = config.big_blind.min(s.stacks[bb_seat]);
        s.put(bb_seat, bb);
        for p in 0..n {
            s.needs_action[p] = s.can_act(p);
        }
        s.settle(first);
        Ok(s)
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    #[inline]
    pub fn num_players(&self) -> usize {
        self.n as usize
    }
    #[inline]
    pub fn button(&self) -> usize {
        self.button as usize
    }
    /// 0 preflop, 1 flop, 2 turn, 3 river.
    #[inline]
    pub fn street(&self) -> usize {
        self.street as usize
    }
    /// Visible board cards (0, 3, 4 or 5).
    #[inline]
    pub fn board(&self) -> &[Card] {
        let start = 2 * self.n as usize;
        &self.cards[start..start + self.board_len as usize]
    }
    /// Hole cards of `player` in deal order; `None` when hidden (masked state).
    #[inline]
    pub fn hole_cards(&self, player: usize) -> Option<[Card; 2]> {
        if player >= self.n as usize {
            return None;
        }
        let h = [self.cards[2 * player], self.cards[2 * player + 1]];
        if h[0] == NO_CARD {
            None
        } else {
            Some(h)
        }
    }
    /// Chips behind, per seat. Not updated with winnings at the end of the
    /// hand; use [`GameState::payoffs`].
    #[inline]
    pub fn stacks(&self) -> &[Chips] {
        &self.stacks[..self.n as usize]
    }
    /// Chips committed on the current street, per seat.
    #[inline]
    pub fn street_bets(&self) -> &[Chips] {
        &self.street_bets[..self.n as usize]
    }
    /// Chips committed in the whole hand (antes included), per seat.
    #[inline]
    pub fn contributed(&self) -> &[Chips] {
        &self.contributed[..self.n as usize]
    }
    /// All chips committed by everyone so far, including the current street.
    #[inline]
    pub fn pot(&self) -> Chips {
        self.contributed().iter().sum()
    }
    /// Seat to act, `None` when terminal.
    #[inline]
    pub fn current_player(&self) -> Option<usize> {
        if self.current < 0 {
            None
        } else {
            Some(self.current as usize)
        }
    }
    #[inline]
    pub fn is_terminal(&self) -> bool {
        self.terminal
    }
    #[inline]
    pub fn folded(&self) -> &[bool] {
        &self.folded[..self.n as usize]
    }
    #[inline]
    pub fn all_in(&self) -> &[bool] {
        &self.all_in[..self.n as usize]
    }
    #[inline]
    pub fn history(&self) -> &[HistoryEntry] {
        &self.history
    }
    /// The street bet level every player must match.
    #[inline]
    pub fn current_bet(&self) -> Chips {
        self.current_bet
    }
    /// The minimum raise increment on this street.
    #[inline]
    pub fn last_raise_size(&self) -> Chips {
        self.last_raise
    }
    /// Number of raises (bets included) on the current street.
    #[inline]
    pub fn num_raises_this_street(&self) -> usize {
        self.raises_this_street as usize
    }
    #[inline]
    pub fn small_blind(&self) -> Chips {
        self.small_blind
    }
    #[inline]
    pub fn big_blind(&self) -> Chips {
        self.big_blind
    }
    #[inline]
    pub fn ante(&self) -> Chips {
        self.ante
    }
    #[inline]
    pub fn is_masked(&self) -> bool {
        self.masked
    }
    /// Players who have not folded.
    pub fn num_active(&self) -> usize {
        self.folded().iter().filter(|&&f| !f).count()
    }
    /// Players who have neither folded nor gone all-in.
    pub fn num_can_act(&self) -> usize {
        (0..self.n as usize).filter(|&p| self.can_act(p)).count()
    }

    // ------------------------------------------------------------------
    // Rules
    // ------------------------------------------------------------------

    #[inline(always)]
    fn can_act(&self, p: usize) -> bool {
        !self.folded[p] && !self.all_in[p]
    }

    #[inline(always)]
    fn put(&mut self, p: usize, chips: Chips) {
        debug_assert!(chips >= 0 && chips <= self.stacks[p]);
        self.stacks[p] -= chips;
        self.street_bets[p] += chips;
        self.contributed[p] += chips;
        if self.stacks[p] == 0 {
            self.all_in[p] = true;
        }
    }

    /// Whether seat `p` still has to act in the current betting round.
    /// A lone player who can still act only acts when facing a bet.
    #[inline(always)]
    fn wants_action(&self, p: usize, n_can_act: usize) -> bool {
        self.needs_action[p] && self.can_act(p) && (n_can_act >= 2 || self.street_bets[p] < self.current_bet)
    }

    /// Set `current` to the next seat (clockwise from `from`, inclusive)
    /// that must act; close betting rounds, deal streets, run out the board
    /// and finish the hand as needed.
    fn settle(&mut self, mut from: usize) {
        let n = self.n as usize;
        loop {
            if self.num_active() <= 1 {
                self.finish();
                return;
            }
            let n_can_act = self.num_can_act();
            for k in 0..n {
                let q = (from + k) % n;
                if self.wants_action(q, n_can_act) {
                    self.current = q as i8;
                    return;
                }
            }
            // Betting round closed.
            if self.street == 3 {
                self.finish();
                return;
            }
            if n_can_act < 2 {
                // Run out the remaining board; no more betting is possible.
                self.street = 3;
                self.board_len = 5;
                self.street_bets = [0; MAX_PLAYERS];
                self.finish();
                return;
            }
            self.street += 1;
            self.board_len = BOARD_LEN[self.street as usize];
            self.street_bets = [0; MAX_PLAYERS];
            self.acted = [false; MAX_PLAYERS];
            self.acted_level = [0; MAX_PLAYERS];
            self.current_bet = 0;
            self.last_raise = self.big_blind;
            self.raises_this_street = 0;
            for p in 0..n {
                self.needs_action[p] = self.can_act(p);
            }
            from = (self.button as usize + 1) % n;
        }
    }

    fn finish(&mut self) {
        self.terminal = true;
        self.current = -1;
        self.needs_action = [false; MAX_PLAYERS];
    }

    /// Legal actions of the player to act.
    pub fn legal_actions(&self) -> LegalActions {
        if self.terminal {
            return LegalActions::default();
        }
        let p = self.current as usize;
        let to_call = (self.current_bet - self.street_bets[p]).max(0);
        let stack = self.stacks[p];
        let mut la = LegalActions {
            can_fold: to_call > 0,
            can_check: to_call == 0,
            call_amount: to_call.min(stack),
            min_raise_to: 0,
            max_raise_to: 0,
        };
        let all_in_to = self.street_bets[p] + stack;
        let reopened = !self.acted[p] || self.current_bet - self.acted_level[p] >= self.last_raise;
        if all_in_to > self.current_bet && reopened && self.others_can_act(p) {
            la.min_raise_to = self.current_bet + self.last_raise;
            la.max_raise_to = all_in_to;
        }
        la
    }

    fn others_can_act(&self, p: usize) -> bool {
        (0..self.n as usize).any(|q| q != p && self.can_act(q))
    }

    /// Apply `action` for the player to act. On error the state is unchanged.
    pub fn apply(&mut self, action: Action) -> Result<(), GameError> {
        if self.masked {
            // Never deal hidden cards: apply on a copy and check.
            let mut next = self.clone();
            next.apply_inner(action)?;
            if next.board().contains(&NO_CARD) {
                return Err(GameError::HiddenCards(
                    "cannot deal a street from a masked state".into(),
                ));
            }
            *self = next;
            return Ok(());
        }
        self.apply_inner(action)
    }

    fn apply_inner(&mut self, action: Action) -> Result<(), GameError> {
        if self.terminal {
            return Err(GameError::IllegalAction("the hand is over".into()));
        }
        let p = self.current as usize;
        let n = self.n as usize;
        let to_call = (self.current_bet - self.street_bets[p]).max(0);
        match action.kind {
            ActionKind::Fold => {
                if action.amount != 0 {
                    return Err(GameError::IllegalAction("fold takes no amount".into()));
                }
                if to_call == 0 {
                    return Err(GameError::IllegalAction("cannot fold when checking is possible".into()));
                }
                self.folded[p] = true;
            }
            ActionKind::CheckCall => {
                if action.amount != 0 {
                    return Err(GameError::IllegalAction("check/call takes no amount".into()));
                }
                let pay = to_call.min(self.stacks[p]);
                self.put(p, pay);
            }
            ActionKind::Raise => {
                let la = self.legal_actions();
                if !la.can_raise() {
                    return Err(GameError::IllegalAction("no raise is legal".into()));
                }
                if !la.is_legal(action) {
                    return Err(GameError::IllegalAction(format!(
                        "raise to {} not in [{}, {}] and not all-in",
                        action.amount, la.min_raise_to, la.max_raise_to
                    )));
                }
                let to = action.amount;
                self.put(p, to - self.street_bets[p]);
                let size = to - self.current_bet;
                if size >= self.last_raise {
                    self.last_raise = size;
                }
                self.current_bet = to;
                self.raises_this_street = self.raises_this_street.saturating_add(1);
                for q in 0..n {
                    if q != p && self.can_act(q) {
                        self.needs_action[q] = true;
                    }
                }
            }
        }
        self.acted[p] = true;
        self.acted_level[p] = self.current_bet;
        self.needs_action[p] = false;
        self.history.push(HistoryEntry { street: self.street, player: p as u8, action });
        self.settle((p + 1) % n);
        Ok(())
    }

    /// A copy with `action` applied.
    pub fn child(&self, action: Action) -> Result<GameState, GameError> {
        let mut c = self.clone();
        c.apply(action)?;
        Ok(c)
    }

    /// Net chip change per seat (terminal states only). Sums to zero.
    pub fn payoffs(&self) -> Result<Vec<Chips>, GameError> {
        let mut out = [0; MAX_PLAYERS];
        self.payoffs_into(&mut out)?;
        Ok(out[..self.n as usize].to_vec())
    }

    /// Allocation-free [`GameState::payoffs`]: fills `out[..num_players]`.
    pub fn payoffs_into(&self, out: &mut [Chips; MAX_PLAYERS]) -> Result<(), GameError> {
        if !self.terminal {
            return Err(GameError::NotTerminal);
        }
        let n = self.n as usize;
        let mut won = [0 as Chips; MAX_PLAYERS];
        if self.num_active() == 1 {
            let w = (0..n).find(|&p| !self.folded[p]).unwrap();
            won[w] = self.pot();
        } else {
            let mut ranks = [0 as HandRank; MAX_PLAYERS];
            let board = self.board();
            if board.len() != 5 || board.contains(&NO_CARD) {
                return Err(GameError::HiddenCards("board is not complete".into()));
            }
            for p in 0..n {
                if !self.folded[p] {
                    let h = self
                        .hole_cards(p)
                        .ok_or_else(|| GameError::HiddenCards(format!("hole cards of seat {p}")))?;
                    ranks[p] = evaluate_hole_board(h, board);
                }
            }
            self.award_pots(&ranks, &mut won);
        }
        for p in 0..MAX_PLAYERS {
            out[p] = if p < n { won[p] - self.contributed[p] } else { 0 };
        }
        Ok(())
    }

    /// Split the pot into main and side pots at the contribution levels of
    /// the players still in the hand, and award each to the best eligible
    /// hand(s). Odd chips go one at a time to tied winners in clockwise
    /// order starting with the seat after the button.
    fn award_pots(&self, ranks: &[HandRank; MAX_PLAYERS], won: &mut [Chips; MAX_PLAYERS]) {
        let n = self.n as usize;
        let c = &self.contributed;
        let mut levels = [0 as Chips; MAX_PLAYERS];
        let mut nl = 0;
        for p in 0..n {
            if !self.folded[p] && !levels[..nl].contains(&c[p]) {
                levels[nl] = c[p];
                nl += 1;
            }
        }
        levels[..nl].sort_unstable();
        let mut prev = 0;
        for k in 0..nl {
            let lvl = levels[k];
            let mut amount: Chips = (0..n).map(|p| c[p].min(lvl) - c[p].min(prev)).sum();
            if k == nl - 1 {
                // Chips of folded players above the top live level.
                amount += (0..n).map(|p| (c[p] - lvl).max(0)).sum::<Chips>();
            }
            prev = lvl;
            if amount == 0 {
                continue;
            }
            let mut best = 0;
            let mut nwin: Chips = 0;
            for p in 0..n {
                if !self.folded[p] && c[p] >= lvl {
                    if ranks[p] > best {
                        best = ranks[p];
                        nwin = 1;
                    } else if ranks[p] == best {
                        nwin += 1;
                    }
                }
            }
            let share = amount / nwin;
            let mut rem = amount % nwin;
            for off in 1..=n {
                let p = (self.button as usize + off) % n;
                if !self.folded[p] && c[p] >= lvl && ranks[p] == best {
                    won[p] += share;
                    if rem > 0 {
                        won[p] += 1;
                        rem -= 1;
                    }
                }
            }
        }
    }

    /// Order in which hands are shown at showdown: the last player to bet or
    /// raise on the final betting round shows first, otherwise the first
    /// live seat clockwise from the button; then clockwise. Empty unless the
    /// hand reached a showdown.
    pub fn showdown_order(&self) -> Vec<usize> {
        if !self.terminal || self.num_active() < 2 {
            return Vec::new();
        }
        let n = self.n as usize;
        let mut start = (self.button as usize + 1) % n;
        if let Some(last) = self.history.last() {
            if let Some(h) = self
                .history
                .iter()
                .rev()
                .take_while(|h| h.street == last.street)
                .find(|h| h.action.kind == ActionKind::Raise)
            {
                start = h.player as usize;
            }
        }
        (0..n).map(|k| (start + k) % n).filter(|&p| !self.folded[p]).collect()
    }

    // ------------------------------------------------------------------
    // Keys
    // ------------------------------------------------------------------

    /// Append the public key (button, board and betting history; no hole
    /// cards) to `out`. Format:
    /// `[button, board_len, board cards..., then per action:
    ///   (street << 4 | kind), and for raises the raise-to as u32 LE]`.
    pub fn write_public_key(&self, out: &mut Vec<u8>) {
        out.push(self.button);
        out.push(self.board_len);
        out.extend_from_slice(self.board());
        for h in &self.history {
            out.push((h.street << 4) | h.action.kind as u8);
            if h.action.kind == ActionKind::Raise {
                out.extend_from_slice(&(h.action.amount as u32).to_le_bytes());
            }
        }
    }

    /// Public key as a new vector.
    pub fn public_key(&self) -> Vec<u8> {
        let mut v = Vec::with_capacity(8 + 3 * self.history.len());
        self.write_public_key(&mut v);
        v
    }

    /// Append the information-set key of `player` to `out`: the public key,
    /// then the seat index, then the player's two hole cards in ascending
    /// order.
    pub fn write_infoset_key(&self, player: usize, out: &mut Vec<u8>) -> Result<(), GameError> {
        if player >= self.n as usize {
            return Err(GameError::InvalidPlayer(player));
        }
        let h = self
            .hole_cards(player)
            .ok_or_else(|| GameError::HiddenCards(format!("hole cards of seat {player}")))?;
        self.write_public_key(out);
        out.push(player as u8);
        out.push(h[0].min(h[1]));
        out.push(h[0].max(h[1]));
        Ok(())
    }

    /// Information-set key of `player` as a new vector.
    pub fn infoset_key(&self, player: usize) -> Result<Vec<u8>, GameError> {
        let mut v = Vec::with_capacity(11 + 3 * self.history.len());
        self.write_infoset_key(player, &mut v)?;
        Ok(v)
    }

    /// A copy as seen by `viewer`: other players' hole cards and undealt
    /// board cards are removed. Actions that would deal a hidden card fail.
    pub fn masked(&self, viewer: usize) -> GameState {
        let mut s = self.clone();
        let n = self.n as usize;
        for p in 0..n {
            if p != viewer {
                s.cards[2 * p] = NO_CARD;
                s.cards[2 * p + 1] = NO_CARD;
            }
        }
        for i in 2 * n + self.board_len as usize..MAX_DEAL {
            s.cards[i] = NO_CARD;
        }
        s.masked = true;
        s
    }
}

impl fmt::Display for GameState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let cs = |cards: &[Card]| -> String {
            cards.iter().map(|&c| card_to_str(c).unwrap_or_else(|_| "??".into())).collect::<Vec<_>>().join("")
        };
        write!(f, "street={} board=[{}] pot={} ", self.street, cs(self.board()), self.pot())?;
        for p in 0..self.n as usize {
            let h = self.hole_cards(p).map(|h| cs(&h)).unwrap_or_else(|| "????".into());
            write!(
                f,
                "| p{p}{} {h} stack={} bet={}{}{} ",
                if p == self.button as usize { "(btn)" } else { "" },
                self.stacks[p],
                self.street_bets[p],
                if self.folded[p] { " folded" } else { "" },
                if self.all_in[p] { " all-in" } else { "" },
            )?;
        }
        let hist: Vec<String> = self.history.iter().map(|h| format!("{}:{}", h.player, h.action)).collect();
        write!(f, "| to_act={} history=[{}]", self.current, hist.join(" "))
    }
}
