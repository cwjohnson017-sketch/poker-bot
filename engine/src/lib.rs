//! No-Limit Texas Hold'em engine: exact betting rules for 2..=9 players, a
//! fast 5/6/7-card hand evaluator, and (with the `python` feature) PyO3
//! bindings exposed as the `poker_engine` Python module.
//!
//! The Rust API mirrors the interface contract in `docs/INTERFACES.md`:
//!
//! * [`cards`]: card encoding (`rank * 4 + suit`) and string helpers.
//! * [`eval`]: `evaluate5/6/7`, batch evaluation and `hand_category`.
//! * [`game`]: [`GameConfig`], [`Action`], [`LegalActions`], [`GameState`].
//! * [`sim`]: deterministic RNG and random-play helpers.

#![allow(clippy::needless_range_loop)]

pub mod cards;
pub mod eval;
pub mod game;
pub mod sim;

#[cfg(feature = "python")]
mod python;

pub use cards::{card_from_str, card_to_str, cards_from_str, Card};
pub use eval::{evaluate, evaluate5, evaluate6, evaluate7, hand_category, HandRank};
pub use game::{
    Action, ActionKind, Chips, GameConfig, GameError, GameState, HistoryEntry, LegalActions, CHECK_CALL, FOLD,
    MAX_PLAYERS, RAISE,
};
