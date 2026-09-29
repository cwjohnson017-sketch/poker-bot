//! Card encoding: `card = rank * 4 + suit`.
//!
//! * rank: `0 = 2, 1 = 3, ..., 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A`
//! * suit: `0 = c, 1 = d, 2 = h, 3 = s`
//!
//! String form is a rank char in `23456789TJQKA` followed by a suit char in
//! `cdhs`, e.g. `"As"`, `"Td"`.

use std::fmt;

/// A card in `0..52`.
pub type Card = u8;

/// Number of cards in a deck.
pub const NUM_CARDS: usize = 52;

/// Sentinel for "no card / hidden card" in internal buffers.
pub const NO_CARD: Card = 255;

/// Rank characters, indexed by rank.
pub const RANK_CHARS: &[u8; 13] = b"23456789TJQKA";
/// Suit characters, indexed by suit.
pub const SUIT_CHARS: &[u8; 4] = b"cdhs";

/// Error produced when parsing or validating cards.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CardError(pub String);

impl fmt::Display for CardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CardError {}

/// Rank of a card (`0..13`, `12` = ace).
#[inline(always)]
pub const fn rank(card: Card) -> u8 {
    card >> 2
}

/// Suit of a card (`0..4`).
#[inline(always)]
pub const fn suit(card: Card) -> u8 {
    card & 3
}

/// Build a card from rank and suit.
#[inline(always)]
pub const fn make_card(rank: u8, suit: u8) -> Card {
    rank * 4 + suit
}

/// Parse a two-character card string such as `"As"` or `"Td"`.
///
/// The rank character is case-insensitive (`"ts"` == `"Ts"`); the suit
/// character is also accepted in upper case.
pub fn card_from_str(s: &str) -> Result<Card, CardError> {
    let b = s.trim().as_bytes();
    if b.len() != 2 {
        return Err(CardError(format!("invalid card string {s:?}: expected 2 characters")));
    }
    let r = b[0].to_ascii_uppercase();
    let su = b[1].to_ascii_lowercase();
    let rank = RANK_CHARS
        .iter()
        .position(|&c| c == r)
        .ok_or_else(|| CardError(format!("invalid rank in card string {s:?}")))?;
    let suit = SUIT_CHARS
        .iter()
        .position(|&c| c == su)
        .ok_or_else(|| CardError(format!("invalid suit in card string {s:?}")))?;
    Ok(make_card(rank as u8, suit as u8))
}

/// Format a card as a two-character string (`"As"`).
pub fn card_to_str(card: Card) -> Result<String, CardError> {
    if card as usize >= NUM_CARDS {
        return Err(CardError(format!("invalid card {card}: must be in 0..52")));
    }
    let mut s = String::with_capacity(2);
    s.push(RANK_CHARS[rank(card) as usize] as char);
    s.push(SUIT_CHARS[suit(card) as usize] as char);
    Ok(s)
}

/// Parse a whitespace-optional sequence of cards: `"AsKd"`, `"As Kd 7c"`.
pub fn cards_from_str(s: &str) -> Result<Vec<Card>, CardError> {
    let compact: Vec<u8> = s.bytes().filter(|b| !b.is_ascii_whitespace() && *b != b',').collect();
    if !compact.len().is_multiple_of(2) {
        return Err(CardError(format!("invalid card list {s:?}")));
    }
    compact
        .chunks(2)
        .map(|c| card_from_str(std::str::from_utf8(c).unwrap_or("??")))
        .collect()
}

/// Check that every card is in `0..52` and no card repeats.
pub fn validate_cards(cards: &[Card]) -> Result<(), CardError> {
    let mut seen: u64 = 0;
    for &c in cards {
        if c as usize >= NUM_CARDS {
            return Err(CardError(format!("invalid card {c}: must be in 0..52")));
        }
        let bit = 1u64 << c;
        if seen & bit != 0 {
            return Err(CardError(format!("duplicate card {}", card_to_str(c).unwrap())));
        }
        seen |= bit;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_all_cards() {
        for c in 0..52u8 {
            let s = card_to_str(c).unwrap();
            assert_eq!(card_from_str(&s).unwrap(), c);
        }
        assert_eq!(card_from_str("2c").unwrap(), 0);
        assert_eq!(card_from_str("As").unwrap(), 51);
        assert_eq!(card_from_str("Td").unwrap(), 8 * 4 + 1);
        assert_eq!(card_from_str("th").unwrap(), 8 * 4 + 2);
        assert!(card_from_str("1c").is_err());
        assert!(card_from_str("Ax").is_err());
        assert!(card_from_str("A").is_err());
        assert!(card_to_str(52).is_err());
        assert_eq!(cards_from_str("As Kd,2c").unwrap(), vec![51, 45, 0]);
        assert!(validate_cards(&[1, 2, 1]).is_err());
        assert!(validate_cards(&[1, 2, 52]).is_err());
    }
}
