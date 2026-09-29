# poker_engine

Rust No-Limit Texas Hold'em engine with PyO3 bindings (Python module
`poker_engine`). It implements the contract in `docs/INTERFACES.md`:
exact betting rules for 2 to 9 players, side pots, and a fast 5/6/7-card
hand evaluator.

## Build

Requirements: Rust stable, and Python 3.9+ with `maturin` and `numpy`
for the bindings.

```bash
cd engine

# Rust library and tests (no Python needed)
cargo test --release          # full suite, including all 133,784,560 7-card hands
cargo test                    # debug build; the exhaustive 7-card test is skipped

# Throughput numbers (single thread)
cargo run --release --example throughput

# Python extension, installed into the active virtualenv
maturin develop --release
python -m pytest tests/test_python_bindings.py
```

The PyO3 code is behind the `python` cargo feature. `maturin` turns it on
(see `pyproject.toml`), so `cargo test` never links against libpython. To
type-check the bindings without maturin:
`PYO3_PYTHON=$(which python) cargo check --features python`.

## Python API

```python
import numpy as np
import poker_engine as pe
from poker_engine import Action, GameConfig, GameState, FOLD, CHECK_CALL, RAISE

cfg = GameConfig()                  # 2 players, 20,000 chips each, 50/100, no ante
cfg = GameConfig(num_players=6, stacks=[10_000] * 6, small_blind=50, big_blind=100, ante=0)

deck = np.random.default_rng(0).permutation(52).tolist()
s = GameState.new_hand(cfg, button=0, deck=deck)   # blinds posted, cards dealt
la = s.legal_actions()   # can_fold, can_check, call_amount, min_raise_to, max_raise_to
s.apply(Action.raise_to(la.min_raise_to))          # mutates; ValueError if illegal
t = s.child(Action.check_call())                   # copy + apply
while not t.is_terminal:
    t.apply(Action.check_call())
t.payoffs()          # net chip change per seat, sums to 0
t.public_key()       # bytes: button, board, betting history
t.infoset_key(0)     # bytes: public key + seat + seat 0's hole cards

pe.evaluate7([pe.card_from_str(c) for c in ["As", "Ks", "Qs", "Js", "Ts", "2c", "3d"]])
pe.evaluate_batch(np.zeros((0, 7), np.uint8))      # uint8 [N, 7] -> int32 [N]
pe.hand_category(rank)                             # 0 high card ... 8 straight flush
```

Contract items: `GameConfig`, `Action` (`fold()`, `check_call()`,
`raise_to(amount)`, fields `kind` and `amount`), `LegalActions`,
`GameState.new_hand`, `legal_actions`, `apply`, `child`, `clone`,
`payoffs`, `public_key`, `infoset_key`, `hole_cards(player)` and the
properties `num_players`, `button`, `street`, `board`, `stacks`,
`street_bets`, `pot`, `current_player` (-1 when terminal), `is_terminal`,
`folded`, `all_in`, `history` (list of `(street, player, Action)`).
Card helpers: `card_from_str`, `card_to_str`. Evaluator: `evaluate5`,
`evaluate6`, `evaluate7`, `evaluate_batch`, `hand_category`.

Extras beyond the contract:

| Name | Purpose |
|---|---|
| `evaluate(cards)` | 5, 6 or 7 cards |
| `GameState.contributed` | chips committed in the whole hand, per seat (antes included) |
| `GameState.current_bet`, `last_raise_size`, `num_raises_this_street` | betting-round state (for abstractions that cap raises per street) |
| `GameState.config` | the `GameConfig` the hand was started with |
| `GameState.masked(viewer)` | copy with other seats' hole cards and undealt cards removed (`hole_cards(other) == []`); for the match runner. Applying an action that would deal a hidden card raises `ValueError`. |
| `GameState.showdown_order()` | order in which live hands are shown |
| `LegalActions.can_raise`, `is_legal(action)`, `clamp_raise_to(amount)` | helpers for action mapping |
| `HAND_CATEGORY_NAMES`, `MAX_PLAYERS` | constants |

`Action`, `GameConfig` pickle; `GameState` supports `copy.copy`,
`copy.deepcopy` and `==`. Type stubs are in `poker_engine.pyi`.

## Rust API

The crate is also a normal library (`rlib`) for Rust code such as the
tabular MCCFR solver:

- `game::GameState`: `new_hand(&GameConfig, button, &[Card])`,
  `legal_actions() -> LegalActions` (`Copy`), `apply(Action) -> Result`,
  `child`, `payoffs() -> Vec<Chips>` / allocation-free `payoffs_into`,
  `public_key()`, `infoset_key(p)`, and `write_public_key` /
  `write_infoset_key` that append into a reused buffer. The state is
  fixed-size arrays plus one `Vec` of history entries, so `clone` is cheap.
- `eval::{evaluate, evaluate5, evaluate6, evaluate7, evaluate_masks,
  evaluate_hole_board, evaluate_batch, hand_category}` and `eval::naive`
  (the slow reference used in tests).
- `sim::Rng` (xoshiro256**), `shuffled_deck`, `random_action`,
  `play_random_hand`.

## Rules and conventions

Where the contract leaves a choice open, the engine uses the standard
(TDA-style) rule. The reference engine in `pokerbot/reference/` must make
the same choices for the two to agree.

**Dealing.** Hole cards go to seat 0, 1, ..., then flop, turn, river, taken
from `deck` in order. `deck` must hold at least `2 * num_players + 5`
distinct cards in `0..52`; normally it is a full permutation.

**Blinds and order.** Antes are posted first, go to the pot as dead money,
and do not count toward `street_bets`. Heads-up: the button posts the
small blind and acts first preflop; the other seat acts first after the
flop. Three or more players: the small blind is the seat after the button,
the big blind the next seat, and the seat after the big blind acts first
preflop (three-handed that is the button). After the flop, the first live
seat clockwise from the button acts first. A player who cannot cover a
blind or ante posts what they have and is all-in.

**Short big blind.** The amount to call preflop is always the full big
blind, even if the big blind poster was all-in for less. Any excess is
returned through the side-pot logic.

**Bet sizes.** `min_raise_to = current_bet + last_raise`, where `last_raise`
starts each street at the big blind and becomes the size of each full raise
(`raise_to - current_bet`). The minimum bet after the flop is therefore the
big blind. `max_raise_to` is the player's all-in total
(`street_bet + stack`), even when it exceeds what opponents can call; the
uncalled part comes back to the bettor in `payoffs`. A raise is legal only
if the player's all-in total is above the current bet, at least one other
player can still act (not folded, not all-in), and action is open to the
player (below). When the all-in total is below `min_raise_to`, only
`raise_to(max_raise_to)` is legal as a raise, exactly as the contract says.

**Re-opening of action.** An all-in raise smaller than a full raise does not
change `last_raise` and does not re-open action for players who have
already acted. A player who has acted on this street may raise again only
if the total increase since their last action is at least one full raise
(`current_bet - bet level after their last action >= last_raise`). This is
the TDA rule: several short all-ins that together add up to a full raise do
re-open action. A player who has not yet acted on the street (the big blind
preflop included) may always raise.

**Folding and checking.** `FOLD` is legal only when facing a bet
(`can_fold == False` whenever checking is possible). `CHECK_CALL` is always
legal in a non-terminal state; a call for more than the stack puts the
player all-in. Folds and check/calls must carry `amount == 0`.

**Closing rounds and run-outs.** A betting round ends when every player who
can act has acted since the last raise and matched the current bet. A
player who is the only one left able to act does not act unless facing a
bet. When the round closes with fewer than two players able to act, the
engine deals the rest of the board in the same `apply` call and ends the
hand. After a run-out, `street == 3`, the board has 5 cards and
`street_bets` are all zero. A hand can also end inside `new_hand` when the
blinds put everyone (or all but one player not facing a bet) all-in.

**Terminal state.** `stacks` stay as the chips behind; winnings are not
added back. Use `payoffs()` (net chip change per seat) and
`stacks[p] + contributed[p] + payoffs[p]` for the final stack. `payoffs()`
raises `ValueError` before the hand is over.

**Side pots.** Pots are cut at the distinct total contributions of the
players still in the hand. Each pot goes to the best eligible hand; chips
that folded players put in above the largest live contribution join the
top pot. A pot with one eligible player (an uncalled bet) goes back to that
player.

**Split pots and odd chips.** Each pot is split separately among its tied
winners. The remainder is paid one chip at a time to the tied winners in
clockwise order starting with the seat after the button (heads-up: the big
blind first).

**Showdown order** (`showdown_order()`, informational). If the last betting
round with any action had a bet or raise, the last aggressor shows first;
otherwise the first live seat after the button. Then clockwise.

**Keys.** `public_key()` is
`[button, board_len, board cards..., then per action (street << 4 | kind),
followed by the raise-to as 4 little-endian bytes for raises]`. Blinds and
antes are implied by the config and are not in the history.
`infoset_key(p)` is `public_key() + [p, low hole card, high hole card]`:
hole cards are sorted so the key does not depend on deal order, and the
seat index keeps the two players' information sets apart. Keys do not
encode the config (stacks, blinds); a solver keys tables per config.

## Hand evaluator

Per-suit 13-bit rank masks: flush check, then quads, full house, straight,
trips, two pair, pair and high card with bit tricks and two small tables
built at compile time (no large lookup table). One code path handles 5, 6
and 7 cards.

`rank = category << 20 | tiebreak`, with up to five card ranks in 4-bit
nibbles (see `src/eval.rs`). Higher is better; compare with integer
comparison only.

Tests: 1.2M random 7-card hands, and 200k random 5- and 6-card hands,
agree exactly with a naive enumerating evaluator (fewer in debug builds).
All 2,598,960 five-card hands give the known category counts (1,302,540
high card, 1,098,240 pairs, ..., 40 straight flushes of which 4 are royal)
and 7,462 distinct values. All 133,784,560 seven-card hands give the known
counts and 4,824 distinct values.

## Throughput

Single thread, release build, `cargo run --release --example throughput`
on the 4-core development container:

| Benchmark | Rate |
|---|---|
| `evaluate7` | ~46M evaluations/s |
| random hands, heads-up (deal, random legal actions, payoffs) | ~3.0M hands/s |
| random hands, 6-max | ~1.3M hands/s |
| heads-up hands with `child()` + `public_key()` at every node | ~1.1M hands/s |
| `evaluate_batch` called from Python | ~35M evaluations/s |

## Layout

```
engine/
  Cargo.toml, pyproject.toml, poker_engine.pyi
  src/lib.rs        crate root and re-exports
  src/cards.rs      card encoding and string helpers
  src/eval.rs       evaluator + naive reference
  src/game.rs       rules: config, actions, state, pots, keys
  src/sim.rs        RNG and random play
  src/python.rs     PyO3 bindings (feature "python")
  tests/evaluator.rs            evaluator vs naive, category distributions
  tests/scenarios.rs            hand-written betting scenarios + random invariants
  tests/test_python_bindings.py pytest suite for the bindings
  examples/throughput.rs        speed measurements
```
