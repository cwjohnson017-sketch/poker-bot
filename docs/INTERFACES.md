# Interface contract

Every component codes against this contract. Two engines implement it: the
Rust crate `engine/` (Python module `poker_engine`, fast) and the pure-Python
reference `pokerbot/reference/` (slow, used for tests and cross-checks). They
must produce identical results on identical inputs.

## Package layout

```
engine/                 Rust crate + PyO3 bindings -> python module `poker_engine`
pokerbot/               Python package
  reference/            pure-Python reference rules + evaluator (same API)
  env/                  torch vectorized env, GPU equity, observation encoders
  abstraction/          action sets, suit isomorphism, buckets
  blueprint/mccfr/      tabular MCCFR driver + exporter (solver core in Rust)
  blueprint/deepcfr/    traversal, memories, networks, training loop
  search/               subgame builder, range CFR+, leaf rollouts, gadget
  eval/                 match runner, duplicate matches, LBR, approximate BR
  agents/               agent interface and implementations
  protocol/             ACPC-style socket protocol and human CLI
configs/                YAML run configs
scripts/                entry points
tests/                  pytest suites (CPU only in CI)
```

## Cards

- A card is an `int` in `0..52`: `card = rank * 4 + suit`.
- `rank`: `0 = 2, 1 = 3, ..., 8 = T, 9 = J, 10 = Q, 11 = K, 12 = A`.
- `suit`: `0 = c, 1 = d, 2 = h, 3 = s`.
- String form: rank char in `23456789TJQKA` followed by suit char in `cdhs`
  (`"As"`, `"Td"`). Helpers: `card_from_str(s) -> int`, `card_to_str(c) -> str`.
- A deck for a hand is a permutation of `0..52`; cards are dealt in order:
  hole cards for player 0 (two cards), player 1, ..., then flop (three), turn,
  river. Player 0 is the seat index, not the button. Given the same deck and
  the same button, both engines deal identically.

## Hand evaluation

- `evaluate7(cards: Sequence[int]) -> int` for exactly 7 cards; higher is a
  better hand. `evaluate5` and `evaluate6` exist as well.
- `evaluate_batch(cards: np.ndarray[uint8, (N, 7)]) -> np.ndarray[int32, (N,)]`.
- Ranks of two hands compare by integer comparison only. The absolute values
  are engine-specific; tests compare orderings and hand categories.
- `hand_category(rank: int) -> int` maps a rank to
  `0 high card, 1 pair, 2 two pair, 3 trips, 4 straight, 5 flush,
   6 full house, 7 quads, 8 straight flush`.

## Game configuration and state

```python
class GameConfig:
    num_players: int = 2
    stacks: list[int]              # starting stacks in chips, per seat
    small_blind: int = 50
    big_blind: int = 100
    ante: int = 0

FOLD, CHECK_CALL, RAISE = 0, 1, 2

class Action:
    kind: int                      # FOLD | CHECK_CALL | RAISE
    amount: int                    # RAISE only: the total this player will have
                                   # committed on the current street after the
                                   # action ("raise to"). 0 for other kinds.
    # constructors: Action.fold(), Action.check_call(), Action.raise_to(amount)

class LegalActions:
    can_fold: bool                 # False when checking is possible
    can_check: bool
    call_amount: int               # chips added to call (0 when can_check)
    min_raise_to: int              # 0 when no raise is legal
    max_raise_to: int              # all-in raise-to; 0 when no raise is legal
    # A raise is legal iff min_raise_to > 0; any raise_to in
    # [min_raise_to, max_raise_to] is legal, plus max_raise_to itself even
    # when it is below min_raise_to (all-in for less).

class GameState:
    @staticmethod
    def new_hand(config: GameConfig, button: int, deck: Sequence[int]) -> GameState
    # Blinds are posted immediately. Heads-up: button posts the small blind
    # and acts first preflop; the other seat acts first on later streets.
    # Multiway: small blind is the seat after the button.

    num_players: int
    button: int
    street: int                    # 0 preflop, 1 flop, 2 turn, 3 river
    board: list[int]               # 0, 3, 4, or 5 cards
    def hole_cards(self, player: int) -> list[int]
    stacks: list[int]              # chips behind, per seat
    street_bets: list[int]         # chips committed this street, per seat
    pot: int                       # all chips committed by everyone so far,
                                   # including the current street
    current_player: int            # -1 when terminal
    is_terminal: bool
    folded: list[bool]
    all_in: list[bool]
    history: list[tuple[int, int, Action]]   # (street, player, action)
    def legal_actions(self) -> LegalActions
    def apply(self, action: Action) -> None  # mutates; raises ValueError if illegal
    def child(self, action: Action) -> GameState
    def clone(self) -> GameState
    def payoffs(self) -> list[int]  # net chip change per seat; terminal only
    def public_key(self) -> bytes   # board + betting history, no hole cards
    def infoset_key(self, player: int) -> bytes  # public_key + player's hole cards
```

Street transitions happen inside `apply`: when the betting round closes the
engine deals the next street from the deck and sets `current_player`. If all
but one player is all-in (or everyone is all-in) the engine runs out the
remaining board and marks the hand terminal in the same `apply` call.
Hands end at showdown or when all but one player has folded. Side pots are
awarded to the best eligible hand; odd chips go to the eligible player
nearest the button clockwise, starting with the seat after the button.

## Agents

```python
class Agent(Protocol):
    name: str
    def new_hand(self, seat: int, config: GameConfig) -> None: ...
    def act(self, state: GameState, seat: int, rng: np.random.Generator) -> Action: ...
    def observe_end(self, state: GameState) -> None: ...
```

Agents receive the full `GameState` but must only read their own hole cards.
The match runner enforces this in tests by handing agents a state whose
other players' hole cards are masked (`hole_cards(other)` returns `[]`).

## Abstract actions

The abstraction layer defines per-street abstract action sets as lists of
`("fold",) | ("check_call",) | ("raise", fraction_of_pot: float) | ("allin",)`.
Mapping from an abstract action and a `GameState` to a concrete `Action` and
from a concrete opponent `Action` back to the nearest abstract action
(pseudo-harmonic mapping) lives in `pokerbot/abstraction/actions.py`.
The abstract index of an action in a state is what networks and tables use.

## Torch vectorized environment

`pokerbot.env.VecNLHE(n: int, config: GameConfig, device, seed)` holds `n`
heads-up hands as tensors and exposes:

- `reset(mask: BoolTensor | None)` deals new hands in the masked slots.
- `obs()` returns a dict of tensors describing the acting player's view.
- `legal_mask()` returns `[n, A]` over the abstract action set.
- `step(abstract_action: LongTensor[n])` applies one abstract action per
  slot for the slot's current actor, returning `(payoffs[n, 2], done[n])`.
- `replay_check(deck, actions)` is a test hook that runs one hand through the
  same tensor code path so it can be compared to the scalar engine.

It must run on CPU (tests) and CUDA (training). Chip accounting and dealing
order match the scalar engine bit for bit.
