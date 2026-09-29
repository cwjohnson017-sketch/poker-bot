# `pokerbot.env`: vectorized heads-up NLHE in torch

Pure-torch, device-agnostic building blocks for GPU self-play:

| Module | Contents |
|---|---|
| `cards.py` | card encoding (`card = rank*4 + suit`), string helpers, `shuffled_decks` (argsort of random keys per row) |
| `evaluator.py` | batched 5/6/7-card evaluator `evaluate_batch`, `evaluate7_batch`, `hand_category`, scalar `evaluate5/6/7` |
| `actions.py` | abstract action spec (`ActionSpec`, `DEFAULT_SPEC`), raise-size mapping, legal mask |
| `vec_env.py` | `VecNLHE(n, config, device, seed)` |
| `obs.py` | `encode_obs`: network inputs for the acting player |
| `equity.py` | `equity_vs_random`, `equity_river` (exact, 990 opponents), `equity_histogram` |
| `config.py` | minimal `GameConfig` (the env duck-types any config with `stacks`, `small_blind`, `big_blind`, `ante`) |

Everything runs under `torch.no_grad`, with no per-hand Python loops.

## Quick start

```python
import torch
from pokerbot.env import VecNLHE, GameConfig

env = VecNLHE(131072, GameConfig(), device="cuda", seed=0, validate=False)
while True:
    obs = env.obs()                       # dict of [n, ...] tensors
    logits = net(obs)                     # your network
    logits = logits.masked_fill(~obs["legal"], -1e9)
    a = torch.distributions.Categorical(logits=logits).sample()
    payoffs, done = env.step(a)           # [n, 2] chips, [n] bool
    ...                                   # consume payoffs[done]
    env.reset(done)                       # re-deal finished slots
```

## Running on CUDA

Pass `device="cuda"`; every tensor, lookup table and the random generator
live on that device. There are no `.cuda()` calls and no CPU-only ops in
`step`/`legal_mask`/`obs`. Things to know:

* `step()` with `validate=True` (the default) does one host sync to raise on
  illegal actions. Construct with `validate=False` for training; illegal
  actions are then replaced by check/call.
* `reset(mask)` does one host sync (`nonzero`) to deal only the finished
  slots. Call it once per step, not per slot.
* `step`, `legal_mask` and `obs` trace into a single graph with no graph
  breaks, so `torch.compile(env.step)` works (checked on CPU: compiled and
  eager states match exactly, about 2.5x faster). For CUDA graphs, keep `n`
  fixed.
* Equity functions take an optional `generator`; it must be on the same
  device as the inputs.

## Rules and chip accounting

The Rust engine (`poker_engine`, rule choices in `engine/README.md`) is the
source of truth. `tests/test_cross_env_engine.py` plays a few thousand random
hands on it (deep, unequal, short and random stacks, antes, blinds that put
players all-in, `sb == bb`, `sb == 0`) and checks that `replay_check` and a
batched `step_concrete` give the same legal info before every action
(current player, street, pot, can_fold, can_check, call_amount,
min_raise_to, max_raise_to), payoffs, board and terminal street. The scalar
oracle `tests/env/scalar_rules.py` follows the same rules.

* The deck is dealt as P0 hole, P1 hole, flop, turn, river (seat order).
  `button` picks the seat that posts the small blind.
* The button posts the small blind and acts first preflop. The other seat
  acts first after the flop. Antes go into the pot but not the street bets.
  A player who cannot cover the ante or blind posts what they have and is
  all-in.
* The current bet (`LegalInfo.max_bet`) is the largest street bet, and
  preflop never less than the full big blind. Facing a big blind that is
  all-in for less, the small blind still calls up to the full big blind
  (and may fold); the excess comes back at showdown.
* `min_raise_to = current bet + max(last full raise increment this street,
  BB)`. The increment starts at BB on every street. `max_raise_to` is the
  actor's all-in total (`street bet + stack`), even when it exceeds what the
  opponent can call. An all-in below `min_raise_to` is legal and does not
  change the increment. A raise is legal only if the actor's all-in total is
  above the current bet and the opponent still has chips behind. Heads-up,
  the TDA re-opening rule never matters: an incomplete raise is always
  all-in, so nobody can raise after it.
* A seat is done for the round when it has folded, is all-in, or has matched
  the current bet and either acted since the last raise or is the only
  player left who can bet: a lone player acts only when facing a bet. The
  round closes when both seats are done. If at most one player can still
  bet, the board is run out and the hand is shown down in the same `step`.
* The same rule is applied right after the blinds. The hand ends at the deal
  when both players are all-in, or when the small blind is all-in and the
  big blind faces no bet. When only the big blind is all-in, the small
  blind still acts (call or fold, no raise).
* `street` is 3 after any showdown (run-outs and hands decided at the deal
  included) and stays at the fold street after a fold.
* Payoffs are net chip changes. On a fold, the folder loses its whole
  contribution. At showdown, the better hand wins `min(contribution)`: the
  uncalled excess goes back. A tie pays 0/0, since heads-up pots split
  evenly (no odd chips).
* Terminal `stacks` are the final stacks, `start + payoff`. The Rust engine
  instead keeps the chips behind at the end (final stack there is
  `stacks + contributed + payoffs`). Use payoffs to compare.

## Abstract actions (`actions.py`)

An `ActionSpec` lists the abstract actions for each street. `DEFAULT_SPEC`
follows DESIGN.md §5.3:

| idx | Preflop | Flop | Turn | River |
|---|---|---|---|---|
| 0 | fold | fold | fold | fold |
| 1 | check/call | check/call | check/call | check/call |
| 2 | `raise_x 2.5` | `raise 0.33` | `raise 0.5` | `raise 0.5` |
| 3 | `raise_x 3.0` | `raise 0.75` | `raise 1.0` | `raise 1.0` |
| 4 | `raise 1.0` (pot) | `raise 1.5` | allin | `raise 2.0` |
| 5 | allin | allin | (padding) | allin |

* `("raise", f)` raises to `max_bet + f * (pot + to_call)`, where `pot`
  includes the current street's bets. `f = 1` is a pot-sized raise, and
  when there is no bet to call it is a bet of `f * pot`.
* `("raise_x", m)` raises to `m * max_bet`: 2.5x the big blind as an open,
  or 3x the open as a 3-bet.
* Sizes are held in thousandths. Amounts use exact integers with
  round-half-up, `(f_milli * X + 500) // 1000`, so other engines can
  reproduce them exactly. The result is clamped to
  `[min_raise_to, max_raise_to]`.
* Legal mask rules:
  * Fold is legal only when facing a bet. Check/call is always legal.
  * Raises need a legal raise and fewer than `max_raises = 4` voluntary
    raises on this street. Blinds do not count.
  * A sized raise that would be all-in is masked, because `allin` covers it.
  * With `dedupe=True`, a sized raise with the same amount as an earlier
    one is masked. Example: the preflop pot raise equals `3x` when opening.
  * Padding entries are never legal.
  * Finished slots allow only check/call, which is a no-op.
* `A = spec.num_actions` (6 by default) is the width of the action axis.

`step_concrete(kind, amount)` applies contract-style concrete actions
(`FOLD`, `CHECK_CALL`, `RAISE` + raise-to amount) through the same code path.
`replay_check(deck, actions, button)` uses it to replay one hand and returns
the payoffs, the visible board and the legal info before each action.

## State tensors (`VecNLHE`)

| Field | Shape / dtype | Meaning |
|---|---|---|
| `deck` | `[n, 52]` uint8 | the deal; `cards` = `deck[:, :9]` as long |
| `button` | `[n]` long | button seat (alternates per slot on each reset unless given) |
| `stacks` | `[n, 2]` long | chips behind; after the hand, `start + payoff` (Rust keeps chips behind) |
| `street_bets` | `[n, 2]` long | chips committed this street |
| `contrib` | `[n, 2]` long | chips committed this hand (incl. antes); `pot = contrib.sum(1)` |
| `street` | `[n]` long | 0..3 (3 after a showdown, including run-outs) |
| `actor` | `[n]` long | seat to act, -1 when finished |
| `folded`, `all_in`, `acted` | `[n, 2]` bool | per-seat flags (`acted`: since the last raise on this street) |
| `last_raise` | `[n]` long | last full raise increment this street (BB at the start of each street) |
| `n_raises` | `[n]` long | voluntary raises this street |
| `hist_tok`, `hist_amt`, `hist_len` | `[n, 24]`, `[n, 24]`, `[n]` long | action tokens (0 = pad), chips added per action, count |
| `done` | `[n]` bool | hand finished |
| `payoffs` | `[n, 2]` long | net chips per seat; 0 until done |
| `ranks` | `[n, 2]` long | showdown hand ranks, precomputed at the deal |
| `last_kind`, `last_amount` | `[n]` long | the concrete action applied by the last step |

A history token is `1 + (street * 2 + actor_is_button) * A + abstract_index`,
which gives `env.vocab_size = 1 + 8A` tokens. The default abstraction never
needs more than 23 tokens per hand.

`select(idx)` returns a new env with copies of the chosen slots. Indices may
repeat, which is how a traversal frontier fans out over actions. `clone()`
and `state_dict()` are also available.

## Observations (`obs.py`)

`env.obs()` returns tensors for the acting player:

* `cards [n, 7]`: own hole cards, then the board. Undealt slots hold 52.
  `card_mask [n, 7]` marks the visible cards.
* `hist [n, 24]`, `hist_mask`, and `hist_amt`. `hist_amt` is chips added
  per action, divided by the starting stack.
* `scalars [n, 14]`, as fractions of the actor's starting stack: pot, own
  and opponent stack, own and opponent street bet, to-call, min and max
  raise-to. These are followed by the street one-hot, is-button and
  raises / max_raises.
* `legal [n, A]` and `actor [n]`.
* Equity hooks, off by default:
  * `obs(equity_samples=k)` adds `equity [n]`, the Monte Carlo equity
    against a random hand.
  * `obs(hist_runouts=r)` adds `equity_hist [n, 10]`, a histogram of river
    equity over `r` sampled runouts. Each runout is exact against all 990
    opponent hands unless `hist_opp_samples > 0`.

## Evaluator

`evaluate_batch` accepts `[..., 5|6|7]` distinct cards and returns
`category << 26 | tiebreak`. Tiebreaks are pairs of 13-bit rank masks, and
the only lookup tables have 8192 entries, so nothing large goes to the
device. `hand_category(r) = r >> 26` gives 0 = high card through
8 = straight flush. It is tested against every 5-card hand (category
counts, 7462 classes) and against an enumerating reference on 250k random
and constructed 7-card hands.

## Tests

```
PYTHONPATH=. python -m pytest tests/env -q          # add -s for the throughput report
```
