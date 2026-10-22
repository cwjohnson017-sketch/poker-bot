# `pokerbot.eval`: evaluation harness

Everything that says whether a bot got better lives here (DESIGN.md 5.7).

| Tool | Module / entry point | Answers |
|---|---|---|
| Match runner | `match.py`, `scripts/play_match.py` | How does A do against B? (plain or duplicate, mbb/h ± CI) |
| Checkpoint ladder | `ladder.py`, `scripts/run_ladder.py` | Is each checkpoint better than the previous ones and the anchors? |
| Local best response | `lbr.py`, `scripts/run_lbr.py` | Does the policy have gross holes? (cheap exploitability lower bound) |
| Approximate best response | `abr.py`, `scripts/run_abr.py` | How much can a trained exploiter win? (stronger lower bound) |
| Soak test | `soak.py`, `scripts/soak.py` | Does the bot run cleanly for 100k hands? (crashes, illegal actions, fallbacks, p99 time, memory) |
| Preflop off-tree audit | `preflop_audit.py`, `scripts/audit_preflop.py` | How does the bot answer every preflop size, on its tree or not? |
| Run logger | `logging.py` | CSV + TensorBoard scalars for all of the above and for training loops |
| Agent registry | `pokerbot/agents/registry.py` | Build any agent from a string such as `blueprint:runs/x.pkl` |
| Archetype suite | `pokerbot/agents/archetypes.py`, `scripts/run_archetypes.py` | Does the bot beat scripted "average players"? (roadmap 1.3) |

## Units and statistics

* **mbb/h**: milli-big-blinds per hand, `1000 * chips / (big_blind * hands)`.
  With 50/100 blinds, +100 mbb/h is +10 chips a hand. For scale: folding
  every hand loses 750 mbb/h, and a calling station loses tens of thousands
  of mbb/h to a decent exploiter because 200bb stacks go in often.
* **CI**: percentile bootstrap over the independent sampling units (one
  hand, or one duplicate deal of two hands) at 95% (`stats.py`). "CI
  excludes zero" is the significance test used by the adoption rule.
* **Duplicate**: every deal is played twice with seats swapped, so card
  luck mostly cancels. This cuts the variance by a large factor between
  similar agents. The ladder always plays duplicate.
* With 200bb stacks the per-hand standard deviation is several big blinds
  even between similar agents, and far more for all-in-heavy baselines. The
  CI half-width shrinks as `1/sqrt(hands)`: going from 20k to 200k hands cuts it
  by about 3.2x. Read the half-width off a short run first, then size the
  real one.
* **Luck adjustment** (`luck.py`; `run_match` / `run_duplicate_match` with
  `luck_adjust=True`; `play_match.py --adjust`; `match.luck_adjust` in a
  config): heads-up results are also reported with card luck removed. A hand
  that ends in a called all-in before the river scores its equity, and each
  new street subtracts `2c * (E_new - E_old)`, the change in seat 0's
  check-down value (`c` chips in per player, `E` the equity against the other
  hand given the board so far). Both terms have zero mean, so the adjusted
  win rate estimates the same quantity with a narrower interval. These are
  the chance terms of AIVAT; the bot's action terms are in `aivat.py` (below). Equities
  are shared by the two seatings of a duplicate deal, so mirror matches
  still cancel exactly.

## AIVAT: action terms for a recording bot

`aivat.py` (roadmap item 1.2) adds AIVAT's action terms for the bot's own
decisions to the chance terms of `luck.py`. It is meant for results against a
person (`pokerbot/play/`), where the opponent's strategy is unknown and
duplicate dealing is impossible. At every searched decision of a
`SearchAgent(record=True)` the bot's result gets `sum_a sigma(a) v(a) -
v(a_taken)`: `sigma` is the strategy row the bot sampled its action from, and
`v(a)` is the solve's expected chips for the bot's hand after `a` against the
opponent's actual hand (`values_vs`, `pokerbot.search.agent.pair_values`).
Each chance and action term has zero mean given everything before it, so the
total is unbiased whatever the value functions are; the module docstring has
the argument. The opponent's actions get no term. Preflop decisions (the
solved table has no values) get none either.

* `aivat_adjusted(hands, decisions, bot_seats, config)`: `(chance, aivat)`,
  `[n, 2]` zero-sum results of in-memory hands plus the bot's decision records.
* `score_session(dir)` / `scripts/score_play_session.py runs/play/<session>`:
  a play session from its logs.
* `values="range"` values actions against the modelled range (`values[c, a]`)
  instead. A converged solve is indifferent between the actions it mixes, so
  those terms are about zero by construction.

**Validation** (`scripts/eval_aivat.py`, 2026-10-21, CPU): a recording
`SearchAgent` (uniform blueprint, the tiny config of `tests/search/test_agent.py`
at 32 DCFR iterations per solve) against three baselines, 3000 hands each, 100bb,
button alternating, no duplicate dealing. Per-hand SD from the bot's seat, in
bb/hand, and the bot's win rate in mbb/hand with 95% bootstrap CIs:

| Opponent | raw SD | chance SD | chance + range-valued actions | AIVAT SD | raw win rate | AIVAT win rate |
|---|---|---|---|---|---|---|
| `equity` | 51.3 | 28.3 | 28.5 | **27.3** | -3925 [-5698, -2070] | -4360 [-5330, -3358] |
| `random` | 51.0 | 31.1 | 31.4 | **30.8** | +7941 [+6123, +9826] | +7025 [+5947, +8142] |
| `always_call` | 74.1 | 45.7 | 45.8 | **44.0** | +3027 [+492, +5838] | +1475 [-41, +3020] |

* **Unbiased:** the AIVAT mean is inside the raw CI in all three. Paired
  `raw - aivat` has z = +0.56 / +1.23 / +1.47, and the action terms alone
  average to z = +0.20 / -0.40 / -1.91. At the clock-limited tiny config (2-4
  iterations, `--iterations 0`) the six z scores are within ±1.3.
* **Variance:** the chance terms do most of the work, removing 64-72% of the
  raw variance. The action terms remove a further 2-8% of what is left
  (variance ratio AIVAT / chance 0.93 / 0.98 / 0.93), so the same CI takes
  about 7% fewer hands against `equity` and `always_call`. Valued against
  the range instead, they add variance (ratio 1.01). With 2-4 iterations per
  solve the bot's values are too rough to help: AIVAT SD 32.5 / 35.1 / 50.2
  against chance-only 32.1 / 34.5 / 50.9.
* **Why the action terms do little here.** The opponents' decisions are where
  most of the remaining variance comes from, and they get no terms. The
  action values assume the opponent plays the solved strategy, and these
  baselines do not. The chance terms value states by check-down equity, the
  action terms by the solve, so the two do not telescope as a single value
  function would. The production bot (converged value-net solves on the GPU)
  against a person may do better or worse. Its numbers need a GPU and a
  person: score a real session with `scripts/score_play_session.py`.

Time: 32-iteration CPU solves take about 1.3-2 s per decision. The table above
took about 3.5 hours of 12 workers (sharing the CPU with a test run).

## Agent specs

`pokerbot.agents.registry.make_agent(spec, **defaults)` (also exported as
`pokerbot.agents.make_agent`, so `scripts/play_match.py` accepts specs too):

```
always_call | always_raise | random | equity | uniform | fixed:allin
equity:raise_threshold=0.8,samples=100          # key=value -> kwargs (YAML-typed)
blueprint:runs/mccfr/ckpt_0100.pkl               # positional arg (checkpoint path)
neural:runs/deepcfr/it40.pt,device=cuda
search:runs/mccfr/final.pkl,device=cuda       # kwargs are whatever the agent accepts
```

Names resolve to `@register("name")` entries first, then the baseline table
`pokerbot.agents.AGENTS`, then a lazy import table (`LAZY_AGENTS`):
`blueprint -> pokerbot.blueprint.mccfr.agent:BlueprintAgent`,
`neural -> pokerbot.blueprint.deepcfr.agent:NeuralBlueprintAgent`,
`search -> pokerbot.search.agent:SearchAgent`. Those modules are imported
only when used, and a missing module raises `AgentSpecError` with the
reason. With a positional argument, a factory's `from_checkpoint(path,
**kw)` classmethod is used when it has one, else `cls(path, **kw)`.
Config files pass per-name constructor defaults under `agents:`, and the
spec's own `key=value` items override them.

## Checkpoint ladder

```
python scripts/run_ladder.py --agents always_call always_raise random equity --hands 2000 --workers 4
python scripts/run_ladder.py --checkpoints "runs/mccfr/ckpt_*.pkl" --kind blueprint \
    --schedule newest -k 5 --anchor equity --anchor blueprint:runs/mccfr/v1.pkl \
    --hands 20000 --workers 8 --out results/ladder_mccfr.json --log-dir runs/ladder
python scripts/run_ladder.py --config configs/ladder.yaml
```

* **Schedules**: `all` (round robin), `previous` (each entry vs its `k`
  predecessors and every anchor), `newest` (only the last entry vs its `k`
  predecessors and the anchors: the per-checkpoint job).
* **Incremental**: results live in one JSON file keyed by the pair of spec
  strings. A pair is skipped when it is already there with at least
  `--hands` hands. If you ask for more hands, the pair is replayed. The file
  is rewritten after every finished pair, so an interrupted ladder resumes.
  The file records the game. Opening it with a different `game:` is refused,
  because such results are not comparable.
* **Parallelism**: `--workers N` plays pairs in `N` spawned processes (one
  torch/BLAS thread each). Every pair uses the same deck seed, so the ladder
  has common random numbers across pairs.
* **Outputs**: `ladder.json` (matches, ratings, chain) and `ladder.md`. The
  markdown has a ratings table, a pairwise matrix (up to 12 agents; `*` marks
  CIs that exclude zero), every match with its deal W/L/T, and a
  **Regressions** section.

How to read it:

* `rating` (mbb/h) is the weighted least-squares fit of
  `r_a - r_b = mbb(a vs b)`, with weights `1/se^2` and a field mean of 0.
  When the agents are transitive it reproduces the pairwise numbers. Large
  disagreements between the matrix and rating differences mean
  intransitivity, for example a checkpoint that beats its parent but loses
  to the grandparent.
* `Elo` is Bradley-Terry fitted to duplicate deals won or lost, with the
  field mean at 1500. It ignores win size. An agent can win most deals and
  still lose chips (`equity` vs `always_raise` shows this). Treat mbb/h as
  primary.
* **Regressions**: a chain entry (checkpoint order) that loses to its
  predecessor with a CI below zero. DESIGN.md treats this as a red flag for
  the run, not a sample-size problem.

## Local best response (LBR)

```
python scripts/run_lbr.py --opponent blueprint:runs/mccfr/final.pkl --hands 20000 --workers 8
python scripts/run_lbr.py --config configs/lbr.yaml
```

The algorithm follows Lisý & Bowling (2017), heads-up. LBR plays real hands
against the opponent and tracks the opponent's range over the 1,326 hole-card
pairs. It updates the range by Bayes' rule with the opponent's policy after
every opponent action, and removes cards as they appear. At each decision it
computes its equity `wp` against the range with the batched torch
evaluator: exact on the turn and river, and `--runouts` sampled runouts on
the flop. It then picks the action with the best value, assuming the
opponent checks or calls down afterwards:

* fold: `-c_me`
* call: `(2wp - 1) * stake`
* raise: `fp * c_opp + (1 - fp) * (2wp - 1) * stake'`

Here `fp` is the range-weighted probability that the opponent folds to that
raise, and the stakes are the chips at risk at showdown. This is the
paper's rule, written in final-chip terms. Preflop LBR calls (`--preflop
call`, as in the paper); `fc` and `full` are options. The candidate raises are
the abstract spec's sizes (`--raises spec`) or pot plus all-in (`fcpa`).

**Opponent interface.** The opponent should be a `PolicyAgent`
(`pokerbot/agents/policy.py`). It must implement `policy(state, seat) ->
{abstract_index: prob}` (or an array over the `ActionSpec` indices) for
the hand `state.hole_cards(seat)`. It can optionally implement
`policy_batch(state, seat, holes[K, 2]) -> [K, A]`, which LBR prefers: one
call per decision instead of up to 1,326. Agents that only `act` are wrapped
by `SampledPolicyAgent`, which calls `act` `--samples` times per hand and
query and warns once. This is slow and noisy, so use it for small checks
only. `AlwaysCallAgent` gets its exact policy automatically. Both trained
blueprints (`blueprint:<strategy file>`, `neural:<run dir>`) implement
`policy` and `policy_batch` directly, over their own action spec, which LBR
then uses for its own raise candidates too.

**Outputs.** LBR's win rate in mbb/h with a CI, all-in adjusted and raw. When
all the money went in before the river, the realized result is replaced by
its expectation over the remaining board. This is unbiased and has much
lower variance, so it is the headline number. There is also a per-street
table:

| column | meaning |
|---|---|
| `ended` / `share` | hands that ended on that street (fold or all-in / showdown after that street's last action) |
| `mbb/h|ended` | LBR's average result in those hands |
| `contrib` | their contribution to the total mbb/h; the column sums to the headline number |
| `LBR actions` | what LBR chose on that street |

How to read it: LBR is a **lower bound** on exploitability. A large
positive number is a real hole. A number near zero does not prove the
strategy is good, because LBR is myopic. The per-street table shows where
the hole is. For example, most `contrib` on the river from LBR raises that
the opponent folds means the opponent over-folds rivers. Several strong
ACPC agents of 2016 lost more than a big blind per game (1,000+ mbb/g) to
LBR. A sound blueprint should come in well below that and trend down across
checkpoints. `range reset` counts hands where the observed action had
essentially zero probability under the reported policy (likelihoods are
floored at `1e-4`). That points to a policy/act mismatch in the opponent.

### LBR against the search (roadmap 1.4)

```
python scripts/run_lbr.py --config configs/lbr_search_vn.yaml --raises fcpa --preflop call \
    --hands 2000 --out results/lbr_search_fcpa_call.json --log-dir runs/lbr/search_fcpa_call
```

`--opponent search:...` works like any other spec (its search YAML from
`agents.search.config`, or `--search-config`). `as_policy_agent` turns the
`SearchAgent` into a `SearchPolicyAgent` (`pokerbot/search/policy.py`), which
plays through the agent with `record=True` and answers LBR's range queries
with `range_strategy(state, seat) -> (actions, [1326, n])` over the concrete
`(kind, amount)` actions. LBR uses that interface directly: the range update
takes the exact probability of the action the bot played, and `fp` is the
range-weighted fold column, with no mapping onto the action spec.

* **Real decisions**: the strategy the agent recorded there, the one it played.
* **After LBR's candidate raises**: a search at that hypothetical state, run on
  a fork of the agent (its own copies of the continual-resolving cache, the
  street roots and the played strategies; nets, blueprint and config shared).
  The fork resolves exactly as the real agent would on reaching that state, and
  the real hand's caches are untouched (`tests/search/test_policy.py` checks
  both, and that a hypothetical search equals the real one when LBR then
  plays that raise). Cached by public state for the hand. Off-tree sizes are
  added at the root as in real play; preflop goes through the solved table
  and its base blueprint off it, so preflop queries are lookups.
* The fork uses the bot's real hole cards (never LBR's). With
  `remove_own_blockers` the solve removes them from its model of LBR's range,
  so the rows of other hands are what this solve computed, not what the bot
  would compute holding them: a few blocker combos of difference.
* `--hypothetical-budget 0.5` halves the time of the hypothetical searches
  (cheaper, slightly less faithful fold probabilities). Default 1.
* The summary adds `opponent decisions per hand`: real and hypothetical
  searches by street with their mean seconds, and `*/fallback` if any search
  failed and played the blueprint.

**Cost.** CPU, `TINY` search config (`tests/search/test_agent.py`, uniform
blueprint, 100bb), 40 hands, 1 thread: a plain match costs 0.77 s per hand;
LBR `fcpa` 1.0-2.1 s per hand with 1.3 hypothetical searches per hand next to
1.0 real ones, and `spec` 2.3-4.7 s with 2.3-2.4 hypothetical ones (the
uniform blueprint's default spec). `full` is cheaper per hand because LBR
ends many hands preflop.

Production, GPU (RTX 4070 Ti, `configs/lbr_search_vn.yaml`, 2026-10-21), 6
hands of `fcpa`/`fc`: 22.2 s per hand. A real flop search took 5.0 s (4 s
budget plus recording and cache store), a turn search 2.3 s, a hypothetical
flop search 4.6 s and turn 2.5 s; per hand, 2.2 real postflop searches and 2.8
hypothetical ones. A normal hand of the bot is about 9 s, so LBR is about 2.5x
a match. `spec` raises use the blueprint's sizes: 7 open sizes plus all-in in
an unopened pot (8 searches per LBR decision, against 2 for `fcpa`), and the
same 2 facing a bet; a 3-hand `spec`/`fc` smoke ran 8 hypothetical flop
searches at its one flop decision. Facing bets are common, so expect `spec`
at roughly 2-3x the hypothetical cost of `fcpa`, about 40-50 s per hand.

| configuration | s per hand (est.) | 2,000 hands | 5,000 hands |
|---|---|---|---|
| `fcpa` / `call`, `fc` | ~22 (measured on `fc`) | ~12 h | ~31 h |
| `fcpa` / `full` | below that (hands end preflop) | < 12 h | < 31 h |
| `spec` / `call`, `fc` | ~40-50 | ~22-28 h | ~55-70 h |
| `spec` / `full` | below that | < 28 h | < 70 h |

The six configurations at 2,000 hands are about 4-5 GPU days on one card;
`--hypothetical-budget 0.5` cuts about a quarter (`fcpa`) to a third
(`spec`). One process per GPU (`workers: 1`); spot GPUs can run
configurations in parallel. Gate 1 wants every configuration under +1,000
mbb/hand; the published search bots made LBR lose (DeepStack -383 to -775,
ReBeL -881, Supremus -951), ACPC-era abstraction bots gave it 3,300-4,700.

## Approximate best response (ABR)

```
python scripts/run_abr.py --config configs/abr_tiny.yaml                 # CPU smoke run
python scripts/run_abr.py --config configs/abr_4070ti.yaml --opponent neural:<ckpt> \
    --log-dir runs/abr/<name> --out results/abr_<name>.json --checkpoint runs/abr/<name>/q.pt
```

The learner is a double DQN with a dueling head over the abstract actions,
trained on `VecNLHE` against a frozen opponent. Its features are the
`encode_obs` card embeddings (card, rank and suit), a bag of history tokens,
the scalar pot and stack features, and optionally Monte Carlo equity. The
learner sits in seat `slot % 2`, and the button alternates per re-deal, so
both positions are trained. Each hand is one episode (`gamma = 1`), and the
reward is the learner's net chips at the end of the hand. After the fixed
budget, the greedy learner and the untrained learner (the baseline) are
evaluated on the same deals (common random numbers). The report gives mbb/h
± CI, split by position.

**Opponent interface.** `VecPolicy.act(env, mask) -> LongTensor[n]` of
abstract actions. Only the `mask` slots, where the opponent is to act, are
used. `uniform` and `call` are vectorized. Any other spec is built through
the registry and used via the agent's `vec_policy(device)` method when it
has one. `neural:<run dir>` provides it (`NeuralVecPolicy`, the SD-CFR
average on `env.obs()` with per-slot reach tracking). The envs use the
opponent's `spec` when it has one, so the learner plays in the blueprint's
action abstraction. Otherwise the agent runs through `ScalarVecPolicy`, which
rebuilds each slot as a scalar `GameState` and calls `policy()` one slot at
a time. That path is for tests and small checks, not for the 4070 Ti config.

**Richer learner actions** (`configs/abr_4070ti_rich.yaml`). By default the
learner is confined to the opponent's action abstraction, so it can't use bet
sizes the blueprint never considers, as a real opponent would. With
`abr.learner_actions` (a spec mapping: `streets`, `max_raises`, `dedupe`), the
learner picks among its own action set, and every choice is played as real
chips (`VecNLHE.step_concrete`). The opponent records each off-tree size the
way it would in real play: translated into its own abstraction by the
randomized pseudo-harmonic mapping (`abr.offtree: harmonic`,
`env.actions.harmonic_abstract`, which mirrors `abstraction.actions.map_offtree`),
or by the nearest size (`nearest`). The learner keeps its own history of the
real actions and its own legal mask (`LearnerView`). This needs a vectorized
opponent: `ScalarVecPolicy` rebuilds slots from their abstract history, which
off-tree raises make inexact, so it is rejected. A run on a richer set can find
more than one confined to the blueprint's abstraction, so compare results only
under the same config.

How to read it: `final` is what the exploiter wins, which is a lower bound on
exploitability, like LBR's but able to find multi-street lines. The
`untrained` row is the baseline, and the difference shows the learning
worked. Compare opponents only under the same config and budget, since a
bigger budget finds more. The train curve (`abr/train_mbb`, epsilon-greedy
play) should rise and flatten. If it is still rising at the end, the budget
is too small for the number to mean much.

## Archetype suite

Scripted opponents that stand in for "the average player" (roadmap 1.3,
Gate 1), in `pokerbot/agents/archetypes.py`. They are ordinary agent
specs (`population`, `nit:fold_to_bet=0.7`, ...), take about a millisecond
per decision on CPU and are always legal (tested over 3,000 hands each,
short and uneven stacks included).

* `ProfileAgent` plays by frequencies: VPIP, PFR, 3-bet, 4-bet, fold to
  3-bet, c-bet, fold to c-bet / to a bet, aggression, bluff share, sizes.
  Hand strength is a percentile: preflop among the 1,326 combos by equity
  against a random hand; postflop by Monte Carlo equity (the `equity`
  baseline's helper), ranked within the agent's own preflop range.
* Presets: `population` (the average losing player: 35/12, 3-bet 4%, c-bet
  55%, fold to c-bet 50%, Agg 35%, from Poker Copilot's published ranges; the
  module docstring cites each), `regular` (23/20), `calling_station`, `nit`,
  `maniac`, `loose_passive` (limper), `fit_or_fold`, `min_raiser` (every raise
  a min-raise), `overbet_bluffer` (polarised 1.5-2.5x pot).
* `translation_stress` builds on `RandomAgent`: sizes from a min-raise to
  all-in on every street (uniform, log-uniform and both ends), odd opens of
  2-12bb, limps, limp-reraises and donk bets.
* `equity`, the existing baseline, is in the suite too (`SUITE`).

Each agent counts its own HUD statistics (`agent.hud`), and the runner reports
them. Against the blueprint they come out near the targets, with drift where
the ranges narrow: VPIP and PFR land within 3 points, c-bets within 10, folds
to c-bets lower than the target because the blueprint bets small (folds scale
with bet size, `size_sensitivity`).

```
python scripts/run_archetypes.py --agent neural:runs/dcfr4_distilled_v2,device=cpu \
    --deals 3000 --chunk 500 --threads 2 --out runs/archetypes/blueprint_cpu.json
python scripts/run_archetypes.py --config configs/match_search_vn.yaml \
    --agent search:preflop:runs/preflop/preflop_r15.pt --reference neural:runs/dcfr4_distilled_v2 \
    --deals 10000 --chunk 250 --out runs/archetypes/production.json --resume
```

Every archetype plays `--deals` duplicate deals, luck-adjusted. With
`--reference` the second agent plays the same deals with the same seeds, and
the table adds the paired difference per deal (Gate 1's regression check).
`--out` is rewritten after each chunk; `--resume` continues it. Run it from
the checkout that holds `runs/` and `data/` (relative blueprint paths). On a
busy machine pass `--threads 2`: with torch's default thread count a CPU
blueprint match ran 10-25x slower when other jobs held the cores.

**Blueprint `dcfr4_distilled_v2` (CPU), 2026-10-21:** 3,000 duplicate deals
(6,000 hands) per archetype, 100bb, luck-adjusted mbb/hand with 95% CI. 47 min
on a shared machine.

| Archetype | Blueprint margin | Archetype VPIP/PFR/3-bet/c-bet/fold-c-bet/Agg (measured) |
|---|---|---|
| population | +145 [-24, +308] | 31/9/3/62/38/29 |
| regular | +303 [+163, +454] | 22/15/7/60/33/40 |
| calling_station | +3,056 [+2,548, +3,604] | 54/3/1/28/10/12 |
| nit | +477 [+367, +581] | 13/9/3/52/55/21 |
| maniac | +1,114 [+665, +1,582] | 83/59/32/90/3/73 |
| loose_passive | -86 [-325, +148] | 46/2/1/33/31/15 |
| fit_or_fold | +386 [+196, +565] | 36/12/4/43/72/26 |
| min_raiser | +158 [-15, +324] | 41/23/9/69/23/43 |
| overbet_bluffer | +503 [+216, +767] | 37/19/6/62/23/46 |
| translation_stress | +2,191 [+1,645, +2,800] | 91/60/50/48/28/68 |
| equity | +1,428 [+1,072, +1,789] | - |

The blueprint beats 8 of 11 with the CI above zero. It does not clearly
beat `population` or `min_raiser`, and it is even with `loose_passive` (raw
-310 [-631, +4]): limped pots and passive calling are the weak spots to watch.
For Gate 1 the production bot must beat each one over 20,000 hands and not
by significantly less than the blueprint.

The bot makes 0.84 preflop, 0.48 flop, 0.29 turn and 0.19 river decisions per
hand on average over these archetypes (from 0.23 / 0.10 / 0.06 postflop
against `nit` to 0.70 / 0.60 / 0.54 against `calling_station`). At
production's budgets (flop 4 s, turn 2 s, river 1 s) that is about 2.7 s per
hand, so the Gate 1 run (10,000 deals = 20,000 hands for each of 11
archetypes, 220,000 hands) is about 165 GPU hours, roughly a week; the
blueprint reference adds under an hour.

## Logging

`RunLogger(log_dir, tensorboard=None, config=None)` writes `scalars.csv`
(`wall_time, step, tag, value`, appended across restarts), `run.json` (git
hash, start time, config) and TensorBoard events when `tensorboard` is
installed. The import is guarded; install it with `uv pip install tensorboard`
or the `logging` extra. `read_scalars(dir)` loads the CSV back. The ladder
logs `ladder/<pair>`, `rating/<agent>` and `elo/<agent>`. LBR logs
`lbr/mbb`, its CI and per-street contributions. ABR logs `abr/train_mbb`,
`abr/loss`, `abr/eps` and `abr/eval_mbb`.

## Expected run times

Measured in the development container: 4 shared CPU cores, Rust engine, no
GPU, torch limited to 1 thread per process.

| Job | Time |
|---|---|
| Ladder, 4 baselines, 2,000 hands per pair, 4 workers | about 1-3 s per pair (Monte Carlo equity in `equity` dominates) |
| LBR vs `always_call` / `uniform` | ~20-40 ms per hand per process (flop equity with 64 runouts dominates) |
| LBR vs an act-only agent (`SampledPolicyAgent`) | seconds per hand (1,326 hands x samples `act` calls per opponent action) |
| LBR vs the search, `TINY` config on CPU | 1-5 s per hand (see "LBR against the search"; production on the GPU about 22 s per hand with `fcpa`) |
| ABR `abr_tiny.yaml` (256 envs, 400 steps, 2 x 4,096 eval hands) | ~20 s on 1 thread. With all threads on a busy box it can be 10x slower. Set `threads: 1` |
| `pytest tests/eval` | ~20 s |

On the 4070 Ti box (8-16 cores) these are **estimates, not measurements**:

* **Ladder**: time is dominated by the agents. A tabular blueprint lookup is
  microseconds, so a 200k-hand duplicate pair takes a few minutes per worker.
  A search agent at 1-2 s per decision needs about 400k decisions for a
  200k-hand pair, which is days. Play search agents over far fewer hands, or
  only against the anchors.
* **LBR**: CPU-bound per decision. Run it on CPU with one process per core
  (`--workers`). With a `policy_batch`-capable blueprint, expect about
  50-100 ms per hand per worker, so 20k hands take roughly 5-15 minutes on 12
  workers. A 20k-hand LBR still has a CI of several hundred mbb/h. Use the
  all-in adjusted number and more hands for release candidates.
* **ABR** with `abr_4070ti.yaml` (16k envs, 40k steps, 2 updates per step,
  200k-hand evaluations): the env and equity features run on the GPU. With an
  opponent network on `env.obs()`, expect roughly 30-80 ms per step, so 1-2
  hours of training plus a few minutes per evaluation. Treat this as a
  weekly job, as DESIGN.md plans. Tune `n_envs`, `updates_per_step` and
  `train_steps` on the box, and watch whether `abr/train_mbb` has flattened.

## Tests

```
python -m pytest tests/eval -q
```

They check the registry round trip, including lazy-import errors. They check
that the scalar abstract actions match `VecNLHE` exactly on random play. The
ladder test plays three baselines, checks the ratings and markdown, and checks
that incremental reruns skip finished pairs. There is a regression-flagging
CLI run. LBR tests check exact river equity, the Bayesian range update
(normalized, card removal), and that LBR beats `always_call` with a CI far
above zero. The ABR test checks that a 300-step CPU run beats a uniform
random opponent by a clear margin over the untrained learner. The logger test
checks the files it writes. `test_aivat.py` checks that an action term
averages to zero over the bot's draws at a fixed decision (and exactly in
expectation), the pair-to-range fallback, that zero-probability actions are
ignored, and that a recording search bot's AIVAT is its luck-adjusted result
plus its action terms, with no preflop terms.
