# `pokerbot.eval`: evaluation harness

Everything that says whether a bot got better lives here (DESIGN.md 5.7).

| Tool | Module / entry point | Answers |
|---|---|---|
| Match runner | `match.py`, `scripts/play_match.py` | How does A do against B? (plain or duplicate, mbb/h ± CI) |
| Checkpoint ladder | `ladder.py`, `scripts/run_ladder.py` | Is each checkpoint better than the previous ones and the anchors? |
| Local best response | `lbr.py`, `scripts/run_lbr.py` | Does the policy have gross holes? (cheap exploitability lower bound) |
| Approximate best response | `abr.py`, `scripts/run_abr.py` | How much can a trained exploiter win? (stronger lower bound) |
| Run logger | `logging.py` | CSV + TensorBoard scalars for all of the above and for training loops |
| Agent registry | `pokerbot/agents/registry.py` | Build any agent from a string such as `blueprint:runs/x.pkl` |

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
  the chance terms of AIVAT; its action terms are not implemented. Equities
  are shared by the two seatings of a duplicate deal, so mirror matches
  still cancel exactly.

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
checks the files it writes.
