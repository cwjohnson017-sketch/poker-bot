# Self-Trained No-Limit Texas Hold'em Bot — Design

Target hardware: one NVIDIA RTX 4070 Ti (12 GB VRAM) in a desktop with a
multi-core CPU and (assumed) 32 GB+ system RAM. No hand histories, no
outside solvers: the bot learns entirely from self-play.

## 1. Recommendation in one paragraph

Build a **heads-up, 100 big-blind, cash-game** bot first. Train an
equilibrium-approximating **blueprint** strategy with counterfactual regret
minimization (CFR) self-play, then add **real-time search** at play time
that re-solves the current subgame on the GPU. Do the blueprint in two
passes: a cheap **tabular Monte Carlo CFR (MCCFR)** blueprint on the CPU
that gives us a working bot and a benchmark opponent within about two
weeks, followed by a **neural blueprint (Deep CFR family)** that uses the
GPU for batched self-play and training and must beat the tabular one to
be adopted. The GPU earns its keep in three places: a vectorized poker
environment that plays hundreds of thousands of hands in lockstep, neural
network training and inference, and range-vs-range CFR for the real-time
search. Six-max is an extension, not the starting point.

## 2. Goals and non-goals

Goals

- A bot that plays heads-up NLHE (blinds 50/100, 20,000-chip stacks, the
  ACPC standard) at a strength clearly above simple rule-based agents and
  above its own earlier checkpoints, measured in milli-big-blinds per hand
  (mbb/h) with confidence intervals.
- Pure self-play: no human data, no external solver output.
- Every training phase fits on one 4070 Ti plus the host CPU, and a full
  training run takes days, not months.
- Reproducible runs: seeds, configs, and checkpoints are versioned.

Non-goals for version 1

- Six-max or tournament play (the design leaves room for both).
- Opponent modeling or exploitation. Version 1 aims for a Nash
  approximation. An exploitative layer can sit on top later.
- Playing on real-money sites. The play interface is a local protocol for
  bot-vs-bot and bot-vs-human matches.

## 3. Hardware budget

| Resource | Figure | What it constrains |
|---|---|---|
| GPU compute | ~40 TFLOPS FP32, up to ~160 TFLOPS FP16 tensor | Network training and batched inference. Plan on 20–30% utilization. |
| VRAM | 12 GB | Batch size of the vectorized environment, model size, on-GPU lookup tables. |
| VRAM bandwidth | ~504 GB/s | Hand evaluation and equity computation are gather-bound. |
| System RAM | assume 32 GB (64 GB is comfortable) | Tabular blueprint regret tables and Deep CFR reservoir buffers. |
| CPU cores | assume 8–16 | Tabular MCCFR traversals, search-tree building, data loading. |
| Disk | ~100 GB free | Checkpoints, equity tables, replay buffers on disk. |

Neural networks for poker are small (a few million parameters). VRAM will
be spent on environment state and lookup tables, not on models. Nothing in
this design needs more than one GPU.

## 4. System overview

```
                 ┌────────────────────────────────────────────┐
                 │              Game engine                    │
                 │  Rust core (rules, side pots, evaluator)    │
                 │  Torch vectorized env (N hands in lockstep) │
                 └───────┬──────────────────┬──────────────────┘
                         │                  │
        ┌────────────────▼──────┐   ┌───────▼────────────────────┐
        │ Phase 1: tabular MCCFR│   │ Phase 2: neural blueprint  │
        │ CPU, card+action      │   │ Deep CFR / SD-CFR on GPU   │
        │ abstraction           │   │ raw cards + equity feats   │
        └────────────┬──────────┘   └───────┬────────────────────┘
                     │  blueprint strategy   │
                     └──────────┬────────────┘
                                │
              ┌─────────────────▼──────────────────┐
              │ Phase 3: real-time search           │
              │ range-vs-range CFR on GPU, depth-   │
              │ limited, blueprint rollouts at leaf │
              └─────────────────┬──────────────────┘
                                │
       ┌────────────────────────▼───────────────────────────┐
       │ Evaluation harness (always on)                      │
       │ duplicate matches, checkpoint ladder, local best     │
       │ response, approximate best response, unit tests      │
       └────────────────────────────────────────────────────┘
```

## 5. Components

### 5.1 Game engine

Two implementations of the same rules, cross-checked against each other.

**Rust core (`engine/`)**, exposed to Python through PyO3.

- Exact NLHE rules: blinds, min-raise tracking, all-ins, side pots, split
  pots, showdown ordering, heads-up button rules. Parameterized by number
  of players so six-max works later.
- A 7-card hand evaluator (perfect-hash 5-card ranks with a 21-combination
  max, or the 2+2 lookup table). Verified against a slow reference
  evaluator on millions of random hands.
- Public-state and information-set encoders that produce a stable byte key
  (used by the tabular blueprint) and a fixed-size integer vector (used by
  the networks).
- Single-threaded speed target: 5M+ hand evaluations per second per core,
  and full random hands at 500k+ per second per core.

**Torch vectorized environment (`env/`)**.

- State stored as tensors with a leading batch dimension: cards
  `[N, 9]`, stacks `[N, 2]`, bets, pot, street, actor, betting-history
  tokens `[N, T]`, done flags. Every step is a handful of masked tensor
  operations. Nothing per-hand in Python.
- Dealing by argsort of random keys per hand. Showdown evaluation by
  gathers into the 2+2 table held in VRAM (about 130 MB).
- `torch.compile` or CUDA graphs to cut kernel-launch overhead.
- Batch of 128k–256k hands. Target 1M+ agent decisions per second before
  network inference; the network is expected to be the bottleneck.
- Agreement test: replay the same seeds through Rust and Torch and require
  identical outcomes.

Why both: the tabular blueprint and the search-tree builder need a fast
scalar engine on the CPU. The neural blueprint needs throughput that only
a batched GPU environment gives. Numba is an acceptable fallback if we
want to stay Python-only, at roughly 3–5x slower for the tabular phase.

### 5.2 Hand evaluation and equity on the GPU

- Equity-vs-random-hand and equity-vs-range for a batch of (hand, board)
  pairs, computed by enumeration on the river and turn and by sampling on
  the flop and preflop. This is a pure gather workload and runs at
  billions of evaluations per second.
- Used for three things: card abstraction for Phase 1, input features for
  Phase 2, and leaf evaluation in search.
- Preflop 169-class equity and the flop equity histograms are computed
  once and cached on disk.

### 5.3 Abstractions

**Action abstraction** (shared by all phases, configurable per street):

| Street | Actions |
|---|---|
| Preflop | fold, call, raise 2.5x (open), 3x (3-bet), pot, all-in |
| Flop | fold, check/call, 0.33 pot, 0.75 pot, 1.5 pot, all-in |
| Turn | fold, check/call, 0.5 pot, 1.0 pot, all-in |
| River | fold, check/call, 0.5 pot, 1.0 pot, 2.0 pot, all-in |

Raises are capped at four per street. Sizes are rounded to legal amounts.
Off-tree opponent bets are mapped to the nearest size with the
pseudo-harmonic mapping (Ganzfried & Sandholm, 2013) during blueprint
play, and added to the subgame tree as an extra branch during search.

**Card abstraction** (Phase 1 only):

- Preflop: the 169 lossless classes.
- Flop, turn, river: suit-isomorphism canonicalization, then k-means on
  equity-distribution histograms (earth mover's distance on the flop and
  turn, plain equity on the river). Start at 1,000 buckets per street and
  grow if RAM allows. Bucketing runs on the GPU from the cached equity
  tables.

Phase 2 uses no card abstraction. Networks see raw card embeddings plus
the equity features from 5.2.

### 5.4 Phase 1: tabular MCCFR blueprint (CPU)

- Algorithm: external-sampling MCCFR with linear weighting (Linear CFR)
  and Pluribus-style pruning of actions with very negative regret. This is
  the best-understood recipe for the strength it gives.
- Storage: regret and strategy-sum tables in a Rust hash map keyed by the
  info-set byte key. With 169/1,000/1,000/1,000 buckets and the action
  set above, expect tens of millions of info sets and a few gigabytes of
  RAM. Grow buckets only if the memory allows.
- Parallelism: one traverser thread per core, lock-free updates on
  sharded tables (regret updates tolerate races in practice).
- Budget: about 1–2 days on 8–16 cores for a usable blueprint.
- Output: an averaged strategy exported to a compact file the play and
  search components can load, plus a checkpoint every hour.

This phase gives us a bot to play against, a baseline for every later
experiment, and the ranges that the real-time search uses on the flop.

### 5.5 Phase 2: neural blueprint (GPU)

Algorithm: **Deep CFR** with external sampling, using the
**single-deep-CFR (SD-CFR)** variant so the final strategy is the
iteration-weighted average of the sequence of advantage networks rather
than a separately trained policy network. That removes one approximation
error and the networks are small enough to keep every checkpoint.

**Batched traversal.** External sampling branches at the traverser's
nodes and samples at the opponent's and chance nodes. On the vectorized
environment we run traversals as a frontier: all live nodes across all
traversals at the current depth are expanded together, traverser nodes fan
out across every abstract action, and the advantage network is queried
once per frontier in a single batch. Traversal depth is capped, with a
blueprint rollout used past the cap, so a single traversal cannot blow up.

**Networks.**

- Input: 7 card slots (rank and suit embeddings, board slots masked when
  undealt), betting history as a sequence of up to 24 action tokens with
  street and position, pot and stack as fractions of the starting stack,
  equity-vs-random and a 10-bin equity histogram from 5.2.
- Body: card branch (embedding sum plus MLP) and a history branch
  (small transformer or GRU over the tokens), concatenated into a 3-layer
  MLP of width 512. Roughly 2–4M parameters.
- Output: one advantage per abstract action, masked to legal actions.
- Precision: bfloat16 autocast for the forward pass, fp32 master weights.

**Memories.** Reservoir buffers of 40M samples per player, stored on the
host in a compact binary layout (cards as bytes, history as bytes,
advantages as float16, iteration weight as uint16) at around 60 bytes per
sample. That is under 10 GB for both players including a strategy memory.
Sampling minibatches from host memory into pinned buffers is faster than
it sounds and keeps VRAM free for the environment.

**Loop per CFR iteration.**

1. For each player: run K traversals on the GPU environment, appending
   regret samples to that player's memory.
2. Re-initialize that player's advantage network from scratch (as in the
   paper) and train it on its memory for a fixed number of SGD steps with
   Adam, batch 10k, iteration-weighted loss.
3. Save the network. Every M iterations, run the evaluation harness
   against the tabular blueprint and the previous checkpoint ladder.

**Alternative kept in reserve.** If the frontier traversal turns out
awkward, ESCHER (McAleer et al., 2022) estimates regrets from plain
sampled trajectories using a learned history-value network. It maps onto
the vectorized environment exactly like a PPO rollout loop. It is the
fallback, not the plan, because it has less track record on full NLHE.

**Adoption rule.** The neural blueprint replaces the tabular one only
when it wins a 200k-hand duplicate match against it with a 95% confidence
interval that excludes zero. Until then the tabular blueprint stays the
shipping strategy.

### 5.6 Phase 3: real-time search

This is the largest single strength multiplier and is independent of
which blueprint we end up with.

- Preflop is played from the blueprint.
- From the flop onward the bot re-solves the remaining game from the
  current public state. Both players' ranges (1,326 combos each, with
  blocked cards removed) come from the blueprint reach probabilities,
  updated by the observed actions along the hand.
- Solver: discounted CFR+ over the subgame with ranges represented as
  tensors, so every node update is a vectorized operation over hand
  combos. This is how commercial solvers work and it fits the GPU well.
- Depth limit: solve to the end of the current street plus one. At the
  depth limit, leaf values come from rolling out the blueprint with a small
  set of biased continuation strategies (Pluribus's four: blueprint,
  fold-biased, call-biased, raise-biased), with the opponent choosing the
  best one. Rollouts are batched on the GPU environment. A learned
  public-belief-state value network (DeepStack/ReBeL style) is the
  version-2 replacement for rollouts.
- Budget: under 2 seconds per decision on the flop, under 1 second on
  later streets. Tree sizes are tuned to that budget by removing bet sizes
  on deep subgames.
- Safety: the subgame gadget from safe resolving is used so that the
  re-solved strategy cannot be exploited by an opponent who deviates on an
  earlier street. Unsafe resolving is available as a config flag for
  comparison.

### 5.7 Evaluation harness

Runs continuously from the first week and gates every adoption decision.

- **Engine tests.** Hand evaluator against a reference on 10M random
  hands; side-pot and split-pot scenarios from a hand-written table;
  Rust-vs-Torch replay agreement.
- **Duplicate matches.** Every pair of agents plays each deal from both
  seats with the same cards. Report mbb/h with a bootstrap 95% confidence
  interval. 200k hands is the standard match length.
- **Checkpoint ladder.** Every new checkpoint plays the last five
  checkpoints and the tabular blueprint. A checkpoint that loses to its
  predecessor is a red flag for the run, not a sample-size problem.
- **Local best response** (Lisý & Bowling, 2017). A cheap lower bound on
  exploitability that catches gross holes such as never bluffing on the
  river. Run on every checkpoint.
- **Approximate best response.** Train a DQN-style best-response agent
  against a frozen policy on the GPU environment for a fixed budget and
  report how much it wins. This is the closest we can get to true
  exploitability at this game size. Run weekly and on release candidates.
- **Baselines.** Always-call, always-raise, a simple equity-threshold
  agent, and the tabular blueprint. External bots such as Slumbot are used
  if their public interfaces are still reachable, but nothing depends on
  them.
- **Dashboards.** Training curves, mbb/h ladder, LBR trend, and search
  timing per street, logged to TensorBoard or Weights & Biases.

### 5.8 Play interface

- A local ACPC-style text protocol over a socket so any agent (bot,
  human CLI, external bot) can sit at the table.
- Agents: `BlueprintAgent`, `SearchAgent`, `HumanCLIAgent`, and the
  baselines above, all behind one interface: `act(observation) -> action`.
- A match runner that handles duplicate seating, seeds, logging, and hand
  history export in a plain text format.

## 6. Training plan and milestones

| Week | Milestone | Done when |
|---|---|---|
| 1–2 | Engine and evaluator, Torch env, engine tests | Rust and Torch agree on 1M seeded hands; env runs 100k hands in lockstep. |
| 2 | Equity tables and card buckets | Preflop, flop, turn, river caches on disk; bucket assignment reproducible. |
| 3–4 | Tabular MCCFR blueprint | Beats all rule baselines by a wide margin; LBR shows no gross holes. Match runner and duplicate evaluation working. |
| 5–8 | Neural blueprint | Deep CFR loop runs end to end; checkpoint ladder trending up; at least one checkpoint beats the tabular blueprint by the adoption rule. |
| 8–10 | Real-time search | Search agent beats the blueprint it is built on; per-decision time within budget. |
| 10–12 | Hardening and long runs | Multi-day runs with no regressions, approximate best response numbers on the release candidate, written results. |

Wall-clock training expectations on the 4070 Ti: the tabular blueprint is
a 1–2 day CPU job; a neural blueprint run is 3–7 days of GPU time; the
search needs no training. A full cycle from empty repo to a search-enabled
bot is roughly three months of part-time work, with the first playable
bot at the end of week four.

## 7. Risks and mitigations

- **Deep CFR fails to beat the tabular blueprint.** This is a real
  possibility at hobbyist scale. Mitigation: the tabular blueprint plus
  search is already a strong bot, and the adoption rule prevents
  regressions. ESCHER is the second attempt before abandoning the neural
  blueprint.
- **Traversal blow-up in external sampling.** Mitigation: depth cap with
  blueprint rollouts, and a smaller action set on later streets.
- **Rules bugs that silently distort training.** Mitigation: dual engines
  with replay agreement, and hand-written side-pot cases in CI.
- **RAM pressure from memories and tables.** Mitigation: compact binary
  layouts, memory-mapped buffers, and bucket counts sized to the machine.
- **Search too slow per decision.** Mitigation: fewer bet sizes deep in
  the tree, shallower depth limit, and caching of the flop solve across
  the hand.
- **Overfitting to self.** Mitigation: the checkpoint ladder and the
  approximate best response, both of which catch a strategy that only
  wins against its own lineage.

## 8. Repository layout and tooling

```
poker-bot/
  engine/          Rust crate: rules, evaluator, encoders, PyO3 bindings
  env/             Torch vectorized environment and GPU equity kernels
  abstraction/     action sets, suit isomorphism, bucket generation
  blueprint/
    mccfr/         tabular MCCFR (Rust) and exporter
    deepcfr/       traversal, memories, networks, training loop
  search/          subgame builder, range CFR+, leaf rollouts, gadget
  eval/            match runner, duplicate matches, LBR, approximate BR
  agents/          agent interface and implementations
  protocol/        ACPC-style socket protocol and human CLI
  configs/         YAML run configs (one per experiment)
  scripts/         entry points: build tables, train, evaluate, play
  tests/           engine, env, abstraction, and solver tests
  DESIGN.md
```

- Python 3.11+, PyTorch 2.x with CUDA 12, Rust stable with maturin for the
  bindings. `uv` for Python dependencies, `cargo` for the crate.
- Every run is a config file plus a git commit hash. Checkpoints carry
  both.
- CI runs the unit tests on CPU. GPU tests run locally before merging.

## 9. Decisions and open questions

Decided

- Heads-up first, 100 bb cash, ACPC blinds. Six-max later.
- Equilibrium-seeking, no opponent modeling in version 1.
- Tabular blueprint first, neural blueprint second, search third.
- Rust for the scalar engine and the tabular solver; Python and PyTorch for
  everything neural.

Open

- Exact bucket counts for Phase 1 depend on measured RAM use; start at
  1,000 per street and revisit.
- Whether the history branch of the network is a transformer or a GRU is
  an experiment, not a decision. Start with the GRU as it is cheaper.
- Whether to keep flop solves cached for the whole hand or re-solve on
  every action. Cache first, measure exploitability difference later.
- Six-max needs a multi-player blueprint (MCCFR handles it, Deep CFR is
  less proven) and a redesigned search with more than one opponent range.
  Not in scope until the heads-up bot is done.
