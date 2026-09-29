# Tabular MCCFR blueprint

Phase 1 of `DESIGN.md` (section 5.4): an equilibrium-approximating strategy
for heads-up no-limit hold'em, trained on the CPU by external-sampling Monte
Carlo CFR over an abstracted game. The solver is Rust (`engine/src/`), driven
from Python:

| File | What |
|---|---|
| `engine/src/abstraction.rs` | Action abstraction (abstract -> concrete, de-duplication, raise cap, pseudo-harmonic mapping), card abstraction (`Bucketer` trait, preflop classes, default hand-strength buckets, table buckets). |
| `engine/src/isomorphism.rs` | Suit-isomorphic canonical hand index `canonical_index` / `canonical_unindex`. |
| `engine/src/mccfr.rs` | Tables, traversal, `Trainer`, checkpoints, strategy export, `BlueprintStrategy` loader. |
| `engine/src/python_mccfr.rs` | PyO3 bindings (`poker_engine.Trainer`, `ActionAbstraction`, `CardAbstraction`, `BlueprintStrategy`, helpers). |
| `engine/examples/mccfr_bench.rs` | Throughput and betting-tree size benchmark. |
| `train.py` | YAML config -> trainer config, training loop with logs, checkpoint, export. Entry point `scripts/train_mccfr.py`. |
| `agent.py` | `BlueprintAgent` (the `Agent` protocol) playing an exported strategy. |
| `export.py` | Pure-numpy reader of strategy files; key encode/decode. |
| `br.py` | Best response in the abstract game over sampled deals (an exploitability proxy for small games). |
| `interfaces.py` | Action-spec normalization, the card-abstraction protocol, `write_bucket_table`. |

## Running it

```bash
# once: build the extension into the active venv
cd engine && maturin develop --release && cd ..

# smoke run: 100bb, reduced actions, 20 postflop buckets, ~1 minute on 4 cores
python scripts/train_mccfr.py --config configs/mccfr_small.yaml
python scripts/play_match.py --a blueprint:runs/mccfr_small/strategy.bin --b equity \
    --hands 20000 --duplicate --seed 0

# the real blueprint
python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml --threads 16
# after a stop (Ctrl-C saves a checkpoint and exports first):
python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml --threads 16 \
    --resume runs/mccfr_hunl/checkpoint.bin
```

`--iterations` is the total target, so a resumed run continues up to it.
`--out-dir DIR` redirects the checkpoint and strategy. When the match config
has no `game:` section, `play_match.py` uses the game the blueprint was
trained on.

### On the target machine (RTX 4070 Ti box, 8-16 cores, 32 GB)

The GPU is not used in this phase. Set `threads` to the number of physical
cores. `configs/mccfr_hunl.yaml` (100bb, the DESIGN.md action table,
169/1000/1000/1000 buckets) has at most ~73M infosets. That is 6-10 GB of RAM
while training, about 2.5 GB more during the final export, and a ~2 GB
strategy file. Checkpoints are written every hour to
`runs/mccfr_hunl/checkpoint.bin`. They are the full tables, so allow for
~6 GB on disk; the write goes to a temporary file and is renamed at the end.
Expect 1 to 1.5 days for the configured 3B iterations on 16 cores (see the
throughput below). Watch the log's `infosets`, `tables` and `rss` columns.
If RSS stays well under 32 GB, the postflop buckets can go up to ~2,500.

Measured throughput on the 4-core development container (other jobs were
sharing it, so treat these as lower bounds):

| Setup | nodes/s | iterations/s |
|---|---|---|
| `mccfr_small.yaml`, 4 threads | 2.3-4.8M | 18-42k |
| 100bb, default actions, 169/50/50/50, 4 threads (`mccfr_bench`) | ~4.2M | 15-25k |
| 100bb, default actions, 169/1000/1000/1000, 4 threads, tables growing | 1.1-2.0M | 6-10k |

A "node" is one visited history, terminal nodes included. An iteration is
one traversal per player. `cargo run --release --example mccfr_bench --
<threads> <seconds> <stack_bb> <buckets>` reproduces the numbers.

## Algorithm

External-sampling MCCFR (Lanctot et al. 2009) with the Pluribus
modifications (Brown & Sandholm 2019):

- **Sampling.** Each iteration runs one traversal per player. A traversal
  deals one random deck up front, which samples every chance node, and walks
  the abstract betting tree with the engine's `GameState`. The solver
  therefore plays exactly the engine's rules: blinds, min-raises, all-ins and
  side pots.
- **Traverser nodes.** All abstract actions are explored. The node value is
  `v = sum_a sigma(a) v(a)`, and each explored action's regret grows by
  `v(a) - v`.
- **Opponent nodes.** One action is sampled from the current strategy, and
  the current strategy is added to that infoset's strategy sum. This is
  external sampling's unbiased "simple averaging": the sampled player's own
  reach is accounted for by the sampling itself, and the traverser explores
  everything. So the strategy sum is updated along the sampled path of each
  traversal, never weighted by the traverser's reach.
- **Regret matching.** The strategy is `max(R, 0)` normalized, or uniform
  when no regret is positive. Negative regrets are kept (this is not CFR+).
  They are floored at `regret_floor`, as in Pluribus, so pruned actions can
  recover.
- **Linear CFR.** Every `lcfr_discount_every` iterations, until `lcfr_stop`,
  all regrets and strategy sums are multiplied by `k / (k + 1)`, where `k` is
  the number of intervals so far. An iteration in interval `j` then weighs in
  proportion to `j`. That is Linear CFR at interval granularity, and it keeps
  the f32 tables bounded; direct `t`-weighting would lose f32 precision after
  ~10^7 visits.
- **Pruning.** After `prune_start` iterations, 95% (`prune_prob`) of
  traversals skip traverser actions whose regret is below `prune_threshold`
  and whose current probability is 0. Pruning never applies on the river or
  to actions that end the hand. The other 5% of traversals explore
  everything.
- **Parallelism.** `run(iterations, threads)` spreads iterations over
  `std::thread` workers. The tables are sharded: `shards` hash maps behind
  their own mutexes, with a key's shard chosen by a hash independent of the
  map's own. Each update locks one shard briefly. With one thread a run is
  bit-for-bit deterministic for a given seed. With more threads it is not,
  because regret updates interleave.

Units: utilities and regrets are in big blinds.

## Infoset keys

Keys are 128-bit, the same in Rust (`mccfr::make_key`) and Python
(`export.make_key`):

```
bits 126..127  street (0 preflop .. 3 river)
bits  97..125  bucket of the acting player on that street (own cards only)
bits   0..96   betting sequence: a leading 1 bit, then one 4-bit token per
               action of the hand so far (both players, all streets, oldest
               first); the token is the action's index in its street's
               abstract action list
```

Seats are not stored. The sequence determines who acts, so a key is relative
to position (the first preflop actor is the button / small blind). Street
boundaries are implied by the sequence, since the betting rules close the
rounds, and the street field makes them explicit. The sequence field holds 24
tokens. Heads-up with a raise cap of 4, a street has at most 6 actions, so the
cap is limited to 4 and the solver to two players. More players need a wider
key; the traversal itself is written for `n` players.

## Tables and checkpoints

Each shard holds a `HashMap<(u64, u64) key, (u32 offset, u32 n)>` and a
`Vec<f32>`. The vector holds `n` regrets followed by `n` strategy sums per
infoset, `n` being the number of abstract actions available there. Measured
at ~60-75 bytes per infoset. `Trainer.stats()` reports `infosets`,
`table_bytes` (map capacity plus arrays), `rss_bytes`, and throughput;
`stats(detailed=True)` adds `infosets_per_street`.

A checkpoint (`Trainer.save` / `Trainer.load`, `PBMCCFR1`) contains, in
little-endian: magic, version, binary solver config, metadata JSON,
iteration and node counters, training seconds, the entry count, then one
record per infoset: `u64 key_hi, u64 key_lo, u8 n, n x f32 regrets,
n x f32 strategy sums`. The periodic checkpoint (`checkpoint.path`, every
`interval_seconds`) is written between batches.

## Strategy file

`Trainer.export_strategy(path)` writes the averaged strategy, the normalized
strategy sums. When an infoset's sum is 0, the current regret-matching
strategy is used instead, and the infoset is left out when no regret is
positive either. The layout is `PBSTRAT1`, little-endian:

```
magic "PBSTRAT1" | u32 version (1)
u32 len + solver config (binary; read by the Rust loader)
u32 len + solver config as JSON (game, actions, cards, schedule)
u32 len + metadata JSON (git hash, YAML config, timestamp)
zero padding to a multiple of 8
u64 N | u64 P
N x (u64 key_hi, u64 key_lo)    sorted ascending as u128
(N + 1) x u32 offsets           row i = probs[offsets[i]:offsets[i+1]]
P x u16 probs                   round(p * 65535)
```

Row `i` has one probability per entry of `ActionAbstraction.legal(state)` at
that infoset: abstract list order, after de-duplication. Readers are
`poker_engine.BlueprintStrategy(path)` (binary search, used by the agent) and
`pokerbot.blueprint.mccfr.read_strategy(path)` (numpy).

## Abstractions

**Actions.** Each street has a list of `fold`, `check_call`,
`("raise", f)` and `allin` (the contract format). A raise to
`current_bet + f * (pot + to_call)` is rounded to whole chips and clamped to
the legal range; at or above the stack it is all-in. Folding is offered only
when facing a bet. Raises are offered only while fewer than `max_raises` were
made on the street. Sizes that map to the same concrete action are merged:
the all-in keeps its own index, and otherwise the earliest entry wins. The
default is the DESIGN.md table. Heads-up preflop, a raise to `x` times the
current bet is `(x - 1) / 2` of the pot, so "2.5x open" and "3x 3-bet" are
0.75 and 1.0 pot, and the table's separate "pot" size coincides with 3x.

**Off-tree actions.** `ActionAbstraction.translate(abs_state, real_state,
action, u)` uses the pseudo-harmonic mapping (Ganzfried & Sandholm 2013). A
raise's pot fraction `x` in the real state is placed between the pot
fractions `a < x < b` of the neighbouring abstract raises in the abstract
state, and mapped to `a` with probability
`f(x) = (b - x)(1 + a) / ((b - a)(1 + x))`, else to `b`. Outside the range it
maps to the nearest size. A real all-in maps to the abstract all-in. Folds
and calls map to themselves.

**Cards.** Preflop uses the 169 lossless classes (`preflop_class`). A
preflop count below 169 groups the classes into equal-probability buckets
ranked by equity. Postflop, the default bucket is `floor(E[HS] * buckets)`,
where `E[HS]` is the equity against one uniformly random hand: exact
enumeration on the river, and a fixed Monte Carlo sample of `hs_samples`
runouts on the flop and turn, seeded by the canonical index so that it is
deterministic. It is computed once per suit-isomorphism class and cached in
lazily filled arrays (about 280 MB of address space; pages are touched only
when used). Replacing the default with the abstraction package's k-means
buckets is a config change: `cards.tables` names one `.npy` per street,
indexed by `canonical_index` (see `interfaces.write_bucket_table`). The
canonical indexing is documented at the top of `engine/src/isomorphism.rs`.
Hole cards and the board are two unordered groups, giving 169 / 1,286,792 /
13,960,050 / 123,156,254 classes. `canonical_unindex` gives a
representative hand for each index.

## BlueprintAgent

`BlueprintAgent(path)`, or `--a blueprint:path` in `play_match.py`, keeps two
shadow `GameState`s of the hand on a dummy deck. The *real* one replays the
actual actions. The *abstract* one plays the training game (training stacks),
where the agent's own actions are the abstract actions it chose and the
opponent's actions are mapped with `translate` (randomized by default,
`mapping="deterministic"` uses `u = 0.5`). At a decision the agent looks up
(street, own bucket from the real cards, abstract sequence), samples an
action (`greedy=True` takes the most likely one), and sizes it against the
real pot. Missing infosets play uniformly. When the abstract game stops
tracking the hand, the agent checks or calls; `agent.counters` counts these
fallbacks. That happens when the opponent re-raises past the raise cap, or
when a large bet was mapped to an all-in that the real game did not have.

## Tests

`tests/blueprint/` covers: the canonical index under suit permutations,
pseudo-harmonic values against hand computation, action sizes, de-duplication
and the raise cap, checkpoint round trip, determinism, and key/export
agreement between Rust and numpy. On a 10bb game with 8 buckets and
fold/call/pot/all-in, the abstract-game best-response value (`br.py`) falls
from ~1.75 bb/hand (uniform) to ~0.3 (10k iterations) to ~0.12 (120k), and
the blueprint beats `AlwaysCallAgent` and `RandomAgent` in duplicate matches.
`BlueprintAgent` plays 500 hands against `RandomAgent` without an illegal
action, on and off the training stack depth.

## Known limits

- Heads-up only (key layout); raise cap at most 4 per street.
- The default postflop buckets are a single hand-strength number. They do
  not capture draws (no potential-aware histograms); tables from the
  abstraction package fix that.
- Pruning and discount schedules are in iterations, not wall-clock minutes
  as in Pluribus. The HUNL config converts using the expected throughput.
- The average strategy is kept for every street. Pluribus keeps it only
  preflop and uses real-time search later; Phase 3 search can do the same.
