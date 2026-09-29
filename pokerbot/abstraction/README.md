# `pokerbot.abstraction`: actions, isomorphism, card buckets

The abstractions of DESIGN.md section 5.3, shared by the blueprints, search
and agents.

| File | What |
|---|---|
| `actions.py` | The action abstraction: the Python `ActionSpec`, conversions to and from the Rust `poker_engine.ActionAbstraction`, legal abstract actions of a contract `GameState`, pseudo-harmonic off-tree mapping `map_offtree`. |
| `isomorphism.py` | Torch/numpy helpers around the Rust canonical hand index: `canonical_index_batch`, `unindex_batch`, `representatives`, `enumerate_canonical` (cached on disk), `orbit_sizes`. |
| `buckets.py` | Equity features per canonical class, EMD / 1-D k-means, bucket tables in the engine's format; CLI `scripts/build_buckets.py`. |

## Action abstraction (`actions.py`)

There is one Python spec type, `ActionSpec` (defined in
`pokerbot/env/actions.py`, where the torch env uses it, and re-exported
here), with `DEFAULT_SPEC` following DESIGN.md. Per street it lists
`("fold",)`, `("check_call",)`, `("raise", pot_fraction)`,
`("raise_x", multiple)` and `("allin",)`. `as_spec` also accepts four lists,
a dict keyed by street name (YAML shorthands such as `call` and
`[raise, 0.75]`), or a Rust `ActionAbstraction`.

`to_rust(spec)` returns a `poker_engine.ActionAbstraction` that gives the
**same abstract indices, legal sets and raise-to amounts** as the env's
`raise_targets` / `legal_mask` in every state. `from_rust` goes the other
way. `tests/abstraction/test_actions.py` checks this on 1,200 random states:
random stacks and blinds, off-tree raise sizes, every street, up to the raise
cap. There are no remaining differences. The conversion handles four
details:

* **Rounding.** The env rounds half up in exact integers
  (`(f_milli * X + 500) // 1000`, with `X = pot + to_call`). Rust computes
  `round(f * X)` in `f64`. With a raw fraction such as 0.35 or 1.005, some
  products land just below `.5` (`1.005 * 100 = 100.49999...`), and Rust is
  one chip lower. `to_rust` passes `f + 1e-10`, which is provably exact for
  `X < 10^7` chips. The test checks every `X < 2M` for several fractions,
  plus a concrete game state. Note that a Rust abstraction built directly
  from YAML floats, as the MCCFR config does, keeps the raw fractions, so it
  can differ from the env by one chip at such pots.
* **Preflop multiples.** `raise_x m` becomes pot fraction `(m - 1) / 2`.
  This is exact heads-up without antes, where the pot after calling is always
  twice the current bet. It is refused on later streets, and when a config
  with antes or more players is passed.
* **Duplicates.** `DEFAULT_SPEC` has both `3x` and a pot raise preflop, which
  are the same raise heads-up. Rust rejects equal entries, so the repeat gets
  one more epsilon. Both sides then drop it as a duplicate amount, and the
  index layout is unchanged.
* **All-in.** The env masks a size that would be all-in. Rust clamps the size
  and merges it into the all-in entry. The results are identical when the
  street has an `allin` entry. `dedupe=False` has no Rust equivalent and is
  refused.

`map_offtree(spec, state, action, rng, mode)` returns the abstract index for
a concrete action, using the pseudo-harmonic mapping (Ganzfried & Sandholm
2013). Consider a bet of pot fraction `x` between abstract sizes `a < x < b`.
It maps to `a` with probability
`f(x) = (b - x)(1 + a) / ((b - a)(1 + x))`. Otherwise:

* a bet below the smallest size maps to the smallest, and one above the
  largest maps to the largest;
* an all-in maps to the abstract all-in;
* a fold maps to fold, or to check/call where folding is not an option;
* a call maps to check/call.

`mode="randomized"` draws `u` from `rng`. `mode="deterministic"` uses
`u = 0.5`, the nearest size under this metric. For a `poker_engine.GameState`
the call delegates to `ActionAbstraction.translate`. Other contract states
(the reference engine, the match runner's masked view) use a line-by-line
Python mirror, and the tests check that the two agree. `abs_state=` passes
the abstract game's decision point when its pot differs from the real one.
`NeuralBlueprintAgent` and `SearchAgent` can switch to it from their
nearest-amount mapping.

## Canonical index (`isomorphism.py`)

The index is defined in `engine/src/isomorphism.rs`; its module docs give
the exact scheme. In brief, the cards form two rounds: the hole cards and
the whole board, unordered. Per suit, the rank sets of the rounds are indexed
colexicographically in mixed radix. Suits are sorted by (count vector,
index). The sorted configuration picks an offset, and groups of suits with
equal count vectors are indexed as multisets. The result is a dense id in
`0..canonical_size(street)`: 169 / 1,286,792 / 13,960,050 / 123,156,254. It
is equal for every suit permutation and every reordering within a round.
`canonical_unindex` returns the representative: suits dealt c, d, h, s in
canonical order, each round sorted ascending.

* `canonical_index_batch(street, cards)` sends `[N, 2 + board]` tensors or
  arrays through the Rust batch function (~5.7M rows/s) and returns an int64
  tensor on the input's device.
* `unindex_batch` returns representatives through `canonical_unindex`
  (~1.1M/s). `enumerate_canonical(street)` builds every representative once
  and caches it as `data/abstraction/canonical_<street>.npy`: 6.4 MB for
  the flop, 84 MB for the turn, 862 MB for the river (about 2 minutes of
  CPU). `representatives(street, indices)` gathers from that cache when it
  exists.
* `orbit_sizes(street, cards)` gives the number of raw `(hole set, board set)`
  hands a class stands for: 24 divided by the number of suit permutations
  that fix the hand. Over the flop these sum to exactly
  `C(52,2) * C(50,3) = 25,989,600` (tested). The k-means weights classes by
  it, so buckets reflect how often hands actually occur.

## Bucket tables (`buckets.py`)

### Features (per canonical class, on its representative)

| Street | Feature |
|---|---|
| river | exact equity (win + tie/2) against all 990 opponent hands (`pokerbot.env.equity.equity_river`) |
| turn | `bins`-bin histogram (default 10 equal-width bins on [0, 1]) of the exact river equity over **all 46 river cards** (`runouts: 0`; a positive value samples that many) |
| flop | the same histogram over `runouts` **sampled** (turn, river) runouts, exact river equity each (`pokerbot.env.equity.equity_histogram`); `runouts: 0` enumerates all 1,081 |

Each class also gets its mean equity (used to order buckets and for
statistics) and its orbit size (the weight).

### Clustering

* **River:** weighted 1-D k-means on equity. Lloyd's algorithm starts from
  the weighted `(j + 0.5)/k` quantiles. Centres are sorted and every class
  goes to the nearest centre, so bucket ids are monotone in equity (tested).
* **Flop, turn:** weighted k-means under the earth mover's distance. For 1-D
  histograms on equally spaced bins, EMD is exactly the L1 distance between
  the cumulative sums. Points are therefore mapped to CDF space (`bins - 1`
  coordinates), and distances are `torch.cdist(p=1)`, batched over
  `fit_chunk` rows. Seeding is k-means++ (weighted D² sampling under EMD).
  Then come `iters` Lloyd iterations, stopping early when no label changes.
  The centre update is the weighted mean histogram, the usual choice for EMD
  k-means; the exact L1 minimiser would be the coordinate-wise median. An
  empty cluster is re-seeded at the point farthest from its centre. Buckets
  are numbered by the centres' mean equity.
* **Sampled fitting** (`sample: N`, or `--sample N`): centres are fitted on
  N random classes, then every class is assigned in chunks. Features are
  always computed once for all classes and stored, so sampling saves k-means
  time only.

### Pipeline, memory and outputs

For each street, features are computed `chunk` classes at a time on the
device. Equity kernels evaluate at most `max_rows` 7-card hands per call.
Results go to host arrays, memory-mapped under `<out_dir>/features/` when
`feature_cache: true`; a re-run with the same feature settings reuses them,
for example to try another bucket count. Next come the fit (all rows or a
sample) and the chunked assignment. Peak host memory for the river is about
3 GB, and VRAM is bounded by `max_rows` and `fit_chunk`.

Per street, `out_dir` receives:

* `<street>.npy`: a 1-D `uint16` table (`uint32` above 65,535 buckets) of
  length `canonical_size(street)`. Entry `i` is the bucket of every hand with
  `canonical_index(street, hole, board) == i`. This is the format that
  `poker_engine.CardAbstraction(tables=[None, flop, turn, river])` and the
  MCCFR config's `cards.tables` load (`engine/src/abstraction.rs`,
  `engine/src/npy.rs`: any 1-D little-endian integer `.npy`, exact length,
  values below the street's bucket count). Preflop needs no table: the
  engine uses the 169 lossless classes when `buckets[0] == 169`.
* `<street>.json`, the sidecar. It records the full config, the git hash,
  the torch version, feature statistics (count, raw hands, weighted equity
  mean/std/min/max, mean histogram, time, indices/s) and the fit (rows,
  objective per iteration, time). It also has the assignment objective, raw
  hands per bucket, each bucket's mean equity, and the centres.

`limit: N` (`--limit N`) computes only N seeded random classes per street,
for smoke runs and tests. The engine requires full-length tables, so every
other entry is padded with `index % buckets`. That filler only keeps the
plumbing working and is not an abstraction; the sidecar's `padded` field
counts the padded entries.

### Running it

```bash
# smoke run on the CPU (~10 s): 256 classes per street, 8 buckets
python scripts/build_buckets.py --config configs/buckets_tiny.yaml

# the real tables on the RTX 4070 Ti
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --device cuda
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --streets river   # one street
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --limit 20000     # measure idx/s first
```

To train the blueprint on the tables, set these in `configs/mccfr_hunl.yaml`
(the script prints the lines at the end):

```yaml
  cards:
    buckets: [169, 1000, 1000, 1000]
    tables: {preflop: null,
             flop: data/abstraction/buckets_hunl/flop.npy,
             turn: data/abstraction/buckets_hunl/turn.npy,
             river: data/abstraction/buckets_hunl/river.npy}
```

`pokerbot/blueprint/mccfr/train.py` already passes `cards.tables` (paths
relative to the repo root) to `poker_engine.Trainer`, and the Rust side
validates their length and range when it loads them.
`tests/abstraction/test_mccfr_glue.py` trains the small MCCFR config for a
few seconds on tiny generated tables. It checks that every infoset's bucket
is in range and that the solver's bucket equals `table[canonical_index]`.
Checkpoints and strategy files store the tables' absolute paths and load
them again. `BlueprintAgent` therefore needs the same files at the same
place at play time, so do not move or rebuild them under a trained
blueprint.

### Throughput and expected 4070 Ti times

Measured on the 4-core development container (torch 2.14 CPU, 4 threads,
another job sharing the machine), the feature kernels run at 6-7M 7-card
evaluations/s:

| Street | Work per class | Measured CPU rate | Classes | CPU time |
|---|---|---|---|---|
| river | 991 evaluations | 6,900 classes/s | 123,156,254 | 5.0 h |
| turn (46 exact rivers) | 45,600 | 123 classes/s | 13,960,050 | 31.4 h |
| flop (128 sampled runouts) | 127,000 | 52 classes/s | 1,286,792 | 6.9 h |

Supporting steps on the same CPU: `canonical_unindex` at 1.1M/s (2 minutes
for the river cache, once), `orbit_sizes` at 0.3M/s, EMD distances at 680M
point-centre pairs/s. k-means++ plus three Lloyd iterations on 100k points
with 1,000 centres takes 1.4 s.

**Assumption.** The equity kernel is eager-mode torch: about 50 elementwise
and gather kernels per evaluation over int64 tensors. It is bound by memory
bandwidth, not arithmetic. The 4070 Ti has 504 GB/s against roughly 15 GB/s
effective here, so we take a **35x** speedup, with a plausible range of 20x
to 60x. That gives about 230M evaluations/s on the GPU.

| Street | 4070 Ti at 35x | range (60x ... 20x) |
|---|---|---|
| river | ~9 min | 5-15 min |
| turn | ~54 min | 31-94 min |
| flop | ~12 min | 7-21 min |
| caches, orbit sizes, k-means | ~5 min | 3-10 min |
| **total, 1000/1000/1000 buckets** | **~1.3 h** | **~45 min - 2.3 h** |

The k-means share is small even on the CPU: fitting on 2M turn classes with
1,000 centres is about 3 s per Lloyd iteration here, and assigning all 14M is
about 20 s. A smoke run with `--limit 20000 --device cuda` prints the real
idx/s per street, which is worth checking before the full build.

Two cheaper options, not implemented:

* Compute the river equity table first and look the turn histograms up from
  it (46 `canonical_index` lookups per turn class instead of 46 x 990
  evaluations). This would remove about 70% of the run.
* `torch.compile` the evaluator.

The exact flop (`runouts: 0`, 1,081 runouts) costs 8.4x the configured
128-runout flop, about 1.7 h on the GPU.

## Tests

`tests/abstraction/` covers:

* action-spec equivalence between Python and Rust (legal sets and amounts,
  three specs, 1,200 random states);
* the rounding epsilon;
* the conversions;
* pseudo-harmonic values: 0.75 pot between 0.5 and 1.0 maps down with
  probability 0.375/0.875, checked deterministically and by frequency;
* agreement between the Python mirror, Rust `translate` and reference-engine
  states;
* isomorphism invariance, unindex round trips, the cached enumeration and
  orbit sizes (brute force and the flop total);
* exact features, checked against `poker_engine.hand_strength`;
* EMD as the CDF L1 distance;
* EMD k-means recovering separated synthetic clusters;
* 1-D k-means;
* the tiny config producing valid tables that load into `CardAbstraction`;
* river buckets monotone in equity;
* sampled fitting with feature-cache reuse;
* the CLI;
* the MCCFR glue.

The suite takes about 25 s.
