# Preflop off-tree audit and soak harness (roadmap 1.5, 2026-10-21)

Two robustness tools for the production bot (`search:preflop:runs/preflop/preflop_r15.pt`
with `configs/search_value_net.yaml`):

* `scripts/audit_preflop.py` (`pokerbot/eval/preflop_audit.py`): the bot's preflop
  answer to every size, on its tree or not.
* `scripts/soak.py` (`pokerbot/eval/soak.py`): many hands against a mix of opponents,
  for crashes, illegal actions, fallbacks, decision times and memory growth.

## The preflop audit

### Method

* **Spots.** 108 preflop decisions, button = seat 0, 100bb:
  * the big blind against opens of 2, 2.25, 2.5, 2.75, 3, 3.5, 4, 5, 6, 8 and 10bb
    and all-in;
  * the button against 3-bets of a min-raise, 2.5x, 3x, 3.5x, 4x and 5x the open and
    all-in, after each of its five on-tree opens (2, 2.25, 2.5, 3, 4bb);
  * the big blind against 4-bets of a min-raise, 2x, 2.5x, 3x and 4x its 3-bet and
    all-in, after its on-tree 3-bet (3x) of each open;
  * the button against raises of 2-10bb and all-in after it limped;
  * the big blind against re-raises (min, 2.5x-5x, all-in) after it raised a limp to
    2, 3 or 4bb.
* **Frequencies.** At each spot the agent makes one decision with `record=True`. Its
  strategy for all 1326 hands is weighted by its own reach there: fold, call and
  raise frequencies over the range it actually holds.
* **No search runs.** Preflop is played from the table on the 31-start tree and from
  the base blueprint (`dcfr4_distilled_v2`) everywhere else, so the audit runs on the
  CPU in 2 seconds.
* **Reference: minimum defence.** A raise that adds `bet` chips to a pot of `pot`
  (before the raise) profits with any two cards if the bot folds more than
  `bet / (pot + bet)`: against a 2bb open, 50%; against all-in, 99%. It ignores the
  raiser's equity when called, so it is a loose bar: above it is a clear leak, below
  it need not be right.
* **Flags.** `overfold` (above the bar); `non_monotone` (folding 10+ points less
  against a bigger raise than against the next smaller one); `jump` (a change of 25+
  points between sizes at most 1.5x apart).

Run (CPU, from the main checkout):

```
python scripts/audit_preflop.py --config configs/match_search_vn.yaml --device cpu --out-dir runs/preflop_audit
```

### Result

18 of 108 spots flagged (git 6e5e63f). The command above writes all five tables to
`runs/preflop_audit/preflop_audit.{md,json}` in 2 s; the excerpts below are from them.
`source` says who plays the spot: the table or the base blueprint.

**Big blind facing an open:**

| open | source | fold | max fold | call | raise |
|---:|---|---:|---:|---:|---:|
| 2bb | table | 29% | 50% | 54% | 17% |
| 2.25bb | table | 35% | 54% | 49% | 16% |
| 2.5bb | table | 40% | 57% | 45% | 15% |
| **2.75bb** | blueprint | **63%** | 60% | 18% | 19% |
| 3bb | table | 47% | 62% | 39% | 13% |
| **3.5bb** | blueprint | **70%** | 67% | 13% | 17% |
| 4bb | table | 58% | 70% | 30% | 12% |
| 5bb | blueprint | 70% | 75% | 13% | 16% |
| 6bb | blueprint | 71% | 79% | 13% | 16% |
| 8bb | blueprint | 71% | 83% | 13% | 16% |
| 10bb | blueprint | 64% | 86% | 18% | 18% |
| all-in | table | 92% | 99% | 8% | 0% |

**Button facing a 3-bet after a 2bb open** (the other opens look the same):

| 3-bet to | source | fold | max fold | call | raise |
|---:|---|---:|---:|---:|---:|
| 3bb (min) | blueprint | 25% | 40% | 54% | 21% |
| 5bb | blueprint | 26% | 57% | 53% | 21% |
| 6bb (3x) | table | 46% | 62% | 41% | 13% |
| 7bb | blueprint | 27% | 67% | 52% | 21% |
| 10bb | blueprint | 28% | 75% | 51% | 21% |
| all-in | table | 92% | 97% | 8% | 0% |

### Findings

1. **Every open size off the table goes back to the old big blind.** Between and
   above the table's sizes the base blueprint plays, and it folds 63-71% (the
   measured blueprint leak, `docs/preflop_eval.md`), where the table folds 40-58%.
   * A 2.75bb open is folded to 63% of the time, against 40% at 2.5bb and 47% at 3bb.
     It and 3.5bb (70%) are above the minimum-defence bar: opening them with any two
     cards profits at once.
   * So the solved table's main gain (the big blind's defence) is lost to anyone
     who opens 2.75bb or 3.5bb. The table plays only exact on-tree histories
     (`PreflopTable`); nothing maps an off-tree size onto it.
   * The obvious fix: play off-tree preflop decisions from the table by translation,
     mixing the neighbouring on-tree histories' rows by the pseudo-harmonic weights
     (the audit's `translation` column already gives them, e.g. 2.75bb = 58% 2.5bb,
     42% 3bb), and build postflop ranges the same way.
2. **Against off-tree 3-bets the button folds about half as often** (25-44% against
   46-51% at the table's 3x), and folds less to a 5x 3-bet than to 3x. Not a
   minimum-defence leak, but a value 3-bet of 3.5x or more gets called wider than one
   of 3x. Same cause.
3. **The bot can never 5-bet.** The blueprint's abstraction caps preflop at 3 raises
   (`spec.max_raises`), so after open, 3-bet and 4-bet it can only fold or call: raise
   0% at every 4-bet spot, on the table or not. An opponent can 4-bet small (a
   min-raise to 10bb after a 6bb 3-bet) and is never shoved on. Below the 3x size every
   4-bet also translates to the one 4-bet size on the tree, so the postflop ranges
   are built as if it were 3x. The bot does not overfold there (39-52%, under the bar),
   so this is a missing option, not a measured loss.
4. **A min re-raise of the bot's limp-raise overfolds:** after limp, raise to 2bb, a
   re-raise to 3bb is folded 55% of the time against a bar of 40% (+15, the largest
   overfold found). Rare: the bot is in this spot on 4% of deals. Larger re-raises
   are folded 52-61%, under the bar.
5. **After its limp, the button stops re-raising at off-tree sizes:** 9-13% re-raises
   at 2-4bb on the table, 2% at 3.5, 5, 6 and 8bb, then 16% at 10bb.
6. **All-ins are handled well:** 82-92% folds against shoves, under the 95-99% bar,
   everywhere.

Translation stress in the soak and the archetypes (1.3) should include 2.75bb and
3.5bb opens, small 4-bets and min re-raises of limp-raises.

## The soak harness

`scripts/soak.py` plays `--hands` hands of the bot (seat 0, button alternating)
against `--opponents`, rotating every two hands; any agent specs (`random`, `equity`,
`always_call`, archetypes once merged). Each hand's deck and RNG come from
`(seed, hand)`, so hands replay on their own.

* Hands go in chunks (`--chunk`, default 100), each written to `chunk_NNNNN.json`
  when done; a rerun with the same `--out-dir` resumes, losing at most one chunk.
* Exceptions in a hand are caught and recorded (bot or opponent, with the traceback).
  Illegal actions are replaced by fold or check and counted (`play_hand`'s new
  `illegal` list).
* Searched decisions and fallbacks come from `SearchAgent.stats`, which the harness
  empties after every chunk (it grows by one entry per search, also in the play app).
* Decision times are the wall time of each `act`, by street.
* Memory: process RSS and CUDA allocated / reserved / peak at the end of every chunk;
  the trend is a fitted slope in MB per 10k hands, over the chunks after the first.
* `soak.json` and `soak.md` hold the totals and the Gate 1 check: 0 crashes, 0
  illegal actions by the bot, fallbacks under 0.1% of postflop decisions, p99
  decision time under 8 s.

### CPU smoke run

300 hands (6 chunks of 50) of the production preflop table and blueprint on the CPU,
with a stand-in search: blueprint rollout leaves took up to 82 s per flop decision on the CPU, so the
smoke config solves to showdown on small trees (1,500 nodes, 2 chance cards, 0.2-0.3 s
budgets). It tests the harness and the bot's plumbing, not production's timing. The CPU
was shared with the test suite and other sessions, so the run took 3.4 h.

| check | value | bar |
|---|---:|---:|
| crashes | 0 | 0 |
| illegal actions | 0 | 0 |
| fallbacks / postflop decisions | 0 of 362 | < 0.1% |
| p99 decision time | 114 s | < 8 s (not meaningful here) |

* **Decision times:** flop p50 18 s, p99 131 s; turn 21 / 98 s; river 6 / 66 s; preflop
  under 4 ms. The time budget does not cover building a tree, and these
  showdown-depth trees take most of that time on the CPU. Production's value-net trees
  measured 4.1 s on the flop (`docs/preflop_eval.md`); the GPU run checks the bar.
* **Memory:** RSS 399 MB after the first chunk, then 502, 492, 487, 495 and 578 MB.
  The fitted slope (+3,100 MB per 10k hands) is dominated by the last chunk; 300 hands
  cannot tell growth from noise. The blueprint's per-prefix reach cache is bounded (64 MB).
* Bot win rates (raw, 100 hands each, CIs of +/-8,000-14,000 mbb/hand): +5,756 against
  `random`, -302 against `equity`, +8,592 against `always_call`. Not a measurement.

```
python scripts/soak.py --config configs/match_search_vn.yaml --device cpu \
    --bot search:preflop:runs/preflop/preflop_r15.pt --search-config <tiny cpu yaml> \
    --hands 300 --chunk 50 --out-dir runs/soak/cpu300
```

with the tiny config's `search:` section: `time_budget {flop: 0.3, turn: 0.2, river: 0.2}`,
`min_iterations: 2`, `tree {actions: blueprint, depth_streets: 3, max_nodes: 1500,
chance_cards: 2, min_chance_cards: 2, max_raises: 2}`, `solver {iterations: 50,
max_runouts: 6}`, `gadget {safe: true, terminate: unsafe}`.

### The full run (not run yet)

100,000 hands against production on the 4070 Ti. The 1,000-hand smoke match with
the solved preflop took 1.1 h, about 4 s a hand (flop 4.1 s, turn 2.0 s, river under
1 s per decision). At that rate 100,000 hands take about 4.5 days; `always_call` takes
more hands to the river and `random` folds many early, so the mix may differ by a day
either way. Two 50k-hand halves with different `--seed` and `--out-dir` on two GPUs
halve it (the `cloud/aws` tooling, about $20-50 on spot GPUs, roadmap section 11):

```
python scripts/soak.py --config configs/match_search_vn.yaml --hands 100000 \
    --opponents random equity always_call --out-dir runs/soak/prod_100k
```

Stop it at any time and rerun the same command to resume. With `nvidia-smi` showing
other GPU jobs, timings (the p99 bar) are not representative.
