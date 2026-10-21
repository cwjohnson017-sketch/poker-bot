# Preflop: how exploitable is the blueprint's preflop? (2026-10-19)

Production plays preflop from the blueprint (`dcfr4_distilled_v2`) and searches
every later street. Preflop was the only part of the bot never measured. Tool:
`scripts/eval_preflop.py` (`pokerbot/search/preflop_eval.py`); run
`runs/preflop/pf100.cmd`, results `runs/preflop/pf100/preflop_eval.{md,json}`
(gitignored).

## Method

* **The preflop tree.** The blueprint's preflop abstraction (button = seat 0)
  has 54 decisions and 31 lines that reach a flop. Under the blueprint, 48% of
  deals end in a preflop fold, 0.2% in a preflop all-in, and 52% see a flop. The
  biggest single line is limp, check: 25% of all deals.
* **Leaf values.** At each of the 31 flop starts and each of 100 random flops
  (the same flops for every start), the flop-root subgame is solved with
  production's flop search settings (`configs/search_value_net.yaml`: 20k-node
  trees, `turn_v5x` leaves), from the blueprint's ranges, for 100 DCFR
  iterations. That is 3,100 solves, 4.4 h on the 4070 Ti.
  * Each solve's root counterfactual values (both seats, every hand) are kept.
  * Each flop also stands for its 24 suit permutations. That needs a
    suit-symmetric preflop, so the blueprint's preflop is averaged over suits.
    It was about 1% asymmetric (reach-weighted).
  * The solves' own exploitability: 0.2-2% of the pot, and 4.9% in limped pots.
* **Best response.** For each seat: the exact best response to the blueprint's
  preflop, given those leaf values.
  * **In-sample:** chosen and scored on all 100 flops. Noise in the leaf values
    lets a maximum look better than it is, so this is biased up.
  * **Cross-fitted:** chosen on one half of the flops and scored on the other.
    This is biased down.
* **What it measures.** The deviator gains from preflop changes only. The
  postflop play stays the solution for the blueprint's ranges.
  * That is also what production does. The search reconstructs both ranges
    from the blueprint's preflop (`range_reach`), whatever the opponent
    actually did preflop.
  * So, up to the search's postflop error, this is how much an opponent could
    win from the bot by changing its preflop play alone.

## Result

| mbb/hand | button | big blind | exploitability (mean) |
|---|---:|---:|---:|
| blueprint value | +217.0 | -216.7 | (sum +0.3) |
| best response, in-sample | +580.1 | +180.7 | **380** |
| best response, cross-fitted | +537.7 | +138.1 | **338** |

* **Preflop alone is 340-380 mbb/hand exploitable.** The two estimates agree
  within 43, so flop-sampling noise is not what drives it.
* **For scale.** These are measured differently, so they are not directly
  comparable, but they are the same order. Round 4's postflop search has
  two-sided exploitability 148 on 18 turn spots and 91 on 12 flop spots.
* **It is the largest leak measured so far.**
* **The zero-sum check** (+0.3 mbb/hand) confirms the leaf values add up: every
  flop solve is exactly zero-sum, and the sampling evens out with 100 flops.

### Where the gains are

The gain of the best single-decision deviation, with the blueprint everywhere
else, in mbb/hand:

| seat | line | gain | blueprint | best action |
|---|---|---:|---|---|
| button | open (root) | 230 | limp 37%, raise 2x 25%, 2.25x 17%, 2.5x 13%, 3x 5%, 4x 3% | never limp; raise 2x 15%, 2.25x 27%, 2.5x 17%, 3x 18%, 4x 22% |
| BB | vs 2x open | 99 | fold 51%, call 25%, 3-bet 21% | fold 0%, call 74%, 3-bet 25% |
| BB | vs limp | 88 | check 69%, raise 31% | check 21%, raise 79% (mostly 3x-4x) |
| BB | vs 2.25x open | 70 | fold 56% | fold 0%, call 80% |
| BB | vs 2.5x open | 54 | fold 58% | fold 1%, call 76% |
| button | vs 3-bet after 2x open | 23 | fold 44%, call 43%, 4-bet 14% | fold 35%, call 25%, 4-bet 39% |
| BB | vs 3x / 4x open | 22 / 17 | fold 63% / 70% | fold 8% / 11% |

* **The big blind overfolds.** It folds 51-70% to opens of 2x-4x, and the best
  response folds almost nothing. With 3:1 against a 2x open (and 1.7:1 against
  4x), and the search playing postflop, nearly every hand defends profitably.
* **The button limps too much.** It limps 37% of hands and the big blind checks
  69% of limps behind. Against this big blind, open-raising every hand beats
  limping.
* **The big blind under-raises limps.**
* These are one-step gains. They overlap and do not add up to the best
  response.

### Caveats

* **Postflop play is the 100-iteration flop solve, not production's 4 s
  search.** It is similar: production runs about 68 iterations on the same
  trees, then searches the turn and river again.
* **Out-of-range hands.** Values for hands the blueprint's range never holds
  (e.g. the big blind's folding hands after it calls) come from that solve's
  strategy for them, not from a best response. If anything this understates
  the deviations.
* **100 flops x 24 permutations.** The cross-fit bound accounts for the noise
  this leaves.

## What it means for the plan

Stage 2 of `docs/plan_round4.md` (a preflop solve consistent with the postflop
the bot plays) now has a measured target of about 340 mbb/hand. That is more
than any postflop leak left.

The preflop tree is small (31 flop starts), so a preflop solve need not wait
for a flop-root net:

1. **Iterate this pipeline.**
   * Hold the leaf values fixed and solve the preflop tree with CFR.
   * Re-solve the 31 x 100 flop roots for the new ranges (4.4 h locally, or
     well under an hour on a few spot GPUs).
   * Repeat until the preflop strategy settles.
   * The result goes into production as a tabular preflop policy. The search
     takes its ranges from it.
2. **A flop-root net** (DeepStack's) remains the general route. It also allows
   preflop re-solving at decision time, but it adds a lossy step: stage 1's
   turn-start net was about 100 mbb/hand behind depth-1 on the flop.

## Attempt 1: iterated linearised preflop solve (2026-10-19/20)

`scripts/solve_preflop.py` (`pokerbot/search/preflop_solve.py`); run
`runs/preflop/solve1.cmd`, results `runs/preflop/solve1/` (gitignored). 14.3 h
locally.

### Method

Each round:

1. **Flop solves.** At the 31 flop starts and 40 new flops (seeds 100-104),
   the flop roots are solved for the current preflop's ranges, as in the
   measurement.
2. **Linearisation.** With the postflop strategy fixed, every solve is turned
   into class values: the value of each hand against each of the 169 preflop
   hand classes.
   * The turn-end leaves are frozen as a zero-sum payoff,
     `U(h, o) = ev_0(h) - ev_1(o) + kappa`.
   * The flop average is a plain sample mean.
   * Both are needed for an exactly zero-sum game. Freezing each hand's value
     against the range was not zero-sum, and CFR stalled.
3. **Evaluation.** The current preflop is scored on these flops. From round 1
   on, they are new flops, so the score is out of sample.
4. **Preflop CFR.** 2,000 iterations on the linearised game; the solutions
   reach 0.01-0.08 mbb/hand.
5. **Next preflop.** Half the current strategy, half the CFR solution.

### Result

| round | preflop | honest best response, cross-fit / in-sample | linearised, same point | next preflop, linearised (before re-solving) |
|---:|---|---|---:|---:|
| 0 | blueprint | 263 / 411 | 411 | 262 |
| 1 | mix of 1 solve | 342 / 430 | 421 | 223 |
| 2 | mix of 2 | 254 / 409 | 403 | 173 |
| 3 | mix of 3 | 387 / 441 | 432 | 176 |
| 4 | mix of 4 | 410 / 445 | 439 | 187 |

mbb/hand, 40 flops per round. (The blueprint with 100 flops: 338 / 380.)

* **No improvement.** The re-solved preflops are as exploitable as the
  blueprint or more, within this run's noise.
* **The linearisation is accurate where it is taken.** Its exploitability of
  the current preflop matches the honest in-sample best response within 1-9
  mbb/hand every round.
* **The steps are too long.** The next preflop is 173-262 in the round that
  made it. Once the flops are re-solved for its ranges, it is 409-445. Moving
  the ranges moves the postflop solution's values by 200-270 mbb/hand, which
  undoes the step.
* **Oscillation.** For example, the big blind's best reply to a 2.5x open swings
  to "3-bet everything" in rounds 3-4.
  * The solutions fold 9-13% of the button's hands at the root, which the
    blueprint never does (plausibly closer to equilibrium).
  * But each half-step overshoots against the next round's postflop.
* **Production keeps the blueprint's preflop.**

### Next options

1. **Shorter steps, more rounds** (fictitious-play style).
   * For example, mix 0.1-0.2 over 15-20 rounds.
   * Cheaper rounds: 20 flops at 60 iterations is about 1 h per round.
   * The linearisation's local accuracy suggests this converges, but slowly.
2. **Leaves that follow the ranges within a CFR run:** a flop-root value net
   (DeepStack). This is the plan's original stage 2. It costs a large data run
   and adds the fit error stage 1 hit.

## Attempt 2: short steps (2026-10-20/21)

Same method, with steps of 0.15 instead of 0.5.
* **Run:** 16 rounds x 20 flops (seeds 200-215), 60 flop iterations;
  `runs/preflop/solve2.cmd`, results `runs/preflop/solve2/`. 18.3 h locally.
* **Crash:** the run ran out of GPU memory once, in round 4. The leaf nets'
  board-feature caches now clear between flop starts, and the launcher
  retries.

| rounds | honest best response, cross-fit / in-sample (mbb/hand, 20 flops each) |
|---|---|
| 0-2 (blueprint, then 15% steps) | 259-287 / 368-395 |
| 3-8 | 22-144 / 255-310 |
| 9-11 | 188-258 / 309-364 |
| 12-15 | 156-237 / 251-299 |

* **Exploitability falls by about a quarter, then levels off.**
  * The linearisation keeps matching the honest in-sample best response
    within 1-17 mbb/hand.
  * Each step's predicted gain (16-56 in the linearised game) mostly
    holds.
  * The plateau of about 250-300 in-sample is noisy at 20 flops a round.

**Paired check on 100 flops.** Round 15's preflop
(`runs/preflop/solve2/round_14/table.pt`, copied to
`runs/preflop/preflop_r15.pt`) was measured on the same 100 flops and settings
as the blueprint above (`runs/preflop/pf100_r15/`). The solve never saw these
flops.

| mbb/hand | button BR | big blind BR | exploitability |
|---|---:|---:|---:|
| blueprint, in-sample / cross-fit | +580 / +538 | +181 / +138 | 380 / 338 |
| **round 15, in-sample / cross-fit** | +396 / +375 | +108 / +69 | **252 / 222** |

**A third less exploitable:** -128 in-sample and -116 cross-fitted.

**What changed:**

| | blueprint | round 15 |
|---|---:|---:|
| button limps | 37% | 10% |
| button folds | 0% | 15% |
| button opens 2x | 25% | 36% |
| big blind folds to 2x / 2.5x / 4x | 51 / 58 / 70% | 29 / 40 / 58% |
| big blind checks behind a limp | 69% | 62% |

**What is still exploitable** (largest one-step gains):
* the button's open (104 mbb/hand): it would rather open 3x-4x than 2x;
* the button's response to 3-bets (35 after a 2x open): it folds 46% and
  4-bets 11%, where the best response 4-bets about 70%.

**Using it.** `search:preflop:runs/preflop/preflop_r15.pt` in place of
`search:neural:<blueprint>`:
* the search plays from the table on the preflop tree;
* it builds the postflop ranges from it;
* it falls back to the blueprint for off-tree preflop histories.

**Smoke match** (`runs/preflop/match_smoke_pf.json`, 1.1 h): the Round 4
production search with this preflop against the blueprint, on the 500
duplicate deals of Round 4's smoke match (seeds 3000-3009).

* **It plays cleanly:** 0 fallbacks on every street; flop 4.11 s and 78
  iterations, turn 2.01 s and 365, river at the 1,000-iteration cap.
* **Win rate:** +511 mbb/hand luck-adjusted, CI [-43, +1089]. Raw +686.
* **Paired against Round 4 on the same deals:** -237 luck-adjusted,
  CI [-857, +383], so no difference measurable at 1,000 hands.
* A less exploitable preflop need not win more against this blueprint. For
  instance, it no longer limps into a big blind that checks behind 69% of the
  time.

**Adopted (2026-10-21).** `configs/match_search_vn.yaml` now plays
`search:preflop:runs/preflop/preflop_r15.pt`. The table is gitignored, like the
value nets; copies are in both checkouts' `runs/preflop/`.
