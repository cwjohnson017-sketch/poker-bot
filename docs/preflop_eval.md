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
