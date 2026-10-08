# Value-net leaves for flop search: evaluation (2026-10-05)

Branch `claude/value-net` (worktree `poker-bot-search`). Design: `docs/value_net.md`.
Raw results: `runs/vn_eval/*.md|json` and `runs/value_net/*.json` (gitignored).

## Summary

**Round 2 (2026-10-07; details in "Round 2" below):**

* **Correction.** Every blueprint number in this report before Round 2 is too
  high. The old evaluation queried the blueprint's turn play on the wrong turn
  card. Corrected on the six spots: one-sided 1,190 (not 2,398) and two-sided
  1,051 (not 1,981) mbb/hand. Value-net search is still about 5x (one-sided)
  to 10x (two-sided) less exploitable. The margins quoted below are
  overstated by about 2x.
* **Production search changes:**
  * no safe-resolving gadget at the first decision of a street. The rollout
    terminate values were biased: the searcher's one-sided exploitability on 6
    spots goes from 580 to 221 mbb/hand, and the per-hand excess over the
    blueprint from 139 to 5.5;
  * flop trees of 20,000 nodes, the blueprint's whole flop abstraction. They
    are less exploitable than 6,000 or 10,000-node trees under the same 4 s,
    despite 65 instead of 186 iterations: one-sided 147 against 209, two-sided
    186 against 677;
  * the round-2 turn-end net `turn_v3w`: 5% lower leaf error at production
    states, 2% lower exploitability.
* **Measured and not adopted:** a blocker-aware head on the turn-end net, and
  terminate values computed in the search tree (better than rollouts, but
  2.6 s of the budget).
* **Head to head, 10,000 hands** against the blueprint with the new
  production config: **+199 mbb/hand luck-adjusted, 95% CI [+44, +364]**
  (raw +162, CI [-13, +340]). There were no blueprint fallbacks in 12,825
  search decisions, and flop decisions took 4.06 s on average.

**Round 1 (2026-10-05/06):**

* **Search with value-net leaves is far less exploitable than the blueprint,
  and than rollout search, on every spot measured.** The reference game solves
  every river exactly, so no leaf model is involved in scoring.
  * On the 6 original flop spots with safe resolving off (a clean two-sided
    comparison), the means are: value-net search **91** mbb/hand, rollout
    search 1,059, blueprint 1,981. Value-net search is better than the
    blueprint in every spot (14–26x).
  * With safe resolving on (the production default), the fair measure is the
    opponent's best response to the strategy the searcher plays. On 12 spots
    (4 boards x 3 spot types), value-net search is **1,751 mbb/hand** better
    than the blueprint on average and better in all 12, by at least 349.
    Rollout search is 795 better on average but worse than the blueprint in
    2 of the 4 "BB first" spots.
  * The two-sided number also scores the opponent-side strategy, which the
    gadget deliberately leaves unrefined. On that number, value-net search is
    better than the blueprint on average (12 spots: 1,416 vs 1,861) but worse
    in the "BB first" spots (3,238 vs 1,878). The unsafe run shows this gap is
    entirely the gadget's.
* **Speed:** a flop decision with the turn-end net and a 4 s budget takes
  **4.0 s** (about 154 DCFR iterations) and keeps almost all the gain:
  1,732 vs 1,751 mbb/hand better than the blueprint on 12 spots. The river
  net averaged over 48 river cards is the most accurate leaf model, but costs
  48 ms per iteration against 18 ms. With the production tree (5,802 nodes,
  637 leaves), a flop decision is 4.03 s with 150 iterations.
* **Held-out error** in pot units, per combo:

  | Net | Overall MAE | Blueprint-like ranges | Zero baseline | Bucketing floor |
  |---|---|---|---|---|
  | River | 0.031 | 0.022 | 0.41 | 0.010 |
  | Turn-end (vs its bootstrapped targets) | 0.033 | — | 0.30 | — |

  At the turn-end leaves search actually uses, checked against exact 48-river
  solves: the river net's river-card average has MAE **0.015**; the turn-end
  net has 0.036. Both nets are limited by data, not capacity.
* **Caveat:** the 6,000-node trunk used for the comparison (the original
  experiment's) is coarse:
  * the flop keeps check, pot, pot re-raise and all-in;
  * the turn keeps check, pot re-raise and all-in, so no opening bet below all-in.

  The blueprint's other sizes are renormalised onto these, which costs it about
  2.5 off-tree decisions per hand and inflates its score. The searches solve
  exactly this trunk.

  A check on a 20,000-node trunk keeps the blueprint's flop sizes from 0.5 to
  2 pot (1.4 off-tree decisions per hand). There the blueprint scores worse,
  because the best responder gets those sizes too, and value-net search's
  lead grows to -3,605 mbb/hand on 2 spots. So the coarse trunk does not
  explain the result.

* **Head to head**, 2,000 hands (duplicate, 1,000 deals): value-net search
  (production config) against its blueprint scores +168 mbb/hand raw
  (95% CI -264 to +580) and +131 luck-adjusted (CI -227 to +492). Positive
  but not significant at this sample size. All 2,508 search decisions
  searched, with no blueprint fallback; flop decisions took 4.03 s on
  average and 4.08 s at most.
* **Follow-up (2026-10-06):**
  * 4.2x the river data, including 73k on-policy leaf samples, and a retrained,
    wider turn-end net.
  * The production leaf model's error at search leaf states fell 22% against
    exact solves.
  * Exploitability on the six spots with safe resolving off fell from 115 to
    **101** mbb/hand (blueprint 1,981), at the same 4.0 s per flop decision.
  * See "Follow-up" below.

## What was built

| Module | Role |
|---|---|
| `search/batch_solver.py` | `BatchRiverSolver`: DCFR on B river subgames that share one betting tree (same pot), each with its own board and ranges. Matches `RangeSolver` to 1e-9. 46 ms/iteration at B=512 on the largest river tree (237 nodes). |
| `search/value_ranges.py`, `value_data.py`, `scripts/gen_value_data.py` | River-root states from blueprint self-play on `VecNLHE` (both ranges tracked over 1326 combos), plus perturbed and DeepStack-style random ranges. Batched by pot, solved exactly, written as shards. |
| `search/value_net.py`, `value_train.py`, `scripts/train_value_net.py` | River net: ranges bucketed into 256 board-relative strength-percentile buckets; residual MLP (4 x 1024); per-bucket values decoded to combos; DeepStack zero-sum layer. Outputs per-combo values per unit of opponent mass, in pot units. Huber loss, held-out report. |
| `search/value_leaf.py` | `ValueLeafEvaluator`: turn-end `VALUE` leaves valued by the exact chance average of the river net over the river cards, on both players' current reaches every update. `TurnEndLeafEvaluator`: one row per leaf. |
| `search/turn_net.py`, `turn_data.py`, `scripts/gen_turn_data.py`, `train_turn_net.py` | Turn-end auxiliary net (DeepStack style): targets bootstrapped from the river net's river-card average; 32 x 8 buckets (mean x spread of river strength). |
| `search/tree.py`, `solver.py`, `agent.py`, `config.py` | `VALUE` node kind; `leaf.mode: rollouts \| value_net`, `leaf.net`, `leaf.net_every`; `tree.leaf_budget_cost`; the solver passes both reaches to the leaf provider. The rollout path is unchanged and still the default. |
| `search/exact_eval.py` | `trunk_exploitability`: exact exploitability of a flop/turn strategy with every (leaf, river card) river solved by `BatchRiverSolver`. Checked against a full turn-to-showdown solve. |
| `search/spot_eval.py`, `scripts/eval_search_exploit.py` | The acceptance test; replaces `runs/search_noise/exploit.py`. |
| `scripts/check_turn_leaves.py` | Turn-end leaf values against exact 48-river solves. |
| `configs/search_value_net.yaml`, `configs/match_search_vn.yaml` | Production search config (turn-end net, 4 s flop budget) and the head-to-head match. |

Tests: `tests/search/test_batch_solver.py`, `test_value_net.py`,
`test_value_ranges.py`, `test_value_leaf.py`, `test_turn_net.py`,
`test_exact_eval.py`, `test_spot_eval.py`. All tests in `tests/search`
pass (`-m "not slow"`).

## Data and training

**River data** (`runs/value_net/river_a`):

* 204,800 river subgames. Sources: 50% blueprint self-play ranges, 25%
  perturbed, 25% random.
* Each solved exactly with 400 DCFR iterations. Mean solve exploitability is
  0.13% of the pot (p90 0.3%); samples above 1% were dropped (279).
* Generation ran at 27 samples/s (about 2 h). The net trains in 2 minutes
  (20k Adam steps).

**Held-out error** of `river_v1.pt` (6,256 samples, held out by board; pot
units per combo):

| | samples | MAE | RMSE | range-weighted MAE | game-value error (per player) | zero MAE | bucket-oracle MAE |
|---|---:|---:|---:|---:|---:|---:|---:|
| overall | 6256 | 0.0306 | 0.106 | 0.0238 | 0.0056 | 0.408 | 0.0104 |
| c in [100, 250) | 1050 | 0.0308 | 0.183 | 0.0189 | 0.0048 | 0.400 | 0.0092 |
| c in [250, 500) | 1313 | 0.0313 | 0.120 | 0.0217 | 0.0048 | 0.425 | 0.0105 |
| c in [500, 1000) | 1231 | 0.0300 | 0.080 | 0.0233 | 0.0048 | 0.422 | 0.0115 |
| c in [1000, 2000) | 1239 | 0.0298 | 0.062 | 0.0257 | 0.0060 | 0.410 | 0.0111 |
| c in [2000, 4000) | 779 | 0.0313 | 0.049 | 0.0287 | 0.0066 | 0.396 | 0.0109 |
| c in [4000, 10000] | 644 | 0.0305 | 0.047 | 0.0273 | 0.0080 | 0.367 | 0.0085 |
| blueprint ranges | 3169 | 0.0219 | 0.046 | 0.0202 | 0.0044 | 0.389 | 0.0083 |
| perturbed | 1506 | 0.0244 | 0.046 | 0.0221 | 0.0047 | 0.390 | 0.0109 |
| random (DeepStack) | 1581 | 0.0538 | 0.195 | 0.0327 | 0.0087 | 0.463 | 0.0143 |

* `c` is the chips each player has put in, so the pot is `2c`.
* The bucket oracle replaces each target by its strength bucket's mean: the
  floor for bucket-level outputs.
* Training loss is about 10x below held-out error, so the net is limited by
  data:
  * dropout 0.1 gave 0.0310 MAE;
  * weight decay gave 0.0308;
  * a per-combo residual head with blocker features gave 0.0290, but costs
    about 7x at inference.

**Turn-end net** (`turn_v1.pt`): 300,000 turn-end states, targets the river
net's exact river-card average (no solving; 277 samples/s). Held-out MAE
against those targets is 0.0327 pot (range-weighted 0.032, game value 0.008;
zero 0.30). Its error against exact river solves is in the next section.

**Turn-end leaves against exact targets** (`scripts/check_turn_leaves.py`,
`runs/value_net/turn_leaf_check.json`):

* 400 fresh turn-end states: 50% self-play, 25% perturbed, 25% random, with
  new seeds.
* For each, all 48 river subgames were solved exactly (400 DCFR iterations)
  and the best-response values chance-averaged.
* These are the values a perfect leaf model would give.

| leaf model | MAE | range-weighted MAE | game-value error | self-play MAE | random MAE |
|---|---:|---:|---:|---:|---:|
| river net, averaged over the river cards (search's `ValueLeafEvaluator`) | 0.0149 | 0.0124 | 0.0030 | 0.0114 | 0.0247 |
| turn-end net | 0.0355 | 0.0339 | 0.0085 | 0.0290 | 0.0543 |
| zero | 0.3015 | | | | |

* The river net's per-river errors (0.031) partly cancel over 44 river cards.
* The turn-end net adds its own fit error, about 2.4x larger. This matches
  its slightly higher exploitability.

## Evaluation method

The task's original experiment scored strategies in a reference game whose
leaves were blueprint rollouts. Here the reference game has no leaf model:

1. Run the search agent (`agent.act`, the real code path) on a flop spot and
   capture its solver. The value-net tree is built with
   `leaf_budget_cost = 1 + k`, so its trunk equals the rollout tree's. A check
   asserts identical decision nodes, child actions, leaves and root ranges.
2. Map every profile onto that trunk: value-net search, rollout search, and
   the blueprint (mass on sizes missing from the trunk renormalised).
3. Score each profile with `trunk_exploitability`:
   * For each (leaf, river card), solve the river subgame exactly with both
     players' reaches under that profile, each mixed with 5% uniform.
   * Compute best-response river values against that river play with the
     opponent's actual reach, and back them up through the trunk.
   * Flop all-ins are scored over all 1,176 run-outs; the searches themselves
     sample 48.

   This is the exact exploitability, in the flop-to-showdown game, of the
   trunk strategy followed by an equilibrium river.
4. Report two numbers:
   * the two-sided `(BR_0 + BR_1) / 2`, the original metric;
   * the opponent's best response to the strategy of the player to act, the
     one the agent plays. Its differences across profiles are exact
     differences in that player's exploitability, because the game value is
     the same for every profile.

Spots are the original six: 2 boards x {BB first, BTN vs check, BTN vs half-pot
lead}, after a 2.5x open and call. Trees are capped at 6,000 nodes, with 196
or 343 leaves (9.4k or 16.5k river subgames per profile). Searches run a fixed
300 iterations unless noted.

## Results

### Six original spots, production config (safe resolving)

`runs/vn_eval/exploit6.md`; river solves 300 iterations.

**Opponent's best response to the searcher's strategy** (mbb/hand; lower is
better; the last columns are the difference from the blueprint):

| spot | value-net (river) | value-net (turn-end) | turn-end, 4 s budget | rollout search | blueprint | VN - BP | turn 4 s - BP | rollout - BP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| board0 BB first | 358 | 392 | 407 | 986 | 1335 | -977 | -928 | -348 |
| board0 BTN vs check | 485 | 520 | 528 | 1847 | 4124 | -3639 | -3596 | -2277 |
| board0 BTN vs 1/2 lead | 632 | 649 | 646 | 1467 | 3530 | -2898 | -2884 | -2063 |
| board1 BB first | 260 | 387 | 382 | 1005 | 822 | -562 | -439 | +183 |
| board1 BTN vs check | 670 | 742 | 756 | 1874 | 2390 | -1721 | -1635 | -516 |
| board1 BTN vs 1/2 lead | 840 | 879 | 874 | 1955 | 2176 | -1336 | -1302 | -222 |
| **mean** | **541** | 595 | 599 | 1522 | 2396 | **-1855** | -1797 | -874 |
| mean BB first | 309 | 389 | 394 | 996 | 1078 | -769 | -684 | -82 |
| mean BTN vs check | 577 | 631 | 642 | 1860 | 3257 | -2680 | -2615 | -1397 |
| mean BTN vs 1/2 lead | 736 | 764 | 760 | 1711 | 2853 | -2117 | -2093 | -1142 |

**Two-sided exploitability** (mbb/hand):

| spot | value-net (river) | value-net (turn-end) | turn-end 4 s | rollout search | blueprint | value-net in its own game |
|---|---:|---:|---:|---:|---:|---:|
| board0 BB first | 4632 | 4028 | 4039 | 3985 | 2614 | 4605 |
| board0 BTN vs check | 386 | 422 | 419 | 1587 | 2610 | 333 |
| board0 BTN vs 1/2 lead | 457 | 469 | 466 | 1107 | 1856 | 370 |
| board1 BB first | 2164 | 2464 | 3002 | 2796 | 1591 | 2096 |
| board1 BTN vs check | 710 | 741 | 746 | 1478 | 1653 | 637 |
| board1 BTN vs 1/2 lead | 576 | 615 | 612 | 1273 | 1552 | 545 |
| **mean** | 1488 | 1457 | 1547 | 2037 | 1980 | 1431 |

The "BB first" two-sided numbers are dominated by the button's strategy inside
the search: BR_1 is 890 chips against 36 for BR_0 on board 0. That is the
safe-resolving gadget's opponent side. The gadget enters the opponent's hands
through terminate/enter choices whose terminate values come from 256 noisy
blueprint rollouts, so hands that would terminate keep unrefined subgame
strategies. The search never plays them, and the next table removes the gadget.

### Same spots, unsafe resolving (gadget off)

`runs/vn_eval/exploit6_unsafe.md`; river solves 200 iterations.

| spot | value-net (river) | value-net (turn-end) | rollout search | blueprint | value-net in its own game |
|---|---:|---:|---:|---:|---:|
| board0 BB first | 100 | 128 | 1137 | 2614 | 4 |
| board0 BTN vs check | 107 | 138 | 1252 | 2611 | 4 |
| board0 BTN vs 1/2 lead | 133 | 153 | 904 | 1858 | 2 |
| board1 BB first | 74 | 98 | 977 | 1593 | 3 |
| board1 BTN vs check | 71 | 95 | 1000 | 1655 | 3 |
| board1 BTN vs 1/2 lead | 63 | 79 | 1085 | 1553 | 2 |
| **mean** | **91** | 115 | 1059 | 1981 | 3 |
| mean BB first | 87 | 113 | 1057 | 2103 | 3 |
| mean BTN vs check | 89 | 117 | 1126 | 2133 | 3 |
| mean BTN vs 1/2 lead | 98 | 116 | 994 | 1705 | 2 |

* Value-net search converges to near equilibrium in its own game (3 mbb/hand).
  The 91 mbb/hand left in the exact game is the leaf net's error.
* Rollout search improves on the blueprint by about half, against about 95%
  for value-net search.
* The opponent's best response against the searcher also falls without the
  gadget (mean 213 vs 541 mbb/hand). At the first flop decision the gadget's
  noisy terminate values cost about 330 mbb/hand of the searcher's own
  exploitability. The ranges here are the blueprint's own, which is exactly
  the case where unsafe resolving is right. Safe resolving still guards
  against opponents whose ranges differ.

### Twelve spots, production config (safe resolving)

`runs/vn_eval/exploit12.md`: the six spots above plus boards 2 and 3, same
settings.

**Opponent's best response to the searcher's strategy**, minus the
blueprint's (mbb/hand; negative = better than the blueprint):

| spot | value-net (river) | value-net (turn-end) | turn-end 4 s | rollout search |
|---|---:|---:|---:|---:|
| board0 BB first | -977 | -943 | -928 | -348 |
| board0 BTN vs check | -3639 | -3604 | -3596 | -2277 |
| board0 BTN vs 1/2 lead | -2898 | -2881 | -2884 | -2063 |
| board1 BB first | -562 | -435 | -439 | +183 |
| board1 BTN vs check | -1721 | -1649 | -1635 | -516 |
| board1 BTN vs 1/2 lead | -1336 | -1297 | -1302 | -222 |
| board2 BB first | -349 | -328 | -331 | +434 |
| board2 BTN vs check | -3045 | -2997 | -2996 | -1883 |
| board2 BTN vs 1/2 lead | -2782 | -2742 | -2740 | -1801 |
| board3 BB first | -1014 | -981 | -987 | -270 |
| board3 BTN vs check | -1668 | -1647 | -1635 | -529 |
| board3 BTN vs 1/2 lead | -1016 | -998 | -1306 | -243 |
| **mean (12)** | **-1751** | -1709 | -1732 | -795 |
| mean BB first (4) | -725 | -672 | -671 | -0 |
| mean BTN vs check (4) | -2518 | -2474 | -2465 | -1301 |
| mean BTN vs 1/2 lead (4) | -2008 | -1980 | -2058 | -1082 |

The absolute best-response means are: value-net 507, turn-end 549,
turn-end 4 s 526, rollout 1,463, blueprint 2,257 mbb/hand.

**Two-sided** means (mbb/hand):

| spot type | value-net (river) | value-net (turn-end) | turn-end 4 s | rollout search | blueprint |
|---|---:|---:|---:|---:|---:|
| all 12 | 1416 | 1401 | 1431 | 1958 | 1861 |
| BB first (4) | 3238 | 3147 | 3275 | 3292 | 1878 |
| BTN vs check (4) | 498 | 525 | 530 | 1421 | 1979 |
| BTN vs 1/2 lead (4) | 513 | 531 | 487 | 1161 | 1725 |

### Richer trunk: the blueprint's flop sizes kept

`runs/vn_eval/exploit_rich.md`:

* 20,000-node trees (leaves count 1 + k) keep flop bets of 0.5, 0.75, 1, 1.5
  and 2 pot, the pot re-raise and all-in. 0.25 and 0.33 are dropped, and the
  turn is still pot re-raise and all-in.
* Each tree has 5,736 decision nodes and 1,421 leaves (68k river subgames per
  profile).
* River solves use 200 iterations; safe resolving; board 0 only.

| spot | value-net (river) | value-net (turn-end) | rollout search | blueprint |
|---|---:|---:|---:|---:|
| **BR against the searcher** (mbb/hand) | | | | |
| board0 BB first | 320 | 370 | 1048 | 3073 |
| board0 BTN vs check | 560 | 588 | 1901 | 5016 |
| mean | **440** | 479 | 1474 | 4045 |
| **two-sided** (mbb/hand) | | | | |
| board0 BB first | 5264 | 5394 | 5639 | 3921 |
| board0 BTN vs check | 491 | 526 | 1682 | 3995 |
| **decision time**, 300 iterations (s) | 61.6 | 11.9 | 26.2 | |

* The blueprint now leaves only 1.4 off-tree decisions per hand, but its
  exploitability rises (2,730 -> 4,045 mbb/hand one-sided, mean of these two
  spots). The best responder can use the extra flop sizes too.
* Value-net search improves on the blueprint by 3,605 mbb/hand (one-sided).
* The river-net fan-out takes 200 ms per iteration at 1,421 leaves, against
  about 35 ms for the turn-end net.
* The two-sided BB-first number again carries the gadget's opponent side.

### Timing per flop decision (RTX 4070 Ti)

Means over the six spots with safe resolving; trees of 2,680 nodes and 294
leaves on average.

| search | total s | setup s | solve s | iterations | ms/iteration |
|---|---:|---:|---:|---:|---:|
| value-net, river net (48 rows/leaf) | 15.65 | 1.17 | 14.33 | 300 | 47.8 |
| value-net, turn-end net | 6.57 | 1.11 | 5.31 | 300 | 17.7 |
| **value-net, turn-end net, 4 s budget** | **4.02** | 1.11 | 2.77 | 154 | 18.0 |
| rollout search (default leaves) | 7.59 | 2.53 | 4.92 | 300 | 16.4 |

* Setup includes the gadget's 256 blueprint rollouts, about 0.9 s.
* The one-time preflop equity table (about 2 s) is warmed up before the runs.
* With the gadget off, setup is 0.27 s.

### Head-to-head match

`runs/vn_match/match.json` (`scripts/match_search.py`, `configs/match_search_vn.yaml`):

* A is the value-net search agent, B the blueprint
  (`neural:runs/dcfr4_distilled_v2`); both play preflop with the blueprint.
* 1,000 duplicate deals, i.e. 2,000 hands, in 20 chunks of 50 deals (seeds
  0–19).
* Search config `configs/search_value_net.yaml`:
  * turn-end net `turn_v1.pt`;
  * flop 4 s, turn 2 s, river 1 s;
  * blueprint actions, trees of at most 6,000 nodes (leaves count 1);
  * safe resolving.

| | mbb/hand | 95% CI |
|---|---:|---|
| raw | +168 | [-264, +580] |
| luck-adjusted (all-in EV + street control variates) | +131 | [-227, +492] |

| street | search decisions | fallbacks | mean time | p90 | max | mean DCFR iterations |
|---|---:|---:|---:|---:|---:|---:|
| flop | 1303 | 0 | 4.03 s | 4.04 s | 4.08 s | 163 |
| turn | 730 | 0 | 2.02 s | 2.04 s | 2.06 s | 136 |
| river | 475 | 0 | 1.00 s | 1.01 s | 1.01 s | 179 |

The match took 2 h on the 4070 Ti (about 3.6 s per hand). Separating +100 to
+200 mbb/hand from zero needs roughly 12–26k hands (the CI half-width shrinks with the square root of the hand count).

## Follow-up (2026-10-06): more river data, on-policy data, turn-end net v2

**Data added**

| set | samples | DCFR iterations | mean solve exploitability (pot) | notes |
|---|---:|---:|---:|---|
| `river_b` | 573,440 | 300 | 0.21% | same mix as `river_a`, seed 1; 40 samples/s (1.5x faster than 400 iterations) |
| `river_onpolicy` | 72,602 | 300 | 0.15% | **on-policy** (source 3): turn-end leaf reaches recorded during 422 value-net flop searches (300 duplicate deals against the blueprint, turn-end net v1 as the leaf model), 2 random river cards each |
| `river_heldout` | 16,384 | 400 | 0.14% | fixed held-out set (seed 999), same mix |
| `river_onpolicy_heldout` | 12,589 | 400 | 0.10% | on-policy held-out from 59 other searches (other deals) |
| `turn_b` | 1,500,000 | - | - | turn-end states, targets bootstrapped from river v2 (252 samples/s) |
| `turn_onpolicy` (+ held-out) | 36,304 (+6,296) | - | - | the recorded turn-end leaf states themselves, labelled with river v2 |

The recorder (`search/leaf_recorder.py`) wraps the leaf provider. On every
20th regret update it stores 8 random turn-end leaves whose reach is
non-negligible for both players. Those are exactly the inputs the net is
queried with.

**River net v2** (`river_v2.pt`: `river_a` + `river_b` + `river_onpolicy`,
850k samples; 40k steps, 13.5 min, samples in host memory). The comparison
below uses the same 29k held-out samples (`scripts/eval_value_net.py`); pot
units per combo.

| | v1 MAE | v2 MAE | v1 wMAE | v2 wMAE | v1 game value | v2 game value |
|---|---:|---:|---:|---:|---:|---:|
| all held-out | 0.0391 | 0.0347 | 0.0317 | 0.0299 | 0.0088 | 0.0081 |
| blueprint ranges | 0.0219 | 0.0205 | 0.0203 | 0.0195 | 0.0044 | 0.0041 |
| perturbed | 0.0247 | 0.0229 | 0.0228 | 0.0217 | 0.0048 | 0.0046 |
| random | 0.0573 | 0.0472 | 0.0329 | 0.0290 | 0.0089 | 0.0076 |
| on-policy | 0.0490 | 0.0437 | 0.0415 | 0.0397 | 0.0129 | 0.0120 |

On-policy states are the hardest: their bucket-oracle floor is 0.019 against
0.008 for blueprint ranges, because the ranges are wider and blockers matter
more. 4x the data gave a modest gain. Training loss rose from 0.0006 to 0.0009,
so the net now overfits less and capacity starts to matter (see the capacity
test below).

**Turn-end net v2** (`turn_v2.pt`: `turn_b` + `turn_onpolicy`, 1.49M samples;
40k steps, 6 min). Held-out against its own bootstrapped targets: MAE 0.0285
(v1: 0.0327), range-weighted 0.0289 (0.0321), game value 0.0067 (0.0079).

**Exact turn-end leaf check** (`scripts/check_turn_leaves.py`): 400 states,
each against all 48 river subgames solved exactly; the same states for v1 and
v2.

| states | leaf model | MAE v1 -> v2 | wMAE v1 -> v2 | game value v1 -> v2 |
|---|---|---|---|---|
| training-style mix | river net x river cards | 0.0149 -> 0.0136 | 0.0124 -> 0.0122 | 0.0030 -> 0.0031 |
| training-style mix | turn-end net | 0.0355 -> 0.0312 | 0.0339 -> 0.0308 | 0.0085 -> 0.0069 |
| **on-policy** (search leaf states) | river net x river cards | 0.0278 -> 0.0242 | 0.0231 -> 0.0229 | 0.0076 -> 0.0067 |
| **on-policy** (search leaf states) | turn-end net | **0.0543 -> 0.0446** | 0.0502 -> 0.0448 | **0.0172 -> 0.0131** |

The turn-end net, the production leaf model, improved most on the states
search actually queries: -18% MAE and -24% game-value error.

**Exploitability**, the six original spots with safe resolving off, two-sided
(mbb/hand; `runs/vn_eval/exploit6_unsafe_v2.md`):

| spot | turn-end v1 | **turn-end v2** | river v1 | river v2 | rollout search | blueprint |
|---|---:|---:|---:|---:|---:|---:|
| board0 BB first | 128 | 118 | 100 | 99 | 1137 | 2614 |
| board0 BTN vs check | 138 | 125 | 107 | 103 | 1252 | 2611 |
| board0 BTN vs 1/2 lead | 153 | 143 | 133 | 128 | 904 | 1858 |
| board1 BB first | 98 | 86 | 74 | 74 | 977 | 1593 |
| board1 BTN vs check | 95 | 85 | 71 | 70 | 1000 | 1655 |
| board1 BTN vs 1/2 lead | 79 | 67 | 63 | 57 | 1085 | 1553 |
| **mean** | 115 | **104** | 91 | 88 | 1059 | 1981 |

Turn-end net v2 is 10% less exploitable than v1 and better on every spot. The
opponent's best response against the searcher fell from 237 to 225 mbb/hand
(blueprint 2,397). The river-card fan-out improved 3%.

**Capacity test** (width 2048 instead of 1024, same data, steps and held-out
split): held-out MAE in pot units.

| net | width 1024 | width 2048 | on-policy, 1024 -> 2048 | parameters | training time |
|---|---:|---:|---:|---:|---:|
| turn-end | 0.0285 | **0.0258** | 0.0369 -> 0.0344 | 4.3M -> 14.8M | 6 -> 9 min |
| river | 0.0347 | 0.0336 | 0.0437 -> 0.0425 | 4.3M -> 14.8M | 13.5 -> 16 min |

* The wide turn-end net (`turn_v2w.pt`) also wins against exact solves:
  * on-policy states: MAE 0.0446 -> 0.0423, game value 0.0131 -> 0.0125;
  * training-style mix: MAE 0.0312 -> 0.0288, game value 0.0069 -> 0.0064.
* It runs one row per leaf, so its extra cost does not show: flop decisions
  are still 4.03 s with about 156 iterations.
* The wide river net gains only 3% but would make the 48-row fan-out 3.5x
  dearer, so it is not used.
* The production config (`configs/search_value_net.yaml`) now uses
  `turn_v2w.pt`.

**Exploitability of the wide turn-end net** (`runs/vn_eval/exploit6_unsafe_v2w.md`;
same six spots, unsafe, two-sided, mbb/hand):

| spot | turn-end v1 | turn-end v2 | **turn-end v2 wide** |
|---|---:|---:|---:|
| board0 BB first | 128 | 118 | 111 |
| board0 BTN vs check | 138 | 125 | 119 |
| board0 BTN vs 1/2 lead | 153 | 143 | 138 |
| board1 BB first | 98 | 86 | 87 |
| board1 BTN vs check | 95 | 85 | 83 |
| board1 BTN vs 1/2 lead | 79 | 67 | 68 |
| **mean** | 115 | 104 | **101** |

The opponent's best response against the searcher is 237 -> 225 -> 222 mbb/hand
across the three versions. The retrained turn-end net closes about 40% of the
gap between the v1 turn-end net (115) and the river-card fan-out (v2: 88),
at the turn-end net's speed: 19 ms per iteration against 49.

**Summary of the follow-up**

* 4.2x the river data (205k -> 862k), including 73k on-policy samples, made
  the river net 11% more accurate on held-out data and 13% more accurate at
  on-policy turn-end leaves.
* 5x the turn-end data, bootstrapped from the better river net and including
  on-policy states, plus a wider net, made the production leaf model 22% more
  accurate at on-policy leaves against exact solves (MAE 0.054 -> 0.042, game
  value 0.017 -> 0.0125).
* Exploitability fell 12% (115 -> 101 mbb/hand, unsafe two-sided) with
  unchanged decision time.
* The remaining error is mostly at on-policy states, which are wide ranges
  where blockers matter. Their bucket-oracle floor is 0.019, against 0.008 for
  blueprint ranges, so a per-combo (blocker-aware) output is the next accuracy
  lever. More data alone gives diminishing returns: 4x the data gave -11%.

## Round 2 (2026-10-07)

Branch `claude/value-net-r2`. Raw results are in `runs/vn_eval/r2_*`,
`runs/tree_size/size3.*` and `runs/value_net/*r2*` / `*v3*` (gitignored). All
exploitability numbers use the exact-river evaluation above. Unless noted, the
settings are the six original flop spots, 300 search iterations on the
6,000-node evaluation trunk (leaves count 1 + k), river solves of 200
iterations, and the production leaf net `turn_v2w`.

### Correction: the blueprint baseline was mis-scored

* **Cause.** `tree.states[n]` below a chance node is the engine state of the
  chance *template*: its betting is right, but its board has whatever card the
  tree builder dealt. The old `spot_eval.blueprint_profile` queried the
  blueprint with that state. So at 864 of the 888 decision nodes of a typical
  flop tree, the blueprint played its turn strategy for the wrong turn card.
* **Scope.** Only the blueprint column of every earlier table above is
  affected. The searches use the right boards (leaves, rollouts and gadget
  read `tree.boards`).
* **Fix.** `tree_policy.blueprint_profile` takes each node's board from
  `tree.boards`. It also batches every node into a few network calls via
  `vec_rollouts`: 1–2 s instead of 3–7 s per evaluation tree. A regression
  test checks that turn nodes are queried on their own board.

| spot | blueprint one-sided, old | **corrected** | blueprint two-sided, old | **corrected** |
|---|---:|---:|---:|---:|
| board0 BB first | 1335 | **1082** | 2614 | **1450** |
| board0 BTN vs check | 4125 | **2025** | 2611 | **1421** |
| board0 BTN vs 1/2 lead | 3532 | **1151** | 1858 | **632** |
| board1 BB first | 825 | **446** | 1593 | **993** |
| board1 BTN vs check | 2391 | **1396** | 1655 | **976** |
| board1 BTN vs 1/2 lead | 2177 | **1042** | 1553 | **837** |
| **mean** | 2398 | **1190** | 1981 | **1051** |

The blueprint is about half as exploitable as reported. Value-net search is
still far less exploitable, but by smaller factors:

* two-sided, unsafe: 101 against 1,051 mbb/hand, about 10x rather than 20x;
* one-sided, production: 221 against 1,190.

The 12-spot and richer-trunk "better than the blueprint" margins above are
overstated by a similar factor and have not been re-run.

### 1. Gadget terminate values at the first decision of a street

New `gadget.terminate` modes (`pokerbot/search/README.md`, "Safe resolving
gadget"):

* `rollouts`: 256 (or more) blueprint-vs-blueprint rollouts, as before;
* `blueprint`: the opponent's values with both players on the blueprint, computed
  in the search tree itself with its own net leaves;
* `blueprint_br`: the opponent's best response to the blueprint there (CFR-D's
  `T`);
* `unsafe`: no gadget when no earlier solve is cached.

All modes were searched on one trunk per spot (`eval_search_exploit.py --extra`).

The safety columns use the new per-hand metric (`spot_eval.safety_vs`). For each
hand the searcher's opponent may hold, it is how much more that hand's best
response gets against the profile than against the blueprint, in the exact
game. Safe resolving promises zero for every hand. The excess is weighted by
the assumed opponent range, or by a uniformly random hand for an opponent
whose range differs.

**Opponent's best response against the searcher (one-sided), mbb/hand:**

| spot | rollouts 256 (old prod.) | rollouts 4096 | blueprint (tree) | blueprint BR (tree) | **unsafe** | blueprint |
|---|---:|---:|---:|---:|---:|---:|
| board0 BB first | 392 | 156 | -11 | 251 | **-97** | 1082 |
| board0 BTN vs check | 501 | 1161 | 531 | 1132 | **443** | 2025 |
| board0 BTN vs 1/2 lead | 639 | 1727 | 777 | 889 | **647** | 1151 |
| board1 BB first | 384 | 27 | -41 | 73 | **-113** | 446 |
| board1 BTN vs check | 694 | 982 | 251 | 606 | **193** | 1396 |
| board1 BTN vs 1/2 lead | 868 | 1279 | 257 | 828 | **255** | 1042 |
| **mean** | 580 | 889 | 294 | 630 | **221** | 1190 |

**Two-sided, safety and cost** (means over the six spots):

| mode | two-sided | excess, assumed range | excess, uniform hand | worst hand | hands worse by > 1% pot | violation in own game (chips) | T seconds | flop iterations in 4 s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| rollouts 256 | 1468 | 139 | 147 | 2563 | 11% | 159 | 0.87 | 141 |
| rollouts 4096 | 1359 | 216 | 246 | 1876 | 36% | 116 | about 14 | - |
| blueprint (tree) | 401 | 7.0 | 7.6 | 33 | 4.8% | 19 | 2.62 | 52 |
| blueprint BR (tree) | 793 | 21 | 24 | 898 | 8.2% | 2.5 | 2.63 | 51 |
| **unsafe** | **101** | **5.5** | **5.7** | **8** | 4.3% | - | 0 | **190** |

Timing is per production decision (5.8k-node trees, `scripts/time_flop_search.py`,
12 decisions each on an idle machine). "Worst hand" is the mean over spots of
the largest per-hand excess.

* **The rollout gadget was the worst option, on strength and on safety.** Its
  terminate values come from blueprint play to showdown, while the entry values
  come from the search's own game (net leaves). The two disagree systematically,
  and 4096 rollouts are no better than 256: this is bias, not noise. Some
  opponent hands gained 2.5 bb/hand more than against the blueprint.
* **Terminate values computed in the search tree** remove that mismatch.
  * `blueprint`: one-sided 294, excess 7 mbb.
  * `blueprint_br`: almost exactly safe in its own game (violation 0–12
    chips). In the exact game it is less safe than `blueprint` because of the
    net's error, and weaker. Loose terminate values let most opponent hands
    terminate, so the re-solve stops refining against them.
  * Both cost about 2.6 s of the 4 s budget (a batched blueprint pass over
    2,572 nodes plus one best-response pass): only about 50 iterations.
* **Unsafe resolving at the first decision is best on every measure.** At the
  first flop decision the root ranges are the blueprint's own, so its solve is
  near-equilibrium against the right range. Per hand it is about as safe as the
  blueprint (excess 5.5 mbb, uniform 5.7), and it leaves the whole budget to
  DCFR.
* **Production change.** `configs/search_value_net.yaml` sets
  `gadget.terminate: unsafe`. The searcher's own exploitability on these spots
  falls from 580 to 221 mbb/hand (-62%). On the 6,000-node tree, flop decisions
  get 190 instead of 141 iterations. Later streets still re-solve safely from
  the cached solve.

### 2. Blocker-aware turn-end net

`train_turn_net.py --residual-head` (per-combo head with blocker features) on
`turn_b` + `turn_onpolicy`, 40k steps, same split as `turn_v2w`:

| net | held-out MAE | on-policy rows | exact MAE, on-policy states | exact game value, on-policy | exact MAE, training mix | iterations in 4 s | training time |
|---|---:|---:|---:|---:|---:|---:|---:|
| `turn_v2w` (2048, production) | 0.0258 | 0.0344 | 0.0423 | 0.0125 | 0.0288 | 180 | 9 min |
| `turn_v2r` (1024 + head) | 0.0281 | 0.0366 | 0.0438 | 0.0127 | 0.0313 | 151 | 29 min |
| `turn_v2wr` (2048 + head) | 0.0252 | 0.0339 | 0.0420 | 0.0130 | 0.0283 | 149 | 34 min |

"Exact" columns: `check_turn_leaves.py` on the same 400 held-out on-policy
states as before (and 400 fresh training-mix states), each against all 48
river subgames solved exactly. River-net fan-out there: 0.0242 and 0.0136.

**Six spots, safe resolving off** (mbb/hand; `runs/vn_eval/r2_turnres6_unsafe.md`):

| | `turn_v2w` | `turn_v2r` | `turn_v2wr` |
|---|---:|---:|---:|
| two-sided mean | 101 | 103 | 99 |
| one-sided mean | 222 | 224 | 221 |

* **The head does not help.** Against exact solves at on-policy states it gains
  0.7% MAE and loses 4% in game value. Exploitability is unchanged within 2%,
  and it costs 17% of the flop iterations.
* **Why.** The turn-end net's targets are bootstrapped from the bucket-level
  river net, averaged over river cards with exact card removal. A per-combo
  head can learn only the blocker structure that is in its targets, and these
  hold little beyond the exact card-removal weights.
* **What would help.** Blocker-aware *targets*: a residual-head river net for
  bootstrapping (its 7x inference cost only matters offline), or exact 48-river
  solves at on-policy states.
* **Production.** Stays on `turn_v2w`.

### 3. Second on-policy round

ReBeL style: record the leaf states the current production search queries,
solve them exactly, retrain, and re-bootstrap (`runs/value_net/r2_onpolicy.cmd`).

* **Recording.** The new production search (unsafe first decisions,
  20000-node flop trees, `turn_v2w`) played 300 duplicate deals against the
  blueprint, seeds new. It recorded 8 turn-end leaves every 7 regret updates:
  76,880 turn-end states from 407 flop searches. Two random river cards each
  gave 153,747 river subgames, solved exactly at 300 iterations (mean
  exploitability 0.14% of the pot). A held-out set of 40 other deals gave
  8,760 turn states and 17,520 river samples at 400 iterations.
* **`river_v3`.** `river_a` + `river_b` + both on-policy rounds: 1.0M samples,
  same architecture and 40k steps as `river_v2`.
* **Turn data.** `turn_b`'s 1.5M states relabelled with `river_v3`
  (`label_turn_states.py --shards`, 12 min instead of 99 for new data), plus
  both rounds' on-policy turn states (113k).
* **`turn_v3w`.** 2048 wide, no head, 40k steps.

**River net on fixed held-out data** (46.5k samples including both rounds'
on-policy held-out; pot units):

| | MAE | on-policy MAE | on-policy game value | blueprint ranges MAE |
|---|---:|---:|---:|---:|
| `river_v2` | 0.0347 | 0.0384 | 0.0103 | 0.0205 |
| `river_v3` | 0.0343 | 0.0377 | 0.0101 | 0.0207 |

**Turn-end leaves against exact 48-river solves** (400 held-out on-policy
states per round; pot units):

| states | model | MAE | game value |
|---|---|---:|---:|
| round 1 (old production search) | river net fan-out, v2 / v3 | 0.0242 / 0.0240 | 0.0067 / 0.0066 |
| round 1 | `turn_v2w` / **`turn_v3w`** | 0.0423 / 0.0419 | 0.0125 / 0.0127 |
| round 2 (new production search) | river net fan-out, v3 | 0.0197 | 0.0053 |
| round 2 | `turn_v2w` / **`turn_v3w`** | 0.0371 / **0.0354** | 0.0109 / **0.0100** |

**Six spots, unsafe** (mbb/hand; `runs/vn_eval/r2_onpolicy6_unsafe.md`):
`turn_v3w` scores 99 two-sided and 219 one-sided, against 101 and 222 for
`turn_v2w`. It is better or equal in every spot, with the same per-hand excess
over the blueprint (5.4 mbb).

* **A small gain.** On the states today's production search queries,
  `turn_v3w` cuts the leaf error by 5% (game value 8%); exploitability falls 2%.
  `configs/search_value_net.yaml` now uses `turn_v3w`.
* **The river net has saturated in this form.** 18% more data, all of it
  on-policy, gave -2% at on-policy states. Its on-policy bucket-oracle floor
  is 0.015.
* **The turn-end net is about twice the river fan-out's error** at the same
  states (0.035 against 0.020). That gap is its own fit to the bootstrapped
  targets, not the river net.
* **The new production states are easier.** Round-2 leaf states, from 20000-node
  flop trees with unsafe first decisions, show lower errors for every model
  than round 1's.

### 4. Production tree size

`scripts/eval_tree_size.py` (`runs/tree_size/size3.md`) setup:

* **Searches.** Production searches (turn-end net, one-row leaves, unsafe first
  decision) at `max_nodes` 6000 / 10000 / 20000, under the 4 s flop budget.
* **Spots.** board0 BB first, board0 BTN vs check, board1 BB first.
* **Trunk.** The 20000-node tree: 18,294 nodes, 2,009 leaves, and the blueprint's
  whole flop abstraction (opens 0.25 to 2 pot, the pot re-raise, all-in). The
  smaller trees keep subsets of its sizes; the turn is the same in all.
* **Translated.** A smaller tree's strategy is put on the trunk with
  `tree_map.translate_sigma`. An opponent size it lacks maps to its nearest size
  (pseudo-harmonic), for 68% (6000) / 54% (10000) of trunk nodes. This is
  pessimistic: it answers a 0.25-pot bet as if it were 0.75.
* **Re-searched.** The same agent searches again at every off-tree opponent flop
  action, with that size forced into its tree and its own earlier decisions
  locked, as it does in play: 5 re-searches per spot at 6000, 4 at 10000.
  These columns are the ones to compare.

**Mean over the three spots** (mbb/hand; one-sided = opponent's best response
against the searcher):

| flop tree | nodes | leaves | iterations in 4 s | one-sided, re-searched | one-sided, translated | two-sided, re-searched | own game |
|---|---:|---:|---:|---:|---:|---:|---:|
| 6000 (old production) | 5,802 | 637 | 188 | 209 | 966 | 677 | 16 |
| 10000 | 8,478 | 931 | 159 | 221 | 585 | 482 | 57 |
| **20000** | 18,294 | 2,009 | **65** | **147** | 147 | **186** | 103 |
| blueprint | | | | 2,878 | 2,878 | 2,987 | |

| spot | 6000 | 10000 | 20000 | 20000 - 6000 |
|---|---:|---:|---:|---:|
| board0 BB first (one-sided / two-sided) | -26 / 707 | -0 / 494 | -107 / 178 | -81 / -529 |
| board0 BTN vs check | 673 / 548 | 664 / 400 | 595 / 168 | -78 / -381 |
| board1 BB first | -20 / 776 | -0 / 551 | -47 / 213 | -27 / -563 |

* **Richer trees beat more iterations.** The 20000 tree gets only 65 DCFR
  iterations (54 ms each against 19) and is the least converged in its own
  game. Even so, it is the least exploitable searcher in all three spots:
  * one-sided, 30% below the 6000 tree;
  * two-sided, 73% below;
  * 10000 sits in between on two-sided and level with 6000 on one-sided.
* **Production change.** `tree.max_nodes_flop: 20000` (new option) gives flop
  searches the whole flop abstraction. Turn and river trees keep
  `max_nodes: 6000`, since this test covers flop decisions only.
* **Caveat.** The trunk itself keeps the turn's all-in-only opening
  (see `keep_open` below), so turn bet sizing is untested here.
* The blueprint is far more exploitable on this richer trunk (2,878 one-sided
  against 1,190 on the coarse trunk): the best responder gets every flop size
  too.

### 5. Head-to-head

`runs/vn_match/match_r2.json` (`scripts/match_search.py`, `configs/match_search_vn.yaml`):

* **Players.** A is the production search (`configs/search_value_net.yaml` as
  copied to `runs/vn_match/r2_search_config_used.yaml`):
  * leaf net `turn_v3w`;
  * flop trees of 20,000 nodes, turn and river 6,000;
  * no gadget at the first decision of a street, the safe gadget from the
    cached solve afterwards;
  * 4 / 2 / 1 s per flop / turn / river decision.

  B is the blueprint `dcfr4_distilled_v2`; both play preflop with it.
* **Hands.** 5,000 duplicate deals (10,000 hands) in 100 chunks of 50, seeds
  1000-1099, independent of round 1's seeds 0-19. The run was paused once at
  chunk 84 and resumed; only the unfinished chunk was replayed.

| | mbb/hand | 95% CI |
|---|---:|---|
| raw | +162 | [-13, +340] |
| **luck-adjusted** (all-in EV + street control variates) | **+199** | **[+44, +364]** |

| street | search decisions | fallbacks | mean time | p90 | max | mean DCFR iterations |
|---|---:|---:|---:|---:|---:|---:|
| flop | 6,383 | 0 | 4.06 s | 4.09 s | 4.21 s | 81 |
| turn | 3,857 | 0 | 2.02 s | 2.04 s | 2.08 s | 129 |
| river | 2,585 | 0 | 1.00 s | 1.01 s | 1.02 s | 162 |

* **Significant.** The production search beats its own blueprint by about
  0.2 bb/hand: the luck-adjusted CI excludes zero, the raw one barely
  includes it.
* **Consistent with round 1.** Round 1's 2,000-hand match (old config)
  measured +131, CI [-227, +492].
* **Not a comparison of configs.** At this sample size the old and new
  configs' win rates cannot be told apart.
* **Stability.** All 12,825 search decisions searched (no fallback). The
  slowest flop decision took 4.21 s.
* **Speed.** The match took about 10 h on the RTX 4070 Ti, about 3.6 s per
  hand.

### 6. Turn-start net and batched turn solves (stretch)

Code and CPU tests only so far; design and measurements are in
`docs/turn_start_net.md`.

* `BatchTurnSolver` (`search/batch_turn_solver.py`) solves B turn subgames
  with turn-end-net leaves and exact river-averaged all-ins. Per instance it
  matches `RangeSolver` to 1e-13.
* The turn-start data pipeline (`scripts/gen_turn_start_data.py`) and
  `TurnStartPredictor` (`kind: turn_start`).
* `FlopEndLeafEvaluator` averages a turn-start net over the 49 turn cards.
* `tree.depth_streets_turn` and `leaf.turn_net` give depth-limited turn
  solves with turn-end-net leaves. `configs/search_turn_start.yaml` is the
  flop `depth_streets: 0` setup.
* GPU data generation and training are queued after the second on-policy
  round, so they can bootstrap from the better turn-end net.

### Other changes

* **Lock fix** (`agent.played_lock`). When a re-search's forced off-tree branch
  pushes the budget into dropping one of our earlier played sizes, the played
  strategy used to be renormalised over the remaining actions. That inflated
  the action we took and distorted our range below it. Now the action taken
  keeps its exact probability.
* **`tree.keep_open`** (off). With the blueprint's spec, the node budget drops
  the turn's 1.0 open before its 1.0 re-raise (a tie in the drop order). So at
  every flop-search budget the turn's only opening bet is all-in.
  * `keep_open` drops the re-raise instead.
  * At the 6000 / 10000 / 20000 budgets that gives 3,108 / 6,066 / 16,122
    nodes with fewer flop sizes.
  * Evaluating it needs a scoring trunk with turn opens (34k nodes, 4,655
    leaves).
* **Tooling.**
  * `spot_eval`: `--extra NAME=JSON` variants on one trunk, `--no-rollout`, and
    the per-hand safety table.
  * `scripts/time_flop_search.py`: production decision timing.
  * `scripts/eval_tree_size.py`: tree size under a time budget, translated and
    re-searched (`search/tree_map.py`, `search/size_eval.py`).
  * `check_turn_leaves.py` compares several turn nets on one set of exact
    solves.
  * `label_turn_states.py --shards` relabels existing turn data with a new
    river net, about 7x faster than regenerating it.

## What worked, what didn't

**Worked**

* **Exact batched river solving makes clean targets cheap.** About 27 solved
  subgames/s at 0.13% of the pot mean exploitability; 205k samples in 2 h.
  The same solver makes an exact, leaf-free evaluation possible.
* **The bucketed MLP with a zero-sum layer is good enough at the leaves search
  uses.** Its river error, 0.031 pot per combo, averages down to 0.015 at
  turn-end leaves over the 44 river cards. Search converges to 3 mbb/hand in
  the net's game and lands at 91 mbb/hand in the exact game. The rollout
  game's noise is gone: two solves of a spot no longer disagree.
* **The turn-end auxiliary net is the right production default.** It is about
  3x cheaper per iteration than the river-card fan-out, meets the < 5 s target
  with 150 iterations, and loses only about 2% of the improvement.
* **The leaf-free evaluation changed the picture.** The old metric's "search
  is no better than the blueprint, and 50–90% worse when BB is first to act"
  was partly a measurement artifact: the two-sided number scores the
  opponent-side strategy that safe resolving leaves unrefined. Measured
  one-sided, even rollout search beats the blueprint on average, though not in
  every BB-first spot.

**Didn't work, or still open**

* **Two-sided exploitability in BB-first spots is still worse than the
  blueprint's with safe resolving on.** That is the gadget's opponent side, not
  the net: unsafe resolving gives 87 vs 2,103 mbb/hand there. If the two-sided
  number is the acceptance criterion as literally stated, value-net search
  passes it only with the gadget off.
* **The gadget costs the searcher itself.** At the first flop decision its
  terminate values come from 256 noisy blueprint rollouts. The one-sided BR
  against the searcher is 541 mbb/hand with the gadget and 213 without, on the
  6 spots.
* **The river net is limited by data.** Train loss is about 10x below held-out;
  dropout and weight decay did not help. A per-combo blocker head helped 5% at
  7x inference cost and is not used.
* **The river-card fan-out is too slow for production trees.** It needs 48 rows
  per leaf: 48 ms/iteration at 343 leaves, and over 100k rows per call at
  20k nodes.
* **The 6,000-node evaluation trunk is coarse.** The blueprint's other sizes
  are renormalised (about 2.4 off-tree decisions per hand); the turn has no
  opening bet below all-in.
  * The 20,000-node check keeps the blueprint's flop sizes from 0.5 to 2 pot,
    and value-net search's lead grows there.
  * Turn opening sizes only appear at about 120k nodes: about 15k leaves and
    700k river subgames per profile, too many to score exactly.
* **Turn solves (to showdown, no leaves) get only about 56 iterations per
  second.** The match used 2 s on the turn. The value net does not help there;
  a turn-start net would.
* **Not built:** a turn-start net for `depth_streets: 0` flop solves. It needs
  turn subgames with value-net leaves solved in batches, which is a batched
  version of `RangeSolver` with leaf providers.

## Next steps

Status of the round-1 list after Round 2:

1. *Blocker-aware outputs:* done for the turn-end net, with no gain (Round 2,
   section 2). It needs blocker-aware targets first.
2. *More on-policy data, iterated:* done once more (section 3); a small gain.
3. *Better first-decision gadget values:* done (section 1). Production now
   resolves unsafely there.
4. *Bigger production trees:* done for the flop (section 4): 20,000 nodes.
5. *Turn-start net and batched turn solving:* code done (section 6); GPU
   data and training queued.
6. *Head-to-head at scale:* done (section 5). +199 mbb/hand luck-adjusted,
   significant at 10,000 hands.

Next, in order of expected value:

1. **Turn bet sizing in flop searches.**
   * Every budget leaves all-in as the turn's only opening bet (`keep_open`,
     section "Other changes").
   * Test `keep_open` and a turn open plus re-raise on a trunk that has turn
     opens: about 34k nodes and 4,655 leaves, about 45 min of scoring per
     profile per spot.
2. **The turn-end net's own fit.**
   * At production leaf states it has about twice the error of the river-net
     fan-out it imitates (0.035 against 0.020 pot).
   * Its targets are not the limit: more steps, a larger or deeper net, or a
     loss weighted towards on-policy states are cheap to try.
3. **Blocker-aware targets**, to get below the bucket floor:
   * bootstrap the turn-end data from a residual-head river net (its
     inference cost only matters offline);
   * or train on exact 48-river solves of on-policy states.
4. **Turn-start net:**
   * held-out accuracy and depth-0 flop timing once trained;
   * then an exploitability comparison against depth-1 flop search, which
     needs exact turn solves at the flop leaves (expensive) or the
     depth-1 trunk with turn play taken from a depth-1 search.
   * Also evaluate turn decisions with turn-end-net leaves
     (`depth_streets_turn: 0`) against today's turn solves to showdown.
5. **Re-run the 12-spot and richer-trunk blueprint comparisons** with the
   corrected blueprint profile.
6. **Cheaper in-tree terminate values** (bf16, per-node history and board
   branches computed once), if safe resolving at first decisions is wanted
   against opponents whose preflop play differs from the blueprint's.
