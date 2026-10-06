# Value-net leaves for flop search: evaluation (2026-10-05)

Branch `claude/value-net` (worktree `poker-bot-search`). Design: `docs/value_net.md`.
Raw results: `runs/vn_eval/*.md|json` and `runs/value_net/*.json` (gitignored).

## Summary

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

* **Head to head**, 2,000 hands (duplicate, 1,000 deals): value-net search
  (production config) against its blueprint scores +168 mbb/hand raw
  (95% CI -264 to +580) and +131 luck-adjusted (CI -227 to +492). Positive
  but not significant at this sample size. All 2,508 search decisions
  searched, with no blueprint fallback; flop decisions took 4.03 s on
  average and 4.08 s at most.

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
  are renormalised (about 2.4 off-tree decisions per hand), which favours the
  searches; the turn has no opening bet below all-in. PENDING: richer-trunk
  check.
* **Turn solves (to showdown, no leaves) get only about 56 iterations per
  second.** The match used 2 s on the turn. The value net does not help there;
  a turn-start net would.
* **Not built:** a turn-start net for `depth_streets: 0` flop solves. It needs
  turn subgames with value-net leaves solved in batches, which is a batched
  version of `RangeSolver` with leaf providers.

## Next steps

1. **More river data, then retrain.**
   * 1M samples is about 10 h at the current rate.
   * Add on-policy samples: leaf reaches recorded during value-net flop
     searches (ReBeL style). The net is queried on the solver's iterates,
     which the self-play, perturbed and random mix only approximates.
2. **Retrain the turn-end net on millions of bootstrapped samples.** Targets
   cost 48 river-net rows, no solving; this would close most of its 2.4x gap to
   the fan-out. The random-range states can be generated without self-play.
3. **Better first-decision gadget values.** For example, evaluate the
   blueprint's trunk strategy with value-net leaves instead of 256 rollouts, or
   use unsafe resolving when the root ranges are the blueprint's own. The
   measured stake is about 330 mbb/hand of the searcher's exploitability.
4. **Bigger production trees.** With the turn-end net, a leaf costs one row, so
   20–50k-node trees, with the blueprint's full flop sizes, fit in a few
   seconds. Re-measure exploitability there.
5. **A turn-start net and batched turn solving.** This enables
   `depth_streets: 0` on the flop (smaller, faster flop trees) and value-net
   leaves for turn decisions.
6. **Head-to-head and ABR at scale.** A 2,000-hand match has a CI of about
   ±350 mbb/hand.
