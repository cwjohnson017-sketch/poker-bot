# Leaf value network for real-time search: design note

Goal: replace the noisy blueprint-rollout values at depth-limit leaves
(`search/leaf.py`) with a learned counterfactual value function, as in DeepStack
and ReBeL. The default flop search solves flop and turn (`depth_streets: 1`),
so its leaves sit at the **end of turn betting**, just before the river card.

## 1. Which leaves first: a river net, averaged over river cards

**River-start net `N_R`.** It predicts the counterfactual values at the
*start* of the river, before any river action. Its targets come from river
subgames solved exactly to showdown, with no leaves, so they are clean. A
turn-end leaf on board `b4` becomes an exact chance average over the 44
river cards that can still come for a given pair of hands:

    v_i(leaf)(c) = sum_{x not in b4} (1/44) * [c avoids x] * m^x_{-i}(c) * N_R(b4+x, r^x)_i(c)

Here `r^x_p = pi_p(leaf) * [avoids x]` are the reaches at the river root, and
`m^x_{-i}(c)` is the opponent's mass there that is disjoint from `c`. This is
the solver's own chance-node identity, so card removal stays exact and the
net never has to learn the river-card average.

Cost: one net row per (leaf, river card). That is about 15k rows for the
6000-node evaluation trees and about 90k for a full 20k-node flop tree.
**Fallback if that is too slow:** distil `N_R`'s chance average into a
turn-end net, one row per leaf. That needs no extra solving (the same idea as
DeepStack's auxiliary net). A turn-start net for `depth_streets: 0` is
bootstrapped the same way later.

## 2. Inputs and outputs

All river trees under the v4 abstraction have at most 240 nodes. They depend
only on `c`, the chips each player has put in, because bets are equal at the
river root.

* **Player order:** `(OOP, IP)`, i.e. (non-button, button). The integration
  permutes seats into this order.
* **Ranges:** both normalised to sum 1. They are bucketed into `K` (default
  256) **board-relative strength buckets**. Each combo's percentile on the
  5-card board is `(valid weaker + (ties - 1) / 2) / (valid - 1)`, and its
  bucket is `floor(K * percentile)`. Tied combos share a bucket. Percentile
  buckets put "the nuts" in bucket `K - 1` on every board.
* **Context:** pot and stack, as `c / (c + stack)` and
  `log(stack / pot)`, plus a 52-dim board one-hot.
* **Outputs:** per-combo expected values per unit of opponent mass,
  `ev_p(c) = v_p(c) / m_{-p}(c)`, divided by the pot `2c`, for both players.
  The trunk is an MLP over the bucket inputs, giving `2K` bucket values that
  are decoded to combos. An optional per-combo residual head with blocker
  features (opponent mass blocked by `c`, and strength-weighted mass blocked)
  is kept only if the measured "bucket oracle" error says bucketing loses too
  much.
* **Zero-sum layer** (DeepStack), at combo level. Let
  `w_p(c) = r_p(c) * m_{-p}(c)` and `Z = sum_c w_0(c) = sum_c w_1(c)` (the
  pair mass). Subtract `delta = (sum_c w_0 ev_0 + sum_c w_1 ev_1) / (2Z)`
  from every `ev_p(c)`. The range-weighted game values then sum to exactly
  zero. The layer is used in training and inference.
* **Targets:** the **best-response** values of each player against the
  other's average strategy after the solve. On range they equal the
  equilibrium values to within the exploitability. Off range (combos with zero
  reach) they are what a trunk deviation into the leaf would actually get, so
  they are the right values for regrets and for the gadget.

## 3. Data: batched exact river solves

* **`BatchRiverSolver`** (new). DCFR with the same update order and discounting
  as `RangeSolver`, run over `B` instances that share one river betting tree
  (same `c`) but have their own boards and ranges. Tensors are `[N, B, 1326]`.
  Regrets are indexed by child edge, not padded to `A`. Fold and showdown
  rows are `(node, instance)` pairs on the instance's board, using the
  existing `ShowdownTables` and `fold_values`. It has exact per-instance best
  responses and exploitability. Every sample stores its own exploitability,
  and samples above a threshold are flagged.
* **States:**
  * About half come from blueprint self-play (`neural:runs/dcfr4_distilled_v2`
    on `VecNLHE`, sampling actions for the dealt cards and multiplying each
    actor's 1326-combo reach by the policy). A hand is kept when it reaches
    the river root without an all-in. The rest are discarded.
  * The other half are perturbed copies of those ranges (log-normal per-combo
    noise, strength tilts, uniform mixing), plus DeepStack-style random
    ranges: recursive random splits of probability mass along the strength
    order, on random boards with log-uniform `c`.
  * Samples are sorted by `c` and chunked into batches. Each batch is solved at
    the batch's median `c`, which is fine because the ranges are only inputs.
    This gives hundreds of distinct pot sizes rather than a grid.
* **Storage:**
  * per sample: board, `c`, ranges and targets as fp16 `[2, 1326]`, the
    exploitability, and the source tag;
  * about 11 kB per sample, cached in shard files as in `distill.py`.
* **Size:** enough for a few hours of GPU time. The rate is measured first on
  a pilot of about 20k samples.

## 4. Integration (`leaf.mode: value_net`)

* **Tree.** `TreeConfig.leaf_mode = "value_net"` makes depth-limit leaves a new
  terminal kind, `VALUE`, with no continuation children. A
  `leaf_budget_cost` option lets the evaluation build a value-net tree with
  exactly the trunk abstraction of a rollout tree.
* **Solver.** `TerminalEvaluator` takes an optional leaf-value provider. It
  receives **both** players' current reaches at the `VALUE` nodes, because the
  net is nonlinear in both ranges. `RangeSolver` passes the full reach to the
  terminal evaluation. The net is evaluated on the current reaches at every
  update, or every `net_every` updates with cached `ev` re-weighted by the
  current opponent mass. All (leaf, river card) rows are batched in chunks in
  bf16, with per-board bucket tables built once per solve.
* **Rollouts.** The rollout path is unchanged and stays the default.

## 5. Validation: trunk exploitability with an exact river

The score must not use the net. For a trunk profile `sigma_T` (flop and turn
decisions of both players) on the evaluation tree:

1. Run a forward pass to get both players' reaches at every leaf.
2. For each (leaf, river card), solve the river subgame exactly
   (`BatchRiverSolver`) on both reaches, each mixed with 5% uniform. The
   mixing gives the defender a defined river strategy where the attacker's
   reach is zero.
3. For each player `p`, compute best-response values at every river root
   against that river strategy, with the opponent's **actual** reach.
   Chance-average them to the leaf.
4. Back the values up through the trunk, with `p` maximising and the opponent
   playing `sigma_T`.

The score is `(BR_0 + BR_1) / 2` in mbb/hand. This is the exact
exploitability, in the full flop-to-showdown game, of `sigma_T` followed by an
equilibrium river for the reached ranges. The same procedure applies to:

* value-net search;
* default rollout search;
* the blueprint (its strategy mapped onto the same trunk).

Candidates are mapped onto one shared trunk by the key (action history,
board). The 6 flop spots of `runs/search_noise` are evaluated, plus more
boards. Timings per decision are reported. Instances where both reaches are
negligible are skipped, with a stated error bound.
