# `pokerbot.search`: real-time depth-limited re-solving

Phase 3 of `DESIGN.md` (sections 5.3, 5.6). From the flop on, `SearchAgent`
re-solves the current street at every decision with range-vs-range CFR on
tensors, using a blueprint for the ranges, for the values at the depth limit
and for preflop play.

| Module | Contents |
|---|---|
| `combos.py` | canonical order of the 1326 hole combos (`combo_index`, `combo_cards`), card-conflict tables, `blocked_sum` |
| `abstract.py` | scalar abstract actions matching `pokerbot.env.actions` exactly, pseudo-harmonic mapping, `CardView`, `make_state` |
| `blueprint.py` | `Blueprint` protocol, `UniformBlueprint`, `TabularBlueprintFromCallable`, `policy_matrix`, `range_reach`, the `search:` registry |
| `adapters.py` | `TabularBlueprint` (MCCFR strategy file) and `NeuralBlueprint` (Deep CFR run), both with a batched `policy_combos` |
| `tree.py` | `TreeBuilder` / `build_tree` -> `SubgameTree` (flat tensors), node budget |
| `showdown.py` | O(n) range-vs-range showdown with card removal (`ShowdownTables`), fold kernel, dense reference |
| `leaf.py` | depth-limit leaf values from blueprint rollouts with `k` biased continuation strategies |
| `value_leaf.py` | depth-limit leaf values from a value net (`leaf.mode: value_net`): river net averaged over the river cards (`ValueLeafEvaluator`, `river_average`) or a turn-end net (`TurnEndLeafEvaluator`), `ShowdownOracle` |
| `value_net.py` | the river value net: board strength-percentile buckets, `RiverValueNet` (zero-sum output), `ValueNetPredictor` |
| `value_train.py` | value-net training data (shards), training loop, held-out report |
| `value_ranges.py` | river-root (or turn-end) states for training: blueprint self-play ranges on `VecNLHE`, perturbed and DeepStack-style random ranges |
| `value_data.py` | data generation: states batched by pot, solved exactly, written as shards (`scripts/gen_value_data.py`) |
| `turn_net.py` | the turn-end net (DeepStack's auxiliary net): turn-end buckets (`TurnFeatureCache`), `TurnEndPredictor`, `load_leaf_predictor`, `train_turn_net` |
| `turn_data.py` | turn-end data with targets bootstrapped from the river net, no solving (`scripts/gen_turn_data.py`) |
| `batch_solver.py` | `BatchRiverSolver`: DCFR on many river subgames sharing one betting tree, `river_tree` |
| `exact_eval.py` | exact exploitability of a flop/turn strategy with every river subgame solved (`trunk_exploitability`), `map_sigma` |
| `tree_policy.py` | the blueprint's strategy on every decision node of a tree (`blueprint_profile`), batched for distilled neural blueprints (`node_policies`) |
| `solver.py` | `RangeSolver`: DCFR / CFR+, alternating updates, exact best response and exploitability |
| `gadget.py` | safe resolving gadget, continual-resolving cache |
| `agent.py` | `SearchAgent`, `make_search_agent` (match runner: `search:<blueprint spec>`) |
| `config.py` | `SearchConfig` loaded from `configs/search_default.yaml` |

## Decision loop (`SearchAgent`)

1. Preflop: sample from the blueprint.
2. Postflop, at the start of each street (cached for later decisions on the
   same street): both ranges over the 1326 combos. If the previous street's
   solve reached this public state, take our reach, the opponent's reach and
   the opponent's best-response values from it (continual resolving);
   otherwise replay the observed history through the blueprint
   (`range_reach`), multiplying each actor's reach by the blueprint
   probability of the observed action (off-tree sizes via the
   pseudo-harmonic mapping). Combos that hit the board get zero reach, and
   combos holding our own cards are removed from the opponent's range
   (`remove_own_blockers`).
3. Build the tree rooted at the **start of the current street**, with this
   street's observed actions forced in. An observed size that is not an
   abstract action (an off-tree opponent bet, or our own size that the
   budget dropped) becomes an extra branch at the node where it happened. Our
   own earlier actions on this street are locked to the strategy we played
   (cached from that solve), as in Pluribus.
4. Solve within the street's time budget (tree building and rollouts count;
   at least `min_iterations` always run). With `gadget.safe` the opponent
   enters through the resolve gadget.
5. Read the average strategy of our actual combo at the current node, sample a
   child and play its concrete action. Cache what we played (for locks) and
   the values at every next-street root the game can still reach.

## Tree layout

`SubgameTree` holds one entry per node in flat tensors (BFS order, so node ids
are depth-sorted and a node's children are contiguous):

| Field | Meaning |
|---|---|
| `parent`, `slot`, `depth` | parent id (-1 at the root), position among the parent's children, depth |
| `kind` | `DECISION`, `CHANCE`, `FOLD_NODE`, `SHOWDOWN`, `LEAF`, `CONTINUATION`, `VALUE` |
| `actor` | player to act (decisions, and the leaf chooser at `LEAF`), else -1 |
| `street`, `contrib [N, 2]`, `bets [N, 2]` | street, chips committed this hand, chips committed this street |
| `board_id` | index into `boards` (the public board at the node) |
| `deal_card`, `chance_weight` | card dealt by the chance parent and its weight `1 / (52 - |board| - 4)` |
| `action_kind`, `action_amount`, `action_abstract` | concrete action into the node (`-1` abstract index = off-tree) |
| `first_child`, `num_children`, `children [N, max]` | child ranges |
| `cont`, `folder` | continuation strategy index, player who folded |
| `level_start` | node ids of each depth |
| `current_node`, `path_nodes` | the observed decision node, and the (node, slot) pairs along the observed path |
| `states`, `histories`, `boards` | engine state per decision / leaf node, concrete action history from the root, distinct boards (`board_id` indexes them) |

`states[n]` below a chance node is the chance **template's** engine state: its
betting is right, but its board has whatever card the builder dealt. Take a
node's board from `boards[board_id[n]]`, e.g. `CardView(states[n], board)`
for a blueprint query (`tree_policy.blueprint_profile` does).

Building happens in two steps. A recursive walk over scalar-engine states
(`poker_engine` or the reference engine) produces a skeleton. Betting never
depends on the cards, so a chance event keeps a single template subtree, and
the BFS expansion then copies it once per card. Depth limit: the current street
plus `depth_streets` more (default 1). Turn and river solves always run to
showdown. At the limit a `LEAF` sits where the next street's card would be
dealt. It is a decision of the leaf chooser (the searcher's opponent) among
`k` `CONTINUATION` children.

**Node budget** (`max_nodes`). The skeleton gives exact expanded counts
cheaply, so the builder reduces the abstraction until the count fits. It
first drops bet sizes deepest street first, farthest from pot-sized first,
down to one size per street. Then it lowers raise caps deepest-first, then
goes all-in only, and finally subsamples chance cards (`min_chance_cards`).
The final action sets are in `tree.street_actions` and `tree.raise_caps`.
`tree.summary()` reports the node count by kind.

## Solver

Ranges are `[2, 1326]` vectors. Regrets, current strategy and strategy sums
are `[D, A, 1326]` tensors (`D` decision nodes, player 0's first; `A` the
widest action set). So each decision node carries `[A, 1326]`, the transpose
of `[combos, actions]`, which lets `regret[node, action]` be one advanced
index. One iteration updates player 0 and then player 1 (alternating). Each
update does the following:

1. **Forward pass**, level by level: `pi[child] = pi[parent] * sigma[parent, a]`
   for the actor, and `* avoids(card)` for both players at chance nodes.
2. **Terminal values** of the updating player `i` against `pi_-i` (below).
3. **Backward pass**: `v[parent] = sum_a sigma(a) * v[child]` at `i`'s
   nodes, `sum_a v[child]` at the opponent's nodes, and
   `sum_x chance_weight * avoids(x) * v[child_x]` at chance nodes.
   Instantaneous regrets `v[child] - v[parent]` are accumulated at `i`'s
   nodes (`index_put_` with accumulate).
4. Strategy sum `+= pi_i * sigma`, then regret matching.

Values are counterfactual:
`v_i(n)(c) = sum_{c' disjoint from c} pi_-i(n)(c') E[u_i | c, c', n]`. A dealt
card `x` is disjoint from both hands with probability `1 / (52 - |board| - 4)`
for every pair, so masking the child's reaches by `x` and weighting by that
constant makes card removal exact (`test_chance_node_values_are_exact`).

* **DCFR** (default): `alpha = 1.5, beta = 0, gamma = 2`. After iteration `t`,
  positive regrets are multiplied by `t^a / (t^a + 1)`, negative ones by
  `t^b / (t^b + 1)`, and the strategy sum by `(t / (t + 1))^g`.
* **CFR+**: regrets are floored at 0 and the strategy sum is weighted by `t`
  (linear averaging).
* `exploitability()` runs exact best responses for both players over the
  whole tree (max per combo at the best responder's nodes) against the
  average strategy. It reports `(BR_0 + BR_1) / 2` in chips.

### Terminal kernels

* **Fold**: `+-stake * (opponent mass disjoint from c)`. The disjoint mass is
  `total - S[x1] - S[x2] + pi(c)`, with `S[x]` the mass of combos holding
  card `x` (one `[C, 52]` matmul).
* **Showdown, full board** (`showdown.py`): the sorted-strengths trick. Per
  board, sort the combos by strength once (`evaluate7_batch`). A prefix sum
  of `pi` in that order gives `W_all(c)`, the mass strictly below `c`'s tie
  group, and `L_all(c)`, the mass strictly above it. Card removal: every card
  lies in exactly 51 combos, and each card keeps its 51 combos sorted by
  strength. A prefix sum along each list gives the weaker and stronger mass
  holding that card. Then `W(c) = W_all - W_x1 - W_x2` (combo `c` itself is
  never strictly weaker than itself), likewise `L`, and `value = stake * (W - L)`.
  Each query is a few gathers and cumsums over `[rows, 1326]` and
  `[rows, 52 * 52]`, batched over rows on different boards. The dense
  `1326 x 1326` product exists only as `naive_showdown` for the tests.
* **All-in before the river**: a showdown averaged over the run-outs, all of
  them or `max_runouts` sampled with the matching unbiased scale. `dense`
  mode builds one `[1326, 1326]` equity-sign matrix per board (integer
  `G - G^T` counts) and applies it as one matmul for every all-in node on
  that board. `runouts` mode feeds one kernel row per (node, run-out).
  `auto` picks dense on CUDA, and on CPU for boards with at least
  `dense_min_nodes` all-in nodes.
* **Continuation** (depth-limit leaf): see below.

### Leaf values (`leaf.py`)

The leaf chooser picks one of `k = 4` continuations: the blueprint, or the
blueprint with fold, call or raise probabilities multiplied by `bias = 5`
and renormalised (Pluribus). The other player plays the plain blueprint. The
choice is a decision node for CFR, so per combo the chooser converges to the
best of the `k`. Each continuation's value is a linear operator on the
opponent's reach, estimated by importance-weighted rollouts. Each rollout:

* samples the remaining board uniformly;
* samples each rollout action from a combo-independent proposal `q`: the
  combo-averaged policy mixed with `explore` uniform;
* multiplies every combo's weight by `sigma(c, a) / q(a)`, separately per
  continuation for the chooser.

The final fold or showdown is evaluated with the kernels above. Each sampled
board is disjoint from any given pair of hands with the same probability, so
the scale `|All| / (R * N_k)` makes the estimate unbiased
(`test_leaf_rollouts_estimate_the_continuation_value`). Card-independent
blueprints share betting paths across leaves.

### Value-net leaves (`value_leaf.py`, `leaf.mode: value_net`)

Design: `docs/value_net.md`. `leaf.mode: value_net` (it sets
`tree.leaf_mode`) makes every depth-limit leaf a `VALUE` terminal: no
children, actor -1, its engine state kept in `tree.states`. Only turn-end
leaves on 4-card boards are supported, i.e. flop solves with
`depth_streets: 1` (a flop solve with `depth_streets: 0` would need a turn net
and raises `ValueError`). Turn and river solves have no leaves either way.

`ValueLeafEvaluator` values a leaf on board `b4` by the exact chance average
of a river-start net `N_R` over the 48 river cards, using the solver's
chance-node identity, so card removal stays exact:

    v_i(c) = sum_{x not in b4} (1/44) * [c avoids x] * m^x_-i(c) * pot * ev^x_i(c)

Here `ev^x = N_R(b4 + x, ranges, c, stack)` is in pot units per unit of
disjoint opponent mass, and the ranges are both reaches at the leaf masked by
`x`, in `(OOP, IP)` order. `m^x_-i` is the opponent's masked mass disjoint
from `c`. It is computed for all 48 cards from one per-card sum per leaf
(`m - S[x] + r({x, c1}) + r({x, c2})`), without a matmul per row.

* The net is nonlinear in both ranges. So `RangeSolver` passes the full
  `[2, N, 1326]` reach from its forward pass to `TerminalEvaluator`, which
  hands `reach[:, VALUE ids]` to the provider at every update (DeepStack
  style).
* There is one net row per (leaf, river card), leaf-major. They are batched
  in chunks of whole leaves, about 16k rows per `predict`.
* `leaf.net_every: n` runs the net on only every n-th regret update per
  player. The updates in between reuse the cached `ev` (fp16), re-weighted
  by the current opponent mass. Value, best-response and exploitability
  passes (and the continual-resolving cache) always run the net.
* `tree.leaf_budget_cost` sets how many nodes a leaf counts for in
  `max_nodes`: 1 by default for `VALUE`, 1 + k for `LEAF`. Setting it to
  `1 + k` gives a value-net tree exactly the trunk abstraction of the
  rollout tree.

The predictor comes from `SearchAgent(..., value_predictor=...)`, or is
loaded once per agent from `leaf.net` with
`turn_net.load_leaf_predictor(path, device)`: a river net
(`ValueNetPredictor`) or a turn-end net (`TurnEndPredictor`, below), by the
checkpoint's `meta` kind. The provider follows the predictor's `kind`
(`value_leaf.make_leaf_evaluator`). A river predictor's interface:

```python
predict(boards: LongTensor[n, 5], ranges: Tensor[n, 2, 1326],  # (OOP, IP), any positive scale
        c: Tensor[n], stack: Tensor[n]) -> Tensor[n, 2, 1326]  # ev in pot units, 0 on invalid combos
```

`ShowdownOracle` implements it exactly for a checked-down river. A turn solve
whose river is check-down reproduces the full solve to showdown
(`test_value_leaf.py`). `FixedLeafValues` returns precomputed leaf values.

**Cost on the RTX 4070 Ti.** fp32 reaches, chunks of 16k rows, measured with
a stand-in MLP (512-1024-512) as the net. The evaluator's own work per
`values()` call comes on top of the net:

| Leaves (net rows) | Evaluator work per call | Cached call (`net_every > 1`) |
|---|---|---|
| 343 (16k) | about 2 ms | 1.4 ms |
| 1,900 (91k) | about 7 ms | 6 ms |

The evaluator's work is writing the masked net input, one batched matmul and
a few gathers. The largest item is writing the `[rows, 2, 1326]` fp32 input:
968 MB at 1,900 leaves, about 2.8 ms. A net that accepted bf16 ranges would
halve it. One solver iteration makes two calls, one per player.

#### Turn-end net: one row per leaf (`turn_net.py`, `turn_data.py`)

DeepStack's auxiliary net. `river_average(predictor, boards4, c, stack,
reach)` is the chance average above without a tree (both players, from one
net pass). A turn-end net `N_TE` learns it directly, in the same units:

    ev_TE_p(c) = v_p(c) / (m_-p(c) * pot),   m_-p = blocked_sum(pi_-p) on b4

so `TurnEndLeafEvaluator` needs one net row per leaf:
`v_p = ev_TE_p * m_-p * pot`, with the same `ids`, `values()` and `net_every`
caching as `ValueLeafEvaluator`.

* **Targets** are bootstrapped from the river net: no solving, 48 river-net
  rows per sample (`turn_data.turn_targets`). With `ShowdownOracle` they are
  the exact check-down turn-end values.
* **States**: blueprint self-play stopped at the end of turn betting
  (`selfplay_river_states(..., turn_end=True)`: the river-root hands without
  their river card), perturbed copies, and random ranges along the turn-end
  strength order. Shards are river shards with `boards [n, 4]`, `exploit` 0
  and `kind: turn_end` in `meta.json`:
  `scripts/gen_turn_data.py --river-net <ckpt> --blueprint <run> --out <dir> --samples N`.
* **Features**: the model is `RiverValueNet` unchanged; only the buckets
  differ. Per combo, its river-strength ranks on the 46 river boards it can
  see give the mean (equity against a random hand) and the spread. 1-D:
  percentile buckets of the mean. 2-D (`spread_buckets = Ks`): `K / Ks` mean
  buckets, each split into `Ks` equal-count spread quantiles, so draws and
  made hands of equal equity separate. `TurnFeatureCache` keeps the ranks per
  turn board (48 river boards are evaluated once per new turn board).
* **Training**: `scripts/train_turn_net.py --data <dir> --out turn.pt
  --spread-buckets 8` (`turn_net.train_turn_net`). The checkpoint's meta has
  `kind: turn_end`, so an agent with `leaf.net: turn.pt` uses
  `TurnEndLeafEvaluator` (`last_stats["value_provider"]`).

**Cost.** Measured on this CPU (8 threads) on a 100 bb flop tree (5,490
nodes, 931 leaves on 49 turn boards), fp32 reaches, both nets the default
`ValueNetConfig` (4.3 M parameters, K = 256):

| Provider | Net rows per call | `values()` call | Without the net | Solver iteration |
|---|---|---|---|---|
| `ValueLeafEvaluator` (river net x 48) | 44,688 | 1,650 ms | 67 ms | 3.46 s |
| `TurnEndLeafEvaluator` | 931 | 36 ms | 4 ms | 0.24 s |

A 2,220-node tree (441 leaves): 785 ms against 18 ms per call. The first
call on new turn boards builds the feature tables (about 0.25 s for 49 turn
boards on this CPU). On the RTX 4070 Ti (estimated from FLOPs, not measured):
the net is about 8.5 MFLOP per row, so a river call at 931 leaves is about
380 GFLOP plus about 3 GB of memory traffic, roughly 15-20 ms with the
evaluator's own work, i.e. 30-40 ms of every iteration. The turn-end call is
about 8 GFLOP and 65 MB, so it is bound by kernel launches: roughly 0.5 ms per
call, 1 ms per iteration.

**1-D or 2-D buckets.** Held-out error on check-down targets (16k training
samples from `dcfr4_distilled_v2` self-play, perturbed and random states; a
separate 4k held-out run; K = 256, width 512, 4 layers, 2,000 steps, two
seeds that agree to 0.0002; pot units):

| Buckets | MAE | wMAE | Bucket oracle MAE |
|---|---|---|---|
| 256 (1-D) | 0.0300 | 0.0311 | 0.0041 |
| 64 x 4 | 0.0284 | 0.0293 | 0.0040 |
| 32 x 8 | 0.0275 | 0.0285 | 0.0037 |

(zero prediction: MAE 0.250). 32 x 8 is 8% better than 1-D on every source
(self-play, perturbed, random), so `--spread-buckets 8` is the default. At
6,000 steps, which overfit the 16k samples, the order is the same: 0.0312
(1-D), 0.0292 (16 x 16), 0.0283 (32 x 8). The nets are far above the bucket
oracle, so more data matters more than the bucketing.

## Safe resolving gadget (`gadget.py`)

This is the CFR-D resolve gadget (Burch, Johanson & Bowling 2014), as used by
DeepStack's continual re-solving. For each combo `c'`, the opponent first
chooses between two options:

* **terminate**: take `T(c')`, the counterfactual value they could already
  secure against how we have played;
* **enter**: play the subgame, with root reach `prior(c') * sigma_enter(c')`.

`sigma_enter` is learned by regret matching inside the solve. At a solution,
`sum_c' prior(c') * max(0, BR_enter(c') - T(c'))` is close to zero, so the
re-solved strategy is not more exploitable than the one it refines. The
tests check it drops from 52 to 0.015 chips on a 200 pot.

* `T` and the subgame values are both weighted by our root reach vector, so
  they must come from the same range. After every solve, `ContinualCache`
  stores our reach, the opponent's reach and the opponent's
  **best-response** values at every chance child under the action we take.
  The next street starts from exactly those vectors.
* Without a cached solve (the first decision of a street, e.g. the first flop
  decision), `gadget.terminate` says where `T` comes from. It is computed
  once per street root and reused by later decisions on that street.
  * `rollouts` (default): the opponent's blueprint-vs-blueprint value from
    `gadget.rollouts` rollouts.
  * `blueprint`: the opponent's counterfactual values in the search tree itself,
    with its own leaves, when both players play the blueprint
    (`tree_policy.blueprint_profile`, then `gadget.tree_terminate_values`).
    It has no rollout noise and comes from the same game as the entry values.
  * `blueprint_br`: the same with the opponent best-responding to the
    blueprint. This is CFR-D's `T`: the re-solve is then no more exploitable
    than the blueprint in the search's game.
  * `unsafe`: no gadget at such decisions. The root ranges there are the
    blueprint's own, the case unsafe resolving assumes.

  `last_stats["gadget"]` records the source (`cache` after a solved earlier
  street), and `terminate_seconds` the time spent on `T`.
* `prior` is the opponent's range mixed with `prior_mix` uniform.
* `safe: false` (unsafe resolving) skips the gadget: the opponent's root
  range is the cached or blueprint range.

## Time budget

Defaults (`configs/search_default.yaml`) are 2 s per flop decision and 1 s on
the turn and river, `max_nodes: 20000`, `min_iterations: 10`.

**Measured on this 4-core CPU container** (2 torch threads, other jobs
sharing the cores, so treat these as rough). A flop subgame at 200 bb after
a 2.5x open and call, with the default config and the uniform blueprint:

| Step | Result |
|---|---|
| Tree | 19,602 nodes (4,428 decision, 3,544 fold, 2,066 showdown, 1,911 depth-limit leaves with 7,644 continuations), built in 0.25 s |
| Setup | 7.3 s: 50 dense all-in matrices about 6 s, rollouts about 1.4 s |
| Iterations | about 2 s each |

River subgames of a few dozen nodes run at about 5-10 ms per iteration.

**RTX 4070 Ti guidance.** These are estimates: nothing here ran on a GPU.

* **Per iteration.** A 20k-node flop tree holds `[2, N, 1326]` reach,
  `[N, 1326]` values and three `[D, A, 1326]` tables, about 1 GB. One
  iteration moves a few GB of memory, runs about 15k rollout kernel rows and
  about 7 GFLOP of all-in matmuls. Expect roughly 15-30 ms per iteration,
  so about 50-100 iterations in the 2 s flop budget. DCFR is usually good
  enough after 100-200.
* **Setup.** The dense all-in matrices are cheap on the GPU (about 4 G
  integer compares per flop solve). The Python side is not: rollout
  generation, gadget rollouts and blueprint queries.
* **With a real blueprint**, rollouts query it for every combo at every
  rollout node (see the measurement below). Both adapters implement a
  vectorised `policy_combos`; the tabular one is then dominated by bucketing
  the fresh turn and river boards of the rollouts, which is CPU work in Rust
  that a GPU does not speed up.
* **If a decision runs over budget**, lower these in order:
  `leaf.max_total_rollouts`, `tree.max_nodes` (10k is a good flop value),
  `solver.max_runouts`, `tree.chance_cards`.
* **Turn and river** trees are much smaller (a turn tree at 20k nodes has
  no leaves). The 1 s budget there is mostly iterations.

### With the tabular blueprint (`search:blueprint:<strategy file>`)

Measured by `tests/search/test_adapters.py::test_flop_decision_timing_tabular_blueprint`
(marked `slow`; run with `-s`): `configs/mccfr_small.yaml` trained for 20k
iterations, a 100 bb flop after a 2.5x open and call, default search config,
CPU, `min_iterations: 1`, on this shared 4-core container.

| Step | Result |
|---|---|
| Tree | 14,604 nodes, 1,323 depth-limit leaves, 0.1 s |
| Leaf rollouts | 33 s: 10,584 rollout rows (2 rollouts x 4 continuations per leaf), **25 ms per leaf** |
| Blueprint queries | 7,084 `policy_combos` calls, 17.8 s, 2.5 ms per call on average |
| All-in matrices, gadget rollouts | about 6 s and 2 s |
| Iterations | 1.5 s each |
| Decision total | 43 s (against a 2 s budget) |

Where the time goes (cProfile of the same decision): the leaf rollouts. In
them, `CardAbstraction.buckets_batch` for every new turn or river runout
board is about 14 s (about 8 ms per cold board of 1,326 combos; a board seen
before is a cached lookup, and a query on it costs 0.2 to 0.7 ms), the rollout
loop itself about 12 s of Python and small-tensor overhead, and
`combos.valid_mask` on each rollout's full board about 5.5 s. So the
bottleneck is the per-rollout work on fresh boards, not the lookups. To fit
the budget: lower `leaf.max_total_rollouts` (the time is linear in it),
`tree.max_nodes` (fewer leaves), or precompute turn and river bucket tables
(`cards.tables`) so bucketing is a table lookup. A neural blueprint costs
about 23 ms per `policy_combos` call with the tiny test network (one forward
pass per net over 1,326 rows, plus the cached own-reach passes), so leaf
rollouts with it would take minutes on this CPU; it needs the GPU and a small
`max_total_rollouts`.

## Plugging in a blueprint

The two trained blueprints plug in from the registry:

```
python scripts/play_match.py --a search:blueprint:runs/mccfr_small/strategy.bin --b equity
python scripts/play_match.py --a search:neural:runs/deepcfr_tiny --b equity
```

* `search:blueprint:<strategy file>` -> `TabularBlueprint`
  (`pokerbot.blueprint.mccfr.policy.TabularPolicy`). The public history is
  replayed into the training game with the pseudo-harmonic translation
  (`u = 0.5`, cached per history prefix). `policy_combos` buckets all 1,326
  combos of the board with one `CardAbstraction.buckets_batch` call (built
  from the strategy's own card config, cached per board) and looks up each
  distinct bucket's infoset once. `spec` is the training action list.
  Actions that are illegal at the real stacks move their mass to check/call,
  and when the abstract game stops tracking the hand the row is check/call,
  as in `BlueprintAgent`.
* `search:neural:<run dir>` -> `NeuralBlueprint`
  (`pokerbot.blueprint.deepcfr.range_policy.NeuralRangePolicy`).
  `policy_combos` encodes the public state once, writes each combo's hole
  cards into the batch and runs one forward pass per net. The SD-CFR own-reach
  weights come from one pass per earlier own decision, cached per history
  prefix and board prefix.
* Rows of combos that share a card with the board are zero. Per-agent
  options: `blueprint: {...}` goes to the blueprint factory (e.g.
  `{last_n: 8}` for a neural run), the rest to the search config.


Any other blueprint: anything with `spec` (an `ActionSpec`) and `policy(state, player)` works.
`policy` returns `{abstract_index: prob}`, a length-`A` sequence or a
tensor, for the actor at a scalar `GameState`. It reads the actor's cards
with `state.hole_cards(player)`. The search may pass a `CardView`: a
read-only wrapper with the board and hole cards replaced. It supports
`board`, `hole_cards`, `history`, `street`, `pot`, `street_bets`, `stacks`,
`legal_actions()`, `public_key()` and `infoset_key()`, and delegates the
rest.

Optional extras:

* `policy_combos(state, player) -> Tensor[1326, A]` for every combo at once.
  This matters for speed: without it, the search calls `policy` once per
  combo.
* `card_independent = True` when the policy ignores cards.

```python
from pokerbot.search import SearchAgent, register_blueprint, search_config

class MyBlueprint:
    def __init__(self, path):
        self.spec = ...                     # the ActionSpec its indices refer to
    def policy(self, state, player):        # {abstract index: prob} for state.hole_cards(player)
        ...
    def policy_combos(self, state, player): # optional: [1326, A] tensor
        ...

register_blueprint("mine", lambda arg, **kw: MyBlueprint(arg))   # -> search:mine:<path>
agent = SearchAgent(MyBlueprint("runs/x"), search_config())
```

From the match runner: `python scripts/play_match.py --a search:uniform --b random`.
Per-agent options go under `agents: {"search:uniform": {...}}` in the match
YAML, for example `config: configs/search_default.yaml`, `time_budget: 1.0`
or `tree: {max_nodes: 5000}`. `TabularBlueprintFromCallable(fn, spec)` wraps
any `fn(state, player)`.

## Tests

```
python -m pytest tests/search -q        # add -s for the CPU timing line
```

* (a) `test_showdown.py`: the O(n) showdown and fold kernels against the
  dense product on random and tie-heavy boards.
* (b) `test_solver.py`: exploitability falls to under 1% of the pot on a
  river subgame (DCFR and CFR+) and falls on a turn subgame with chance
  nodes. It also checks that chance values are exact and that the dense and
  run-out all-in modes agree.
* (c) `test_tree.py`: node budget, off-tree branch at the right node, layout
  invariants, every river card enumerated from a turn root.
* (d) `test_gadget.py`: the safe gadget's opponent entry values never exceed
  the terminate values, and the leaf-rollout estimate is unbiased.
* (e) `test_agent.py`: 100 legal hands against `RandomAgent` through the
  masked match runner, unsafe mode with cached roots, and the
  `search:uniform` registry.
* (f) `test_timing.py`: prints the CPU flop timing above.
* `test_adapters.py`: batched `policy_combos` equals per-combo `policy` for
  both trained blueprints (tabular on and off the training stacks), zero
  mass on card conflicts, the tabular buckets equal the strategy's own,
  `search:blueprint:` and `search:neural:` agents play 20 legal hands, and
  the (slow) tabular timing above.
* `test_abstract.py`: scalar abstract actions against `pokerbot.env.actions`,
  pseudo-harmonic mapping, `CardView`, `range_reach`.
* `test_value_leaf.py`: covers the following.
  * Value-net leaves with `ShowdownOracle` match a turn solve to showdown
    with a checked-down river: values, best responses, exploitability and
    every turn strategy.
  * The leaf values match the dense showdown enumerated over river cards,
    and the per-card identity for the masked opponent masses holds.
  * `net_every`, a flop value-net solve, and `leaf_budget_cost` reproducing
    the rollout tree under a binding budget.
  * A value-net agent plays legal hands.
* `test_turn_net.py` (turn-end net, about 15 s on CPU): covers the following.
  * `river_average` equals `ValueLeafEvaluator.values` on a flop tree (exact
    check-down oracle and a nonlinear toy river net; any leaf order).
  * Turn-end targets bootstrapped from `ShowdownOracle` equal the dense
    check-down enumeration over river cards and `BatchRiverSolver` on a
    check-down river.
  * Turn-end features match their definition (1-D and 2-D buckets), the
    feature cache, `TurnEndPredictor` and checkpoint kinds.
  * A small turn-end net learns check-down targets (held-out MAE about 15% of
    the zero baseline).
  * `TurnEndLeafEvaluator` with an exact turn-end predictor reproduces the
    river evaluator's leaf values and solve, and `net_every`.
  * The agent picks the provider from the checkpoint kind and plays; turn-end
    self-play states; the data and training CLIs.
* `test_tree_policy.py`: covers the following.
  * The batched blueprint profile equals per-node `policy_matrix` queries on
    every combo that can reach a node, and the fallback path for other
    blueprints.
  * Turn nodes are queried on their own board, not the chance template's.
  * Terminate values computed in the tree; each `gadget.terminate` mode in the
    agent.
  * A `blueprint_br` re-solve keeps every opponent combo below its terminate
    value.

## Shortcuts and limits

* One leaf chooser (the searcher's opponent), not both players as in
  Pluribus. The searcher's continuation is the plain blueprint.
* Ranges at the first flop root come from the blueprint, not from our
  actual preflop play. Later streets use the cached solve.
* By default the first gadget terminate values are blueprint-vs-blueprint
  rollout values, not a best response to the blueprint, so they are noisy.
  `gadget.terminate: blueprint | blueprint_br` computes them in the search
  tree instead (see "Safe resolving gadget").
* Sampled run-outs (`max_runouts`, rollouts) and subsampled chance cards
  give an unbiased but noisy game. The solver treats that sampled game as
  exact.
* Rollout generation and blueprint queries are Python loops on the CPU.
* The tabular adapter's history replay uses the deterministic pseudo-harmonic
  split (`u = 0.5`), not the randomized one `BlueprintAgent.act` uses.
