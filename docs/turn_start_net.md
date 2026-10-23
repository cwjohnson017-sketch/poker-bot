# Turn-start value net: design notes and measurements

Goal: let flop search stop at the **end of flop betting** (`tree.depth_streets: 0`)
instead of at the end of the turn. Its `VALUE` leaves sit on 3-card boards and are
valued by a **turn-start net** `N_TS`, which predicts the counterfactual values at the
turn root (after the turn card, before any turn betting). As in DeepStack, `N_TS` is
bootstrapped from the next net: its targets are turn subgames solved with
`VALUE` leaves at the end of turn betting, valued by the existing turn-end net
`N_TE` (`runs/value_net/turn_v2w.pt`), which was itself bootstrapped from the river net.

Status (2026-10-08): trained on GPU (`runs/value_net/turn_start_v1.pt`, 200k samples from
`turn_v3w` leaves, held-out MAE 0.039 pot). Depth-0 flop search runs about 210 iterations in
2 s, but its flop decisions are 131 mbb/hand more exploitable than depth-1 search on 3 spots,
so it is not used in production. Results: `docs/value_net_eval.md`, Round 2, section 6.

| Piece | Module | Tests |
|---|---|---|
| Batched turn solves | `search/batch_turn_solver.py`: `BatchTurnSolver`, `turn_tree`, `allin_turn_matrix` | `tests/search/test_batch_turn_solver.py` |
| Turn-root states | `value_ranges.selfplay_river_states(turn_start=True)`, `turn_data.make_turn_states(turn_start=True)` | `tests/search/test_value_ranges.py` |
| Data | `search/turn_start_data.py`, `scripts/gen_turn_start_data.py` | `tests/search/test_turn_start_net.py` |
| Net | `turn_net.TurnStartPredictor`, `train_turn_net(kind="turn_start")`, `scripts/train_turn_net.py --kind turn_start` | same |
| Flop-end leaves | `value_leaf.FlopEndLeafEvaluator`, `flop_end_leaves`, `make_leaf_evaluator` | same |
| Agent | `tree.depth_streets_turn`, `leaf.turn_net`, `SearchAgent(turn_value_predictor=...)`, `configs/search_turn_start.yaml` | same |

## 1. `BatchTurnSolver`

DCFR over `B` turn subgames that share one turn betting tree. Each instance has its
own 4-card board and ranges.

* **Tree** (`turn_tree(game_config, c, spec, button=1)`): the turn root of a hand in
  which both players committed `c` chips, built by `TreeBuilder` with
  `depth_streets: 0` and `leaf_mode: value_net`, as `river_tree` does for the river.
  It has decisions, folds, all-in showdowns, and `VALUE` leaves where turn betting
  ends with chips behind. The tree depends only on `c`, the button and the spec.
  With the dcfr4 blueprint spec at 100bb:

  | `c` | Nodes | `VALUE` leaves | All-in showdowns |
  |---:|---:|---:|---:|
  | 100 | 159 | 31 | 22 |
  | 250 | 183 | 35 | 26 |
  | 1,500 | 105 | 17 | 18 |
  | 8,000 | 9 | 1 | 2 |

* **Algorithm**: `BatchRiverSolver`'s, by subclassing. It has the same alternating
  DCFR updates, regrets on edges and segments of equal fan-out, and the same
  float32 discount factors. It runs on the 1,128 combos disjoint from a 4-card
  board. `BatchRiverSolver` gained a class-level `board_len` / `num_valid`, a
  `_kernels` hook and an optional showdown matrix; the river behaviour is
  unchanged.
* **Leaves**: at the start of every backward pass, the updating player's leaf
  values come from the turn-end predictor. It is evaluated on both players'
  current reaches, with one `predict` row per (leaf, instance), batched in chunks
  of whole leaves:
  `v_p = ev_p * blocked_sum(r_-p) * 2c_leaf`. This is exactly where `RangeSolver`
  calls `TurnEndLeafEvaluator`.
  * `leaf_every > 1` caches `ev` per player, as `leaf.net_every` does.
  * Root values, best responses and exploitability evaluate the leaves on both
    players' reaches under the scored strategy. The best responder's own reach is
    taken under the average strategy, as in `RangeSolver.values`.
  * A river predictor (e.g. `ShowdownOracle`) is wrapped in `RiverAveragePredictor`.
* **All-ins**: one exact matrix per instance serves every all-in node of the tree:

      E_b[c', c] = (1/44) * sum_{x not in b4} [c, c' avoid x] * sign(s_x(c) - s_x(c'))

  It is `RangeSolver`'s dense all-in matrix on the compact layout (the test checks
  this). Each update applies it as `B` batched matmuls, one per instance. It is
  built once per solve from `48 * 1128^2` integer compares per instance, in int16,
  and skipped when the tree has no all-in node.
  * Memory: 5.09 MB per instance in fp32.
  * The alternative, one sorted-strength kernel row per (all-in node, instance,
    river), was rejected. At the blueprint spec's ~26 all-in nodes it would need
    about 1,250 kernel rows per instance per update, each doing a few dozen passes
    over 1,326 combos: an order of magnitude more memory traffic than reading
    `E_b` once.
* **No CUDA graph.** A predictor call does host-side board-cache lookups.

**Verified** (CPU, float64):

* A batch of 3 instances, including a tie-heavy quad-deuce board, matches one
  `RangeSolver` per instance on the same turn tree. The reference uses
  `TurnEndLeafEvaluator(RiverAveragePredictor(ShowdownOracle()))` leaves and
  dense all-ins. The average strategy matches to 5e-13, and root values (plain
  and best response, both players) to 5e-15 relative. Exploitability and other
  root ranges also match.
* `E_b` equals `solver.allin_matrix` and a naive river enumeration. An all-in
  node's value equals the enumerated river showdown.
* Exploitability falls: 43%, 12% and 0.9% of the pot at 3, 10 and 40 iterations
  (check-down leaves).
* `leaf_every` call counts, the river-predictor wrapping, empty ranges, and a real
  `TurnEndPredictor` through its board cache are also tested.

**CPU measurements**:

* Setup: dcfr4 spec at `c = 250` (183 nodes, 35 leaves, 26 all-in nodes), B = 64,
  float32, 24 threads.
* Tree passes and all-ins: 86 ms per iteration, with a zero predictor.
* With `turn_v2w.pt` on the CPU: 240 ms per iteration. That is 2 net calls of
  2,240 rows, about 77 ms each.
* Setup including the all-in matrices: 1.4 s.
* Memory per instance: tree tensors (regret, strategy sum, sigma, reach, values)
  4.13 MB, all-in matrix 5.09 MB, fold incidence 0.47 MB. Total 9.7 MB.

**GPU extrapolation (RTX 4070 Ti, 12 GB; not measured)**:

* *Memory at B = 512.*
  * Instances: 9.7 MB x 512 = 5.0 GB.
  * Net input and output per call: `[17,920, 2, 1326]` fp32, about 0.4 GB.
  * Net activations in 8,192-row chunks: a few hundred MB.
  * Total: about 6 GB. It fits. Use B = 256 if the GPU is shared.
* *Tree passes.* The structure is `BatchRiverSolver`'s with a similar tree (183
  vs at most 240 nodes) and combo count (1,128 vs 1,081). That solver measured
  about 46 ms per iteration at B = 512 (`c = 250`, with its CUDA graph). This
  solver has no graph, and its all-in matmuls read 5 MB per instance per update.
  Expect about 45-60 ms here.
* *Net.* `turn_v2w` costs 29.6 MFLOP per row.
  * Per iteration: 2 x 35 x 512 = 35,840 rows, so 1.06 TFLOP in bf16.
  * At 60-80 TFLOP/s effective that is 13-18 ms, plus about 2 ms to write the
    net input.
* *Total.* About 60-80 ms per iteration at B = 512. At 300 iterations that is
  about 20-25 s per 512 samples, i.e. **about 20-25 samples/s** before self-play
  state generation, so roughly 2.5-3 hours for 200k samples. `--leaf-every 2`
  would save only the net's share (about a quarter) and converges worse (below),
  so it stays at 1.

**Convergence with `turn_v2w` leaves** (CPU, 16 DeepStack random states):

| Iterations | `c = 250`: mean exploit / pot | max | `c = 1000`: mean | max |
|---:|---:|---:|---:|---:|
| 25 | 0.102 | 0.225 | 0.044 | 0.103 |
| 50 | 0.047 | 0.099 | 0.019 | 0.046 |
| 100 | 0.025 | 0.045 | 0.011 | 0.028 |
| 200 | 0.013 | 0.030 | 0.006 | 0.018 |
| 400 | 0.009 | 0.027 | 0.005 | 0.012 |

This is slower than the river (0.13% of the pot at 400 iterations), and it
flattens after about 200 iterations. The net's leaf values are nonlinear in both
reaches, so the "game" moves while CFR runs. The exploitability is measured with
the leaves frozen at the average profile. Every sample stores its own
exploitability, so `--max-exploit` in training can drop the worst. 300
iterations is the recommended setting.

Reusing the cached leaf `ev` (`leaf_every`, CPU, 16 random states at `c = 400`;
mean exploitability / pot, max in brackets):

| `leaf_every` | 100 iterations | 200 | 300 |
|---:|---:|---:|---:|
| 1 | 0.016 (0.046) | 0.009 (0.022) | 0.007 (0.022) |
| 2 | 0.024 (0.044) | 0.014 (0.028) | 0.012 (0.026) |
| 4 | 0.075 (0.221) | 0.031 (0.061) | 0.021 (0.052) |

## 2. Turn-start data

* **States**: as for the turn-end data, but stopped at the turn root, mixed
  50/25/25 (`make_turn_states(..., turn_start=True)`):
  * blueprint self-play stopped at the turn root
    (`selfplay_river_states(turn_start=True)`, lockstep and scalar paths, tested
    against `range_reach`);
  * perturbed copies of more self-play ranges;
  * DeepStack random ranges along the turn strength order, on random 4-card boards
    with log-uniform `c`.
* **Targets**: `solve_turn_batch`.
  * States are sorted by `c` and batched. Each batch is solved at one `c`, drawn
    log-uniformly within the batch and snapped to a reachable amount
    (`value_data.make_batches`).
  * The solve is `BatchTurnSolver` with turn-end-net leaves.
  * The targets are the root best-response values of both players,
    `ev = v / (m_opp * 2c)`, in `(OOP, IP)` order. They are zero on board
    conflicts and where `m_opp <= 1e-9`.
  * The exploitability is stored per sample, in pot units.
* **Shards**: the river format with `boards [n, 4]`. `meta.json` has
  `kind: turn_start`, the turn-end net path, the spec and per-shard timing and
  exploitability. Shards are seeded per `(seed, shard)` and resumable.
* **Turn tree spec**: the blueprint's own spec, with no sizes dropped. The turn
  tree is at most ~183 nodes and cheap, so dropping sizes would only make the
  targets less like the real game. At play time the same abstraction is used
  when `depth_streets_turn: 0` (a turn-root tree with blueprint actions, no budget
  reduction). With `depth_streets_turn: 1` the 6,000-node budget drops turn and
  river sizes, so the net's values assume a richer turn game than the one the
  turn search then plays.
* **Check-down sanity**: with a passive spec the turn tree is check, check, leaf.
  The pipeline's targets then equal `turn_targets(ShowdownOracle())` (the exact
  check-down values). A small net trained on such data with `kind=turn_start`
  reaches a held-out MAE under 0.35 x the zero baseline.

## 3. Net and checkpoints

The model is `RiverValueNet` with turn features: `TurnFeatureCache` (32 x 8
mean/spread buckets by default) on the same 4-card board. `TurnStartPredictor` is
`TurnEndPredictor` with `kind = "turn_start"`.

* `save_turn_net(..., kind="turn_start")`, `turn_meta(..., kind=)`.
* `train_turn_net(..., kind="turn_start")`, or
  `scripts/train_turn_net.py --kind turn_start`.
* `load_leaf_predictor` loads the new kind.
* Training rejects shard directories whose `meta.json` names another kind.

## 4. Flop-end leaves and the agent

`FlopEndLeafEvaluator` values `VALUE` leaves at the end of flop betting (3-card
boards) by the exact chance average of the turn-start net over the 49 turn cards:

    v_i(c) = sum_{t not in b3} (1/45) * [c avoids t] * m^t_-i(c) * pot * ev^t_i(c)

* It is `RiverAverage` generalised by a class attribute `board_len` (4 for the
  river average, 3 here). The cards per board (`52 - board_len`) and the chance
  weight `1 / (52 - board_len - 4)` are derived from it. Existing classes behave
  exactly as before.
* `make_leaf_evaluator` picks it for `kind == "turn_start"`. A turn-start net on
  turn-end leaves raises, and so does a turn-end net on flop-end leaves; both say
  what to set.
* **Verified**:
  * With an exact check-down turn-start oracle, the leaf values equal the dense
    all-in enumeration over all 1,176 turn-river run-outs (`allin_matrix`).
  * A `depth_streets: 0` flop solve with a nonlinear toy turn-start predictor
    `P` matches a flop + checked-down-turn solve. That second solve deals all 49
    turn cards by chance nodes and uses `P` as its turn-end predictor. Values,
    best responses and exploitability match to 1e-10, and every flop strategy to
    1e-9, once that tree's float32 chance weights are replaced by the exact
    1/45.
* **Cost**: a 100bb flop tree after a 2.5x open, dcfr4 spec, `depth_streets: 0`.
  It has 213 nodes and 41 leaves, so 2,009 net rows per call (vs 5,490 nodes and
  931 rows for the `depth_streets: 1` evaluation tree).
  * CPU with `turn_v2w`'s architecture (8 threads on a CPU shared with other
    jobs): 180-240 ms per `values()` call and 380-580 ms per iteration. The
    feature tables of the 49 turn boards take 0.6-1 s on the first call.
  * GPU estimate: 59 GFLOP per call, about 1-2 ms, so a few ms per iteration.

**Agent/config** (defaults unchanged):

* `tree.depth_streets_turn` (`TreeConfig`): the depth of trees rooted on the turn.
  `None` (the default) means `depth_streets`. 0 gives turn solves `VALUE` leaves
  at the end of turn betting.
* `leaf.turn_net` (`LeafConfig`): a second checkpoint for trees rooted on the turn
  (a turn-end or river net). It is loaded lazily, or passed as
  `SearchAgent(turn_value_predictor=...)`. Without it `leaf.net` serves every tree
  (today's behaviour).
* Before the first search, the agent checks a turn-start `leaf.net`. It requires
  `depth_streets: 0`, and a turn-end `leaf.turn_net` when turn solves have leaves
  (`depth_streets_turn` 0, or unset with `depth_streets` 0). Errors are raised
  rather than hidden by `fallback_on_error`. Other setups are not checked, as
  before.
* `configs/search_turn_start.yaml`: `depth_streets: 0`, `depth_streets_turn: 1`
  (the turn solves to showdown, as today), `leaf.net` = the turn-start
  checkpoint. A commented `leaf.turn_net` is for `depth_streets_turn: 0`.

## 5. Commands (GPU; not run yet)

From a checkout with this code, `PYTHONPATH=.`, the value-net artifacts in
`V = poker-bot-search/runs/value_net`:

```
# pilot: rate and exploitability, then the data and a held-out set
python scripts/gen_turn_start_data.py --turn-net %V%/turn_v2w.pt --blueprint %BP% \
    --out %V%/turn_start_pilot --samples 4096 --iterations 300 --batch 256
python scripts/gen_turn_start_data.py --turn-net %V%/turn_v2w.pt --blueprint %BP% \
    --out %V%/turn_start_a --samples 200000 --iterations 300 --batch 256 --resume
python scripts/gen_turn_start_data.py --turn-net %V%/turn_v2w.pt --blueprint %BP% \
    --out %V%/turn_start_heldout --samples 16384 --iterations 300 --batch 256 --seed 999
# the net (turn_v2w's size)
python scripts/train_turn_net.py --kind turn_start --data %V%/turn_start_a \
    --heldout-data %V%/turn_start_heldout --out %V%/turn_start_v1.pt --steps 40000 --width 2048
# flop decisions with depth_streets 0
python scripts/time_flop_search.py --blueprint %BP% --config configs/search_turn_start.yaml \
    --value-net %V%/turn_start_v1.pt --repeats 2
```

`BP = C:/Users/Connor/Documents/Poker-Bot/poker-bot/runs/dcfr4_distilled_v2`.
`--batch 512` if the GPU is not shared.

## 6. Doubts and limits

* **Bootstrapped error compounds.** `N_TS` learns targets valued by `N_TE`, whose
  on-policy exact MAE is 0.042 pot. That error now enters through every turn
  leaf, and `N_TS` adds its own fit error on top. Measure it before trusting it:
  run `check_turn_leaves`-style exact checks at turn roots, and the 6-spot
  trunk exploitability of `depth_streets: 0` against `depth_streets: 1`.
* **Convergence floor.** With net leaves the turn solves flatten at about 1% of
  the pot (random ranges, 400 iterations). This is far below the net's own error,
  but the per-sample `exploit` should be used to filter.
* **Continual resolving.** A `depth_streets: 0` flop tree, or a
  `depth_streets_turn: 0` turn tree, has no chance nodes, so `ContinualCache`
  has no chance children to store for the next street.
  * **Turn-end leaves with a river net: stored** (2026-10-08). When the turn
    tree's leaves are valued by a river net averaged over the 48 river cards
    (`ValueLeafEvaluator`: `leaf.turn_net` a river net, or `leaf.net` one with
    `leaf.turn_net` unset), the cache stores the river roots below the leaves
    under our action. Each (leaf, river card `x`) gets the key a chance child
    would have, both leaf reaches times `[avoids x]`, and the opponent's value
    `[c avoids x] * m^x_us(c) * pot * ev^x_opp(c)` in chips: the terms of the
    leaf's river average (`RiverAverage.card_values`). The river search starts
    from them as after a turn solved to showdown (`gadget: cache`). With a
    checked-down river and `ShowdownOracle` the entries equal those of the
    showdown tree (`tests/search/test_river_cache.py`). See "Safe resolving
    gadget" in `pokerbot/search/README.md` for the definition and the cost.
  * **Turn-end nets: not stored.** `TurnEndLeafEvaluator` (e.g. `leaf.turn_net:
    turn_v2w.pt`) predicts only the average over the river cards, so there is
    no per-card value. The river starts from the blueprint ranges, with the
    gadget's terminate values from `gadget.terminate` (256 blueprint rollouts
    by default, none with `unsafe`): the noisy first-decision path.
    `last_stats["cache_skipped_leaves"]` counts the leaves this affects.
  * **Flop-end leaves: not stored.** The turn after a `depth_streets: 0` flop
    solve starts the same way. `FlopEndLeafEvaluator` is a `RiverAverage` too,
    so `card_values` already gives its per-turn-card values (49 cards, weight
    `1 / 45`; checked once by hand, not in the tests). Storing them would mean
    allowing flop-rooted trees in `gadget.card_value_provider`, with the
    turn-start net's value at the turn root as `T`. Not done.
* **Flop all-ins** stay as before: a sampled 48 of the 1,176 run-outs
  (`solver.max_runouts`), unbiased but noisy. Raise `max_runouts` if all-in
  nodes matter; flop trees are now small.
* **Turn tree abstraction mismatch** with `depth_streets_turn: 1` (section 2).
  Running the turn with `depth_streets_turn: 0` and `leaf.turn_net:
  turn_v2w.pt` makes play and data consistent, and is faster.
