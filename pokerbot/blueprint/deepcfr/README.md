# `pokerbot.blueprint.deepcfr`: neural blueprint (Deep CFR, SD-CFR average)

Phase 2 of `DESIGN.md` (section 5.5). Heads-up NLHE, self-play only, trained on
the torch vectorized env `pokerbot.env.VecNLHE`, played through the scalar
engines by `NeuralBlueprintAgent`.

| Module | Contents |
|---|---|
| `features.py` | canonical network inputs (`features_from_obs`), `FeatureConfig` (optional equity inputs) |
| `networks.py` | `AdvantageNet`, `NetConfig`, `regret_matching`, `StrategyHead` |
| `memory.py` | `ReservoirMemory`: compact host reservoir buffer, minibatch sampling, save/load |
| `traversal.py` | `FrontierTraverser` (batched external sampling), `rollout`, `actor_probs`, `NetPolicy`, `deal_env` |
| `trainer.py` | `DeepCFRTrainer`: the CFR loop, checkpoints, resume, logging, evaluation hook |
| `policy.py` | `SDCFRPolicy`: reach- and iteration-weighted average of the advantage nets |
| `scalar.py` | one-hand mirror of the env's action mapping and observation encoder, for `GameState`s |
| `agent.py` | `NeuralBlueprintAgent` (`neural:<run dir>` in `scripts/play_match.py`); also a `PolicyAgent` (`policy`, `policy_batch`, `policy_all`) with `vec_policy(device)` |
| `range_policy.py` | `NeuralRangePolicy`: the SD-CFR average as a function of the public state and a hand, batched over all 1326 combos (own reach recomputed from the history) |
| `vec_policy.py` | `NeuralVecPolicy`: the average policy acting on a `VecNLHE` batch from `env.obs()` (ABR opponent) |
| `checkpoint.py` | checkpoint layout, `save_net` / `load_net` / `list_checkpoints` |
| `config.py` | `DeepCFRConfig` (YAML) |

## Algorithm as implemented

Deep CFR (Brown et al., 2019) with external sampling and linear CFR weighting,
using the single-deep-CFR average (Steinberger, 2019). No policy network is
trained.

For iteration `t = 1, 2, ...` and each seat `p` in 0, 1:

1. **Traverse.** `traversals_per_iter` root hands are dealt, in batches of
   `roots_per_batch`. Seat `p` branches on every legal abstract action. The
   opponent samples one action from its current policy. Chance is sampled once
   per root, because the deck is fixed at the deal. The current policy of a
   seat is regret matching on its latest advantage net; when no legal action
   has a positive advantage it plays the best one (`fallback: argmax`, the
   paper's rule: uniform play there was about 50% more exploitable in its
   ablation). Before a seat has a net, its policy is uniform over the legal
   actions. Leaf values can be variance-reduced (below). Every node where `p`
   branches adds one sample to `p`'s advantage memory: features, legal mask,
   instantaneous regrets `r(a) = v(a) - sum_b sigma(b) v(b)` in value units,
   and the iteration `t`. Seat 1's traversal already uses the net seat 0
   trained in this iteration, as in Algorithm 1 of the paper.
2. **Train.** The net is reinitialized, or warm-started with `reinit: false`.
   It then runs `sgd_steps` Adam steps on uniform minibatches from the
   reservoir, with the linear-CFR loss
   `sum_i t_i * sum_a legal_ia (net(x_i)_a - r_ia)^2 / sum_i t_i`.
   Gradient-norm clipping is applied. On CUDA the MLPs run under bf16
   autocast and the GRU and the loss stay in fp32.
3. **Save.** The net is saved as `checkpoints/p{p}/iter{t}.pt`. The average
   strategy is `avg(I) = sum_t t * pi_t(I) * sigma_t(I) / sum_t t * pi_t(I)`,
   where `pi_t(I)` is the seat's own reach under `sigma_t`. `SDCFRPolicy`
   computes it exactly during play. At each own decision it evaluates every
   net, and after acting it multiplies each net's reach by that net's
   probability of the action taken. `last_n` averages only the most recent
   nets. `reach_weighted=False` gives a plain `t`-weighted mixture.

Values are chips divided by `value_scale`, which defaults to the big blind.
The 4070 Ti config uses 1000, which is 10bb. Regret matching is
scale-invariant, so the scale only conditions the regression.

## The frontier scheme (`traversal.py`)

The loop state is an env holding the live slots, plus these per-slot tensors:

| Tensor | Meaning |
|---|---|
| `owner [n]` long | edge the slot's value belongs to: `node_id * A + action`, or -1 before the first branching node of its root |
| `root [n]` long | root hand index |
| `cut [n]` bool | slot is being rolled out (no more branching or samples) |
| `depth [n]` long | branching nodes on the slot's path |
| `corr [n]` float | chance control-variate corrections since the slot's last branching node |

One frontier step, the only Python loop:

1. `obs()` on all live slots, then one network batch per seat (`actor_probs`).
2. The traverser's slots that may branch (not cut, `depth < max_depth`) take
   the frontier budget in slot order. The budget is
   `max_frontier_nodes - n`, and a slot's fan-out costs `#legal - 1`. Slots
   over the budget become `cut`.
3. Branching slots become nodes. The node records its compact features,
   policy, parent edge (`owner`) and root. The slot is replicated with
   `env.select` once per legal action, and each copy's owner is the new edge.
   All other slots keep one copy with a sampled action. Opponents sample from
   their own policy. Cut and depth-capped traverser slots sample from the
   traverser's policy, which is outcome sampling.
4. `env.step(actions)`. Finished slots write `payoff[p] / value_scale` (plus
   `corr`) into their owner edge. A slot with owner -1 writes its root value
   instead. The finished slots are then dropped with a second `select`.

If slots are still live after `max_steps`, every slot is cut and they play on
as rollouts.

**Backup.** Each edge receives exactly one value: a leaf payoff or the value
of the next branching node below it. Child nodes are always created at a later
frontier step than their parents. One reverse pass over the steps computes
`v(node) = sum_a sigma(a) v(node, a)` for all nodes of that step as one tensor
op and writes the results into the parent edges (plus the corrections the
node's slot collected on the way down). Regrets are `edge_value - node_value`,
masked to the legal actions.

**Variance reduction** (`traversal.allin_equity`, `traversal.chance_cv`).
The regret targets are dominated by sampling noise, much of it from the board.
Both players' hands and the board are fixed at the root, so the traverser's
equity against the opponent's actual hand is computed once per root for every
street, `E_0..E_3` (`street_equities`: exact on the flop, turn and river,
`preflop_equity_samples` Monte Carlo runouts preflop).

* `allin_equity`: a hand that ends in a called all-in before the river is
  scored `stake * (2 E_s - 1)` instead of the pre-dealt runout.
* `chance_cv: beta`: a slot that moves to a new street with `c` chips in per
  player adds `-beta * 2c * (E_new - E_old)` to `corr`, a check-down control
  variate. `E_old` is the exact expectation of `E_new` over the dealt cards
  (unbiased preflop), so the correction has mean zero.

Both leave every regret target unbiased. Measured on the 100bb run's
iteration-33 nets over 1024 roots per seat, they cut the mean squared regret
target by 73% preflop, 51% on the flop, 34% on the turn and 0 on the river (21%
overall; the all-in part alone: 67%, 46%, 28%). The cost is about 34M hand
evaluations per 8192 roots.

**Why the caps keep regrets unbiased.** A cut slot plays on by sampling the
traverser's current policy. Its payoff is therefore an unbiased single-sample
estimate of the counterfactual value of the edge above it. Branching nodes
above a cut get noisier regret estimates, but the estimates stay unbiased.
Nodes below a cut are not recorded. Recording them would weight samples by
the traverser's own reach.

**Memory bounds.** Live slots never exceed `max_frontier_nodes`, so env and
observation memory are `O(max_frontier_nodes)`. That is about 0.6 KB of env
state and 0.4 KB of features per slot. Recorded nodes are bounded by
`max_steps * max_frontier_nodes` and are stored compactly on the device
(about 140 B each including the policy). Network calls run in chunks of
`infer_chunk` rows, which bounds activation memory. A batch of `R` roots with
the default abstraction peaks at about 20-35 live slots per root. The
widest step is on the flop or turn.

`test_traversal.py` checks the backup against a brute-force recursion that
uses the scalar engine and the scalar feature encoder on the same decks. The
traverser policy is a random net and the opponent is deterministic. Every node
value, child value and regret must match. The same check runs with the
frontier and depth caps forcing rollouts, using a deterministic traverser, and
again with `allin_equity` and `chance_cv` against a brute force that applies
the same equity values and corrections on the scalar engine.

## Tensor layouts

**Features** (`features.py`), leading dim `n`:

| Key | Shape and type | Contents |
|---|---|---|
| `cards` | `[n, 7]` long | own hole cards, then board; 52 = undealt |
| `card_mask` | `[n, 7]` bool | visible cards |
| `hist` | `[n, 24]` long | env tokens `1 + (street*2 + is_button)*A + idx`, 0 = pad |
| `hist_amt` | `[n, 24]` float | chips added per action / starting stack |
| `scalars` | `[n, 14 (+1) (+10)]` | `obs["scalars"]`, then optional equity and 10-bin equity histogram |
| `legal` | `[n, A]` bool | legal abstract actions |

**Advantage memory** (`memory.py`, per sample, `A = 6`, `S = 14`):

| Field | Type | Bytes |
|---|---|---|
| cards | uint8 x 7 | 7 |
| hist | uint8 x 24 | 24 |
| hist_amt | fp16 x 24 | 48 |
| scalars | fp16 x 14 | 28 |
| legal | bool x 6 | 6 |
| target | fp16 x 6 | 12 |
| iteration | uint16 | 2 |

That totals 127 B per sample, about 5.1 GB per 40M-sample player. The arrays
are numpy on the host, allocated lazily with `np.zeros`. Only sampled
minibatches become torch tensors. On CUDA they go through reused pinned
buffers with a non-blocking copy, and a host thread prefetches `prefetch`
batches. `save` writes one `.npy` per field and `load` streams them back
memory-mapped. `load(mmap=True)` keeps a full buffer memory-mapped
copy-on-write.

**Network** (`networks.py`, about 2.2M parameters at the defaults):

- Card branch: per slot `rank_emb + suit_emb + slot_emb` (zeroed when
  undealt), summed per group (hole, flop, turn, river), then an MLP to 768.
- History branch: token embedding plus position embedding plus a linear
  projection of the amount, then a packed GRU (hidden 256, final state).
  `hist_type: transformer` swaps in a transformer encoder with a summary
  token.
- Scalars: a linear layer to 64.
- Trunk: 3 layers of width 512 (residual and LayerNorm after the first),
  then a linear head to `A` advantages. The caller applies the legal mask
  in `regret_matching`.

## Running on the RTX 4070 Ti

```
python scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml --out runs/dcfr1
tensorboard --logdir runs/dcfr1/tb
python scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml --resume runs/dcfr1
python scripts/play_match.py --a neural:runs/dcfr1 --b equity --hands 20000 --duplicate
```

- Everything is device-agnostic. `device: auto` picks CUDA. bf16 autocast is
  used only on CUDA, and TF32 is enabled there for the fp32 GRU.
- Resume points are written every `memory.save_every` iterations: memories,
  RNG states and `trainer_state.pt`. On `--resume`, checkpoints newer than
  the resume point are deleted, because their data is not in the saved
  memories.
- In the first iteration, check that `slot_steps_per_s` and the per-phase
  seconds match the estimates in the config header. Also check VRAM with
  `nvidia-smi`. The config comments explain which knobs to turn:
  `roots_per_batch` and `max_frontier_nodes` together, and `infer_chunk`.
- The CPU smoke test is
  `python scripts/train_deepcfr.py --config configs/deepcfr_tiny.yaml`,
  which takes about 1 s per iteration.

## What to watch in the logs

`log.csv` has one row per iteration and seat. TensorBoard has the same
values under `p0/` and `p1/`.

| Column | Healthy behaviour |
|---|---|
| `loss` vs `loss_first` | Each fit should end well below its first-step loss. If the final loss stops dropping, add SGD steps or width. The absolute level rises while early iterations add variety, then flattens. |
| `adv_mem_size`, `adv_mem_seen` | Size reaches capacity after a few iterations. After that, `seen` keeps growing and old iterations are replaced uniformly. |
| `nodes`, `slot_steps` | Samples and frontier work per iteration. A sudden drop means the policies fold early, which is common in the first iterations. |
| `cut_slots` | Frontier cap hits. Many cuts mean fewer regret samples deep in the hand. Raise `max_frontier_nodes` or lower `roots_per_batch`. |
| `max_frontier` | Must stay at or below `max_frontier_nodes`. |
| `regret_abs_mean` | Mean absolute regret in value units. It should shrink slowly as the strategy converges. |
| `allin_leaves` | Leaves scored by all-in equity (`allin_equity`). Zero means the feature is off. |
| `val_r2_preflop` ... `val_r2_river`, `val_r2_all` | Iteration-weighted R^2 of the new net on the held-out samples (`memory.holdout`): `1 - MSE / mean square target`, so predicting zero scores 0. The targets are mostly noise, so values are low (on the first 100bb run, about 0.02 preflop and 0.6 on the river), but a change to the network, loss or inputs that lowers them is fitting worse. Compare runs on these, not on `loss`. |
| `traversal_s`, `train_s`, `slot_steps_per_s` | Throughput. Training should dominate once the memories are full. |

`eval.csv` and the TensorBoard tags `eval/mbb_vs_equity` and
`eval/mbb_vs_previous` hold duplicate-match results with bootstrap 95% CIs.
The current average strategy plays `EquityThresholdAgent` and the average
strategy of the previous evaluation. `vs_equity` should turn and stay clearly
positive. `vs_previous` should hover at or above zero. A significantly
negative `vs_previous` is a red flag for the run (DESIGN.md 5.7).

Every evaluation plays the same deals (`eval.seed`), so results of different
iterations are paired. With `eval.sample_net` each hand is played by one net
drawn with probability proportional to its iteration, which is the SD-CFR
average in distribution at one forward pass per decision. With
`eval.luck_adjust` the `mbb_adj` columns (TensorBoard `eval/mbb_adj_vs_*`)
report the same matches with all-in EV and chance corrections
(`pokerbot.eval.luck`): unbiased, with a narrower interval.

## Measured on this container's CPU

The container has 4 shared cores, with other jobs running. The numbers below
use 2 torch threads, the DESIGN.md 5.3 abstraction, 200bb stacks, 512 roots
and `max_frontier_nodes = 16384`. A slot-step is one live frontier slot
processed in one step.

| Policy network | slot-steps/s | branching nodes/s |
|---|---|---|
| none (uniform; env only) | ~140-240k | ~45-75k |
| small net (~0.1M params) | ~35-60k | ~10-17k |
| default net (2.2M params, fp32) | ~9k | ~2.8k |

On the CPU the network dominates. The tiny config takes about 0.7 s per
iteration, or 1.8 s for the first iteration including warm-up.

## Known limitations and shortcuts

- **Blinds and stacks are baked in.** Features are fractions of the starting
  stack, but the abstraction and the training distribution use one game
  config. The agent warns when it plays a different one.
- **Off-tree opponent bets** get a history token chosen by the `offtree`
  option of `NeuralBlueprintAgent` (`neural:<dir>,offtree=nearest`) and
  `NeuralRangePolicy`. The default `"harmonic"` uses the pseudo-harmonic
  mapping of `pokerbot.abstraction.actions.map_offtree` between the
  neighbouring legal sizes: randomized with the `act` rng, drawn once per
  opponent action per hand, in the agent; deterministic (`u = 0.5`) in the
  stateless `policy` / `policy_batch` / `policy_all` path, as the tabular
  policy does. `"nearest"` records the nearest *legal* size (`nearest_abstract`:
  closest amount, first on ties), the index `VecNLHE.step_concrete` records.
  Either way, own actions, on-tree sizes and folds/calls are unchanged; a
  raise to exactly the all-in is the `allin` index; and on the tree the
  token is the one `VecNLHE.step` records.
- **Stateless queries.** `policy(state, seat)` and the batched paths
  recompute the own-reach weights from the history (one forward pass per net
  per earlier own decision, cached per history prefix in the batched path),
  so a query costs more than the agent's own incremental tracking. The
  `NeuralVecPolicy` tracks reach per slot and resets it when a slot's deck
  changes; it requires the env to use the blueprint's action spec.
- **The frontier budget is spent in slot order**, so hands later in a batch
  are cut first when the cap binds. Regrets stay unbiased, but deep nodes of
  those hands are under-sampled.
- **One chance sample per root.** All branches of a root share the board.
  This is unbiased and correlates the child values of a node.
- **Evaluation is small.** It uses `last_n` nets per seat and about a
  thousand duplicate deals. It shows trends, not adoption-grade results. The
  DESIGN.md adoption rule needs a 200k-hand match.
- **SD-CFR play cost grows with the number of nets.** The agent runs one
  forward pass per net per decision. Use `sample_net=true`
  (`neural:<dir>,sample_net=true`: one net per hand, the same strategy in
  distribution), `last_n`, or the GPU for long runs. Range queries
  (`policy_all`, search) still need every net.
- **Untested on CUDA.** This container has no GPU. The code has no
  CPU-only paths and does no per-element host work, but the VRAM and time
  estimates in `configs/deepcfr_4070ti.yaml` are calculations, not
  measurements. Whether bf16 GRUs are safe was not checked, so the GRU
  always runs in fp32.
- **Strategy memory** (`record_strategy`) is recorded if asked, but nothing
  trains on it. SD-CFR does not need it.
- **No `torch.compile` or CUDA graphs yet.** The frontier changes size every
  step. Compiling `env.step` with dynamic shapes is the obvious next speed-up.
