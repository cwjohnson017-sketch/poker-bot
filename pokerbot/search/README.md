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
| `tree.py` | `TreeBuilder` / `build_tree` -> `SubgameTree` (flat tensors), node budget |
| `showdown.py` | O(n) range-vs-range showdown with card removal (`ShowdownTables`), fold kernel, dense reference |
| `leaf.py` | depth-limit leaf values from blueprint rollouts with `k` biased continuation strategies |
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
| `kind` | `DECISION`, `CHANCE`, `FOLD_NODE`, `SHOWDOWN`, `LEAF`, `CONTINUATION` |
| `actor` | player to act (decisions, and the leaf chooser at `LEAF`), else -1 |
| `street`, `contrib [N, 2]`, `bets [N, 2]` | street, chips committed this hand, chips committed this street |
| `board_id` | index into `boards` (the public board at the node) |
| `deal_card`, `chance_weight` | card dealt by the chance parent and its weight `1 / (52 - |board| - 4)` |
| `action_kind`, `action_amount`, `action_abstract` | concrete action into the node (`-1` abstract index = off-tree) |
| `first_child`, `num_children`, `children [N, max]` | child ranges |
| `cont`, `folder` | continuation strategy index, player who folded |
| `level_start` | node ids of each depth |
| `current_node`, `path_nodes` | the observed decision node, and the (node, slot) pairs along the observed path |

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
* Without a cached solve (the first flop decision), `T` is the opponent's
  blueprint-vs-blueprint value from `gadget.rollouts` rollouts.
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
  rollout node. Several thousand leaves make this the bottleneck unless the
  blueprint implements a vectorised `policy_combos`.
* **If a decision runs over budget**, lower these in order:
  `leaf.max_total_rollouts`, `tree.max_nodes` (10k is a good flop value),
  `solver.max_runouts`, `tree.chance_cards`.
* **Turn and river** trees are much smaller (a turn tree at 20k nodes has
  no leaves). The 1 s budget there is mostly iterations.

## Plugging in a blueprint

Anything with `spec` (an `ActionSpec`) and `policy(state, player)` works.
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

class MCCFRBlueprint:            # adapter around the tabular BlueprintAgent
    def __init__(self, path):
        self.agent = BlueprintAgent.load(path)
        self.spec = self.agent.spec
    def policy(self, state, player):
        return self.agent.action_probs(state, player)   # {abstract index: prob}

register_blueprint("blueprint", MCCFRBlueprint)          # -> search:blueprint:<path>
agent = SearchAgent(MCCFRBlueprint("runs/mccfr.bin"), search_config())
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
* `test_abstract.py`: scalar abstract actions against `pokerbot.env.actions`,
  pseudo-harmonic mapping, `CardView`, `range_reach`.

## Shortcuts and limits

* One leaf chooser (the searcher's opponent), not both players as in
  Pluribus. The searcher's continuation is the plain blueprint.
* Ranges at the first flop root come from the blueprint, not from our
  actual preflop play. Later streets use the cached solve.
* The first gadget terminate values are blueprint-vs-blueprint rollout
  values, not a best response to the blueprint, so they are noisy.
* Sampled run-outs (`max_runouts`, rollouts) and subsampled chance cards
  give an unbiased but noisy game. The solver treats that sampled game as
  exact.
* Rollout generation and blueprint queries are Python loops on the CPU.
