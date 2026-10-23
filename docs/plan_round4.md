# Round 4 plan: value nets back to the flop root, then preflop (2026-10-10)

Goal: the strongest bot, measured as before: exact exploitability on spots, plus
duplicate matches with confidence intervals.

## Where the bot stands

The production bot (`configs/search_value_net.yaml`, merged in b1a23c9):

| street | who decides | measured |
|---|---|---|
| preflop | blueprint `dcfr4_distilled_v2` | not measured (2026-10-19: 338-380 local best response; replaced 2026-10-21 by the solved table `preflop_r15.pt`, 222-252, `docs/preflop_eval.md`) |
| flop | search, flop + turn, `turn_v4e` leaves, 4 s | exploitability 95 two-sided / 216 one-sided (12 spots) |
| turn | depth-0 search, `river_v3` x 48 leaves, 2 s | 190 / 648 (18 spots) |
| river | search to showdown, 1 s | starts from the cached turn solve |

* **Head to head:** +255 mbb/hand against its own blueprint over 10,000 hands,
  CI [+91, +430].
* **Diminishing returns:** Round 3's search changes gained +56 [-88, +200] over
  Round 2 on the same deals.

## Why the next step is not a new Deep CFR blueprint

**What the blueprint still does in production.**
* It plays every preflop decision.
* It supplies both players' ranges at the start of every flop search, by
  replaying the preflop.
* It is the self-play source of the leaf nets' training states.

Everything after the flop is search. A better blueprint therefore helps only
through its preflop strategy.

**That preflop strategy was solved against the blueprint's own postflop play.**
* That postflop play is far weaker than the search. dcfr3's ABR found +2,720
  mbb/hand, mainly from folding to river shoves after postflop raise wars.
* A Deep CFR v5 would still optimise its preflop against its own postflop, not
  against what the bot plays.
* dcfr3 -> dcfr4 (richer actions, more fit compute) did not win head to head:
  -58 [-233, +125].

**The direct route is DeepStack's: extend the value nets back to the flop root.**
Preflop can then be solved against the values of the postflop game the bot
actually plays. The chain exists up to the turn root:

* river net -> turn-end net (flop search leaves);
* river net x 48 (turn search leaves);
* turn-start net (Round 2, depth-0 flop search).

Only the turn-start net is weak. At 200k samples it is limited by data (held-out
MAE 0.039), and depth-0 flop search with it was 131 mbb/hand behind production
on 3 spots.

## Stage 1: a turn-start net good enough to build on (local GPU, started 2026-10-10)

`runs/r4/chain1.cmd`, about 20 h:

1. **`turn_start_b`.**
   * 800k turn-root subgames: 4x the data, seed 1, the 50/25/25 self-play /
     perturbed / random mix.
   * Solved by `BatchTurnSolver` for 300 iterations, with `turn_v4e` leaves
     (11% more accurate than Round 2's `turn_v3w`).
   * Held-out: `turn_start_heldout_b`, 16k samples, seed 998.
   * Rate about 13-16 samples/s, so 14-17 h.
2. **`turn_start_v2`.** 2048 x 8 layers, 120k steps: the recipe that gave the
   turn-end net its 11%.
3. **Gate** (`scripts/eval_depth0.py`, 3 spots, exact river):
   * depth-0 flop decisions with `turn_start_v1` and with `turn_start_v2`;
   * each against the production depth-1 search.

| result | next |
|---|---|
| v2 closes most of the 131 gap | stage 2 |
| v2 improves on v1 but a gap remains | more data, with on-policy turn roots recorded from production flop searches; this is where AWS helps |
| v2 does not improve on v1 | the chain is too lossy; go to stage 3 |

If depth-0 flop search becomes competitive, it is also worth having on its own:
the whole flop abstraction in about 200 nodes, and about 3x the iterations.

### Stage 1 result (2026-10-11): v2 does not improve on v1, so stage 3

* **`turn_start_v2`:** held-out MAE 0.029 against v1's 0.039 (the held-out sets
  differ).
* **Gate, re-solved** (`runs/r4/depth0_resolved.json`, `--resolve-iters 400`):
  * every flop decision of both players is locked to the profile's;
  * every turn decision is solved again on the production trunk with production's
    leaf net;
  * so the profiles differ only in their flop strategies.

  mbb/hand, exact river, lower is better:

  | spot | production, one-sided | v1, one-sided | v2, one-sided | production, two-sided | v1, two-sided | v2, two-sided |
  |---|---:|---:|---:|---:|---:|---:|
  | board0 BB first | -105 | 96 | 86 | 172 | 302 | 287 |
  | board0 BTN vs check | 582 | 638 | 631 | 161 | 290 | 282 |
  | board1 BB first | -72 | 12 | 27 | 174 | 204 | 214 |
  | **mean** | **135** | **249** | **248** | **169** | **266** | **261** |

* **The hybrid scoring** (production's turn strategy behind the depth-0 flop) gave
  the same picture:
  * one-sided: v1 280, v2 278, production 147-157;
  * two-sided: 307 / 310 against 187-197.

  Re-solving the turn shrinks the gap only a little:
  * one-sided, from about 127 to 114 mbb/hand;
  * two-sided, from about 116 to 95.
* **Reading.**
  * 4x the data and a lower held-out MAE left the flop decisions as they were.
  * So the loss is not fit error. Likely sources are the targets themselves
    (turn-start values solved with `turn_v4e` leaves) or a mismatch between the
    training ranges and the flop search's ranges.
  * A flop-root net trained on turn-start leaves would add another lossy step.

### After stage 1 (2026-10-11/12): the turn and the flop instead

Stage 3 lists a new Deep CFR blueprint first. But the reasoning above says a
blueprint helps only through preflop, and dcfr3 -> dcfr4 did not win. The turn
was the largest measured leak, so the work went there. Details are in
`docs/value_net_eval.md`, Round 4.

| | Round 3 production | Round 4 production |
|---|---|---|
| turn leaf model | `river_v3` | `river_v4rx` (residual head, fused Triton kernel, third on-policy round with the turn searches' leaves, 160k steps) |
| flop leaf model | `turn_v4e` | `turn_v5x` (12 layers, 240k steps, `river_v4rx` targets) |
| solver | eager | CUDA-graph replay (turn: 318 instead of 183 iterations in 2 s) |
| 18 turn spots, two-sided | 189 | **148** |
| 12 flop spots, two-sided | 95 | **91** |

Stages 2 and 3 are unchanged as options. Preflop is still unmeasured.

## Stage 2: flop-root net and preflop (AWS-scale data)

* **Flop-root data.** Flop-root subgames (both ranges at the start of the flop)
  solved with depth-0 flop trees and turn-start leaves averaged over the turn
  cards (`FlopEndLeafEvaluator`). This is DeepStack's flop network.
  * It needs a batched flop solver like `BatchTurnSolver` (new code).
  * Probably 1M+ samples, which is embarrassingly parallel and suits spot
    instances: resumable shards, seeded per shard.
* **Preflop solve.**
  * The preflop abstraction is small. Solve it with CFR against flop-root net
    values at every preflop terminal x flop.
  * That gives a preflop strategy consistent with the postflop the bot plays.
  * Later the same net allows depth-limited preflop re-solving at decision
    time.
* **Measure:**
  * the best response to the old and the new preflop strategy in the
    flop-root net's game;
  * then exact checks of the flop-root net on held-out flop roots;
  * then a 10,000-hand match.

## Stage 3: fallback and longer-term options

* **Deep CFR blueprint v5** if stages 1-2 stall. Known levers, measured on
  dcfr3:
  * more fit compute;
  * no frontier cuts.

  Each run is about a day locally; AWS would let 3-4 configs run in parallel.
* **200bb play.** The ACPC standard and external bots such as Slumbot play 200bb
  (as far as we know). The bot and all its nets are 100bb. An external benchmark
  needs a 200bb blueprint plus regenerated value-net data.
* **From the Round 3 report:**
  * a cheaper residual head (`river_v3r` at equal iterations: -13 / -14 on turn
    spots);
  * a third on-policy round.

## AWS

* **Quota.** The account had 0 GPU quota in us-east-2. Requests for 16 vCPUs of
  G/VT instances (on-demand and spot) were submitted on 2026-10-10 and are
  pending.
* **First use: a pilot.**
  * A Linux setup script, the test suite, and `check_turn_leaves.py`, which
    should reproduce `turn_v4e`'s MAE 0.0315.
  * A short data-generation job, timed against the local 16 samples/s.
  * Then stage 1's extra data or stage 2's data in parallel.
* **Security.** The CLI here uses the root account's access keys. Create an IAM
  user for this work and remove the root keys first.
* **Time budgets depend on the GPU.** Compare across machines with fixed
  iteration counts, or re-time.
