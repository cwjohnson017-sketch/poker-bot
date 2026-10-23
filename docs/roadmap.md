# Long-term roadmap (2026-10-21)

Where the project goes after Round 4 and the preflop solve. The four goals, in
the author's order:

1. **As good as the average player.**
2. **Better than the author and the average player.**
3. **Up to 6 players and different stack sizes.**
4. **Professional level.**

The plan keeps the project's way of working: rounds that end in a measured gate,
and production changes only when a measurement says so. Each goal gets a
definition that can be measured, the work that leads to it, and a gate. What the
strongest poker AIs did, and what this plan takes from each, is in section 3.

**Status (2026-10-23):** Phase 1 in progress. The tools of 1.1-1.5 are built
and on `main`: the play app (evaluation mode; review mode not yet), AIVAT
action terms, archetype opponents, LBR against the search, the soak harness
and the preflop audit. Gate 1's production runs (archetypes, LBR and a
20,000-hand soak) started on 2026-10-22 on spot GPUs and the local GPU
(`cloud/aws/fleet.py`); a quarter had finished after 30 hours. The first
results (section 2) show the search leaving weak players' money on the table,
so a safe exploitation layer (1.6) joins Phase 1. Update this line as gates
pass.

## 1. Summary

| phase | goal | main work | gate, short | rough length |
|---|---|---|---|---|
| 1 | 1 | play app; AIVAT with the bot's action terms; archetype opponents; full-game LBR; soak test; safe exploitation layer | beats every archetype, not significantly less than the blueprint does; LBR under +1,000 everywhere; clean soak | 4-6 weeks |
| 2 | 2 | sessions against the author; the preflop leak; 200bb nets and preflop; Slumbot and GTO Wizard Benchmark clients | the author loses over a fixed AIVAT sample; first outside scores at 200bb | 4-8 weeks |
| 3 | 3, depths | nets for 10-250bb; preflop tables per depth; push/fold | 20-200bb within 25% of 100bb's exploitability | 3-5 weeks |
| 4 | 3, six-max | N-player plumbing; a Pluribus-style blueprint; heads-up pots through the existing search; three-way search | search beats blueprint-only tables; archetype tables and people lose | 3-5 months |
| 5 | 4 | preflop search, nets trained on the search's own play, Supremus-scale data, off-tree safety, a bigger blueprint | GTO Wizard Benchmark at least -5 bb/100; Slumbot at least +100 mbb/hand; LBR loses; strong people lose | open-ended, 6+ months |

The lengths assume the pace of the project's first three weeks and one GPU,
with spot instances for batch jobs. They are guesses: each gate can send work back a step.

**The main points:**

* **The bot has never played anything outside its own lineage:** no person and
  no outside bot; a full-game local best response (LBR) first played it on
  2026-10-22. Phase 1 builds those yardsticks, and every later gate uses them.
* **Goal 1 takes exploitation, not only strength.** The median online player
  loses about 20 bb/100 before rake (3.5), but equilibrium play does not
  collect it all: in the first Gate 1 results the search breaks even against
  a calling station that its own blueprint beats by 2.7 bb/hand, because it
  plays every opponent as a sound one (section 2). Phase 1 adds a safe
  exploitation layer (1.6).
* **200bb comes early.** Both public bot benchmarks, Slumbot and the GTO
  Wizard Benchmark, play only 200bb. It is also the first step of goal 3.
* **The closest published relative is Supremus:** DeepStack with a GPU DCFR+
  solver, nets at the end of every street, more sizes, and far more training
  data. It beat Slumbot by 176 mbb/hand, where a plain DeepStack
  reimplementation lost by 63. More and better net targets is the same lever as
  this project's value-net rounds.
* **Six-max follows Pluribus**, an MCCFR blueprint plus depth-limited search,
  about 12,400 core-hours, a one-off cloud job. When two players see the flop,
  the existing heads-up search plays the rest.
* **"Professional" gets numbers:**
  * at least -5 bb/100 on the GTO Wizard Benchmark, where the best independent
    bots score -3;
  * at least +100 mbb/hand against Slumbot;
  * LBR losing;
  * strong people losing over a fixed sample.
* **The bot never plays on real-money sites.** The big sites ban bots and
  confiscate funds (section 12).

## 2. Where the bot stands

Heads-up no-limit hold'em, 100bb (10,000 chips, 50/100 blinds), stacks reset
every hand.

| street | who decides | time | exploitability measured (mbb/hand) |
|---|---|---|---|
| preflop | solved table `preflop_r15.pt` over the blueprint `dcfr4_distilled_v2` | instant | 222-252, preflop deviations only (`docs/preflop_eval.md`) |
| flop | search, flop + turn, `turn_v5x` leaves | 4 s | 91 two-sided / 212 one-sided, 12 spots |
| turn | depth-0 search, `river_v4rx` x 48 leaves | 2 s | 148 two-sided / 605 one-sided, 18 spots |
| river | search to showdown from the cached turn solve | 1 s | not measured on its own |

* **Head to head:** +198 mbb/hand luck-adjusted against its own blueprint over
  10,000 hands, CI [+29, +366]. Round 4 against Round 3: +11 [-75, +97].
* **Never measured:**
  * any human opponent;
  * any bot outside its own lineage;
  * the full-game exploitability of the production bot by ABR, which only ever
    ran against blueprints (dcfr3: +2,720 mbb/hand). LBR against production
    started on 2026-10-22 (below);
  * any stack depth but 100bb, any table but heads-up.

**First Gate 1 results (2026-10-23, about a quarter of the runs).** Production
(the preflop table, then the search with fixed iterations per street,
`configs/search_value_net_fixed.yaml`, so the result does not depend on the
GPU) and the blueprint play the same duplicate deals against each archetype.
Luck-adjusted mbb/hand with 95% CIs, 5,000 hands each:

| archetype | production | blueprint | production - blueprint |
|---|---:|---:|---:|
| calling station | -46 [-428, +323] | +2,662 [+2,130, +3,231] | -2,708 [-3,248, -2,142] |
| maniac | +233 [-212, +707] | +1,073 [+573, +1,529] | -840 [-1,363, -282] |
| overbet bluffer | +3 [-221, +246] | +500 [+183, +812] | -497 [-843, -158] |
| nit | +431 [+334, +523] | +575 [+460, +685] | -144 [-261, -31] |
| fit-or-fold | +492 [+326, +666] | +468 [+248, +697] | +24 [-205, +250] |
| min-raiser | +211 [+27, +374] | +173 [+1, +333] | +38 [-158, +234] |
| population | +204 [+75, +343] | +131 [-40, +285] | +73 [-105, +258] |
| loose-passive | +37 [-199, +251] | -62 [-338, +214] | +99 [-205, +415] |
| regular | +337 [+202, +478] | +211 [+24, +382] | +126 [-43, +318] |

* **LBR** (250-500 hands per configuration so far) finds no exploit. LBR
  loses in the four `call` and `fc` configurations (-916 to -3,082 mbb/hand)
  and is near zero in the two `full` ones (+251 and -88). All six are under
  Gate 1's +1,000 bar; four CIs still reach above it until their samples fill.
* **Soak:** 16,000 of 20,000 hands, no crashes, illegal actions or fallbacks,
  p99 decision time 5.5 s.
* **Why production trails the blueprint** against three archetypes. A third
  player, the preflop table followed by the blueprint with no search, replayed
  1,000 of the same deals and splits each gap:

  | archetype | gap | from the preflop table | from the search |
  |---|---:|---:|---:|
  | calling station | -2,557 | -218 (not significant) | -2,339 |
  | maniac | -1,512 | -641 | -871 |
  | overbet bluffer | -936 | -476 | -460 |

  * The search plays every opponent as a sound one: the opponent's range comes
    from the blueprint's own policy (`range_reach`) and its responses from the
    subgame equilibrium. Against the calling station, which calls any size and
    never bluffs, it puts about three times as many chips into river bluffs as
    the blueprint does, and bets strong hands about 1x the pot where the
    blueprint shoves about 3.5x. Nearly all of the gap is in hands that reach
    the river.
    Against the maniac, which bluffs 70% of its bets, it calls down and then
    folds the river: strong hands fold 32% of the time facing a bet, against
    the blueprint's 1%.
  * The preflop table is closer to equilibrium than the blueprint. In the small
    blind it folds 15% of hands (the blueprint none) and limps 10% (the
    blueprint 37%); in the big blind it defends more. That costs against loose,
    aggressive players. The split above plays the blueprint after the table,
    whose ranges it was not trained for, so the table's share is approximate.
  * The blueprint is not the stronger player: the search beats it head to head,
    and LBR finds nothing in the search. Its leaks (huge value bets, rarely
    folding, limping) are what these archetypes pay for. Matching it there
    takes exploitation (1.6); Gate 1's regression check (section 4) fails
    without it.

**What the code supports today:**

| piece | 2 players | 3-6 players | other stack depths |
|---|---|---|---|
| Rust engine (`engine/`) | yes | yes: rules, side pots, 2-9 players | yes |
| torch `VecNLHE` | yes | no ("heads-up only") | yes |
| tabular MCCFR (`engine/src/mccfr.rs`) | yes | no: the infoset key layout is heads-up; `MAX_RAISES_CAP = 4`, `MAX_ABSTRACT_ACTIONS = 16` | yes |
| search (`pokerbot/search`) | yes | no: two ranges | trees yes |
| value nets | yes | no | the inputs have stack-to-pot features (`value_net.context_features`), but every sample was 100bb |
| preflop | `preflop_r15.pt`, 31 flop starts | no | no: one table, 100bb |
| match runner (`eval/match.py`) | yes | plain matches only: duplicate and luck adjustment are heads-up only | yes |
| luck adjustment (`eval/luck.py`) | AIVAT's chance terms | no | yes |
| human play | a terminal prompt over the socket protocol (`serve_match.py` + `play_client.py --agent human`) | through plain matches | yes |

**Hardware:** i9-13900K (24 cores, 32 threads), 64 GB RAM, RTX 4070 Ti (12 GB),
about 520 GB free disk. AWS spot `g5.xlarge` (one A10G, about 1.2x slower than
the 4070 Ti here at the same iterations) at about $0.48-0.55/h. The GPU spot
quota is 32 vCPUs in each of us-east-1 and us-east-2 since 2026-10-22, eight
`g5.xlarge` per region; spot capacity, not quota, held Gate 1's runs to one or
two instances at a time.

## 3. What the strongest poker AIs teach

Checked against the papers and the benchmark sites on 2026-10-21. mbb/hand =
milli-big-blinds per hand; 10 mbb/hand = 1 bb/100. The papers mix standard
deviations, standard errors and 95% CIs under "+/-"; the type is given where the
source states it.

### 3.1 Heads-up

| system | what it was | result | what this plan takes |
|---|---|---|---|
| **Libratus** (CMU, 2017) | MCCFR blueprint; nested safe re-solving from the turn with the opponent's actual size in the tree; a self-improver that added the sizes the pros used, between sessions. About 25M core-hours | +147 mbb/hand over 120,000 hands against four top heads-up pros | nested re-solving of off-tree sizes (2.2, 5.1); the self-improver (5.1) |
| **DeepStack** (Alberta, 2017) | continual re-solving with flop and turn value nets; 3 s per action on a GTX 1080. Its turn targets took 175 core-years | +486 mbb/hand with AIVAT (raw +492) over 44,852 hands against 33 pros; LBR lost 383-775 | the architecture this bot already has; AIVAT for people (1.2); LBR as a gate (1.4) |
| **Modicum** (CMU, 2018) | MCCFR blueprint + depth-limited solving; at the leaves the opponent picks among 4 continuation strategies. 700 core-hours, a 4-core CPU | +11 +/-9 against Slumbot, +6 +/-5 against Baby Tartanian8 (95% CI). Single-valued leaves lost | multi-valued leaves for six-max search (4.4) |
| **ReBeL** (Facebook AI, 2020) | self-play RL on public belief states, value and policy nets, CFR-D search; trained on random stacks of 50-250bb; data generation on 720 V100s | +45 +/-5 against Slumbot; +165 +/-69 over 7,500 hands against a top pro; LBR lost 881 | value nets trained on the search's own play (5.1); random stacks in training (3.1) |
| **Supremus** (2020) | DeepStack plus DCFR+, a river net and nets at the end of every street, 4,000-iteration targets, 5M flop / 20M turn / 50M river samples, a GPU solver, more sizes | +176 +/-44 (1 SE) over 150,000 hands against Slumbot. The same paper's DeepStack reimplementation: -63 +/-40 | the closest relative of this bot, and its upgrade list: more and better targets (2.3, 5.1) |
| **AlphaHoldem** (2022) | self-play PPO, no search, 2.9 ms per decision, 580 GPU-hours | +112 +/-16 over 100,000 hands against Slumbot, but +17 +/-22 against OpenStack | results are not transitive: test against several opponents |
| **Student of Games** (DeepMind, 2023) | one algorithm for perfect- and imperfect-information games: growing-tree CFR with value and policy nets | +7 +/-3 over 3.1M hands against the open-source Slumbot2019; LBR lost 434 | nothing specific: general, not stronger |
| **GTO Wizard AI** (from Ruse, 2022 on) | self-play RL, neural values of the future streets, solves one street at a time, any stack up to 200bb, about 3 s per street | +19.4 +/-4.1 bb/100 (+194 mbb/hand) over 150,000 hands against Slumbot | stack-conditioned values with street-by-street solving cover every depth (Phase 3). It is the opponent in the GTO Wizard Benchmark |
| **Slumbot** (Eric Jackson, ACPC 2018 winner) | a large abstraction solved with CFR and played from a table, no search; about 250k core-hours | the standard opponent. LBR beat its 2016 version by 4,020 mbb/hand | the first outside bot (2.4) |

**Since 2024.**
* RL-CFR (ICML 2024) learns action abstractions over a ReBeL-style solver:
  +84 +/-17 against Slumbot. Its ReBeL replication on 8 GPUs reached only
  +18 +/-16.
* EVPA (ICLR 2025): +96 to +187 against Slumbot, depending on time per decision
  and on off-tree sizes. Its DeepStack replication went from -573 at 0.02 s per
  decision to +33 at 2 s.
* Language models: every LLM tested so far loses to GTO Wizard AI. The best on
  its leaderboard is about -5 bb/100.

### 3.2 Benchmarks a bot can play today

Both are heads-up, 50/100, 200bb, stacks reset every hand. Neither plays 100bb.

**Slumbot** (slumbot.com).
* Live in October 2026. Its JSON API exists "so that researchers and hobbyists
  can test their bots". No rate limit is published; hands left idle for 24 h are
  forfeited.
* It reports raw and "baseline" results (minus Slumbot against itself on the
  same cards), not AIVAT.
* Published results against it, mbb/hand (n/r: not reported):

  | agent | result | hands |
  |---|---:|---:|
  | GTO Wizard AI | +194 | 150,000 |
  | Supremus | +176 | 150,000 |
  | AlphaHoldem | +112 | 100,000 |
  | RL-CFR | +84 | >250,000 |
  | ReBeL | +45 | n/r |
  | Baby Tartanian8 | +36 | n/r |
  | Modicum | +11 | n/r |
  | DeepStack reimplementation (Supremus paper) | -63 | 150,000 |

* **A sanity check, not an oracle.** Slumbot is static and translates bet
  sizes; LBR beat its 2016 version by 4 bb/hand, and accounts with unknown
  methods sit above +120 bb/100 on its leaderboard. Do not tune sizes against
  it.

**GTO Wizard Benchmark** (benchmark.gtowizard.com, since March 2026).
* Plays GTO Wizard AI. API key on request; 100,000 hands per user per month;
  every hand scored with AIVAT; ranked by the lower end of the 95% CI.
* Leaderboard on 2026-10-21, AIVAT bb/100 +/- the leaderboard's AIVAT error
  (it shrinks with hands, so it reads as a standard error):

  | entry | hands | AIVAT bb/100 |
  |---|---:|---:|
  | the three best independent bots | 52,000-200,000 | -3.0 +/-0.5, -3.1 +/-0.9, -3.2 +/-0.5 |
  | the only entry marked "Human" | 2,958 | -3.9 +/-3.2 |
  | GPT-6, low reasoning | 10,000 | -5.4 +/-2.4 |
  | always fold (benchmark paper) | | -64.6 |

* No entry is positive. It is the most informative heads-up yardstick there is:
  a strong, fixed opponent, AIVAT built in, and a scale from always-fold to the
  best bots.

### 3.3 Measuring exploitability and luck

* **LBR** (Lisy & Bowling, 2017) tracks the bot's range from its exact action
  probabilities and best-responds with a fixed action set, assuming
  check/call to showdown. A lower bound.
  * ACPC-era abstraction bots: 3,300-4,700 mbb/hand (Hyperborean 2014 4,675,
    Slumbot 2016 4,020, Act1 2016 3,302). Always-fold would lose 750.
  * Search bots: LBR loses. DeepStack -383 to -775, Student of Games -434,
    ReBeL -881, Supremus -951.
  * The action set matters: one bot ranged from 496 to 3,763 across sets.
* **ABR / IS-MCTS best response** (Timbers et al., IJCAI 2022): a learned best
  response. It won 1,259 against a Slumbot-2016 reproduction where LBR found
  522-763.
* **AIVAT** (Burch et al., AAAI 2018): control variates for chance and for the
  player whose strategy is known, plus "imaginary observations" of the hands
  that player could hold. In DeepStack's human study it cut the standard
  deviation by 85%, about 44x fewer hands: +486 +/-40 against raw +492 +/-220.
* **Duplicate** needs both seats dealt by the evaluator. It is impossible
  against Slumbot, GTO Wizard or a person playing live.

### 3.4 Six-max: Pluribus, and after

**Pluribus** (CMU and Facebook AI, *Science* 2019):

* **Game:** six players, 100bb (10,000 chips, 50/100), stacks reset every hand.
* **Blueprint:**
  * Monte Carlo CFR that samples actions;
  * Linear CFR in early iterations only, because the discounting stops paying
    for its multiplications later;
  * actions with very negative regret skipped in 95% of iterations;
  * between 1 and 14 sizes per decision; no card abstraction preflop.
  * 8 days on one 64-core server: 12,400 core-hours, under 512 GB of RAM.
    About $144 at 2019 spot prices, and no GPUs.
  * Sized so a compressed copy fits in 128 GB for play.
* **Search:**
  * played as-is only preflop. Every later round is searched, and preflop too
    after a bet far off the tree; slightly off-tree preflop bets are rounded with
    the pseudo-harmonic mapping;
  * depth-limited. At the leaves each player picks one of 4 continuation
    strategies: the blueprint, and the blueprint biased toward folding,
    calling or raising;
  * a balanced strategy for every hand it could hold. Card abstraction only on
    future rounds;
  * Monte Carlo Linear CFR for big or early subgames, a vector Linear CFR that
    samples only chance otherwise;
  * 2 CPUs (28 cores), under 128 GB; 1-33 s per search, about 20 s per hand in
    self-play.
* **Results** (AIVAT):
  * +48 mbb/hand (SE 25) over 10,000 hands, five pros (from a pool of 13) with
    one Pluribus;
  * +32 (SE 15) over 10,000 hands, one pro with five Pluribus copies (two pros,
    5,000 hands each).
* **Theory:** beyond two players there is no equilibrium guarantee. CFR only
  removes dominated actions, and an exact equilibrium can still lose. Pluribus
  never adapted to its opponents. Six-max strength is measured, not proven.

**After Pluribus:**

* No published extension of ReBeL- or DeepStack-style value nets past two
  players turned up; ReBeL's guarantees hold for two-player zero-sum games
  only.
* Commercial solvers:
  * GTO Wizard has solved three-way postflop since August 2025 (equilibrium
    approximations plus nets: turn Nash distance 0.24% of the pot). A third
    player "can increase computational complexity by a factor of 1000 or more".
  * Since February 2026 it solves multiway preflop for up to 9 players, with at
    most 3 players reaching the flop. A six-max 100bb preflop tree has about
    622,000 nodes, and the three-player cap cut solve time about 20x.
  * MonkerSolver solves any number of players with abstraction; PioSOLVER is
    heads-up only.
* **No public six-max benchmark bot.** Husky Hold'em Bench (NeurIPS 2025) runs
  LLM-written bots in a six-player round robin, with open code: a possible
  sparring pool.

### 3.5 People: win rates, variance, data

* **The population.** 76.9M hands of online no-limit cash games (2009-10,
  611,484 players; Potter van Loon et al., *PLoS ONE* 2015):
  * the median player lost 20 bb/100 before rake and 30 after;
  * 37.5% were winners before rake, 32% after;
  * rake cost about 16 bb/100.

  So "the average player" is a losing player, and a bot that breaks even
  against the population is already above average.
* **What winning looks like.** Pros "consider 50 mbb/g a sizable margin"
  (DeepStack paper). Libratus won 147 against top heads-up pros, and Pluribus
  48 against elite six-max pros.
* **Typical six-max profiles** (VPIP/PFR, % of hands): regulars about 19/17 to
  25/23; losing players around 30-40 with low PFR; the loosest 50-75.
* **Variance.**
  * Six-max raw: about 75-120 bb/100 standard deviation, 7,500-12,000
    mbb/hand.
  * Heads-up 200bb against people: the Libratus and DeepStack matches imply
    about 14,000-24,000 mbb/hand raw.
  * After AIVAT: about 2,500 mbb/hand in Pluribus's experiment and about 2,000
    on the GTO Wizard Benchmark.
* **Data.** The PHH dataset (Zenodo, CC BY 4.0, 20.3 GB) has:
  * 21.6M online no-limit hands from HandHQ (July 2009, 25NL-1000NL, six
    sites). These are anonymized site logs, so hole cards are probably known
    only where shown down: check before relying on them;
  * the ACPC logs, including 279M heads-up no-limit hands (with duplicates);
  * Pluribus's 10,000 hands.

  Uses here: population profiles for the archetypes (1.3), the share of
  heads-up and multiway flops at six-max (4.0), and imitation opponents of old
  ACPC bots if Slumbot goes away.

### 3.6 What this plan takes from them

1. **Measure outside the lineage first.** Every system above was judged
   against people or outside bots, with AIVAT, LBR or both. This bot has
   neither yet (Phase 1).
2. **200bb early.** Both public bot benchmarks play only 200bb (Phase 2).
3. **Scale the targets.** Supremus is this bot's architecture with more and
   better training targets, and it beat Slumbot by 176 where a plain DeepStack
   reimplementation lost. Data and targets are the lever (2.3, 5.1).
4. **Condition on the stack; don't build a bot per depth.** ReBeL trained on
   random 50-250bb stacks, and GTO Wizard trains on random stacks, blinds and
   rake. Below about 10bb, push/fold tables (Phase 3).
5. **Six-max is Pluribus plus this bot's search.** A tabular blueprint is a
   one-off cloud job, and heads-up pots go to the existing search. Limit the
   new multiway machinery to three-way pots (Phase 4).
6. **Results are not transitive.** AlphaHoldem beat Slumbot by 112 but barely
   beat OpenStack. Judge against several opponents and people.
7. **Stay off real-money sites** (section 12).

## 4. The goals as gates

"Average", "better than me" and "professional" need numbers before work can aim
at them. The thresholds below come from section 3. Each gate is met when all of
its items hold. Win rates are for the bot, with 95% CIs, unless stated.

**Gate 1: as good as the average player** (heads-up, 100bb). The average
player loses money (3.5), so this gate asks for clear wins against stand-ins
for that player, no gross hole, and a bot that runs cleanly.

1. **Archetypes:** beats each one over 20,000 duplicate hands with the CI above
   zero, and not by significantly less than the blueprint does against the same
   archetype (a regression check). The blueprint's margins come partly from its
   own leaks, so this takes the exploitation layer (1.6). The layer counts only
   if item 2 holds with it on and its margins against `regular` and
   `population` do not drop.
2. **LBR:** under +1,000 mbb/hand in every configuration. ACPC-era bots gave
   LBR 3,300-4,700.
3. **Soak:** 100,000 hands with 0 crashes and 0 illegal actions, fallbacks under
   0.1% of decisions, p99 decision time under 8 s.
4. **A first human session** (300-500 hands) runs without bugs. Reported, not
   judged.

**Gate 2: better than the author and the average player.**

1. **The author loses** over a sample fixed in advance (at least 2,000 hands;
   the exact number from 1.2's measured variance), scored with AIVAT: the bot's
   CI lies above zero.
2. **Outside anchors at 200bb are on record:** at least 20,000 hands against
   Slumbot (luck-adjusted) and against the GTO Wizard Benchmark (AIVAT).
   Stretch: Slumbot's CI above zero.
3. **Gate 1 still holds** at 100bb, and holds at 200bb.

**Gate 3: six players and any stack.**

*3a, heads-up at any depth (Phase 3):*

1. At 20, 50, 100 and 200bb, the flop and turn spot suites are no more than
   25% worse than 100bb production on the same spot types.
2. Preflop local best response at most about 250 mbb/hand at every depth with a
   preflop table, the level of `preflop_r15.pt`.
3. Push/fold depths: within 5 mbb/hand of the exact equilibrium.
4. Gate 1's archetype and LBR bars at 20, 50 and 200bb.
5. Slumbot and the GTO Wizard Benchmark no worse than at Gate 2.

*3b, six-max (Phase 4):*

1. **Search beats blueprint:** one search bot with five blueprint bots wins
   (CI above zero, seat-rotated deals); five search bots with one blueprint
   bot: the blueprint loses.
2. **Archetype tables:** one bot with five archetypes wins by a wide margin.
3. **People:** the author, and friends who agree, in a fixed sample of one human
   with five bots (Pluribus's 1H+5AI format), scored with AIVAT: the humans'
   CI lies below zero.
4. **Soak:** 100,000 six-max hands with 0 crashes, mixed stacks from 20 to
   200bb included.

**Gate 4: professional.**

*Heads-up (200bb, the benchmark format):*

1. **GTO Wizard Benchmark:** at least -5 bb/100 (AIVAT) over a month's
   100,000 hands, a 95% CI of about +/-1.2 bb/100. That is the band of the best
   independent bots (-3.0 to -3.2) and of the board's one human entry
   (-3.9 +/-3.2 over 2,958 hands). Stretch: the top of the board.
2. **Slumbot:** at least +100 mbb/hand over at least 50,000 hands,
   luck-adjusted, with the CI above zero. Published strong bots: +84 to +194.
3. **LBR loses** in every configuration (CI below zero), as it did against
   DeepStack, ReBeL, Supremus and Student of Games.
4. **A strong human** (a winning heads-up regular) loses over a fixed sample of
   at least 3,000 hands, scored with AIVAT. DeepStack's 33 pros played up to
   3,000 hands each.

*Six-max:*

1. **Pluribus's test, at hobby scale:** consenting strong players in the
   1H+5AI or 5H+1AI format, about 10,000 hands with AIVAT, the bot's CI above
   zero. Pluribus: +48 (SE 25) and +32 (SE 15).
2. **Until such players are available:** wins against population-profile tables
   built from the HandHQ statistics, and no exploit found by deliberate
   exploit hunting.

## 5. Phase 1: yardsticks outside the lineage (goal 1)

**Why first.** Every number before this phase compared the bot with itself or
its blueprint. Goal 1 is about people, and goals 2-4 need the same yardsticks,
so they come first. The first results (section 2) found no bug but a gap: the
search plays sound poker and does not exploit weak play, so this phase also
builds a safe exploitation layer (1.6).

**1.1 A play app.** A local web table (a stdlib HTTP server with long polling,
plain HTML and JavaScript), with the production `SearchAgent` on the GPU. Built
on 2026-10-21 except review mode: `pokerbot/play/README.md`.

* One click per action: fold, check/call, preset sizes (33, 50, 75, 100, 150%
  of the pot), a slider, all-in. The next hand deals itself.
* Stacks reset every hand (to 100bb), as in the Libratus, DeepStack and
  Pluribus experiments, so every hand counts the same.
* Every hand is logged: the deck, the actions, decision times and, at every bot
  decision, the solve's strategy for the whole range and its per-action values.
  1.2, 1.4 and the leak reviews need these.
* Hand histories export in PokerStars text format, so tracking software can
  read them.
* Two modes:
  * **evaluation:** nothing is shown beyond a showdown;
  * **review:** after a session, replay hands with the bot's range and strategy
    at each decision.
* A session panel: hands played, raw and AIVAT win rates with 95% CIs.
* It reuses `eval/match.py` and `pokerbot/protocol` where it can. The terminal
  path stays.

**1.2 AIVAT with the bot's action terms.** A human cannot play duplicate, so
variance reduction carries the human evaluation.

* `eval/luck.py` has AIVAT's chance terms with a check-down value function.
* Add the action terms at the bot's own decisions: `sum_a sigma(a) v(a) -
  v(a_taken)`, with `v` from the search's per-action values (logged by 1.1).
  The term has zero mean for any `v`, because the bot samples its action from
  `sigma`. A better `v` only lowers the variance.
* Validate on bot-against-archetype matches dealt without duplicate: the
  adjusted mean must match the raw mean within its CI. Report the drop in the
  standard deviation. That number sizes every human sample in Phase 2
  (section 10).

**1.3 Archetype opponents,** stand-ins for "the average player"
(`pokerbot/agents/archetypes.py`):

* calling station, nit, maniac, loose-passive limper, fit-or-fold, min-raiser,
  overbet bluffer, and the existing `equity` baseline;
* a population profile whose frequencies (VPIP, PFR, 3-bet, c-bet, fold to
  c-bet, aggression) come from published population statistics or from the
  public HandHQ hands (3.5);
* translation stress: random legal sizes from a min-raise to all-in on every
  street, odd preflop sizes, limp-reraises, donk bets.

Each plays 20,000 duplicate hands against production. Report the margin per
archetype, next to the blueprint's margin against the same archetype.

**1.4 Full-game LBR against production.**

* The search solves for its whole range at every decision. Expose that as a
  `PolicyAgent` (keep the strategy for all 1,326 hands at each decision) so
  `eval/lbr.py` can play against it.
* LBR's candidate raises need the bot's response to each raise it considers.
  Re-run the search there: a few extra searches per LBR decision.
* Configurations: raises `fcpa` and `spec`; preflop `call`, `fc` and `full`.
* Cost: about 3-5x a normal match per hand. 2,000-5,000 hands per configuration
  fits overnight, or on spot GPUs with the `cloud/aws` head-to-head tooling.

**1.5 Soak test and off-tree audit.**

* 100,000 hands against the archetypes (spot GPUs): crashes, illegal actions,
  fallbacks, decision time p50 and p99, memory growth.
* Preflop lines the 31-start tree does not have: every open size from a
  min-raise to all-in, odd 3-bet and 4-bet sizes, limp-reraises. Check the
  bot's defence against each size. The search rebuilds both ranges from the
  preflop table through translation (`range_reach`), so odd preflop lines are
  the likely weak point.

**1.6 A safe exploitation layer** (moved here from 5.1 on 2026-10-23). The
first results show what equilibrium play leaves against weak players (section
2), and Gate 1's regression check needs it back. Behind a flag:

* **Opponent statistics** per session: fold to a bet by street and size, bet
  and raise frequencies, the bluffs shown down, and preflop limps, opens and
  3-bets. Shrink them toward equilibrium frequencies, so the bot moves away
  from equilibrium only as the evidence grows.
* **A modeled opponent in the search.** Fix the opponent's decision nodes in
  each subgame to the model, or mix the model with the free equilibrium
  response (restricted Nash response) to bound the worst case.
  `RangeSolver(locked=...)` already fixes any node, the opponent's included.
  Make the model depend on the cards where the statistics allow: a station
  folds the bottom of its range, not a random part of it. The safe-resolving
  gadget lets the opponent's weak hands drop out of a subgame, which would undo
  the model, so it gives way while the layer exploits.
* **Preflop:** the same statistics adjust the table against limp-heavy and
  raise-heavy opponents.
* **Measured on:** the archetypes, where section 2's gaps should close;
  `regular` and `population`, with no significant loss; LBR with the layer on,
  where Gate 1's bar must still hold; and a wrong model, such as a station
  model against the regular profile, to price a misread opponent.

**Gate 1:** see section 4.

## 6. Phase 2: beat the author, then an outside bot (goal 2)

**2.1 Sessions against the author.**

* Fix the sample before the first hand: for example 3,000 hands in blocks of
  300. The number follows from the standard deviation measured in 1.2
  (section 10). Decide only at the end: stopping on a good run inflates false
  wins.
* Evaluation mode during the sample, review mode afterwards.
* A line that wins for the author is the most useful result of the phase:
  script it as an archetype, measure it, fix it, and add it to the regression
  suite.

**2.2 Preflop, the largest leak measured** (222-252 mbb/hand).

* More rounds of `solve_preflop` (more flops per round, smaller mixing steps),
  each validated with `eval_preflop.py` on fresh flops.
* A richer preflop abstraction: more open, 3-bet and 4-bet sizes. People use
  sizes the 31 flop starts do not have.
* Off-tree preflop lines: today the opponent's range comes from the nearest
  abstract line. Re-solve the preflop subtree below an off-tree action instead,
  with the flop-root values the pipeline already computes (Libratus-style
  nested solving).
  * The audit of 1.5 (`docs/preflop_audit.md`) measured the leak: sizes off
    the solved table fall back to the old blueprint and overfold, 63-70%
    against 2.75bb and 3.5bb opens where table sizes get 40-58%. A first fix is
    to mix the neighbouring table rows by their pseudo-harmonic weights.

**2.3 200bb.** Slumbot plays 200bb, and this is also the first step of goal 3.

* River and turn-end net data with effective stacks up to 250bb. The nets
  already take stack-to-pot features; they have only seen 100bb. Generate with
  a stack distribution, retrain, and check exact leaf error per stack-to-pot
  band.
* Search trees for deep stacks: more raise levels and node budgets per street.
* A 200bb preflop: the same iterated solve, with 4-bet and 5-bet levels,
  started from the 100bb table.
* Cut the dependence on `dcfr4`. Today the search takes its action menus from
  the blueprint (`tree.actions: blueprint`) and half of the river training
  states from blueprint self-play. At 200bb there is no blueprint: the menus
  come from config, flop ranges from the preflop table, and training states from
  the search's own play (the on-policy rounds) plus random and perturbed ranges.

**2.4 Two outside bots at 200bb** (section 3.2 has the details).

* **Slumbot:** a client in `pokerbot/protocol/slumbot.py` that plays an agent
  through Slumbot's HTTP API (`new_hand`, `act`, action strings such as
  `b200c/kk/kk/kb200`). Register a username so the hands count on its
  leaderboard. Play an even number of hands (the first hand starts in the big
  blind). One stream at a time.
* **GTO Wizard Benchmark:** the author requests an API key. 100,000 hands a
  month against GTO Wizard AI, scored with AIVAT, on a public leaderboard. The
  leaderboard's AIVAT errors (0.46 bb/100 over 200,000 hands) imply about
  2,000 mbb/hand per hand, so a full month gives a 95% CI of about
  +/-1.2 bb/100.
* Slumbot reports no AIVAT. Report our luck adjustment next to the raw result:
  the chance terms where Slumbot's cards are shown, and our own action terms
  from 1.2 always.
* Speed: about 3-6 s per hand with today's budgets (the Round 4 head to head
  took 12.3 s per two-hand deal with both sides searching), so 20,000 hands is
  one to two days of the local GPU. Spot GPUs can play GTO Wizard's 100,000
  hands in parallel if its terms allow.

**Gate 2:** see section 4.

## 7. Phase 3: any stack depth, heads-up (goal 3, part 1)

**3.1 Nets for every depth.** Value-net data across 10-250bb effective stacks.
Held-out error per stack-to-pot band; exact leaf checks at 25, 50, 100 and
200bb.

**3.2 Preflop by depth.** Preflop tables on a grid: 15, 20, 25, 30, 40, 50,
60, 80, 100, 125, 150, 200 and 250bb.

* Each is about 16 h of the iterated solve on the 4070 Ti: about two days for
  the whole grid on the four spot GPUs the last known quota allows, less with
  a larger quota.
* Play from the nearest depth first; interpolate later if the grid shows it
  matters.
* Below about 10bb, solve push/fold directly: the tree is small enough for a
  near-exact solution. Between 10 and 15bb, the 15bb table or a small exact
  tree, whichever measures better.

**3.3 Menus by stack-to-pot ratio.** Short stacks need fewer sizes and more
all-ins; deep stacks need more raise levels.

**3.4 Unequal stacks.** Heads-up only the effective stack matters, so this is
a mapping. Check that every piece accepts `stacks` that differ.

**3.5 Evaluation at 20, 50, 100 and 200bb:** spot suites, preflop local best
response, LBR, archetypes. Slumbot again at 200bb.

**Gate 3a:** see section 4.

## 8. Phase 4: six-max (goal 3, part 2)

**Approach.** Pluribus's recipe (an MCCFR blueprint plus depth-limited search),
with one addition this project is well placed for: when only two players see
the flop, the rest of the hand is a two-player game with dead money in the pot,
and the existing heads-up search plays it.

**4.0 Measure first.** Count heads-up, three-way and larger flops in the HandHQ
hands from six-seat tables (PHH dataset). No published figure turned up. The share of
multiway flops decides how much of Phase 4 is reuse and how much is new
search.

**4.1 N-player plumbing.**

* Infoset keys for N players in `engine/src/mccfr.rs` (today the layout is
  heads-up), and an N-player action abstraction in `engine/src/abstraction.rs`
  and its Python twin.
* A six-max match runner: seat rotation (the same deck dealt through the seat
  rotations), N-player chance-term luck adjustment, hand histories.
* `VecNLHE` stays heads-up unless a neural six-max blueprint is tried later.

**4.2 The six-max blueprint.**

* External-sampling MCCFR with linear weighting early and negative-regret
  pruning, as in `DESIGN.md` 5.4 and Pluribus, at 100bb.
* Abstraction: 169 preflop classes; postflop buckets from `build_buckets.py`;
  many preflop sizes (Pluribus used up to 14 per decision), two or three
  postflop sizes.
* Track the average strategy for preflop only, where the blueprint is played
  as it is. Later streets can use the final regret-matching strategy, since
  search re-solves them. That drops a table from every postflop infoset.
* First a coarse run on the i9 (24 cores, 64 GB) to validate the pipeline,
  then the full-size run on a high-memory cloud instance (section 11).

**4.3 Heads-up pots: the existing search.**

* When two players reach the flop, hand the state to the heads-up search with
  both ranges from the six-max blueprint (its preflop plays the role
  `preflop_r15.pt` plays now).
* Dead money: the folded players' chips make the subgame constant-sum, not
  zero-sum. Add it to the trees, the value-net inputs and their training data.
* Position: the player out of position acts first, as in heads-up postflop.
* Value-net data from six-max ranges: one on-policy round.

**4.4 Multiway pots.**

* First the blueprint plays them.
* Then depth-limited search for three-way pots, with Pluribus's continuation
  strategies at the leaves (the blueprint, and the blueprint biased toward
  folding, calling and raising), each player choosing among them. Modicum showed
  single-valued leaves lose.
* Range-vs-range CFR on the GPU needs a three-way showdown kernel. Monte Carlo
  CFR on the CPU, as Pluribus searched, is the fallback.
* Four or more players on the flop: the blueprint, unless 4.0 shows those pots
  matter. GTO Wizard caps its multiway preflop solves at three players to the
  flop for the same reason: each extra player multiplies the cost.

**4.5 Six-max evaluation.**

* One search bot against five blueprint bots, and five against one: the search
  must win in both.
* Tables of archetypes: one bot against five, five against one.
* The author and friends who agree, with AIVAT.

**4.6 Stacks at six-max.** 100bb first. Then 40-200bb with the stacks exact in
the search and the nearest-depth blueprint for preflop; push/fold tables for
short stacks.

**Gate 3b:** see section 4.

## 9. Phase 5: professional level (goal 4)

Open-ended. Work in rounds as now, each led by the largest leak the
yardsticks show.

**5.1 Heads-up strength.** The levers, roughly in order of measured or expected
size:

* **Preflop at decision time.** Preflop is still the weakest street measured.
  DeepStack re-solved preflop at every decision. Two routes:
  * nested solves on the preflop table below each decision, with the flop-root
    values the preflop pipeline already computes;
  * the flop-root net of `docs/plan_round4.md` stage 2, which allows
    depth-limited preflop search. Stage 1 showed each extra net in the chain
    costs accuracy (the turn-start net was about 100 mbb/hand behind), so this
    is the second route.
* **Value nets trained on the search's own play (ReBeL).** The training states
  still come from blueprint self-play plus random and perturbed ranges, with
  on-policy rounds on top. ReBeL closes the loop: self-play with search, and
  nets trained on the search's own subgame values. That takes the blueprint
  out of the data path.
* **More of what the search does:** iterations per second (the fused head and
  CUDA graphs were the start), bigger trees with the sizes people use, better
  leaf nets (they are data-limited, so more data).
* **Off-tree actions on every street:** nested safe re-solving with the actual
  size in the tree (Libratus).
* **A self-improver:** log the sizes opponents use most and add them to the
  trees and the preflop abstraction (Libratus did this between sessions of its
  2017 match).
* **Exploitation beyond 1.6:** opponent models that carry across sessions and
  learn from hand histories, exploitation in multiway pots, and what the layer
  wins against what it gives away, measured at each level of evidence. Pros
  earn most of their money from weak players. This was the last and optional
  item here until the first archetype results (2026-10-23) showed Gate 1 needs
  a first version (1.6).

**5.2 Six-max strength:** a larger blueprint (cloud), finer abstraction where
it pays (preflop sizes by position), more search iterations in multiway pots,
and value-net data drawn from six-max ranges for the heads-up pots.

**5.3 Find holes the way an opponent would.** ABR is too slow against the
search itself. Distil the production policy into a fast network, run ABR
against the copy, and replay any exploit line it finds against the real search.

**5.4 Strong people.** Volunteers who are winning players, heads-up and
six-max, with AIVAT and a fixed sample.

**Gate 4:** see section 4.

## 10. Evaluation ground rules

**Three kinds of evidence, three uses.**

| evidence | examples | decides |
|---|---|---|
| exact exploitability on fixed spots | `spot_eval`, `eval_depth0`, `eval_preflop` | engineering choices: nets, trees, budgets |
| matches between versions | `play_match.py`, `match_search.py`, AWS head to head | adoption: no regressions |
| yardsticks outside the lineage | humans with AIVAT, Slumbot, LBR, archetypes | the goal gates |

**Self-play matches cannot see small gains.** In the Round 4 match against the
blueprint, the per-deal standard deviation was about 6,600 mbb/hand raw and
6,100 with the chance terms (5,000 duplicate deals; 63% of deals cancel
exactly). A 10,000-hand match resolves about +/-170 mbb/hand. Keep using the
spot suites for anything smaller.

**Sample sizes.** A 95% CI of half-width `h` needs about `n = (1.96 s / h)^2`
hands, `s` the per-hand standard deviation:

| s (mbb/hand) | h = 200 | h = 100 | h = 50 |
|---:|---:|---:|---:|
| 2,000 | 400 | 1,500 | 6,100 |
| 4,000 | 1,500 | 6,100 | 25,000 |
| 8,000 | 6,100 | 25,000 | 98,000 |

Raw heads-up results against people are noisier than the bottom row: the
Libratus and DeepStack matches imply 14,000-24,000 mbb/hand at 200bb (3.5).
AIVAT cut DeepStack's by 85%, and the GTO Wizard Benchmark's AIVAT runs at
about 2,000. That is why 1.2 comes before any human sample.

**Fix the sample in advance.** Write the hand count (or a sequential rule
with valid error rates, such as a confidence sequence) into the run's config
before the first hand. Stopping when the numbers look good inflates false
wins, for humans and for bots.

**A regression suite that grows.** The archetypes, every exploit line found by
a person or by LBR, and the spot suites run on every production change.

## 11. Compute and cost

| job | phase | where | size | cost |
|---|---|---|---|---|
| play app, sessions | 1-2 | local GPU | while someone plays | none |
| LBR against the search | 1 | local overnight, or spot `g5.xlarge` | 3-5x a match per hand | under $50 |
| soak test | 1 | spot `g5.xlarge` | 100,000 hands | $20-50 |
| 200bb river and turn-end data | 2 | local GPU, spot for the rest | a few million samples; the river data ran 27-40 samples/s | $20-60 |
| 200bb preflop solve | 2 | local GPU | about 16 h, like attempt 2 | none |
| Slumbot, 20,000-50,000 hands | 2 | local GPU | 1-4 days | none |
| GTO Wizard Benchmark, 100,000 hands | 2, 5 | local GPU or spot | 4-7 GPU-days | up to about $80 on spot |
| preflop grid, 13 depths | 3 | spot `g5.xlarge` | about 210 GPU-hours | about $100 |
| six-max blueprint, coarse | 4 | local i9 | days; must fit 64 GB | none |
| six-max blueprint, Pluribus-size | 4 | spot `r7i.16xlarge` (64 vCPUs, 512 GiB) | 12,400 core-hours: 1-2.5 weeks | $200-500 |
| Supremus-scale net data | 5 | spot `g5.xlarge` | 5-50M samples per street, 10-100x today's | hundreds of dollars |

* `r7i.16xlarge` cost about $1.0-1.3/h on spot in July 2026 ($4.23 on demand),
  with a 15-20% interruption rate. The blueprint run must checkpoint and
  resume.
* The account's GPU spot quota is 32 vCPUs in each of us-east-1 and us-east-2
  (2026-10-22). The six-max run needs a standard-instance spot quota of 64
  vCPUs: request it a few weeks ahead.
* Check prices again before each large run.

## 12. Where the bot may play

**Never on real-money sites.** `DESIGN.md` already lists this as a non-goal,
and the rules are explicit:

* PokerStars bans tools that play "without human intervention" and may
  withhold funds;
* GGPoker bans bots, AI included, "fully and partially automated", with a
  permanent ban and confiscation of funds;
* ACR prohibits bots, with a permanent ban and confiscation.

Beyond the rules, a bot at a real-money table takes money from people who did
not agree to play one.

**Where it can play:**

* **Slumbot,** whose API exists for testing bots.
* **The GTO Wizard Benchmark,** under its terms: an API key on request, play
  only, no solver use.
* **People who know they are playing a bot:** the author, friends,
  volunteers. Research teams paid for strong opponents: DeepStack CAD 5,000 /
  2,500 / 1,250 to its top three, Libratus a $200,000 pool, Pluribus $50,000
  plus $2,000 per player in the one-human test. A hobby version can offer a
  small prize or none.
* MIT Pokerbots (January) plays a secret variant and needs MIT IAP
  registration, so it is not a venue for this bot. The LLM arenas (Kaggle Game
  Arena, PokerBattle.ai) take language models only.

## 13. Risks

| risk | where | mitigation |
|---|---|---|
| Human samples too small to conclude anything | Gates 2-4 | AIVAT with action terms first (1.2); fixed samples; several short sessions; more players |
| People find off-tree lines the bot handles badly | Phases 1-2, 5 | the translation audit (1.5), nested re-solving (2.2, 5.1), every found line in the regression suite |
| The exploitation layer gets exploited: an opponent plays like a station to draw it out, then changes | Phases 1-2, 5 | statistics shrunk toward equilibrium and weighted to recent hands; a bounded worst case (restricted Nash response); LBR and the regular profile with the layer on (1.6) |
| Slumbot's API changes or closes | Gate 2 stretch, Gate 4 | the archetypes, LBR and people still stand; ACPC logs in the PHH dataset can train imitation opponents of old ACPC bots |
| One GPU for the play app, data generation and evaluation | all | batch jobs on spot GPUs; the play app gets the local GPU while someone plays |
| A six-max blueprint does not fit 64 GB | Phase 4 | a coarse local run first; the average strategy kept for preflop only; a high-memory cloud instance for the full run |
| Multiway search has no exact reference to test against | Phase 4 | exact solves of small three-way spots and toy games; the blueprint as the fallback in multiway pots |
| Six-max has no public benchmark bot | Gates 3b, 4 | search vs blueprint-only tables, archetype tables, people |
| Self-play numbers mislead | all | spot suites for engineering, outside yardsticks for gates (section 10) |
| Scope: goal 4 has no natural end | Phase 5 | gates per round; stop a line of work when its gate fails, as with the turn-start net |

## 14. The next two weeks

Updated 2026-10-23. The tools of 1.1-1.5 are built (see the status line).

1. **Finish Gate 1's runs:** the archetypes, LBR and a 20,000-hand soak, on
   spot GPUs and the local GPU. About 3-4 more days at the spot capacity of
   2026-10-23.
2. **The exploitation layer** (1.6): a prototype against the calling station
   and the maniac first, then the archetype suite and LBR with it on.
3. **The preflop table against loose, aggressive players:** the table against
   the blueprint's preflop, both followed by the search, on the maniac and the
   overbet bluffer. Section 2's split put the blueprint after the table, which
   was not trained for the table's ranges.
4. **A first session against the author**, 300-500 hands, to shake out the app.
   Reported, not gated.
5. **In parallel on spot instances:** start the 200bb river data (2.3).

**Decisions for the author along the way:**

* **Cash games or tournaments.** This plan assumes cash games: chips are money,
  no ICM. Tournaments would add ICM, antes and much shorter stacks after
  Phase 3.
* **Cloud budget** for Phase 4's blueprint (section 11).
* **Who else plays the bot:** friends or volunteers, for Gates 2-4.

## Sources

Checked on 2026-10-21.

**Heads-up systems**
* Libratus: [Science 2017](https://noambrown.com/papers/17-Science-Superhuman.pdf), [IJCAI 2017](https://www.ijcai.org/proceedings/2017/0772.pdf)
* DeepStack: [arXiv 1701.01724](https://arxiv.org/abs/1701.01724)
* Modicum, depth-limited solving: [arXiv 1805.08195](https://arxiv.org/abs/1805.08195)
* ReBeL: [arXiv 2007.13544](https://arxiv.org/abs/2007.13544)
* Supremus: [arXiv 2007.10442](https://arxiv.org/abs/2007.10442)
* AlphaHoldem: [AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20394)
* Student of Games: [arXiv 2112.03178](https://arxiv.org/abs/2112.03178)
* RL-CFR: [arXiv 2403.04344](https://arxiv.org/abs/2403.04344)
* EVPA: [ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/file/8c1b5863a6b0f925617b917bb2f55be0-Paper-Conference.pdf)
* GTO Wizard AI: [explained](https://blog.gtowizard.com/gto-wizard-ai-explained/), [benchmarks](https://blog.gtowizard.com/gto-wizard-ai-benchmarks/)

**Benchmarks**
* Slumbot: [site](https://slumbot.com/), [API sample](https://slumbot.com/sample_api.py)
* GTO Wizard Benchmark: [site](https://benchmark.gtowizard.com/), [paper, arXiv 2603.23660](https://arxiv.org/abs/2603.23660), [leaderboard](https://researcher.gtowizard.com/leaderboard?game_name=HUNL%20200BB&limit=30)
* LLM arenas: [Kaggle Game Arena report](https://arxiv.org/abs/2609.31473), [PokerBattle.ai](https://pokerbattle.ai/), [Husky Hold'em Bench](https://neurips.cc/virtual/2025/122372)

**Evaluation**
* LBR: [arXiv 1612.07547](https://arxiv.org/abs/1612.07547)
* Approximate best response / IS-MCTS-BR: [arXiv 2004.09677](https://arxiv.org/abs/2004.09677)
* AIVAT: [arXiv 1612.06915](https://arxiv.org/abs/1612.06915)

**Six-max**
* Pluribus: [Science 2019](https://par.nsf.gov/servlets/purl/10119653), [Meta blog](https://ai.meta.com/blog/pluribus-first-ai-to-beat-pros-in-6-player-poker/)
* GTO Wizard multiway: [three-way postflop](https://blog.gtowizard.com/now_live_3_way_solving_nodelocking_2_0_and_50k_icm_ft_sims/), [multiway preflop](https://blog.gtowizard.com/gto-wizard-ai-custom-multiway-solving/)
* [MonkerSolver](https://monkerware.com/solver.html), [PioSOLVER](https://piosolver.com/products/)

**People and data**
* Population win rates: [Potter van Loon et al., PLoS ONE 2015](https://pmc.ncbi.nlm.nih.gov/articles/PMC4346402/)
* Player profiles: [Poker Copilot statistics guide](https://pokercopilot.com/assets/poker_statistics_guide.pdf)
* Variance: [PrimeDope variance calculator](https://www.primedope.com/poker-variance-calculator/)
* PHH dataset: [Zenodo](https://zenodo.org/doi/10.5281/zenodo.10796885), [GitHub](https://github.com/uoftcprg/phh-dataset)

**Rules and venues**
* [PokerStars prohibited tools](https://www.pokerstars.fr/en/poker/room/prohibited/), [GGPoker security policy](https://ggpoker.com/network/security-ecology-policy/), [ACR security](https://www.acrpoker.eu/security/)
* [MIT Pokerbots](https://pokerbots.org/)

**Cloud prices**
* [r7i.16xlarge prices](https://calculator.holori.com/aws/ec2/r7i.16xlarge)
