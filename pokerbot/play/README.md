# `pokerbot.play`: play the bot in a browser

A local web table where a person plays the bot heads-up, one hand at a time,
with every hand logged for evaluation. This is item 1.1 of `docs/roadmap.md`:
the measurement against people that Goals 1 and 2 need.

```bash
python scripts/play_web.py                      # the production bot, a new session
python scripts/play_web.py --hands 300          # a sample fixed at 300 hands
python scripts/play_web.py --resume runs/play/20261021-140509
python scripts/play_web.py --bot equity --no-warmup   # a cheap opponent, to try the app
```

The script prints the address (`http://127.0.0.1:8765/` unless that port is
busy) and opens it. The bot is `match.a` of `--config`
(`configs/match_search_vn.yaml`: the solved preflop table and value-net search
on the GPU), built with that file's `agents:` settings. A warm-up hand against a
calling station runs first, so the first real decisions do not pay for loading
the nets and compiling kernels. Ctrl+C stops the server and prints the session's
result from the bot's side.

## Evaluation mode

The only mode so far. The page shows what a player at the table sees: your
cards, the board, stacks, bets, the actions of the hand, and the bot's cards
only when they are shown down. Nothing about the bot's strategy or ranges ever
reaches the page.

* Stacks reset to 100bb every hand and the button alternates.
* Deals come from the session's seed (random by default, kept in
  `session.json`): hand `h` uses the deck `default_rng((seed, h, 1))`.
* **A fixed sample.** `--hands N` ends the session after `N` finished hands.
  Decide `N` before the first hand (`docs/roadmap.md`, section 10): stopping on a
  good run inflates false wins.
* **Several sittings.** `--resume <dir>` continues a session: same bot, same
  seed, same sample size. Every hand is noted in `dealt.jsonl` before it is
  played, so a hand cut short (Ctrl+C mid-hand) is counted as abandoned and its
  deck is never dealt again. The session panel shows the count.
* **Keys:** `F` fold, `C` check or call, `R` raise size, `Enter` raise, `1`-`6`
  presets, `N` next hand. Auto-deal (2.5 s after a fold, 5 s after a showdown)
  and hiding the session statistics are remembered in the browser. The side
  panel keeps the last hand's result, with the bot's cards if they were shown.

The session panel shows hands played, your net result, and your win rate in
bb/100 with a 95% bootstrap CI three ways: raw, luck-adjusted and AIVAT. The
luck adjustment is AIVAT's chance terms (`pokerbot/eval/luck.py`: all-in
equity and street equity changes, with both hands known). AIVAT adds the bot's
action terms (`pokerbot/eval/aivat.py`, roadmap item 1.2): at every searched
bot decision, `sum_a sigma(a) v(a) - v(a_taken)` with `v` the solve's value of
each action for the bot's hand against your actual hand. All three estimate
the same win rate, with less variance in that order (see below). Preflop
decisions come from the solved table, which has no values, and get no action
term.

### Scoring a session with AIVAT

```bash
python scripts/score_play_session.py runs/play/20261021-140509
```

prints your win rate and the bot's, raw / luck-adjusted / AIVAT, with 95% CIs
and per-hand standard deviations. It recomputes the action terms from the
logged arrays (`d<k>_strategy`, `d<k>_values_vs`). Sessions logged before
`values_vs` was recorded are scored with the range values (`d<k>_values`),
which remove almost nothing (the panel counts those hands with no action
terms). `--values range` forces the range values.

**Measured** (`scripts/eval_aivat.py`, a tiny CPU search bot at 32
iterations per solve, 3000 hands against each of `equity`, `random` and
`always_call`; details in `pokerbot/eval/README.md`): per-hand SD in bb/hand
raw / luck-adjusted / AIVAT 51.3 / 28.3 / 27.3, 51.0 / 31.1 / 30.8 and
74.1 / 45.7 / 44.0. The chance terms cut the hands needed for a given CI by
about 3x. The action terms take off a further 2-8% here, and the AIVAT mean
agreed with the raw mean in all three. The production bot against a person
has not been measured yet: that needs a GPU and a person.

## The session directory

`runs/play/<session>/` (`log.py`):

| file | contents |
|---|---|
| `session.json` | the game, the bot spec with its full search config, the seed, the sample size, the git commit, start and resume times |
| `dealt.jsonl` | one line per dealt hand, written before it is played |
| `hands.jsonl` | one line per finished hand: deck, button, hole cards, board, every action with its decision time, payoffs, your raw, luck-adjusted (`human_adjusted`) and AIVAT (`human_aivat`) result, the bot's summed action terms (`bot_action_terms`), and a summary of each bot decision (the actions it had, the one it took, timings, search stats) |
| `solves/hand_<n>.npz` | the bot's arrays for hand `n`, decision `k`: `d<k>_strategy` `[1326, A]` (its strategy for every combo over the `A` actions it had), `d<k>_reach` `[2, 1326]` (both players' reach at the decision, seat order) and, for searched decisions, `d<k>_values` `[1326, A]` and `d<k>_values_vs` `[1326, A]` |
| `hands.txt` | the repo's plain-text hand histories (`pokerbot/eval/history.py`) |
| `pokerstars.txt` | PokerStars-format histories from your seat, for tracking software. Whether a given tracker imports them is untested. |
| `errors.log` | the traceback if the session stopped on an error |

`d<k>_values[c, a]` is the bot's expected chips for the whole hand with combo
`c` after action `a`, under the solved strategies: the counterfactual value of
that child divided by the opponent's reach disjoint from `c` (NaN where that is
zero). `d<k>_values_vs[o, a]` is the same for the bot's actual combo against
opponent combo `o` (NaN where `o` shares a card with it; averaged by the
opponent's reach it gives `d<k>_values[c, a]`). AIVAT's action terms read
`values_vs`; LBR against the search (1.4) can read these arrays. Preflop decisions come from the solved table and have a strategy and
reach but no values.

Recording is `SearchAgent(record=True)` (`pokerbot/search/agent.py`). It costs
two extra value passes per searched decision (plus one value-net pass at the
leaves for `values_vs`) and never changes the action
played (`tests/search/test_agent.py`).

## Design notes

* `session.py`: `PlaySession` runs the hand loop on its own thread. The
  person's actions come in through `submit_action` (validated against the legal
  actions, then checked by the engine). The browser reads `snapshot`, a
  versioned public view.
* `server.py`: a stdlib `ThreadingHTTPServer`. The page long-polls
  `GET /api/state?since=<version>` and posts `/api/action`, `/api/next` and
  `/api/stop`. A turn-based game needs no websocket, and the app needs no new
  dependency.
* It listens on 127.0.0.1 and answers only requests whose `Host` names this
  machine, so another web page cannot drive it; POST bodies must be JSON.
* `static/`: plain HTML, CSS and JavaScript, no build step and no network
  access.

## Limits

* Heads-up only (Phase 4 adds six-max).
* Review mode (replaying a session with the bot's ranges and strategies) is not
  built yet. The data it needs is all in the session directory.
* The bot plays one hand at a time on the local GPU. A decision takes up to the
  search budget: 4 s on the flop, 2 s on the turn, 1 s on the river.

Tests: `tests/play/` (session, server, PokerStars export), and the recording
test in `tests/search/test_agent.py`.
