# poker-bot

A heads-up No-Limit Texas Hold'em bot that learns only from self-play. It is
built to train on one consumer GPU (an RTX 4070 Ti). The plan, following
[`DESIGN.md`](DESIGN.md):

1. A tabular MCCFR blueprint on the CPU.
2. A neural Deep CFR blueprint on the GPU.
3. Real-time subgame search.

An evaluation harness based on duplicate matches decides whether each new
version is adopted.

Every component codes against the interface contract in
[`docs/INTERFACES.md`](docs/INTERFACES.md). The contract has two
implementations:

- `engine/` is a Rust crate with PyO3 bindings, imported as the Python module
  `poker_engine`. It is fast.
- `pokerbot/reference/` is pure Python. It is slow, but simple enough to use
  as the ground truth in tests.

## What is here

| Path | Contents |
|---|---|
| `pokerbot/reference/` | Pure-Python rules engine and hand evaluator: `GameConfig`, `Action`, `LegalActions`, `GameState`, `evaluate5/6/7`, `evaluate_batch`, `hand_category`, card helpers. |
| `pokerbot/engine_select.py` | `get_engine()` returns `poker_engine` when it is built and `pokerbot.reference` otherwise. `POKERBOT_ENGINE=reference\|rust\|auto` overrides the choice. |
| `pokerbot/agents/` | The `Agent` protocol and the baselines `AlwaysCallAgent`, `AlwaysRaiseAgent`, `RandomAgent`, `EquityThresholdAgent` and `HumanCLIAgent`. |
| `pokerbot/eval/` | `play_hand`, `run_match` and `run_duplicate_match`, the hole-card masking wrapper, mbb/h with a bootstrap CI (`stats.py`), and hand-history export (`history.py`). |
| `pokerbot/blueprint/mccfr/` | Tabular MCCFR blueprint: training driver, `BlueprintAgent`, strategy files. The solver is Rust (`engine/src/mccfr.rs`). See [`pokerbot/blueprint/mccfr/README.md`](pokerbot/blueprint/mccfr/README.md). |
| `pokerbot/protocol/` | ACPC-style line protocol over TCP: the message codec, a match server, and a client. |
| `configs/` | YAML run configs. See [`configs/README.md`](configs/README.md). |
| `scripts/` | Entry points: `play_match.py`, `serve_match.py`, `play_client.py`, `train_mccfr.py`. |
| `tests/` | CPU pytest suite. |

## Setup

This needs Python 3.11+, [uv](https://docs.astral.sh/uv/), and a Rust stable
toolchain for the fast engine.

```bash
uv venv --python 3.11 .venv
source .venv/bin/activate

# PyTorch: pick the wheel for your machine
uv pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA 12
# uv pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU only

uv pip install -e ".[dev]"

# Optional: build the Rust engine into the venv (Python falls back to the
# reference engine without it)
cd engine && maturin develop --release && cd ..
```

## Tests and lint

```bash
make test        # pytest -m "not slow" (under a minute on a laptop CPU)
make test-all    # also runs the full 2.6M five-card evaluator enumeration
make lint        # ruff check + ruff format --check
```

The tests always use the reference engine (`POKERBOT_ENGINE=reference`), so
they pass whether or not the Rust engine is built. When `poker_engine` is
importable, `tests/test_cross_engine.py` also replays random 2-9 player hands
through both engines. It checks that states, legal actions, payoffs and keys
are identical. CI (`.github/workflows/ci.yml`) runs ruff and the CPU suite.

## Playing matches

```bash
# duplicate match: mbb/hand for A with a 95% bootstrap confidence interval
python scripts/play_match.py --a always_call --b equity --hands 2000 --duplicate --seed 0

# the same match from a config file, writing hand histories
python scripts/play_match.py --config configs/match.yaml --history hands.txt
```

The available agents are `always_call`, `always_raise`, `random`, `equity`,
`human` and `blueprint:path/to/strategy.bin` (an MCCFR blueprint trained with
`scripts/train_mccfr.py`; needs the Rust engine). A duplicate match plays
every deal twice with the same cards, once from each seat. Results are in milli-big-blinds per hand. The confidence
interval comes from bootstrapping over deals.

To play against a bot yourself over the socket protocol:

```bash
python scripts/serve_match.py --seat0 remote --seat1 equity --hands 50
python scripts/play_client.py --agent human          # in a second terminal
```

At the prompt, type `f` to fold, `c` or `k` to check or call, `r <amount>` to
raise to that street total, or `a` to go all-in.

## Conventions

- Cards are `rank * 4 + suit` (`2c` = 0, `As` = 51). A deck is dealt in order:
  seat 0's hole cards, then seat 1's, and so on, then the flop, turn and river.
- `Action.raise_to(x)` means "raise to `x` chips committed on this street".
- Agents get a masked view of the state. Other seats' `hole_cards` return
  `[]`. `clone()` and `child()` re-sample every hidden card, so an agent can
  search without seeing the real deck.
- Every run is a config file plus a git commit hash. See
  [`configs/README.md`](configs/README.md).
