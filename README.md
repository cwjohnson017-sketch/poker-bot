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

## Status

Every phase of the design is implemented and tested on CPU. Nothing has been
run on a GPU yet: the CUDA configs (`*_4070ti.yaml`) carry estimates, not
measurements, and the first real run happens on the target machine.

| Phase | Component | State |
|---|---|---|
| 0 | Rust engine, torch vectorized env, reference engine | Done. The three agree on random hands; the Rust engine is the source of truth. |
| 1 | Tabular MCCFR blueprint (Rust solver) | Done. The small config beats the equity baseline by about +660 to +820 mbb/h in duplicate matches. The full run needs the bucket tables below. |
| 1 | Card bucket tables (equity histograms, EMD k-means) | Done. A 1000/1000/1000 build is estimated at 1 to 2 hours on the 4070 Ti. |
| 2 | Deep CFR neural blueprint (SD-CFR averaging) | Done. Traversal values verified against brute force; untested at scale. |
| 3 | Real-time search (range CFR, depth limit, safe gadget) | Done. Converges on river and turn subgames; per-decision speed with a real blueprint is the open risk. |
| - | Evaluation: duplicate matches, ladder, LBR, approximate best response | Done. |

## What is here

| Path | Contents |
|---|---|
| `engine/` | Rust crate: rules, hand evaluator, suit isomorphism, action abstraction, the MCCFR solver, PyO3 bindings (`poker_engine`). See [`engine/README.md`](engine/README.md). |
| `pokerbot/reference/` | Pure-Python rules engine and hand evaluator with the same API, used as ground truth in tests. |
| `pokerbot/env/` | Torch vectorized heads-up environment, batched evaluator, equity kernels, observation encoder. See [`pokerbot/env/README.md`](pokerbot/env/README.md). |
| `pokerbot/abstraction/` | Canonical action spec (Python and Rust in agreement), pseudo-harmonic off-tree mapping, canonical hand indexing, bucket-table builder. See [`pokerbot/abstraction/README.md`](pokerbot/abstraction/README.md). |
| `pokerbot/blueprint/mccfr/` | Tabular blueprint training driver, `BlueprintAgent`, strategy files. See [`pokerbot/blueprint/mccfr/README.md`](pokerbot/blueprint/mccfr/README.md). |
| `pokerbot/blueprint/deepcfr/` | Deep CFR: frontier traversal, reservoir memories, networks, trainer, `NeuralBlueprintAgent`. See [`pokerbot/blueprint/deepcfr/README.md`](pokerbot/blueprint/deepcfr/README.md). |
| `pokerbot/search/` | Subgame tree, range-based DCFR/CFR+, showdown kernel, leaf rollouts, safe resolving gadget, `SearchAgent`. See [`pokerbot/search/README.md`](pokerbot/search/README.md). |
| `pokerbot/agents/` | The `Agent` protocol, baselines, the `PolicyAgent` protocol, and the spec registry (`equity:samples=100`, `blueprint:<file>`, `neural:<dir>`, `search:<blueprint>`). |
| `pokerbot/eval/` | Match runner, duplicate matches, bootstrap CIs, checkpoint ladder, LBR, approximate best response, logging. See [`pokerbot/eval/README.md`](pokerbot/eval/README.md). |
| `pokerbot/protocol/` | ACPC-style line protocol over TCP: codec, match server, client. |
| `configs/` | YAML run configs, one per experiment. See [`configs/README.md`](configs/README.md). |
| `scripts/` | Entry points: `build_buckets.py`, `train_mccfr.py`, `train_deepcfr.py`, `play_match.py`, `run_ladder.py`, `run_lbr.py`, `run_abr.py`, `serve_match.py`, `play_client.py`. |
| `tests/` | CPU pytest suite (about 260 tests). |

## Training on the RTX 4070 Ti

Run the phases in this order. Each step has a tiny CPU config next to the
real one, so run the tiny one first to check the plumbing.

```bash
# 1. Card buckets (GPU, about 1-2 h). Print the real per-street rate first.
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --limit 20000 --device cuda
python scripts/build_buckets.py --config configs/buckets_hunl.yaml --device cuda --out data/buckets

# 2. Tabular blueprint (CPU, 1-1.5 days on 16 cores, 6-10 GB RAM).
#    Point mccfr.cards.tables in configs/mccfr_hunl.yaml at the tables from step 1.
python scripts/train_mccfr.py --config configs/mccfr_hunl.yaml --threads 16 --out-dir runs/mccfr_hunl

# 3. Evaluate it against the baselines and with LBR.
python scripts/play_match.py --a blueprint:runs/mccfr_hunl/strategy.bin --b equity --hands 200000 --duplicate
python scripts/run_lbr.py --opponent blueprint:runs/mccfr_hunl/strategy.bin --hands 2000 --workers 8

# 4. Neural blueprint (GPU, days). Watch the ladder; adopt only if it beats step 2.
python scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml --out runs/deepcfr --device cuda
python scripts/run_ladder.py --agents blueprint:runs/mccfr_hunl/strategy.bin neural:runs/deepcfr --hands 200000

# 5. Search on top of the adopted blueprint.
python scripts/play_match.py --a search:blueprint:runs/mccfr_hunl/strategy.bin --b blueprint:runs/mccfr_hunl/strategy.bin --hands 20000 --duplicate
```

The CUDA configs were sized from CPU measurements and the card's specs. If a
step runs out of VRAM or over its time budget, the README of that component
says which knobs to lower.

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

The scalar-engine tests default to the reference engine
(`POKERBOT_ENGINE=reference`), so the core suite passes whether or not the
Rust engine is built. When `poker_engine` is importable, the cross-engine
tests also run: `tests/test_cross_engine.py` replays random 2-9 player hands
through the reference and Rust engines, and `tests/test_cross_env_engine.py`
replays heads-up hands through the torch environment and the Rust engine.
The blueprint, abstraction and search tests need the Rust engine and are
skipped without it. CI (`.github/workflows/ci.yml`) runs ruff and the CPU
suite.

## Playing matches

```bash
# duplicate match: mbb/hand for A with a 95% bootstrap confidence interval
python scripts/play_match.py --a always_call --b equity --hands 2000 --duplicate --seed 0

# the same match from a config file, writing hand histories
python scripts/play_match.py --config configs/match.yaml --history hands.txt
```

Agent specs are parsed by `pokerbot/agents/registry.py`: `always_call`,
`always_raise`, `random`, `equity` (`equity:samples=100`), `human`,
`blueprint:<strategy.bin>` (needs the Rust engine), `neural:<run dir>`, and
`search:<blueprint spec>` such as `search:uniform` or
`search:blueprint:<strategy.bin>`. A duplicate match plays
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
