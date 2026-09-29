# Run configs

Every run is **a config file plus a git commit hash**. Nothing that changes
results lives only on the command line or in someone's shell history.

## Convention

- One YAML file per experiment, checked in under `configs/`. Name it after
  what it does (`match.yaml`, `mccfr_hu_1000b.yaml`, `deepcfr_v2.yaml`).
- Top-level sections by component: `engine`, `game`, then one section per
  subsystem (`match`, `agents`, `mccfr`, `deepcfr`, `search`, ...).
- The `game` section is always the same shape and maps onto `GameConfig`:
  `num_players`, `stacks`, `small_blind`, `big_blind`, `ante`. Defaults are the
  ACPC heads-up game (50/100, 20,000-chip stacks).
- Seeds live in the config (`seed:`). A run with the same config, commit and
  engine must reproduce its numbers.
- Command-line flags may override config values for quick experiments. Any
  result you keep should be re-run from a committed config.
- Scripts print (and store with their outputs) the provenance from
  `pokerbot.config.run_info()`: the git hash (with `-dirty` if the tree had
  uncommitted changes), the config path and the engine. Checkpoints must
  carry the same record.

## Files

| File | Used by | What |
|---|---|---|
| `match.yaml` | `scripts/play_match.py --config configs/match.yaml` | Duplicate match between two agents, with per-agent constructor arguments under `agents:`. |
| `deepcfr_tiny.yaml` | `scripts/train_deepcfr.py --config configs/deepcfr_tiny.yaml` | Deep CFR smoke test on the CPU (20bb, reduced action set, tiny nets; about 1 s per iteration). |
| `deepcfr_4070ti.yaml` | `scripts/train_deepcfr.py --config configs/deepcfr_4070ti.yaml` | The neural blueprint run on one RTX 4070 Ti, with VRAM, RAM and time estimates in the header. |
| `ladder.yaml` | `scripts/run_ladder.py --config configs/ladder.yaml` | Checkpoint ladder: agent list or checkpoint glob, schedule, hands per pair, workers. |
| `lbr.yaml` | `scripts/run_lbr.py --config configs/lbr.yaml` | Local best response against one opponent spec. |
| `abr_tiny.yaml` | `scripts/run_abr.py --config configs/abr_tiny.yaml` | Approximate best response, CPU smoke run (~20 s). |
| `abr_4070ti.yaml` | `scripts/run_abr.py --config configs/abr_4070ti.yaml --opponent <spec>` | Approximate best response sized for one RTX 4070 Ti. |
| `mccfr_small.yaml` | `scripts/train_mccfr.py --config configs/mccfr_small.yaml` | Small MCCFR blueprint (100bb, reduced actions, 20 postflop buckets); about a minute on 4 cores. Smoke runs and tests. |
| `mccfr_hunl.yaml` | `scripts/train_mccfr.py --config configs/mccfr_hunl.yaml` | The Phase 1 blueprint: 100bb, DESIGN.md action table, 169/1000/1000/1000 buckets; 6-10 GB RAM, 1-1.5 days on 16 cores. |
