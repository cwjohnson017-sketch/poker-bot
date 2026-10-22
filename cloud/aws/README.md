# AWS pilot

Run one GPU instance, reproduce known local numbers, measure speed, and terminate.
The scripts run from Git Bash on the local machine. They use the configured AWS CLI.
Settings are in `common.sh`:
* region `us-east-2`;
* `g5.2xlarge`: 8 vCPUs, A10G 24 GB, 32 GB RAM, $1.212/h on-demand on 2026-10-10;
* the Deep Learning Base AMI (Ubuntu 22.04).

```bash
bash cloud/aws/pilot_launch.sh      # billed from here; self-terminates after MAX_HOURS (4)
bash cloud/aws/pilot_sync.sh        # code at HEAD, blueprint + tables, nets (~3.3 GB)
source cloud/aws/common.sh && load_state
remote_detached setup bash poker-bot/cloud/aws/remote_setup.sh    # ~5 min; log ~/setup.log
remote_detached pilot bash poker-bot/cloud/aws/remote_pilot.sh    # ~40 min; poker-bot/pilot/summary.txt
bash cloud/aws/pilot_terminate.sh   # results to runs/aws/pilot/, then terminate
```

What the pilot checks (`remote_pilot.sh`), against the local RTX 4070 Ti:

| check | local reference |
|---|---|
| test suite, `-m "not slow"` | 498 passed |
| exact turn-end leaf check, round-2 on-policy states | river fan-out 0.0197, `turn_v3w` 0.0354, `turn_v4e` 0.0315 MAE |
| turn-start data generation, `turn_v4e` leaves | 11.6 samples/s |
| flop decision, production config | 58 DCFR iterations in 4 s |

**State and setup.**
* The launched instance is recorded in `runs/aws/<name>.env`.
* The SSH key pair (`~/.ssh/pokerbot-pilot.pem`) and a security group that
  allows SSH from this machine's IP only are created once and reused.

**Time budgets depend on the GPU.** Compare across machines with fixed iteration
counts, or re-time first.

# Fleet runner (`fleet.py`)

Runs the long Phase 1 jobs of `docs/roadmap.md` on spot GPU instances (and the local
GPU with `--local`): `calibrate`, `archetypes`, `lbr` and `soak`. A job is split into
resumable blocks, each a directory `runs/fleet/<job>/blocks/<block>/` that the same
command fills here or on an instance (`run_block.py`). Launches are EC2 Fleet `instant`
requests over several instance types and every zone, with the `price-capacity-optimized`
allocation; AWS calls the older Spot Fleet API legacy. The guards of `h2h_coordinator.py`
carry over: a spend budget, self-termination after `--max-hours`, idle instances
terminated, restart from `state.json`. See the docstring for every option.

```bash
git switch -c <branch> && git commit ...    # the instances run the bundle's commit
REF=<branch> bash cloud/aws/fleet_bundle.sh # code at REF + nets, table, blueprint (~1 GB)
python cloud/aws/fleet.py --job calibrate --types g5.xlarge g6.xlarge --budget 3
python cloud/aws/fleet.py --job archetypes --dry-run      # the blocks, nothing launched
python cloud/aws/fleet.py --job archetypes --budget 150 --slots 2 --local
python cloud/aws/fleet.py --job archetypes --merge-only   # summary.md of what finished
```

**Same strength on every GPU.** The search stops on a time budget, so a slower GPU
would play a weaker bot. Every job but `calibrate` plays
`configs/search_value_net_fixed.yaml` instead: production with the iterations per street
fixed at the 4070 Ti's means (`street_iterations` in `pokerbot/search/config.py`). On
other GPUs it only takes longer. The soak's decision-time check means something only on
hardware like the 4070 Ti (the A10G of `g5`).

**Calibration** (`scripts/calibrate_search.py`): the same 24 hands against `always_call`,
once with production's budgets (iterations per street this GPU fits) and once with the
fixed iterations (its seconds per hand, so its cost). `--job calibrate` runs both on one
instance of each `--types`.

**Before a run:** spot quota for G instances (`L-3819A6DF`, vCPUs per region), the key
pair of each region in `KEYS` (`~/.ssh/<key>.pem`), the `pokerbot-ssh` security group.
