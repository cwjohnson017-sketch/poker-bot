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
