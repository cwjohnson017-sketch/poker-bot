#!/usr/bin/env bash
# Runs ON the instance after remote_setup.sh: checks with known local answers, and speed.
# Results in ~/poker-bot/pilot/ (summary.txt). Local references (RTX 4070 Ti):
#   check_turn_leaves, round-2 on-policy states: river fan-out MAE 0.0197, turn_v3w 0.0354,
#     turn_v4e 0.0315 (exact solves; tiny float differences across GPUs are fine)
#   gen_turn_start_data with turn_v4e leaves: 11.6 samples/s
#   time_flop_search, production config: 58 DCFR iterations per 4 s flop decision
set -uo pipefail
cd ~/poker-bot
source .venv/bin/activate
export PYTHONPATH=.
mkdir -p pilot
BP=runs/dcfr4_distilled_v2
V=runs/value_net
step() { echo "== $(date +%T) $*" | tee -a pilot/summary.txt; }

step "tests (CPU, not slow)"
python -m pytest tests -m "not slow" -q -p no:cacheprovider > pilot/tests.log 2>&1
tail -1 pilot/tests.log | tee -a pilot/summary.txt

step "exact turn-end leaf check (expect 0.0197 / 0.0354 / 0.0315)"
python scripts/check_turn_leaves.py --river-net $V/river_v3.pt --turn-net $V/turn_v3w.pt \
    --turn-net $V/turn_v4e.pt --blueprint $BP --states 400 --iterations 400 \
    --states-file $V/river_onpolicy2_heldout/turn_states.pt --out pilot/leaf_check.json \
    > pilot/leaf_check.log 2>&1
grep "MAE" pilot/leaf_check.log | grep -v "    " | tee -a pilot/summary.txt

step "turn-start data rate (expect about 11.6 samples/s locally)"
python scripts/gen_turn_start_data.py --turn-net $V/turn_v4e.pt --blueprint $BP \
    --out pilot/turn_start --samples 8192 --iterations 300 --batch 512 --seed 7 \
    > pilot/turn_start.log 2>&1
grep "shard_" pilot/turn_start.log | tail -1 | tee -a pilot/summary.txt

step "flop decision timing (expect about 58 iterations in 4 s locally)"
python scripts/time_flop_search.py --blueprint $BP --boards 1 --repeats 1 \
    --variant "v4e={\"leaf\": {\"net\": \"$V/turn_v4e.pt\"}}" > pilot/timing.log 2>&1
tail -4 pilot/timing.log | tee -a pilot/summary.txt

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a pilot/summary.txt
step "done"
