#!/usr/bin/env bash
# Runs ON the instance: one block of the head-to-head match (cloud/aws/h2h_coordinator.py).
# Round 4 production search (A) against Round 3's (B), BLOCK's seeds, resumable: an
# existing h2h/block_<BLOCK>.json (uploaded by the coordinator after an interruption)
# is continued after its last finished chunk. Writes h2h/block_<BLOCK>.done when complete.
#   bash poker-bot/cloud/aws/remote_h2h_block.sh BLOCK SEED DEALS CHUNK
set -euo pipefail
BLOCK=$1 SEED=$2 DEALS=$3 CHUNK=$4
cd ~/poker-bot
source .venv/bin/activate
mkdir -p h2h
BP=runs/dcfr4_distilled_v2
PYTHONPATH=. python scripts/match_search.py --config configs/match_search_vn.yaml \
    --a "search:neural:$BP" --b "search:neural:$BP" \
    --b-search-config configs/search_value_net_r3.yaml \
    --deals "$DEALS" --chunk "$CHUNK" --seed "$SEED" --resume --out "h2h/block_$BLOCK.json"
touch "h2h/block_$BLOCK.done"
