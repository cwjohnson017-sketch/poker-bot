#!/usr/bin/env bash
# Upload what the pilot needs (~3.3 GB): the code at this checkout's HEAD, the blueprint
# and its strength tables (main checkout), the production value nets and one held-out
# state file. No git credentials are needed on the instance.
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_state
remote "mkdir -p ~/$REMOTE_DIR"
echo "code (HEAD of $(git -C "$REPO" rev-parse --abbrev-ref HEAD))"
git -C "$REPO" archive --format=tar HEAD | remote "tar -x -C ~/$REMOTE_DIR"
echo "blueprint and strength tables"
tar -C "$MAIN" -cf - runs/dcfr4_distilled_v2 data/abstraction | remote "tar -x -C ~/$REMOTE_DIR"
echo "value nets and held-out states"
tar -C "$REPO" -cf - runs/value_net/turn_v4e.pt runs/value_net/turn_v3w.pt runs/value_net/river_v3.pt \
    runs/value_net/river_onpolicy2_heldout/turn_states.pt | remote "tar -x -C ~/$REMOTE_DIR"
remote "du -sh ~/$REMOTE_DIR"
