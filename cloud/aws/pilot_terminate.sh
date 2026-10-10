#!/usr/bin/env bash
# Copy the pilot results back (runs/aws/pilot/) and terminate the instance, which ends its
# billing and deletes its disk. The key pair and security group are kept for reuse.
#   bash cloud/aws/pilot_terminate.sh            # fetch results, then terminate
#   NO_FETCH=1 bash cloud/aws/pilot_terminate.sh # terminate only
set -euo pipefail
source "$(dirname "$0")/common.sh"
load_state
if [[ -z ${NO_FETCH:-} ]]; then
    mkdir -p "$STATE_DIR/pilot"
    remote "tar -C ~/$REMOTE_DIR --exclude=pilot/turn_start -cf - pilot" \
        | tar -x -C "$STATE_DIR" || echo "fetch failed; terminating anyway" >&2
    echo "results in $STATE_DIR/pilot"
fi
aws_ ec2 terminate-instances --instance-ids "$INSTANCE_ID" \
    --query "TerminatingInstances[0].CurrentState.Name" --output text
aws_ ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
rm -f "$STATE"
echo "terminated $INSTANCE_ID"
