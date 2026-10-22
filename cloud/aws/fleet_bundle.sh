#!/usr/bin/env bash
# Build the fleet bundle and upload it to the private S3 bucket once; instances of
# cloud/aws/fleet.py fetch it through a short-lived pre-signed URL, so relaunching after
# spot interruptions does not re-upload from this machine.
#
# Contents: the code at REF (default HEAD: commit what the instances should run first),
# the strength features the blueprint reads, the production search's two leaf nets, the
# solved preflop table and the blueprint. The preflop table names its base blueprint by
# this machine's absolute path; on the instances `local_base` finds it under runs/.
#   bash cloud/aws/fleet_bundle.sh            # -> s3://pokerbot-<account>/fleet/bundle.tar
#   REF=claude/spot-fleet bash cloud/aws/fleet_bundle.sh
set -euo pipefail
source "$(dirname "$0")/common.sh"
ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text | tr -d '\r')
BUCKET=${BUCKET:-pokerbot-$ACCOUNT}
KEY_S3=${KEY_S3:-fleet/bundle.tar}
REF=${REF:-HEAD}
OUT=${OUT:-$STATE_DIR/fleet_bundle.tar}
mkdir -p "$STATE_DIR"

if ! aws_ s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    aws_ s3api create-bucket --bucket "$BUCKET" \
        --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
    aws_ s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo "created private bucket $BUCKET ($REGION)"
fi

echo "building $OUT (code at $REF = $(git -C "$REPO" rev-parse --short "$REF"))"
git -C "$REPO" archive --format=tar -o "$OUT" "$REF"
F=data/abstraction/buckets_hunl/features
tar -C "$MAIN" -rf "$OUT" \
    $F/flop_equity.npy $F/flop_features.npy $F/flop_features.json $F/flop_weights.npy \
    $F/turn_equity.npy $F/turn_features.npy $F/turn_features.json $F/turn_weights.npy \
    $F/river_equity.npy $F/river_features.json \
    runs/value_net/turn_v5x.pt runs/value_net/river_v4rx.pt \
    runs/preflop/preflop_r15.pt runs/dcfr4_distilled_v2
ls -la "$OUT"
echo "uploading to s3://$BUCKET/$KEY_S3"
SRC=$OUT   # aws_ turns Git Bash's path conversion off: give aws.exe a Windows path
if command -v cygpath >/dev/null; then SRC=$(cygpath -m "$OUT"); fi
aws_ s3 cp "$SRC" "s3://$BUCKET/$KEY_S3" --only-show-errors
aws_ s3api head-object --bucket "$BUCKET" --key "$KEY_S3" --query ContentLength --output text
