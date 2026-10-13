#!/usr/bin/env bash
# Build the head-to-head bundle and upload it to a private S3 bucket once; instances fetch
# it through a short-lived pre-signed URL (cloud/aws/h2h_coordinator.py), so relaunching
# after spot interruptions does not re-upload from this machine.
#
# Contents (about 1.55 GB): the code at this checkout's HEAD, the blueprint (main
# checkout), the strength features a match reads (traced: the five equity / feature
# tables below, not the 0.5 GB river features or the canonical tables) and the four nets
# of the Round 3 and Round 4 production searches.
#   bash cloud/aws/h2h_bundle.sh            # -> s3://pokerbot-<account>/h2h/bundle.tar
set -euo pipefail
source "$(dirname "$0")/common.sh"
ACCOUNT=$(aws_ sts get-caller-identity --query Account --output text | tr -d '\r')
BUCKET=${BUCKET:-pokerbot-$ACCOUNT}
KEY_S3=${KEY_S3:-h2h/bundle.tar}
OUT=${OUT:-$STATE_DIR/h2h_bundle.tar}
mkdir -p "$STATE_DIR"

if ! aws_ s3api head-bucket --bucket "$BUCKET" >/dev/null 2>&1; then
    aws_ s3api create-bucket --bucket "$BUCKET" \
        --create-bucket-configuration "LocationConstraint=$REGION" >/dev/null
    aws_ s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
        BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
    echo "created private bucket $BUCKET ($REGION)"
fi

echo "building $OUT (code at $(git -C "$REPO" rev-parse --short HEAD))"
git -C "$REPO" archive --format=tar -o "$OUT" HEAD
F=data/abstraction/buckets_hunl/features
tar -C "$REPO" -rf "$OUT" \
    $F/flop_equity.npy $F/flop_features.npy $F/flop_features.json $F/flop_weights.npy \
    $F/turn_equity.npy $F/turn_features.npy $F/turn_features.json $F/turn_weights.npy \
    $F/river_equity.npy $F/river_features.json \
    runs/value_net/turn_v5x.pt runs/value_net/river_v4rx.pt \
    runs/value_net/turn_v4e.pt runs/value_net/river_v3.pt
tar -C "$MAIN" -rf "$OUT" runs/dcfr4_distilled_v2
ls -la "$OUT"
echo "uploading to s3://$BUCKET/$KEY_S3"
aws_ s3 cp "$OUT" "s3://$BUCKET/$KEY_S3" --only-show-errors
aws_ s3api head-object --bucket "$BUCKET" --key "$KEY_S3" --query ContentLength --output text
