#!/usr/bin/env bash
# Launch one GPU instance for the pilot. It is billed from now until it terminates:
# run pilot_terminate.sh when done. As a guard it also shuts itself down (and so
# terminates) MAX_HOURS after boot.
#   bash cloud/aws/pilot_launch.sh                 # g5.2xlarge, on-demand, us-east-2
#   MARKET=spot bash cloud/aws/pilot_launch.sh     # spot (cheaper, can be interrupted)
set -euo pipefail
source "$(dirname "$0")/common.sh"
mkdir -p "$STATE_DIR"
if [[ -f $STATE ]]; then
    echo "an instance is already recorded in $STATE; terminate it first" >&2
    exit 1
fi

# AMI: AWS Deep Learning Base (Ubuntu 22.04, NVIDIA driver)
AMI=$(aws_ ssm get-parameter \
    --name /aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-ubuntu-22.04/latest/ami-id \
    --query Parameter.Value --output text)
ROOT_DEV=$(aws_ ec2 describe-images --image-ids "$AMI" --query "Images[0].RootDeviceName" --output text)
echo "AMI $AMI (root device $ROOT_DEV)"

# key pair (private key kept locally)
if ! aws_ ec2 describe-key-pairs --key-names "$KEY" >/dev/null 2>&1; then
    mkdir -p "$HOME/.ssh"
    aws_ ec2 create-key-pair --key-name "$KEY" --key-type ed25519 \
        --query KeyMaterial --output text | tr -d '\r' > "$SSH_KEY"   # aws.exe writes CRLF
    chmod 600 "$SSH_KEY"
    echo "created key pair $KEY -> $SSH_KEY"
fi

# security group: SSH from this machine's public IP only
MY_IP=$(curl -s https://checkip.amazonaws.com | tr -d '[:space:]')
VPC=$(aws_ ec2 describe-vpcs --filters Name=is-default,Values=true --query "Vpcs[0].VpcId" --output text)
SG=$(aws_ ec2 describe-security-groups --filters Name=group-name,Values="$SG_NAME" Name=vpc-id,Values="$VPC" \
    --query "SecurityGroups[0].GroupId" --output text)
if [[ $SG == None ]]; then
    SG=$(aws_ ec2 create-security-group --group-name "$SG_NAME" --vpc-id "$VPC" \
        --description "pokerbot SSH from the owner IP only" --query GroupId --output text)
fi
aws_ ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 22 \
    --cidr "$MY_IP/32" >/dev/null 2>&1 || true   # already allowed
echo "security group $SG allows SSH from $MY_IP"

MARKET_OPT=()
if [[ $MARKET == spot ]]; then
    MARKET_OPT=(--instance-market-options
        'MarketType=spot,SpotOptions={SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')
fi
USER_DATA=$(printf '#!/bin/bash\nshutdown -h +%d\n' $((MAX_HOURS * 60)))
INSTANCE_ID=$(aws_ ec2 run-instances --image-id "$AMI" --instance-type "$TYPE" --key-name "$KEY" \
    --security-group-ids "$SG" "${MARKET_OPT[@]}" \
    --block-device-mappings "DeviceName=$ROOT_DEV,Ebs={VolumeSize=$DISK_GB,VolumeType=gp3,DeleteOnTermination=true}" \
    --instance-initiated-shutdown-behavior terminate \
    --user-data "$USER_DATA" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=$NAME},{Key=project,Value=pokerbot}]" \
    --query "Instances[0].InstanceId" --output text)
echo "launched $INSTANCE_ID ($TYPE, $MARKET, self-terminates after $MAX_HOURS h); waiting..."
aws_ ec2 wait instance-running --instance-ids "$INSTANCE_ID"
HOST=$(aws_ ec2 describe-instances --instance-ids "$INSTANCE_ID" \
    --query "Reservations[0].Instances[0].PublicDnsName" --output text)
printf 'INSTANCE_ID=%s\nHOST=%s\n' "$INSTANCE_ID" "$HOST" > "$STATE"
echo "running: $HOST (state in $STATE)"
for _ in $(seq 30); do   # sshd takes a minute
    if remote true 2>/dev/null; then echo "ssh ok"; exit 0; fi
    sleep 10
done
echo "ssh not answering yet; try: bash cloud/aws/pilot_sync.sh in a minute" >&2
