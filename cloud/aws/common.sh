# Shared settings for the AWS scripts (sourced, not run). Override any of them in the
# environment, e.g. `REGION=us-east-1 TYPE=g6.2xlarge bash cloud/aws/pilot_launch.sh`.

REGION=${REGION:-us-east-2}
TYPE=${TYPE:-g5.2xlarge}           # 8 vCPUs, 1x A10G (24 GB), 32 GB RAM
MARKET=${MARKET:-on-demand}        # on-demand | spot
NAME=${NAME:-pokerbot-pilot}
KEY=${KEY:-pokerbot-$REGION}       # EC2 key pair (per region); private key in ~/.ssh/$KEY.pem
SG_NAME=${SG_NAME:-pokerbot-ssh}   # security group: SSH from this machine's IP only
DISK_GB=${DISK_GB:-150}
MAX_HOURS=${MAX_HOURS:-4}          # the instance shuts itself down (and terminates) after this

# local sources: this checkout (code, value nets) and the main checkout (blueprint, tables)
REPO=$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)
if command -v cygpath >/dev/null; then REPO=$(cygpath -u "$REPO"); fi   # Git Bash: C:/x -> /c/x
MAIN=${MAIN:-/c/Users/Connor/Documents/Poker-Bot/poker-bot}
STATE_DIR=${STATE_DIR:-$REPO/runs/aws}
STATE=$STATE_DIR/$NAME.env         # INSTANCE_ID, HOST of the launched instance
SSH_KEY=$HOME/.ssh/$KEY.pem
SSH_OPTS=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new -o ServerAliveInterval=30)
REMOTE_USER=ubuntu
REMOTE_DIR=poker-bot

# MSYS_NO_PATHCONV: Git Bash would otherwise rewrite arguments such as /aws/service/... or
# /dev/sda1 into Windows paths before they reach aws.exe
aws_() { MSYS_NO_PATHCONV=1 aws --region "$REGION" "$@"; }

load_state() {
    [[ -f $STATE ]] || { echo "no instance recorded in $STATE (run pilot_launch.sh)" >&2; exit 1; }
    # shellcheck disable=SC1090
    source "$STATE"
}

remote() { ssh "${SSH_OPTS[@]}" "$REMOTE_USER@$HOST" "$@"; }
# start a long job on the instance, detached from this SSH session (log in ~/<name>.log)
remote_detached() {
    local name=$1; shift
    ssh "${SSH_OPTS[@]}" -n "$REMOTE_USER@$HOST" \n        "nohup $* > ~/$name.log 2>&1 < /dev/null & echo started $name"
}
