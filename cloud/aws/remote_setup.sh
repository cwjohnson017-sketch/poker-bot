#!/usr/bin/env bash
# Runs ON the instance (Deep Learning Base AMI, Ubuntu 22.04): Python 3.12 venv with the
# same torch as the local machine (2.11, CUDA 12.8), the package, and the Rust engine.
#   ssh ... 'bash poker-bot/cloud/aws/remote_setup.sh'
set -euo pipefail
cd ~/poker-bot
if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH=$HOME/.local/bin:$PATH
if ! command -v cargo >/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
fi
source "$HOME/.cargo/env"
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "torch==2.11.*" --index-url https://download.pytorch.org/whl/cu128
uv pip install -e ".[dev,train]"   # train: tensorboard (a trainer test expects its logs)
(cd engine && maturin develop --release)
python - <<'EOF'
import torch, poker_engine
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
print("engine", poker_engine.__file__)
EOF
