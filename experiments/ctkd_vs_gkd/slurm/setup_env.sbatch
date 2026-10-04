#!/bin/bash
#SBATCH --job-name=opd-setup
#SBATCH --partition=normal
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Builds ~/opd/.venv. Run on a compute node: login nodes cap virtual memory at 1 GB.
set -euo pipefail
export PATH=$HOME/.local/bin:$PATH
export TMPDIR=/tmp UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy  # /tmp inside a job is private node-local scratch
cd $HOME/opd
[ -d .venv ] || uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python \
  "trl[vllm]==1.14.1" accelerate datasets peft liger-kernel \
  wandb rich math-verify pandas matplotlib hf_transfer
.venv/bin/python - <<'PY'
import torch, transformers, trl, vllm, accelerate, datasets
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("transformers", transformers.__version__, "trl", trl.__version__, "vllm", vllm.__version__)
from trl.experimental.gold import GOLDConfig, GOLDTrainer
from trl.experimental.gkd import GKDConfig, GKDTrainer
print("GOLD + GKD import OK")
PY
