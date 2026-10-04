#!/bin/bash
#SBATCH --job-name=opd-download
#SBATCH --partition=normal
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Fetches teacher, students and the Countdown data into ~/opd/hf_cache.
set -euo pipefail
source $HOME/opd/env.sh

MODELS=(
  Qwen/Qwen3-4B-Instruct-2507          # teacher
  Qwen/Qwen2.5-1.5B-Instruct           # student, same tokenizer   (Jaccard 1.00)
  allenai/OLMo-2-0425-1B-Instruct      # student, high overlap     (Jaccard 0.65)
  HuggingFaceTB/SmolLM2-1.7B-Instruct  # student, low overlap      (Jaccard 0.24)
)
# Gated models are fetched only when a token is present.
if hf auth whoami >/dev/null 2>&1; then
  MODELS+=(meta-llama/Llama-3.2-1B-Instruct)  # student used in the GOLD paper (Jaccard 0.64)
fi

for m in "${MODELS[@]}"; do
  echo "== $m"
  hf download "$m" --exclude "*.pth" --exclude "original/*" --exclude "*.gguf" --exclude "onnx/*"
done
hf download --repo-type dataset HuggingFaceTB/Countdown-Task-GOLD
du -sh $HF_HOME
