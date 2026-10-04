#!/bin/bash
#SBATCH --job-name=opd-chain
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# One 3-hour link of a resumable OPD run on the `gpu` partition (gpu-long A100-80s are
# often draining). Submit with slurm/submit_chain.sh; each link resumes from the last
# full checkpoint, and links after completion only evaluate or exit.
# Usage: sbatch slurm/opd_chain.sbatch <name> <student> <gkd|uld|gold> [opd.py flags...]
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
name=$1 student=$2 loss=$3; shift 3
out=$OPD/runs/opd_$name
if [ ! -f $out/final/config.json ]; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
  export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
  accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port $MASTER_PORT opd.py \
    --student $student --loss $loss --out $out --resumable --save_steps 10 "$@"
  rm -rf $out/checkpoint-*  # full checkpoints are only needed to resume
fi
[ -f $OPD/results/eval/opd_$name/metrics.json ] || \
  python eval_countdown.py --model $out/final --out $OPD/results/eval/opd_$name
