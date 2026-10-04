#!/bin/bash
#SBATCH --job-name=opd-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-96:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Plumbing, memory and step-time check on base students.
# Usage: sbatch slurm/opd_smoke.sbatch "<loss> <student> [opd.py flags]" ...
set -uo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader >> $OPD/logs/smoke-mem-$SLURM_JOB_ID.txt; sleep 5; done ) &
for cfg in "$@"; do
  set -- $cfg
  echo "=== $cfg  $(date +%T)"
  rm -rf /tmp/smoke
  loss=$1 student=$2; shift 2
  accelerate launch --num_processes 1 --mixed_precision bf16 opd.py --student $student --loss $loss --out /tmp/smoke \
    --max_steps 2 --grad_accum 8 --save_steps 1000 "$@" 2>&1 | grep -vE "it/s\]|s/it\]" | tail -40
  echo "=== done $loss $(date +%T)  peak_mem_MiB=$(sort -n $OPD/logs/smoke-mem-$SLURM_JOB_ID.txt | tail -1)"
done
