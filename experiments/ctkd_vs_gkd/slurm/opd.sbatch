#!/bin/bash
#SBATCH --job-name=opd
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Usage: sbatch --job-name=<name> slurm/opd.sbatch <name> <student> <gkd|uld|gold> [opd.py flags...]
# Trains into ~/opd/runs/opd_<name>, then evaluates the final model.
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
name=$1 student=$2 loss=$3; shift 3
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
# MIG slices put several jobs on one node; give each its own rendezvous port.
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
accelerate launch --num_processes 1 --mixed_precision bf16 --main_process_port $MASTER_PORT opd.py \
  --student $student --loss $loss --out $OPD/runs/opd_$name "$@"
python eval_countdown.py --model $OPD/runs/opd_$name/final --out $OPD/results/eval/opd_$name
