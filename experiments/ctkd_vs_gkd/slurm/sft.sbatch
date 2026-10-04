#!/bin/bash
#SBATCH --job-name=opd-sft
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Usage: sbatch [--gres=...] slurm/sft.sbatch <student-hf-id> <name> [sft.py flags...]
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
model=$1 name=$2; shift 2
python sft.py --model $model --out $OPD/runs/sft_$name "$@"
python eval_countdown.py --model $OPD/runs/sft_$name --out $OPD/results/eval/student_${name}_sft
