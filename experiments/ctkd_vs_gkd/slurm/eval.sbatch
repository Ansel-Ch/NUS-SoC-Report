#!/bin/bash
#SBATCH --job-name=opd-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-80:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Usage: sbatch slurm/eval.sbatch <model-path-or-id> <run-name>
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
python eval_countdown.py --model $1 --out $OPD/results/eval/$2
