#!/bin/bash
#SBATCH --job-name=opd-diag
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Usage: sbatch slurm/diagnostics.sbatch <student-path-or-id> <eval-run-name> [n]
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
python analysis/failure_diagnostics.py --student $1 --samples $OPD/results/eval/$2/samples.jsonl \
  --n ${3:-200} --out $OPD/results/diag/$2
