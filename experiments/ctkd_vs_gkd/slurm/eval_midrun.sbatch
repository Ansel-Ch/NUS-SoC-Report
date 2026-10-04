#!/bin/bash
#SBATCH --job-name=eval-midrun
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100-40:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# Snapshot a running OPD job's latest checkpoint (they rotate) and evaluate it.
# Usage: sbatch slurm/eval_midrun.sbatch <run-name> [n_prompts]
set -euo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd
ck=$(ls -d $OPD/runs/opd_$1/checkpoint-* | sort -t- -k2 -n | tail -1)
step=${ck##*-}
snap=/mnt/scratch/a/anselch/opd_snapshots/opd_$1_step$step
mkdir -p $snap
cp $ck/{config.json,generation_config.json,model.safetensors,tokenizer.json,tokenizer_config.json,chat_template.jinja} $snap/
python eval_countdown.py --model $snap --n_prompts ${2:-200} --out $OPD/results/eval/midrun_$1_step$step
