#!/bin/bash
#SBATCH --job-name=opd-eval-base
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100-47:1
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/home/a/anselch/opd/logs/%x-%j.out
# B0 and T rows of the results table, plus the F9 template-mismatch probe.
set -uo pipefail
source $HOME/opd/env.sh
cd $HOME/NUS-SoC-Report/experiments/ctkd_vs_gkd  # sbatch runs a spooled copy, so $0 is not usable

T=Qwen/Qwen3-4B-Instruct-2507
RES=$OPD/results/eval
run() {  # name model [template_from]
  [ -f $RES/$1/metrics.json ] && { echo "skip $1"; return; }
  echo "== $1"
  python eval_countdown.py --model $2 ${3:+--template_from $3} --out $RES/$1 || echo "FAILED $1"
}

run teacher                $T
run student_qwen25_base    Qwen/Qwen2.5-1.5B-Instruct
run student_olmo2_base     allenai/OLMo-2-0425-1B-Instruct
run student_smollm2_base   HuggingFaceTB/SmolLM2-1.7B-Instruct
# F9: the teacher prompted with each student's template, as GOLD feeds it.
run teacher_tpl_qwen25     $T Qwen/Qwen2.5-1.5B-Instruct
run teacher_tpl_olmo2      $T allenai/OLMo-2-0425-1B-Instruct
run teacher_tpl_smollm2    $T HuggingFaceTB/SmolLM2-1.7B-Instruct

python - <<'PY'
import json, glob, os
for f in sorted(glob.glob(os.path.expandvars("$OPD/results/eval/*/metrics.json"))):
    m = json.load(open(f))
    print(f"{f.split('/')[-2]:26s} acc={m['accuracy']:.3f} trunc={m['truncation_rate']:.3f} "
          f"len={m['mean_tokens']:.0f} {m['reasons']}")
PY
