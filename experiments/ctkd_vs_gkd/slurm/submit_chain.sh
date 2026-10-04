#!/bin/bash
# Submit a resumable OPD run as N chained 3-hour jobs.
# Usage: slurm/submit_chain.sh <links> <name> <student> <gkd|uld|gold> [opd.py flags...]
set -euo pipefail
cd "$(dirname "$0")"
links=$1 name=$2; shift
dep=""
for i in $(seq 1 $links); do
  dep=$(sbatch --parsable --job-name=$name-$i ${dep:+--dependency=afterany:$dep} opd_chain.sbatch "$@")
  echo "$name link $i: $dep"
done
