# Source from every job: shared paths for the OPD experiments.
export OPD=$HOME/opd
export PATH=$HOME/.local/bin:$PATH
source $OPD/.venv/bin/activate
export HF_HOME=$OPD/hf_cache
export HF_XET_HIGH_PERFORMANCE=1
export TMPDIR=${SLURM_JOB_ID:+/tmp}
export TMPDIR=${TMPDIR:-$HOME/.tmp}   # login-node /tmp has a 50 MB quota
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}
# FlashInfer JIT-compiles its sampler for sm_80 and the nodes have no nvcc; vLLM's
# PyTorch sampler is equivalent.
export VLLM_USE_FLASHINFER_SAMPLER=0
# Trainer log dicts go to stdout, which SLURM block-buffers; keep them live.
export PYTHONUNBUFFERED=1
