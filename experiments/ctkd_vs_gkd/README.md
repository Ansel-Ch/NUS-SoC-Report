# Experiment 1: Cross-tokenizer OPD (GOLD) vs same-tokenizer OPD (GKD)

Motivating experiment for the thesis: quantify what white-box on-policy distillation
loses when teacher and student do not share a tokenizer, and characterise *why*.
This feeds the argument in Chapter 2 that cross-tokenizer alignment carries
irreducible error, which black-box (text-only) supervision avoids by construction.

## Research questions

1. **RQ1 (gap).** Starting from the same SFT warm-start protocol, how much of the
   teacher–student gap does GOLD recover for a cross-tokenizer student, relative to
   GKD for a same-tokenizer student?
2. **RQ2 (attribution).** How much of that gap is due to the *loss approximation*
   (sorted ULD / merged probabilities) versus the *tokenizer mismatch itself*?
3. **RQ3 (dose-response).** Does performance degrade monotonically with vocabulary
   overlap and with token-boundary disagreement on task-relevant spans?
4. **RQ4 (failure modes).** Which of the hypothesised failure modes (below) are
   observable in training diagnostics and in the outputs?

## Models

| Role | Model | Vocab | Jaccard w/ teacher | Gated |
|---|---|---|---|---|
| Teacher | `Qwen/Qwen3-4B-Instruct-2507` | 151,669 | 1.00 | no |
| Student, same tok | `Qwen/Qwen2.5-1.5B-Instruct` | 151,665 | 1.00 | no |
| Student, high overlap | `allenai/OLMo-2-0425-1B-Instruct` | 100,278 | 0.65 | no |
| Student, low overlap | `HuggingFaceTB/SmolLM2-1.7B-Instruct` | 49,152 | 0.24 | no |
| Student, GOLD paper | `meta-llama/Llama-3.2-1B-Instruct` | 128,256 | ~0.64 | **yes** |

Jaccard is computed over vocabulary strings (`analysis/tokenizer_diff.py`), matching
the GOLD paper's definition. Llama-3.2-1B is optional: it reproduces the published
GOLD setting and needs an HF token with the Llama licence accepted.

## Task

Countdown (`HuggingFaceTB/Countdown-Task-GOLD`): reach a target from a set of numbers
with `+ - * /`, answer inside `<answer></answer>`. Verifiable, cheap to score, and
numeric-heavy, which stresses tokenizer disagreement on digits. Score on the held-out
`test` split; an answer is correct iff it uses each number exactly once and evaluates
to the target.

## Conditions

All OPD runs start from that student's SFT checkpoint and use identical
hyperparameters, the same trainer (`trl.experimental.gold.GOLDTrainer`), and the same
number of optimizer steps. Only the loss and the student change.

| ID | Student | Method | Purpose |
|---|---|---|---|
| B0 | all | none (base) | floor |
| B1 | all | SFT on verified teacher completions | off-policy baseline, OPD warm-start |
| T  | teacher | none | ceiling |
| **G1** | Qwen2.5-1.5B | **GKD**: generalized JSD on exact logits | same-tokenizer reference |
| **G2** | Qwen2.5-1.5B | pure ULD (no hybrid) on a *same* tokenizer pair | isolates loss-approximation cost (RQ2); with hybrid on, every token is an exact match and the loss reduces to G1 |
| **X1** | OLMo-2-1B | **GOLD** hybrid ULD | cross-tokenizer, high overlap |
| **X2** | SmolLM2-1.7B | **GOLD** hybrid ULD | cross-tokenizer, low overlap |
| X3 | OLMo-2-1B | pure ULD (no hybrid) | ablation of exact-match path |
| X4 | Llama-3.2-1B | GOLD hybrid ULD | replicate published result (if token) |
| X5 | OLMo-2-1B | GOLD, `--merge_strategy bayesian` | fixes the F1 off-by-one; isolates its cost |
| X6 | SmolLM2-1.7B (and OLMo) | GOLD, `--hybrid_weights 1.0 0.1` | F8: weight the exact path by mass, not vocab count |
| X7 | OLMo-2-1B | GOLD at lr 1e-7 (the GOLD recipe's rate) | is the hybrid collapse an artefact of our 10× higher lr? |

Primary metric: Countdown test accuracy. Because students differ in capacity, the
headline comparison is **teacher-gap recovery**
`R = (acc_OPD − acc_SFT) / (acc_teacher − acc_SFT)`, which normalises away the
student's starting point.

## Hypothesised failure modes of cross-tokenizer OPD

Each is paired with the diagnostic that would confirm or refute it.

| # | Hypothesis | Mechanism | Diagnostic |
|---|---|---|---|
| F1 | **Merge approximation bias** | For a teacher span that maps to one student token, the full distribution is only available at the first teacher position; counterfactual tokens are scaled by the *observed* continuation's probability, so alternatives are mis-weighted. | Loss and student/teacher agreement split by merged vs 1:1 positions; compare `uld_token_merge_strategy=observed` vs `bayesian`. |
| F2 | **Identity-blind ULD signal** | Sorted-probability L1 matches distribution *shape*, not *which token* gets the mass. On unmatched positions the student can match the teacher's entropy while putting mass on the wrong token. | Entropy-matching vs top-1 agreement on unmatched positions; X3 vs X1. |
| F3 | **Overlap dose-response** | Less overlap → more positions fall back to ULD → weaker signal. | R vs Jaccard and vs boundary agreement across G2, X1, X2, X4. |
| F4 | **Numeric segmentation mismatch** | Qwen splits digits individually; cl100k/Llama-3 group up to three. Errors concentrate on exactly the tokens that decide Countdown correctness. | Per-token-class loss (digits / operators / prose / tags); error analysis of wrong answers by digit position. |
| F5 | **Termination / special-token asymmetry** | EOS is skipped in the ULD loss by default and chat-template / BOS tokens have no counterpart, so the stopping decision gets no teacher signal. | Length distribution, truncation rate, missing `</answer>` rate vs G1. |
| F6 | **Tokenization bias** | Student samples are re-tokenised canonically for the teacher; the teacher's conditionals are for one segmentation, not the byte-level marginal. | Fraction of student outputs whose teacher re-tokenisation is non-trivial; correlation with per-sample loss. |
| F7 | **Cost overhead** | Alignment and merging are CPU-side, per sample. | Wall-clock per step and tokens/s vs G1 at equal batch. |
| F8 | **Hybrid weight by vocab count, not mass** | `ULDLoss._compute_hybrid_uld_loss` sets matched weight = \|matched vocab\| / \|teacher vocab\| (0.65 OLMo, 0.26 SmolLM2), regardless of how much teacher probability mass actually lies on shared tokens. Low-overlap students get mostly identity-blind signal even where an exact comparison was available. | Log teacher mass on matched ids per position; ablation with fixed weights (`uld_hybrid_matched_weight=1.0`, unmatched small). |
| F9 | **Teacher sees the student's chat template** | `compute_loss` passes `original_prompt_text`, rendered with the *student's* template, to the teacher tokenizer as plain text. For OLMo (`<\|user\|>`) the Qwen teacher conditions on a foreign, out-of-distribution context; SmolLM2 uses ChatML, which Qwen parses natively. Template compatibility is therefore confounded with vocab overlap. | Teacher accuracy and NLL on its own answers under each student's template vs its native template; ablation that renders the teacher prompt with the teacher's template. |
| F10 | **Prefix-event mismatch in merged groups** | When a student token (e.g. `"123"`) aligns with several teacher tokens (`"1","2","3"`), the merged teacher distribution lives on first-token events (`"1"`), and the hybrid path compares it by string with the student's probability of `"1"` — a different event (`"1"` then stop vs `"1"` as a prefix). Merged sub-vectors are also unnormalised, so the matched "JSD" is not a proper divergence. | Per-group-size loss; fraction of matched-path loss coming from multi-token groups; sign/scale of matched loss. |
| F11 | **Loss-scale mismatch between conditions** | JSD is a per-token mean; ULD is per-sequence L1 sum / length then batch mean. At equal learning rate the effective step size differs, so "GOLD underperforms" could be an LR artefact. | Gradient-norm logs; small LR sweep per condition before the main runs. |

F5 and the special-token half of F4 are expected to matter far more for agentic
tool-use (e.g. Qwen's `<tool_call>` is one special token with no counterpart in the
other vocabularies), which is the thesis' target setting.

## Hardware plan (SoC cluster, per-user QOS)

| Resource | Per-user cap | Use |
|---|---|---|
| `h100-96` (H100 NVL, 94 GB) | 1 | OPD runs (teacher 4B + student ≤1.7B full FT, vLLM colocated) |
| `a100-80` | 2 | OPD runs in parallel, SFT |
| `h100-47` (MIG half H100) | 2 | eval, SFT for 1B students |
| `h200-141` | 1 | `gpu` partition only (3 h); eval sweeps |
| total GPUs | 8 | |

Wall time: 3 h on `gpu`, 3 days on `gpu-long` (no H200 there). All jobs are
single-GPU: MIG slices cannot be combined, and the per-user cap allows only one
H100-96. An OPD step (GA 64, 2048-token rollouts) takes ~130 s on an H100 and peaks at
41–52 GB. Measured: GKD ~25 s/step on a full H100 (200 steps ≈ 1.4 h); OLMo GOLD
~150 s/step on a 47 GB MIG slice, where it also OOMs on long batches, so GOLD runs use
H100-96 / A100-80 only; there is no resume path and
checkpoints are model-only because the home quota is unknown.

Cluster gotchas (all handled in `env.sh` / the scripts):
- Login nodes cap virtual memory at 1 GB and `/tmp` at 50 MB: install and download
  inside jobs. Inside a job `/tmp` is private node-local scratch; `/temp` is root-only.
- Nodes have no `nvcc`: FlashInfer's JIT sampler fails on A100 (`VLLM_USE_FLASHINFER_SAMPLER=0`).
- cuDNN SDPA fails on some H100 nodes (`enable_cudnn_sdp(False)`).
- vLLM 0.30 has no native OLMo-2; its Transformers backend patches the HF class in
  place, breaking the colocated training model (`_bypass_vllm_compile_wrapper`).

## Findings so far (2026-10-03)

Baselines, Countdown test[:1000], greedy, 2048 tokens:

| Model | Base | SFT (10k) |
|---|---|---|
| Teacher Qwen3-4B-2507 | 0.713 | – |
| Qwen2.5-1.5B | 0.113 | 0.665 |
| OLMo-2-1B | 0.021 | 0.630 |
| SmolLM2-1.7B | 0.013 | 0.488 |

Teacher misses are almost all truncations from genuine search on hard instances (no
repetition loops). Full SFT leaves too little headroom for OPD to separate methods, so
the warm-start budget was swept:

| SFT examples | 500 | 1000 | 2000 | 10k |
|---|---|---|---|---|
| Qwen2.5-1.5B | 0.295 | **0.379** | 0.527 | 0.665 |
| OLMo-2-1B | 0.045 | 0.154 | **0.381** | 0.630 |
| SmolLM2-1.7B | 0.013 | 0.046 | 0.191 | 0.488 |

OPD starts from a **matched starting accuracy (~0.38)** rather than a matched SFT budget,
because the GKD/GOLD comparison is necessarily across students and OPD dynamics depend
on the starting point: Qwen2.5 n=1000 (0.379), OLMo n=2000 (0.381), SmolLM2 n=5000 (0.375).

OPD results (200 steps × 64 rollouts, lr 1e-6, λ=1, β=0; same test prompts):

| Run | Student | Loss | SFT → OPD | R | s/step (GPU) |
|---|---|---|---|---|---|
| G1 | Qwen2.5-1.5B | GKD (JSD) | 0.379 → 0.600 | 0.66 | 25 (H100) |
| G2 | Qwen2.5-1.5B | ULD, same tokenizer | 0.379 → 0.487 | 0.32 | 116 (H100) |
| X3 | OLMo-2-1B | ULD, cross tokenizer | 0.381 → 0.426 | 0.14 [0.07, 0.20] | ~80 (H100) |

**RQ2, the loss approximation alone halves the gain.** Same student, teacher, data and
tokenizer; only the loss differs. GKD − ULD = +0.113 [95% CI +0.091, +0.136] (paired
bootstrap); 130 prompts solved only by G1 vs 17 only by G2. G2 also truncates more
(0.432 vs 0.367).

**RQ2 decomposition, under the same ULD loss.** Going from GKD to ULD on a shared
tokenizer roughly halves recovery (0.66 → 0.32); then adding the tokenizer mismatch
roughly halves it again (0.32 → 0.14, OLMo). Caveat: G2 and X3 are different students,
matched only on starting accuracy.

### Every GOLD hybrid run collapsed (2026-10-04)

| Run | Student | Accuracy | Truncated | Arithmetic steps correct | Failure |
|---|---|---|---|---|---|
| X6 (final) | SmolLM2 | 0.375 → **0.000** | 100% | – | restates task, then newlines to the length limit |
| X1 (step 110) | OLMo | 0.381 → **0.035** | 96% | **0.67** | endless search with wrong arithmetic |
| X5 (step 100) | OLMo | 0.381 → **0.005** | 99.5% | **0.25** | repetition loops (`81 - 73 = 60` ×N) |
| X3 (final, ULD only) | OLMo | 0.381 → 0.426 | 50% | 0.98 | healthy |
| X2 (adaptive weights) | SmolLM2 | crashed at 133 (F12) | | | loss tracked X6's trajectory; rerun pending |

X5 was stopped at step 100: fixing the merge off-by-one does not rescue GOLD, it makes
OLMo worse, so its question is answered. X7 reruns X1 at lr 1e-7 (the GOLD recipe's
rate) to test whether the collapse is a learning-rate artefact.

Arithmetic = fraction of `a op b = c` steps in the reasoning that are correct; every
non-hybrid run, the SFT checkpoints and the teacher are 0.95–0.98. The losses fell
throughout (X6: 112 → 33; X1: 108 → 18), so **the optimiser gamed the hybrid objective
rather than learning the task**. The ULD-only runs (G2, X3) did not collapse, which
isolates the hybrid exact-match (JSD) term.

**F10 confirmed causally for OLMo.** Digit-token lengths in OLMo's outputs (1/2/3 digits):
SFT 0.14/0.82/0.05, X3 0.12/0.81/0.07, **X1 and X5 1.00/0.00/0.00**. The exact-match term
forced OLMo into the teacher's single-digit segmentation, which is non-canonical for OLMo,
and its arithmetic broke. X5 (`bayesian`, merge off-by-one fixed) is *worse* than X1: the
F1 bug had been shrinking merged digit groups to near zero, which partly shielded OLMo.

**SmolLM2 found a different exploit.** Qwen packs up to 32 newlines into one token,
SmolLM2 uses 4-newline tokens, so newline runs form merged groups. Diagnostics on X6's
outputs: 97% of sampled tokens are non-canonical; merged-group scalars average 8e-6
(`observed`); **41.5% of matched-KL terms are negative**, so the unnormalised matched
JSD can be driven to or below zero there; teacher P(EOS) at the end is ~5e-13 but EOS is
never trained (F5), and per-sequence length normalisation dilutes the remaining loss.

Robustness (F12, F13): every 3-hour A100 link nets only ~20 steps (restart cost, up to
10 lost steps per link, steps slowing from ~240 s to ~430 s as rollouts lengthen). X2
crashed at step 133 on F12 before the patch and is being rerun. A second GOLD crash,
**F13**: OLMo's embedding has 100,352 rows but only 100,278 tokens, so sampling at
temperature 1 occasionally draws a padding id with no token string, and
`piece_byte_len(None)` raises `TypeError` (killed X1 link 3). `opd.py` now treats such
ids as zero-width and logs `[out-of-vocab sample]`.

**F7 confirmed:** with identical models and data, the ULD path costs 4.7× GKD per step
(per-sample Python alignment and full-vocabulary sorts).

Failure-mode evidence from `analysis/failure_diagnostics.py` on 50 base-OLMo rollouts:

- **F1 confirmed, default merge is off by one.** With `uld_token_merge_strategy="observed"`
  (default) the scalars multiplied into merged groups average 0.023, i.e. P(next token =
  current token); with `"bayesian"` they average 0.93. Merged groups (multi-digit numbers
  for OLMo) are shrunk ~40× and carry almost no signal. Whether this changes training is
  open: from the same SFT checkpoint, step-1 loss is 108.5 under `observed` (X1) and
  96.8 under `bayesian` (X5), on different random batches, so the total loss does not
  separate them. (An earlier note compared against a ~13 loss from the *base* model in
  the smoke test; that comparison was invalid.) X1 vs X5 accuracy will decide it.
- **F10 refined, segmentation-convention conflict.** Positions that *predict digits* are
  99.5% 1:1 aligned yet have 16.7% top-1 agreement and 11.6 nats matched-KL (prose: 75%,
  0.63). After `Ġ` the teacher predicts `5` while OLMo predicts `567`, absent from
  Qwen's vocab, so the "exact" JSD pushes OLMo toward the teacher's segmentation, which
  is non-canonical for OLMo. Prediction: GOLD training raises OLMo's non-canonical rate.
- **F8 confirmed.** 98.5% of teacher mass is on shared tokens; the adaptive weight
  gives the exact path only 0.65 (OLMo) / 0.26 (SmolLM2).
- **F6 real.** Every rollout contains non-canonical sampled spans (e.g. `<`,`think`,`>`
  vs canonical `<th`,`ink`,`>`); GOLD aligns these against canonical teacher tokens.
- **F9 refuted for capability.** The teacher scores 0.722 with OLMo's template (0.713
  native); Qwen2.5 and SmolLM2 render identical ChatML. Only a distribution shift remains.
- **F11 mostly moot.** Grad-norm at step 1: GKD 2.0, ULD 29, GOLD 12–48 (SmolLM2), 130
  (OLMo). Adam is invariant to a global loss scale, so this does not confound the LR;
  what matters is the *relative* weight inside the GOLD loss, where matched-KL dominates.
- **F12 (new), alignment can crash training.** X5 died at step 25 with
  `ValueError: Tokenizer produced overlapping byte offsets that could not be normalized`
  from `trl.experimental.utils._normalize_byte_offsets`, raised while re-tokenizing a
  sampled student rollout with the teacher tokenizer. 0/25,000 greedy evaluation outputs
  trigger it, so it is rare and comes from temperature-1 sampling, but one occurrence kills a
  run. `opd.py` now clamps such offsets to be monotonic (`_tolerate_nonmonotonic_byte_offsets`)
  and logs every fallback as `[byte-offset fallback]`; X5 resumed from step 20 with it.
- SmolLM2 (Jaccard 0.24) segments Countdown text identically to Qwen, so Jaccard is a
  poor predictor of on-task alignment; OLMo (Jaccard 0.65) is the harder case here.

## Layout

- `env.sh` — shared environment (venv, HF cache in `~/opd`, outside this repo).
- `slurm/` — batch scripts. `setup_env.sbatch`, `download.sbatch`, then train/eval.
- `analysis/` — tokenizer and failure-mode analysis.
- Checkpoints and logs go to `~/opd/runs` and `~/opd/logs`, never into git.
