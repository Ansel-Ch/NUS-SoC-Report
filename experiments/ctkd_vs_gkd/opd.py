"""On-policy distillation from Qwen3-4B-Instruct-2507 with TRL's GOLDTrainer.

Every condition uses the same trainer, generation path and hyperparameters; only the
loss changes:

  gkd   generalized JSD on exact logits (same tokenizer only)          -> G1
  uld   ULD sorted-L1 with GOLD's byte-offset alignment, no hybrid      -> G2, X3
  gold  ULD + hybrid exact-match JSD (the GOLD method)                  -> X1, X2, X4

Ablations: --merge_strategy (F1), --hybrid_weights (F8).
"""

import argparse
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint
from trl.experimental.gold import GOLDConfig, GOLDTrainer

from countdown import opd_split

TEACHER = "Qwen/Qwen3-4B-Instruct-2507"


def main():
    # cuDNN SDPA fails on some H100 nodes (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH);
    # flash / mem-efficient SDPA are numerically equivalent here.
    torch.backends.cuda.enable_cudnn_sdp(False)
    _tolerate_nonmonotonic_byte_offsets()
    _tolerate_out_of_vocab_samples()
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="SFT checkpoint or HF id")
    ap.add_argument("--student_tokenizer", default=None, help="defaults to --student")
    ap.add_argument("--loss", choices=["gkd", "uld", "gold"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--max_steps", type=int, default=200)
    ap.add_argument("--grad_accum", type=int, default=64)
    ap.add_argument("--lmbda", type=float, default=1.0)
    ap.add_argument("--beta", type=float, default=0.0, help="0 = forward KL, as in the GOLD recipe")
    ap.add_argument("--max_completion_length", type=int, default=2048)
    ap.add_argument("--merge_strategy", choices=["observed", "bayesian"], default="observed")
    ap.add_argument("--hybrid_weights", type=float, nargs=2, default=None, metavar=("MATCHED", "UNMATCHED"))
    ap.add_argument("--vllm_mem", type=float, default=0.25)
    ap.add_argument("--save_steps", type=int, default=50)
    ap.add_argument(
        "--resumable",
        action="store_true",
        help="save full checkpoints (model + optimizer, ~18 GB) and resume from the latest; "
        "for chains of 3-hour jobs on the `gpu` partition",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.student_tokenizer or args.student)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.float32, attn_implementation="sdpa")
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16, attn_implementation="sdpa")

    cross = args.loss in ("uld", "gold")
    cfg = GOLDConfig(
        output_dir=args.out,
        # optimisation
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=10,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        bf16=True,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
        # distillation
        lmbda=args.lmbda,
        beta=args.beta,
        temperature=1.0,
        top_p=1.0,
        max_completion_length=args.max_completion_length,
        max_length=args.max_completion_length + 512,
        teacher_model_name_or_path=TEACHER,
        use_uld_loss=cross,
        teacher_tokenizer_name_or_path=TEACHER if cross else None,
        uld_use_hybrid_loss=args.loss == "gold",
        uld_hybrid_matched_weight=args.hybrid_weights[0] if args.hybrid_weights else None,
        uld_hybrid_unmatched_weight=args.hybrid_weights[1] if args.hybrid_weights else None,
        uld_token_merge_strategy=args.merge_strategy,
        # student rollouts
        use_vllm=True,
        vllm_mode="colocate",
        vllm_gpu_memory_utilization=args.vllm_mem,
        vllm_max_model_length=args.max_completion_length + 512,
        vllm_enable_sleep_mode=True,
        # bookkeeping
        dataloader_drop_last=True,
        logging_steps=1,
        log_completions=True,
        log_completions_steps=25,
        num_completions_to_print=2,
        save_strategy="steps",
        save_steps=args.save_steps,
        # Model-only checkpoints (~6 GB vs ~18 GB with optimizer state; home quota is
        # unknown) unless the run is split across wall-clock-limited jobs.
        save_only_model=not args.resumable,
        save_total_limit=1,
        report_to="none",
        seed=0,
    )
    trainer = GOLDTrainer(
        model=student,
        teacher_model=teacher,
        args=cfg,
        train_dataset=opd_split(),
        processing_class=tok,
    )
    _bypass_vllm_compile_wrapper(trainer.model)
    last = get_last_checkpoint(args.out) if args.resumable and os.path.isdir(args.out) else None
    trainer.train(resume_from_checkpoint=last)
    trainer.model.to(torch.bfloat16)
    trainer.save_model(f"{args.out}/final")
    tok.save_pretrained(f"{args.out}/final")


def _tolerate_nonmonotonic_byte_offsets():
    """GOLD raises when the teacher tokenizer splits a multi-byte character into tokens
    whose byte offsets overlap in a way `_normalize_byte_offsets` cannot repair, which
    kills the whole run on one rollout. Fall back to clamping those offsets to be
    monotonic; only texts that would otherwise crash are affected, and each fallback is
    counted so its frequency can be reported."""
    import trl.experimental.utils as gold_utils

    original = gold_utils._normalize_byte_offsets
    counts = {"calls": 0, "fallbacks": 0}

    def normalize(byte_offsets, tokens, text_bytes):
        counts["calls"] += 1
        try:
            return original(byte_offsets, tokens, text_bytes)
        except ValueError:
            counts["fallbacks"] += 1
            print(f"[byte-offset fallback] {counts['fallbacks']}/{counts['calls']} texts", flush=True)
            out, cursor = [], 0
            for start, end in byte_offsets:
                start = max(start, cursor)
                end = max(end, start)
                out.append((start, end))
                cursor = end
            return out

    gold_utils._normalize_byte_offsets = normalize


def _tolerate_out_of_vocab_samples():
    """Embedding matrices are padded past the tokenizer vocabulary (OLMo-2: 100,352 rows,
    100,278 tokens). Sampling at temperature 1 can draw a padding id, which has no token
    string, and GOLD's byte-offset code crashes on it. Treat such ids as zero-width;
    each occurrence is counted."""
    import trl.experimental.gold.gold_trainer as gold_trainer

    original = gold_trainer.piece_byte_len
    counts = {"oov": 0}

    def piece_byte_len(piece):
        if piece is None:
            counts["oov"] += 1
            print(f"[out-of-vocab sample] {counts['oov']} so far", flush=True)
            return 0
        return original(piece)

    gold_trainer.piece_byte_len = piece_byte_len


def _bypass_vllm_compile_wrapper(model):
    """vLLM has no native OLMo-2 implementation, so its Transformers backend wraps the HF
    model class in place for torch.compile when the colocated engine starts. The training
    model was built before that and lacks the wrapper's state; the wrapper calls `forward`
    directly when `do_not_compile` is set. Harmless for unwrapped classes."""
    for m in model.modules():
        m.do_not_compile = True


if __name__ == "__main__":
    main()
