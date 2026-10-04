"""SFT warm-start on verified teacher solutions (baseline B1 and the OPD initialisation).

The GOLD recipe distils from an SFT checkpoint rather than the raw student: base
students solve under 8% of Countdown, so their own rollouts carry little signal.
"""

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from countdown import sft_split


def to_prompt_completion(row):
    # Completion-only loss through prompt/completion fields works with every chat
    # template; `assistant_only_loss` would need {% generation %} markers that the
    # OLMo and SmolLM2 templates lack.
    return {"prompt": row["messages"][:-1], "completion": row["messages"][-1:]}


def main():
    # cuDNN SDPA fails on some H100 nodes (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH);
    # flash / mem-efficient SDPA are numerically equivalent here.
    torch.backends.cuda.enable_cudnn_sdp(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=3072)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--n_examples", type=int, default=None, help="SFT data budget (default: whole SFT split)")
    args = ap.parse_args()

    ds = sft_split()
    if args.n_examples:
        ds = ds.select(range(args.n_examples))
    ds = ds.map(to_prompt_completion, remove_columns=["messages"])
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # fp32 master weights; bf16 autocast for compute.
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")

    cfg = SFTConfig(
        output_dir=args.out,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=0.03,  # a float < 1 is a ratio of total steps (transformers v5)
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=0,
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    trainer.model.to(torch.bfloat16)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
