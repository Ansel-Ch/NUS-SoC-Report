"""Evaluate a model on the Countdown test split with vLLM.

Writes per-sample generations (for failure analysis) and a metrics summary.

`--template_from` renders prompts with another model's chat template and feeds the
result to the evaluated model as text. Evaluating the teacher with a student's
template reproduces the context the teacher sees inside GOLD's cross-tokenizer loss
(failure mode F9).
"""

import argparse
import json
import os
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from countdown import score, test_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--template_from", default=None)
    ap.add_argument("--n_prompts", type=int, default=1000)
    ap.add_argument("--n_samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--gpu_mem", type=float, default=0.85)
    args = ap.parse_args()

    ds = test_split(args.n_prompts)
    tok = AutoTokenizer.from_pretrained(args.template_from or args.model)
    prompts = [tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=False) for r in ds]

    llm = LLM(
        args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        seed=0,
    )
    params = SamplingParams(
        n=args.n_samples,
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        seed=0,
    )
    # Prompts already contain the rendered template (and BOS where the template adds
    # one), so they are passed as raw text without re-applying a template.
    outputs = llm.generate(prompts, params)

    os.makedirs(args.out, exist_ok=True)
    reasons, lengths, truncated, correct = Counter(), [], 0, 0
    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        for idx, (row, out) in enumerate(zip(ds, outputs, strict=True)):
            for o in out.outputs:
                ok, reason = score(o.text, row["nums"], row["target"])
                correct += ok
                reasons[reason] += 1
                lengths.append(len(o.token_ids))
                truncated += o.finish_reason == "length"
                rec = {
                    "idx": idx,
                    "nums": row["nums"],
                    "target": row["target"],
                    "correct": ok,
                    "reason": reason,
                    "n_tokens": len(o.token_ids),
                    "finish_reason": o.finish_reason,
                    "text": o.text,
                    # Sampled ids, which may be a non-canonical tokenisation of `text` (F6).
                    "token_ids": list(o.token_ids),
                }
                f.write(json.dumps(rec) + "\n")

    total = len(lengths)
    metrics = {
        "model": args.model,
        "template_from": args.template_from,
        "n_prompts": len(ds),
        "n_samples": args.n_samples,
        "temperature": args.temperature,
        "accuracy": correct / total,
        "truncation_rate": truncated / total,
        "mean_tokens": sum(lengths) / total,
        "reasons": {k: v / total for k, v in reasons.items()},
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)
    print(json.dumps(metrics, indent=1))


if __name__ == "__main__":
    main()
