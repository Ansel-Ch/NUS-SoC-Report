"""Measure the hypothesised cross-tokenizer failure modes on real student rollouts.

Reuses GOLD's own alignment, merging and teacher-input construction
(`ULDLoss`, `build_teacher_inputs_from_texts`, `piece_byte_len`) so every quantity is
what `GOLDTrainer` computes during training. Input is the `samples.jsonl` written by
`eval_countdown.py` for a student; nothing is trained.

Per aligned group it records the group shape, a token class, the teacher's probability
mass on vocabulary shared with the student (F8), the sorted-L1 and matched-subset KL
terms GOLD optimises (F2, F10), identity-level top-1 agreement, and the scalar
conditionals used to merge multi-token groups under each merge strategy (F1). Per
sample it records teacher NLL and end-of-sequence probability under the student's
template (what GOLD feeds the teacher) and under the teacher's native template
(F5, F9), and whether the student's sampled ids are a canonical tokenisation (F6).
"""

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.gold import GOLDConfig
from trl.experimental.gold.gold_trainer import ULDLoss, build_teacher_inputs_from_texts
from trl.experimental.utils import piece_byte_len

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from countdown import test_split  # noqa: E402

TEACHER = "Qwen/Qwen3-4B-Instruct-2507"


def token_class(text):
    t = text.strip()
    if not t:
        return "whitespace"
    if re.fullmatch(r"</?(answer|think)>", t) or "<answer" in t or "answer>" in t:
        return "tag"
    if re.fullmatch(r"\d+", t):
        return "digits"
    if re.fullmatch(r"[\d\s+\-*/×÷=().]+", t):
        return "arith"
    return "prose"


def student_offsets(tok, ids):
    offs, cum = [], 0
    for tid in ids:
        nb = piece_byte_len(tok.convert_ids_to_tokens([tid])[0])
        offs.append((cum, cum + nb))
        cum += nb
    return offs


def answer_span(labels):
    idx = (labels != -100).nonzero(as_tuple=True)[0]
    return int(idx[0]), int(len(idx))


def teacher_nll_and_eos(model, ids, labels, eos_id):
    """Mean NLL over completion tokens, and P(EOS) at the final completion position."""
    with torch.no_grad():
        logp = F.log_softmax(model(ids[None]).logits[0].float(), -1)
    start, size = answer_span(labels)
    # logits at p predict token p+1; the last label is the appended EOS.
    tgt = ids[start : start + size - 1]
    lp = logp[start - 1 : start + size - 2].gather(-1, tgt[:, None]).squeeze(-1)
    p_eos = logp[start + size - 2, eos_id].exp().item()
    return -lp.mean().item(), p_eos


def main():
    # cuDNN SDPA fails on some H100 nodes (CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH);
    # flash / mem-efficient SDPA are numerically equivalent here.
    torch.backends.cuda.enable_cudnn_sdp(False)
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True)
    ap.add_argument("--student_tokenizer", default=None)
    ap.add_argument("--samples", required=True, help="samples.jsonl from eval_countdown.py")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    dev = "cuda"

    s_tok = AutoTokenizer.from_pretrained(args.student_tokenizer or args.student)
    t_tok = AutoTokenizer.from_pretrained(TEACHER)
    student = AutoModelForCausalLM.from_pretrained(args.student, dtype=torch.bfloat16).to(dev).eval()
    teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.bfloat16).to(dev).eval()

    cfg = GOLDConfig(output_dir="/tmp/diag", use_uld_loss=True, uld_use_hybrid_loss=True, beta=0.0)
    uld = ULDLoss(cfg, student_tokenizer=s_tok, teacher_tokenizer=t_tok, device=dev)
    t_matched = torch.tensor(sorted(uld._teacher_matched_ids), device=dev)
    s_matched = uld.mapping_tensor[t_matched]
    adaptive_matched_weight = len(t_matched) / len(t_tok)

    prompts = test_split()
    rows = [json.loads(l) for l in open(args.samples)][: args.n]
    groups, samples = [], []

    for r_i, rec in enumerate(rows):
        msgs = prompts[rec.get("idx", r_i)]["prompt"]
        s_prompt = s_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        t_prompt_native = t_tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

        comp = list(rec.get("token_ids") or s_tok(rec["text"], add_special_tokens=False)["input_ids"])
        if comp and comp[-1] == s_tok.eos_token_id:
            comp = comp[:-1]
        if not comp:
            continue
        # Decode exactly as GOLD does; some tokenizers (OLMo) clean up spaces by default.
        comp_text = s_tok.decode(comp, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        # F6: fraction of sampled tokens that are not 1:1 with the canonical re-encoding.
        canon = s_tok(comp_text, add_special_tokens=False)["input_ids"]
        sg_c, cg_c = ULDLoss._align_by_byte_offsets(student_offsets(s_tok, comp), student_offsets(s_tok, canon))
        n_1to1 = sum(1 for a, c in zip(sg_c, cg_c, strict=True) if len(a) == len(c) == 1)
        noncanon_frac = 1 - n_1to1 / len(comp) if canon != comp else 0.0

        # Student sequence exactly as GOLD builds it on-policy.
        p_ids = s_tok(s_prompt, add_special_tokens=False)["input_ids"]
        s_ids = torch.tensor(p_ids + comp + [s_tok.eos_token_id], device=dev)
        s_labels = s_ids.clone()
        s_labels[: len(p_ids)] = -100
        s_offs = [(0, 0)] * len(p_ids) + student_offsets(s_tok, comp) + [(0, 0)]
        s_offs = torch.tensor(s_offs, device=dev)

        # Teacher sequences: GOLD's (student template as text) and native.
        t_ids, t_labels, _, t_offs = build_teacher_inputs_from_texts(t_tok, [s_prompt], [comp_text])
        tn_ids, tn_labels, _, _ = build_teacher_inputs_from_texts(t_tok, [t_prompt_native], [comp_text])
        t_ids, t_labels, t_offs = t_ids[0].to(dev), t_labels[0].to(dev), t_offs[0].to(dev)
        tn_ids, tn_labels = tn_ids[0].to(dev), tn_labels[0].to(dev)

        nll_gold, eos_gold = teacher_nll_and_eos(teacher, t_ids, t_labels, t_tok.eos_token_id)
        nll_native, eos_native = teacher_nll_and_eos(teacher, tn_ids, tn_labels, t_tok.eos_token_id)

        with torch.no_grad():
            s_logits = student(s_ids[None]).logits[0].float()
            t_logits = teacher(t_ids[None]).logits[0].float()

        s_start, s_size = answer_span(s_labels)
        t_start, t_size = answer_span(t_labels)
        s_size, t_size = s_size - 1, t_size - 1  # uld_skip_*_eos=True (default)
        s_ans_offs = s_offs[s_start : s_start + s_size].tolist()
        t_ans_offs = t_offs[t_start : t_start + t_size].tolist()
        sg, tg = uld._align_by_byte_offsets(s_ans_offs, t_ans_offs)
        pairs = [(a, b) for a, b in zip(sg, tg, strict=False) if a and b]

        per_strategy = {}
        for strategy, shift in (("observed", 0), ("bayesian", 1)):
            uld.token_merge_strategy = strategy
            s_probs = F.softmax(s_logits[s_start - shift : s_start + s_size - shift], -1)
            t_probs = F.softmax(t_logits[t_start - shift : t_start + t_size - shift], -1)
            s_tids = s_ids[s_start : s_start + s_size].tolist()
            t_tids = t_ids[t_start : t_start + t_size].tolist()
            s_al = uld._merge_probabilities_with_alignment_groups(s_probs, [a for a, _ in pairs], s_tids)
            t_al = uld._merge_probabilities_with_alignment_groups(t_probs, [b for _, b in pairs], t_tids)
            # Scalar conditionals multiplied into each merged teacher group.
            scal = []
            for _, b in pairs:
                pos = b[:-1] if strategy == "bayesian" else b[1:]
                scal.append([t_probs[p, t_tids[p]].item() for p in pos])
            per_strategy[strategy] = (s_al, t_al, scal)

        del s_logits, t_logits
        s_al, t_al, scal_obs = per_strategy["observed"]  # GOLD's default, i.e. what trains
        s_al_b, t_al_b, scal_bay = per_strategy["bayesian"]
        t_mm = t_al[:, t_matched]
        s_mm = s_al[:, s_matched]
        mass_matched = (t_mm.sum(-1) / t_al.sum(-1).clamp_min(1e-12)).tolist()
        kl_matched = (t_mm * (t_mm.clamp_min(1e-8).log() - s_mm.clamp_min(1e-8).log())).sum(-1).tolist()
        # Sorted L1 with the smaller vocabulary zero-padded, as in ULDLoss; chunked
        # because full-vocabulary sorts of 2k-token rollouts do not fit in 40 GB.
        l1_sorted = []
        for c in range(0, len(pairs), 256):
            sc = s_al[c : c + 256].sort(-1, descending=True).values
            tc = t_al[c : c + 256].sort(-1, descending=True).values
            v = max(sc.size(-1), tc.size(-1))
            l1_sorted += (F.pad(sc, (0, v - sc.size(-1))) - F.pad(tc, (0, v - tc.size(-1)))).abs().sum(-1).tolist()
            del sc, tc
        # Identity-level agreement: does the teacher's argmax map to the student's argmax?
        t_top = t_al.argmax(-1)
        s_top = s_al.argmax(-1)
        mapped = torch.where(
            t_top < len(uld.mapping_tensor), uld.mapping_tensor[t_top.clamp_max(len(uld.mapping_tensor) - 1)], -1
        )
        top1_agree = (mapped == s_top).tolist()
        t_top_b = t_al_b.argmax(-1)
        mapped_b = torch.where(
            t_top_b < len(uld.mapping_tensor), uld.mapping_tensor[t_top_b.clamp_max(len(uld.mapping_tensor) - 1)], -1
        )
        top1_agree_b = (mapped_b == s_al_b.argmax(-1)).tolist()
        texts = [s_tok.decode([s_ids[s_start + i].item() for i in a]) for a, _ in pairs]

        for g, (a, b) in enumerate(pairs):
            groups.append(
                {
                    "sample": r_i,
                    "correct": rec["correct"],
                    "n_s": len(a),
                    "n_t": len(b),
                    "cls": token_class(texts[g]),
                    # Under "observed" the distribution at a group predicts what follows it.
                    "next_cls": token_class(texts[g + 1]) if g + 1 < len(texts) else "end",
                    "mass_matched": mass_matched[g],
                    "kl_matched": kl_matched[g],
                    "l1_sorted": l1_sorted[g],
                    "top1_agree": bool(top1_agree[g]),
                    "top1_agree_bayesian": bool(top1_agree_b[g]),
                    "scalar_observed": scal_obs[g],
                    "scalar_bayesian": scal_bay[g],
                }
            )
        samples.append(
            {
                "sample": r_i,
                "correct": rec["correct"],
                "noncanonical_token_frac": noncanon_frac,
                "n_groups": len(pairs),
                "nll_gold_ctx": nll_gold,
                "nll_native_ctx": nll_native,
                "p_eos_gold_ctx": eos_gold,
                "p_eos_native_ctx": eos_native,
            }
        )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "groups.jsonl"), "w") as f:
        f.writelines(json.dumps(g) + "\n" for g in groups)
    with open(os.path.join(args.out, "samples.jsonl"), "w") as f:
        f.writelines(json.dumps(s) + "\n" for s in samples)

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    one = [g for g in groups if g["n_s"] == g["n_t"] == 1]
    multi = [g for g in groups if not (g["n_s"] == g["n_t"] == 1)]
    by_cls = defaultdict(list)
    for g in groups:
        by_cls[g["next_cls"]].append(g)
    summary = {
        "student": args.student,
        "n_samples": len(samples),
        "n_groups": len(groups),
        "F1_F10_frac_groups_1to1": len(one) / max(1, len(groups)),
        "F1_mean_merge_scalar_observed": mean(x for g in multi for x in g["scalar_observed"]),
        "F1_mean_merge_scalar_bayesian": mean(x for g in multi for x in g["scalar_bayesian"]),
        "F2_top1_agree_1to1": mean(g["top1_agree"] for g in one),
        "F2_top1_agree_multi": mean(g["top1_agree"] for g in multi),
        "F1_top1_agree_multi_bayesian": mean(g["top1_agree_bayesian"] for g in multi),
        "F8_adaptive_matched_weight": adaptive_matched_weight,
        "F8_mean_teacher_mass_on_matched": mean(g["mass_matched"] for g in groups),
        "F10_kl_matched_1to1": mean(g["kl_matched"] for g in one),
        "F10_kl_matched_multi": mean(g["kl_matched"] for g in multi),
        "F10_frac_negative_kl_matched": mean(g["kl_matched"] < 0 for g in groups),
        "F9_teacher_nll_gold_ctx": mean(s["nll_gold_ctx"] for s in samples),
        "F9_teacher_nll_native_ctx": mean(s["nll_native_ctx"] for s in samples),
        "F5_p_eos_gold_ctx": mean(s["p_eos_gold_ctx"] for s in samples),
        "F5_p_eos_native_ctx": mean(s["p_eos_native_ctx"] for s in samples),
        "F6_frac_samples_noncanonical": mean(s["noncanonical_token_frac"] > 0 for s in samples),
        "F6_frac_tokens_noncanonical": mean(s["noncanonical_token_frac"] for s in samples),
        "by_predicted_class": {
            c: {
                "n": len(gs),
                "frac_1to1": mean(g["n_s"] == g["n_t"] == 1 for g in gs),
                "top1_agree": mean(g["top1_agree"] for g in gs),
                "kl_matched": mean(g["kl_matched"] for g in gs),
                "l1_sorted": mean(g["l1_sorted"] for g in gs),
            }
            for c, gs in sorted(by_cls.items())
        },
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
