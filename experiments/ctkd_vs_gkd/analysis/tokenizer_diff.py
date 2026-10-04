"""Compare how teacher and student tokenizers segment Countdown-relevant text.

For each student, reports vocabulary overlap with the teacher and, on a set of probe
strings, whether the two segmentations share token boundaries. A span where the
boundaries disagree is where GOLD must merge probabilities, so the boundary
agreement rate approximates how much of the loss is computed exactly versus
approximately.
"""

import argparse
import json

from transformers import AutoTokenizer

TEACHER = "Qwen/Qwen3-4B-Instruct-2507"
STUDENTS = [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "allenai/OLMo-2-0425-1B-Instruct",
    "HuggingFaceTB/SmolLM2-1.7B-Instruct",
    "meta-llama/Llama-3.2-1B-Instruct",
]
PROBES = {
    "numbers": "Using the numbers [19, 36, 55, 7], create an equation that equals 65.",
    "multidigit": "1234 + 5678 = 6912, 100 - 37 = 63, 999 / 3 = 333",
    "equation": "<answer> (55 + 36 - 19) - 7 </answer>",
    "reasoning": "Let me try 55 + 36 = 91, then 91 - 19 = 72, and 72 - 7 = 65. That works!",
    "tool_call": '<tool_call>\n{"name": "search", "arguments": {"query": "weather in Singapore"}}\n</tool_call>',
}


def boundaries(tok, text):
    """Character offsets at which each token ends."""
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    return [end for _, end in enc["offset_mapping"]], enc["input_ids"]


def align_groups(bs, bt):
    """GOLD's byte-offset alignment (ULDLoss._align_by_byte_offsets) on token end offsets."""
    groups, i, j, gi, gj = [], 0, 0, 0, 0
    while i < len(bs) and j < len(bt):
        if bs[i] < bt[j]:
            i += 1
        elif bs[i] > bt[j]:
            j += 1
        else:
            i += 1
            j += 1
            groups.append((i - gi, j - gj))
            gi, gj = i, j
    return groups


def vocab_overlap(t, s):
    vt, vs = set(t.get_vocab()), set(s.get_vocab())
    inter = len(vt & vs)
    return {"jaccard": inter / len(vt | vs), "student_in_teacher": inter / len(vs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    teacher = AutoTokenizer.from_pretrained(TEACHER)
    report = {}
    for name in STUDENTS:
        try:
            student = AutoTokenizer.from_pretrained(name)
        except Exception as e:  # gated model without token
            print(f"skip {name}: {type(e).__name__}")
            continue
        entry = {"overlap": vocab_overlap(teacher, student), "probes": {}}
        print(f"\n=== {name}  {entry['overlap']}")
        for key, text in PROBES.items():
            bt, ids_t = boundaries(teacher, text)
            bs, ids_s = boundaries(student, text)
            # Fraction of student tokens in 1:1 groups: only these are compared
            # exactly; the rest fall into merged groups (F1, F10).
            groups = align_groups(bs, bt)
            agree = sum(ns for ns, nt in groups if ns == nt == 1) / len(bs)
            entry["probes"][key] = {
                "n_teacher": len(ids_t),
                "n_student": len(ids_s),
                "one_to_one_frac": agree,
                "teacher_tokens": teacher.convert_ids_to_tokens(ids_t),
                "student_tokens": student.convert_ids_to_tokens(ids_s),
            }
            print(f"  {key:10s} T={len(ids_t):3d} S={len(ids_s):3d} 1:1={agree:.2f}")
            if key in ("multidigit", "tool_call"):
                print("    T:", teacher.convert_ids_to_tokens(ids_t)[:16])
                print("    S:", student.convert_ids_to_tokens(ids_s)[:16])
        report[name] = entry
    if args.out:
        with open(args.out, "w") as f:
            json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
