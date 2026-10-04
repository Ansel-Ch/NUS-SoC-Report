"""Results table: accuracy, teacher-gap recovery R, and paired bootstrap CIs.

Every evaluation scores the same test prompts in the same order, so differences are
estimated per prompt (paired), which is much tighter than comparing two accuracies.

R = (acc_OPD - acc_SFT) / (acc_teacher - acc_SFT) normalises away the student's
starting point; its CI resamples prompts jointly for the three runs involved.
"""

import argparse
import json
import os

import numpy as np

# OPD run -> the SFT checkpoint it started from.
RUNS = {
    "opd_G1_qwen25_gkd": "student_qwen25_n1000_sft",
    "opd_G2_qwen25_uld": "student_qwen25_n1000_sft",
    "opd_X1_olmo2_gold": "student_olmo2_n2000_sft",
    "opd_X3_olmo2_uld": "student_olmo2_n2000_sft",
    # X5 was stopped at step 100 once it had answered its question; evaluated mid-run.
    "midrun_X5_olmo2_gold_bayes_step100": "student_olmo2_n2000_sft",
    "opd_X7_olmo2_gold_lr1e-7": "student_olmo2_n2000_sft",
    "opd_X2_smollm2_gold": "student_smollm2_n5000_sft",
    "opd_X6_smollm2_gold_w": "student_smollm2_n5000_sft",
}
TEACHER = "teacher"


def load(res, name):
    path = os.path.join(res, name, "samples.jsonl")
    if not os.path.exists(path):
        return None
    return np.array([json.loads(line)["correct"] for line in open(path)], dtype=float)


def ci(samples):
    return np.percentile(samples, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", default=os.path.expanduser("~/opd/results/eval"))
    ap.add_argument("--boot", type=int, default=10_000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    t = load(args.res, TEACHER)
    print(f"teacher acc={t.mean():.3f}\n")
    print(f"{'run':28s} {'SFT':>6s} {'OPD':>6s} {'Δacc [95% CI]':>22s} {'R [95% CI]':>22s}")
    for run, sft_name in RUNS.items():
        o, s = load(args.res, run), load(args.res, sft_name)
        if o is None or s is None:
            print(f"{run:28s} pending")
            continue
        n = len(o)
        s, tt = s[:n], t[:n]  # mid-run evaluations use a prefix of the same test prompts
        idx = rng.integers(0, n, size=(args.boot, n))
        d = o[idx].mean(1) - s[idx].mean(1)
        gap = tt[idx].mean(1) - s[idx].mean(1)
        r = d / np.where(gap > 0, gap, np.nan)
        delta, R = o.mean() - s.mean(), (o.mean() - s.mean()) / (tt.mean() - s.mean())
        lo, hi = ci(d)
        rlo, rhi = np.nanpercentile(r, [2.5, 97.5])
        print(f"{run:28s} {s.mean():6.3f} {o.mean():6.3f} {delta:+7.3f} [{lo:+.3f},{hi:+.3f}] "
              f"{R:+7.2f} [{rlo:+.2f},{rhi:+.2f}]")


if __name__ == "__main__":
    main()
