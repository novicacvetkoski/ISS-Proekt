"""
RQ1 — does a panel of independent agents produce usable verdicts, and does
DELIBERATION add anything over five independent opinions?

The B2 comparison is the one that matters. Accuracy against gold is contamination-
sensitive (these judgments are public and pre-2020), but "full jury vs round-0 majority"
is an internal contrast on identical inputs, so it measures the deliberation mechanism
itself rather than the model's recall.

Usage:
    python -m jury.analysis.rq1 runs/eval_v1 --pool eval
"""

import argparse
from pathlib import Path

from ..domain.verdicts import VERDICT_TO_LABEL
from .metrics import (
    REFERENCES,
    accuracy,
    cohens_kappa,
    confusion,
    load_gold,
    load_runs,
    macro_f1,
    mcnemar,
    predicted_label,
    round0_majority,
    verdict_counts,
    wilson_interval,
)


def report(run_dir: Path, pool: str, condition: str = "control") -> dict:
    gold = load_gold(pool)
    records = [r for r in load_runs(run_dir, condition) if r.case_id in gold]
    if not records:
        raise SystemExit(f"No {condition} runs found in {run_dir}")

    jury_pairs, b2_pairs = [], []
    jury_correct, b2_correct = [], []
    for r in records:
        g = gold[r.case_id]
        p = predicted_label(r)
        if p is None:                      # hung: counted as an error, and reported separately
            jury_correct.append(False)
        else:
            jury_pairs.append((p, g))
            jury_correct.append(p == g)

        v0 = round0_majority(r)
        if v0 is None:
            b2_correct.append(False)
        else:
            b2_pairs.append((VERDICT_TO_LABEL[v0], g))
            b2_correct.append(VERDICT_TO_LABEL[v0] == g)

    hits = sum(jury_correct)
    lo, hi = wilson_interval(hits, len(records))

    result = {
        "condition": condition,
        "n": len(records),
        "hung": sum(1 for r in records if r.verdict.verdict is None),
        "verdicts": verdict_counts(records),
        "jury": {
            "accuracy": accuracy(jury_pairs),
            "accuracy_ci95": (lo, hi),
            "macro_f1": macro_f1(jury_pairs),
            "cohens_kappa": cohens_kappa(jury_pairs),
            "confusion": confusion(jury_pairs),
        },
        "b2_round0_majority": {
            "accuracy": accuracy(b2_pairs),
            "macro_f1": macro_f1(b2_pairs),
        },
        "deliberation_effect": mcnemar(b2_correct, jury_correct),
        "unanimity_rate": sum(1 for r in records if r.verdict.unanimous) / len(records),
        "mean_rounds": sum(r.rounds_run for r in records) / len(records),
        "references": REFERENCES,
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--pool", default="eval")
    parser.add_argument("--condition", default="control")
    args = parser.parse_args()

    r = report(Path(args.run_dir), args.pool, args.condition)
    j, b2 = r["jury"], r["b2_round0_majority"]

    print(f"RQ1 — {r['condition']}, n={r['n']}")
    print(f"  verdicts:        {r['verdicts']}  (hung: {r['hung']})")
    print(f"  accuracy:        {j['accuracy']:.3f}  95% CI [{j['accuracy_ci95'][0]:.3f}, {j['accuracy_ci95'][1]:.3f}]")
    print(f"  macro-F1:        {j['macro_f1']:.3f}")
    print(f"  Cohen's kappa:   {j['cohens_kappa']:.3f}")
    print(f"  confusion:       {j['confusion']}")
    print(f"  unanimity rate:  {r['unanimity_rate']:.2f}   mean rounds: {r['mean_rounds']:.2f}")
    print()
    print("  Does deliberation help? (B2 = round-0 majority, no debate)")
    print(f"    B2 accuracy:   {b2['accuracy']:.3f}   macro-F1 {b2['macro_f1']:.3f}")
    print(f"    jury accuracy: {j['accuracy']:.3f}")
    print(f"    McNemar:       {r['deliberation_effect']}")
    print()
    print("  Reference points:")
    for k, v in r["references"].items():
        print(f"    {k:34s} {v:.4f}")


if __name__ == "__main__":
    main()
