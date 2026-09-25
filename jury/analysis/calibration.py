"""
Persona calibration check.

SYSTEM_DESIGN.md names this as a gate before spending the eval budget: no persona
should collapse into near-identical reasoning with the others'. analysis/
differentiation.py already has the metric (differentiation_report, Jaccard
similarity, threshold 0.25 — see that module for the full reasoning) but nothing
calls it. This is that missing call site.

differentiation_report() expects each juror as {"positions": [...]}, the prototype's
transcript shape. The current architecture's persisted artifact (RunRecord.positions)
is a flat list across ALL jurors AND rounds instead, so _group_by_juror() below adapts
one into the other — grouping by juror_id and sorting each juror's positions by round,
which differentiation_report needs since it reads positions[-1] as "the final round".

Usage:
    python -m jury.analysis.calibration runs/debug --condition control
    python -m jury.analysis.calibration runs/debug --condition treatment
"""

import argparse
from pathlib import Path

from ..domain.models import RunRecord
from .differentiation import differentiation_report
from .metrics import load_runs

# Not a value from differentiation.py or SYSTEM_DESIGN.md — a practical rule of thumb
# for THIS script's summary line only: if more than a fifth of the debug pool shows
# low differentiation, that is worth fixing before the eval spend rather than after.
# Change it, or ignore the line, as you see fit; it does not affect the per-case flags.
ESCALATE_FRACTION = 0.2


def _group_by_juror(record: RunRecord) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for p in record.positions:
        grouped.setdefault(p.juror_id, []).append(p.model_dump(mode="json"))
    for positions in grouped.values():
        positions.sort(key=lambda p: p["round"])
    return [{"positions": positions} for positions in grouped.values()]


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def report(run_dir: Path, condition: str) -> dict:
    records = load_runs(run_dir, condition)
    if not records:
        raise SystemExit(f"No {condition} runs found in {run_dir}")

    per_case = []
    for r in records:
        rep = differentiation_report(_group_by_juror(r))
        rep["case_id"] = r.case_id
        per_case.append(rep)

    # "scored" excludes cases where fewer than 2 jurors produced final-round reasoning
    # text (differentiation_report returns final_avg_similarity=None there) — nothing
    # to compare in that case, not a 0.
    scored = [c for c in per_case if c["final_avg_similarity"] is not None]
    flagged = [c for c in scored if c["low_differentiation_flag"]]

    return {
        "run_dir": str(run_dir),
        "condition": condition,
        "n_cases": len(records),
        "n_scored": len(scored),
        "n_flagged": len(flagged),
        "flagged_fraction": (len(flagged) / len(scored)) if scored else None,
        "mean_round0_similarity": _mean(
            [c["round0_avg_similarity"] for c in scored if c["round0_avg_similarity"] is not None]
        ),
        "mean_final_similarity": _mean([c["final_avg_similarity"] for c in scored]),
        "per_case": per_case,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--condition", default="control")
    args = parser.parse_args()

    r = report(Path(args.run_dir), args.condition)

    print(f"Persona calibration — {r['run_dir']}, condition={r['condition']}, n={r['n_cases']}")
    if r["n_scored"] == 0:
        print("  No cases had enough final-round reasoning text (>=2 jurors) to score.")
        return

    r0 = r["mean_round0_similarity"]
    print(f"  mean round-0 similarity:  {r0:.3f}" if r0 is not None else "  mean round-0 similarity:  n/a")
    print(f"  mean final similarity:    {r['mean_final_similarity']:.3f}")
    print(
        f"  flagged (final similarity >= threshold 0.25): "
        f"{r['n_flagged']}/{r['n_scored']} ({r['flagged_fraction']:.0%})"
    )

    if r["n_flagged"]:
        print("\n  Flagged cases — read these transcripts before trusting the panel's diversity:")
        for c in r["per_case"]:
            if c["low_differentiation_flag"]:
                print(
                    f"    {c['case_id']}: round0={c['round0_avg_similarity']} "
                    f"final={c['final_avg_similarity']} — {c['interpretation']}"
                )

    if r["flagged_fraction"] and r["flagged_fraction"] > ESCALATE_FRACTION:
        print(
            f"\n  More than {ESCALATE_FRACTION:.0%} of scored cases show low differentiation — "
            f"worth fixing the persona prompts or debate protocol BEFORE spending the eval "
            f"budget, not after."
        )


if __name__ == "__main__":
    main()
