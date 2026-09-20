"""
Shared scoring helpers.

This is where the gold label finally enters the system: run artifacts are written
label-free, and analysis joins them against the pool afterwards. That ordering is what
makes it structurally impossible for an outcome to leak into a prompt.
"""

import json
import math
from pathlib import Path

from ..domain.models import RunRecord
from ..domain.verdicts import VERDICT_TO_LABEL, Verdict

# Reference points from the literature, for reporting alongside our numbers.
REFERENCES = {
    "majority_class_test_split": 0.5023,     # ILDC test split is 50.23% accepted
    "gpt4_zero_shot_macro_f1": 0.6829,       # IL-TUR (ACL 2024), CJPE
    "finetuned_sota_macro_f1": 0.8131,       # IL-TUR
    "human_expert_accuracy": 0.94,           # ILDC paper, 56 expert-annotated cases
}


def load_runs(run_dir: Path, condition: str) -> list[RunRecord]:
    records = []
    for path in sorted(run_dir.glob(f"*/{condition}.json")):
        records.append(RunRecord.model_validate_json(path.read_text(encoding="utf-8")))
    return records


def load_gold(pool: str) -> dict[str, int]:
    path = Path(__file__).resolve().parents[2] / "data" / f"{pool}_pool.json"
    pool_records = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["id"]): int(r["label"]) for r in pool_records}


def predicted_label(record: RunRecord) -> int | None:
    v = record.verdict.verdict
    return VERDICT_TO_LABEL[v] if v is not None else None


def accuracy(pairs: list[tuple[int, int]]) -> float:
    """pairs: (predicted, gold)."""
    if not pairs:
        return float("nan")
    return sum(1 for p, g in pairs if p == g) / len(pairs)


def confusion(pairs: list[tuple[int, int]]) -> dict[str, int]:
    tp = sum(1 for p, g in pairs if p == 1 and g == 1)
    tn = sum(1 for p, g in pairs if p == 0 and g == 0)
    fp = sum(1 for p, g in pairs if p == 1 and g == 0)
    fn = sum(1 for p, g in pairs if p == 0 and g == 1)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def macro_f1(pairs: list[tuple[int, int]]) -> float:
    c = confusion(pairs)
    f1s = []
    for pos, neg in (("tp", "fn"), ("tn", "fp")):
        tp = c[pos]
        fn = c[neg]
        fp = c["fp"] if pos == "tp" else c["fn"]
        denom = 2 * tp + fp + fn
        f1s.append((2 * tp / denom) if denom else 0.0)
    return sum(f1s) / 2


def cohens_kappa(pairs: list[tuple[int, int]]) -> float:
    """Agreement with the court, corrected for chance."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    po = accuracy(pairs)
    pred1 = sum(1 for p, _ in pairs if p == 1) / n
    gold1 = sum(1 for _, g in pairs if g == 1) / n
    pe = pred1 * gold1 + (1 - pred1) * (1 - gold1)
    return (po - pe) / (1 - pe) if pe != 1 else float("nan")


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score CI — behaves at small n and near 0/1, unlike the normal approximation."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def mcnemar(a_correct: list[bool], b_correct: list[bool]) -> dict:
    """
    Exact-ish McNemar on paired correctness. Only discordant pairs carry information,
    which is why this is under-powered at n=100 — stated up front in SYSTEM_DESIGN §5.5.
    """
    if len(a_correct) != len(b_correct):
        raise ValueError("Paired comparison requires equal-length inputs")
    b = sum(1 for x, y in zip(a_correct, b_correct) if x and not y)
    c = sum(1 for x, y in zip(a_correct, b_correct) if y and not x)
    result = {"a_only_correct": b, "b_only_correct": c, "discordant": b + c}
    try:
        from statsmodels.stats.contingency_tables import mcnemar as sm_mcnemar

        table = [[sum(1 for x, y in zip(a_correct, b_correct) if x and y), b],
                 [c, sum(1 for x, y in zip(a_correct, b_correct) if not x and not y)]]
        res = sm_mcnemar(table, exact=(b + c) < 25)
        result["statistic"] = float(res.statistic)
        result["p_value"] = float(res.pvalue)
    except ImportError:
        result["p_value"] = None
    return result


def bootstrap_delta(
    a: list[bool], b: list[bool], iterations: int = 10_000, seed: int = 42
) -> tuple[float, float, float]:
    """Paired bootstrap of (mean(b) - mean(a)). Returns (delta, lo, hi) at 95%."""
    import random as _random

    rng = _random.Random(seed)
    n = len(a)
    if n == 0:
        return (float("nan"),) * 3
    observed = (sum(b) - sum(a)) / n
    deltas = []
    indices = range(n)
    for _ in range(iterations):
        sample = [rng.choice(indices) for _ in indices]
        deltas.append((sum(b[i] for i in sample) - sum(a[i] for i in sample)) / n)
    deltas.sort()
    return observed, deltas[int(0.025 * iterations)], deltas[int(0.975 * iterations)]


def verdict_counts(records: list[RunRecord]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        key = r.verdict.verdict.value if r.verdict.verdict else "hung"
        counts[key] = counts.get(key, 0) + 1
    return counts


def round0_majority(record: RunRecord) -> Verdict | None:
    """The no-deliberation baseline (B2), recoverable from any control run for free."""
    from ..domain.verdicts import majority_verdict

    votes = [p.verdict for p in record.positions if p.round == 0]
    verdict, _ = majority_verdict(votes)
    return verdict
