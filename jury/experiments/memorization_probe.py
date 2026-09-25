"""
Memorization / contamination probe.

ILDC/CJPE judgments are public and pre-2020 (docs/HANDOFF.md §2.3): Gemini may already
know a case's outcome from pretraining, not from reasoning over the case materials.
That would inflate RQ1's absolute agreement-with-outcome number for reasons that have
nothing to do with the jury's reasoning quality. RQ2 (the paired persuader contrast) is
argued to be robust to this regardless — but RQ1 is not, so this has to be reported
*before* the full eval spend (SYSTEM_DESIGN.md §6, phase 5 — a gate, not an afterthought).

The probe asks the pinned juror model, on the SAME case text a juror would see
(get_case_text() — the gold label never reaches this prompt, same as everywhere else,
CLAUDE.md rule 1), two separate things: does it recognize this as a specific real case
(not just a familiar area of law), and if so, what outcome does it recall for THAT
case — as memory, not as a reasoned guess from the materials. Recall accuracy is only
scored among cases the model claims to recognize; a low recognition rate paired with
high recall accuracy on the recognized subset is the signature of genuine
contamination, as distinct from the model just being a good legal reasoner.

This is diagnostic, not part of the deliberation pipeline: it never touches jurors/,
deliberation/, or the real experiment's run directory. Its artifacts live under their
own runs/memorization_probe/ tree so they can never be mistaken for a RunRecord, and
MemorizationProbeResult is a distinct schema for the same reason.

Usage:
    python -m jury.experiments.memorization_probe --config configs/eval.yaml --pool eval --sample-size 20
    python -m jury.experiments.memorization_probe --config configs/eval.yaml --pool eval --report-only
"""

import argparse
import json
import random
import sys
from pathlib import Path

from ..config import ExperimentConfig, load_config
from ..data.case_file import build_case_file
from ..data.ildc_loader import get_case_text
from ..domain.models import MemorizationProbeResponse, MemorizationProbeResult
from ..llm.factory import build_client
from .runner import load_pool, write_atomic

REPO_ROOT = Path(__file__).resolve().parents[2]

# Deliberately separated from any legal-reasoning framing: a jury-style prompt invites
# the model to reason its way to a plausible-sounding outcome, which is exactly the
# false positive this probe exists to rule out. This is a recall question, not a task.
PROBE_SYSTEM = """You are being asked a narrow factual question, separate from any legal \
reasoning task. Do not analyze or reason about this case's merits — that is not what is \
being asked.

You will be shown the text of a real court case with party names and identifying \
details removed. Answer only:

1. Do you specifically RECOGNIZE this as a real, particular case you have prior \
knowledge of — not just a familiar area of law, but this specific dispute? Most cases \
shown to you will be ones you do not recognize; say so plainly when that is true. Only \
answer yes if you can point to a specific fact, phrase, or procedural detail that \
identifies it for you, not because the general topic feels familiar.
2. If, and only if, you recognize it: what outcome do you recall for THIS case — from \
memory of the actual case, not inferred from the materials below? If you do not \
recognize the case, or recognize the topic but not this specific outcome, say UNSURE. \
Guessing from legal reasoning here defeats the entire point of the question."""

PROBE_TASK = (
    "Based only on genuine prior recognition — never on reasoning about the materials "
    "below — answer whether you recognize this specific case and what you recall of "
    "its outcome."
)


def run_probe(cfg: ExperimentConfig, pool: str, sample_size: int, seed: int, force: bool) -> None:
    records = load_pool(pool)
    rng = random.Random(seed)
    sample = rng.sample(records, k=min(sample_size, len(records))) if sample_size else records

    client = build_client(cfg.juror_model)  # the exact model the real jurors will use
    out_dir = REPO_ROOT / "runs" / "memorization_probe" / pool

    if pool == "eval":
        print(
            "NOTE: this spends real calls against the HELD-OUT eval pool. That is the "
            "point — it must run before the full eval spend, not instead of it.",
            flush=True,
        )

    for i, record in enumerate(sample, 1):
        case_id = str(record["id"])
        out_path = out_dir / f"{case_id}.json"
        if out_path.exists() and not force:
            print(f"  [{i}/{len(sample)}] {case_id} already probed — skipping")
            continue

        # get_case_text() is the only sanctioned path to case text — same choke point
        # the real jurors go through. The label is never read here.
        case = build_case_file(case_id, get_case_text(record))

        try:
            parsed, usage = client.generate(
                system=PROBE_SYSTEM,
                prefix=case.render(),
                task=PROBE_TASK,
                schema=MemorizationProbeResponse,
                temperature=0.0,  # factual recall, not reasoning diversity — low on purpose
                seed=cfg.seed,
            )
        except Exception as e:
            print(f"  [{i}/{len(sample)}] {case_id} FAILED: {e}", file=sys.stderr, flush=True)
            continue

        result = MemorizationProbeResult(
            case_id=case_id,
            pool=pool,
            recognizes_case=parsed.recognizes_case,
            case_identification=parsed.case_identification,
            recalled_outcome=parsed.recalled_outcome,
            recall_basis=parsed.recall_basis,
            usage=usage,
            provenance={**cfg.provenance(), "probe": "memorization", "pool": pool},
        )
        write_atomic(out_path, result.model_dump_json(indent=2))
        flag = "RECOGNIZES" if result.recognizes_case else "clean"
        print(f"  [{i}/{len(sample)}] {case_id} — {flag}", flush=True)


def report(pool: str) -> None:
    # Imported here, not at module top: this keeps the probe importable/runnable even
    # in a minimal install, same lazy-import spirit as the rest of jury/data/.
    from ..analysis.metrics import load_gold
    from ..domain.verdicts import LABEL_TO_VERDICT

    out_dir = REPO_ROOT / "runs" / "memorization_probe" / pool
    paths = sorted(p for p in out_dir.glob("*.json") if p.name != "_summary.json")
    if not paths:
        sys.exit(f"No probe results under {out_dir} yet — run the probe first.")

    results = [MemorizationProbeResult.model_validate_json(p.read_text(encoding="utf-8")) for p in paths]
    gold = load_gold(pool)

    recognized = [r for r in results if r.recognizes_case]
    scored = [
        r for r in recognized
        if r.case_id in gold and r.recalled_outcome in ("ALLOW", "DISMISS")
    ]
    correct = [
        r for r in scored
        if r.recalled_outcome == LABEL_TO_VERDICT[gold[r.case_id]].value.upper()
    ]

    print(f"\nMemorization probe — {pool} pool, {len(results)} cases")
    print(f"  recognized as a specific case: {len(recognized)}/{len(results)} "
          f"({len(recognized) / len(results):.0%})")
    if scored:
        print(f"  of those, recalled outcome matched gold: {len(correct)}/{len(scored)} "
              f"({len(correct) / len(scored):.0%})")
    else:
        print("  no recognized case gave a scoreable (non-UNSURE) recalled outcome")

    if recognized:
        print("\n  Recognized cases — report these individually regardless of the aggregate:")
        for r in recognized:
            gold_verdict = LABEL_TO_VERDICT.get(gold.get(r.case_id))
            if gold_verdict is None:
                match = "no gold"
            elif r.recalled_outcome == gold_verdict.value.upper():
                match = "MATCH"
            else:
                match = "no match"
            print(f"    {r.case_id}: recalled={r.recalled_outcome} ({match}) — {r.case_identification!r}")

    summary = {
        "pool": pool,
        "n": len(results),
        "recognized": len(recognized),
        "recognition_rate": len(recognized) / len(results),
        "scored": len(scored),
        "recall_accuracy": (len(correct) / len(scored)) if scored else None,
        "recognized_case_ids": [r.case_id for r in recognized],
    }
    write_atomic(out_dir / "_summary.json", json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--pool", choices=["debug", "eval"], help="Defaults to the config's own pool")
    parser.add_argument(
        "--sample-size", type=int, default=20,
        help="Cases to sample (0 = whole pool). Default 20 — cheap, run before the full eval spend.",
    )
    parser.add_argument("--seed", type=int, help="Defaults to the config's seed")
    parser.add_argument("--force", action="store_true", help="Re-probe cases that already have a result")
    parser.add_argument(
        "--report-only", action="store_true",
        help="Skip calling the model; just report on results already on disk",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool = args.pool or cfg.pool
    seed = args.seed if args.seed is not None else cfg.seed

    if not args.report_only:
        run_probe(cfg, pool=pool, sample_size=args.sample_size, seed=seed, force=args.force)
    report(pool)


if __name__ == "__main__":
    main()
