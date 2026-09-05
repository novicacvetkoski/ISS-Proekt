"""
run_case.py — drive one case through the full AI jury pipeline via local Ollama.

Usage (placeholder case, for pipeline testing):
    python run_case.py

Usage (real ILDC case from the debug pool — safe to iterate on):
    python run_case.py --pool debug --index 0

Usage (real ILDC case from the held-out eval pool — Phase 6 only):
    python run_case.py --pool eval --index 0

Run data/prepare_held_out.py first to generate debug_pool.json / eval_pool.json.

Requires Ollama running locally with the model pulled, e.g.:
    ollama pull qwen3:4b
"""

import argparse
import json
import sys
from pathlib import Path

from jurors import (
    TextualistJuror,
    PrecedentHawkJuror,
    EquityAdvocateJuror,
    SkepticalCrossExaminerJuror,
    PragmatistJuror,
)
from jurors.synthesizer import Synthesizer
from orchestrator import DebateOrchestrator, save_transcript

DATA_DIR = Path(__file__).parent / "data"

# Placeholder case for pipeline testing only — NOT a real ILDC case. Used when
# no --pool/--index is given.
PLACEHOLDER_CASE_TEXT = """\
The appellant was convicted under Section 302 read with Section 34 of the Indian
Penal Code for the death of the complainant's brother, arising from an altercation
at a village gathering. The prosecution relies primarily on the testimony of two
eyewitnesses, both relatives of the deceased, and a recovered weapon not conclusively
linked to the appellant by forensic evidence. The defence argues the eyewitnesses are
interested witnesses whose testimony was not corroborated by independent evidence, and
that the appellant acted, if at all, in the heat of the moment without premeditation
following provocation from the deceased's group. The trial court convicted; the High
Court affirmed the conviction on appeal. The appellant now appeals to the Supreme Court.
"""

# Rough word-count budget beyond which the case text risks overflowing num_ctx
# once persona/shared-rules prompt overhead and debate-round context are added.
# This is a coarse heuristic (word count, not real token count) — real ILDC
# judgments regularly run several thousand words, well past what a 4GB card's
# num_ctx=3072 can hold alongside everything else in the prompt.
CASE_LENGTH_WARNING_WORDS = 900


def load_case_from_pool(pool: str, index: int) -> tuple[str, str]:
    """Returns (case_id, case_text) — case_text has the label stripped."""
    pool_path = DATA_DIR / f"{pool}_pool.json"
    if not pool_path.exists():
        print(
            f"{pool_path} not found. Run data/prepare_held_out.py first to "
            "generate it (requires Hugging Face access to the ILDC/CJPE dataset).",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(pool_path, encoding="utf-8") as f:
        records = json.load(f)

    if not (0 <= index < len(records)):
        print(f"Index {index} out of range — {pool_path} has {len(records)} cases (0-{len(records)-1}).",
              file=sys.stderr)
        sys.exit(1)

    record = records[index]
    # get_case_text() is the only sanctioned way to build juror-facing text —
    # deliberately does NOT touch record["label"], so the real outcome never
    # reaches the model.
    from data.ildc_loader import get_case_text
    case_text = get_case_text(record)
    case_id = f"{pool}_{record['id']}"
    return case_id, case_text


def build_jurors():
    return [
        TextualistJuror(),
        PrecedentHawkJuror(),
        EquityAdvocateJuror(),
        SkepticalCrossExaminerJuror(),
        PragmatistJuror(),
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool", choices=["debug", "eval"], default=None,
                         help="Pull a real ILDC case from data/<pool>_pool.json instead of the placeholder")
    parser.add_argument("--index", type=int, default=0, help="Index within the chosen pool")
    args = parser.parse_args()

    if args.pool:
        case_id, case_text = load_case_from_pool(args.pool, args.index)
        if args.pool == "eval":
            print(
                "NOTE: running a case from the HELD-OUT eval pool. This is fine for a "
                "one-off pipeline check, but repeatedly running/inspecting eval cases "
                "during development risks implicitly tuning prompts against them — "
                "prefer --pool debug for iteration.",
                flush=True,
            )
    else:
        case_id, case_text = "placeholder_case_001", PLACEHOLDER_CASE_TEXT

    word_count = len(case_text.split())
    print(f"Case: {case_id} ({word_count} words)", flush=True)
    if word_count > CASE_LENGTH_WARNING_WORDS:
        print(
            f"WARNING: case text is {word_count} words, above the ~{CASE_LENGTH_WARNING_WORDS}-word "
            "heuristic budget for num_ctx=3072 once prompt overhead and debate rounds are added. "
            "This may silently truncate context on Ollama rather than error — watch reasoning "
            "quality closely, and consider raising num_ctx (JurorAgent(num_ctx=...)) if you have "
            "VRAM headroom, or expect degraded results on longer cases.",
            file=sys.stderr,
        )

    print("Initializing jurors...", flush=True)
    jurors = build_jurors()
    synthesizer = Synthesizer()

    orchestrator = DebateOrchestrator(jurors=jurors, synthesizer=synthesizer, max_rounds=3)

    print("Running case through jury (this will take a while on a 4GB card)...", flush=True)
    transcript = orchestrator.run_case(case_id=case_id, case_text=case_text)

    out_path = f"transcript_{case_id}.json"
    save_transcript(transcript, out_path)

    print(f"\nDone. Rounds run: {transcript['rounds_run']} "
          f"(converged early: {transcript['converged_early']})")
    print(f"Final verdict: {transcript['synthesis'].get('final_verdict')}")

    diff = transcript["differentiation"]
    print(f"\nDifferentiation check: round-0 avg similarity = {diff['round0_avg_similarity']}, "
          f"final-round avg similarity = {diff['final_avg_similarity']}")
    if diff["low_differentiation_flag"]:
        print(f"WARNING: {diff['interpretation']}", file=sys.stderr)
    else:
        print(diff["interpretation"])

    if transcript["flagged_citations"]:
        print(f"\nWARNING: {len(transcript['flagged_citations'])} possible fabricated "
              f"citation(s) detected: {transcript['flagged_citations']}", file=sys.stderr)

    total_retries = sum(
        pos.get("format_retries", 0)
        for juror in transcript["jurors"]
        for pos in juror["positions"]
    )
    gave_up = sum(
        1
        for juror in transcript["jurors"]
        for pos in juror["positions"]
        if "[UNPARSED OUTPUT" in pos["reasoning"]
    )
    print(f"\nFormat retries used: {total_retries} total across all positions "
          f"({gave_up} position(s) never parsed even after retries)")
    if gave_up:
        print(f"WARNING: {gave_up} position(s) failed JSON parsing even after retries. "
              "Inspect transcript raw_output fields.", file=sys.stderr)

    print(f"\nFull transcript written to {out_path}")


if __name__ == "__main__":
    main()
