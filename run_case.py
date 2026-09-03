"""
run_case.py — drive one case through the full AI jury pipeline.

Usage:
    python run_case.py

Requires Ollama running locally with the model pulled, e.g.:
    ollama pull qwen3:4b

This script uses a short placeholder case for pipeline testing. Swap CASE_TEXT
for a real preprocessed ILDC case (facts only — outcome stripped, per Phase 3)
once the data pipeline is in place.
"""

import sys

from jurors import (
    TextualistJuror,
    PrecedentHawkJuror,
    EquityAdvocateJuror,
    SkepticalCrossExaminerJuror,
    PragmatistJuror,
)
from jurors.synthesizer import Synthesizer
from orchestrator import DebateOrchestrator, save_transcript

# Placeholder case for pipeline testing only — NOT a real ILDC case.
# Replace with actual preprocessed case facts (outcome stripped) once
# the Phase 3 data pipeline produces them.
CASE_TEXT = """\
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


def build_jurors():
    return [
        TextualistJuror(),
        PrecedentHawkJuror(),
        EquityAdvocateJuror(),
        SkepticalCrossExaminerJuror(),
        PragmatistJuror(),
    ]


def main():
    print("Initializing jurors...", flush=True)
    jurors = build_jurors()
    synthesizer = Synthesizer()

    orchestrator = DebateOrchestrator(jurors=jurors, synthesizer=synthesizer, max_rounds=3)

    print("Running case through jury (this will take a while on a 4GB card)...", flush=True)
    transcript = orchestrator.run_case(case_id="pilot_case_001", case_text=CASE_TEXT)

    out_path = "transcript_pilot_case_001.json"
    save_transcript(transcript, out_path)

    print(f"\nDone. Rounds run: {transcript['rounds_run']} "
          f"(converged early: {transcript['converged_early']})")
    print(f"Final verdict: {transcript['synthesis'].get('final_verdict')}")
    print(f"Full transcript written to {out_path}")

    # Quick parse-failure sanity check — worth watching closely at this model size.
    unparsed_count = sum(
        1
        for juror in transcript["jurors"]
        for pos in juror["positions"]
        if "[UNPARSED OUTPUT" in pos["reasoning"]
    )
    if unparsed_count:
        print(f"\nWARNING: {unparsed_count} juror position(s) failed JSON parsing. "
              "Inspect transcript raw_output fields.", file=sys.stderr)


if __name__ == "__main__":
    main()
