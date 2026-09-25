"""
DebateOrchestrator: runs one case through the full jury protocol.

Sequence (per Phase 2 architecture doc):
  1. Independent initial analysis — every juror sees the case alone, no cross-talk.
  2. Debate rounds — each juror sees a summary of the others' current positions and
     may restate or revise, up to max_rounds or until convergence.
  3. Synthesis — the Synthesizer aggregates final positions into one verdict + rationale.

The full structured transcript (every juror, every round, raw + parsed) is retained
as a single JSON-serializable object — this is both the debugging artifact for Phase 4
and the evaluation artifact for Phase 5 (consistency re-runs, explainability review,
anti-collusion audit of raw text).
"""

import json
from datetime import datetime, timezone

from jurors.base import JurorAgent, Position
from jurors.synthesizer import Synthesizer
from jurors.analysis import differentiation_report


class DebateOrchestrator:
    def __init__(
        self,
        jurors: list[JurorAgent],
        synthesizer: Synthesizer,
        max_rounds: int = 3,
    ):
        self.jurors = jurors
        self.synthesizer = synthesizer
        self.max_rounds = max_rounds

    def run_case(self, case_id: str, case_text: str) -> dict:
        started_at = datetime.now(timezone.utc).isoformat()

        # --- Stage 1: independent initial analysis ---------------------
        for juror in self.jurors:
            juror.initial_analysis(case_text)

        # --- Stage 2: debate rounds -------------------------------------
        rounds_run = 0
        for round_num in range(1, self.max_rounds + 1):
            rounds_run = round_num
            summary = self._summarize_positions()
            for juror in self.jurors:
                # Exclude the responding juror's own last position from what it's shown,
                # so it's reacting to others, not restating itself.
                others_summary = self._summarize_positions(exclude=juror.name)
                juror.respond_to_debate(round_num, others_summary)

            if self._has_converged():
                break

        # --- Stage 3: synthesis ------------------------------------------
        final_positions = [
            {
                "juror": juror.name,
                "verdict": juror.position_log[-1].normalized_verdict or juror.position_log[-1].verdict,
                "confidence": juror.position_log[-1].confidence,
                "reasoning": juror.position_log[-1].reasoning,
            }
            for juror in self.jurors
        ]
        synthesis = self.synthesizer.synthesize(case_text, final_positions)

        finished_at = datetime.now(timezone.utc).isoformat()

        # Aggregate every citation-shaped string flagged across every juror/round —
        # a non-empty list here means fabricated precedent likely leaked into the
        # jury's reasoning and should be reviewed before trusting this case's verdict.
        all_flagged_citations = sorted(set(
            citation
            for juror in self.jurors
            for pos in juror.position_log
            for citation in pos.flagged_citations
        ))

        jurors_log = [juror.to_log_dict() for juror in self.jurors]
        differentiation = differentiation_report(jurors_log)

        return {
            "case_id": case_id,
            "started_at": started_at,
            "finished_at": finished_at,
            "rounds_run": rounds_run,
            "converged_early": rounds_run < self.max_rounds,
            "flagged_citations": all_flagged_citations,
            "differentiation": differentiation,
            "jurors": jurors_log,
            "synthesis": synthesis,
        }

    # ---------------------------------------------------------------- #
    # Helpers
    # ---------------------------------------------------------------- #

    def _summarize_positions(self, exclude: str | None = None) -> str:
        """
        Compact per-juror summary for the shared debate context — deliberately NOT
        the full raw output of each juror, to keep context length manageable on a
        4GB-VRAM local model. Each juror's full raw text still lives in its own
        position_log / history for the transcript.
        """
        lines = []
        for juror in self.jurors:
            if juror.name == exclude or not juror.position_log:
                continue
            last: Position = juror.position_log[-1]
            lines.append(
                f"- {juror.name}: verdict = \"{last.normalized_verdict or last.verdict}\" "
                f"(confidence {last.confidence}). Key reasoning: {self._truncate(last.reasoning)}"
            )
        return "\n".join(lines)

    @staticmethod
    def _truncate(text: str, max_chars: int = 400) -> str:
        text = text or ""
        return text if len(text) <= max_chars else text[:max_chars].rstrip() + "..."

    def _has_converged(self) -> bool:
        """All jurors currently hold the same normalized (non-null, non-'unclear') verdict."""
        verdicts = [j.position_log[-1].normalized_verdict for j in self.jurors if j.position_log]
        if not verdicts or any(v is None or v == "unclear" for v in verdicts):
            return False
        return len(set(verdicts)) == 1


def save_transcript(transcript: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=2, ensure_ascii=False)
