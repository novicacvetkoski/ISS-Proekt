"""
The Clerk — non-voting.

Two jobs: set the agenda after round 0 (the orientation phase of real deliberation),
and write the jury's rationale at the end. It does not hold a position, does not
deliberate, and does not vote.

Why non-voting: a real foreperson votes, but giving one of our five personas procedural
control over the agenda AND a vote would confound persona effects with process power —
and with five jurors it would also change the majority arithmetic. The deviation is
documented in docs/SYSTEM_DESIGN.md §4.2.

The verdict is computed arithmetically from the ballots BEFORE the clerk is called. The
clerk explains the decision; it never makes it. Letting a model decide the verdict from a
transcript would quietly replace the jury with a summariser.
"""

from ..domain.models import ClerkAgenda, ClerkVerdict, Position, PrivateAssessment, Usage
from ..domain.verdicts import Verdict
from ..llm.base import LLMClient

CLERK_SYSTEM = """\
You are the clerk of a five-member deliberative panel considering an appeal to the Supreme
Court of India. You are NOT a juror. You hold no view on the outcome, you do not vote, and
you must not introduce any legal argument of your own.

Do NOT name, cite, or refer to any specific decided case (no case names, years, or
SCC/AIR/SCR numbers), even if a juror's reasoning mentioned one. Restate the underlying
principle in your own words instead.
"""

AGENDA_TASK = """\
TASK — SET THE AGENDA

Below are the issues each juror independently identified as decisive, before they had seen
each other's views.

{issue_blocks}

Merge them into at most five disputed issues for the panel to work through. Combine issues
that are the same question in different words. Keep an issue only one juror raised if it is
genuinely distinct — a point only one juror sees is often the point the deliberation turns on.
State each as a question the panel can answer from the materials.
"""

VERDICT_TASK = """\
TASK — WRITE THE PANEL'S RATIONALE

The panel has finished deliberating. The verdict has already been determined by the jurors'
votes: {verdict} ({breakdown}). Your job is to explain it, not to decide it.

FINAL POSITIONS
{positions}

Write a rationale a legal professional could read on its own, without the transcript, and
understand why the panel landed here:
- Name the arguments that were actually decisive, in your own words. Do not write that
  "the jurors discussed several views".
- If the panel was split, say so plainly and explain how the disagreement stood — do not
  smooth it over.
- Where a juror dissented, state their reasoning in `dissent_summary`. Dissent stays visible.
"""


class Clerk:
    def __init__(self, client: LLMClient, temperature: float = 0.4, seed: int | None = None):
        self.client = client
        self.temperature = temperature
        self.seed = seed

    def build_agenda(
        self, case_prefix: str, assessments: list[tuple[str, PrivateAssessment]]
    ) -> tuple[list[str], Usage]:
        blocks = "\n\n".join(
            f"{juror_id}:\n" + "\n".join(f"  - {issue}" for issue in a.disputed_issues)
            for juror_id, a in assessments
        )
        agenda, usage = self.client.generate(
            system=CLERK_SYSTEM,
            prefix=case_prefix,
            task=AGENDA_TASK.format(issue_blocks=blocks),
            schema=ClerkAgenda,
            temperature=self.temperature,
            seed=self.seed,
        )
        return agenda.issues[:5], usage

    def write_verdict(
        self,
        case_prefix: str,
        verdict: Verdict | None,
        vote_breakdown: dict[str, Verdict],
        final_positions: list[Position],
    ) -> tuple[ClerkVerdict, Usage]:
        breakdown = ", ".join(f"{j}: {v.value}" for j, v in vote_breakdown.items())
        positions = "\n\n".join(
            f"{p.juror_id} — {p.verdict.value.upper()} (confidence {p.confidence:.2f})\n"
            f"  {p.reasoning}"
            for p in final_positions
        )
        result, usage = self.client.generate(
            system=CLERK_SYSTEM,
            prefix=case_prefix,
            task=VERDICT_TASK.format(
                verdict=verdict.value.upper() if verdict else "HUNG — no majority",
                breakdown=breakdown,
                positions=positions,
            ),
            schema=ClerkVerdict,
            temperature=self.temperature,
            seed=self.seed,
        )
        return result, usage
