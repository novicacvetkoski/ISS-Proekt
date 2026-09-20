"""
Shared fixtures, including a scripted fake backend.

The fake matters more than it looks: it lets the whole deliberation graph — fan-out,
digest, rounds, stopping rule, persuader overlay, clerk — be exercised deterministically
with no API key and no cost. Protocol changes get caught here rather than 40 cases into
a paid run.
"""

import pytest
from pydantic import BaseModel

from jury.config import DeliberationConfig, ExperimentConfig, ModelConfig
from jury.data.case_file import build_case_file
from jury.domain.models import (
    Claim,
    ClerkAgenda,
    ClerkVerdict,
    PrivateAssessment,
    Statement,
    Usage,
)
from jury.domain.verdicts import Verdict

CASE_TEXT = """\
The appellant was convicted under the relevant penal provision for an offence arising out of
an altercation at a village gathering, and sentenced by the trial court.

The prosecution relied principally on the testimony of two eyewitnesses, both related to the
deceased, and on a weapon recovered some days later which the forensic report did not
conclusively connect to the appellant.

The defence argued that the eyewitnesses were interested witnesses whose account was not
corroborated by any independent source, and that the recovery was not proved in accordance
with the requirements of the Evidence Act.

The trial court convicted the appellant. The High Court affirmed the conviction, holding that
the testimony of related witnesses is not for that reason alone unreliable.

The appellant contends that the courts below failed to consider the absence of independent
corroboration, and that the incident occurred without premeditation following provocation.
"""


@pytest.fixture
def case():
    return build_case_file("case_test_001", CASE_TEXT)


class FakeClient:
    """
    Scripted backend. `verdict_script` maps a juror's persona name fragment to the
    sequence of verdicts it will return, so a test can stage a flip, a holdout, or a
    unanimous panel exactly.
    """

    def __init__(self, verdict_script: dict[str, list[Verdict]] | None = None,
                 default: Verdict = Verdict.DISMISS):
        self.model = "fake-model"
        self.verdict_script = verdict_script or {}
        self.default = default
        self.calls: list[dict] = []
        self._counters: dict[str, int] = {}

    def verify_model(self) -> None:
        return None

    @staticmethod
    def _speaker(task: str) -> str:
        """
        Whose turn is this? Read it from the persona card's 'YOU ARE:' line ONLY.

        Matching anywhere in the task would match jurors named in the DIGEST too, so
        every juror would inherit the first scripted juror's verdict — which silently
        turned a 3-2 split into a unanimous panel the first time this was written.
        """
        for line in task.splitlines():
            if line.startswith("YOU ARE:"):
                return line
        return ""

    def _next_verdict(self, task: str) -> Verdict:
        speaker = self._speaker(task)
        for key, sequence in self.verdict_script.items():
            if key in speaker:
                i = self._counters.get(key, 0)
                self._counters[key] = i + 1
                return sequence[min(i, len(sequence) - 1)]
        return self.default

    def generate(self, *, system, prefix, task, schema: type[BaseModel], temperature, seed=None):
        self.calls.append({"system": system, "prefix": prefix, "task": task, "schema": schema.__name__})
        usage = Usage(input_tokens=100, output_tokens=50, cached_tokens=80, model=self.model)

        if schema is PrivateAssessment:
            return (
                PrivateAssessment(
                    story="The altercation occurred and the appellant was convicted.",
                    competing_story="The appellant may not have been the assailant.",
                    disputed_issues=["Is related-witness testimony sufficient without corroboration?"],
                    claims=[
                        Claim(
                            claim_id="c1",
                            text="The eyewitnesses were interested witnesses without corroboration.",
                            anchor_pids=["P2", "P3"],
                            quote=None,
                        )
                    ],
                    verdict=self._next_verdict(task),
                    confidence=0.7,
                    uncertainties="The forensic report is inconclusive.",
                ),
                usage,
            )

        if schema is Statement:
            return (
                Statement(
                    claims=[
                        Claim(
                            claim_id="c1",
                            text="The recovery was not proved as the Evidence Act requires.",
                            anchor_pids=["P3"],
                            quote=None,
                        )
                    ],
                    responses=[],
                    questions=[],
                    verdict=self._next_verdict(task),
                    confidence=0.75,
                    changed_vote=False,
                    moved_by_claim_ids=[],
                    change_reasoning=None,
                ),
                usage,
            )

        if schema is ClerkAgenda:
            return ClerkAgenda(issues=["Was the conviction safe without corroboration?"]), usage

        if schema is ClerkVerdict:
            return (
                ClerkVerdict(rationale="The panel reached its view on corroboration.", dissent_summary=None),
                usage,
            )

        raise AssertionError(f"FakeClient has no script for {schema.__name__}")


@pytest.fixture
def fake_client():
    return FakeClient()


@pytest.fixture
def config():
    return ExperimentConfig(
        name="test",
        pool="debug",
        juror_model=ModelConfig(backend="gemini", model="fake-model", verify_on_start=False),
        clerk_model=ModelConfig(backend="gemini", model="fake-model", verify_on_start=False),
        deliberation=DeliberationConfig(min_rounds=1, max_rounds=2),
    )
