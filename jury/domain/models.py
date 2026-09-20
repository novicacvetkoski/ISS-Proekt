"""
Schemas for everything that crosses a model boundary or lands in a run artifact.

TWO CLASSES OF MODEL LIVE HERE, and the distinction matters:

1. RESPONSE schemas (PrivateAssessment, Statement, ClerkAgenda, ClerkVerdict) are
   handed to the provider as a structured-output schema. Gemini REJECTS schemas whose
   fields carry default values (googleapis/python-genai#699), so every field in these
   is REQUIRED. Optionality is expressed as `X | None`, never as a default.

2. RECORD schemas (CaseFile, Ballot, VoteChange, JuryVerdict, RunRecord) are ours alone
   and never sent as a schema, so they may use defaults freely.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from .verdicts import Verdict

# --------------------------------------------------------------------------- #
# Case materials
# --------------------------------------------------------------------------- #


class Paragraph(BaseModel):
    """One numbered paragraph of case text. `pid` is what claims anchor to."""
    pid: str          # "P17"
    text: str


class CaseFile(BaseModel):
    """
    Juror-facing case materials. Never contains the gold label — construction goes
    through data.case_file.build_case_file(), which only ever sees stripped text.
    """
    case_id: str
    paragraphs: list[Paragraph]

    def render(self) -> str:
        return "\n\n".join(f"[{p.pid}] {p.text}" for p in self.paragraphs)

    @property
    def pids(self) -> set[str]:
        return {p.pid for p in self.paragraphs}

    def text_of(self, pid: str) -> str | None:
        for p in self.paragraphs:
            if p.pid == pid:
                return p.text
        return None


# --------------------------------------------------------------------------- #
# Response schemas — NO DEFAULTS BELOW THIS LINE (see module docstring)
# --------------------------------------------------------------------------- #


class Claim(BaseModel):
    """
    One argumentative move, anchored to the case text.

    The anchor is what makes grounding checkable without an LLM judge, and what
    lets a later vote change be attributed to a specific argument (informational)
    rather than to the mood of the room (normative). It is required for that reason.
    """
    claim_id: str                 # assigned by the juror as "c1", "c2"; namespaced downstream
    text: str = Field(description="The claim, in one or two sentences")
    anchor_pids: list[str] = Field(description="Case paragraph ids this claim relies on, e.g. ['P17']")
    quote: str | None = Field(description="Optional verbatim quote of <=30 words from an anchored paragraph")


class PrivateAssessment(BaseModel):
    """Round 0: independent analysis, before any exposure to another juror."""
    story: str = Field(description="Your causal account of what happened, from the materials")
    competing_story: str = Field(description="The strongest account that points the other way")
    disputed_issues: list[str] = Field(description="The 2-5 questions this appeal actually turns on")
    claims: list[Claim] = Field(description="Your supporting claims, each anchored to case paragraphs")
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainties: str = Field(description="What would change your mind, and what the materials do not settle")


class ClaimResponse(BaseModel):
    """An explicit engagement with another juror's claim."""
    target_claim_id: str = Field(description="The claim id you are answering, e.g. 'JB-R1-c2'")
    stance: str = Field(description="One of: accept, reject, qualify")
    reasoning: str


class Statement(BaseModel):
    """One juror's contribution to one deliberation round."""
    claims: list[Claim] = Field(description="New or restated claims, anchored")
    responses: list[ClaimResponse] = Field(description="Direct engagement with other jurors' claims")
    questions: list[str] = Field(description="Questions to other jurors, each naming the juror")
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    changed_vote: bool
    moved_by_claim_ids: list[str] = Field(
        description="If you changed your vote, the claim ids that moved you. Empty if unchanged. "
                    "A majority forming against you is NOT a reason and must not be listed here."
    )
    change_reasoning: str | None = Field(description="Why you changed, or null if you did not")


class ClerkAgenda(BaseModel):
    """Orientation phase: the disputed issues the panel will work through."""
    issues: list[str] = Field(description="At most 5 disputed issues, merged across jurors")


class ClerkVerdict(BaseModel):
    """Final written product of the jury."""
    rationale: str = Field(description="Why the jury reached this verdict, naming the decisive arguments")
    dissent_summary: str | None = Field(description="Dissenting reasoning, or null if unanimous")


# --------------------------------------------------------------------------- #
# Record schemas — defaults allowed, never sent as a provider schema
# --------------------------------------------------------------------------- #


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    model: str = ""


class Ballot(BaseModel):
    juror_id: str
    round: int
    verdict: Verdict
    confidence: float


class Position(BaseModel):
    """A juror's state at one round, as persisted."""
    juror_id: str
    round: int
    verdict: Verdict
    confidence: float
    claims: list[Claim] = Field(default_factory=list)
    responses: list[ClaimResponse] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    changed_vote: bool = False
    moved_by_claim_ids: list[str] = Field(default_factory=list)
    change_reasoning: str | None = None
    reasoning: str = ""          # story (round 0) or rendered claims (later rounds)
    is_persuader: bool = False
    usage: Usage = Field(default_factory=Usage)
    timestamp: str = Field(default_factory=_now)


class VoteChange(BaseModel):
    """
    One flip, classified. `kind` is the RQ2 measurement:
      "informational" — cited at least one real, case-grounded claim by another juror
      "normative"     — cited nothing, or cited only a tally/consensus
    """
    juror_id: str
    round: int
    from_verdict: Verdict
    to_verdict: Verdict
    moved_by_claim_ids: list[str] = Field(default_factory=list)
    kind: str = "unclassified"
    change_reasoning: str | None = None


class PersuaderAssignment(BaseModel):
    juror_id: str
    target_verdict: Verdict
    selected_because: str        # "round0_minority" | "seeded_random"
    agreed_with_target_at_round0: bool


class JuryVerdict(BaseModel):
    verdict: Verdict | None      # None = hung (even split; only possible if a juror is excluded)
    unanimous: bool
    vote_breakdown: dict[str, Verdict]
    rationale: str
    dissent_summary: str | None = None
    non_persuader_verdict: Verdict | None = None   # verdict among the 4 non-persuader jurors


class RunRecord(BaseModel):
    """
    One (case, condition) result. This is the unit of the experiment and the unit
    of resumability: written atomically, skipped by the runner if present.
    """
    case_id: str
    condition: str               # "control" | "treatment" | "single_agent" | "noise"
    experiment: str
    verdict: JuryVerdict
    gold_label: int | None       # filled in by ANALYSIS, never present during the run
    positions: list[Position]
    ballots: list[Ballot]
    vote_changes: list[VoteChange] = Field(default_factory=list)
    persuader: PersuaderAssignment | None = None
    agenda: list[str] = Field(default_factory=list)
    rounds_run: int = 0
    stopped_because: str = ""
    provenance: dict = Field(default_factory=dict)   # config hash, model id, seeds, prompt version
    usage_total: Usage = Field(default_factory=Usage)
    timestamp: str = Field(default_factory=_now)
