"""
The Juror.

STATELESS BY DESIGN. A juror holds no conversation history; both operations are pure
functions of (profile, case, context). The prototype's JurorAgent accumulated a full
message history per juror, so the case text was re-sent every round and cost grew with
round count. Here the prompt is rebuilt from graph state each call, so prompt size is
O(1) in the number of rounds, and any single call can be replayed in isolation.

Responsibilities R1-R8 are enumerated in docs/SYSTEM_DESIGN.md §2.1.
"""

from ..config import DeliberationConfig
from ..domain.models import (
    CaseFile,
    Claim,
    Position,
    PrivateAssessment,
    Statement,
    Usage,
)
from ..llm.base import LLMClient
from ..personas.registry import JurorProfile
from ..personas.render import render_persona_card
from .prompts.instructions import COURT_INSTRUCTIONS

REVIEW_TASK = """\
TASK — INDEPENDENT REVIEW (round 0)

You have not seen any other juror's view, and you will not until after you commit to this
one. Work through the materials yourself:

1. Build your account of what happened and what is legally at stake (`story`).
2. State the strongest account pointing the other way (`competing_story`) — genuinely,
   not as a straw man.
3. Name the 2-5 questions this appeal actually turns on (`disputed_issues`).
4. Give the claims that support your view (`claims`), each anchored to case paragraph ids.
5. Commit to a verdict and a confidence you would defend.
6. Say what the materials do not settle, and what would change your mind (`uncertainties`).

This is your honest starting position. It is recorded privately.
"""

DELIBERATE_TASK = """\
TASK — DELIBERATION ROUND {round_num}

YOUR CURRENT POSITION
{own_position}

THE OTHER JURORS' CURRENT POSITIONS
{digest}

THE PANEL IS WORKING THROUGH THESE ISSUES
{agenda}

Now take your turn:
- Engage directly with other jurors' claims by id in `responses` — accept, reject, or
  qualify, with reasoning. Address the strongest argument against you, not the weakest.
- Add or sharpen your own claims, anchored to case paragraph ids.
- Ask other jurors questions by name where their reasoning has a gap.
- State your verdict and confidence.

ON CHANGING YOUR VOTE
If you change it, set `changed_vote` true and list in `moved_by_claim_ids` the specific
claim ids that moved you. If no particular argument moved you, you do not have a reason to
change. That {tally} is the current split is NOT a reason, and listing it as one is a
failure of your duty as a juror. If you hold your position, say why the arguments against
it do not persuade you.
"""


class Juror:
    def __init__(
        self,
        profile: JurorProfile,
        client: LLMClient,
        cfg: DeliberationConfig,
        temperature: float,
        seed: int | None = None,
    ):
        self.profile = profile
        self.client = client
        self.cfg = cfg
        self.temperature = temperature
        self.seed = seed

    # ------------------------------------------------------------------ #

    @property
    def juror_id(self) -> str:
        return self.profile.name

    @property
    def short_id(self) -> str:
        return self.profile.short_id

    def _system(self) -> str:
        return COURT_INSTRUCTIONS

    def _prefix(self, case: CaseFile) -> str:
        """
        The cacheable head: identical across all five jurors for a given case, so the
        implicit cache can hit. The persona card follows it; see SYSTEM_DESIGN §4.3.
        """
        return f"CASE MATERIALS — {case.case_id}\n\n{case.render()}"

    def _persona_block(self, overlay: str | None = None) -> str:
        card = render_persona_card(self.profile)
        return f"{card}\n\n{overlay}" if overlay else card

    # ------------------------------------------------------------------ #

    def review_case(self, case: CaseFile) -> tuple[PrivateAssessment, Usage]:
        """Round 0 — R1, R3. No exposure to any other juror."""
        task = f"{self._persona_block()}\n\n{REVIEW_TASK}"
        assessment, usage = self.client.generate(
            system=self._system(),
            prefix=self._prefix(case),
            task=task,
            schema=PrivateAssessment,
            temperature=self.temperature,
            seed=self.seed,
        )
        return assessment, usage

    def deliberate(
        self,
        case: CaseFile,
        round_num: int,
        own_position: Position,
        digest: str,
        agenda: list[str],
        tally: dict[str, int],
        overlay: str | None = None,
    ) -> tuple[Statement, Usage]:
        """One deliberation round — R4, R5, R6. `overlay` carries the persuader directive."""
        task = DELIBERATE_TASK.format(
            round_num=round_num,
            own_position=_render_own_position(own_position),
            digest=digest,
            agenda="\n".join(f"  {i}. {issue}" for i, issue in enumerate(agenda, 1)) or "  (none recorded)",
            tally=", ".join(f"{k}: {v}" for k, v in sorted(tally.items())) or "unknown",
        )
        statement, usage = self.client.generate(
            system=self._system(),
            prefix=self._prefix(case),
            task=f"{self._persona_block(overlay)}\n\n{task}",
            schema=Statement,
            temperature=self.temperature,
            seed=self.seed,
        )
        return statement, usage


def _render_own_position(position: Position) -> str:
    lines = [
        f"You currently vote {position.verdict.value.upper()} (confidence {position.confidence:.2f}).",
    ]
    if position.reasoning:
        lines.append(f"Your reasoning: {position.reasoning}")
    if position.claims:
        lines.append("Your claims so far:")
        lines += [f"  [{c.claim_id} | {','.join(c.anchor_pids)}] {c.text}" for c in position.claims]
    return "\n".join(lines)


def position_from_assessment(
    profile: JurorProfile, assessment: PrivateAssessment, usage: Usage
) -> Position:
    return Position(
        juror_id=profile.name,
        round=0,
        verdict=assessment.verdict,
        confidence=assessment.confidence,
        claims=assessment.claims,
        reasoning=assessment.story,
        usage=usage,
    )


def position_from_statement(
    profile: JurorProfile,
    statement: Statement,
    round_num: int,
    usage: Usage,
    is_persuader: bool = False,
) -> Position:
    return Position(
        juror_id=profile.name,
        round=round_num,
        verdict=statement.verdict,
        confidence=statement.confidence,
        claims=statement.claims,
        responses=statement.responses,
        questions=statement.questions,
        changed_vote=statement.changed_vote,
        moved_by_claim_ids=statement.moved_by_claim_ids,
        change_reasoning=statement.change_reasoning,
        reasoning=_render_claims(statement.claims),
        is_persuader=is_persuader,
        usage=usage,
    )


def _render_claims(claims: list[Claim]) -> str:
    return " ".join(c.text for c in claims)
