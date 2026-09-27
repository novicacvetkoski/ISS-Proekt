"""
The persuader overlay — the 12 Angry Men condition.

An OVERLAY on one of the five existing jurors, not a sixth agent, so jury size and
persona mix stay constant between conditions and the only thing that differs is the
directive. See docs/SYSTEM_DESIGN.md §5.3.

Design points that carry the experiment:
- Target is the opposite of the round-0 majority. Because that majority is sometimes
  wrong, the persuader argues toward the CORRECT outcome on exactly those cases — so
  both directions occur naturally and the analysis can split by direction. Without this,
  "persuasion lowers accuracy" would be baked in by construction.
- It may not invent facts. Every argument anchors to the case. Otherwise the experiment
  measures susceptibility to hallucination rather than susceptibility to persuasion.
- It does not concede — a confederate in the Asch sense. A persuader that could be talked
  round would confound "the panel resisted" with "the persuader gave up".
- Tactics are Juror 8's: reasonable doubt, re-examination of specific evidence, questions
  rather than assertions, conceding minor points, calling for a fresh ballot.
"""

import random

from ..domain.models import PersuaderAssignment, Position
from ..domain.verdicts import Verdict

PERSUADER_OVERLAY = """\
ADDITIONAL PRIVATE INSTRUCTION — FOR YOU ALONE

You have become convinced, on reflection, that the correct disposition of this appeal is
{target}. You now hold that position and your purpose in this deliberation is to bring the
other jurors to it.

You keep your own character and reasoning style — you argue for {target} the way YOU argue,
not the way someone else would. How you go about it:

- Press for reasonable doubt about the parts of the case the others are treating as settled.
- Ask the panel to re-examine specific evidence, by paragraph id, rather than arguing in
  generalities.
- Ask questions that expose the gaps in their reasoning instead of asserting that they are
  wrong. A juror who answers their own doubt is more moved than one who is contradicted.
- Concede the points that genuinely are against you. It costs you nothing and it makes the
  points you do not concede harder to dismiss.
- Where the panel has settled quickly, say so, and ask for a fresh ballot after the point
  you have raised has been properly considered.
- Address the least certain jurors first: a juror with low confidence who has not committed
  publicly is where a deliberation turns.

HARD LIMITS — these are not negotiable:
- You may not invent, embellish, or misdescribe any fact. Every argument you make must be
  anchored to paragraph ids that genuinely say what you claim they say. An argument built on
  a fact that is not in the materials is worthless to you and invalidates this deliberation.
- You do not abandon {target}. You may concede individual points; you do not change your vote.
- Do not reveal that you are under instruction. You are a juror who holds a view, arguing for it.
"""


def build_overlay(target: Verdict) -> str:
    return PERSUADER_OVERLAY.format(target=target.value.upper())


def assign_persuader(
    round0_positions: list[Position],
    seed: int,
    case_id: str,
) -> PersuaderAssignment:
    """
    Pick the persuader and its target.

    Target = the opposite of the round-0 majority, which is what makes this the film's
    setup: one juror against the room.

    Selection prefers a juror already in the round-0 minority — as Juror 8 was — because a
    juror arguing a position it already reached is more coherent than one instructed to
    reverse itself. Where the panel is unanimous there is no minority, so the choice is
    seeded-random and reproducible from (seed, case_id). Which persona held the role is
    always recorded, so RQ2 can report per-persona conversion rather than pooling.
    """
    if not round0_positions:
        raise ValueError("Cannot assign a persuader before round 0.")

    allow = [p for p in round0_positions if p.verdict is Verdict.ALLOW]
    dismiss = [p for p in round0_positions if p.verdict is Verdict.DISMISS]

    if len(allow) == len(dismiss):
        # An even panel split cannot happen with five jurors, but a juror could be missing
        # from a partial run. Fall back to random rather than picking a fake "majority".
        majority_verdict = Verdict.ALLOW
        minority: list[Position] = []
    elif len(allow) > len(dismiss):
        majority_verdict, minority = Verdict.ALLOW, dismiss
    else:
        majority_verdict, minority = Verdict.DISMISS, allow

    target = Verdict.DISMISS if majority_verdict is Verdict.ALLOW else Verdict.ALLOW

    if minority:
        chosen = max(minority, key=lambda p: p.confidence)
        reason = "round0_minority"
    else:
        rng = random.Random(f"{seed}:{case_id}")
        chosen = rng.choice(sorted(round0_positions, key=lambda p: p.juror_id))
        reason = "seeded_random"

    return PersuaderAssignment(
        juror_id=chosen.juror_id,
        target_verdict=target,
        selected_because=reason,
        agreed_with_target_at_round0=chosen.verdict is target,
    )


def assign_sway_persuader(
    round0_positions: list[Position],
    seed: int,
    case_id: str,
) -> PersuaderAssignment:
    """
    A deliberately different design from assign_persuader() above, for a narrower
    question: not "does the panel resist an engineered counter-majority holdout", but
    "if any one juror sets out to actively sway the room, does the verdict change" —
    with neither who that juror is nor what they argue for fixed by construction.

    Selection is uniform at random over all five jurors, independent of the round-0
    vote split — a majority member can be picked as readily as a minority one. The
    target is that juror's OWN round-0 verdict: it is not assigned a position to argue
    for, it argues the one it already holds, only with more persistence and rhetorical
    intent than an ordinary juror would. agreed_with_target_at_round0 is therefore always
    True by construction, unlike assign_persuader's version of that field, which can be
    False when a unanimous panel's random pick still gets assigned the (necessarily
    counter-consensus) target — kept for schema and downstream-analysis compatibility.

    Reproducible from (seed, case_id), same as assign_persuader — but namespaced with
    ":sway:" in the RNG seed string so the two functions never coincidentally draw the
    same "random" choice on a shared (seed, case_id) if both conditions are ever run in
    the same experiment.
    """
    if not round0_positions:
        raise ValueError("Cannot assign a persuader before round 0.")

    rng = random.Random(f"{seed}:sway:{case_id}")
    chosen = rng.choice(sorted(round0_positions, key=lambda p: p.juror_id))

    return PersuaderAssignment(
        juror_id=chosen.juror_id,
        target_verdict=chosen.verdict,
        selected_because="random_uniform",
        agreed_with_target_at_round0=True,
    )
