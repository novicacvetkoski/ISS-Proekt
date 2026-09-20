"""
The deliberation digest — the only view a juror gets of the other jurors.

This is the core context-management device. Jurors never see the raw transcript;
they see a HARD-CAPPED summary of each other juror's current position. The cap is
what makes prompt size O(1) in the number of rounds instead of O(rounds x jurors),
and it also forces jurors to make a point compactly rather than restating the case.

Claim ids are namespaced here (JTEX-R1-c2) so that a later vote change can name
exactly which argument moved it. That attribution is what separates informational
from normative influence in RQ2 — see docs/SYSTEM_DESIGN.md §4.4.
"""

from ..config import DeliberationConfig
from ..domain.models import Position


def namespaced_claim_id(juror_short_id: str, round_num: int, claim_id: str) -> str:
    return f"{juror_short_id}-R{round_num}-{claim_id}"


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + " …"


def render_position_summary(position: Position, short_id: str, cfg: DeliberationConfig) -> str:
    """One juror's current position, capped."""
    header = (
        f"{short_id} — {position.juror_id} — votes {position.verdict.value.upper()} "
        f"(confidence {position.confidence:.2f})"
    )
    lines = [header]

    for claim in position.claims[: cfg.max_claims_per_juror_in_digest]:
        cid = namespaced_claim_id(short_id, position.round, claim.claim_id)
        anchors = ",".join(claim.anchor_pids) if claim.anchor_pids else "unanchored"
        lines.append(f"  [{cid} | {anchors}] {_truncate_words(claim.text, cfg.max_words_per_claim_in_digest)}")

    if position.changed_vote:
        moved = ", ".join(position.moved_by_claim_ids) or "no claim cited"
        lines.append(f"  (changed vote this round; cited: {moved})")

    return "\n".join(lines)


def build_digest(
    positions: list[Position],
    short_ids: dict[str, str],
    for_juror_id: str,
    cfg: DeliberationConfig,
) -> str:
    """
    What `for_juror_id` sees of the others: their latest position each, plus any
    questions addressed to them. The juror's own position is excluded — it gets its
    own, fuller block in the task prompt.
    """
    latest: dict[str, Position] = {}
    for p in positions:
        if p.juror_id == for_juror_id:
            continue
        current = latest.get(p.juror_id)
        if current is None or p.round > current.round:
            latest[p.juror_id] = p

    if not latest:
        return "(No other positions yet.)"

    blocks = [
        render_position_summary(p, short_ids.get(p.juror_id, "J??"), cfg)
        for p in sorted(latest.values(), key=lambda x: x.juror_id)
    ]

    # Questions naming this juror. Matched on the juror's display name and short id,
    # which is coarse but cheap; a missed question costs nothing, a false positive
    # just shows the juror one extra line.
    me_short = short_ids.get(for_juror_id, "")
    directed = [
        f"  - {q}"
        for p in latest.values()
        for q in p.questions
        if for_juror_id.lower() in q.lower() or (me_short and me_short in q)
    ]
    if directed:
        blocks.append("QUESTIONS ADDRESSED TO YOU:\n" + "\n".join(directed))

    return "\n\n".join(blocks)


def vote_tally(positions: list[Position]) -> dict[str, int]:
    """Current vote counts from each juror's latest position."""
    latest: dict[str, Position] = {}
    for p in positions:
        current = latest.get(p.juror_id)
        if current is None or p.round > current.round:
            latest[p.juror_id] = p
    tally: dict[str, int] = {}
    for p in latest.values():
        tally[p.verdict.value] = tally.get(p.verdict.value, 0) + 1
    return tally
