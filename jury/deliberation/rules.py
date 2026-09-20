"""
Stopping rule and vote-change classification.

Both are pure functions over state so they are unit-testable without a model, and so
the RQ2 measurement can be re-derived from a stored transcript rather than trusting
what was computed at run time.
"""

from ..config import DeliberationConfig
from ..domain.models import Position, VoteChange
from ..domain.verdicts import Verdict, majority_verdict

# A claim id looks like "JTEX-R1-c2" — namespaced by digest.namespaced_claim_id.
# A flip citing something that is not a real claim id (a tally, "the majority",
# an empty list) is normative, not informational.


def should_stop(
    round_num: int,
    positions: list[Position],
    cfg: DeliberationConfig,
) -> tuple[bool, str]:
    """
    Returns (stop, reason).

    Unanimity ends deliberation, but not before `min_rounds` — a panel that happens to
    agree at round 0 still has to test that agreement once, or the control and treatment
    conditions would not be comparable (the persuader needs at least one round to act).

    Stability stop (two consecutive rounds with no vote change) follows the prior work's
    early-stopping rule and keeps cost bounded on cases where nobody is moving.
    """
    if round_num < cfg.min_rounds:
        return False, ""

    latest = _latest_positions(positions)
    verdicts = [p.verdict for p in latest.values()]
    _, unanimous = majority_verdict(verdicts)
    if unanimous:
        return True, "unanimous"

    if round_num >= cfg.max_rounds:
        return True, "max_rounds"

    stable = 0
    for r in range(round_num, 0, -1):
        changes = [p for p in positions if p.round == r and p.changed_vote]
        if changes:
            break
        stable += 1
        if stable >= cfg.stable_rounds_to_stop:
            return True, "stable"

    return False, ""


def _latest_positions(positions: list[Position]) -> dict[str, Position]:
    latest: dict[str, Position] = {}
    for p in positions:
        current = latest.get(p.juror_id)
        if current is None or p.round > current.round:
            latest[p.juror_id] = p
    return latest


def final_votes(positions: list[Position]) -> dict[str, Verdict]:
    return {jid: p.verdict for jid, p in _latest_positions(positions).items()}


def classify_vote_changes(
    positions: list[Position], short_ids: dict[str, str] | None = None
) -> list[VoteChange]:
    """
    Turn the position log into classified flips. This is the RQ2 measurement.

    "informational" — the juror cited at least one well-formed claim id belonging to
                      another juror, i.e. a specific argument moved it.
    "normative"     — it cited nothing, or cited only its own claims, meaning the change
                      came from the room rather than from an argument.

    Deliberately conservative: a flip is only counted as informational when it names an
    argument. Kaplan & Miller's distinction is exactly this, and prior work shows flip
    rate alone conflates conformity with genuine persuasion (arXiv 2606.00820).
    """
    by_juror: dict[str, list[Position]] = {}
    for p in sorted(positions, key=lambda x: (x.juror_id, x.round)):
        by_juror.setdefault(p.juror_id, []).append(p)

    changes: list[VoteChange] = []
    for juror_id, history in by_juror.items():
        for prev, current in zip(history, history[1:]):
            if current.verdict is prev.verdict:
                continue
            own_prefix = f"{short_ids[juror_id]}-" if short_ids and juror_id in short_ids else None
            cited = [c for c in current.moved_by_claim_ids if _is_claim_id(c)]
            # A juror citing only its own earlier claims has not been moved by anyone.
            foreign = [c for c in cited if own_prefix is None or not c.startswith(own_prefix)]
            kind = "informational" if foreign else "normative"
            changes.append(
                VoteChange(
                    juror_id=juror_id,
                    round=current.round,
                    from_verdict=prev.verdict,
                    to_verdict=current.verdict,
                    moved_by_claim_ids=current.moved_by_claim_ids,
                    kind=kind,
                    change_reasoning=current.change_reasoning,
                )
            )
    return changes


def _is_claim_id(value: str) -> bool:
    """Well-formed namespaced claim id: '<SHORT>-R<n>-c<n>'."""
    parts = value.strip().split("-")
    return (
        len(parts) == 3
        and parts[0].startswith("J")
        and parts[1].startswith("R")
        and parts[1][1:].isdigit()
        and parts[2].startswith("c")
    )
