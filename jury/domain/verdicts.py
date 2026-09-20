"""
Verdict vocabulary.

Binary, and deliberately so: ILDC gold labels are binary (the appellant's claim is
"accepted" or "rejected"), and ILDC marks a case accepted if even ONE of several
appeals succeeds. The prototype's third option ("modify") had no gold counterpart
and would have collapsed to ALLOW anyway, producing predictions that cannot be scored.
"""

from enum import Enum


class Verdict(str, Enum):
    ALLOW = "allow"      # appeal/petition accepted — appellant wins — ILDC label 1
    DISMISS = "dismiss"  # appeal rejected — decision below stands — ILDC label 0


LABEL_TO_VERDICT = {1: Verdict.ALLOW, 0: Verdict.DISMISS}
VERDICT_TO_LABEL = {Verdict.ALLOW: 1, Verdict.DISMISS: 0}

# Free-text fallback. Structured outputs make this nearly dead code, but a model
# can still emit a schema-valid string that isn't one of ours when the backend
# doesn't enforce enums (older Ollama builds). Keep it as a safety net, and log
# whenever it fires rather than silently coercing.
_ALLOW_KEYWORDS = ("allow", "accept", "quash", "set aside", "acquit", "reverse", "overturn", "succeed")
_DISMISS_KEYWORDS = ("dismiss", "reject", "uphold", "affirm", "conviction stands", "decline")


def normalize_verdict(raw: str | None) -> Verdict | None:
    """Map a free-text verdict onto the controlled vocabulary. None if unmappable."""
    if not raw:
        return None
    lowered = raw.strip().lower()
    for verdict in Verdict:
        if lowered == verdict.value:
            return verdict
    for kw in _ALLOW_KEYWORDS:
        if kw in lowered:
            return Verdict.ALLOW
    for kw in _DISMISS_KEYWORDS:
        if kw in lowered:
            return Verdict.DISMISS
    return None


def majority_verdict(votes: list[Verdict]) -> tuple[Verdict | None, bool]:
    """
    Returns (verdict, unanimous). With an odd panel there are no ties; an even
    split returns (None, False) so callers must handle a hung panel explicitly
    rather than silently inheriting a default.
    """
    if not votes:
        return None, False
    allow = sum(1 for v in votes if v is Verdict.ALLOW)
    dismiss = len(votes) - allow
    if allow == dismiss:
        return None, False
    return (Verdict.ALLOW if allow > dismiss else Verdict.DISMISS), (allow == 0 or dismiss == 0)
