"""
Differentiation diagnostics.

Purpose: detect when jurors' reasoning has collapsed into near-identical text —
i.e. the debate protocol produced five copies of the same argument instead of
five independently-reasoned positions, which defeats the point of a multi-agent
jury (see roadmap: isolating the debate mechanism's contribution vs. a
single-agent baseline requires the agents to actually reason differently).

Deliberately stdlib-only, no embedding model — this is a cheap triage signal to
flag transcripts for human review, not a rigorous similarity metric. Jaccard
similarity over stopword-filtered word sets: fast, dependency-free, and good
enough to catch "these five paragraphs are basically the same argument reworded."
"""

import re
from itertools import combinations

# Small stopword list — enough to stop generic connective words (the, and, that)
# from dominating the similarity score, without pulling in an NLP dependency.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "being", "to", "of", "in", "on", "for", "with", "as", "by", "at", "this", "that",
    "these", "those", "it", "its", "not", "no", "which", "who", "whom", "such",
    "does", "do", "did", "has", "have", "had", "would", "could", "should", "must",
    "can", "will", "shall", "under", "without", "beyond", "must", "case", "cases",
}

_WORD_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> set[str]:
    words = _WORD_RE.findall(text.lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 2}


def jaccard_similarity(text_a: str, text_b: str) -> float:
    """0.0 = no shared vocabulary, 1.0 = identical word sets."""
    set_a, set_b = _tokenize(text_a), _tokenize(text_b)
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union else 0.0


def average_pairwise_similarity(texts: list[str]) -> float | None:
    """Mean Jaccard similarity across all pairs. None if fewer than 2 texts."""
    texts = [t for t in texts if t]
    if len(texts) < 2:
        return None
    pairs = list(combinations(texts, 2))
    scores = [jaccard_similarity(a, b) for a, b in pairs]
    return sum(scores) / len(scores)


# Above this average pairwise similarity, jurors' reasoning is flagged as
# suspiciously convergent — worth a human read before trusting the "unanimous"
# result as five independent agreements rather than one argument repeated five
# times.
#
# Calibrated against two real transcripts from this project (not synthetic data):
#   - A run with genuine 4-1 disagreement across distinct persona arguments
#     scored ~0.08 final-round average similarity.
#   - A run that read as near-identical repeated reasoning across all 5 jurors
#     (manually confirmed by inspection) scored ~0.35.
# 0.25 sits between the two with margin on both sides. This is a two-datapoint
# calibration, not a validated cutoff — tighten or loosen it as more real
# transcripts accumulate, ideally by logging this score against your own
# manual "does this look differentiated" judgment on each case for a while.
LOW_DIFFERENTIATION_THRESHOLD = 0.25


def differentiation_report(jurors_log: list[dict]) -> dict:
    """
    jurors_log: the "jurors" list from a run_case() transcript (each juror's
    to_log_dict() output — i.e. transcript["jurors"]).

    Compares round-0 (independent, pre-debate) similarity against final-round
    similarity. A big jump from low round-0 similarity to high final similarity
    suggests the DEBATE process itself is erasing differentiation (a process
    problem — prompt or protocol). High similarity already at round 0 suggests
    the case itself doesn't have much room for disagreement, or the personas
    aren't differentiated enough in the first place (a persona-design problem).
    """
    round0_texts = []
    final_texts = []

    for juror in jurors_log:
        positions = juror.get("positions", [])
        if not positions:
            continue
        round0 = next((p for p in positions if p["round"] == 0), None)
        if round0 and round0.get("reasoning"):
            round0_texts.append(round0["reasoning"])
        final = positions[-1]
        if final.get("reasoning"):
            final_texts.append(final["reasoning"])

    round0_avg = average_pairwise_similarity(round0_texts)
    final_avg = average_pairwise_similarity(final_texts)

    low_differentiation = final_avg is not None and final_avg >= LOW_DIFFERENTIATION_THRESHOLD
    delta = (final_avg - round0_avg) if (round0_avg is not None and final_avg is not None) else None

    if not low_differentiation:
        interpretation = "Final positions show reasonable textual diversity across jurors."
    elif delta is not None and delta > 0.15:
        interpretation = (
            "Jurors started differentiated but converged toward near-identical reasoning "
            "during debate — likely a debate-protocol or prompt issue eroding persona "
            "distinctiveness, not just an easy case."
        )
    else:
        interpretation = (
            "Jurors were already similar at round 0, before any debate — likely reflects "
            "a one-sided case, or personas that aren't differentiated enough to disagree "
            "on this fact pattern. Worth testing against a harder case before concluding "
            "the personas themselves are the problem."
        )

    return {
        "round0_avg_similarity": round(round0_avg, 3) if round0_avg is not None else None,
        "final_avg_similarity": round(final_avg, 3) if final_avg is not None else None,
        "similarity_delta": round(delta, 3) if delta is not None else None,
        "low_differentiation_flag": low_differentiation,
        "threshold": LOW_DIFFERENTIATION_THRESHOLD,
        "interpretation": interpretation,
    }
