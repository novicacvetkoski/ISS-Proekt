"""
Grounding check — are claims actually anchored to the case?

No LLM judge involved: anchors are validated against the real paragraph ids, and quotes
by (normalised) substring match. Cheap, deterministic, and re-runnable from stored
transcripts, which is what makes it usable as a reported metric rather than a vibe.

What it cannot do: tell you whether the cited paragraph SUPPORTS the claim. That needs a
judge model or a human. This catches the coarser failure — citing paragraphs that do not
exist, or quoting text that was never written.
"""

import re

from ..domain.models import CaseFile, RunRecord

_WS = re.compile(r"\s+")


def _normalise(text: str) -> str:
    return _WS.sub(" ", text.lower().strip())


def check_record(record: RunRecord, case: CaseFile, max_quote_words: int = 30) -> dict:
    total = valid_anchor = invalid_anchor = unanchored = 0
    quotes = quote_ok = quote_bad = quote_too_long = 0
    bad_examples: list[str] = []

    normalised_case = _normalise(case.render())
    pids = case.pids

    for position in record.positions:
        for claim in position.claims:
            total += 1
            if not claim.anchor_pids:
                unanchored += 1
            elif all(p in pids for p in claim.anchor_pids):
                valid_anchor += 1
            else:
                invalid_anchor += 1
                missing = [p for p in claim.anchor_pids if p not in pids]
                if len(bad_examples) < 5:
                    bad_examples.append(f"{claim.claim_id}: cites {missing}")

            if claim.quote:
                quotes += 1
                if len(claim.quote.split()) > max_quote_words:
                    quote_too_long += 1
                if _normalise(claim.quote) in normalised_case:
                    quote_ok += 1
                else:
                    quote_bad += 1
                    if len(bad_examples) < 5:
                        bad_examples.append(f"{claim.claim_id}: quote not in case text")

    return {
        "case_id": record.case_id,
        "condition": record.condition,
        "claims": total,
        "valid_anchor": valid_anchor,
        "invalid_anchor": invalid_anchor,
        "unanchored": unanchored,
        "grounding_rate": valid_anchor / total if total else float("nan"),
        "quotes": quotes,
        "quote_verified": quote_ok,
        "quote_unverified": quote_bad,
        "quote_too_long": quote_too_long,
        "examples": bad_examples,
    }


def citation_flags(record: RunRecord) -> list[str]:
    """
    Case-citation shapes in juror text — the prompt forbids naming decided cases because
    models fabricate Indian case law, observed directly in this project's earlier runs.
    This is the detector; the prompt rule is the preventer.
    """
    pattern = re.compile(
        r"[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)*\s+v\.?\s+[A-Z][A-Za-z.]+"
        r"|\(\d{4}\)\s*\d*\s*SCC\s*\d*"
        r"|\bAIR\s*\d{4}\b"
    )
    found: set[str] = set()
    for position in record.positions:
        for text in [position.reasoning, position.change_reasoning or ""] + [c.text for c in position.claims]:
            found.update(m.group(0).strip() for m in pattern.finditer(text))
    return sorted(found)
