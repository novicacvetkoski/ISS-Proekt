"""
Turning a raw ILDC record into juror-facing case materials.

Paragraph numbering is not cosmetic: it is the anchor space for the claims ledger.
Every claim a juror makes cites paragraph ids, which makes grounding checkable by
substring match rather than by an LLM judge, and lets a vote change be attributed to a
specific argument. See docs/SYSTEM_DESIGN.md §4.4.

THE GOLD LABEL NEVER PASSES THROUGH HERE. build_case_file() takes text, not a record,
so there is no path by which `label` can reach a prompt (CLAUDE.md rule 1).
"""

import re

from ..domain.models import CaseFile, Paragraph

# ILDC documents are noisy OCR-ish text with inconsistent breaks. Split on blank lines
# first; fall back to sentence-grouping for documents that have none, which is common.
_BLANK_LINE = re.compile(r"\n\s*\n")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

MIN_PARAGRAPH_WORDS = 15      # merge slivers into the previous paragraph
TARGET_WORDS_PER_PARA = 120   # when falling back to sentence grouping


def split_paragraphs(text: str) -> list[str]:
    chunks = [c.strip() for c in _BLANK_LINE.split(text) if c.strip()]
    if len(chunks) < 3:
        chunks = _group_sentences(text)

    merged: list[str] = []
    carry = ""     # a leading sliver has no previous paragraph, so it merges forward
    for chunk in chunks:
        if carry:
            chunk, carry = f"{carry} {chunk}", ""
        if len(chunk.split()) < MIN_PARAGRAPH_WORDS:
            if merged:
                merged[-1] = f"{merged[-1]} {chunk}"
            else:
                carry = chunk          # ILDC documents often open with a short header line
            continue
        merged.append(chunk)
    if carry:
        if merged:
            merged[-1] = f"{merged[-1]} {carry}"
        else:
            merged.append(carry)   # the whole document is shorter than one paragraph
    return merged


def _group_sentences(text: str) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE_END.split(text.replace("\n", " ")) if s.strip()]
    paragraphs, current, count = [], [], 0
    for sentence in sentences:
        current.append(sentence)
        count += len(sentence.split())
        if count >= TARGET_WORDS_PER_PARA:
            paragraphs.append(" ".join(current))
            current, count = [], 0
    if current:
        paragraphs.append(" ".join(current))
    return paragraphs


def build_case_file(case_id: str, case_text: str) -> CaseFile:
    """The only sanctioned way to build juror-facing materials from stripped text."""
    paragraphs = [
        Paragraph(pid=f"P{i}", text=chunk)
        for i, chunk in enumerate(split_paragraphs(case_text), start=1)
    ]
    if not paragraphs:
        raise ValueError(f"Case {case_id} produced no paragraphs — check the source text.")
    return CaseFile(case_id=case_id, paragraphs=paragraphs)


def approx_tokens(case: CaseFile) -> int:
    """Rough token count (words x 1.3). Enough to predict implicit-cache eligibility."""
    return int(len(case.render().split()) * 1.3)
