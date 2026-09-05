"""
Base class for AI jury agents.

Design notes (per project roadmap, Phase 2/4):
- Each juror keeps its own independent message history, so its transcript
  is traceable in isolation for the explainability evaluation (Phase 5).
- Position changes must be logged with an explicit reason, never silently
  overwritten — this is the anti-collusion / groupthink safeguard informed
  by Motwani et al.'s findings on steganographic coordination between agents.
- Output parsing uses a fenced-JSON contract rather than regex over free
  text, since Qwen3-4B follows explicit structured-output instructions far
  more reliably than it can be reverse-parsed after the fact.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import ollama

DEFAULT_MODEL = "qwen3:4b"
DEFAULT_NUM_CTX = 3072  # conservative for 4GB VRAM — see project notes


# Controlled vocabulary for verdicts. Jurors are instructed to use one of these
# exactly; normalize_verdict() is a safety-net fallback for when they don't, since
# pilot runs showed free-text verdicts ("Quash conviction", "Conviction set aside",
# "uphold") which broke exact-string convergence checks and vote tallying.
CONTROLLED_VERDICTS = ["uphold", "quash", "modify", "unclear"]

_VERDICT_KEYWORDS = {
    "uphold": ["uphold", "affirm", "conviction stands", "dismiss the appeal", "conviction stand"],
    "quash": ["quash", "set aside", "acquit", "reverse", "overturn"],
    "modify": ["modify", "remand", "reduce", "partial", "downgrade"],
}

# Citation-shaped strings ("X v. Y", "(2005)", "SCC", "AIR ...") are a proxy for
# fabricated case law — Qwen3-4B at this size reliably invents plausible-looking
# but non-existent citations. This regex is a detection net, not a preventer;
# the actual prevention lives in the "no named citations" prompt rule below.
_CITATION_PATTERN = re.compile(
    r"[A-Z][A-Za-z.]+(?:\s+[A-Z][A-Za-z.]+)*\s+v\.?\s+[A-Z][A-Za-z.]+(?:\s+[A-Za-z.]+)*"
    r"|\(\d{4}\)\s*\d*\s*SCC\s*\d*"
    r"|\bAIR\s*\d{4}\b"
)


def normalize_verdict(raw_verdict: str | None) -> str | None:
    """Map free-text verdict onto CONTROLLED_VERDICTS via keyword match. Returns
    None if raw_verdict is None; returns 'unclear' if no keyword matches."""
    if raw_verdict is None:
        return None
    lowered = raw_verdict.lower()
    for canonical, keywords in _VERDICT_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return canonical
    return "unclear"


def find_citation_flags(text: str) -> list[str]:
    """Return distinct citation-shaped substrings found in text, for audit review."""
    if not text:
        return []
    return sorted(set(m.group(0).strip() for m in _CITATION_PATTERN.finditer(text)))


@dataclass
class Position:
    """A single structured position taken by a juror at a point in the debate."""
    round: int  # 0 = independent initial analysis, 1..N = debate rounds
    verdict: str | None            # raw verdict string as the model wrote it
    normalized_verdict: str | None  # mapped onto CONTROLLED_VERDICTS
    confidence: float | None
    reasoning: str
    changed_from_previous: bool
    changed_reason: str | None
    raw_output: str
    flagged_citations: list[str] = field(default_factory=list)
    format_retries: int = 0  # how many reformat attempts were needed before parsing (or giving up)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class JurorAgent:
    """
    Base class for a single juror in the debate.

    Subclasses are expected to override `persona_prompt` (a class attribute
    or property) with their persona's system prompt. Everything else —
    calling the model, tracking history, parsing structured output, logging
    position changes — is shared.
    """

    name: str = "Juror"
    persona_prompt: str = ""  # overridden by subclasses

    # Number of reformat attempts if the model's response doesn't parse as valid
    # JSON. Deliberately small — the retry prompt is a narrow, easy ask ("just
    # reformat this as JSON"), so if it's still failing after this many attempts
    # the problem is unlikely to be fixed by trying again.
    DEFAULT_FORMAT_RETRIES = 2

    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = DEFAULT_NUM_CTX):
        if not self.persona_prompt:
            raise ValueError(f"{self.__class__.__name__} must define persona_prompt")
        self.model = model
        self.num_ctx = num_ctx
        self.history: list[dict] = []       # full raw conversation, this juror only
        self.position_log: list[Position] = []  # structured, parsed positions

    # ------------------------------------------------------------------ #
    # Model call
    # ------------------------------------------------------------------ #

    def _call_model(self, user_content: str) -> str:
        self.history.append({"role": "user", "content": user_content})
        response = ollama.chat(
            model=self.model,
            messages=[{"role": "system", "content": self._full_system_prompt()}] + self.history,
            options={"temperature": 0.7, "num_ctx": self.num_ctx},
        )
        output = response["message"]["content"]
        self.history.append({"role": "assistant", "content": output})
        return output

    def _call_model_with_format_retry(
        self, user_content: str, max_retries: int | None = None
    ) -> tuple[str, int]:
        """
        Calls the model, and if the response doesn't contain valid fenced JSON,
        re-prompts (in the same conversation, so the juror sees its own failed
        attempt) with a narrow "reformat only" instruction, up to max_retries
        times. Returns (raw_output, retries_used) — raw_output is the LAST
        attempt regardless of whether it eventually parsed, so a persistent
        failure is still visible downstream via the existing
        [UNPARSED OUTPUT] flag in _parse_position rather than silently retried
        forever or masked.
        """
        max_retries = self.DEFAULT_FORMAT_RETRIES if max_retries is None else max_retries
        raw = self._call_model(user_content)
        retries_used = 0

        while self._extract_json(raw) is None and retries_used < max_retries:
            retries_used += 1
            reformat_prompt = (
                "Your previous response was not valid JSON. Do not change your "
                "reasoning or verdict — reformat EXACTLY the same position into "
                "a single fenced ```json``` block matching the required schema, "
                "with nothing before or after it."
            )
            raw = self._call_model(reformat_prompt)

        return raw, retries_used

    def _full_system_prompt(self) -> str:
        """Persona prompt plus the shared structured-output and conduct rules."""
        return f"{self.persona_prompt}\n\n{SHARED_RULES}"

    # ------------------------------------------------------------------ #
    # Public debate protocol methods
    # ------------------------------------------------------------------ #

    def initial_analysis(self, case_text: str) -> Position:
        """Independent analysis, before any cross-juror communication."""
        prompt = (
            f"CASE MATERIALS:\n{case_text}\n\n"
            "Give your independent initial verdict and full reasoning. "
            "You have not seen any other juror's position.\n\n"
            "IMPORTANT: Output ONLY the JSON block below. Do not write a judgment "
            "document, a case caption, headers, or any text before or after the "
            "JSON. Your entire response must be a single fenced ```json``` block."
        )
        raw, retries_used = self._call_model_with_format_retry(prompt)
        position = self._parse_position(raw, round_num=0, previous=None)
        position.format_retries = retries_used
        self.position_log.append(position)
        return position

    def respond_to_debate(self, round_num: int, other_positions_summary: str) -> Position:
        """One debate round: react to a summary of other jurors' current positions."""
        prompt = (
            f"DEBATE ROUND {round_num} — other jurors' current positions:\n"
            f"{other_positions_summary}\n\n"
            "Restate or revise your position. If you change it, you MUST name "
            "exactly which juror's argument moved you and why, in changed_reason. "
            "Do not change your verdict merely because a majority holds a "
            "different view — deference without a distinct new argument is invalid.\n\n"
            "Your reasoning must stay identifiably grounded in YOUR persona's distinct "
            "reasoning style, not converge on a generic summary shared with other jurors. "
            "If you agree with the outcome others have reached, say so, but justify it "
            "through your own persona's lens (e.g. the textualist through statutory "
            "language, the equity advocate through fairness/hardship) rather than "
            "repeating their argument in your own words.\n\n"
            "IMPORTANT: Output ONLY the JSON block below. Do not write bare values, "
            "a list, or any text before or after the JSON. Your entire response must "
            "be a single fenced ```json``` block, structured exactly like the schema "
            "you were given."
        )
        raw, retries_used = self._call_model_with_format_retry(prompt)
        previous = self.position_log[-1] if self.position_log else None
        position = self._parse_position(raw, round_num=round_num, previous=previous)
        position.format_retries = retries_used
        self.position_log.append(position)
        return position

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def _parse_position(self, raw: str, round_num: int, previous: Position | None) -> Position:
        parsed = self._extract_json(raw)

        verdict = parsed.get("verdict") if parsed else None
        confidence = parsed.get("confidence") if parsed else None
        # Only fall back to the full raw text when "reasoning" is truly absent —
        # not when parsed successfully but is simply missing that key, which
        # pilot runs showed produces a "reasoning" field containing the raw
        # JSON dump rather than useful prose. Fall back to raw output text in
        # that case, and flag it, rather than nesting JSON-in-a-field.
        if parsed and "reasoning" in parsed:
            reasoning = parsed["reasoning"]
        elif parsed is not None:
            reasoning = "[NO REASONING FIELD PROVIDED BY MODEL]"
        else:
            reasoning = raw
        changed_reason = parsed.get("changed_reason") if parsed else None
        if changed_reason in ("null", "", None):
            changed_reason = None

        normalized = normalize_verdict(verdict)
        prev_normalized = previous.normalized_verdict if previous else None
        changed = bool(previous) and prev_normalized is not None and normalized != prev_normalized

        # Log a debugging flag rather than silently swallowing the problem —
        # malformed JSON at this model size should be visible, not hidden.
        if parsed is None:
            reasoning = f"[UNPARSED OUTPUT — fenced JSON not found]\n{raw}"

        flagged = find_citation_flags(raw)

        return Position(
            round=round_num,
            verdict=verdict,
            normalized_verdict=normalized,
            confidence=confidence,
            reasoning=reasoning,
            changed_from_previous=changed,
            changed_reason=changed_reason,
            raw_output=raw,
            flagged_citations=flagged,
        )

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Pull the first fenced ```json ... ``` block, or a bare {...} block, out of text."""
        fence_match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
        candidate = fence_match.group(1) if fence_match else None

        if candidate is None:
            brace_match = re.search(r"\{.*\}", text, re.DOTALL)
            candidate = brace_match.group(0) if brace_match else None

        if candidate is None:
            return None

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            return None

    # ------------------------------------------------------------------ #
    # Export
    # ------------------------------------------------------------------ #

    def to_log_dict(self) -> dict:
        """Full structured log for this juror — feeds the case-level transcript JSON."""
        return {
            "juror": self.name,
            "model": self.model,
            "positions": [
                {
                    "round": p.round,
                    "verdict": p.verdict,
                    "normalized_verdict": p.normalized_verdict,
                    "confidence": p.confidence,
                    "reasoning": p.reasoning,
                    "changed_from_previous": p.changed_from_previous,
                    "changed_reason": p.changed_reason,
                    "flagged_citations": p.flagged_citations,
                    "format_retries": p.format_retries,
                    "raw_output": p.raw_output,
                    "timestamp": p.timestamp,
                }
                for p in self.position_log
            ],
        }


SHARED_RULES = """\
CONDUCT RULES (apply regardless of persona):
- You are reasoning about a real Indian Supreme Court case drawn from the ILDC dataset. \
Ground your reasoning in Indian law: the Constitution of India, the Indian Penal Code, \
the Code of Criminal Procedure, and the Indian Evidence Act, as relevant to your persona.
- Never defer to another juror's position merely because they sound confident or because \
a majority has formed. If you change your verdict, you must cite a specific new argument \
that moved you.
- Do not use shorthand, codes, or non-standard notation. Reason in plain, complete \
natural language so your reasoning can be audited.
- You do not know the real outcome of this case. Do not guess at or reference any real \
case's actual disposition — reason only from the materials provided.

CITATION RULE — READ CAREFULLY:
- Do NOT name, cite, or refer to any specific case (no case names like "X v. State of Y", \
no year, no SCC/AIR/SCR reference numbers), even ones that sound plausible. You are a small \
local model and you WILL invent citations that do not exist if you attempt this — this has \
been observed directly in prior runs of this system and materially corrupted the jury's \
reasoning. Instead, reason from general legal doctrine and principle only, described in \
your own words without attaching a case name to it (e.g. say "Indian courts generally treat \
testimony of interested witnesses with caution absent corroboration" — NOT "as held in \
[invented case name]").
- If you are not 100% certain a precedent is real, do not mention it at all, in any form.

VERDICT RULE:
- Your "verdict" field must be exactly one of: "uphold", "quash", "modify". Use "uphold" if \
the conviction/High Court decision should stand, "quash" if it should be reversed/set aside/ \
the appellant acquitted, "modify" for any partial or reduced-charge outcome. Do not use other \
wording — the pipeline compares this field by exact string.

REQUIRED OUTPUT FORMAT — respond with a single fenced JSON block and NOTHING else outside it \
(no preamble, no judgment document, no headers, no text after the closing fence):
```json
{
  "verdict": "uphold | quash | modify",
  "confidence": <float between 0 and 1>,
  "reasoning": "<your full reasoning, several sentences, in your own words, no case citations>",
  "changed_reason": "<null on your first turn, or if you changed your verdict this round: name the juror and argument that moved you>"
}
```
"""
