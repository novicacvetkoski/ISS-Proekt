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


@dataclass
class Position:
    """A single structured position taken by a juror at a point in the debate."""
    round: int  # 0 = independent initial analysis, 1..N = debate rounds
    verdict: str | None
    confidence: float | None
    reasoning: str
    changed_from_previous: bool
    changed_reason: str | None
    raw_output: str
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
            "You have not seen any other juror's position. "
            "Respond using the required JSON format."
        )
        raw = self._call_model(prompt)
        position = self._parse_position(raw, round_num=0, previous=None)
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
            "different view — deference without a distinct new argument is invalid. "
            "Respond using the required JSON format."
        )
        raw = self._call_model(prompt)
        previous = self.position_log[-1] if self.position_log else None
        position = self._parse_position(raw, round_num=round_num, previous=previous)
        self.position_log.append(position)
        return position

    # ------------------------------------------------------------------ #
    # Parsing
    # ------------------------------------------------------------------ #

    def _parse_position(self, raw: str, round_num: int, previous: Position | None) -> Position:
        parsed = self._extract_json(raw)

        verdict = parsed.get("verdict") if parsed else None
        confidence = parsed.get("confidence") if parsed else None
        reasoning = parsed.get("reasoning", raw) if parsed else raw
        changed_reason = parsed.get("changed_reason") if parsed else None

        changed = bool(previous) and previous.verdict is not None and verdict != previous.verdict

        # Log a debugging flag rather than silently swallowing the problem —
        # malformed JSON at this model size should be visible, not hidden.
        if parsed is None:
            reasoning = f"[UNPARSED OUTPUT — fenced JSON not found]\n{raw}"

        return Position(
            round=round_num,
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            changed_from_previous=changed,
            changed_reason=changed_reason,
            raw_output=raw,
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
                    "confidence": p.confidence,
                    "reasoning": p.reasoning,
                    "changed_from_previous": p.changed_from_previous,
                    "changed_reason": p.changed_reason,
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
the Code of Criminal Procedure, the Indian Evidence Act, and actual or plausible Supreme \
Court / High Court precedent, as relevant to your persona.
- Never defer to another juror's position merely because they sound confident or because \
a majority has formed. If you change your verdict, you must cite a specific new argument \
that moved you.
- Do not use shorthand, codes, or non-standard notation. Reason in plain, complete \
natural language so your reasoning can be audited.
- You do not know the real outcome of this case. Do not guess at or reference any real \
case's actual disposition — reason only from the materials provided.

REQUIRED OUTPUT FORMAT — respond with a single fenced JSON block and nothing else outside it:
```json
{
  "verdict": "<your verdict on the case, in a few words>",
  "confidence": <float between 0 and 1>,
  "reasoning": "<your full reasoning, several sentences>",
  "changed_reason": "<null on your first turn, or if you changed your verdict this round: name the juror and argument that changed your mind>"
}
```
"""
