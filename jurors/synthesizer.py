"""
Synthesizer agent.

Distinct from JurorAgent: it doesn't hold an independent position through the
debate, doesn't participate in rounds, and is only called once at the end to
aggregate the jury's final positions into a single verdict + rationale. Kept
as its own class rather than a JurorAgent subclass since its interface
(single aggregation call) is genuinely different from the debate protocol.
"""

import json
import re

import ollama

from .base import DEFAULT_MODEL, DEFAULT_NUM_CTX, find_citation_flags, normalize_verdict

SYNTHESIZER_PROMPT = """\
You are the synthesizer for a five-juror AI jury reasoning about an Indian Supreme
Court case (from the ILDC dataset). You are not a juror yourself and you must not
introduce new legal arguments of your own. Your job is to aggregate the five jurors'
final positions into a single collective verdict and a rationale that a legal
professional could read on its own, without access to the underlying debate
transcript, and understand why the jury reached that verdict.

Rules:
- Do not simply pick the majority verdict without explanation. If the jury is split,
  say so explicitly and explain how you are resolving it (e.g. weight of reasoning,
  not just headcount).
- Your rationale must reference the specific arguments that were decisive, not just
  restate that "the jury discussed several views."
- Note any juror who dissented and briefly state their reasoning, even in the final
  summary — dissent should remain visible, not smoothed over.
- Your "final_verdict" field must be exactly one of: "uphold", "quash", "modify".
- Do NOT name, cite, or refer to any specific case (no case names, years, or SCC/AIR
  reference numbers) even if a juror's reasoning mentioned one. You are a small local
  model and are highly likely to fabricate or repeat a fabricated citation if you
  attempt this. Summarize the underlying legal principle a juror relied on in your
  own words instead, without attaching a case name to it.

Respond with a single fenced JSON block and nothing else outside it:
```json
{
  "final_verdict": "uphold | quash | modify",
  "vote_breakdown": {"<juror name>": "uphold | quash | modify", ...},
  "rationale": "<several sentences explaining why this verdict was reached, citing the decisive arguments in your own words, no case citations>",
  "dissent_summary": "<null if unanimous, otherwise a summary of dissenting reasoning>"
}
```
"""


class Synthesizer:
    def __init__(self, model: str = DEFAULT_MODEL, num_ctx: int = DEFAULT_NUM_CTX):
        self.model = model
        self.num_ctx = num_ctx
        self.raw_output: str | None = None

    def synthesize(self, case_text: str, final_positions: list[dict]) -> dict:
        """
        final_positions: list of {"juror": name, "verdict": ..., "confidence": ...,
        "reasoning": ...} for each juror's LAST logged position.
        """
        positions_block = "\n\n".join(
            f"{p['juror']} — verdict: {p['verdict']} (confidence: {p['confidence']})\n"
            f"Reasoning: {p['reasoning']}"
            for p in final_positions
        )
        user_content = (
            f"CASE MATERIALS:\n{case_text}\n\n"
            f"FINAL JUROR POSITIONS:\n{positions_block}\n\n"
            "Produce the jury's final verdict and rationale using the required JSON format."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": SYNTHESIZER_PROMPT},
                {"role": "user", "content": user_content},
            ],
            options={"temperature": 0.4, "num_ctx": self.num_ctx},  # lower temp: aggregation, not debate
        )
        raw = response["message"]["content"]

        self.raw_output = raw
        parsed = self._extract_json(raw)
        if parsed is None:
            parsed = {
                "final_verdict": None,
                "vote_breakdown": {p["juror"]: p["verdict"] for p in final_positions},
                "rationale": f"[UNPARSED SYNTHESIZER OUTPUT]\n{raw}",
                "dissent_summary": None,
            }
        else:
            # Normalize even though the prompt requests controlled vocabulary directly —
            # safety net for when the model doesn't comply, same rationale as juror parsing.
            parsed["final_verdict"] = normalize_verdict(parsed.get("final_verdict"))
        parsed["raw_output"] = raw
        parsed["flagged_citations"] = find_citation_flags(raw)
        return parsed

    @staticmethod
    def _extract_json(text: str) -> dict | None:
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
