"""
Meta Model API backend (muse-spark family), via the OpenAI-compatible Chat Completions
endpoint at api.meta.ai/v1.

Notes that cost time to rediscover:
- This is Meta's own first-party Model API, not a third-party Llama host — different
  from the "Llama API" naming you'll see elsewhere. muse-spark-1.2-contributor is the
  model the cost math was built around: $0.10/1M input, $0.20/1M output, $0.002/1M
  CACHED input (a 50x discount) — see prompt_cache_retention below. The "-contributor"
  suffix means Meta may use prompts/completions to improve their models; the non
  -contributor tier (muse-spark-1.2) costs ~12.5x more and is never trained on. That
  trade-off is a deliberate choice for the person running this, not something to
  silently default one way.
- response_format={"type": "json_schema", ...} is confirmed supported (Meta's Model API
  docs / promptfoo's provider page). Unlike Gemini this API does NOT reject default
  -valued fields, but strict:true DOES require additionalProperties: false on every
  object in the schema (OpenAI's own strict-mode contract, which Meta mirrors exactly)
  — pydantic's model_json_schema() does not set that on its own, and a bare
  schema.model_json_schema() gets rejected with "'additionalProperties' is required to
  be supplied and to be false." _to_strict_schema() below walks the WHOLE schema tree,
  including $defs for nested models, and sets it everywhere an object appears. The
  other strict-mode requirement — every field listed in "required" — is already
  satisfied for free, because domain.models expresses optionality as `X | None` rather
  than a default (see that module's own docstring), so nothing extra was needed there.
- prompt_cache_retention is passed as "in_memory" so the repeated case-file prefix
  (byte-identical across all five jurors, same reasoning as gemini.py) can hit the
  cached-input rate instead of the full rate. Not guaranteed — same "measured, not
  assumed" posture as Gemini's cache: usage.prompt_tokens_details is read back per call
  rather than trusted blindly.
- `seed` is best-effort determinism here too, same caveat as the other two backends.
- No key-pool for this backend, unlike Gemini. The cost math this was built for shows a
  single key's $20 cap comfortably covers a full eval run several times over — the
  constraint that made Gemini's free tier need 16 keys (a 20-requests/day quota) doesn't
  exist here. If you instead hit a requests-per-minute rate limit under real load, that's
  a different problem than budget and would need its own handling — not built preemptively.
- MUSE SPARK REASONS MANDATORILY. reasoning_effort accepts minimal/low/medium/high/xhigh
  but NOT "none" — Meta's API 400s if you try to disable it. Those reasoning tokens bill
  as output (usage.completion_tokens_details.reasoning_tokens) and count against the SAME
  budget as max_output_tokens. Meta's own default effort is "medium" with no cap unless
  you set one, which is exactly how a ClerkAgenda call (five short strings — nothing
  about the SCHEMA is expensive) can burn its whole implicit budget thinking and return
  empty content with finish_reason="length": the model never got to the visible output.
  reasoning_effort defaults to "low" here (see config.py's ModelConfig — not hardcoded
  only here, so it stays overridable per role) as a deliberate safety default, not a
  measured-optimal one. Raise max_output_tokens generously regardless of effort level —
  reasoning tokens are inherently variable per case, and this failure mode is silent
  content-loss, not a clean error, until finish_reason is checked.
"""

import os

from openai import OpenAI
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..domain.models import Usage
from .base import StructuredCallError

DEFAULT_BASE_URL = "https://api.meta.ai/v1"


def _to_strict_schema(schema: type[BaseModel]) -> dict:
    """
    pydantic's model_json_schema() doesn't set additionalProperties: false anywhere,
    but OpenAI-style strict:true requires it on every object subschema — top-level AND
    inside $defs, since nested models (e.g. a list[Claim] field) show up there. Walk
    the whole tree rather than patching just the top level, or a nested model's schema
    fails the same check one level down.
    """
    raw = schema.model_json_schema()

    def _walk(node):
        if isinstance(node, dict):
            if node.get("type") == "object" or "properties" in node:
                node.setdefault("additionalProperties", False)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    _walk(raw)
    return raw


class MetaClient:
    def __init__(
        self,
        model: str = "muse-spark-1.2-contributor",
        api_key: str | None = None,
        base_url: str | None = None,
        max_output_tokens: int | None = None,
        prompt_cache_retention: str | None = "in_memory",
        reasoning_effort: str | None = "low",
    ):
        key = api_key or os.environ.get("MODEL_API_KEY")
        if not key:
            raise RuntimeError(
                "Set MODEL_API_KEY to use the Meta backend (developer.meta.com/ai/products/meta-model-api)."
            )
        self.client = OpenAI(
            api_key=key,
            base_url=base_url or os.environ.get("MODEL_API_BASE_URL") or DEFAULT_BASE_URL,
        )
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.prompt_cache_retention = prompt_cache_retention
        self.reasoning_effort = reasoning_effort

    def verify_model(self) -> None:
        """
        Fail fast if the pinned model id isn't available to this key. Same reasoning as
        the Gemini/Ollama verify_model: an eval run must not silently fall back to a
        different (and differently priced, differently trained) model mid-run.
        """
        try:
            names = {m.id for m in self.client.models.list()}
        except Exception as e:
            raise RuntimeError(f"Meta Model API not reachable: {e}") from e
        if self.model not in names:
            raise RuntimeError(
                f"Model {self.model!r} is not available to this API key. "
                f"Available (sample): {sorted(names)[:10]}"
            )

    @retry(
        retry=retry_if_exception_type(StructuredCallError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        reraise=True,
    )
    def generate(
        self,
        *,
        system: str,
        prefix: str,
        task: str,
        schema: type[BaseModel],
        temperature: float,
        seed: int | None = None,
    ) -> tuple[BaseModel, Usage]:
        kwargs = {}
        if seed is not None:
            kwargs["seed"] = seed
        if self.max_output_tokens is not None:
            kwargs["max_tokens"] = self.max_output_tokens
        if self.prompt_cache_retention is not None:
            kwargs["prompt_cache_retention"] = self.prompt_cache_retention
        if self.reasoning_effort is not None:
            # extra_body, not a direct kwarg: reasoning_effort isn't a field the base
            # OpenAI SDK's request schema knows about, so it has to go through as a
            # raw pass-through field rather than risk the SDK stripping/rejecting it.
            kwargs.setdefault("extra_body", {})["reasoning_effort"] = self.reasoning_effort

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    # Cache-shaped order, same reasoning as gemini.py: large shared
                    # prefix first, small volatile task last, so the cacheable portion
                    # of the message is byte-identical across all five jurors.
                    {"role": "user", "content": f"{prefix}\n\n{task}"},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": _to_strict_schema(schema),
                        "strict": True,
                    },
                },
                temperature=temperature,
                **kwargs,
            )
        except Exception as e:
            raise StructuredCallError(f"Meta call failed: {e}") from e

        choice = response.choices[0]
        content = choice.message.content
        usage_obj = response.usage
        reasoning = 0
        comp_details = getattr(usage_obj, "completion_tokens_details", None)
        if comp_details is not None:
            reasoning = getattr(comp_details, "reasoning_tokens", 0) or 0

        if not content:
            hint = (
                f" — {reasoning} reasoning tokens were spent before the budget ran out; "
                f"raise max_output_tokens and/or lower reasoning_effort (currently "
                f"{self.reasoning_effort!r})"
                if choice.finish_reason == "length"
                else ""
            )
            raise StructuredCallError(
                f"Meta returned no content for {schema.__name__}. "
                f"finish_reason={choice.finish_reason}{hint}"
            )

        try:
            parsed = schema.model_validate_json(content)
        except Exception as e:
            raise StructuredCallError(
                f"Meta output did not validate as {schema.__name__} "
                f"(finish_reason={choice.finish_reason}): {e}"
            ) from e

        cached = 0
        prompt_details = getattr(usage_obj, "prompt_tokens_details", None)
        if prompt_details is not None:
            cached = getattr(prompt_details, "cached_tokens", 0) or 0

        usage = Usage(
            input_tokens=getattr(usage_obj, "prompt_tokens", 0) or 0,
            output_tokens=getattr(usage_obj, "completion_tokens", 0) or 0,
            cached_tokens=cached,
            reasoning_tokens=reasoning,
            model=self.model,
        )
        return parsed, usage
