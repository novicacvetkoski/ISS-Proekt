"""
Gemini backend (google-genai).

Notes that cost time to rediscover:
- response_schema rejects pydantic models with DEFAULT values
  (googleapis/python-genai#699). Every response model in domain.models is
  all-required for this reason. Do not add a default to "make it optional" —
  use `X | None` instead.
- Implicit caching is on by default for 2.5+ and needs >=4,096 tokens on 3.x Flash.
  We put the case file in `prefix` so it is byte-identical across all five jurors
  in a round; cached_content_token_count is recorded per call so cache behaviour
  is measured rather than assumed.
- `seed` is best-effort determinism, not a guarantee. Reruns still differ; that is
  exactly why the experiment measures a noise floor instead of trusting seeds.
"""

import os

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..domain.models import Usage
from .base import StructuredCallError


class GeminiClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        thinking_level: str | None = None,
        max_output_tokens: int | None = None,
    ):
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini backend.")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens

    def verify_model(self) -> None:
        """
        Fail fast if the pinned model id no longer exists. Model ids churn; an eval
        run must not silently fall back to a different model halfway through.
        """
        names = {m.name.split("/")[-1] for m in self.client.models.list()}
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
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json",
            response_schema=schema,
            seed=seed,
            max_output_tokens=self.max_output_tokens,
        )
        if self.thinking_level:
            config.thinking_config = types.ThinkingConfig(thinking_level=self.thinking_level)

        try:
            # Cache-shaped order: large shared prefix first, small volatile tail last.
            response = self.client.models.generate_content(
                model=self.model,
                contents=[prefix, task],
                config=config,
            )
        except Exception as e:  # network, quota, safety block
            raise StructuredCallError(f"Gemini call failed: {e}") from e

        parsed = response.parsed
        if parsed is None:
            raise StructuredCallError(
                f"Gemini returned no schema-valid object for {schema.__name__}. "
                f"finish_reason={getattr(response.candidates[0], 'finish_reason', None) if response.candidates else None}"
            )

        meta = response.usage_metadata
        usage = Usage(
            input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
            output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
            cached_tokens=getattr(meta, "cached_content_token_count", 0) or 0,
            model=self.model,
        )
        return parsed, usage
