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
- With a key_pool, a quota-exhausted key rotates to the next key and rebuilds
  `self.client` mid-run. This costs the implicit cache: Gemini's cache is scoped
  to the key/project that wrote it, so a rotation forces a cache miss on the next
  call. That's an accepted trade against the alternative (the run just stops).
- 503 "high demand" is a model-capacity problem, not a key problem — it happens on
  brand-new model releases before Google finishes scaling serving capacity, and it
  is the SAME for every key in the pool. Rotating keys on a 503 doesn't fix
  anything; it just burns free keys for no benefit. So 503 gets its own recovery
  path: wait OVERLOAD_RETRY_SECONDS and retry the SAME key, indefinitely, instead
  of treating it like quota exhaustion. This can genuinely hang a run for as long
  as the outage lasts (these have run from minutes to a couple of weeks in the
  wild) — Ctrl+C is safe any time, since every case is written atomically and
  skipped on the next run, so nothing already-completed is lost.
"""

import os
import time

from google import genai
from google.genai import types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..domain.models import Usage
from .base import StructuredCallError
from .key_pool import GeminiKeyPool, KeyPoolExhausted, is_quota_error

OVERLOAD_RETRY_SECONDS = 30

# google-genai does not expose a typed exception per HTTP status (yet), so this is
# matched on message content — same workaround is_quota_error uses in key_pool.py.
# Deliberately a DIFFERENT marker set from is_quota_error's: a 503 must never be
# treated as a reason to rotate keys (see module docstring).
_OVERLOAD_MARKERS = ("UNAVAILABLE", "503", "overloaded", "high demand")


def is_overloaded_error(exc: Exception) -> bool:
    """True if `exc` is Gemini reporting the model itself is temporarily out of
    serving capacity — the same for every key, unlike quota exhaustion."""
    text = str(exc)
    return any(marker in text for marker in _OVERLOAD_MARKERS)


class GeminiClient:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        key_pool: GeminiKeyPool | None = None,
        thinking_level: str | None = None,
        max_output_tokens: int | None = None,
    ):
        self.key_pool = key_pool
        if key_pool is not None:
            key = key_pool.current()
        else:
            key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) to use the Gemini backend.")
        self._current_key = key
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

        # Cache-shaped order: large shared prefix first, small volatile tail last.
        response = self._call_with_recovery(prefix=prefix, task=task, config=config)

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

    def _call_with_recovery(self, *, prefix: str, task: str, config: "types.GenerateContentConfig"):
        """
        Try the call on the current key, with two independent recovery paths —
        deliberately not the same mechanism, because they fix different problems:

        - Overload (503, a MODEL problem, identical across every key): waiting
          doesn't cost anything but time, and rotating keys wouldn't help, so
          wait OVERLOAD_RETRY_SECONDS and retry the SAME key, indefinitely. This
          never reaches the outer @retry on generate() and never raises — it
          only returns once a response comes back.
        - Quota exhaustion (a KEY problem): rotate to the pool's next key and
          retry immediately. A dead key doesn't get better after a wait, so it
          doesn't get one either.

        Anything else (network blip, safety block, schema miss) is left to the
        @retry decorator on generate(), which retries the same key 3x with
        exponential backoff — unchanged from before this method existed.

        Without a key_pool, quota errors behave exactly as before (raise
        immediately); overload errors still wait-and-retry regardless, since
        that recovery has nothing to do with having a pool.
        """
        while True:
            try:
                return self.client.models.generate_content(
                    model=self.model,
                    contents=[prefix, task],
                    config=config,
                )
            except Exception as e:  # network, overload, quota, safety block
                if is_overloaded_error(e):
                    print(
                        f"  [GeminiClient] {self.model} overloaded (503) — waiting "
                        f"{OVERLOAD_RETRY_SECONDS}s before retrying the same key "
                        f"(Ctrl+C is safe; nothing completed so far is lost)",
                        flush=True,
                    )
                    time.sleep(OVERLOAD_RETRY_SECONDS)
                    continue  # same key, same call, forever until it succeeds

                if self.key_pool is None or not is_quota_error(e):
                    raise StructuredCallError(f"Gemini call failed: {e}") from e
                try:
                    next_key = self.key_pool.rotate(self._current_key)
                except KeyPoolExhausted:
                    raise StructuredCallError(f"Gemini call failed, pool exhausted: {e}") from e
                self._current_key = next_key
                self.client = genai.Client(api_key=next_key)
                # loop: retry the same call on the freshly rotated key