"""
Local Ollama backend — the secondary model arm.

Kept because the closest prior work's headline claim is that ALIGNMENT STYLE, not
capability, drives deliberative flexibility (12 Angry AI Agents, arXiv 2605.01986:
GPT-4o 1.0 vote changes/run vs Llama-4-Scout 2.0-6.0). Testing that on our task
needs a second model family. Nothing in the main design depends on this arm.

Ollama takes a JSON schema directly via `format=`, so the structured-output
contract is identical to Gemini's and the prototype's regex/fenced-JSON/retry
machinery is not needed here either.

Context note: a 4GB card runs num_ctx ~3072, which a full ILDC case (avg ~3.2k
tokens) does not fit alongside prompt overhead. Use the summarised case file for
this backend, or expect truncation — silently, because Ollama does not error.
"""

import json

import ollama
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from ..domain.models import Usage
from .base import StructuredCallError

DEFAULT_NUM_CTX = 3072


class OllamaClient:
    def __init__(self, model: str = "qwen3:4b", num_ctx: int = DEFAULT_NUM_CTX):
        self.model = model
        self.num_ctx = num_ctx

    def verify_model(self) -> None:
        try:
            available = {m.model for m in ollama.list().models}
        except Exception as e:
            raise RuntimeError(f"Ollama daemon not reachable: {e}") from e
        if self.model not in available and f"{self.model}:latest" not in available:
            raise RuntimeError(f"Model {self.model!r} not pulled. Run: ollama pull {self.model}")

    @retry(
        retry=retry_if_exception_type(StructuredCallError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=20),
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
        options = {"temperature": temperature, "num_ctx": self.num_ctx}
        if seed is not None:
            options["seed"] = seed

        try:
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"{prefix}\n\n{task}"},
                ],
                format=schema.model_json_schema(),
                options=options,
            )
        except Exception as e:
            raise StructuredCallError(f"Ollama call failed: {e}") from e

        content = response["message"]["content"]
        try:
            parsed = schema.model_validate(json.loads(content))
        except (json.JSONDecodeError, ValidationError) as e:
            raise StructuredCallError(f"Ollama output did not validate as {schema.__name__}: {e}") from e

        usage = Usage(
            input_tokens=response.get("prompt_eval_count", 0) or 0,
            output_tokens=response.get("eval_count", 0) or 0,
            cached_tokens=0,
            model=self.model,
        )
        return parsed, usage
