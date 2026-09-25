"""Backend selection. Keeps `if backend == ...` in exactly one place."""

from ..config import ModelConfig
from .base import LLMClient

# One pool per process, built lazily from env on the first Gemini client a
# run asks for. runner.py builds juror_client and clerk_client as two
# separate build_client() calls, but they must draw on the same 16-key
# budget — a key marked exhausted by jurors must not be handed to the clerk
# a moment later. Module-level cache is what makes them share it without
# threading a pool argument through every call site that constructs a config.
_shared_gemini_pool = None


def _get_gemini_pool():
    global _shared_gemini_pool
    if _shared_gemini_pool is None:
        from .key_pool import GeminiKeyPool

        _shared_gemini_pool = GeminiKeyPool()
    return _shared_gemini_pool


def build_client(cfg: ModelConfig) -> LLMClient:
    if cfg.backend == "gemini":
        from .gemini import GeminiClient

        client = GeminiClient(
            model=cfg.model,
            key_pool=_get_gemini_pool(),
            thinking_level=cfg.thinking_level,
            max_output_tokens=cfg.max_output_tokens,
        )
    elif cfg.backend == "ollama":
        from .ollama import OllamaClient

        client = OllamaClient(model=cfg.model, num_ctx=cfg.num_ctx)
    elif cfg.backend == "meta":
        from .meta import MetaClient

        # No shared pool here, unlike Gemini — see meta.py's module docstring for why
        # one key's budget is enough that rotation isn't needed for this backend.
        client = MetaClient(
            model=cfg.model,
            max_output_tokens=cfg.max_output_tokens,
            reasoning_effort=cfg.reasoning_effort,
        )
    else:  # pragma: no cover - pydantic Literal already constrains this
        raise ValueError(f"Unknown backend {cfg.backend!r}")

    if cfg.verify_on_start:
        client.verify_model()
    return client
