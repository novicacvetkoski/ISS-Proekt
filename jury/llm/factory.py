"""Backend selection. Keeps `if backend == ...` in exactly one place."""

from ..config import ModelConfig
from .base import LLMClient


def build_client(cfg: ModelConfig) -> LLMClient:
    if cfg.backend == "gemini":
        from .gemini import GeminiClient

        client = GeminiClient(
            model=cfg.model,
            thinking_level=cfg.thinking_level,
            max_output_tokens=cfg.max_output_tokens,
        )
    elif cfg.backend == "ollama":
        from .ollama import OllamaClient

        client = OllamaClient(model=cfg.model, num_ctx=cfg.num_ctx)
    else:  # pragma: no cover - pydantic Literal already constrains this
        raise ValueError(f"Unknown backend {cfg.backend!r}")

    if cfg.verify_on_start:
        client.verify_model()
    return client
