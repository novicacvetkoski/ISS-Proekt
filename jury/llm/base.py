"""
The one interface every model backend implements.

Deliberately narrow: one structured call. Jurors are stateless, so there is no
chat-session concept here and no history to manage — the caller rebuilds the full
prompt each time from graph state. That is what keeps prompt size O(1) in rounds.

PROMPT LAYOUT IS PART OF THE CONTRACT. `prefix` is the large, shared, byte-identical
portion (court instructions + case file) and `task` is the small volatile tail.
Backends must emit them in that order and must not interleave anything, because
Gemini's implicit cache keys on the prefix (>=4,096 tokens on 3.x Flash) and a
reordering silently costs a cache miss on every call.
"""

from typing import Protocol, TypeVar

from pydantic import BaseModel

from ..domain.models import Usage

T = TypeVar("T", bound=BaseModel)


class StructuredCallError(RuntimeError):
    """Raised when a backend cannot produce a schema-valid response."""


class LLMClient(Protocol):
    """Backends: GeminiClient, OllamaClient."""

    model: str

    def generate(
        self,
        *,
        system: str,
        prefix: str,
        task: str,
        schema: type[T],
        temperature: float,
        seed: int | None = None,
    ) -> tuple[T, Usage]:
        """Return a schema-valid object plus token usage. Raises StructuredCallError."""
        ...
