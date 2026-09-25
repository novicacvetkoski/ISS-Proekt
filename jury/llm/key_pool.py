"""
Multi-key rotation for the Gemini backend.

Sixteen keys (from separate Google accounts) only buy more headroom than one
key's quota if an exhausted key is retired and swapped out, rather than
retried into the ground by the existing @retry in gemini.py. That retry is
tuned for transient failures (network blips, momentary 5xx) on a *single*
key with exponential backoff; quota exhaustion is a different failure mode
and should move to the next key immediately instead of waiting out a backoff
that key will never recover from this run.

Key identity never enters provenance (CLAUDE.md rule 5: config hash, model
id, seeds, prompt version). Which of the 16 keys served a given call is not
part of what makes a run a result, so the pool doesn't record or expose that
mapping beyond a status print for debugging.
"""

import os

# google-genai does not expose a typed exception per HTTP status (yet), so
# quota/rate-limit errors are matched on message content — the same
# workaround gemini.py already leans on for StructuredCallError.
_EXHAUSTION_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota")


def is_quota_error(exc: Exception) -> bool:
    """True if `exc` looks like Gemini reporting this key is out of quota,
    as opposed to a network blip, safety block, or schema failure — those
    should retry the *same* key (gemini.py's existing @retry), not burn
    through the pool."""
    text = str(exc)
    return any(marker in text for marker in _EXHAUSTION_MARKERS)


class KeyPoolExhausted(RuntimeError):
    """Every key in the pool has been marked exhausted for this run."""


class GeminiKeyPool:
    """
    Round-robins over a list of Gemini API keys, retiring ones that come
    back quota-exhausted. One pool is shared by every GeminiClient built in
    a run — juror_client and clerk_client both draw against the same 16-key
    budget, so a key exhausted by jurors must not be handed to the clerk.
    """

    def __init__(self, keys: list[str] | None = None):
        keys = keys or self._load_keys_from_env()
        if not keys:
            raise RuntimeError(
                "No Gemini API keys found. Set GEMINI_API_KEYS (comma-separated), "
                "or GEMINI_API_KEY_1..GEMINI_API_KEY_N, or fall back to a single "
                "GEMINI_API_KEY / GOOGLE_API_KEY."
            )
        self._keys = keys
        self._exhausted: set[str] = set()
        self._index = 0

    @staticmethod
    def _load_keys_from_env() -> list[str]:
        # Preferred: one comma-separated var — this is what `export
        # GEMINI_API_KEYS="k1,k2,...,k16"` populates.
        blob = os.environ.get("GEMINI_API_KEYS")
        if blob:
            return [k.strip() for k in blob.split(",") if k.strip()]

        # Alternative: GEMINI_API_KEY_1, GEMINI_API_KEY_2, ... (stops at the
        # first gap, so keep the numbering contiguous).
        numbered = []
        i = 1
        while True:
            k = os.environ.get(f"GEMINI_API_KEY_{i}")
            if not k:
                break
            numbered.append(k)
            i += 1
        if numbered:
            return numbered

        # Single-key fallback — existing `export GEMINI_API_KEY=...` setups
        # keep working unchanged, just as a one-element pool.
        single = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        return [single] if single else []

    @property
    def remaining(self) -> int:
        return len(self._keys) - len(self._exhausted)

    def current(self) -> str:
        """The key callers should be using right now."""
        if self.remaining <= 0:
            raise KeyPoolExhausted(f"All {len(self._keys)} Gemini keys are exhausted for this run.")
        # Defensive: rotate() is expected to leave _index on a live key, but
        # don't trust that blindly across future edits.
        while self._keys[self._index] in self._exhausted:
            self._index = (self._index + 1) % len(self._keys)
        return self._keys[self._index]

    def rotate(self, dead_key: str) -> str:
        """Mark `dead_key` exhausted and return the next live key. Raises
        KeyPoolExhausted if that was the last one."""
        if dead_key not in self._exhausted:
            self._exhausted.add(dead_key)
            print(
                f"  [GeminiKeyPool] key ...{dead_key[-4:]} exhausted "
                f"({self.remaining}/{len(self._keys)} remaining)"
            )
        self._index = (self._index + 1) % len(self._keys)
        return self.current()
