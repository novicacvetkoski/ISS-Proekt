"""
Experiment configuration.

One YAML per experiment. The config is hashed into every run artifact: a result
whose provenance does not pin the model id, the seeds, and the prompt version is
not a result (see CLAUDE.md, rule 5).
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

# Bump when any prompt template changes in a way that could move results.
# Recorded per run so two runs are never silently compared across a prompt edit.
PROMPT_VERSION = "1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent


class ModelConfig(BaseModel):
    backend: Literal["gemini", "ollama", "meta"] = "gemini"
    model: str = "gemini-3.8-flash"
    temperature: float = 1.0
    thinking_level: str | None = None
    max_output_tokens: int | None = None
    num_ctx: int = 3072              # ollama only
    reasoning_effort: str | None = "low"   # meta only — Muse Spark reasons mandatorily
                                            # (a 400 if you pass "none"), and those tokens
                                            # bill as output and count against
                                            # max_output_tokens. Meta's own default is
                                            # "medium", which is what let a merely-5-string
                                            # ClerkAgenda call burn its whole budget on
                                            # reasoning and return empty content with
                                            # finish_reason="length". "low" is a deliberate,
                                            # overridable-per-role safety default, not a
                                            # measured-optimal value — raise it for juror_model
                                            # specifically if verdict quality seems to suffer.
    verify_on_start: bool = True


class DeliberationConfig(BaseModel):
    min_rounds: int = 1
    max_rounds: int = 4
    # Stop when this many consecutive rounds pass with no vote changes.
    stable_rounds_to_stop: int = 2
    max_claims_per_juror_in_digest: int = 3
    max_words_per_claim_in_digest: int = 40
    max_quote_words: int = 30


class ExperimentConfig(BaseModel):
    name: str
    pool: Literal["debug", "eval"] = "debug"
    conditions: list[str] = Field(default_factory=lambda: ["control", "treatment"])
    seed: int = 42
    juror_model: ModelConfig = Field(default_factory=ModelConfig)
    clerk_model: ModelConfig = Field(default_factory=ModelConfig)
    deliberation: DeliberationConfig = Field(default_factory=DeliberationConfig)
    personas: list[str] = Field(
        default_factory=lambda: ["textualist", "precedent_hawk", "equity_advocate",
                                 "cross_examiner", "pragmatist"]
    )
    runs_dir: str = "runs"
    prompt_version: str = PROMPT_VERSION

    @property
    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(), sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:12]

    def provenance(self) -> dict:
        return {
            "experiment": self.name,
            "config_hash": self.config_hash,
            "prompt_version": self.prompt_version,
            "juror_model": self.juror_model.model,
            "clerk_model": self.clerk_model.model,
            "seed": self.seed,
        }

    def run_dir(self) -> Path:
        return REPO_ROOT / self.runs_dir / self.name


def load_config(path: str | Path) -> ExperimentConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ExperimentConfig.model_validate(data)
