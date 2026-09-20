# CLAUDE.md — AI Jury (ISS-Proekt)

Multi-agent AI jury that deliberates over real Indian Supreme Court appeals (ILDC/CJPE) and is
scored against the published outcome. Two research questions: (RQ1) can a panel of independent
personality-bearing agents reach defensible verdicts, and (RQ2) can one agent that sets out to sway
the others change the jury's verdict — the *12 Angry Men* condition.

Design rationale lives in [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md). Research notes and
sources are in [docs/HANDOFF.md](docs/HANDOFF.md). Read the design doc before changing the
deliberation protocol, the personas, or the metrics.

## Environment

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # memorizz needs >=3.10
uv pip install -e ".[dev]"
export GEMINI_API_KEY=...        # jurors
export HF_TOKEN=...              # ILDC is a gated dataset
```
Run everything with `.venv/bin/python`. The system `python3` is 3.9 and will fail on
`X | None` annotations and on memorizz.

## Non-negotiable rules

These protect the validity of the experiment. Breaking one silently invalidates a run.

1. **The gold label never reaches a model.** `data/ildc_loader.get_case_text()` is the only
   sanctioned way to build juror-facing text. Nothing else may read `record["label"]` on the path
   to a prompt. Labels are for scoring, after the fact.
2. **`data/eval_pool.json` is write-once.** Iterate prompts on `debug_pool.json` only. The pool
   builder already refuses to regenerate the eval pool; do not "fix" that.
3. **Personas are frozen during a run.** memorizz persona *evolution* (`update()`,
   `update_persona`) stays off — a persona that mutates leaks information between the 100 cases.
   The registry asserts `version == 1`.
4. **No cross-case memory.** Each case is an independent trial. No conversation carries over.
5. **Every experiment artifact records provenance**: config hash, exact model id, seeds, prompt
   version. A run without provenance is not a result.
6. **Verdicts are binary**: `ALLOW` (appeal accepted, gold label 1) / `DISMISS` (rejected, 0).
   There is no `modify` — the dataset has no such class, and it would collapse to ALLOW anyway.

## Architecture in one paragraph

LangGraph owns orchestration. memorizz owns persona identity (storage + versioning) **only** —
never model calls, never the agent loop. Gemini is called directly through `jury/llm/`. Jurors are
**stateless functions** of (profile, case, context): no growing chat history, all state in the graph
state object, so prompt size is O(1) in the number of rounds. Round 0 is private and shared by both
experimental conditions.

## Conventions that bite

- **Structured output only.** Gemini `response_schema=` + Ollama `format=`. Do not parse free text.
  ⚠️ Gemini rejects Pydantic models whose fields have **default values**
  (googleapis/python-genai#699) — response models in `jury/domain/models.py` must have all fields
  required. Put defaults in the *call site*, not the schema.
- **memorizz `Persona.__init__` pollutes personas.** It appends your text onto generic
  `PREDEFINED_INFO` defaults ("Provide versatile support across various domains.") and calls an
  embedding provider at construction. Always build with `Persona.from_dict(...)`, which bypasses
  both. memorizz has **no Gemini provider** — don't try to route calls through it.
- **Prompt layout is cache-shaped**: `[court instructions][case file][persona card][task]`. The
  shared prefix is identical across all five jurors so Gemini's implicit cache hits (≥4,096 tokens
  for 3.x Flash). Don't reorder it casually; log `cached_content_token_count` if you do.
- **Claims carry anchors.** Every claim cites a case paragraph id (`[P17]`) and may quote ≤30 words
  verbatim. Anchors are validated by substring match — this is the cheap grounding metric and the
  mechanism that lets us tell informational from normative vote changes. Don't make anchors optional.
- **No case citations by name.** Models fabricate Indian case law; this was observed directly in
  this project's own earlier runs. Jurors reason from doctrine in their own words.
  `find_citation_flags()` is the detector, the prompt rule is the preventer.
- `datasets<4.0.0` is required — IL-TUR is a script-based dataset and 4.x removed that entirely.

## Commands

```bash
.venv/bin/python -m jury.data.prepare_pools --debug-size 15 --eval-size 100
.venv/bin/python -m jury.experiments.runner --config configs/debug.yaml --case-index 0
.venv/bin/python -m jury.experiments.runner --config configs/eval.yaml --condition control
.venv/bin/python -m jury.analysis.rq1 runs/<exp>/     # accuracy, F1, kappa, baselines
.venv/bin/python -m jury.analysis.rq2 runs/<exp>/     # paired flips, McNemar, noise floor
.venv/bin/pytest -q
```

Runs are resumable: one JSON per (case, condition), written atomically; the runner skips what
already exists. Delete a file to recompute it.

## Repo map

```
jury/llm/            LLMClient protocol, Gemini + Ollama backends, factory
jury/domain/         pydantic schemas, verdict vocabulary
jury/personas/       YAML profiles (source of truth) + memorizz registry + prompt rendering
jury/jurors/         Juror, Clerk (non-voting), persuader overlay, prompt templates
jury/deliberation/   graph state, nodes, LangGraph wiring, digest, stopping rules
jury/data/           ILDC loader, pool builder, case-file paragraph numbering
jury/experiments/    runner, conditions
jury/analysis/       metrics, RQ1, RQ2, persona differentiation, grounding
```

Legacy top-level `jurors/`, `data/`, `run_case.py` are the Ollama-era prototype. `run_case.py`
imports a module (`orchestrator`) that does not exist in the repo — it is dead. Migrate, don't extend.
