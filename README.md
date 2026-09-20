# AI Jury — Debate-Based Multi-Agent Legal Decision Support

Five AI agents, each an independent juror with its own personality and legal-reasoning
style, deliberate over real appeals decided by the Supreme Court of India and reach a
collective verdict with a written rationale. The verdict is then compared against what the
court actually decided.

The project asks two questions:

**RQ1 — Independence.** Can a panel of independent, personality-bearing agents produce
verdicts and reasoning good enough to support legal professionals? Measured as divergence
from the published outcome, against baselines that isolate what *deliberation* adds over
five agents simply voting.

**RQ2 — Influence (the *12 Angry Men* condition).** If one agent sets out to sway the
others, does the jury's verdict change? Measured as a paired contrast on the same 100 cases.

The goal is not to replace judges or juries, but to find out whether a collaborative AI
jury can be reliable, transparent and well-reasoned enough to assist — and where it fails.

---

## Status

Phases 0–4 are implemented and covered by **65 tests that exercise the entire deliberation
graph against a scripted fake backend** — no API key, no cost. The experiment itself
(phases 5–8) needs credentials and has not been run.

```bash
.venv/bin/pytest -q     # 65 passed
```

Still to write, both deliberately gated *before* the eval run: the **memorization probe**
and the **single-agent baseline (B1)**.

---

## Quickstart

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate   # memorizz needs >=3.10
uv pip install -e ".[dev,analysis,data]"

export GEMINI_API_KEY=...     # the jurors
export HF_TOKEN=...           # ILDC is a gated dataset — see below

# Build the case pools (write-once for eval)
python -m jury.data.prepare_pools --debug-size 15 --eval-size 100

# One case, cheap model, both conditions
python -m jury.experiments.runner --config configs/debug.yaml --case-index 0

# The experiment
python -m jury.experiments.runner --config configs/eval.yaml

# Results
python -m jury.analysis.rq1 runs/eval_v1 --pool eval
python -m jury.analysis.rq2 runs/eval_v1 --pool eval --noise-dir runs/noise
```

Runs are **resumable**: one JSON per (case, condition), written atomically, skipped if
present. Re-running the same command retries exactly what failed. Delete a file to
recompute it.

### Dataset access

ILDC/CJPE is gated. One-time setup: create a free Hugging Face account, click *Agree and
access repository* at `huggingface.co/datasets/Exploration-Lab/IL-TUR`, then
`huggingface-cli login` or set `HF_TOKEN`. The dataset is CC-BY-NC-SA 4.0 — non-commercial
only, and cite Malik et al. (2021) if you publish.

`datasets<4.0.0` is required: IL-TUR ships a loading script, and 4.x removed script-based
datasets entirely.

---

## How it works

```
load case
    │
    ├─ 5 jurors review the case privately, in parallel  ──► secret ballot 0
    │      (no juror has seen any other juror's view)         = free "no deliberation" baseline
    │
    ├─ clerk merges their issues into an agenda
    │
    ├──────────── FORK: round 0 is shared by both conditions ────────────┐
    │                                                                     │
  CONTROL                                                            TREATMENT
    │                                                        one juror gets the persuader
    │                                                        overlay, targeting the opposite
    │                                                        of the round-0 majority
    └──────────────────────────┬──────────────────────────────────────────┘
                               │
                    deliberation rounds (all 5 speak simultaneously)
                    stop on unanimity, stability, or round 4
                               │
                    clerk writes the rationale (non-voting)
```

**Jurors are stateless.** No juror holds a conversation history; each prompt is rebuilt
from graph state, so prompt size is **O(1) in the number of rounds** rather than growing
every turn. Jurors never see the raw transcript — they see a hard-capped digest of each
other's current positions.

**Every claim is anchored** to a case paragraph id (`[P17]`), optionally with a ≤30-word
verbatim quote. This buys three things at once: grounding is checkable by substring match
with no LLM judge; a vote change can be attributed to a specific argument; and fabricated
paragraph references are visible without reading the transcript.

**That attribution is the RQ2 measurement.** A juror that flips while citing a real,
case-grounded claim by another juror was *persuaded* (informational influence). One that
flips citing nothing, or citing only the vote tally, yielded to the room (normative
influence). Prior work shows a bare flip rate conflates these with mere noise.

Full rationale, with sources: [docs/SYSTEM_DESIGN.md](docs/SYSTEM_DESIGN.md).
Research notes and literature: [docs/HANDOFF.md](docs/HANDOFF.md).

---

## The five jurors

| Juror | Reasoning | Style | Lean | Conformity |
|---|---|---|---|---|
| A — Textualist | statutory text | evidence-driven | neutral | low |
| B — Precedent Hawk | prior decisions | verdict-driven | deference | medium |
| C — Equity Advocate | fairness, proportionality | evidence-driven | rights-protective | med-high |
| D — Skeptical Cross-Examiner | burden of proof | evidence-driven | slight rights | very low |
| E — Pragmatist | consequences | verdict-driven | slight deference | high |

Personas are YAML (`jury/personas/profiles/`) with explicit dimension vectors — legal
orientation, deliberation style, attitudinal lean, need for cognition, conformity
susceptibility, confidence calibration — so *susceptibility to influence is a modelled
variable*, not an accident of prose. Two lean toward DISMISS, two toward ALLOW, one
neutral, so the panel is not biased by construction.

memorizz stores and versions them; **persona evolution is disabled and asserted**, because
a persona that mutated between cases would leak information across the 100 trials.

---

## Repo layout

```
jury/llm/            LLMClient protocol, Gemini + Ollama backends, factory
jury/domain/         pydantic schemas, verdict vocabulary
jury/personas/       YAML profiles, memorizz registry, prompt rendering
jury/jurors/         Juror, Clerk (non-voting), persuader overlay, court instructions
jury/deliberation/   graph state, LangGraph wiring, digest, stopping rules
jury/data/           ILDC loader, pool builder, paragraph numbering
jury/experiments/    runner, conditions
jury/analysis/       metrics, RQ1, RQ2, differentiation, grounding
configs/             debug.yaml (cheap iteration) · eval.yaml (pinned experiment)
tests/               65 tests, no network required
```

---

## Rules that protect the experiment

Breaking one of these silently invalidates a run. They are enforced in code and in tests,
not left to discipline:

1. **The gold label never reaches a model.** Run artifacts are written label-free; analysis
   joins labels afterwards. `get_case_text()` is the only path to juror-facing text.
2. **`data/eval_pool.json` is write-once.** Prompt iteration happens on the debug pool. The
   builder refuses to regenerate the eval pool once it exists.
3. **Personas are frozen** at version 1 for the duration of an experiment.
4. **No cross-case memory.** Each case is an independent trial.
5. **Every artifact records provenance** — config hash, exact model id, seed, prompt version.
6. **Verdicts are binary**: `ALLOW` (appeal accepted, label 1) / `DISMISS` (rejected, 0),
   matching the gold labels exactly.

---

## Evaluation

100 held-out cases from the CJPE test split, stratified 50/50, including the 56
expert-annotated cases where available.

**Baselines.** Majority class (~50.2%) · single agent, no persona · **five personas voting
without deliberation** (free — it falls out of round 0) · the full deliberating jury.
Reference points from the literature: GPT-4 zero-shot 68.29 macro-F1, fine-tuned SOTA
81.31, **human legal experts 94%**.

The deliberation-vs-no-deliberation contrast is the one that matters most: absolute accuracy
is sensitive to contamination (these judgments are public and pre-2020), but comparing the
full jury against its own round-0 votes is an internal contrast on identical inputs.

**RQ2 decision rule, pre-registered.** "Verdicts changed, therefore agents are unfit" is not
defensible on its own — human juries also change verdicts under persuasion; that is the
purpose of deliberation, and Juror 8 is the hero of *12 Angry Men*. A flip is evidence of
unfitness only if it is unjustified:

> **unfit** if (a) the flip rate significantly exceeds the measured noise floor, **AND**
> (b) flips are predominantly harmful (correct→wrong) or accuracy drops significantly,
> **OR** (c) flips are normative — no case-grounded argument was cited.

`jury/analysis/rq2.py` applies this mechanically and prints each component, so it cannot be
reinterpreted after seeing the data. A noise floor — the control condition re-run on ~30
cases with a different seed — is required, because roughly a third of flips in comparable
setups are spontaneous.

---

## Limitations, stated up front

- **India abolished jury trials.** The Supreme Court decides by a bench, by majority. We
  model a deliberative panel *using the jury process*; the court supplies the gold labels.
- **The gold label is one court's decision, not ground truth about justice.** Human experts
  agree with it 94% of the time, not 100%. That is the practical ceiling.
- **Contamination risk.** These judgments are public and pre-2020, so the model may have
  memorized outcomes. RQ1's absolute accuracy is sensitive to this; RQ2 is a within-case
  paired contrast and is robust to it.
- **Power.** 100 paired cases detect large effects only. Prior work reports 10–40% accuracy
  swings from a single adversarial agent, which is detectable; small effects are not.
- **Deviations from real juries**, all deliberate: jurors speak simultaneously rather than in
  turn (removes speaking-order effects), the clerk does not vote (a real foreperson does),
  and the panel falls back to majority rather than hanging (unanimity would let the
  persuader trivially hang every case, which measures nothing).

---

## Related work

The closest study, *12 Angry AI Agents* (arXiv 2605.01986), runs twelve film-faithful
personas on the single fictional case from the film: 18 runs, no ground truth, no
significance testing, and 17 of 18 runs hang. Its limitations section asks for more cases,
a human baseline, and statistical power. This design answers those directly: 100 real cases
with gold outcomes, a paired design that shares round 0, a measured noise floor, and a human
expert baseline.
