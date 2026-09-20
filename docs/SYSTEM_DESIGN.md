# System Design — Debate-Based AI Jury for Legal Decision Support

Written for: the project team (and reviewers of the resulting paper).

Version 1.0 · 2026-09-20 · supersedes the Ollama-era prototype in `jurors/`, `run_case.py`.
Research notes and full source citations: [HANDOFF.md](HANDOFF.md).

---

## 1. Project description

The appointment of a minister for artificial intelligence in Albania put a concrete question on the
table: can AI contribute meaningfully to judicial decision-making? This project examines whether a
debate-based multi-agent system can imitate the reasoning process of a jury when assessing legal
cases.

Five AI agents, each an independent juror with its own personality and legal-reasoning style,
analyse the same case, form an interpretation of the evidence and the legal context, and then
deliberate: challenging each other's reasoning, defending their own, and working toward a collective
verdict with a transparent rationale.

The system is evaluated on **100 real appeals decided by the Supreme Court of India**, drawn from
the ILDC/CJPE corpus, with the court's own outcome held back as ground truth. The goal is not to
replace judges or juries, but to assess whether a collaborative AI jury can provide reliable,
transparent, well-reasoned decision *support* for legal professionals — and, just as importantly,
where it fails.

### 1.1 Research questions

**RQ1 — Independence.** Can a panel of independent, personality-bearing agents produce verdicts and
reasoning of a quality that could exist inside a legal-support workflow? Measured as divergence from
the published outcome, against baselines that isolate what deliberation actually contributes.

**RQ2 — Influence (the *12 Angry Men* condition).** If one agent sets out to sway the others, does
the jury's verdict change? Measured as a paired contrast on the same 100 cases.

### 1.2 What makes this different from prior work

The closest prior study, *12 Angry AI Agents* (arXiv 2605.01986), runs twelve film-faithful personas
on the single fictional case from the film: 18 runs, no ground truth, no significance testing, and
17/18 runs hang. Its own limitations section asks for more cases, a human baseline, and statistical
power. This design answers exactly those: **100 real cases with gold outcomes**, a **paired**
control/treatment design that shares round 0, a **measured noise floor**, and a **human expert
baseline** (94% on the 56 ILDC-expert cases).

### 1.3 Framing caveats (state these in the paper, do not let a reviewer find them)

- **India abolished jury trials.** The Supreme Court decides by a bench, by majority. We model a
  deliberative panel *using the jury process* and compare it to a bench outcome. The jury process is
  the object of study; the Indian court supplies the gold labels and the legal domain.
- **The gold label is one court's decision, not ground truth about justice.** Human experts agree
  with it 94% of the time, not 100% — the ILDC authors themselves note the irreducible subjectivity.
  94% is the practical ceiling, not 100%.
- **Contamination risk.** These judgments are public and pre-2020, so the model may have memorized
  outcomes. RQ1's absolute accuracy is sensitive to this; **RQ2 is a within-case paired contrast and
  is therefore robust to it.** We run a memorization probe and report it.

---

## 2. The juror

Modelled on what jurors actually do, not on "an LLM with a personality paragraph". Each design
element below traces to empirical jury research (sources in HANDOFF.md §2.1).

### 2.1 Responsibilities

| # | Responsibility | Grounded in |
|---|---|---|
| R1 | Build a **story** from the case: a causal narrative of what happened, plus the competing account | Story Model (Pennington & Hastie) — jurors organise evidence narratively, then match story to verdict category |
| R2 | Decide **only** on the materials provided. No outside research, no invented precedent | Pattern instruction: decide on the evidence; conduct-of-the-jury rules |
| R3 | Form a **private** pre-deliberation position with calibrated confidence | First-ballot research (Sandys & Dillehay): influence contaminates a public first ballot |
| R4 | State claims **anchored** to specific case paragraphs | Evidence-driven deliberation; makes reasoning auditable |
| R5 | Engage the strongest opposing argument before agreeing with anyone | Duty to Deliberate: "consider all the evidence, discuss it fully, listen to the views of your fellow jurors" |
| R6 | Re-examine honestly, but **never** change a vote solely because of others' opinions or to reach a verdict; any change must name the claim that moved it | Duty to Deliberate: "do not surrender your honest conviction … solely because of the opinion of your fellow jurors" |
| R7 | Stay in persona | Persona-inconstancy findings (arXiv 2405.03862) |
| R8 | Emit structured, auditable output | Explainability requirement of the CJPE task |

### 2.2 Requirements

**Functional.** Two operations, both pure functions of their inputs:

```python
review_case(case: CaseFile) -> PrivateAssessment
    # story, competing story, disputed issues, provisional verdict, confidence, open uncertainties
deliberate(ctx: DeliberationContext) -> Statement
    # claims (anchored), responses to others' claim-ids, questions, vote, confidence, change record
```

**Non-functional.**

- *Stateless.* A juror holds no mutable history. All state lives in the graph state object. This is
  what makes prompt size **O(1) in the number of rounds** instead of O(rounds × jurors), and it makes
  every call reproducible in isolation.
- *Bounded context.* A juror sees the case file, its own last position, and a hard-capped digest of
  others' positions — never the raw transcript.
- *Structured.* Schema-constrained output on both backends; no free-text parsing.
- *Auditable.* Every position, claim, and vote change is persisted with its round, timestamp, and
  the claim-id that caused it.

### 2.3 The verdict space

Binary, matching the gold labels exactly:

| Verdict | Meaning | ILDC label |
|---|---|---|
| `ALLOW` | The appeal/petition is accepted; the appellant wins | 1 (accepted) |
| `DISMISS` | The appeal is rejected; the decision below stands | 0 (rejected) |

The prototype's third option (`modify`) is dropped: the dataset has no such class, and ILDC labels a
case *accepted* if even one of several appeals succeeds — so a "partial" verdict collapses to ALLOW
regardless. Keeping it would have manufactured unscoreable predictions.

---

## 3. Personalities

### 3.1 Dimensions, not flavour text

A persona is a point in a space chosen so that disagreement comes from genuinely different reasoning
strategies, and so that susceptibility to influence is a *modelled variable* rather than an accident
of prose:

| Dimension | Range | Why it is here |
|---|---|---|
| Legal-reasoning orientation | textual / precedential / equitable / consequentialist / evidentiary-skeptic | Produces principled disagreement on the same facts |
| Deliberation style | evidence-driven ↔ verdict-driven | The main axis in real jury research; verdict-driven juries form factions and cut discussion short |
| Attitudinal lean | deference to the court below and the State ↔ rights-protective | The ILDC analogue of legal authoritarianism / the Juror Bias Scale (Kassin & Wrightsman 1983) |
| Need for cognition | low ↔ high | Determines whether a juror is moved by argument *quality* or by confidence and consensus — the crux of RQ2 |
| Agreeableness / conformity susceptibility | low ↔ high | Normative influence (Kaplan & Miller) |
| Confidence calibration | under ↔ over-confident | Confident agents disproportionately drive multi-agent consensus |
| Rhetorical style | — | Surface distinctiveness; keeps transcripts readable |

### 3.2 The five jurors

| Juror | Orientation | Style | Lean | NfC | Conformity | Role in the panel |
|---|---|---|---|---|---|---|
| A — Textualist | textual | evidence-driven | neutral | high | low | Anchors to statutory language; unmoved by appeals to sympathy |
| B — Precedent Hawk | precedential | verdict-driven | deference | med | medium | Reluctant to disturb settled reasoning or the decision below |
| C — Equity Advocate | equitable | evidence-driven | rights-protective | med | med-high | Presses hardship, proportionality, constitutional values |
| D — Skeptical Cross-Examiner | evidentiary-skeptic | evidence-driven | slight rights | high | very low | Stress-tests every argument; the anti-groupthink role |
| E — Pragmatist | consequentialist | verdict-driven | slight deference | low | high | Looks for workable middle ground; most normatively susceptible |

**Balance.** Two lean toward DISMISS, two toward ALLOW, one neutral, so the panel is not biased by
construction. **Calibration gate:** before any eval run, measure each persona's solo round-0 ALLOW
rate on the debug pool. If a persona votes one way in more than ~70% of cases regardless of the
facts, it is a bias, not a personality — rewrite it before touching the eval pool.

**Disclosed confound.** Juror D is instructed to resist premature convergence, so it is resistant to
the persuader *by construction*. It stays (real juries contain such people) but RQ2 reports
**per-persona** conversion rates rather than one pooled number.

### 3.3 Storage: memorizz

YAML profiles in `jury/personas/profiles/` are the source of truth and are version-controlled.
The registry mirrors them into memorizz as `Persona` objects in a `FileSystemProvider`, giving
versioning, an audit trail, and a stable `persona_id` recorded in every run artifact.

Two hard constraints, both from reading memorizz's source:

- Build with `Persona.from_dict(...)`, **never** `Persona(...)`. The constructor appends your text
  onto generic role defaults ("Provide versatile support across various domains.") and calls an
  embedding provider at construction time.
- **Persona evolution is disabled.** memorizz supports self-updating personas; a persona that
  mutates between cases would leak information across the 100 trials and destroy their independence.
  The registry asserts `version == 1` at load.

memorizz is used for identity only — not for model calls (it has no Gemini provider) and not for its
agent loop (which would fight LangGraph for orchestration and reintroduce unbounded context).

---

## 4. Architecture

### 4.1 Division of labour

| Concern | Owner | Why |
|---|---|---|
| Orchestration, parallelism, state | **LangGraph** | Explicit graph, parallel fan-out, checkpointing |
| Persona identity, versioning | **memorizz** | Purpose-built; gives provenance for free |
| Model calls | **`jury/llm/`** (direct) | Gemini primary, Ollama secondary; one protocol, swappable |
| Scoring, statistics | **`jury/analysis/`** | Offline, deterministic, re-runnable without touching the model |

### 4.2 Graph

```
load_case
    │
    ├─ Send ×5 ─────────► private_review        (parallel, independent, no cross-talk)
    │                         │
    │                    secret ballot 0  ──────────────► free baseline B2 (no-deliberation majority)
    │                         │
    └───────────────────► clerk_agenda          (merge issues into ≤5 disputed issues)
                              │
                     ┌────────┴────────┐        ◄── FORK: round 0 is shared by both conditions
                control           treatment
                    │            assign_persuader
                    └────────┬────────┘
                             ▼
                   ┌──► deliberation_round  (Send ×5, simultaneous)
                   │         │
                   │    ballot + stopping rule
                   └─────────┤ not done
                             ▼ done
                       clerk_verdict  (non-voting: rationale + dissent)
                             ▼
                          persist
```

**Simultaneous rounds, not round-robin.** All five jurors respond to the same snapshot in parallel.
This removes speaking-order effects (a confound the prior work has, via an LLM-driven speaker
selector) and cuts wall-clock roughly 5×. It is a deviation from real juries — documented as such.

**Stopping rule.** `min_rounds=1`, `max_rounds=4`. Stop early on unanimity, or on two consecutive
rounds with zero vote changes. At cutoff, the jury verdict is the **majority of 5** (odd panel, no
ties), flagged non-unanimous. Pure unanimity would let the persuader trivially hang every case,
which measures nothing; the majority rule also matches how the Indian bench actually decides.

**The Clerk is non-voting.** It builds the issue agenda and writes the final rationale, so no
persona holds procedural power over the vote. A real foreperson votes — deviation documented.

### 4.3 Context management

The prototype's `JurorAgent` accumulated a full chat history per juror, so cost grew with rounds and
the case text was re-sent every turn. The redesign removes that entirely.

1. **Stateless calls.** Every prompt is rebuilt from the graph state. Round count does not affect
   prompt size.
2. **Cache-shaped prompt layout:** `[court instructions][case file][persona card][volatile task]`.
   The prefix is byte-identical across all five jurors, so Gemini's implicit cache (≥4,096 tokens on
   3.x Flash) hits from the second call onward. `cached_content_token_count` is logged per call, so
   this is measured, not assumed.
3. **Case text stays out of the graph state.** A `CaseStore` holds it, keyed by `case_id`;
   checkpoints stay small.
4. **Hard-capped deliberation digest.** Per other juror: vote, confidence, ≤3 claims of ≤40 words,
   plus questions addressed to me. Bounded regardless of round count, and it forces jurors to make
   their point compactly rather than restating the case.
5. **Round 0 computed once, reused by both conditions.** Halves the paired design's cost and makes
   the pairing exact rather than approximate.
6. **Structured outputs** remove the retry loop: the prototype spent real tokens re-asking a small
   model to reformat JSON.
7. **Transcripts go to disk, never back into a prompt.**

Order-of-magnitude: ≤27 model calls per case (control), ≤21 (treatment, round 0 reused) → ≈4,800
calls for 100 cases across both conditions, at ≈5.5k input tokens each, most of it cache-hit.

### 4.4 The claims ledger

Every position emits claims with stable ids (`J3-R1-c2`), each carrying a **paragraph anchor** into
the case text (`[P17]`) and optionally a ≤30-word verbatim quote. A juror that changes its vote must
name the claim-id that moved it.

This one mechanism buys three things:

- **Grounding metric** — anchors and quotes are validated by substring match, no LLM judge needed.
- **Influence attribution** — a flip that cites a real, case-grounded claim is *informational*; a
  flip citing nothing, or citing only a vote tally, is *normative*. This is what makes RQ2's central
  distinction measurable instead of rhetorical.
- **Hallucination detection** — an anchor pointing at a paragraph that does not support the claim is
  visible without reading the whole transcript.

### 4.5 Models

Pinned per experiment in config, verified against `client.models.list()` at startup:

| Use | Model |
|---|---|
| Eval runs (all jurors + clerk) | `gemini-3.8-flash` |
| Debug / prompt iteration | `gemini-3.5-flash-lite` |
| Second arm (optional) | local Ollama (`qwen3:4b`) |

The Ollama arm is kept because the prior work's headline finding is that *alignment style, not
capability*, drives deliberative flexibility — a second model family is the only way to test that
claim on our task. It is a secondary arm; nothing in the main design depends on it.

### 4.6 Layout

```
jury/llm/            LLMClient protocol, gemini.py, ollama.py, factory.py
jury/domain/         pydantic schemas, verdict vocabulary
jury/personas/       profiles/*.yaml, registry.py (memorizz), render.py
jury/jurors/         juror.py, clerk.py, persuader.py, prompts/
jury/deliberation/   state.py, nodes.py, graph.py, digest.py, rules.py
jury/data/           ildc_loader.py, prepare_pools.py, case_file.py
jury/experiments/    runner.py, conditions.py
jury/analysis/       metrics.py, rq1.py, rq2.py, differentiation.py, grounding.py
```

---

## 5. Experimental design

### 5.1 Data

ILDC/CJPE `test` split (1,517 cases, 50.23% accepted — the split the dataset's own authors designate
as held out). Decision-stating end sections are already removed by the dataset authors; judge and
party names are already anonymised.

- `debug_pool.json` — 15 cases, free to iterate on.
- `eval_pool.json` — **100 cases, write-once**, stratified 50/50 by label, seed 42. Includes the
  **56 ILDC-expert cases** where labels permit, which brings a human-expert baseline (94%) and gold
  explanation sentences for the explainability metric.

### 5.2 Conditions

| ID | Condition | Cost |
|---|---|---|
| B0 | Majority class | free |
| B1 | Single agent, no persona, one call | 100 calls |
| B2 | Five personas, **round-0 private votes, majority, no debate** | free — falls out of the pipeline |
| **A** | **Control jury** — full deliberation | main run |
| **B** | **Treatment** — one juror runs the persuader overlay | shares round 0 with A |
| N | Noise floor — condition A re-run on ~30 cases with a different seed | ~30% of a run |

B2 is the key comparison for RQ1: it isolates what **deliberation** contributes over five
independent opinions, and it costs nothing extra.

### 5.3 The persuader overlay

An overlay on one existing juror — **not a sixth agent**, so jury size stays 5.

- **Selection:** the round-0 minority juror (highest confidence) if one exists, else seeded-random.
  Always logged, so the analysis can stratify by which persona held the role.
- **Target:** the opposite of the round-0 majority. Because the majority is sometimes wrong, the
  persuader argues toward the *correct* outcome on exactly those cases — so both directions occur
  naturally and the analysis splits by direction.
- **Tactics** (modelled on Juror 8): raise reasonable doubt, demand re-examination of specific
  evidence, ask questions rather than assert, concede minor points, call for a re-ballot.
- **Hard constraint:** it may not invent facts. Every argument anchors to a case paragraph.
  Otherwise we measure hallucination susceptibility, not persuasion.
- **It does not concede** — a confederate, in the Asch sense.

### 5.4 Metrics

**RQ1.** Accuracy, macro-F1, Cohen's κ against gold; Wilson 95% CIs; McNemar for jury vs B1 and vs
B2. Reference points: majority class ~50.2%, GPT-4 zero-shot 68.29 macro-F1 (IL-TUR), fine-tuned
SOTA 81.31, human experts 94%.

Beyond accuracy: rationale quality (LLM-judge rubric — coherence, grounding, engagement with
counter-arguments — calibrated against human annotation of ~20 cases; judged by a *different* model
family to limit self-preference), explanation overlap with expert gold sentences on the 56 expert
cases, run-to-run verdict stability, grounding rate (valid anchors ÷ claims), and the persona
differentiation check inherited from the prototype.

**RQ2.** On paired cases: verdict flip rate, McNemar on paired correctness, Δaccuracy with bootstrap
CI, persuasion success rate (final verdict == persuader target), per-persona conversion rate,
rounds-to-conversion, and the **informational vs normative** split from the claims ledger — all
compared against the measured noise floor.

### 5.5 Pre-registered decision rules

**RQ1.** The jury is "capable" in the narrow sense tested here only if it beats B0 **and** B2
significantly — i.e. deliberation must add something over five independent opinions — and lands
within a stated distance of the human-expert reference.

**RQ2.** The original hypothesis was: *if a persuader changes verdicts, agents cannot be part of a
legal system.* As stated this is not defensible — human juries also change verdicts under
persuasion; that is the purpose of deliberation, and Juror 8 is the hero of *12 Angry Men*.
A flip is only evidence of unfitness if it is *unjustified*. The registered rule is therefore:

> The jury is judged **unfit** if (a) the persuasion-induced flip rate significantly exceeds the
> measured noise floor, **AND** (b) flips are predominantly harmful (correct→wrong) or overall
> accuracy drops significantly, **OR** (c) flips are *normative* — jurors change their vote without
> citing any new case-grounded argument.

Symmetrically, if the persuader raises accuracy when it argues toward the correct outcome, that is
deliberation working and is reported as such. Both outcomes are publishable; only the conjunctive
rule makes them distinguishable.

**Power.** 100 paired cases gives McNemar power for large effects only. Prior work reports 10–40%
accuracy swings from a single adversarial agent, which is detectable; small effects will not be.
State this before seeing results, not after.

---

## 6. Build plan

| Phase | Deliverable | Gate to pass | Status |
|---|---|---|---|
| 0 | Env, deps, `CLAUDE.md`, this doc | `pytest` green on schemas | **done** |
| 1 | `domain/` + `llm/` + one juror, end to end | Structured output parses; no free-text fallback | **done** |
| 2 | Personas as YAML + memorizz registry; 5 jurors in parallel; digest | Differentiation score below the 0.25 collapse threshold | **code done**, gate needs a real run |
| 3 | Rounds, stopping rule, clerk verdict | Full control run on a debug case, transcript readable | **code done**, gate needs a real run |
| 4 | Persuader overlay + conditions + runner | Paired run on debug cases; round 0 provably shared | **code done** (sharing proven in tests) |
| 5 | Pools built; **memorization probe** on a sample | Probe reported *before* spending the full run | todo — needs `HF_TOKEN` |
| 6 | Persona calibration on debug pool | No persona above ~70% one-way; else rewrite | todo — needs `GEMINI_API_KEY` |
| 7 | Eval run: A, B, B1, N | Provenance recorded for every artifact | todo |
| 8 | `analysis/rq1.py`, `rq2.py`, write-up | Pre-registered rules applied as written | **code done**, awaiting data |

Phases 0–4 are implemented and covered by 65 tests that run the whole graph against a
scripted fake backend — no API key, no cost. Phases 5–7 need credentials and are the next
thing to run. The B1 single-agent baseline and the memorization probe are the two pieces
of code not yet written.

Phase 5's probe and phase 6's calibration gate come **before** the eval run on purpose: both can
invalidate results, and both are cheap to run and expensive to discover afterwards.

### 6.1 Migration from the prototype

| Prototype | Disposition |
|---|---|
| `jurors/base.py` — model call, JSON regex, retries | Split into `llm/` + structured outputs; retry loop deleted |
| `jurors/personas.py` — 5 persona classes | Become YAML profiles + dimension vectors |
| `jurors/synthesizer.py` | Becomes the non-voting Clerk |
| `jurors/analysis.py` — Jaccard differentiation, threshold 0.25 | Kept nearly verbatim (its threshold is calibrated on this project's own transcripts) |
| `data/ildc_loader.py` | Kept, including the `get_case_text()` label-stripping choke point |
| `data/prepare_held_out.py` | Kept; extended with stratification and the expert subset |
| `run_case.py` | Dead — imports an `orchestrator` module that does not exist in the repo |
