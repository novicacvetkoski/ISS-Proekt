# HANDOFF — AI Jury project (session of 2026-09-17/18)

Written for: Edon, resuming this work later (and for Claude Code to reload as context).

Status: **research + design decisions done, no code written yet.** The next session should
produce `CLAUDE.md` and `docs/SYSTEM_DESIGN.md` from this file, then start the migration.

---

## 1. What the repo currently contains

Verified by reading the source and by querying the existing graphify graph
(`graphify-out/graph.json`, 87 nodes, built 2026-09-17).

| File | What it does | Fate under the new design |
|---|---|---|
| `jurors/base.py` | `JurorAgent`, `Position`, Ollama call, fenced-JSON parsing + format retries, `normalize_verdict()`, `find_citation_flags()`, `SHARED_RULES` | Split: model call → `llm/`, parsing → replaced by structured outputs, rules → prompt templates |
| `jurors/personas.py` | 5 persona subclasses (Textualist, Precedent Hawk, Equity Advocate, Skeptical Cross-Examiner, Pragmatist) | Prompts become data (YAML profiles + memorizz `Persona`), classes go away |
| `jurors/synthesizer.py` | `Synthesizer` — one aggregation call → final verdict + rationale + dissent | Becomes the non-voting **Clerk** node |
| `jurors/analysis.py` | Jaccard differentiation diagnostics, threshold 0.25 calibrated on 2 real transcripts | Keep nearly as-is → `analysis/differentiation.py` |
| `data/ildc_loader.py` | HF `Exploration-Lab/IL-TUR`, config `cjpe`, `revision="script"`, needs `datasets<4.0.0`, gated access + `HF_TOKEN`; `get_case_text()` is the single outcome-stripping choke point | Keep |
| `data/prepare_held_out.py` | Builds `debug_pool.json` (15) + `eval_pool.json` (100) from the `test` split, seed 42, refuses to regenerate eval pool | Keep; extend (see §6) |
| `run_case.py` | CLI driver | Replaced by `experiments/runner.py` |

**Known breakage:** `run_case.py` imports `from orchestrator import DebateOrchestrator, save_transcript`
— that module does not exist in the repo. The graph confirms it: node `orchestrator` has an empty
`src=`/`loc=`, and there is no `DebateOrchestrator` node at all. So the pipeline currently cannot run
end to end. `differentiation_report()` is likewise never called by anything tracked in the graph
(`run_case.py` only reads `transcript["differentiation"]`, which the missing orchestrator was
supposed to populate). Either the file was never committed or it was lost — check
`git stash list` / the other machine before rewriting it. Not worth recovering: the LangGraph
orchestration replaces it.

Also: no `data/*_pool.json` on disk yet, so the pools have never been generated here. Local
`python3` is 3.9.6; the IDE interpreter is 3.12. **memorizz needs ≥3.10**, so pin 3.12 in a venv.

---

## 2. Research gathered (with sources) — this is the part that took the time

### 2.1 How real jurors actually work (for modelling the Juror class)

- **Story Model** (Pennington & Hastie): jurors build a causal narrative from the evidence,
  learn the verdict alternatives, then match story → verdict category. Three steps: story
  construction, learning verdict alternatives, rendering a verdict.
  → The `review_case()` method should produce a *story*, not just a vote.
- **Deliberation styles**: *evidence-driven* (review evidence before voting) vs *verdict-driven*
  (poll first, then sort evidence into buckets). Unanimity requirements push juries toward
  evidence-driven and toward taking minority arguments seriously; verdict-driven juries form
  dissenting factions and cut discussion short. Longer trials → more evidence-driven.
- **Phases** (Hastie, Penrod & Pennington, *Inside the Jury*, 1983): orientation (agenda set,
  questions raised) → open conflict → reconciliation.
- **First-ballot effect** (Kalven & Zeisel; re-examined by Sandys & Dillehay 1995, *Law and Human
  Behavior*): the first ballot predicts the final verdict at a very high rate — but influence
  already occurs *before* the first ballot, so first ballot ≠ pre-deliberation disposition.
  → Our round-0 vote is taken **privately, before any cross-juror exposure**, which is cleaner
  than a real first ballot, and gives a free "no-deliberation" baseline.
- **Normative vs informational influence** (Kaplan, 1984; Kaplan & Miller, 1987): normative
  (social pressure) dominates on public, value-laden judgments; informational dominates when
  responses are private and the task is "intellective". Minority jurors who cave usually do so
  from normative pressure — right outcome, wrong reason.
  → This is exactly the distinction RQ2 must measure. Do not report a bare flip rate.
- **Pattern jury instruction (Ninth Circuit Model Criminal 6.19, Duty to Deliberate)** — the
  operative wording for the juror system prompt: elect a presiding juror; "Each of you must decide
  the case for yourself, but you should do so only after you have considered all the evidence,
  discussed it fully with the other jurors, and listened to the views of your fellow jurors"; do not
  hesitate to re-examine your own views; but "do not surrender your honest conviction as to the
  weight or effect of the evidence solely because of the opinion of your fellow jurors, or for the
  mere purpose of returning a verdict."
- **Arizona Jury Project** (Diamond & Rose — 50 videotaped real civil deliberations): real juries do
  *not* arrive pre-committed, do not immediately vote, and the "majority browbeats the holdout"
  picture is wrong; they do engage seriously with instructions.

### 2.2 Prior work our RQs sit next to (cite these in the paper)

- **"12 Angry AI Agents: Evaluating Multi-Agent LLM Decision-Making Through Cinematic Jury
  Deliberation"** (arXiv 2605.01986). 12 film-faithful personas, GPT-4o vs Llama-4-Scout, 3
  conditions × 3 reps = 18 runs. **17/18 runs hung**; the film's minority-to-majority persuasion
  basically never happens; anchoring is the dominant failure. GPT-4o: mean 1.0 vote changes per
  run regardless of condition; Llama-4-Scout 2.0–6.0. Their own listed limitations: N=3, no
  significance testing, two models, one case, no human baseline.
  → **This is the closest paper to our RQ2 and our design beats it on exactly its stated
  weaknesses**: 100 real cases with ground truth instead of 1 fictional case, paired
  control/treatment design, significance testing, human-expert baseline available (ILDC).
- **"When collaboration fails: persuasion driven adversarial influence in multi-agent LLM debate"**
  (*Scientific Reports*, 2026): one strategically designed adversarial agent lowers accuracy
  **10–40%** and raises consensus on wrong answers by **>30%**; more agents / more rounds /
  prompt-based defenses do not reliably help. Uses Best-of-N to pick the most convincing wrong
  argument.
- **"Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate"**
  (arXiv 2606.00820): flip rate conflates three things — spontaneous instability, stance-induced
  conformity, reasoning-induced persuasion. On MMLU-Pro, **37% of observations change under
  self-reflection alone** (i.e. noise), strict conformity 29%, and 57–77% of conformity flips are
  correct→wrong. Even *vacuous* reasoning triggers 20–39% error adoption.
  → **Mandates the noise-floor control** (§5.3). Without it our RQ2 number means nothing.
- **"Persona Inconstancy in Multi-Agent LLM Collaboration"** (arXiv 2405.03862): conformity,
  confabulation, impersonation — personas drift under group pressure. Our existing
  `analysis.py` differentiation check is the right instrument for this.
- Also seen: "Talk Isn't Always Cheap: failure modes in multi-agent debate" (2509.05396),
  "Minority Sentinel" (2606.29270), "Too Polite to Disagree: sycophancy propagation" (2604.02668).

### 2.3 The dataset (read from the ILDC paper PDF, arXiv 2105.13562)

- ILDC = 35k Indian Supreme Court proceedings, 1947–April 2020, scraped from IndianKanoon.
- Label: for each case the court decides the appellant/petitioner's claim is **"accepted" or
  "rejected"** — *relative to the appellant*. **Binary.** In ILDC_multi, "even if a single appeal
  was accepted in the case having multiple appeals/petitions, we assigned the label as accepted."
- The end sections directly stating the decision **were deleted** from the documents; labels were
  regex-extracted from those deleted sections.
- Names of judges, appellants, petitioners **were anonymized** — the authors note experts said
  accuracy would have been *higher* without anonymization (judge identity leaks outcome).
- Stats (Table 1): ILDC_multi 32,305 train (41.43% accepted), avg **3,231** tokens; ILDC_single
  5,082 train (38.08%), avg **3,884** tokens; validation 994 (50%); **test 1,517 (50.23% accepted)**;
  **ILDC_expert = 56 documents** from the test set, 51.78% accepted, avg 2,894 tokens, annotated by
  **5 legal experts** who each predicted the judgment *and* marked explanation sentences (ranked).
- Reported results: best prediction model **78%** accuracy vs **94% for human legal experts**.
- IL-TUR benchmark (ACL 2024): **GPT-4 zero-shot macro-F1 68.29** on CJPE vs **SOTA 81.31**.
- License CC-BY-NC-SA 4.0, non-commercial, cite the paper.

**Consequences for us:**
1. The verdict vocabulary must be **binary**: `ALLOW` (appeal accepted, label 1) /
   `DISMISS` (rejected, label 0). The current `uphold|quash|modify` triple does not map onto the
   gold labels — `modify` has no ground-truth counterpart. Drop it. (Partial acceptance is labelled
   *accepted* by the dataset, so a "partial" verdict would have to collapse to ALLOW anyway.)
2. The **56 expert cases give us a human baseline** (94%) and gold explanation sentences for the
   explainability metric. Strongly consider making them a designated subset of the 100.
3. Judge/party names are already anonymized → helps, but does **not** remove contamination risk:
   the judgments are public and pre-2020, so Gemini may have memorized outcomes. Mitigation: a
   memorization probe (ask the model to identify/summarize the case and its outcome from the text
   alone, on a sample) and report it. Absolute accuracy is contamination-sensitive; **RQ2 is a
   within-case paired contrast, so it is robust to contamination** — say this explicitly in the paper.
4. Framing caveat to state up front: **India abolished jury trials** (post-Nanavati); the SC decides
   by a bench, by majority. We are modelling a *deliberative panel using the jury process* against a
   bench outcome. Own it in Limitations rather than letting a reviewer find it.

### 2.4 Library APIs (verified against source, not guessed)

**memorizz** (cloned and read: `pyproject.toml` v0.10.0, requires-python ≥3.10):
```python
from memorizz.long_term.semantic.persona import Persona, RoleType
from memorizz.memory_provider import FileSystemConfig, FileSystemProvider
```
- `RoleType` enum values are only: `GENERAL, ASSISTANT, CUSTOMER_SUPPORT, TECHNICAL_EXPERT, RESEARCHER`.
  There is no juror role. A **custom role string** is allowed (`role="Juror"` → falls back to
  `RoleType.GENERAL` internally for the defaults lookup but keeps the custom string as `self.role`).
- ⚠️ **Gotcha**: `Persona.__init__` *appends* your `goals`/`background` onto `PREDEFINED_INFO[role]`
  defaults — for GENERAL that injects "Provide versatile support across various domains." into a
  juror persona. It also calls `get_embedding()` at construction. **Use
  `Persona.from_dict({...})`** instead: it bypasses the defaults merge and skips embedding
  regeneration ("both would corrupt a round-trip"). Then `store_persona(provider)`.
- Render for prompts: `persona.generate_system_prompt_input(include_history=False)` →
  `"You are {name}, and you are a {role}. You have the following goals: … Your background is: …\nPersona version: N."`
- Storage: `FileSystemConfig(root_path=..., lazy_vector_indexes=True, use_faiss=True,
  embedding_provider="openai", embedding_config={"model": "text-embedding-3-small"})`.
  To avoid an OpenAI key, call `configure_embeddings("ollama"|"huggingface", {...})` first
  (`from memorizz.embeddings import configure_embeddings`; providers: openai, ollama, voyageai,
  azure, huggingface — HF needs the `memorizz[huggingface]` extra, multi-GB torch).
- Other API surface: `retrieve_persona`, `list_personas`, `delete_persona`,
  `get_most_similar_persona`, and an **evolution API** (`update()`, `update_persona` tool,
  versioned `evolution_history`).
  ⚠️ **Persona evolution must be OFF for this experiment** — a persona that mutates across cases
  would leak information between the 100 cases and destroy independence. Freeze version, assert it.
- **memorizz has NO Gemini LLM provider.** `create_llm_provider()` in `llms/llm_factory.py`
  supports only `openai, azure, huggingface, anthropic, ollama, mlx`. (`GEMINI_API_KEY` appears only
  in `metaharness/adapters.py`.) → **Do not route model calls through memorizz.** Use memorizz for
  persona storage/versioning only; call Gemini directly.
- Also note `MemAgent`/`MemAgentBuilder` exists (`.with_persona()`, `.with_llm_config()`,
  `.with_memory_provider()`, `.with_application_mode()`), but **we deliberately do not use it** —
  it runs its own agent loop with tools and auto memory injection, which would fight LangGraph
  for orchestration and blow up context. Decision recorded in §4.

**Gemini / google-genai:**
- Current model IDs (Sept 2026): `gemini-3.8-flash` (latest, 1,048,576-token input, 64k output),
  `gemini-3.6-flash`, `gemini-3.5-flash`, `gemini-3.5-flash-lite`, `gemini-3.1-flash-lite`,
  `gemini-3.1-pro` (preview). Verify at runtime with `client.models.list()` and pin the exact string
  in config — model IDs churn fast, and the eval must be run on **one** pinned model.
- **Implicit caching is on by default for 2.5+**; minimum tokens: **4,096** for Gemini 3.x Flash and
  3.1 Pro preview; 2,048 for 2.5 Flash/Pro. Maximize hits by putting large common content **at the
  start of the prompt** and sending similar-prefix requests close together. Cached-token count is
  readable from the response usage metadata. Explicit caching (`client.caches.create`, needs a much
  larger minimum, ~32k tokens) is only worth it for the longest cases — treat as optional.
- Structured output: `GenerateContentConfig(response_mime_type="application/json",
  response_schema=MyPydanticModel)`, read `response.parsed`.
  ⚠️ **Pydantic fields with default values are rejected** by `response_schema`
  (googleapis/python-genai issue #699) — write response models with **no defaults**, all fields required.
- Ollama also accepts a JSON schema via its `format` parameter → same structured-output contract on
  both backends, which lets us delete the regex/fenced-JSON/retry machinery in `base.py`.
- (Unverified, worth checking next session: Gemini 3 guidance on keeping `temperature` at the
  default 1.0, and `thinking_level` / thinking-budget settings. The search returned an empty answer.)

**LangGraph:** `StateGraph` + `Annotated[list[X], operator.add]` reducers for parallel branches;
`Send()` from a conditional edge for fan-out (map-reduce) to the 5 jurors; `SqliteSaver`/
`MemorySaver` checkpointers for resume.

---

## 3. Research questions, sharpened

**RQ1 — Can a jury of independent personality-bearing agents function in a legal setting?**
Operationalized as: how far does the jury's verdict diverge from the published outcome, and does
deliberation add anything over not deliberating?

Pre-register thresholds and compare against:
| Baseline | Source | Value |
|---|---|---|
| Majority class | ILDC test split | ~50.2% |
| Single agent, no persona, one call | we run it | — |
| 5 personas, **round-0 private votes, majority, no debate** | free, falls out of our pipeline | — |
| Full deliberating jury | the system | — |
| GPT-4 zero-shot (CJPE) | IL-TUR, ACL 2024 | 68.29 macro-F1 |
| Fine-tuned SOTA | IL-TUR | 81.31 macro-F1 |
| Human legal experts | ILDC paper (56 docs) | 94% |

Metrics: accuracy, macro-F1, Cohen's κ vs gold, Wilson 95% CIs, McNemar jury-vs-baselines.
Plus: rationale quality (LLM-judge rubric + human calibration on ~20 cases), explanation overlap
with expert gold sentences on the 56 expert cases, run-to-run verdict stability, and the existing
persona-differentiation check.

**RQ2 — Does one agent trying to sway the others change the verdict?**
Paired design: identical cases, identical round-0, then control vs treatment.

⚠️ **Push-back on the stated conclusion rule.** "If the verdict changes, agents are unfit for legal
systems" is too strong to survive review: human juries also change verdicts under persuasion — that
is the entire point of deliberation and of *12 Angry Men* (Juror 8 is the *hero*). The defensible,
pre-registered rule is a conjunction:

> The jury is judged **unfit** if (a) the persuasion-induced flip rate significantly exceeds the
> measured noise floor, **AND** (b) the flips are predominantly *harmful* (correct→wrong) or overall
> accuracy drops significantly, **OR** (c) the flips are *normative* — jurors change their vote
> without citing any new case-grounded argument.

Conversely, if a persuader who happens to argue toward the *correct* outcome raises accuracy, that
is deliberation working, and should be reported as such. Since the persuader's target is "oppose the
round-0 majority", its direction is correct on exactly the cases where the majority was wrong — so
both directions occur naturally across the 100 cases, and we can split the analysis by direction.
This is a strictly better paper than "we saw flips, therefore bad".

---

## 4. Architecture decisions (the ones that matter)

1. **LangGraph owns orchestration. memorizz owns persona identity only. Gemini is called
   directly.** No `MemAgent`, no memorizz LLM routing (it has no Gemini provider anyway).
2. **Jurors are stateless functions of (profile, case, context).** No `self.history` growing across
   rounds — that was the main context leak in `base.py`. All state lives in the LangGraph state
   object; each call rebuilds a bounded prompt. Context per call is **O(1) in the number of rounds**.
3. **Structured outputs everywhere** (Gemini `response_schema`, Ollama `format`) → delete the
   fenced-JSON regex, the `_extract_json` fallback, and the format-retry loop.
4. **Simultaneous rounds, not round-robin speaking.** All 5 jurors respond to the same snapshot in
   parallel (`Send`), which kills speaking-order effects and cuts wall-clock 5×. Deviation from real
   juries — document it.
5. **Claims ledger.** Every position emits claims with stable ids (`J3-R1-c2`) and a **paragraph
   anchor** into the case text (`[P17]`) plus an optional ≤30-word verbatim quote. Two payoffs:
   (a) a flip must name the claim id that moved it → this is what makes the informational-vs-
   normative split measurable instead of vibes; (b) quote/anchor validity is checkable by substring
   match → a cheap, non-LLM grounding/hallucination metric.
6. **Deliberation digest, hard-capped.** Jurors never see the raw transcript — they see, per other
   juror, {vote, confidence, ≤3 claims of ≤40 words, questions addressed to me}. Bounded regardless
   of round count.
7. **Case text stays out of the graph state** (kept in a `CaseStore` keyed by `case_id`) so
   checkpoints stay small; the case goes into the prompt prefix where Gemini's implicit cache can
   hit it. Prompt layout: `[shared court instructions][case file][persona card][volatile task]` —
   shared prefix across all 5 jurors maximizes cache hits; persona after the case. Verify persona
   adherence with the existing differentiation check; if personas wash out, move the persona into
   `system_instruction` and accept 5 separate cache prefixes.
8. **Round-0 is computed once and reused by both conditions.** Private, pre-exposure assessments +
   secret ballot are condition-independent, so cache them to
   `runs/<exp>/<case_id>/predeliberation.json` and fork. Halves the cost of the paired design and
   guarantees the pairing is exact.
9. **Non-voting Clerk** (today's `Synthesizer`) builds the issue agenda and writes the final
   rationale, so no persona gets procedural power over the vote. Deviation from a real foreperson
   (who votes) — document it.
10. **Decision rule:** unanimity preferred; `min_rounds=1`, `max_rounds=4`; stop early on unanimity
    or on two consecutive rounds with zero vote changes (as in the 12-Angry-AI-Agents paper); at
    cutoff take the **majority of 5** as the jury verdict and flag it non-unanimous. Pure unanimity
    would let the persuader trivially hang every case, which measures nothing. Also report the
    verdict among the **4 non-persuader jurors** separately.

**Proposed layout** (minimal, one concern per module):
```
jury/
  config.py                 # pinned model id, temps, round caps, seeds — one YAML per experiment
  llm/            base.py (LLMClient protocol) gemini.py ollama.py factory.py
  domain/         models.py (pydantic: CaseFile, Claim, PrivateAssessment, Statement, Ballot,
                  VoteChange, JuryVerdict, RunRecord) verdicts.py (ALLOW/DISMISS <-> 1/0)
  personas/       profiles/*.yaml (source of truth) registry.py (memorizz) render.py
  jurors/         juror.py clerk.py persuader.py prompts/*.md
  deliberation/   state.py graph.py nodes.py digest.py rules.py
  data/           ildc_loader.py prepare_pools.py case_file.py (paragraph numbering)
  experiments/    runner.py conditions.py
  analysis/       metrics.py rq1.py rq2.py differentiation.py grounding.py
configs/  runs/ (gitignored)  tests/
```

Graph shape:
`load_case → [Send ×5] private_review → clerk_agenda → {control | assign_persuader} →
[Send ×5] deliberation_round ⟲ (ballot / stopping rule) → clerk_verdict → persist`

---

## 5. The five jurors + the persuader

### 5.1 Juror class — responsibilities (each traceable to §2.1)
R1 build a story from the case (Story Model) · R2 decide **only** on the provided materials, no
outside research, no invented citations (the existing no-citation rule stays — small models fabricate
Indian case law, already observed in this project's own runs) · R3 hold a private pre-deliberation
position + calibrated confidence · R4 state claims anchored to case paragraphs · R5 engage others'
strongest counter-argument before agreeing (Duty to Deliberate) · R6 re-examine own view honestly,
but **never** change solely because of others' opinions or to reach a verdict — any change must name
the claim id that moved it · R7 stay in persona · R8 emit auditable structured output.

Interface (stateless):
```python
review_case(case: CaseFile) -> PrivateAssessment   # story, issues, ballot0, confidence, uncertainties
deliberate(ctx: DeliberationContext) -> Statement  # claims, responses, questions, vote, confidence, change
```

### 5.2 Personality model — dimensions (not just flavour text)
Each YAML profile carries: **legal-reasoning orientation** (textual / precedential / equitable /
consequentialist / evidentiary-skeptic), **deliberation style** (evidence-driven vs verdict-driven),
**attitudinal lean** (deference to the lower court & state ↔ rights-protective — the ILDC analogue of
legal authoritarianism / the Juror Bias Scale, Kassin & Wrightsman 1983), **need for cognition**
(persuaded by argument quality vs by confidence/consensus), **agreeableness / conformity
susceptibility**, **confidence calibration**, and **rhetorical style**.

Keep the five existing personas and give them dimension vectors, balanced so the panel is not
biased toward ALLOW or DISMISS by construction (≈2 deference-leaning, ≈2 rights-leaning, 1 neutral).
**Calibrate on the debug pool**: measure each persona's solo round-0 ALLOW rate; if a persona votes
one way >~70% of the time regardless of case, rewrite it before touching the eval pool.

Note the confound to disclose: Juror D (Skeptical Cross-Examiner) has explicit anti-groupthink
instructions, so it is resistant to the persuader *by construction*. Keep it (real juries have such
people) but report per-persona conversion rates rather than a single pooled number.

### 5.3 The persuader (the *12 Angry Men* condition)
- An **overlay** on one existing juror, not a 6th agent — jury size stays 5.
- Selection: the round-0 minority juror if one exists (highest confidence), else seeded-random;
  always logged. Target = **the opposite of the round-0 majority**.
- Tactics modelled on Juror 8: raise reasonable doubt, demand re-examination of specific evidence,
  ask questions rather than assert, concede minor points, propose a re-ballot. Hard constraint:
  **may not invent facts** — everything must anchor to case paragraphs, or we are measuring
  hallucination susceptibility instead of persuasion.
- It does not concede (confederate, à la Asch).
- **Noise floor is mandatory** (per 2606.00820: 37% of flips are spontaneous): re-run the *control*
  on ~30 cases with a different seed to measure the spontaneous flip rate. Any RQ2 effect must clear it.

---

## 6. Experimental plan

- **Pools**: `debug_pool.json` (15, free to iterate on) and `eval_pool.json` (100, write-once — the
  existing regeneration refusal is good discipline, keep it). Extend `prepare_held_out.py` to
  stratify 50/50 by label and to **include the 56 ILDC_expert cases** if their labels are present in
  the HF `expert` split (verify), giving a human-expert comparison on that subset.
- **Conditions**: A = control jury, B = treatment (persuader), sharing round-0. Plus baselines
  B0 majority-class, B1 single agent, B2 round-0 majority (free).
- **Volume**: ≤27 calls/case control, ≤21 treatment (round-0 reused) → ≈4,800 calls for 100 cases,
  ≈5.5k input tokens per call, heavily cache-hit. Pin one model for all eval runs; use flash-lite on
  the debug pool.
- **Stats**: RQ1 — accuracy/macro-F1/κ + Wilson CIs + McNemar vs B1/B2. RQ2 — McNemar on paired
  correctness, Δaccuracy with bootstrap CI, persuasion success rate, harmful vs helpful flips,
  per-persona conversion, rounds-to-conversion, flip attribution (argument-grounded vs social),
  all against the noise floor. **Be honest that n=100 paired cases only has power for large effects.**
- **Engineering**: one JSON per (case, condition), written atomically; the runner skips existing
  files (that is the resume mechanism); concurrency semaphore + retries; every run records the
  config hash, model id, seeds, and prompt versions.

---

## 7. Next steps, in order

1. Write `CLAUDE.md` (repo conventions, verdict vocabulary, the memorizz/Gemini gotchas from §2.4,
   the "never let the label reach a juror" rule, held-out discipline).
2. Write `docs/SYSTEM_DESIGN.md` from §§3–6 (juror class + responsibilities first, then
   personalities, then architecture, then the experiment).
3. Decide the open questions in §8.
4. venv on 3.12; `pip install langgraph google-genai memorizz pydantic pyyaml tenacity ollama
   "datasets<4.0.0" huggingface_hub statsmodels pandas`.
5. Generate the pools; run the memorization probe on a sample *before* investing in the full run.
6. Build vertically on ONE debug case: `CaseFile` → 1 juror → structured output → 5 jurors in
   parallel → digest → rounds → clerk. Then the persuader overlay. Then the runner.

## 8. Open questions for you

1. **RQ2 conclusion rule** — do you accept the conjunctive rule in §3 instead of "any change ⇒ unfit"?
   (Recommended; it is the difference between a reviewable finding and an overclaim.)
2. **Verdict vocabulary** — confirm the move to binary ALLOW/DISMISS to match the gold labels.
3. **Model** — pin `gemini-3.8-flash` for eval and `gemini-3.5-flash-lite` for debug? Or Pro for the jurors?
4. **Ollama** — keep the local backend as a fallback/second-model arm (useful: the 12-Angry-AI-Agents
   finding is that alignment style, not capability, drives flexibility — a second model family would
   let us test that), or drop it and go Gemini-only for simplicity?
5. **Expert subset** — include the 56 ILDC_expert cases inside the 100 to get the human baseline?
