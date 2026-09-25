"""
B1 — single-agent baseline.

The roadmap's Phase 6 (and RQ1's own framing in analysis/rq1.py) names the comparison
that actually isolates the debate mechanism: does a panel of five personas, arguing
across rounds, do anything a single model call reasoning alone wouldn't already do?
B2 (round0_majority in analysis/metrics.py) answers half of that for free — five
independent personas, no debate, majority vote. B1 is the other half: not just "no
debate" but no PANEL at all — one agent, no persona framing, reasoning about the same
case materials in a single call.

Deliberately NOT a persona. NEUTRAL_PROFILE is built inline here rather than added to
personas/profiles/ as a selectable YAML, because it must never be pickable via
--persona alongside the five real ones — doing so would let it leak into a real jury
run by accident. dimensions={} so render_persona_card() emits no disposition section:
this agent should read as "no persona applied", not as a sixth, blandly-written persona.

Reuses Juror.review_case() and position_from_assessment() directly rather than
reimplementing the call — B1's one call has to go through the exact same prompt
machinery the panel's round-0 call uses (COURT_INSTRUCTIONS, case prefix, REVIEW_TASK),
or the comparison confounds "single agent" with "differently-prompted agent". Same
juror_model, same temperature, same seed as the real panel, for the same reason:
isolate the debate/panel mechanism, not the model or the prompt.

RunRecord.condition="single_agent" was already anticipated in domain/models.py, so
existing analysis code needs zero changes: metrics.load_runs(run_dir, "single_agent")
works today, unmodified, and lands in the SAME run_dir as the main experiment.

Usage:
    python -m jury.experiments.single_agent --config configs/eval.yaml
    python -m jury.experiments.single_agent --config configs/debug.yaml --limit 3
"""

import argparse
import sys

from ..config import ExperimentConfig, load_config
from ..data.case_file import approx_tokens, build_case_file
from ..data.ildc_loader import get_case_text
from ..deliberation.graph import total_usage
from ..domain.models import JuryVerdict, RunRecord
from ..jurors.juror import Juror, position_from_assessment
from ..llm.factory import build_client
from ..personas.registry import JurorProfile
from .runner import load_pool, write_atomic

NEUTRAL_PROFILE = JurorProfile(
    id="single_agent",
    name="Independent Reviewer",
    role="judge",
    version=1,
    dimensions={},
    goals="Reach the legally correct outcome on the materials given.",
    background=(
        "You are reviewing this case alone, with no panel and no persona to play. "
        "Reason plainly and directly from the case materials and applicable law."
    ),
    conduct="",
)


def run_one(cfg: ExperimentConfig, juror: Juror, record: dict, force: bool) -> RunRecord | None:
    case_id = str(record["id"])
    out_path = cfg.run_dir() / case_id / "single_agent.json"
    if out_path.exists() and not force:
        print(f"  [{case_id}/single_agent] already done — skipping")
        return None

    # get_case_text() is the ONLY sanctioned path to juror-facing text, same as runner.py.
    case = build_case_file(case_id, get_case_text(record))
    assessment, usage = juror.review_case(case)
    position = position_from_assessment(NEUTRAL_PROFILE, assessment, usage)

    verdict = JuryVerdict(
        verdict=position.verdict,
        unanimous=True,               # trivially true at n=1 -- not a claim of agreement
        vote_breakdown={position.juror_id: position.verdict},
        rationale=assessment.story,   # the agent's own reasoning; no clerk synthesis exists here
        dissent_summary=None,
        non_persuader_verdict=None,
    )
    record_out = RunRecord(
        case_id=case_id,
        condition="single_agent",
        experiment=cfg.name,
        verdict=verdict,
        gold_label=None,              # analysis joins the label, same as every other condition
        positions=[position],
        ballots=[],
        vote_changes=[],
        persuader=None,
        agenda=[],
        rounds_run=0,
        stopped_because="single_agent — no deliberation by design",
        provenance={
            **cfg.provenance(),
            "condition": "single_agent",
            "case_approx_tokens": approx_tokens(case),
        },
        usage_total=total_usage([usage]),
    )
    write_atomic(out_path, record_out.model_dump_json(indent=2))
    print(f"  [{case_id}/single_agent] {position.verdict.value} (confidence {position.confidence:.2f})")
    return record_out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="Only the first N cases of the pool")
    parser.add_argument("--force", action="store_true", help="Recompute results that already exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pool = load_pool(cfg.pool)
    if args.limit:
        pool = pool[: args.limit]

    if cfg.pool == "eval":
        print(
            "NOTE: running on the HELD-OUT eval pool. Do not tune prompts from what you "
            "see here — that is what the debug pool is for.",
            flush=True,
        )

    client = build_client(cfg.juror_model)
    juror = Juror(
        profile=NEUTRAL_PROFILE,
        client=client,
        cfg=cfg.deliberation,   # unused by review_case(); required by Juror's constructor only
        temperature=cfg.juror_model.temperature,
        seed=cfg.seed,
    )

    for i, record in enumerate(pool, 1):
        print(f"[{i}/{len(pool)}] case {record['id']}", flush=True)
        try:
            run_one(cfg, juror, record, force=args.force)
        except Exception as e:
            print(f"  [{record['id']}/single_agent] FAILED: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
