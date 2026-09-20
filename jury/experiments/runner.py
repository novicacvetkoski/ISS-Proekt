"""
Experiment runner.

Resumability is file-based on purpose: one JSON per (case, condition), written
atomically, skipped if present. A crashed run resumes by re-running the same command,
and a single bad case is recomputed by deleting its file. That is simpler to reason
about than checkpoint surgery, and it survives changing the code between runs.

ROUND-0 SHARING is the other job here. Private review is condition-independent, so it is
computed once per case, cached, and replayed into both conditions. Halves cost, and makes
the control/treatment pairing exact rather than approximate — the two conditions provably
start from identical juror positions.

Usage:
    python -m jury.experiments.runner --config configs/debug.yaml
    python -m jury.experiments.runner --config configs/eval.yaml --condition control
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

from ..config import ExperimentConfig, load_config
from ..data.case_file import approx_tokens, build_case_file
from ..data.ildc_loader import get_case_text
from ..deliberation.graph import DeliberationEngine, total_usage
from ..deliberation.rules import classify_vote_changes
from ..deliberation.state import empty_state
from ..domain.models import CaseFile, Position, PrivateAssessment, RunRecord
from ..jurors.clerk import Clerk
from ..jurors.juror import Juror
from ..llm.factory import build_client
from ..personas.registry import load_profiles

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def load_pool(pool: str) -> list[dict]:
    path = DATA_DIR / f"{pool}_pool.json"
    if not path.exists():
        sys.exit(
            f"{path} not found. Build it first:\n"
            f"  python -m jury.data.prepare_pools --debug-size 15 --eval-size 100"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def write_atomic(path: Path, payload: str) -> None:
    """Write via a temp file + rename so a crash cannot leave a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(payload)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


class Runner:
    def __init__(self, cfg: ExperimentConfig):
        self.cfg = cfg
        self.profiles = load_profiles(cfg.personas)
        self.juror_client = build_client(cfg.juror_model)
        self.clerk_client = build_client(cfg.clerk_model)
        self.case_store: dict[str, CaseFile] = {}

        self.jurors = [
            Juror(
                profile=p,
                client=self.juror_client,
                cfg=cfg.deliberation,
                temperature=cfg.juror_model.temperature,
                seed=cfg.seed,
            )
            for p in self.profiles
        ]
        self.clerk = Clerk(self.clerk_client, seed=cfg.seed)
        self.engine = DeliberationEngine(self.jurors, self.clerk, cfg, self.case_store)

    # -- round-0 cache ---------------------------------------------------- #

    def _round0_path(self, case_id: str) -> Path:
        return self.cfg.run_dir() / case_id / "predeliberation.json"

    def _load_round0(self, case_id: str) -> dict | None:
        path = self._round0_path(case_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {
            "positions": [Position.model_validate(p) for p in raw["positions"]],
            "assessments": [
                (jid, PrivateAssessment.model_validate(a)) for jid, a in raw["assessments"]
            ],
            "agenda": raw["agenda"],
        }

    def _save_round0(self, case_id: str, state: dict) -> None:
        payload = {
            "positions": [p.model_dump(mode="json") for p in state["positions"] if p.round == 0],
            "assessments": [
                (jid, a.model_dump(mode="json")) for jid, a in state["assessments"]
            ],
            "agenda": state["agenda"],
        }
        write_atomic(self._round0_path(case_id), json.dumps(payload, indent=2, ensure_ascii=False))

    # -- one case --------------------------------------------------------- #

    def run_case(self, record: dict, condition: str, force: bool = False) -> RunRecord | None:
        case_id = str(record["id"])
        out_path = self.cfg.run_dir() / case_id / f"{condition}.json"
        if out_path.exists() and not force:
            print(f"  [{case_id}/{condition}] already done — skipping")
            return None

        # get_case_text() is the ONLY sanctioned path to juror-facing text; it never
        # touches record["label"]. The label below is read for scoring only, after the run.
        case = build_case_file(case_id, get_case_text(record))
        self.case_store[case_id] = case

        state = empty_state(case_id, condition)
        cached = self._load_round0(case_id)
        if cached:
            state.update(cached)
            print(f"  [{case_id}/{condition}] reusing shared round 0")

        result = self.engine.run(state)

        if not cached:
            self._save_round0(case_id, result)

        changes = classify_vote_changes(result["positions"], self.engine.short_ids)
        record_out = RunRecord(
            case_id=case_id,
            condition=condition,
            experiment=self.cfg.name,
            verdict=result["verdict"],
            gold_label=None,            # analysis joins the label; runs stay label-free
            positions=result["positions"],
            ballots=[],
            vote_changes=changes,
            persuader=result.get("persuader"),
            agenda=result.get("agenda", []),
            rounds_run=result.get("round", 0),
            stopped_because=result.get("stopped_because", ""),
            provenance={
                **self.cfg.provenance(),
                "condition": condition,
                "personas": [p.id for p in self.profiles],
                "case_approx_tokens": approx_tokens(case),
            },
            usage_total=total_usage(result.get("usages", [])),
        )
        write_atomic(out_path, record_out.model_dump_json(indent=2))

        verdict = record_out.verdict
        print(
            f"  [{case_id}/{condition}] {verdict.verdict.value if verdict.verdict else 'HUNG'} "
            f"({'unanimous' if verdict.unanimous else 'split'}), "
            f"{record_out.rounds_run} rounds, stop={record_out.stopped_because}, "
            f"{len(changes)} flip(s), "
            f"{record_out.usage_total.cached_tokens} cached tokens"
        )
        return record_out

    def run(self, conditions: list[str], limit: int | None, case_index: int | None, force: bool):
        pool = load_pool(self.cfg.pool)
        if case_index is not None:
            pool = [pool[case_index]]
        elif limit:
            pool = pool[:limit]

        if self.cfg.pool == "eval":
            print(
                "NOTE: running on the HELD-OUT eval pool. Do not tune prompts from what you "
                "see here — that is what the debug pool is for.",
                flush=True,
            )

        for i, record in enumerate(pool, 1):
            print(f"[{i}/{len(pool)}] case {record['id']}", flush=True)
            for condition in conditions:
                try:
                    self.run_case(record, condition, force=force)
                except Exception as e:
                    # One bad case must not end a 100-case run. No result file is written,
                    # so re-running the command retries exactly the failures.
                    print(f"  [{record['id']}/{condition}] FAILED: {e}", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--condition", action="append", dest="conditions",
                        choices=["control", "treatment"],
                        help="Defaults to the conditions listed in the config")
    parser.add_argument("--limit", type=int, help="Only the first N cases of the pool")
    parser.add_argument("--case-index", type=int, help="Run exactly one case by pool index")
    parser.add_argument("--force", action="store_true", help="Recompute results that already exist")
    args = parser.parse_args()

    cfg = load_config(args.config)
    runner = Runner(cfg)
    runner.run(
        conditions=args.conditions or cfg.conditions,
        limit=args.limit,
        case_index=args.case_index,
        force=args.force,
    )


if __name__ == "__main__":
    main()
