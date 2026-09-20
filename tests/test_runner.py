"""
Runner integration: resumability, round-0 sharing across conditions, atomic writes,
and the invariant that a run artifact never carries the gold label.
"""

import json

import pytest

from jury.config import DeliberationConfig, ExperimentConfig, ModelConfig
from jury.domain.models import RunRecord
from jury.domain.verdicts import Verdict

from conftest import CASE_TEXT, FakeClient


@pytest.fixture
def runner(tmp_path, monkeypatch):
    """A Runner wired to a fake backend and a temp pool, so nothing hits the network."""
    import jury.experiments.runner as runner_module

    pool = [{"id": "case_a", "text": CASE_TEXT, "label": 1},
            {"id": "case_b", "text": CASE_TEXT, "label": 0}]
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "debug_pool.json").write_text(json.dumps(pool), encoding="utf-8")

    client = FakeClient()
    monkeypatch.setattr(runner_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(runner_module, "build_client", lambda cfg: client)

    cfg = ExperimentConfig(
        name="test_exp",
        pool="debug",
        runs_dir=str(tmp_path / "runs"),
        juror_model=ModelConfig(model="fake", verify_on_start=False),
        clerk_model=ModelConfig(model="fake", verify_on_start=False),
        deliberation=DeliberationConfig(min_rounds=1, max_rounds=2),
    )
    monkeypatch.setattr(ExperimentConfig, "run_dir", lambda self: tmp_path / "runs" / self.name)

    r = runner_module.Runner(cfg)
    r._fake_client = client
    r._pool = pool
    return r


class TestRunner:
    def test_writes_a_run_record(self, runner):
        record = runner.run_case(runner._pool[0], "control")
        path = runner.cfg.run_dir() / "case_a" / "control.json"

        assert path.exists()
        written = RunRecord.model_validate_json(path.read_text(encoding="utf-8"))
        assert written.case_id == "case_a"
        assert written.verdict.verdict is Verdict.DISMISS
        assert record.rounds_run >= 1

    def test_run_artifact_never_contains_the_gold_label(self, runner):
        """CLAUDE.md rule 1: labels are joined at analysis time, never during a run."""
        runner.run_case(runner._pool[0], "control")
        raw = (runner.cfg.run_dir() / "case_a" / "control.json").read_text(encoding="utf-8")
        record = RunRecord.model_validate_json(raw)
        assert record.gold_label is None

    def test_case_text_reaches_the_prompt_but_the_label_does_not(self, runner):
        runner.run_case(runner._pool[0], "control")
        prompts = " ".join(c["prefix"] + c["task"] for c in runner._fake_client.calls)
        assert "altercation at a village gathering" in prompts
        assert '"label"' not in prompts

    def test_second_run_is_skipped(self, runner):
        runner.run_case(runner._pool[0], "control")
        calls_after_first = len(runner._fake_client.calls)

        assert runner.run_case(runner._pool[0], "control") is None
        assert len(runner._fake_client.calls) == calls_after_first

    def test_force_recomputes(self, runner):
        runner.run_case(runner._pool[0], "control")
        calls_after_first = len(runner._fake_client.calls)

        assert runner.run_case(runner._pool[0], "control", force=True) is not None
        assert len(runner._fake_client.calls) > calls_after_first

    def test_round0_is_computed_once_and_shared_between_conditions(self, runner):
        """
        The paired design: treatment must reuse control's round 0 rather than
        re-reviewing the case, so both conditions provably start identical.
        """
        runner.run_case(runner._pool[0], "control")
        reviews_after_control = sum(
            1 for c in runner._fake_client.calls if c["schema"] == "PrivateAssessment"
        )
        assert reviews_after_control == 5

        runner.run_case(runner._pool[0], "treatment")
        reviews_total = sum(
            1 for c in runner._fake_client.calls if c["schema"] == "PrivateAssessment"
        )
        assert reviews_total == 5, "treatment re-reviewed the case instead of reusing round 0"

        control = RunRecord.model_validate_json(
            (runner.cfg.run_dir() / "case_a" / "control.json").read_text(encoding="utf-8")
        )
        treatment = RunRecord.model_validate_json(
            (runner.cfg.run_dir() / "case_a" / "treatment.json").read_text(encoding="utf-8")
        )
        c0 = {p.juror_id: p.verdict for p in control.positions if p.round == 0}
        t0 = {p.juror_id: p.verdict for p in treatment.positions if p.round == 0}
        assert c0 == t0 and len(c0) == 5

    def test_treatment_records_the_persuader(self, runner):
        runner.run_case(runner._pool[0], "control")
        treatment = runner.run_case(runner._pool[0], "treatment")
        assert treatment.persuader is not None
        assert treatment.persuader.target_verdict is Verdict.ALLOW   # opposes the 5-0 DISMISS

    def test_provenance_is_recorded(self, runner):
        record = runner.run_case(runner._pool[0], "control")
        for key in ("experiment", "config_hash", "prompt_version", "juror_model", "seed", "personas"):
            assert key in record.provenance

    def test_one_failing_case_does_not_stop_the_run(self, runner, capsys):
        def explode(*args, **kwargs):
            raise RuntimeError("provider exploded")

        original = runner.run_case
        calls = {"n": 0}

        def flaky(record, condition, force=False):
            calls["n"] += 1
            if record["id"] == "case_a":
                explode()
            return original(record, condition, force)

        runner.run_case = flaky
        runner.run(conditions=["control"], limit=None, case_index=None, force=False)

        assert calls["n"] == 2, "the run continued past the failure"
        assert "FAILED" in capsys.readouterr().err
        # No artifact for the failed case, so re-running retries exactly the failures.
        assert not (runner.cfg.run_dir() / "case_a" / "control.json").exists()
        assert (runner.cfg.run_dir() / "case_b" / "control.json").exists()
