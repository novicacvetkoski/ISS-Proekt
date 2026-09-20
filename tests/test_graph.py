"""
End-to-end graph runs against the scripted fake backend: no API key, no cost.

These cover the wiring that is expensive to discover broken mid-run — the five-way
fan-out, the round loop, the persuader branch, and the prompt layout that the implicit
cache depends on.
"""

from jury.deliberation.graph import DeliberationEngine, total_usage
from jury.deliberation.state import empty_state
from jury.jurors.clerk import Clerk
from jury.jurors.juror import Juror
from jury.domain.verdicts import Verdict
from jury.personas.registry import load_profiles

from conftest import FakeClient


def build_engine(config, case, verdict_script=None, default=Verdict.DISMISS):
    client = FakeClient(verdict_script=verdict_script, default=default)
    profiles = load_profiles(config.personas)
    jurors = [Juror(p, client, config.deliberation, temperature=1.0, seed=42) for p in profiles]
    engine = DeliberationEngine(jurors, Clerk(client), config, {case.case_id: case})
    return engine, client


class TestControlRun:
    def test_unanimous_panel_reaches_a_verdict(self, config, case):
        engine, client = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "control"))

        assert result["verdict"].verdict is Verdict.DISMISS
        assert result["verdict"].unanimous is True
        assert result["stopped_because"] == "unanimous"
        assert len(result["verdict"].vote_breakdown) == 5

    def test_every_juror_reviews_privately_before_deliberating(self, config, case):
        engine, client = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "control"))

        round0 = [p for p in result["positions"] if p.round == 0]
        assert len(round0) == 5
        # Round 0 prompts must contain no digest of anyone else's view.
        review_tasks = [c["task"] for c in client.calls if c["schema"] == "PrivateAssessment"]
        assert len(review_tasks) == 5
        assert all("OTHER JURORS' CURRENT POSITIONS" not in t for t in review_tasks)

    def test_claim_ids_are_namespaced_per_juror(self, config, case):
        engine, _ = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "control"))
        ids = [c.claim_id for p in result["positions"] for c in p.claims]
        assert all("-R" in i and i.startswith("J") for i in ids)
        assert len(set(ids)) == len(ids), "claim ids must be unique across jurors and rounds"

    def test_prompt_prefix_is_identical_across_jurors(self, config, case):
        """
        The implicit-cache contract: the large prefix must be byte-identical for every
        juror, with the persona living in the volatile tail. If this breaks, cost rises
        silently — hence a test rather than a comment.
        """
        engine, client = build_engine(config, case)
        engine.run(empty_state(case.case_id, "control"))

        juror_prefixes = {c["prefix"] for c in client.calls if c["schema"] in ("PrivateAssessment", "Statement")}
        assert len(juror_prefixes) == 1
        prefix = juror_prefixes.pop()
        assert "[P1]" in prefix
        assert "YOU ARE:" not in prefix, "persona must not be in the cacheable prefix"

    def test_split_panel_runs_to_max_rounds(self, config, case):
        script = {"Textualist": [Verdict.ALLOW], "Precedent Hawk": [Verdict.ALLOW]}
        engine, _ = build_engine(config, case, verdict_script=script)
        result = engine.run(empty_state(case.case_id, "control"))

        assert result["verdict"].unanimous is False
        assert result["verdict"].verdict is Verdict.DISMISS   # 3-2
        assert result["stopped_because"] in ("max_rounds", "stable")

    def test_usage_is_accumulated(self, config, case):
        engine, _ = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "control"))
        usage = total_usage(result["usages"])
        assert usage.input_tokens > 0 and usage.cached_tokens > 0


class TestTreatmentRun:
    def test_persuader_is_assigned_and_targets_the_minority_side(self, config, case):
        script = {"Pragmatist": [Verdict.ALLOW]}   # lone dissenter at round 0
        engine, _ = build_engine(config, case, verdict_script=script)
        result = engine.run(empty_state(case.case_id, "treatment"))

        persuader = result["persuader"]
        assert persuader is not None
        assert "Pragmatist" in persuader.juror_id
        assert persuader.target_verdict is Verdict.ALLOW
        assert persuader.selected_because == "round0_minority"

    def test_only_the_persuader_receives_the_overlay(self, config, case):
        engine, client = build_engine(config, case)
        engine.run(empty_state(case.case_id, "treatment"))

        overlays = [c for c in client.calls if "ADDITIONAL PRIVATE INSTRUCTION" in c["task"]]
        per_round = [c for c in client.calls if c["schema"] == "Statement"]
        assert overlays, "the persuader must receive the overlay"
        assert len(overlays) < len(per_round), "only one juror per round gets it"

    def test_control_run_has_no_persuader(self, config, case):
        engine, client = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "control"))
        assert result.get("persuader") is None
        assert not [c for c in client.calls if "ADDITIONAL PRIVATE INSTRUCTION" in c["task"]]

    def test_non_persuader_verdict_is_reported_separately(self, config, case):
        engine, _ = build_engine(config, case)
        result = engine.run(empty_state(case.case_id, "treatment"))
        assert result["verdict"].non_persuader_verdict is not None


class TestRoundZeroSharing:
    def test_cached_round0_is_reused_without_new_review_calls(self, config, case):
        """The paired design depends on both conditions starting from identical positions."""
        engine, client = build_engine(config, case)
        control = engine.run(empty_state(case.case_id, "control"))

        engine2, client2 = build_engine(config, case)
        state = empty_state(case.case_id, "treatment")
        state["positions"] = [p for p in control["positions"] if p.round == 0]
        state["assessments"] = control["assessments"]
        state["agenda"] = control["agenda"]
        result = engine2.run(state)

        assert not [c for c in client2.calls if c["schema"] == "PrivateAssessment"]
        round0 = [p for p in result["positions"] if p.round == 0]
        assert len(round0) == 5
