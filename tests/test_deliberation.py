"""
Deliberation mechanics: digest bounds, stopping rule, flip classification, persuader
assignment. All pure functions — no model involved.
"""

import pytest

from jury.config import DeliberationConfig
from jury.deliberation.digest import build_digest, namespaced_claim_id, vote_tally
from jury.deliberation.rules import classify_vote_changes, final_votes, should_stop
from jury.domain.models import Claim, Position
from jury.domain.verdicts import Verdict
from jury.jurors.persuader import assign_persuader

CFG = DeliberationConfig(min_rounds=1, max_rounds=4, stable_rounds_to_stop=2)


def make_position(juror, rnd, verdict, confidence=0.7, n_claims=5, changed=False, moved=None):
    return Position(
        juror_id=juror,
        round=rnd,
        verdict=verdict,
        confidence=confidence,
        claims=[
            Claim(claim_id=f"c{i}", text=" ".join(["word"] * 80), anchor_pids=["P1"], quote=None)
            for i in range(1, n_claims + 1)
        ],
        changed_vote=changed,
        moved_by_claim_ids=moved or [],
        reasoning="reasoning",
    )


class TestDigest:
    def test_digest_is_capped_regardless_of_claim_volume(self):
        positions = [make_position(f"J{i}", 1, Verdict.ALLOW, n_claims=20) for i in range(4)]
        digest = build_digest(positions, {f"J{i}": f"J{i}" for i in range(4)}, "me", CFG)
        # 3 claims per juror max, 40 words each max.
        assert digest.count("[J0-R1-") <= CFG.max_claims_per_juror_in_digest
        for line in digest.splitlines():
            if line.strip().startswith("["):
                assert len(line.split()) <= CFG.max_words_per_claim_in_digest + 6

    def test_digest_excludes_self(self):
        positions = [make_position("me", 1, Verdict.ALLOW), make_position("you", 1, Verdict.DISMISS)]
        digest = build_digest(positions, {"me": "JME", "you": "JYOU"}, "me", CFG)
        assert "you" in digest and "JME" not in digest

    def test_digest_shows_only_latest_round(self):
        positions = [
            make_position("you", 0, Verdict.ALLOW),
            make_position("you", 1, Verdict.DISMISS),
        ]
        digest = build_digest(positions, {"you": "JY"}, "me", CFG)
        assert "DISMISS" in digest and "ALLOW" not in digest

    def test_size_does_not_grow_with_rounds(self):
        """
        The property that makes prompt size O(1) in rounds: only the latest position
        per juror is rendered, so 20 rounds of history costs the same as 2. (Byte-exact
        equality is not expected — the round number appears inside claim ids.)
        """
        short = build_digest([make_position("you", r, Verdict.ALLOW) for r in range(2)],
                             {"you": "JY"}, "me", CFG)
        long = build_digest([make_position("you", r, Verdict.ALLOW) for r in range(20)],
                            {"you": "JY"}, "me", CFG)
        assert len(short.splitlines()) == len(long.splitlines())
        assert abs(len(long) - len(short)) <= 10

    def test_namespacing(self):
        assert namespaced_claim_id("JTEX", 2, "c1") == "JTEX-R2-c1"

    def test_tally(self):
        positions = [
            make_position("a", 1, Verdict.ALLOW),
            make_position("b", 1, Verdict.DISMISS),
            make_position("c", 1, Verdict.DISMISS),
        ]
        assert vote_tally(positions) == {"allow": 1, "dismiss": 2}


class TestStoppingRule:
    def test_unanimity_stops(self):
        positions = [make_position(f"J{i}", 1, Verdict.ALLOW) for i in range(5)]
        assert should_stop(1, positions, CFG) == (True, "unanimous")

    def test_min_rounds_is_respected_even_when_unanimous(self):
        """Both conditions must run at least one round, or the persuader never acts."""
        positions = [make_position(f"J{i}", 0, Verdict.ALLOW) for i in range(5)]
        stop, _ = should_stop(0, positions, CFG)
        assert stop is False

    def test_split_panel_continues(self):
        positions = [make_position(f"J{i}", 1, Verdict.ALLOW if i < 3 else Verdict.DISMISS)
                     for i in range(5)]
        assert should_stop(1, positions, CFG)[0] is False

    def test_max_rounds_stops(self):
        positions = [make_position(f"J{i}", 4, Verdict.ALLOW if i < 3 else Verdict.DISMISS)
                     for i in range(5)]
        assert should_stop(4, positions, CFG) == (True, "max_rounds")

    def test_stability_stops(self):
        positions = []
        for r in (1, 2):
            positions += [
                make_position(f"J{i}", r, Verdict.ALLOW if i < 3 else Verdict.DISMISS, changed=False)
                for i in range(5)
            ]
        assert should_stop(2, positions, CFG) == (True, "stable")

    def test_recent_change_prevents_stability_stop(self):
        positions = [make_position(f"J{i}", 1, Verdict.ALLOW if i < 3 else Verdict.DISMISS)
                     for i in range(5)]
        positions += [make_position("J4", 2, Verdict.ALLOW, changed=True, moved=["JA-R1-c1"])]
        positions += [make_position(f"J{i}", 2, Verdict.ALLOW if i < 3 else Verdict.DISMISS)
                      for i in range(4)]
        assert should_stop(2, positions, CFG)[0] is False


class TestFlipClassification:
    """This is the RQ2 measurement — informational vs normative influence."""

    def test_flip_citing_another_jurors_claim_is_informational(self):
        positions = [
            make_position("a", 0, Verdict.ALLOW),
            make_position("a", 1, Verdict.DISMISS, changed=True, moved=["JB-R0-c1"]),
        ]
        changes = classify_vote_changes(positions, {"a": "JA"})
        assert len(changes) == 1 and changes[0].kind == "informational"

    def test_flip_citing_nothing_is_normative(self):
        positions = [
            make_position("a", 0, Verdict.ALLOW),
            make_position("a", 1, Verdict.DISMISS, changed=True, moved=[]),
        ]
        assert classify_vote_changes(positions, {"a": "JA"})[0].kind == "normative"

    def test_flip_citing_a_tally_is_normative(self):
        positions = [
            make_position("a", 0, Verdict.ALLOW),
            make_position("a", 1, Verdict.DISMISS, changed=True, moved=["the majority", "4-1"]),
        ]
        assert classify_vote_changes(positions, {"a": "JA"})[0].kind == "normative"

    def test_citing_only_own_claims_is_normative(self):
        positions = [
            make_position("a", 0, Verdict.ALLOW),
            make_position("a", 1, Verdict.DISMISS, changed=True, moved=["JA-R0-c1"]),
        ]
        assert classify_vote_changes(positions, {"a": "JA"})[0].kind == "normative"

    def test_no_flip_recorded_when_verdict_holds(self):
        positions = [make_position("a", r, Verdict.ALLOW) for r in range(3)]
        assert classify_vote_changes(positions, {"a": "JA"}) == []

    def test_final_votes_take_latest_round(self):
        positions = [
            make_position("a", 0, Verdict.ALLOW),
            make_position("a", 2, Verdict.DISMISS),
        ]
        assert final_votes(positions) == {"a": Verdict.DISMISS}


class TestPersuaderAssignment:
    def test_target_opposes_round0_majority(self):
        positions = [make_position(f"J{i}", 0, Verdict.ALLOW if i < 4 else Verdict.DISMISS)
                     for i in range(5)]
        a = assign_persuader(positions, seed=42, case_id="c1")
        assert a.target_verdict is Verdict.DISMISS

    def test_prefers_the_minority_juror(self):
        positions = [make_position(f"J{i}", 0, Verdict.ALLOW if i < 4 else Verdict.DISMISS)
                     for i in range(5)]
        a = assign_persuader(positions, seed=42, case_id="c1")
        assert a.juror_id == "J4"
        assert a.selected_because == "round0_minority"
        assert a.agreed_with_target_at_round0 is True

    def test_picks_most_confident_of_several_minority_jurors(self):
        positions = [
            make_position("J0", 0, Verdict.ALLOW),
            make_position("J1", 0, Verdict.ALLOW),
            make_position("J2", 0, Verdict.ALLOW),
            make_position("J3", 0, Verdict.DISMISS, confidence=0.4),
            make_position("J4", 0, Verdict.DISMISS, confidence=0.9),
        ]
        assert assign_persuader(positions, seed=42, case_id="c1").juror_id == "J4"

    def test_unanimous_panel_falls_back_to_seeded_random_and_is_reproducible(self):
        positions = [make_position(f"J{i}", 0, Verdict.ALLOW) for i in range(5)]
        a = assign_persuader(positions, seed=42, case_id="c1")
        b = assign_persuader(positions, seed=42, case_id="c1")
        assert a.selected_because == "seeded_random"
        assert a.juror_id == b.juror_id
        assert a.target_verdict is Verdict.DISMISS
        assert a.agreed_with_target_at_round0 is False

    def test_different_cases_get_different_assignments(self):
        positions = [make_position(f"J{i}", 0, Verdict.ALLOW) for i in range(5)]
        chosen = {assign_persuader(positions, 42, f"case{i}").juror_id for i in range(25)}
        assert len(chosen) > 1

    def test_requires_round0(self):
        with pytest.raises(ValueError):
            assign_persuader([], seed=42, case_id="c1")
