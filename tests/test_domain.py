"""Schema and verdict-vocabulary invariants."""

import pytest

from jury.data.case_file import build_case_file
from jury.domain import models
from jury.domain.verdicts import Verdict, majority_verdict, normalize_verdict


class TestVerdicts:
    def test_binary_vocabulary(self):
        assert [v.value for v in Verdict] == ["allow", "dismiss"]

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("allow", Verdict.ALLOW),
            ("ALLOW", Verdict.ALLOW),
            ("the appeal is accepted", Verdict.ALLOW),
            ("quash the conviction", Verdict.ALLOW),
            ("dismiss", Verdict.DISMISS),
            ("uphold the conviction", Verdict.DISMISS),
            ("", None),
            (None, None),
            ("wobble", None),
        ],
    )
    def test_normalize(self, raw, expected):
        assert normalize_verdict(raw) is expected

    def test_majority_and_unanimity(self):
        a, d = Verdict.ALLOW, Verdict.DISMISS
        assert majority_verdict([a, a, a, d, d]) == (a, False)
        assert majority_verdict([d, d, d, d, d]) == (d, True)
        # Even split must be explicit, never silently defaulted.
        assert majority_verdict([a, a, d, d]) == (None, False)
        assert majority_verdict([]) == (None, False)


class TestResponseSchemas:
    """
    Gemini rejects response schemas whose fields carry defaults
    (googleapis/python-genai#699). This test is the guard against someone
    "just adding a default" and breaking every juror call at runtime.
    """

    RESPONSE_SCHEMAS = [
        models.Claim,
        models.PrivateAssessment,
        models.ClaimResponse,
        models.Statement,
        models.ClerkAgenda,
        models.ClerkVerdict,
    ]

    @pytest.mark.parametrize("schema", RESPONSE_SCHEMAS)
    def test_no_defaults_in_response_schemas(self, schema):
        defaulted = [name for name, f in schema.model_fields.items() if not f.is_required()]
        assert not defaulted, (
            f"{schema.__name__} has default values on {defaulted}. Gemini's response_schema "
            "rejects defaults — use `X | None` and pass the value explicitly instead."
        )


class TestCaseFile:
    def test_paragraph_ids_are_sequential_and_anchorable(self, case):
        assert [p.pid for p in case.paragraphs] == [f"P{i}" for i in range(1, len(case.paragraphs) + 1)]
        assert "P1" in case.pids
        assert case.text_of("P1") is not None
        assert case.text_of("P999") is None

    def test_render_includes_anchors(self, case):
        rendered = case.render()
        assert "[P1]" in rendered and "[P2]" in rendered

    def test_short_paragraphs_are_merged(self):
        case = build_case_file("x", "Short.\n\n" + "word " * 60 + "\n\nAlso short.")
        assert all(len(p.text.split()) >= 5 for p in case.paragraphs)

    def test_empty_text_is_rejected(self):
        with pytest.raises(ValueError):
            build_case_file("x", "   ")
