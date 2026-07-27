"""Tests for shared structured-review output normalization."""

from openmc_agent.plan_builder.closed_loop.models import (
    AxialGeometryReviewModelOutput,
    PlanFindingCategory,
    PlanFindingSeverity,
)
from openmc_agent.plan_builder.closed_loop.review_io import normalize_llm_review_candidate


def test_review_normalizer_accepts_finding_code_alias_and_severity_case() -> None:
    candidate = {
        "review_status": "COMPLETE",
        "findings": [
            {
                "finding_code": "axial.test",
                "severity": "ERROR",
                "category": "physical_ambiguity",
                "message": "test",
                "evidence_refs": ["F001"],
            }
        ],
    }

    normalized = normalize_llm_review_candidate(candidate, AxialGeometryReviewModelOutput)
    parsed = AxialGeometryReviewModelOutput.model_validate(normalized)

    assert parsed.review_status == "complete"
    assert parsed.findings[0].code == "axial.test"
    assert parsed.findings[0].severity is PlanFindingSeverity.ERROR
    assert parsed.findings[0].category is PlanFindingCategory.PHYSICAL_AMBIGUITY


def test_review_normalizer_accepts_issue_code_alias_and_blocking_severity() -> None:
    candidate = {
        "review_status": "complete",
        "findings": [
            {
                "issue_code": "axial.blocker",
                "severity": "blocking",
                "message": "test",
                "evidence_refs": ["F001"],
            }
        ],
    }

    normalized = normalize_llm_review_candidate(candidate, AxialGeometryReviewModelOutput)
    parsed = AxialGeometryReviewModelOutput.model_validate(normalized)

    assert parsed.findings[0].code == "axial.blocker"
    assert parsed.findings[0].severity is PlanFindingSeverity.ERROR
