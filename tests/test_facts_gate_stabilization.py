"""Phase 8B Step 3 — Facts Gate stabilization tests.

Covers:

* Deterministic evidence↔Facts consistency checks (Step 2).
* Review stage splitting (Step 1).
* Repair coverage completeness (Step 3).
* Reviewer output stability — empty response, free-text approve (Step 4).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest

from openmc_agent.plan_builder.closed_loop.facts_evidence_consistency import (
    check_facts_evidence_consistency,
    FactsEvidenceConsistencyResult,
)
from openmc_agent.plan_builder.closed_loop.facts_review_stages import (
    FactsReviewStage,
    STAGE_ORDER,
    STAGE_FIELD_MAP,
    extract_facts_subset,
    build_stage_review_prompt,
    FactsReviewStageRequest,
)
from openmc_agent.plan_builder.closed_loop.facts_revision import (
    check_facts_repair_completeness,
    REQUIRED_COVERAGE_PATHS,
)
from openmc_agent.plan_builder.closed_loop.facts_reviewer import (
    _classify_review_failure,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight evidence claim stand-in
# ---------------------------------------------------------------------------


@dataclass
class _FakeClaim:
    """Minimal stand-in for EvidenceClaim used by the consistency checker."""

    claim_id: str
    subject: str = ""
    predicate: str = ""
    value: Any = None


def _claim(cid: str, text: str, predicate: str = "search_hit") -> _FakeClaim:
    return _FakeClaim(claim_id=cid, subject="source", predicate=predicate, value={"text": text})


# ---------------------------------------------------------------------------
# Step 2: deterministic evidence↔Facts consistency
# ---------------------------------------------------------------------------


class TestEvidenceConsistency:
    """Tests for facts_evidence_consistency.check_facts_evidence_consistency."""

    def test_full_core_evidence_with_single_assembly_facts_rejects(self):
        """Evidence indicates multi-assembly but Facts says single_assembly."""
        claims = [
            _claim("c1", "The model covers a 3x3 core lattice."),
            _claim("c2", "Full core physics benchmark with multiple assemblies."),
        ]
        facts = {"model_scope": "single_assembly"}
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        assert not result.ok
        codes = [f.code for f in result.findings]
        assert "facts.scope_evidence_conflict" in codes

    def test_rcca_evidence_with_empty_inserts_rejects(self):
        """Evidence mentions RCCA/control rods but Facts has no inserts."""
        claims = [
            _claim("c1", "RCCA control rods are inserted from the top."),
            _claim("c2", "Pyrex rods are used as burnable poison."),
        ]
        facts = {"localized_insert_requirements": []}
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        assert not result.ok
        codes = [f.code for f in result.findings]
        assert "facts.localized_insert_missing" in codes

    def test_fuel_variants_evidence_with_empty_variants_rejects(self):
        """Evidence mentions multiple enrichments but Facts has no variants."""
        claims = [
            _claim("c1", "Fuel enrichment of 3.1 wt% U-235."),
            _claim("c2", "Different enrichment levels for specific assemblies."),
        ]
        facts = {"fuel_variant_requirements": []}
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        assert not result.ok
        codes = [f.code for f in result.findings]
        assert "facts.fuel_variant_missing" in codes

    def test_grid_evidence_with_false_grid_rejects(self):
        """Evidence mentions spacer grids but Facts says has_spacer_grids=False."""
        claims = [
            _claim("c1", "The assembly includes spacer grids at regular intervals."),
        ]
        facts = {"has_spacer_grids": False}
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        assert not result.ok
        codes = [f.code for f in result.findings]
        assert "facts.grid_feature_missing" in codes

    def test_consistent_facts_and_evidence_passes(self):
        """When Facts matches evidence, no findings are produced."""
        claims = [
            _claim("c1", "3x3 core with multiple assemblies."),
            _claim("c2", "Spacer grids are present."),
            _claim("c3", "Control rods are used."),
        ]
        facts = {
            "model_scope": "multi_assembly_core",
            "has_spacer_grids": True,
            "localized_insert_requirements": [{"requirement_id": "cr1"}],
            "fuel_variant_requirements": [{"variant_id": "v1"}],
        }
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        assert result.ok
        assert result.findings == []

    def test_empty_evidence_produces_no_findings(self):
        """No evidence → no consistency checks can fire."""
        result = check_facts_evidence_consistency(
            facts_patch={"model_scope": "single_assembly"},
            evidence_claims=[],
        )
        assert result.ok

    def test_findings_carry_evidence_claim_ids(self):
        """Each finding should reference the claim IDs that triggered it."""
        claims = [_claim("c_specific", "3x3 full core lattice layout.")]
        facts = {"model_scope": "single_assembly"}
        result = check_facts_evidence_consistency(
            facts_patch=facts, evidence_claims=claims
        )
        scope_findings = [f for f in result.findings if f.code == "facts.scope_evidence_conflict"]
        assert scope_findings
        assert "c_specific" in scope_findings[0].evidence_claim_ids

    def test_to_issue_dicts_format(self):
        """Findings should be convertible to the gate's issue dict format."""
        result = check_facts_evidence_consistency(
            facts_patch={"model_scope": "single_assembly"},
            evidence_claims=[_claim("c1", "3x3 core")],
        )
        dicts = result.to_issue_dicts()
        assert isinstance(dicts, list)
        for d in dicts:
            assert "code" in d
            assert d["severity"] == "error"
            assert d["blocking"] is True
            assert d["owner_patch_type"] == "facts"


# ---------------------------------------------------------------------------
# Step 1: review stage splitting
# ---------------------------------------------------------------------------


class TestReviewStageSplit:
    """Tests for facts_review_stages module."""

    def test_stage_order_is_complete(self):
        assert len(STAGE_ORDER) == 6
        assert FactsReviewStage.SCOPE in STAGE_ORDER
        assert FactsReviewStage.COMPLETENESS in STAGE_ORDER

    def test_stage_field_map_covers_key_fields(self):
        """Each stage maps to the relevant FactsPatch fields."""
        scope_fields = STAGE_FIELD_MAP[FactsReviewStage.SCOPE]
        assert "model_scope" in scope_fields
        assert "assembly_count" in scope_fields

        insert_fields = STAGE_FIELD_MAP[FactsReviewStage.LOCALIZED_INSERT]
        assert "localized_insert_requirements" in insert_fields

        grid_fields = STAGE_FIELD_MAP[FactsReviewStage.GRID_AXIAL]
        assert "has_spacer_grids" in grid_fields

    def test_extract_facts_subset_returns_only_stage_fields(self):
        facts = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "has_spacer_grids": True,
            "fuel_variant_requirements": [{"variant_id": "v1"}],
        }
        scope_subset = extract_facts_subset(facts, FactsReviewStage.SCOPE)
        assert "model_scope" in scope_subset
        assert "assembly_count" in scope_subset
        assert "has_spacer_grids" not in scope_subset
        assert "fuel_variant_requirements" not in scope_subset

    def test_stage_prompt_is_smaller_than_full_patch(self):
        """The stage prompt should not contain irrelevant fields."""
        from openmc_agent.plan_builder.closed_loop.models import PlanEvidencePack, SourceExcerpt, PlanGateId

        facts = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "core_lattice_size": [3, 3],
            "has_spacer_grids": True,
            "fuel_variant_requirements": [{"variant_id": "v1", "enrichment_wt_percent": 3.1}],
            "localized_insert_requirements": [{"requirement_id": "cr1"}],
        }
        pack = PlanEvidencePack(
            evidence_pack_id="test",
            gate_id=PlanGateId.FACTS,
            source_excerpts=[SourceExcerpt(source_id="s1", text="excerpt")],
            relevant_patches={"facts": facts},
        )
        request = FactsReviewStageRequest(
            stage=FactsReviewStage.SCOPE,
            target_fields=("model_scope", "assembly_count"),
            facts_subset=extract_facts_subset(facts, FactsReviewStage.SCOPE),
            evidence_excerpts=[{"evidence_hash": "h1", "text": "excerpt"}],
        )
        prompt = build_stage_review_prompt(request, pack)
        # Scope stage should include model_scope and assembly_count.
        assert "model_scope" in prompt
        assert "assembly_count" in prompt
        # Scope stage should NOT include fuel_variant_requirements.
        assert "fuel_variant_requirements" not in prompt
        assert "localized_insert_requirements" not in prompt


# ---------------------------------------------------------------------------
# Step 3: repair coverage completeness
# ---------------------------------------------------------------------------


class TestRepairCompleteness:
    """Tests for facts_revision.check_facts_repair_completeness."""

    def test_complete_candidate_passes(self):
        candidate = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "assembly_type_counts": {"fuel": 9},
            "fuel_variant_requirements": [{"variant_id": "v1"}],
            "localized_insert_requirements": [{"requirement_id": "cr1"}],
            "has_spacer_grids": True,
        }
        missing = check_facts_repair_completeness(candidate)
        assert missing == []

    def test_missing_model_scope_fails(self):
        candidate = {
            "model_scope": "unknown",
            "assembly_count": 9,
            "assembly_type_counts": {"fuel": 9},
            "fuel_variant_requirements": [{"variant_id": "v1"}],
            "localized_insert_requirements": [{"requirement_id": "cr1"}],
            "has_spacer_grids": True,
        }
        missing = check_facts_repair_completeness(candidate)
        assert "/model_scope" in missing

    def test_empty_fuel_variants_fails(self):
        candidate = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "assembly_type_counts": {"fuel": 9},
            "fuel_variant_requirements": [],
            "localized_insert_requirements": [{"requirement_id": "cr1"}],
            "has_spacer_grids": True,
        }
        missing = check_facts_repair_completeness(candidate)
        assert "/fuel_variant_requirements" in missing

    def test_empty_localized_inserts_fails(self):
        candidate = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "assembly_type_counts": {"fuel": 9},
            "fuel_variant_requirements": [{"variant_id": "v1"}],
            "has_special_pin_map": True,
            "localized_insert_requirements": [],
            "has_spacer_grids": True,
        }
        missing = check_facts_repair_completeness(candidate)
        assert "/localized_insert_requirements" in missing

    def test_empty_localized_inserts_allowed_without_special_pin_map(self):
        candidate = {
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "assembly_type_counts": {"fuel": 9},
            "fuel_variant_requirements": [{"variant_id": "v1"}],
            "has_special_pin_map": False,
            "localized_insert_requirements": [],
            "has_spacer_grids": True,
        }
        missing = check_facts_repair_completeness(candidate)
        assert "/localized_insert_requirements" not in missing

    def test_required_coverage_paths_are_reactor_neutral(self):
        """The required paths must not include reactor-specific fields."""
        for path in REQUIRED_COVERAGE_PATHS:
            # No VERA/PWR/BWR/HTGR-specific field names.
            assert not any(term in path.lower() for term in ("vera", "pwr", "bwr", "htgr"))


# ---------------------------------------------------------------------------
# Step 4: reviewer output stability
# ---------------------------------------------------------------------------


class TestReviewerFailureClassification:
    """Tests for _classify_review_failure."""

    def test_empty_response_classified(self):
        """All attempts with empty raw_text → reviewer_empty_response."""

        @dataclass
        class _Attempt:
            raw_text: str = ""

        @dataclass
        class _Call:
            ok: bool = False
            error_code: str = "structured_review.schema_invalid"
            attempts: list = None

            def __post_init__(self):
                if self.attempts is None:
                    self.attempts = [_Attempt(), _Attempt()]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "facts.reviewer_empty_response"

    def test_empty_response_false_when_call_ok(self):
        """call.ok=True → never classifies as empty even if raw_text empty."""

        @dataclass
        class _Attempt:
            raw_text: str = ""

        @dataclass
        class _Call:
            ok: bool = True
            error_code: str = "structured_review.schema_invalid"
            attempts: list = None

            def __post_init__(self):
                if self.attempts is None:
                    self.attempts = [_Attempt(), _Attempt()]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "structured_review.schema_invalid"

    def test_free_text_approve_classified(self):
        """Short prose 'approve' without JSON → free_text_approve."""

        @dataclass
        class _Attempt:
            raw_text: str = ""

        @dataclass
        class _Call:
            ok: bool = False
            error_code: str = "structured_review.schema_invalid"
            attempts: list = None

            def __post_init__(self):
                self.attempts = [_Attempt(raw_text="Looks good, no issues found.")]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "facts.reviewer_free_text_approve"

    def test_budget_exhausted_takes_priority(self):
        """Budget exhaustion is checked before empty/free-text."""

        @dataclass
        class _Attempt:
            raw_text: str = ""

        @dataclass
        class _Call:
            ok: bool = False
            error_code: str = "planning.closed_loop.budget_exhausted"
            attempts: list = None

            def __post_init__(self):
                self.attempts = [_Attempt(), _Attempt()]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "facts_review.budget_exhausted"

    def test_malformed_json_falls_through_to_schema_invalid(self):
        """JSON present but broken → schema_invalid (not empty/free-text)."""

        @dataclass
        class _Attempt:
            raw_text: str = '{"findings": [broken'

        @dataclass
        class _Call:
            ok: bool = False
            error_code: str = "structured_review.schema_invalid"
            attempts: list = None

            def __post_init__(self):
                self.attempts = [_Attempt(raw_text='{"findings": [broken')]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "structured_review.schema_invalid"

    def test_repeated_hash_truncation_classified(self):
        """A reviewer that loops on evidence hashes gets a precise failure code."""

        @dataclass
        class _Attempt:
            raw_text: str = ""
            truncated_suspected: bool = True

        @dataclass
        class _Call:
            ok: bool = False
            error_code: str = "structured_review.schema_invalid"
            attempts: list = None
            raw_outputs: list[str] = None

            def __post_init__(self):
                repeated = (
                    '{"review_status":"complete_with_gaps",'
                    '"reviewed_evidence_hashes":["'
                    + ("a" * 64) * 160
                )
                self.attempts = [_Attempt(raw_text=repeated)]
                self.raw_outputs = [repeated]

        call = _Call()
        code = _classify_review_failure(call)
        assert code == "facts.reviewer_repeated_hash_truncation"


class TestCompletenessOwnership:
    def test_prompt_keeps_downstream_contracts_nonblocking(self):
        from openmc_agent.plan_builder.closed_loop.models import PlanEvidencePack, PlanGateId, SourceExcerpt

        pack = PlanEvidencePack(
            evidence_pack_id="ownership",
            gate_id=PlanGateId.FACTS,
            source_excerpts=[SourceExcerpt(source_id="s", text="source")],
            relevant_patches={"facts": {"missing_facts": []}},
        )
        request = FactsReviewStageRequest(
            stage=FactsReviewStage.COMPLETENESS,
            facts_subset={"missing_facts": []},
            evidence_excerpts=[{"text": "source"}],
        )
        prompt = build_stage_review_prompt(request, pack)
        assert "downstream requirement contracts" in prompt
        assert "never as a Facts Gate error" in prompt
