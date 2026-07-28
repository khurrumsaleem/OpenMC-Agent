"""Phase 8B Step 4B-1+ — Facts review path normalization and revision trigger tests.

Covers:

* ``affected_json_paths`` with ``facts_subset.`` prefix → normalized to ``/X``.
* Bare field names → normalized to ``/X``.
* Already-canonical ``/X`` paths → unchanged (idempotent).
* ``/materials/...`` and ``/universes/...`` paths → still rejected (scope guard).
* End-to-end: findings with bare paths are accepted after normalization.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openmc_agent.plan_builder.closed_loop.facts_evidence import build_facts_evidence_packs
from openmc_agent.plan_builder.closed_loop.facts_reviewer import _normalize, run_facts_review
from openmc_agent.plan_builder.closed_loop.models import (
    FactsReviewModelOutput,
    PlanClosedLoopPolicy,
    PlanEvidencePack,
    PlanGateId,
    PlanFindingCategory,
    PlanFindingSeverity,
    SourceExcerpt,
)
from openmc_agent.plan_builder.state import PlanBuildState


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "facts_revision"


def _make_pack() -> PlanEvidencePack:
    return PlanEvidencePack(
        evidence_pack_id="test",
        gate_id=PlanGateId.FACTS,
        source_excerpts=[SourceExcerpt(source_id="s1", text="excerpt")],
        relevant_patches={"facts": {"expected_pyrex_count": None}},
    )


def _make_draft_dict(
    code: str = "TEST",
    severity: str = "error",
    paths: list[str] | None = None,
    evidence_hash: str = "",
    repairable: bool = True,
    requires_human: bool = False,
    category: str = "source_coverage",
) -> dict[str, Any]:
    return {
        "code": code,
        "severity": severity,
        "category": category,
        "message": "test finding",
        "evidence_hashes": [evidence_hash] if evidence_hash else [],
        "affected_json_paths": ["/expected_pyrex_count"] if paths is None else paths,
        "repairable_by_llm": repairable,
        "requires_human": requires_human,
        "confidence": 0.9,
    }


def _make_output(drafts: list[dict[str, Any]]) -> FactsReviewModelOutput:
    return FactsReviewModelOutput.model_validate({
        "review_status": "complete_with_gaps",
        "reviewed_evidence_hashes": [],
        "coverage_summary": {},
        "findings": drafts,
    })


class TestPathNormalization:
    """Direct tests for the _normalize path prefix fix."""

    def test_facts_subset_prefix_normalized(self):
        """facts_subset.X → /X."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(code="P1", paths=["facts_subset.expected_pyrex_count"], evidence_hash=eh),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 1
        assert findings[0].affected_json_paths == ["/expected_pyrex_count"]
        assert rejected == []

    def test_bare_field_normalized(self):
        """Bare field name → /field."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(code="P2", paths=["expected_pyrex_count"], evidence_hash=eh),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 1
        assert findings[0].affected_json_paths == ["/expected_pyrex_count"]

    def test_already_slash_prefixed_unchanged(self):
        """/X stays /X (idempotent)."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(code="P3", paths=["/expected_pyrex_count"], evidence_hash=eh),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 1
        assert findings[0].affected_json_paths == ["/expected_pyrex_count"]

    def test_materials_path_still_rejected(self):
        """/materials/... paths are still rejected (scope guard)."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(code="P4", paths=["/materials/fuel"], evidence_hash=eh),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 0
        assert any(r.get("code") == "facts_review.path_out_of_scope" for r in rejected)

    def test_universes_path_still_rejected(self):
        """/universes/... paths are still rejected (scope guard)."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(code="P5", paths=["/universes/cell_1"], evidence_hash=eh),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 0
        assert any(r.get("code") == "facts_review.path_out_of_scope" for r in rejected)

    def test_multiple_paths_normalized(self):
        """Multiple paths in one finding are all normalized."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(
                code="P6",
                paths=["facts_subset.expected_pyrex_count", "expected_thimble_plug_count", "/expected_spacer_grid_count"],
                evidence_hash=eh,
            ),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 1
        assert findings[0].affected_json_paths == [
            "/expected_pyrex_count",
            "/expected_thimble_plug_count",
            "/expected_spacer_grid_count",
        ]

    def test_facts_subset_with_nested_path(self):
        """facts_subset.fuel_variant_requirements.0.variant_id → /fuel_variant_requirements.0.variant_id."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(
                code="P7",
                paths=["facts_subset.fuel_variant_requirements"],
                evidence_hash=eh,
            ),
        ])
        findings, rejected = _normalize(output, pack)
        assert len(findings) == 1
        assert findings[0].affected_json_paths == ["/fuel_variant_requirements"]

    @pytest.mark.parametrize("path", ["/missing_facts", "/missing_facts/0", "/assumptions/0", "/source_notes/source_1"])
    def test_metadata_recording_paths_override_conservative_classification(self, path: str):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        output = _make_output([_make_draft_dict(code="MISSING_OPERATING_STATE", paths=[path], evidence_hash=evidence_hash, repairable=False, requires_human=True)])
        findings, rejected = _normalize(output, pack)
        assert rejected == []
        assert len(findings) == 1
        finding = findings[0]
        assert finding.repairable_by_llm is True
        assert finding.requires_human is False
        assert finding.metadata["classification_override"] == {
            "reason": "facts_metadata_recording",
            "original_repairable_by_llm": False,
            "original_requires_human": True,
        }

    @pytest.mark.parametrize("paths", [["/missing_facts_extra"], ["/missing_facts", "/expected_pyrex_count"]])
    def test_non_metadata_or_mixed_paths_keep_reviewer_classification(self, paths: list[str]):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        output = _make_output([_make_draft_dict(code="NON_METADATA", paths=paths, evidence_hash=evidence_hash, repairable=False, requires_human=True)])
        findings, rejected = _normalize(output, pack)
        assert rejected == []
        assert len(findings) == 1
        assert findings[0].repairable_by_llm is False
        assert findings[0].requires_human is True
        assert "classification_override" not in findings[0].metadata

    def test_empty_path_error_is_not_promoted_to_metadata_repair(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        output = _make_output([_make_draft_dict(code="EMPTY_PATH", paths=[], evidence_hash=evidence_hash, repairable=False, requires_human=True)])
        findings, rejected = _normalize(output, pack)
        assert findings == []
        assert any(item["code"] == "facts_review.invalid_finding_contract" for item in rejected)

    def test_confirmation_not_error_reviewer_finding_is_downgraded_to_warning(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="FUEL_VARIANT_COUNT_TOTAL_MISMATCH",
            paths=["/fuel_variant_requirements"],
            evidence_hash=evidence_hash,
            repairable=False,
            requires_human=False,
        )
        payload["message"] = (
            "The facts_subset lists 2 fuel_variant_requirements entries, but "
            "the expected_assembly_count values sum to 9 and no enrichment "
            "level is omitted; no duplicate variant_id is present. This finding "
            "is recorded as a coverage confirmation, not an error."
        )
        output = _make_output([payload])
        findings, rejected = _normalize(output, pack)
        assert rejected == []
        assert len(findings) == 1
        assert findings[0].severity is PlanFindingSeverity.WARNING
        assert output.findings[0].severity is PlanFindingSeverity.WARNING
        assert findings[0].metadata["classification_override"] == {
            "reason": "facts_reviewer_confirmation_not_error",
            "original_severity": "error",
            "original_repairable_by_llm": False,
            "original_requires_human": False,
        }

    def test_step3d_t3_fixture_contradictory_finding_is_sanitized_and_downgraded(self):
        fixture_path = FIXTURE_ROOT / "phase8c_step3d_facts_stale_closure.json"
        raw = fixture_path.read_text(encoding="utf-8")
        lowered = raw.lower()
        for forbidden in ("api_key", "prompt_text", "raw_provider_output", "reasoning_content", "sk-"):
            assert forbidden not in lowered
        fixture = json.loads(raw)
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        contradictory = fixture["latest_rereview_error_findings"][0]
        contradictory = {
            **contradictory,
            "evidence_hashes": [evidence_hash],
            "confidence": 0.9,
        }
        output = _make_output([contradictory])
        findings, rejected = _normalize(output, pack)
        assert rejected == []
        assert findings[0].code == "FUEL_VARIANT_COUNT_TOTAL_MISMATCH"
        assert findings[0].severity is PlanFindingSeverity.WARNING

    def test_excerpt_limited_unsupported_inference_is_rejected_as_chunk_local(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="unsupported_inference.pyrex_radial_layers_and_isotopics",
            paths=["/expected_pyrex_count"],
            evidence_hash=evidence_hash,
            repairable=False,
            requires_human=True,
            category="unsupported_inference",
        )
        payload["message"] = (
            "The source excerpt truncates before Section 12.2, so the radial "
            "layers are not provided in the source excerpt."
        )
        payload["expected_value"] = "Not provided in the source excerpt"
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert findings == []
        assert rejected == [
            {
                "code": "facts_review.excerpt_limited_unsupported_inference",
                "finding_code": "unsupported_inference.pyrex_radial_layers_and_isotopics",
                "reason": "chunk-local unsupported-inference claim is not a whole-source human blocker",
            }
        ]
        assert output.findings[0].severity is PlanFindingSeverity.WARNING
        assert output.findings[0].requires_human is False


class TestEndToEndPathNormalization:
    """End-to-end: run_facts_review accepts findings with bare paths."""

    def test_run_facts_review_accepts_facts_subset_paths(self):
        """A finding with facts_subset. prefix should be accepted, not rejected."""
        policy = PlanClosedLoopPolicy()
        packs = build_facts_evidence_packs(
            requirement_text="variant A\n",
            facts_patch={"patch_type": "facts"},
            confirmed_facts={},
            planning_metadata={},
            policy=policy,
        )
        evidence = packs[0].source_excerpts[0].evidence_hash
        payload = json.dumps({
            "review_status": "complete_with_gaps",
            "reviewed_evidence_hashes": [evidence],
            "coverage_summary": {},
            "findings": [
                {
                    "code": "WARN_NULL",
                    "severity": "warning",
                    "category": "source_coverage",
                    "message": "expected count is null",
                    "evidence_hashes": [evidence],
                    "affected_json_paths": ["facts_subset.expected_pyrex_count"],
                    "repairable_by_llm": True,
                    "requires_human": False,
                    "confidence": 0.8,
                },
            ],
        })
        result = run_facts_review(
            evidence_packs=packs,
            reviewer_client=lambda _: payload,
            state=PlanBuildState(state_id="s", requirement_text="r"),
            policy=policy,
        )
        assert len(result.findings) == 1
        assert result.findings[0].affected_json_paths == ["/expected_pyrex_count"]
        assert result.coverage_complete  # warning-only, no error findings

    def test_split_review_excerpt_limited_false_positive_does_not_block_coverage(self):
        policy = PlanClosedLoopPolicy()
        packs = [
            PlanEvidencePack(
                evidence_pack_id="chunk-1",
                gate_id=PlanGateId.FACTS,
                source_excerpts=[SourceExcerpt(source_id="s1", text="first chunk without section 12.2")],
                relevant_patches={"facts": {"expected_pyrex_count": 16}},
            ),
            PlanEvidencePack(
                evidence_pack_id="chunk-2",
                gate_id=PlanGateId.FACTS,
                source_excerpts=[SourceExcerpt(source_id="s2", text="section 12.2 confirms sixteen pyrex rods")],
                relevant_patches={"facts": {"expected_pyrex_count": 16}},
            ),
        ]
        payloads = []
        for index, pack in enumerate(packs):
            evidence_hash = pack.source_excerpts[0].evidence_hash
            findings = []
            if index == 0:
                findings.append(
                    {
                        "code": "unsupported_inference.pyrex_radial_layers_and_isotopics",
                        "severity": "error",
                        "category": "unsupported_inference",
                        "message": (
                            "The provided source excerpt does not include Section 12.2, "
                            "so the Pyrex dimensions are not provided in this source excerpt."
                        ),
                        "evidence_hashes": [evidence_hash],
                        "affected_json_paths": ["/expected_pyrex_count"],
                        "repairable_by_llm": False,
                        "requires_human": True,
                        "confidence": 0.8,
                        "expected_value": "Not provided in the source excerpt",
                    }
                )
            payloads.append(
                json.dumps(
                    {
                        "review_status": "complete",
                        "reviewed_evidence_hashes": [evidence_hash],
                        "coverage_summary": {},
                        "findings": findings,
                    }
                )
            )

        def reviewer(_: str) -> str:
            return payloads.pop(0)

        result = run_facts_review(
            evidence_packs=packs,
            reviewer_client=reviewer,
            state=PlanBuildState(state_id="s", requirement_text="r"),
            policy=policy,
        )

        assert result.ok
        assert result.coverage_complete
        assert result.findings == []
        assert result.rejected == [
            {
                "code": "facts_review.excerpt_limited_unsupported_inference",
                "finding_code": "unsupported_inference.pyrex_radial_layers_and_isotopics",
                "reason": "chunk-local unsupported-inference claim is not a whole-source human blocker",
            }
        ]
