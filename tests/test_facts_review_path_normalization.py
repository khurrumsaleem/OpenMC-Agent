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
from openmc_agent.plan_builder.closed_loop.facts_reviewer import (
    _normalize,
    normalize_facts_review_finding,
    reconcile_facts_findings_against_whole_source,
    run_facts_review,
)
from openmc_agent.plan_builder.closed_loop.models import (
    FactsReviewModelOutput,
    PlanClosedLoopPolicy,
    PlanEvidencePack,
    PlanGateId,
    PlanFindingCategory,
    PlanFindingSeverity,
    PlanReviewFinding,
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

    def test_planning_metadata_paths_rejected_as_out_of_scope(self):
        """Findings on evidence-pack / planning-metadata paths (feature
        contract, planning-mode decision) are not FactsPatch fields and the
        Facts revision loop cannot act on them, so they must be rejected
        rather than deadlock closure."""
        pack = _make_pack()
        eh = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(
                code="FR-001",
                severity="error",
                category="representation_error",
                paths=["/patch_summaries/planning_mode_decision/feature_summary/large_lattice_dimension"],
                evidence_hash=eh,
            ),
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
                "code": "facts_review.excerpt_limited_finding",
                "finding_code": "unsupported_inference.pyrex_radial_layers_and_isotopics",
                "reason": "chunk-local missing-evidence claim is not a whole-source human blocker",
            }
        ]
        assert output.findings[0].severity is PlanFindingSeverity.WARNING
        assert output.findings[0].requires_human is False

    def test_relevant_patches_facts_path_is_normalized(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        output = _make_output([
            _make_draft_dict(
                code="FACTS_NOTE",
                severity="warning",
                paths=["/relevant_patches.facts.source_notes[7]"],
                evidence_hash=evidence_hash,
            ),
        ])

        findings, rejected = _normalize(output, pack)

        assert rejected == []
        assert findings[0].affected_json_paths == ["/source_notes/7"]

    def test_excerpt_limited_source_coverage_is_rejected_as_chunk_local(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="PYREX_RADIAL_LAYERS_UNEVIDENCED",
            paths=["/relevant_patches.facts.localized_insert_requirements[0].source_note"],
            evidence_hash=evidence_hash,
            repairable=False,
            requires_human=True,
            category="source_coverage",
        )
        payload["message"] = (
            "The supplied source excerpt is truncated immediately before the "
            "section containing the Pyrex radial structure."
        )
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert findings == []
        assert rejected == [
            {
                "code": "facts_review.excerpt_limited_finding",
                "finding_code": "PYREX_RADIAL_LAYERS_UNEVIDENCED",
                "reason": "chunk-local missing-evidence claim is not a whole-source human blocker",
            }
        ]
        assert output.findings[0].affected_json_paths == [
            "/localized_insert_requirements/0/source_note"
        ]

    def test_normalization_firewall_directly_classifies_non_error_human_as_advisory(self):
        payload = _make_draft_dict(
            code="PYREX_ENDCAP_COORD_AMBIGUITY",
            severity="info",
            paths=["/relevant_patches.facts.localized_insert_requirements[0]"],
            repairable=False,
            requires_human=True,
            category="physical_ambiguity",
        )
        payload["message"] = "Exact endcap coordinates are ambiguous, but this is advisory context."
        draft = FactsReviewModelOutput.model_validate({
            "review_status": "complete_with_gaps",
            "reviewed_evidence_hashes": [],
            "coverage_summary": {},
            "findings": [payload],
        }).findings[0]

        normalized = normalize_facts_review_finding(draft)

        assert normalized.rejected is None
        assert normalized.draft is not None
        assert normalized.draft.requires_human is False
        assert normalized.draft.affected_json_paths == ["/localized_insert_requirements/0"]
        assert normalized.classification_override["reason"] == "facts_non_error_human_advisory"


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
                "code": "facts_review.excerpt_limited_finding",
                "finding_code": "unsupported_inference.pyrex_radial_layers_and_isotopics",
                "reason": "chunk-local missing-evidence claim is not a whole-source human blocker",
            }
        ]


def _make_plan_finding(
    *,
    code: str = "TEST",
    severity: PlanFindingSeverity = PlanFindingSeverity.ERROR,
    category: PlanFindingCategory = PlanFindingCategory.SOURCE_COVERAGE,
    message: str = "test finding",
    paths: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    repairable: bool = True,
    requires_human: bool = False,
) -> PlanReviewFinding:
    return PlanReviewFinding(
        gate_id=PlanGateId.FACTS,
        code=code,
        severity=severity,
        category=category,
        message=message,
        confidence=0.9,
        affected_json_paths=paths if paths is not None else ["/expected_pyrex_count"],
        repairable_by_llm=repairable,
        requires_human=requires_human,
        metadata=metadata or {},
    )


class TestWholeSourceReconciliation:
    """Direct tests for reconcile_facts_findings_against_whole_source.

    Facts review splits the source into chunks; a chunk-local reviewer may
    flag a value as unsupported even though the supporting section lives in a
    different chunk.  The reconciliation downgrades such findings when the
    cited evidence is present in the whole source.
    """

    def test_chunk_local_unsupported_finding_backed_by_whole_source_is_downgraded(self):
        finding = _make_plan_finding(
            code="UNSUPPORTED_RCCA_DATA",
            category=PlanFindingCategory.UNSUPPORTED_INFERENCE,
            message=(
                "RCCA density 10.2 cannot be verified and is unsupported; "
                "the cited section §15.1 is not in this chunk."
            ),
            paths=["/localized_insert_requirements/3/source_note"],
            metadata={"current_value": "AIC_density=10.2 g/cc; source=§15.1"},
        )
        whole_source = (
            "### 15.1 RCCA absorber geometry\n"
            "AIC density 10.2 g/cc; B4C density 1.76 g/cc.\n"
        )

        reconciled, audit = reconcile_facts_findings_against_whole_source([finding], whole_source)

        assert len(reconciled) == 1
        assert reconciled[0].severity is PlanFindingSeverity.WARNING
        assert reconciled[0].repairable_by_llm is False
        assert len(audit) == 1
        assert audit[0]["code"] == "facts_review.whole_source_reconciled"
        assert audit[0]["finding_code"] == "UNSUPPORTED_RCCA_DATA"

    def test_finding_citing_absent_section_stays_blocking(self):
        finding = _make_plan_finding(
            code="UNSUPPORTED_X",
            category=PlanFindingCategory.UNSUPPORTED_INFERENCE,
            message="value 99.99 is unsupported; cited §99.9 is missing.",
            metadata={"current_value": "x=99.99; source=§99.9"},
        )
        whole_source = "### 15.1 RCCA absorber geometry\nAIC density 10.2 g/cc.\n"

        reconciled, audit = reconcile_facts_findings_against_whole_source([finding], whole_source)

        assert len(reconciled) == 1
        assert reconciled[0].severity is PlanFindingSeverity.ERROR
        assert audit == []

    def test_hard_contract_omission_stays_blocking_even_with_section_citation(self):
        finding = _make_plan_finding(
            code="FACTS_INCOMPLETE",
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="patch omits fuel_variant_requirements; §5 not evidenced in this chunk.",
        )
        whole_source = "### 5 fuel variants\nregion1 2.11 wt% U-235.\n"

        reconciled, audit = reconcile_facts_findings_against_whole_source([finding], whole_source)

        assert reconciled[0].severity is PlanFindingSeverity.ERROR
        assert audit == []

    def test_non_coverage_finding_is_not_reconciled(self):
        finding = _make_plan_finding(
            code="COUNT_MISMATCH",
            category=PlanFindingCategory.PHYSICAL_AMBIGUITY,
            message="assembly_count mismatch: patch says 9 but §2.1 states 8.",
        )
        whole_source = "### 2.1\nassembly count is 9.\n"

        reconciled, audit = reconcile_facts_findings_against_whole_source([finding], whole_source)

        assert reconciled[0].severity is PlanFindingSeverity.ERROR
        assert audit == []

    def test_warning_findings_pass_through_unchanged(self):
        finding = _make_plan_finding(
            severity=PlanFindingSeverity.WARNING,
            code="UNSUPPORTED_X",
            message="value 10.2 unsupported; cited §15.1 missing.",
        )
        reconciled, audit = reconcile_facts_findings_against_whole_source(
            [finding], "### 15.1 ... 10.2 ..."
        )
        assert reconciled[0].severity is PlanFindingSeverity.WARNING
        assert audit == []


class TestLocalizedInsertGeometryDetail:
    """The localized-insert detailed-geometry owner boundary."""

    def test_thimble_plug_geometry_demand_on_insert_source_note_is_downgraded(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="THIMBLE_PLUG_GEOMETRY_MISSING",
            paths=["/relevant_patches.facts.localized_insert_requirements[1].source_note"],
            evidence_hash=evidence_hash,
            repairable=True,
            requires_human=False,
            category="source_coverage",
        )
        payload["message"] = (
            "Thimble plug insert requirements lack all geometric parameters "
            "specified in §14: outer_radius=0.538 cm, height=11.0 cm and the "
            "full radial cross-section."
        )
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert len(findings) == 1
        assert findings[0].severity is PlanFindingSeverity.WARNING
        assert findings[0].metadata["classification_override"]["reason"] == "facts_localized_insert_geometry_detail"
        assert findings[0].metadata["classification_override"]["owner_route"] == "downstream_patch_family"

    def test_insert_requirement_existence_gap_stays_blocking(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="LOCALIZED_INSERT_REQUIREMENTS_MISSING",
            paths=["/localized_insert_requirements"],
            evidence_hash=evidence_hash,
            repairable=True,
            requires_human=False,
            category="source_coverage",
        )
        payload["message"] = (
            "The patch is missing localized_insert_requirements entirely; no "
            "thimble plug insert geometry or requirement is declared."
        )
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert len(findings) == 1
        assert findings[0].severity is PlanFindingSeverity.ERROR

    def test_insert_finding_without_geometry_markers_stays_blocking(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="INSERT_COUNT_WRONG",
            paths=["/localized_insert_requirements/1/expected_coordinate_count_per_assembly"],
            evidence_hash=evidence_hash,
            repairable=True,
            requires_human=False,
            category="source_coverage",
        )
        payload["message"] = "expected_coordinate_count_per_assembly is 4 but should be 24."
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert len(findings) == 1
        assert findings[0].severity is PlanFindingSeverity.ERROR


class TestExcerptLimitedFreeFormCode:
    """The loosened excerpt-limited firewall accepts free-form unsupported_* codes."""

    def test_free_form_unsupported_code_with_excerpt_marker_is_rejected(self):
        pack = _make_pack()
        evidence_hash = pack.source_excerpts[0].evidence_hash
        payload = _make_draft_dict(
            code="UNSUPPORTED_RCCA_DATA",
            paths=["/localized_insert_requirements/3/source_note"],
            evidence_hash=evidence_hash,
            repairable=False,
            requires_human=False,
            category="unsupported_inference",
        )
        payload["message"] = (
            "These values cannot be verified against the supplied evidence; "
            "the provided source excerpt ends at §13.4."
        )
        output = _make_output([payload])

        findings, rejected = _normalize(output, pack)

        assert findings == []
        assert any(r.get("code") == "facts_review.excerpt_limited_finding" for r in rejected)


class TestWholeSourceReconciliationEndToEnd:
    """End-to-end: a chunk-local finding not caught by the per-chunk firewall
    is reconciled against the whole source so coverage is not blocked."""

    def test_chunk_local_coverage_finding_unblocked_by_whole_source_reconciliation(self):
        policy = PlanClosedLoopPolicy()
        packs = [
            PlanEvidencePack(
                evidence_pack_id="chunk-1",
                gate_id=PlanGateId.FACTS,
                source_excerpts=[SourceExcerpt(source_id="s1", text="first chunk ends at section 13.4")],
                relevant_patches={"facts": {"expected_pyrex_count": 16}},
            ),
            PlanEvidencePack(
                evidence_pack_id="chunk-2",
                gate_id=PlanGateId.FACTS,
                source_excerpts=[SourceExcerpt(source_id="s2", text="section 15.1 RCCA density 10.2 g/cc")],
                relevant_patches={"facts": {"expected_pyrex_count": 16}},
            ),
        ]
        whole_source = (
            "first chunk ends at section 13.4\n"
            "section 15.1 RCCA density 10.2 g/cc\n"
        )
        payloads = []
        for index, pack in enumerate(packs):
            evidence_hash = pack.source_excerpts[0].evidence_hash
            findings = []
            if index == 0:
                # Phrased WITHOUT an excerpt marker and WITHOUT a matching
                # excerpt-limited code, so the per-chunk firewall leaves it as
                # a blocking error.  Whole-source reconciliation must catch it.
                findings.append(
                    {
                        "code": "REGIONAL_GAP",
                        "severity": "error",
                        "category": "source_coverage",
                        "message": (
                            "RCCA absorber lacks evidence in this chunk; the "
                            "values are missing here. Section 15.1 is cited."
                        ),
                        "evidence_hashes": [evidence_hash],
                        "affected_json_paths": ["/localized_insert_requirements/3/source_note"],
                        "repairable_by_llm": True,
                        "requires_human": False,
                        "confidence": 0.8,
                        "current_value": "AIC_density=10.2; source=§15.1",
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
            whole_source_text=whole_source,
        )

        assert result.ok
        assert result.coverage_complete
        assert len(result.findings) == 1
        assert result.findings[0].severity is PlanFindingSeverity.WARNING
        assert any(r.get("code") == "facts_review.whole_source_reconciled" for r in result.rejected)

    def test_without_whole_source_chunk_local_finding_blocks_coverage(self):
        """Sanity: without whole_source_text the same finding stays blocking."""
        policy = PlanClosedLoopPolicy()
        packs = [
            PlanEvidencePack(
                evidence_pack_id="chunk-1",
                gate_id=PlanGateId.FACTS,
                source_excerpts=[SourceExcerpt(source_id="s1", text="first chunk ends at section 13.4")],
                relevant_patches={"facts": {"expected_pyrex_count": 16}},
            ),
        ]
        evidence_hash = packs[0].source_excerpts[0].evidence_hash
        payload = json.dumps(
            {
                "review_status": "complete",
                "reviewed_evidence_hashes": [evidence_hash],
                "coverage_summary": {},
                "findings": [
                    {
                        "code": "REGIONAL_GAP",
                        "severity": "error",
                        "category": "source_coverage",
                        "message": "RCCA absorber lacks evidence; values missing. Section 15.1 cited.",
                        "evidence_hashes": [evidence_hash],
                        "affected_json_paths": ["/localized_insert_requirements/3/source_note"],
                        "repairable_by_llm": True,
                        "requires_human": False,
                        "confidence": 0.8,
                        "current_value": "AIC_density=10.2; source=§15.1",
                    }
                ],
            }
        )

        result = run_facts_review(
            evidence_packs=packs,
            reviewer_client=lambda _: payload,
            state=PlanBuildState(state_id="s", requirement_text="r"),
            policy=policy,
        )

        assert not result.coverage_complete
        assert result.findings[0].severity is PlanFindingSeverity.ERROR
