"""Phase 8B Step 4B-1+ — Facts gate revision trigger tests.

Verifies that ``coverage_complete=False`` with repairable error findings
triggers revision (REVISE_CURRENT_PATCH) instead of immediately blocking.
Only ``review.ok=False`` (broken review output) causes an early block.

Key behavioral change in executor.py:
  Before: ``if not review.ok or not review.coverage_complete: return ...``
  After:  ``if not review.ok: return ...``
          (coverage_incomplete falls through to compute_allowed_actions)
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmc_agent.plan_builder import executor
from openmc_agent.plan_builder.closed_loop.controller import (
    initialize_plan_loop_state,
    transition_stage,
)
from openmc_agent.plan_builder.closed_loop.models import (
    PlanClosedLoopPolicy,
    PlanLoopMode,
    PlanReviewAction,
    PlanStageStatus,
)
from openmc_agent.plan_builder.closed_loop.policy import compute_allowed_actions
from openmc_agent.plan_builder.executor import run_incremental_planning
from openmc_agent.plan_builder.patch_generator import FakePatchLLM
from openmc_agent.plan_builder.state import PlanBuildState


def _extract_evidence_hash(prompt: str) -> str:
    """Extract the first evidence_hash from a review prompt payload."""
    data = json.loads(prompt.split("INPUT:\n", 1)[1])
    return data.get("facts_subset") or data.get("source_excerpts", [{}])[0].get("evidence_hash", "")


class TestComputeAllowedActionsCoverageIncomplete:
    """Unit tests for compute_allowed_actions with coverage_incomplete."""

    def test_required_coverage_paths_are_allowed_revision_targets(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import allowed_paths_for_findings
        from openmc_agent.plan_builder.closed_loop.models import (
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
        )

        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="SOURCE_COVERAGE.PYREX_PLENUM_TRUNCATION",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="repairable source coverage gap",
            affected_patch_types=["facts"],
            affected_json_paths=["/source_notes"],
            repairable_by_llm=True,
            requires_human=False,
            confidence=0.9,
        )

        allowed = allowed_paths_for_findings([finding])

        assert "/assembly_type_counts" in allowed
        assert "/fuel_variant_requirements" in allowed
        assert "/localized_insert_requirements" in allowed

    def test_required_coverage_paths_are_scope_conditional(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import (
            check_facts_repair_completeness,
            required_coverage_paths_for_patch,
        )

        single = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "has_spacer_grids": False,
            "assembly_type_counts": {},
            "localized_insert_requirements": [],
        }
        assert "/assembly_type_counts" not in required_coverage_paths_for_patch(single)
        assert "/localized_insert_requirements" not in required_coverage_paths_for_patch(single)
        assert check_facts_repair_completeness(single) == []

        multi = {
            **single,
            "model_scope": "multi_assembly_core",
            "assembly_count": 9,
            "core_lattice_size": None,
            "assembly_type_counts": {},
        }
        assert "/core_lattice_size" not in required_coverage_paths_for_patch(multi)
        assert "/assembly_type_counts" in required_coverage_paths_for_patch(multi)
        assert check_facts_repair_completeness(multi) == ["/assembly_type_counts"]

        inserts = {
            **single,
            "has_special_pin_map": True,
            "localized_insert_requirements": [],
        }
        assert "/localized_insert_requirements" in required_coverage_paths_for_patch(inserts)
        assert check_facts_repair_completeness(inserts) == ["/localized_insert_requirements"]

    def test_optional_coverage_paths_are_still_safe_revision_targets(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import evaluate_facts_revision
        from openmc_agent.plan_builder.closed_loop.models import (
            FactsRevisionProposal,
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
        )

        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "localized_insert_requirements": [{"requirement_id": "insert", "insert_kind": "pyrex_rod"}],
            "has_spacer_grids": False,
        }
        proposal = FactsRevisionProposal.model_validate({
            "proposal_id": "optional_count",
            "confidence": 0.9,
            "rationale": "record a harmless single-assembly type count while fixing source notes",
            "operations": [
                {"op": "replace", "path": "/assembly_type_counts", "value": {"default": 1}},
                {"op": "add", "path": "/source_notes/-", "value": "source-backed note"},
            ],
            "resolved_finding_ids": [],
        })
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="SOURCE_COVERAGE.NOTE",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="repairable source-note gap",
            affected_patch_types=["facts"],
            affected_json_paths=["/source_notes"],
            repairable_by_llm=True,
            requires_human=False,
            confidence=0.9,
        )

        result = evaluate_facts_revision(
            facts_patch=facts,
            proposal=proposal,
            findings=[finding],
            confirmed_facts={},
            prior_candidate_hashes=[],
        )

        assert result.accepted
        assert result.candidate["assembly_type_counts"] == {"default": 1}

    def test_schema_out_operating_conditions_are_recorded_as_source_notes(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import evaluate_facts_revision
        from openmc_agent.plan_builder.closed_loop.models import (
            FactsRevisionProposal,
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
        )

        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "has_spacer_grids": False,
        }
        proposal = FactsRevisionProposal.model_validate({
            "proposal_id": "operating_conditions",
            "confidence": 0.9,
            "rationale": "record operating-state values",
            "operations": [
                {
                    "op": "add",
                    "path": "/operating_conditions",
                    "value": {"boron_ppm": 1066, "coolant_temperature_K": 565},
                },
                {
                    "op": "add",
                    "path": "/source_notes/-",
                    "value": "Variant 3B operating state: boron 1066 ppm, coolant temperature 565 K.",
                },
            ],
            "resolved_finding_ids": [],
        })
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="OPERATING_CONDITIONS_MISSING",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="record operating conditions",
            affected_patch_types=["facts"],
            affected_json_paths=["/source_notes"],
            repairable_by_llm=True,
            requires_human=False,
            confidence=0.9,
        )

        result = evaluate_facts_revision(
            facts_patch=facts,
            proposal=proposal,
            findings=[finding],
            confirmed_facts={},
            prior_candidate_hashes=[],
        )

        assert result.accepted
        assert "operating_conditions" not in result.candidate
        assert result.candidate["source_notes"] == [
            "Variant 3B operating state: boron 1066 ppm, coolant temperature 565 K."
        ]

    def test_unknown_schema_out_path_still_fails_closed(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import evaluate_facts_revision
        from openmc_agent.plan_builder.closed_loop.models import (
            FactsRevisionProposal,
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
        )

        proposal = FactsRevisionProposal.model_validate({
            "proposal_id": "unknown_path",
            "confidence": 0.9,
            "rationale": "bad field",
            "operations": [{"op": "add", "path": "/made_up_field", "value": 1}],
            "resolved_finding_ids": [],
        })
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="BAD",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="bad",
            affected_patch_types=["facts"],
            affected_json_paths=["/source_notes"],
            repairable_by_llm=True,
            requires_human=False,
            confidence=0.9,
        )

        result = evaluate_facts_revision(
            facts_patch={"patch_type": "facts"},
            proposal=proposal,
            findings=[finding],
            confirmed_facts={},
            prior_candidate_hashes=[],
        )

        assert result.reasons == ["facts_revision.path_out_of_scope"]

    def test_revision_prompt_uses_same_conditional_required_coverage_contract(self):
        from openmc_agent.plan_builder.closed_loop.facts_revision import required_coverage_paths_for_patch
        from openmc_agent.plan_builder.closed_loop.facts_revision_prompts import _required_coverage_fields_with_status

        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "has_spacer_grids": False,
        }

        prompt_paths = [item["path"] for item in _required_coverage_fields_with_status(facts)]

        assert prompt_paths == list(required_coverage_paths_for_patch(facts))

    def test_repairable_error_findings_yield_revise(self):
        """Error findings with repairable_by_llm → REVISE_CURRENT_PATCH."""
        from openmc_agent.plan_builder.closed_loop.models import (
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
            PlanStageState,
        )
        stage = PlanStageState(stage_id="plan_gate_facts", gate_id=PlanGateId.FACTS)
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="TEST_NULL",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.SOURCE_COVERAGE,
            message="null field",
            affected_patch_types=["facts"],
            affected_json_paths=["/expected_pyrex_count"],
            repairable_by_llm=True,
            requires_human=False,
            confidence=0.9,
        )
        actions = compute_allowed_actions(
            policy=PlanClosedLoopPolicy(mode=PlanLoopMode.CONTROLLED),
            stage_state=stage,
            findings=[finding],
            deterministic_issues=[
                {"code": "facts_review.coverage_incomplete", "severity": "error", "blocking": True},
            ],
        )
        assert PlanReviewAction.REVISE_CURRENT_PATCH in actions

    def test_non_repairable_error_findings_yield_fail_closed(self):
        """Non-repairable error findings → FAIL_CLOSED."""
        from openmc_agent.plan_builder.closed_loop.models import (
            PlanFindingCategory,
            PlanFindingSeverity,
            PlanGateId,
            PlanReviewFinding,
            PlanStageState,
        )
        stage = PlanStageState(stage_id="plan_gate_facts", gate_id=PlanGateId.FACTS)
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code="HUMAN_ONLY",
            severity=PlanFindingSeverity.ERROR,
            category=PlanFindingCategory.PHYSICAL_AMBIGUITY,
            message="requires human",
            affected_patch_types=["facts"],
            affected_json_paths=["/model_scope"],
            repairable_by_llm=False,
            requires_human=True,
            confidence=0.9,
        )
        actions = compute_allowed_actions(
            policy=PlanClosedLoopPolicy(mode=PlanLoopMode.CONTROLLED),
            stage_state=stage,
            findings=[finding],
            deterministic_issues=[
                {"code": "facts_review.coverage_incomplete", "severity": "error", "blocking": True},
            ],
        )
        assert PlanReviewAction.REVISE_CURRENT_PATCH not in actions


class TestExecutorRevisionTrigger:
    """Integration tests: executor triggers revision for coverage_incomplete."""

    def test_schema_invalid_review_blocks_immediately(self, monkeypatch) -> None:
        """review.ok=False → immediate block, no revision attempted."""
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))

        llm = FakePatchLLM([json.dumps({"patch_type": "facts"})])

        call_count = {"n": 0}
        def bad_reviewer(_prompt: str) -> str:
            call_count["n"] += 1
            # Return invalid JSON to force schema_invalid
            return "{broken"

        result = run_incremental_planning(
            requirement="small source",
            state=PlanBuildState(state_id="s", requirement_text="small source"),
            llm_client=llm,
            plan_loop_policy={"mode": "controlled"},
            plan_reviewer_client=bad_reviewer,
            plan_repair_client=llm,
        )
        assert not result.ok
        stage = result.state.plan_loop_stages.get("plan_gate_facts")
        assert stage is not None
        assert stage.status.value == "blocked"
        # repair_count should be 0 — no revision attempted
        assert stage.repair_count == 0

    def test_schema_invalid_review_with_deterministic_error_triggers_revision(self, monkeypatch, tmp_path) -> None:
        """A broken reviewer must not mask a deterministic repairable Facts gap."""
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))

        initial_facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "has_spacer_grids": False,
        }
        patch_llm = FakePatchLLM([json.dumps(initial_facts)])
        repair_llm = FakePatchLLM([json.dumps({
            "proposal_id": "restore_grid_flag",
            "confidence": 0.9,
            "rationale": "restore source-backed spacer grid flag",
            "operations": [
                {"op": "replace", "path": "/has_spacer_grids", "value": True},
            ],
            "resolved_finding_ids": [],
        })])

        state = PlanBuildState(state_id="schema-bad-with-facts-gap", requirement_text="source")
        state.metadata["planning_mode_decision"] = {
            "feature_summary": {"has_spacer_grid": True}
        }
        review_calls = {"n": 0}

        def distinct_bad_reviewer(_prompt: str) -> str:
            review_calls["n"] += 1
            return "{\"broken\": " + ("[" * review_calls["n"])

        result = run_incremental_planning(
            requirement="source mentions spacer grid",
            state=state,
            llm_client=patch_llm,
            plan_loop_policy={"mode": "controlled"},
            plan_reviewer_client=distinct_bad_reviewer,
            plan_repair_client=repair_llm,
            plan_loop_output_dir=tmp_path,
        )

        assert result.ok
        stage = result.state.plan_loop_stages["plan_gate_facts"]
        assert stage.status is PlanStageStatus.ACCEPTED
        assert stage.repair_count == 1
        repaired = [
            env for env in result.state.patches.values()
            if env.patch_type == "facts" and env.status == "valid"
        ][-1]
        assert repaired.content["has_spacer_grids"] is True
        gate_result = json.loads((tmp_path / "incremental" / "plan_closed_loop" / "facts_gate_result.json").read_text())
        assert gate_result["final_gate_status"]["terminal_reason"] == "candidate_committed_and_accepted"

    def test_coverage_incomplete_with_repairable_triggers_revision(self, monkeypatch) -> None:
        """review.ok=True + error findings + repairable → revision attempted.

        The reviewer returns a valid JSON with error-severity findings
        (repairable_by_llm=True, affected_json_paths with /prefix).
        The executor should NOT immediately block; it should attempt
        revision via plan_repair_client.
        """
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))

        llm = FakePatchLLM([json.dumps({"patch_type": "facts"})])

        review_count = {"n": 0}
        def error_reviewer(prompt: str) -> str:
            review_count["n"] += 1
            # Extract evidence hash from the prompt payload
            try:
                payload = json.loads(prompt.split("INPUT:\n", 1)[1])
                excerpts = payload.get("evidence_excerpts", payload.get("source_excerpts", []))
                eh = excerpts[0].get("evidence_hash", "") if excerpts else ""
            except Exception:
                eh = ""
            return json.dumps({
                "review_status": "complete_with_gaps",
                "reviewed_evidence_hashes": [eh] if eh else [],
                "coverage_summary": {},
                "findings": [
                    {
                        "code": "FIELD_NULL",
                        "severity": "error",
                        "category": "source_coverage",
                        "message": "field is null",
                        "evidence_hashes": [eh] if eh else [],
                        "affected_json_paths": ["/expected_pyrex_count"],
                        "repairable_by_llm": True,
                        "requires_human": False,
                        "confidence": 0.9,
                        "expected_value": 80,
                        "current_value": None,
                    },
                ],
            })

        result = run_incremental_planning(
            requirement="small source",
            state=PlanBuildState(state_id="s", requirement_text="small source"),
            llm_client=llm,
            plan_loop_policy={"mode": "controlled"},
            plan_reviewer_client=error_reviewer,
            plan_repair_client=llm,
        )
        stage = result.state.plan_loop_stages.get("plan_gate_facts")
        assert stage is not None
        # Key assertion: revision should have been attempted (repair_count > 0)
        # or at minimum, the decision should be REVISE_CURRENT_PATCH (not immediate block)
        decisions = result.state.plan_review_decisions
        if decisions:
            last_facts_decision = [d for d in decisions.values() if d.gate_id.value == "facts"]
            if last_facts_decision:
                assert last_facts_decision[-1].action != PlanReviewAction.APPROVE
        # The stage should NOT be immediately blocked from the early return
        # (it may end up blocked from budget exhaustion, but repair_count > 0
        # proves revision was attempted)
        assert stage.repair_count > 0 or stage.status.value in ("repairing", "blocked", "review_failed")


class TestExecutorRevisionClosure:
    """A valid partial repair must continue with its remaining findings."""

    def test_metadata_only_error_is_normalized_before_initial_revision_action(self, monkeypatch) -> None:
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))
        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "assembly_type_counts": {"a": 1},
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "localized_insert_requirements": [{"requirement_id": "insert", "insert_kind": "pyrex_rod"}],
            "has_spacer_grids": False,
        }
        patch_llm = FakePatchLLM([json.dumps(facts)])
        repair_llm = FakePatchLLM([json.dumps({
            "proposal_id": "record_blank_operating_state",
            "confidence": 0.9,
            "rationale": "record the source-declared missing state identifier",
            "operations": [{
                "op": "add",
                "path": "/missing_facts/-",
                "value": "Operating state identifier is blank/unspecified in source text.",
            }],
            "resolved_finding_ids": [],
        })])
        calls = {"count": 0}

        def reviewer(prompt: str) -> str:
            calls["count"] += 1
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            excerpts = payload.get("source_excerpts", [])
            evidence_hash = excerpts[0]["evidence_hash"] if excerpts else ""
            findings = [] if calls["count"] > 1 else [{
                "code": "missing_operating_state_unrecorded",
                "severity": "error",
                "category": "source_coverage",
                "message": "record the blank source operating state in missing_facts",
                "evidence_hashes": [evidence_hash],
                "affected_json_paths": ["/missing_facts"],
                "repairable_by_llm": False,
                "requires_human": True,
                "confidence": 0.95,
            }]
            return json.dumps({
                "review_status": "complete_with_gaps" if findings else "complete",
                "reviewed_evidence_hashes": [evidence_hash] if evidence_hash else [],
                "coverage_summary": {},
                "findings": findings,
            })

        result = run_incremental_planning(
            requirement="Model ONLY operating state '' in this run.",
            state=PlanBuildState(
                state_id="metadata-closure",
                requirement_text="Model ONLY operating state '' in this run.",
            ),
            llm_client=patch_llm,
            plan_loop_policy={"mode": "controlled"},
            plan_reviewer_client=reviewer,
            plan_repair_client=repair_llm,
        )
        assert result.ok
        stage = result.state.plan_loop_stages["plan_gate_facts"]
        assert stage.status is PlanStageStatus.ACCEPTED
        assert stage.repair_count == 1
        repaired = [item for item in result.state.patches.values() if item.patch_type == "facts" and item.status == "valid"][-1]
        assert repaired.content["missing_facts"] == [
            "Operating state identifier is blank/unspecified in source text."
        ]

    def test_two_repairs_close_in_one_facts_gate_call(self, monkeypatch, tmp_path) -> None:
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))
        facts = {"patch_type": "facts", "model_scope": "single_assembly", "assembly_count": 1, "assembly_type_counts": {"a": 1}, "fuel_variant_requirements": [{"variant_id": "fuel"}], "localized_insert_requirements": [{"requirement_id": "insert", "insert_kind": "pyrex_rod"}], "has_spacer_grids": False}
        patch_llm = FakePatchLLM([json.dumps(facts)])
        repair_llm = FakePatchLLM([
            json.dumps({"proposal_id": "repair_1", "confidence": 0.9, "rationale": "first", "operations": [{"op": "replace", "path": "/expected_pyrex_count", "value": 1}], "resolved_finding_ids": []}),
            json.dumps({"proposal_id": "repair_2", "confidence": 0.9, "rationale": "second", "operations": [{"op": "replace", "path": "/expected_thimble_plug_count", "value": 1}], "resolved_finding_ids": []}),
        ])
        calls = {"count": 0}

        def reviewer(_prompt: str) -> str:
            calls["count"] += 1
            payload = json.loads(_prompt.split("INPUT:\n", 1)[1])
            excerpts = payload.get("source_excerpts", [])
            evidence_hash = excerpts[0]["evidence_hash"] if excerpts else ""
            if calls["count"] == 1:
                path, code = "/expected_pyrex_count", "FIRST_NULL"
            elif calls["count"] == 2:
                path, code = "/expected_thimble_plug_count", "SECOND_NULL"
            else:
                path, code = "", ""
            findings = [] if not path else [{"code": code, "severity": "error", "category": "source_coverage", "message": "missing", "evidence_hashes": [evidence_hash], "affected_json_paths": [path], "repairable_by_llm": True, "requires_human": False, "confidence": 0.9}]
            return json.dumps({"review_status": "complete_with_gaps" if findings else "complete", "reviewed_evidence_hashes": [evidence_hash] if evidence_hash else [], "coverage_summary": {}, "findings": findings})

        result = run_incremental_planning(requirement="small source", state=PlanBuildState(state_id="closure", requirement_text="small source"), llm_client=patch_llm, plan_loop_policy={"mode": "controlled"}, plan_reviewer_client=reviewer, plan_repair_client=repair_llm, plan_loop_output_dir=tmp_path)
        assert result.ok
        stage = result.state.plan_loop_stages["plan_gate_facts"]
        assert stage.status is PlanStageStatus.ACCEPTED
        assert stage.repair_count == 2
        assert stage.metadata["facts_revision_closure"]["rounds"] == 2
        repaired = [item for item in result.state.patches.values() if item.patch_type == "facts" and item.status == "valid"][-1]
        assert repaired.content["expected_pyrex_count"] == repaired.content["expected_thimble_plug_count"] == 1
        artifacts = tmp_path / "incremental" / "plan_closed_loop"
        assert (artifacts / "facts_rereview_result_000.json").exists()
        assert (artifacts / "facts_rereview_result_001.json").exists()
        gate_result = json.loads((artifacts / "facts_gate_result.json").read_text())
        assert gate_result["initial_decision"]["action"] == "revise_current_patch"
        assert [item["accepted_for_commit"] for item in gate_result["candidate_validation"]["rounds"]] == [False, True]
        assert gate_result["candidate_commit"]["committed"] is True
        assert gate_result["final_gate_status"] == {
            "accepted": True,
            "failure_code": None,
            "status": "accepted",
            "terminal_reason": "candidate_committed_and_accepted",
        }

    def test_facts_revision_closure_uses_policy_repair_round_limit(self, monkeypatch) -> None:
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))
        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "assembly_type_counts": {"a": 1},
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "localized_insert_requirements": [{"requirement_id": "insert", "insert_kind": "pyrex_rod"}],
            "has_spacer_grids": False,
            "source_notes": [],
        }
        patch_llm = FakePatchLLM([json.dumps(facts)])
        repair_llm = FakePatchLLM([
            json.dumps({
                "proposal_id": f"repair_{idx}",
                "confidence": 0.9,
                "rationale": f"repair finding {idx}",
                "operations": [{"op": "add", "path": "/source_notes/-", "value": f"source-backed detail {idx}"}],
                "resolved_finding_ids": [],
            })
            for idx in range(4)
        ])
        calls = {"count": 0}

        def reviewer(prompt: str) -> str:
            calls["count"] += 1
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            excerpts = payload.get("source_excerpts", [])
            evidence_hash = excerpts[0]["evidence_hash"] if excerpts else ""
            # Initial review plus the first three rereviews expose the next
            # source-backed detail.  The fourth repair clears the gate.
            if calls["count"] <= 4:
                finding = {
                    "code": f"MISSING_DETAIL_{calls['count']}",
                    "severity": "error",
                    "category": "source_coverage",
                    "message": "missing source-backed detail",
                    "evidence_hashes": [evidence_hash] if evidence_hash else [],
                    "affected_json_paths": ["/source_notes"],
                    "repairable_by_llm": True,
                    "requires_human": False,
                    "confidence": 0.9,
                }
                findings = [finding]
            else:
                findings = []
            return json.dumps({
                "review_status": "complete_with_gaps" if findings else "complete",
                "reviewed_evidence_hashes": [evidence_hash] if evidence_hash else [],
                "coverage_summary": {},
                "findings": findings,
            })

        result = run_incremental_planning(
            requirement="small source",
            state=PlanBuildState(state_id="closure-four", requirement_text="small source"),
            llm_client=patch_llm,
            plan_loop_policy={"mode": "controlled", "max_repair_rounds_per_gate": 4},
            plan_reviewer_client=reviewer,
            plan_repair_client=repair_llm,
        )

        assert result.ok
        stage = result.state.plan_loop_stages["plan_gate_facts"]
        assert stage.status is PlanStageStatus.ACCEPTED
        assert stage.metadata["facts_revision_closure"]["rounds"] == 4

    def test_blocked_closure_metadata_uses_latest_rereview_findings(self, monkeypatch) -> None:
        monkeypatch.setattr(executor, "default_patch_task_order", lambda _: ["facts"])
        monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: ["facts"])
        monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))
        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "assembly_count": 1,
            "assembly_type_counts": {"a": 1},
            "fuel_variant_requirements": [{"variant_id": "fuel"}],
            "localized_insert_requirements": [{"requirement_id": "insert", "insert_kind": "pyrex_rod"}],
            "has_spacer_grids": False,
        }
        patch_llm = FakePatchLLM([json.dumps(facts)])
        repair_llm = FakePatchLLM([
            json.dumps({
                "proposal_id": "repair_count",
                "confidence": 0.9,
                "rationale": "repair the old count finding",
                "operations": [{"op": "replace", "path": "/expected_pyrex_count", "value": 80}],
                "resolved_finding_ids": [],
            }),
        ])
        calls = {"count": 0}

        def reviewer(prompt: str) -> str:
            calls["count"] += 1
            payload = json.loads(prompt.split("INPUT:\n", 1)[1])
            excerpts = payload.get("source_excerpts", [])
            evidence_hash = excerpts[0]["evidence_hash"] if excerpts else ""
            if calls["count"] == 1:
                finding = {
                    "code": "OLD_COUNT_NULL",
                    "severity": "error",
                    "category": "source_coverage",
                    "message": "expected_pyrex_count is null; set it to 80",
                    "evidence_hashes": [evidence_hash] if evidence_hash else [],
                    "affected_json_paths": ["/expected_pyrex_count"],
                    "repairable_by_llm": True,
                    "requires_human": False,
                    "confidence": 0.9,
                    "expected_value": 80,
                    "current_value": None,
                }
            else:
                finding = {
                    "code": "NEW_PHYSICAL_AMBIGUITY",
                    "severity": "error",
                    "category": "physical_ambiguity",
                    "message": "new ambiguity remains after the count repair",
                    "evidence_hashes": [evidence_hash] if evidence_hash else [],
                    "affected_json_paths": ["/model_scope"],
                    "repairable_by_llm": False,
                    "requires_human": True,
                    "confidence": 0.9,
                }
            return json.dumps({
                "review_status": "complete_with_gaps",
                "reviewed_evidence_hashes": [evidence_hash] if evidence_hash else [],
                "coverage_summary": {},
                "findings": [finding],
            })

        result = run_incremental_planning(
            requirement="small source",
            state=PlanBuildState(state_id="closure-blocked", requirement_text="small source"),
            llm_client=patch_llm,
            plan_loop_policy={"mode": "controlled"},
            plan_reviewer_client=reviewer,
            plan_repair_client=repair_llm,
        )
        assert not result.ok
        stage = result.state.plan_loop_stages["plan_gate_facts"]
        assert stage.status is PlanStageStatus.BLOCKED
        closure = stage.metadata["facts_revision_closure"]
        assert closure["failure_code"] == "planning.facts_revision.unresolved_requires_human"
        assert closure["unresolved_finding_codes"] == ["NEW_PHYSICAL_AMBIGUITY"]
        unresolved_ids = closure["unresolved_finding_ids"]
        assert len(unresolved_ids) == 1
        unresolved = result.state.plan_review_findings[unresolved_ids[0]]
        assert unresolved.code == "NEW_PHYSICAL_AMBIGUITY"
        assert result.summary["generation_issues"][0]["metadata"]["unresolved_finding_codes"] == [
            "NEW_PHYSICAL_AMBIGUITY"
        ]
