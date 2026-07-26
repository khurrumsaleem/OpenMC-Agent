"""Reactor-neutral Phase-2 Placement Gate protocol coverage."""

from __future__ import annotations

import json

from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanGateId, PlanFindingCategory, PlanFindingSeverity, PlanReviewFinding, PlacementPatchEdit, PlacementRevisionProposal, PlanStageStatus
from openmc_agent.plan_builder.closed_loop.controller import initialize_gate_stage, transition_stage
from openmc_agent.plan_builder.closed_loop.placement_evidence import (
    build_placement_evidence_pack, build_placement_binding_view,
    placement_gate_applicable, placement_gate_input_hash, placement_gate_ready,
)
from openmc_agent.plan_builder.closed_loop.placement_preflight import run_placement_preflight
from openmc_agent.plan_builder.closed_loop.placement_reviewer import run_placement_review
from openmc_agent.plan_builder.closed_loop.placement_issue_policy import placement_issue_owner
from openmc_agent.plan_builder.closed_loop.review_io import StructuredReviewCallSpec, run_structured_review_call
from openmc_agent.plan_builder.closed_loop.models import PlacementReviewModelOutput
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
from openmc_agent.plan_builder import executor
from openmc_agent.plan_builder.executor import run_incremental_planning
from openmc_agent.plan_builder.closed_loop.fingerprints import compute_candidate_hash
from openmc_agent.plan_builder.closed_loop.placement_revision import commit_placement_revision, evaluate_placement_revision


def _state(*, missing_intent: bool = False) -> PlanBuildState:
    facts = {
        "patch_type": "facts", "model_scope": "single_assembly",
        "localized_insert_requirements": [{
            "requirement_id": "absorber", "insert_kind": "absorber_insert", "assembly_type_ids": [],
            "expected_coordinate_count_per_assembly": 1, "host_kind": "guide_tube", "required_profile_id": "p1",
            "required_segment_roles": ["absorber"], "expected_insert_universe_ids": ["abs"], "anchor_z_cm": 1.0,
            "control_state_id": "inserted",
        }],
    }
    universes = {"patch_type": "universes", "universes": [
        {"universe_id": "fuel", "kind": "fuel_pin", "cells": []},
        {"universe_id": "abs", "kind": "custom", "cells": []},
    ]}
    profiles = {"patch_type": "localized_insert_profiles", "profiles": [{"profile_id": "p1", "anchor_kind": "bottom", "anchor_z_cm": 1.0, "segments": [{"segment_id": "s", "relative_z_min_cm": 0, "relative_z_max_cm": 1, "universe_id": "abs", "role": "absorber"}]}]}
    pin = {"patch_type": "pin_map", "lattice_size": [3, 3], "default_universe_id": "fuel", "guide_tube_coords": [[1, 1]], "instrument_tube_coords": [], "localized_insert_intents": [] if missing_intent else [{"insert_id": "i", "insert_kind": "absorber_insert", "insert_universe_id": "abs", "coordinates": [[1, 1]], "axial_profile_id": "p1", "anchor_z_cm": 1.0, "control_state_id": "inserted"}]}
    state = PlanBuildState(state_id="placement", requirement_text="reactor-neutral placement source")
    for patch in (facts, universes, profiles, pin):
        state.add_patch(PlanPatchEnvelope(patch_id=patch["patch_type"], patch_type=patch["patch_type"], content=patch, status="valid"))
    return state


def test_single_assembly_view_preflight_and_hash_are_deterministic() -> None:
    state = _state()
    assert placement_gate_applicable(state) and placement_gate_ready(state)
    view = build_placement_binding_view(state=state)
    assert view.scope_kind == "single_assembly" and view.assembly_scopes[0].scope_id == "single_assembly"
    assert not run_placement_preflight(state=state)["issues"]
    assert placement_gate_input_hash(state) == placement_gate_input_hash(state)
    pack = build_placement_evidence_pack(state=state, policy=PlanClosedLoopPolicy(mode="advisory"))
    assert pack.contract_matrix.rows[0].static_binding_status == "pass"
    assert all(item.canonical_hash for item in pack.evidence_items)


def test_missing_intent_is_deterministic_placement_failure() -> None:
    issues = run_placement_preflight(state=_state(missing_intent=True))["issues"]
    assert {item["code"] for item in issues} >= {"localized_insert.required_placement_missing"}


def test_multi_assembly_placement_owner_routing_excludes_pin_map() -> None:
    assert placement_issue_owner(
        "localized_insert.anchor_mismatch",
        canonical_scope="multi_assembly",
    )["owner_patch_types"] == ["assembly_catalog", "localized_insert_profiles"]
    assert placement_issue_owner(
        "localized_insert.control_state_mismatch",
        canonical_scope="multi_assembly",
    )["owner_patch_types"] == ["assembly_catalog"]


def test_reviewer_rejects_unknown_evidence_ref() -> None:
    state = _state()
    policy = PlanClosedLoopPolicy(mode="advisory")
    pack = build_placement_evidence_pack(state=state, policy=policy)
    payload = {"review_status": "complete", "reviewed_contract_row_ids": ["absorber"], "reviewed_evidence_refs": [item.ref_id for item in pack.evidence_items], "coverage_summary": {"omitted_contract_row_count": 0}, "findings": [{"code": "placement.semantic_gap", "severity": "error", "category": "placement_gap", "message": "bad ref", "evidence_refs": ["not-present"], "affected_contract_rows": ["absorber"], "affected_json_paths": ["/localized_insert_intents"], "repairable_by_llm": False, "requires_human": False, "confidence": 0.9}]}
    result = run_placement_review(evidence_pack=pack, reviewer_client=lambda _: json.dumps(payload), state=state, policy=policy)
    assert result.ok and result.rejected[0]["code"] == "placement_review.unknown_evidence_ref"


def test_placement_review_complete_with_gaps_is_coverage_complete() -> None:
    state = _state()
    policy = PlanClosedLoopPolicy(mode="advisory")
    pack = build_placement_evidence_pack(state=state, policy=policy)
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_contract_row_ids": ["absorber"],
        "reviewed_evidence_refs": [item.ref_id for item in pack.evidence_items],
        "coverage_summary": {
            "reviewed_contract_row_count": 1,
            "omitted_contract_row_count": 0,
            "reviewed_evidence_item_count": len(pack.evidence_items),
            "omitted_evidence_item_count": 0,
        },
        "findings": [{
            "code": "placement.semantic_warning",
            "severity": "warning",
            "category": "representation_error",
            "message": "covered review with a non-blocking gap",
            "evidence_refs": [pack.evidence_items[0].ref_id],
            "affected_contract_rows": ["absorber"],
            "affected_json_paths": ["/localized_insert_intents"],
            "repairable_by_llm": False,
            "requires_human": True,
            "confidence": 0.8,
        }],
    }
    result = run_placement_review(evidence_pack=pack, reviewer_client=lambda _: json.dumps(payload), state=state, policy=policy)
    assert result.ok
    assert result.coverage_complete
    assert result.error_code == ""
    assert [finding.code for finding in result.findings] == ["placement.semantic_warning"]


def test_placement_review_missing_coverage_ref_fails_closed() -> None:
    state = _state()
    policy = PlanClosedLoopPolicy(mode="advisory")
    pack = build_placement_evidence_pack(state=state, policy=policy)
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_contract_row_ids": ["absorber"],
        "reviewed_evidence_refs": [item.ref_id for item in pack.evidence_items[:-1]],
        "coverage_summary": {"omitted_evidence_item_count": 1},
        "findings": [],
    }
    result = run_placement_review(evidence_pack=pack, reviewer_client=lambda _: json.dumps(payload), state=state, policy=policy)
    assert not result.ok
    assert result.error_code == "placement_review.coverage_incomplete"


def test_review_io_uses_last_schema_valid_embedded_object() -> None:
    state = PlanBuildState(state_id="io", requirement_text="r")
    policy = PlanClosedLoopPolicy(mode="advisory")
    raw = 'draft {"wrong":true} final {"review_status":"complete","findings":[],"reviewed_evidence_hashes":[],"coverage_summary":{},"concise_summary":""}'
    result = run_structured_review_call(client=lambda _: raw, initial_prompt="input", retry_prompt_builder=lambda raw, error: "retry", output_model=PlacementReviewModelOutput, call_spec=StructuredReviewCallSpec(role_id="test", gate_id=PlanGateId.PLACEMENT, schema_name="PlacementReviewModelOutput", json_schema=PlacementReviewModelOutput.model_json_schema(), artifact_prefix="test"), state=state, stage=None, policy=policy)
    # Facts-shaped result does not satisfy Placement output because the latter
    # has different coverage fields; robust extraction therefore retries and
    # fails rather than accepting a merely feature-shaped object.
    assert not result.ok and result.schema_retry_count == 1


def test_controlled_gate_accepts_after_facts_and_placement_review(monkeypatch) -> None:
    state = _state()
    monkeypatch.setattr(executor, "default_patch_task_order", lambda _: [])
    monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: [])
    monkeypatch.setattr(executor, "assemble_state_if_ready", lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}))

    def reviewer(prompt: str) -> str:
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        if "source_excerpts" in payload:
            return json.dumps({"review_status": "complete", "reviewed_evidence_hashes": [item["evidence_hash"] for item in payload["source_excerpts"]], "coverage_summary": {}, "findings": []})
        return json.dumps({"review_status": "complete", "reviewed_contract_row_ids": [item["requirement_id"] for item in payload["contract_matrix"]["rows"]], "reviewed_evidence_refs": [item["ref_id"] for item in payload["evidence_items"]], "coverage_summary": {"omitted_contract_row_count": 0}, "findings": []})

    result = run_incremental_planning(
        requirement=state.requirement_text, state=state, llm_client=lambda _: (_ for _ in ()).throw(AssertionError("no proposer")),
        plan_loop_policy={"mode": "controlled", "gate_enabled": {"facts": True, "placement": True}}, plan_reviewer_client=reviewer,
    )
    assert result.ok
    assert result.state.plan_loop_stages["plan_gate_facts"].status.value == "accepted"
    assert result.state.plan_loop_stages["plan_gate_placement"].status.value == "accepted"


def test_previously_skipped_placement_stage_reopens_when_inputs_become_applicable(monkeypatch) -> None:
    """A stale not-applicable checkpoint cannot cause skipped -> reviewing."""
    state = _state()
    stage = initialize_gate_stage(PlanGateId.PLACEMENT, [])
    transition_stage(stage, PlanStageStatus.SKIPPED)
    stage.metadata["reason"] = "not_applicable"
    state.plan_loop_stages[stage.stage_id] = stage
    monkeypatch.setattr(executor, "default_patch_task_order", lambda _: [])
    monkeypatch.setattr(executor, "required_patch_types_for_state", lambda _: [])
    monkeypatch.setattr(
        executor,
        "assemble_state_if_ready",
        lambda state, **_: state.model_copy(update={"assembled_plan": {"ok": True}}),
    )

    def reviewer(prompt: str) -> str:
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        return json.dumps({
            "review_status": "complete",
            "reviewed_contract_row_ids": [row["requirement_id"] for row in payload["contract_matrix"]["rows"]],
            "reviewed_evidence_refs": [item["ref_id"] for item in payload["evidence_items"]],
            "coverage_summary": {"omitted_contract_row_count": 0},
            "findings": [],
        })

    result = run_incremental_planning(
        requirement=state.requirement_text,
        state=state,
        llm_client=lambda _: (_ for _ in ()).throw(AssertionError("no proposer")),
        plan_loop_policy={"mode": "advisory", "gate_enabled": {"placement": True}},
        plan_reviewer_client=reviewer,
    )
    assert result.ok
    assert result.state.plan_loop_stages["plan_gate_placement"].status is PlanStageStatus.REVIEWED
    assert any(event.event_type == "planning.placement_gate_reopened" for event in result.state.build_log)


def test_vera4_static_placement_contract_and_mutation() -> None:
    """Offline static challenge: no provider, reference, or gold data lookup."""
    from scripts.vera4_base_fixture import build_all_vera4_patches

    state = PlanBuildState(state_id="vera4-placement", requirement_text="fixture")
    for patch in build_all_vera4_patches():
        content = patch.model_dump(mode="json")
        state.add_patch(PlanPatchEnvelope(patch_id=content["patch_type"], patch_type=content["patch_type"], content=content, status="valid", source="fixture"))
    baseline = run_placement_preflight(state=state)
    assert not baseline["issues"]
    catalog = next(item for item in state.patches.values() if item.patch_type == "assembly_catalog")
    center = next(item for item in catalog.content["assembly_types"] if item["assembly_type_id"] == "center_rcca")
    center["pin_map"]["localized_insert_intents"] = []
    mutated = run_placement_preflight(state=state)
    assert "localized_insert.required_placement_missing" in {item["code"] for item in mutated["issues"]}


def test_transactional_pin_map_revision_is_clone_then_atomic_commit() -> None:
    state = _state(missing_intent=True)
    pin = next(item for item in state.patches.values() if item.patch_type == "pin_map")
    finding = PlanReviewFinding(
        gate_id="placement", code="localized_insert.required_placement_missing", severity=PlanFindingSeverity.ERROR,
        category=PlanFindingCategory.PLACEMENT_GAP, message="missing intent", affected_patch_types=["pin_map"],
        affected_json_paths=["/localized_insert_intents"], repairable_by_llm=True, requires_human=False, confidence=1.0,
    )
    proposal = PlacementRevisionProposal(
        proposal_id="proposal", edits=[PlacementPatchEdit(
            patch_type="pin_map", patch_id=pin.patch_id,
            expected_patch_hash=compute_candidate_hash(target_patch_type="pin_map", candidate_patch=pin.content),
            operations=[{"op": "add", "path": "/localized_insert_intents/-", "value": {"insert_id": "i", "insert_kind": "absorber_insert", "insert_universe_id": "abs", "coordinates": [[1, 1]], "axial_profile_id": "p1", "anchor_z_cm": 1.0, "control_state_id": "inserted"}}],
        )], resolved_finding_ids=[finding.finding_id], confidence=1.0,
    )
    evaluation = evaluate_placement_revision(state=state, proposal=proposal, findings=[finding], prior_candidate_hashes=[])
    assert evaluation.accepted
    old_content = dict(pin.content)
    committed = commit_placement_revision(state=state, evaluated=evaluation, proposal_id=proposal.proposal_id)
    assert committed and pin.content == old_content and pin.status == "repaired"
    assert len([item for item in state.patches.values() if item.patch_type == "pin_map" and item.status == "valid"]) == 1
