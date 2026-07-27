"""Tests for Axial Geometry retry request routing."""

from openmc_agent.plan_builder.closed_loop.axial_geometry_issue_policy import axial_geometry_issue_owner
from openmc_agent.plan_builder.closed_loop.retry_controller import (
    compile_retry_execution_plan,
    normalize_retry_request,
)
from openmc_agent.plan_builder.closed_loop.retry_owner_policy import retry_owner_policy
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanGateId
from openmc_agent.plan_builder.closed_loop.retry_models import RetryTriggerOrigin
from tests._axial_geometry_fixtures import state_with_axial_patches


def test_retry_owner_policy_routes_axial_layer_codes():
    policy = retry_owner_policy("axial.layer_overlap", {"owner_patch_type": "axial_layers"})
    assert policy is not None
    assert "axial_layers" in policy.owner_patch_types
    assert PlanGateId.AXIAL_GEOMETRY in policy.gates_to_invalidate


def test_retry_owner_policy_routes_axial_overlay_codes():
    policy = retry_owner_policy("axial.overlay_interval_invalid")
    assert policy is not None
    assert "axial_overlays" in policy.owner_patch_types


def test_retry_owner_policy_routes_facts_dependency():
    policy = retry_owner_policy("axial.domain_missing")
    assert policy is not None
    assert "facts" in policy.owner_patch_types
    assert PlanGateId.FACTS in policy.gates_to_invalidate


def test_retry_owner_policy_routes_overlay_density_to_axial_overlays():
    policy = retry_owner_policy("axial.overlay_density_required")
    assert policy is not None
    assert "axial_overlays" in policy.owner_patch_types


def test_single_owner_per_finding():
    """Each finding maps to exactly one owner (or one set of mutually-compatible owners)."""
    for code in ("axial.layer_overlap", "axial.overlay_interval_invalid", "axial.base_path_segment_gap"):
        policy = retry_owner_policy(code)
        assert policy is not None
        assert len(policy.owner_patch_types) >= 1


def test_axial_target_seed_rejects_upstream_owner_retry() -> None:
    state = state_with_axial_patches(include_profiles=True)
    state.metadata["accepted_plan_build_state_seed"] = True
    state.metadata["target_gate_seed_gate"] = "axial_geometry"
    request = normalize_retry_request(
        {"code": "axial.localized_insert_profile_missing"},
        state=state,
        origin=RetryTriggerOrigin.AXIAL_GEOMETRY_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["localized_insert_profiles"]

    try:
        compile_retry_execution_plan(
            request,
            state,
            PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled"),
        )
    except ValueError as exc:
        assert "planning.retry_owner_frozen_by_target_seed" in str(exc)
    else:
        raise AssertionError("expected upstream owner retry to be frozen")


def test_axial_target_seed_limits_axial_owner_invalidation() -> None:
    state = state_with_axial_patches(include_profiles=True)
    state.metadata["accepted_plan_build_state_seed"] = True
    state.metadata["target_gate_seed_gate"] = "axial_geometry"
    request = normalize_retry_request(
        {"code": "axial.layer_overlap"},
        state=state,
        origin=RetryTriggerOrigin.AXIAL_GEOMETRY_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["axial_layers"]

    plan = compile_retry_execution_plan(
        request,
        state,
        PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled"),
    )

    assert set(plan.invalidation_patch_types) <= {
        "base_path_axial_profiles",
        "axial_layers",
        "axial_overlays",
    }
    assert plan.gates_to_invalidate == [PlanGateId.AXIAL_GEOMETRY]


def test_axial_target_seed_rejects_task_plan_retry() -> None:
    state = state_with_axial_patches(include_profiles=True)
    state.metadata["accepted_plan_build_state_seed"] = True
    state.metadata["target_gate_seed_gate"] = "axial_geometry"
    request = normalize_retry_request(
        {"code": "axial.required_patch_omitted"},
        state=state,
        origin=RetryTriggerOrigin.AXIAL_GEOMETRY_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["planning_task_plan"]

    try:
        compile_retry_execution_plan(
            request,
            state,
            PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled"),
        )
    except ValueError as exc:
        assert "planning.retry_owner_frozen_by_target_seed" in str(exc)
    else:
        raise AssertionError("expected task-plan retry to be frozen")
