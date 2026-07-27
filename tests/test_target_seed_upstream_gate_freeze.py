"""Regression tests for accepted target-seed upstream gate reuse."""

from __future__ import annotations

from openmc_agent.plan_builder.closed_loop.models import (
    PlanClosedLoopPolicy,
    PlanGateId,
)
from openmc_agent.plan_builder.closed_loop.axial_geometry_reviewer import (
    AxialGeometryReviewResult,
)
from openmc_agent.plan_builder.executor import run_incremental_planning
from tests._axial_geometry_fixtures import state_with_axial_patches


def test_axial_target_seed_reuses_accepted_placement_even_with_stale_hash(monkeypatch) -> None:
    state = state_with_axial_patches(include_profiles=True)
    state.metadata["accepted_plan_build_state_seed"] = True
    state.metadata["target_gate_seed_gate"] = "axial_geometry"
    placement_stage = state.plan_loop_stages["plan_gate_placement"]
    placement_stage.metadata["accepted_input_hash"] = "stale-placement-hash"

    def fail_placement_review(**_kwargs):
        raise AssertionError("accepted upstream Placement Gate must not be re-reviewed")

    def clean_axial_review(**_kwargs):
        return AxialGeometryReviewResult(ok=True, coverage_complete=True)

    monkeypatch.setattr(
        "openmc_agent.plan_builder.closed_loop.placement_reviewer.run_placement_review",
        fail_placement_review,
    )
    monkeypatch.setattr(
        "openmc_agent.plan_builder.closed_loop.axial_geometry_reviewer.run_axial_geometry_review",
        clean_axial_review,
    )

    policy = PlanClosedLoopPolicy(
        mode="controlled",
        stop_after_gate=PlanGateId.AXIAL_GEOMETRY,
        gate_enabled={
            PlanGateId.FACTS: True,
            PlanGateId.MATERIAL_UNIVERSE: True,
            PlanGateId.PLACEMENT: True,
            PlanGateId.AXIAL_GEOMETRY: True,
            PlanGateId.ASSEMBLED_PLAN: False,
        },
        placement_review_mode="controlled",
        material_universe_review_mode="controlled",
        axial_geometry_review_mode="controlled",
    )

    result = run_incremental_planning(
        requirement=state.requirement_text,
        state=state,
        llm_client=lambda _prompt: "{}",
        plan_loop_policy=policy,
        plan_reviewer_client=object(),
    )

    assert all(issue.code != "planning.axial_geometry_requires_accepted_placement" for issue in result.issues)
    assert state.plan_loop_stages["plan_gate_placement"].status.value == "accepted"
    assert any(
        event.event_type == "planning.placement_gate_reused_from_target_seed"
        for event in state.build_log
    )
