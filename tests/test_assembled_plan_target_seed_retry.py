"""Regression tests for Assembled target seed retry isolation."""

from __future__ import annotations

from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.closed_loop.retry_controller import (
    compile_retry_execution_plan,
    normalize_retry_request,
)
from openmc_agent.plan_builder.closed_loop.retry_models import RetryTriggerOrigin
from tests._assembled_plan_fixtures import state_with_assembled_plan


def _seed_state():
    state = state_with_assembled_plan()
    state.metadata["accepted_plan_build_state_seed"] = True
    state.metadata["target_gate_seed_gate"] = "assembled_plan"
    return state


def test_assembled_target_seed_rejects_facts_owner_retry() -> None:
    state = _seed_state()
    request = normalize_retry_request(
        {"code": "assembled.root_missing"},
        state=state,
        origin=RetryTriggerOrigin.ASSEMBLED_PLAN_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["facts"]

    try:
        compile_retry_execution_plan(
            request,
            state,
            PlanClosedLoopPolicy(mode="controlled", assembled_plan_review_mode="controlled"),
        )
    except ValueError as exc:
        assert "planning.retry_owner_frozen_by_target_seed" in str(exc)
    else:
        raise AssertionError("expected facts owner retry to be frozen")


def test_assembled_target_seed_rejects_axial_owner_retry() -> None:
    state = _seed_state()
    request = normalize_retry_request(
        {"code": "assembled.required_axial_layer_unreachable"},
        state=state,
        origin=RetryTriggerOrigin.ASSEMBLED_PLAN_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["axial_layers"]

    try:
        compile_retry_execution_plan(
            request,
            state,
            PlanClosedLoopPolicy(mode="controlled", assembled_plan_review_mode="controlled"),
        )
    except ValueError as exc:
        assert "planning.retry_owner_frozen_by_target_seed" in str(exc)
    else:
        raise AssertionError("expected axial owner retry to be frozen")
