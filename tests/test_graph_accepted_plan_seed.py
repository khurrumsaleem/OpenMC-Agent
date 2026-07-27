"""Regression tests for accepted incremental plan-build-state seed handling."""

from __future__ import annotations

from openmc_agent.graph import _receive_requirement, _run_incremental_plan_generation
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanGateId
from openmc_agent.plan_builder.executor import IncrementalExecutionResult
from tests._axial_geometry_fixtures import state_with_axial_patches
from tests._assembled_plan_fixtures import state_with_assembled_plan


def test_receive_requirement_preserves_accepted_plan_build_state_seed() -> None:
    requirement = (
        "Build a multi-assembly reactor model with 17x17 lattice assemblies, "
        "source-backed materials, generated universes, assembly placement, "
        "axial layers, and spacer-grid axial overlays."
    )
    seed = state_with_axial_patches(include_profiles=True)
    seed.requirement_text = requirement

    updates = _receive_requirement(
        {
            "requirement": requirement,
            "accepted_plan_build_state": seed.model_dump(mode="json"),
        }
    )

    assert updates["planning_mode_decision"]["mode"] == "incremental"
    assert updates["plan_build_state"]["state_id"] == seed.state_id
    assert updates["accepted_plan_build_state"]["state_id"] == seed.state_id


def test_assembled_target_seed_enters_incremental_executor(monkeypatch) -> None:
    seed = state_with_assembled_plan()
    calls: list[dict[str, object]] = []

    def fake_run_incremental_planning(**kwargs):
        calls.append(kwargs)
        state = kwargs["state"]
        state.metadata["executor_called_for_target_seed"] = True
        return IncrementalExecutionResult(
            ok=True,
            state=state,
            assembled_plan=None,
            summary={"stopped_after_gate": "assembled_plan"},
            plan_loop_outcome={"status": "stopped_after_gate"},
        )

    def fail_fast_path(*_args, **_kwargs):
        raise AssertionError("assembled target seed must not bypass the incremental executor")

    monkeypatch.setattr(
        "openmc_agent.plan_builder.executor.run_incremental_planning",
        fake_run_incremental_planning,
    )
    monkeypatch.setattr("openmc_agent.graph._write_final_simulation_plan", fail_fast_path)

    result = _run_incremental_plan_generation(
        {
            "requirement": seed.requirement_text,
            "planning_mode_decision": {"mode": "incremental"},
            "plan_build_state": seed.model_dump(mode="json"),
            "accepted_plan_build_state": seed.model_dump(mode="json"),
        },
        patch_llm_client=lambda _prompt: "",
        plan_loop_policy=PlanClosedLoopPolicy(
            mode="controlled",
            stop_after_gate=PlanGateId.ASSEMBLED_PLAN,
            assembled_plan_review_mode="controlled",
        ),
    )

    assert calls
    assert result["plan_build_state"]["metadata"]["accepted_plan_build_state_seed"] is True
    assert result["plan_build_state"]["metadata"]["target_gate_seed_gate"] == "assembled_plan"
