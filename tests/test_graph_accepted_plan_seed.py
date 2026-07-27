"""Regression tests for accepted incremental plan-build-state seed handling."""

from __future__ import annotations

from openmc_agent.graph import _receive_requirement
from tests._axial_geometry_fixtures import state_with_axial_patches


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
