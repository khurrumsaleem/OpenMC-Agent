"""Tests for AxialGeometryEvidencePack."""

from tests._axial_geometry_fixtures import state_with_axial_patches
from openmc_agent.plan_builder.closed_loop.axial_geometry_evidence import (
    axial_geometry_gate_applicable,
    axial_geometry_gate_ready,
    axial_geometry_gate_input_hash,
    build_axial_geometry_evidence_pack,
)
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.state import PlanComponentTask


def _policy():
    return PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled")


def test_applicable_with_axial_domain():
    state = state_with_axial_patches()
    assert axial_geometry_gate_applicable(state) is True


def test_applicable_when_placement_accepted_state_has_pending_axial_tasks():
    """Placement milestone checkpoints can require Axial before patches exist."""
    state = state_with_axial_patches()
    for patch_id, envelope in list(state.patches.items()):
        if envelope.patch_type in {"base_path_axial_profiles", "axial_layers", "axial_overlays"}:
            del state.patches[patch_id]
    state.component_tasks.clear()
    state.add_task(
        PlanComponentTask(
            task_id="task_axial_layers",
            patch_type="axial_layers",
            description="Define core.axial_layers",
        )
    )
    state.add_task(
        PlanComponentTask(
            task_id="task_axial_overlays",
            patch_type="axial_overlays",
            description="Define core.axial_overlays",
            dependencies=["task_axial_layers"],
        )
    )
    state.canonical_task_plan = None
    assert axial_geometry_gate_applicable(state) is True
    assert axial_geometry_gate_ready(state) is False


def test_ready_with_valid_axial_patches():
    state = state_with_axial_patches()
    assert axial_geometry_gate_ready(state) is True


def test_input_hash_stable():
    state = state_with_axial_patches()
    h1 = axial_geometry_gate_input_hash(state, policy=_policy())
    h2 = axial_geometry_gate_input_hash(state, policy=_policy())
    assert h1 == h2


def test_evidence_pack_has_items():
    state = state_with_axial_patches()
    pack = build_axial_geometry_evidence_pack(state=state, policy=_policy())
    assert pack.gate_id.value == "axial_geometry"
    assert len(pack.evidence_items) > 0
    assert pack.binding_view is not None
    assert pack.input_hash
