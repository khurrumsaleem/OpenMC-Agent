"""Tests for AxialGeometryContractMatrix construction."""

from tests._axial_geometry_fixtures import (
    make_axial_layers_content,
    make_axial_overlays_content,
    state_with_axial_patches,
)
from openmc_agent.plan_builder.closed_loop.axial_geometry_binding import build_axial_geometry_binding_view
from openmc_agent.plan_builder.closed_loop.axial_geometry_evidence import build_axial_geometry_contract_matrix
from openmc_agent.plan_builder.closed_loop.axial_geometry_preflight import run_axial_geometry_preflight
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy


def test_matrix_has_all_nine_row_kinds():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view)
    kinds = {r.row_kind for r in matrix.rows}
    assert "source_domain_coverage" in kinds
    assert "active_fuel_coverage" in kinds
    assert "layer_fill_binding" in kinds
    assert "loading_attachment" in kinds
    assert "overlay_binding" in kinds
    assert "through_path_preservation" in kinds
    assert "spacer_grid_structural_count" in kinds


def test_matrix_layer_fill_binding_passes():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view)
    layer_rows = [r for r in matrix.rows if r.row_kind == "layer_fill_binding"]
    assert len(layer_rows) == 3
    assert all(r.coverage_status == "pass" for r in layer_rows)


def test_matrix_loading_attachment_detected():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view)
    loading_rows = [r for r in matrix.rows if r.row_kind == "loading_attachment"]
    assert len(loading_rows) >= 1
    assert all(r.coverage_status == "pass" for r in loading_rows)


def test_matrix_spacer_grid_count_matches():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view)
    sg_rows = [r for r in matrix.rows if r.row_kind == "spacer_grid_structural_count"]
    assert len(sg_rows) == 1
    assert sg_rows[0].expected_count == 2
    assert sg_rows[0].actual_count == 2
    assert sg_rows[0].coverage_status == "pass"


def test_matrix_has_input_hash():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view)
    assert matrix.input_hash
    assert len(matrix.input_hash) > 0


def test_preflight_blocks_required_spacer_grid_skeleton_without_material():
    overlays = make_axial_overlays_content([
        {
            "overlay_id": "sg1",
            "overlay_kind": "spacer_grid",
            "z_min_cm": 20.0,
            "z_max_cm": 20.5,
            "target_lattice_id": "lat1",
            "geometry_mode": "skeleton",
            "requires_human_confirmation": True,
        },
    ])
    state = state_with_axial_patches(overlays=overlays)
    result = run_axial_geometry_preflight(
        state=state,
        policy=PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled"),
    )
    codes = {issue["code"] for issue in result.issues}
    assert "axial.overlay_skeleton_not_materialized" in codes
    assert "axial.overlay_material_missing" in codes
    assert result.ok is False


def test_preflight_blocks_bare_component_profile_lattice_layer():
    layers = make_axial_layers_content(layers=[
        {"layer_id": "lower", "role": "lower_nozzle", "z_min_cm": 0.0, "z_max_cm": 10.0, "fill_type": "material", "fill_id": "mat_nozzle"},
        {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 10.0, "z_max_cm": 90.0, "fill_type": "lattice", "fill_id": "lat1"},
        {"layer_id": "upper_plenum", "role": "upper_plenum", "z_min_cm": 90.0, "z_max_cm": 95.0, "fill_type": "lattice", "fill_id": "lat1"},
        {"layer_id": "upper", "role": "upper_nozzle", "z_min_cm": 95.0, "z_max_cm": 100.0, "fill_type": "material", "fill_id": "mat_nozzle"},
    ])
    state = state_with_axial_patches(layers=layers)
    result = run_axial_geometry_preflight(
        state=state,
        policy=PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled"),
    )
    codes = {issue["code"] for issue in result.issues}
    assert "axial.base_path_profile_missing" in codes
    assert result.ok is False
