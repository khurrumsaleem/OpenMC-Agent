"""VERA4 Axial Geometry Phase-5 offline qualification.

Reactor-neutral mutation challenges on a VERA4-style multi-component axial
model:
- multi-component axial domain;
- spacer grid bands;
- mass-conserving overlay;
- fuel/guide-tube through-path;
- loading-to-layer attachment;
- overlay material readiness;
- Phase-3B retry routing;
- mutation repair/replay.
"""

from tests._axial_geometry_fixtures import state_with_axial_patches, make_facts_content, make_axial_layers_content, make_axial_overlays_content
from openmc_agent.plan_builder.closed_loop.axial_geometry_preflight import run_axial_geometry_preflight
from openmc_agent.plan_builder.closed_loop.axial_geometry_binding import build_axial_geometry_binding_view
from openmc_agent.plan_builder.closed_loop.axial_geometry_issue_policy import axial_geometry_issue_owner
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanGateId


def _policy():
    return PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled")


def test_vera4_multi_component_axial_domain():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    assert view.axial_domain_cm is not None
    assert len(view.axial_layer_records) >= 3


def test_vera4_spacer_grid_bands():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    grid_overlays = [o for o in view.axial_overlay_records if o.overlay_kind == "spacer_grid"]
    assert len(grid_overlays) >= 2


def test_vera4_mass_conserving_overlay_density():
    overlays = make_axial_overlays_content(overlays=[
        {"overlay_id": "sg1", "overlay_kind": "spacer_grid", "z_min_cm": 20.0, "z_max_cm": 20.5, "target_lattice_id": "lat1", "material_id": "mat_grid", "geometry_mode": "mass_conserving_outer_frame", "effective_density_g_cm3": 7.8, "through_path_preserved": True, "total_mass_g": 100.0},
    ])
    state = state_with_axial_patches(overlays=overlays)
    view = build_axial_geometry_binding_view(state=state)
    assert view.axial_overlay_records[0].density_status == "pass"


def test_vera4_fuel_through_path_preserved():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    lattice_tps = [t for t in view.through_path_records if t.path_kind == "base_lattice"]
    assert len(lattice_tps) >= 1
    assert all(t.preserved for t in lattice_tps)


def test_vera4_loading_to_layer_attachment():
    state = state_with_axial_patches()
    view = build_axial_geometry_binding_view(state=state)
    attached = [l for l in view.lattice_loading_records if l.attachment_status == "attached"]
    assert len(attached) >= 1


def test_vera4_overlay_material_readiness():
    state = state_with_axial_patches()
    preflight = run_axial_geometry_preflight(state=state, policy=_policy())
    codes = [i["code"] for i in preflight.issues]
    assert "axial.overlay_material_missing" not in codes


def test_vera4_mutation_unattached_loading():
    """Declare loading but don't attach it to any layer."""
    layers = make_axial_layers_content(layers=[
        {"layer_id": "l1", "role": "lower_nozzle", "z_min_cm": 0.0, "z_max_cm": 10.0, "fill_type": "material", "fill_id": "mat_nozzle"},
        {"layer_id": "l2", "role": "active_fuel", "z_min_cm": 10.0, "z_max_cm": 90.0, "fill_type": "lattice", "fill_id": "lat1"},
        {"layer_id": "l3", "role": "upper_nozzle", "z_min_cm": 90.0, "z_max_cm": 100.0, "fill_type": "material", "fill_id": "mat_nozzle"},
    ], loadings=[
        {"loading_id": "ld1", "base_lattice_id": "lat1"},
        {"loading_id": "ld_orphan", "base_lattice_id": "lat1"},
    ])
    state = state_with_axial_patches(layers=layers)
    preflight = run_axial_geometry_preflight(state=state, policy=_policy())
    codes = [i["code"] for i in preflight.issues]
    assert "axial.loading_unattached" in codes
    owner = axial_geometry_issue_owner("axial.loading_unattached")
    assert owner is not None
    assert "axial_layers" in owner.owner_patch_types


def test_vera4_mutation_missing_replacement_universe():
    """Overlay references a non-existent target lattice."""
    overlays = make_axial_overlays_content(overlays=[
        {"overlay_id": "sg1", "overlay_kind": "spacer_grid", "z_min_cm": 20.0, "z_max_cm": 20.5, "target_lattice_id": "lat_nonexistent", "material_id": "mat_grid", "geometry_mode": "mass_conserving_outer_frame", "effective_density_g_cm3": 7.8, "through_path_preserved": True, "total_mass_g": 100.0},
        {"overlay_id": "sg2", "overlay_kind": "spacer_grid", "z_min_cm": 50.0, "z_max_cm": 50.5, "target_lattice_id": "lat_nonexistent", "material_id": "mat_grid", "geometry_mode": "mass_conserving_outer_frame", "effective_density_g_cm3": 7.8, "through_path_preserved": True, "total_mass_g": 100.0},
    ])
    state = state_with_axial_patches(overlays=overlays)
    preflight = run_axial_geometry_preflight(state=state, policy=_policy())
    codes = [i["code"] for i in preflight.issues]
    assert "axial.overlay_target_lattice_missing" in codes


def test_vera4_overlay_density_routes_to_axial_overlays():
    """Overlay mass-basis mutation should route to the axial overlay owner."""
    owner = axial_geometry_issue_owner("axial.overlay_density_required")
    assert owner is not None
    assert "axial_overlays" in owner.owner_patch_types
    assert PlanGateId.AXIAL_GEOMETRY in owner.gates_to_invalidate
