"""Tests for the deterministic assembler (Phase 3)."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.assembler import (
    PlanAssemblyResult,
    expand_pin_map,
    assemble_simulation_plan_from_patches,
)
from openmc_agent.plan_builder.patches import (
    AxialLayerPatchItem,
    AxialLayersPatch,
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
    CellLayerPatch,
    CoordinateConvention,
    FactsPatch,
    LatticeLoadingPatchItem,
    LocalizedInsertIntentPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
    PinMapPatch,
    SettingsPatch,
    UniverseSpecPatch,
    UniversesPatch,
)
from openmc_agent.plan_builder.state import (
    PlanBuildState,
    PlanPatchEnvelope,
    add_validated_patch_to_state,
    assemble_state_if_ready,
)
from openmc_agent.plan_builder.validators import (
    PatchValidationContext,
    validate_patch,
)


# ---------------------------------------------------------------------------
# Helper: minimal valid patch set for 3D assembly
# ---------------------------------------------------------------------------


def _minimal_3d_patches() -> list:
    return [
        FactsPatch(
            benchmark_id="TEST",
            lattice_size=(17, 17),
            pin_pitch_cm=1.26,
            assembly_pitch_cm=21.42,
            has_axial_geometry=True,
            has_spacer_grids=True,
            active_fuel_region_cm=(0.0, 100.0),
        ),
        MaterialsPatch(materials=[
            MaterialSpecPatch(material_id="fuel", name="UO2", role="fuel", density_g_cm3=10.0),
            MaterialSpecPatch(material_id="water", name="Water", role="coolant", density_g_cm3=0.74),
            MaterialSpecPatch(material_id="clad", name="Zircaloy-4", role="cladding",
                              density_g_cm3=6.56, composition={"Zr": 1.0},
                              composition_status="approximate",
                              warnings=["Zircaloy-4 approximated as pure Zr"]),
        ]),
        UniversesPatch(universes=[
            UniverseSpecPatch(universe_id="fuel_pin", kind="fuel_pin", cells=[
                CellLayerPatch(id="fuel", role="fuel", material_id="fuel", region_kind="cylinder"),
                CellLayerPatch(id="clad", role="cladding", material_id="clad", region_kind="annulus"),
                CellLayerPatch(id="water", role="coolant", material_id="water", region_kind="background"),
            ]),
            UniverseSpecPatch(universe_id="gt", kind="guide_tube", cells=[
                CellLayerPatch(id="inner_water", role="coolant", material_id="water", region_kind="cylinder"),
                CellLayerPatch(id="wall", role="cladding", material_id="clad", region_kind="annulus"),
                CellLayerPatch(id="bg", role="background", material_id="water", region_kind="background"),
            ]),
        ]),
        PinMapPatch(
            lattice_size=(17, 17),
            default_universe_id="fuel_pin",
            guide_tube_coords=[(2, 2), (2, 5)],
            coordinate_convention=CoordinateConvention(index_base=0),
        ),
        AxialLayersPatch(layers=[
            AxialLayerPatchItem(
                layer_id="active_fuel", role="active_fuel",
                z_min_cm=0.0, z_max_cm=100.0,
                fill_type="lattice", fill_id="assembly_lattice",
            ),
        ]),
        AxialOverlaysPatch(overlays=[
            AxialOverlayPatchItem(
                overlay_id="grid1", overlay_kind="spacer_grid",
                z_min_cm=10.0, z_max_cm=12.0,
                target_lattice_id="assembly_lattice",
                material_id="clad",
                geometry_mode="homogenized_open_region",
                through_path_preserved=True,
            ),
        ]),
        SettingsPatch(),
    ]


# ---------------------------------------------------------------------------
# 1. Assemble minimal 3D plan from valid patches
# ---------------------------------------------------------------------------


def test_assemble_minimal_3d_plan() -> None:
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    assert result.plan is not None
    cm = result.plan.complex_model
    assert len(cm.materials) >= 2
    assert len(cm.universes) >= 2
    assert len(cm.lattices) == 1
    assert cm.core is not None
    assert len(cm.core.axial_layers) >= 1
    assert len(cm.core.axial_overlays) >= 1


def test_assembler_closes_pin_universe_and_honors_explicit_r_min() -> None:
    """LLM cylinder labels must not discard r_min annuli or outer moderator."""
    patches = _minimal_3d_patches()
    universes = next(p for p in patches if isinstance(p, UniversesPatch))
    fuel_pin = next(u for u in universes.universes if u.universe_id == "fuel_pin")
    fuel_pin.cells = [
        CellLayerPatch(
            id="fuel", role="fuel", material_id="fuel",
            region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.40,
        ),
        CellLayerPatch(
            id="gap", role="gap", material_id="water",
            region_kind="cylinder", r_min_cm=0.40, r_max_cm=0.42,
        ),
        CellLayerPatch(
            id="clad", role="cladding", material_id="clad",
            region_kind="cylinder", r_min_cm=0.42, r_max_cm=0.48,
        ),
    ]

    result = assemble_simulation_plan_from_patches(patches)
    assert result.ok is True
    assert result.plan is not None
    model = result.plan.complex_model
    fuel_universe = next(u for u in model.universes if u.id == "fuel_pin")
    assert "fuel_pin_outer_moderator" in fuel_universe.cell_ids
    regions = {region.id: region.expression for region in model.regions}
    assert regions["reg_fuel_pin_gap_annulus"] == "+surf_fuel_pin_fuel -surf_fuel_pin_gap"
    assert regions["reg_fuel_pin_clad_annulus"] == "+surf_fuel_pin_gap -surf_fuel_pin_clad"
    assert regions["reg_fuel_pin_outer_moderator_out"] == "+surf_fuel_pin_clad"


def test_assembler_does_not_duplicate_explicit_background() -> None:
    """An input-provided outer background remains the sole exterior cell."""
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    assert result.plan is not None
    fuel_universe = next(
        universe for universe in result.plan.complex_model.universes
        if universe.id == "fuel_pin"
    )
    assert "fuel_pin_outer_moderator" not in fuel_universe.cell_ids


def test_water_cell_at_half_pitch_remains_renderable() -> None:
    """A moderator cylinder may touch a square lattice-cell side boundary."""
    from openmc_agent.renderers.assembly import RectAssemblyRenderer

    patches = _minimal_3d_patches()
    universes = next(p for p in patches if isinstance(p, UniversesPatch))
    universes.universes.append(UniverseSpecPatch(
        universe_id="water_cell",
        kind="water_cell",
        cells=[CellLayerPatch(
            id="water", role="coolant", material_id="water",
            region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.63,
        )],
    ))
    pin_map = next(p for p in patches if isinstance(p, PinMapPatch))
    pin_map.water_cell_coords = [(0, 0)]

    result = assemble_simulation_plan_from_patches(patches)
    assert result.ok is True
    assert result.plan is not None
    capability = RectAssemblyRenderer().can_render(result.plan)
    assert "surface.cylinder_radius_invalid" not in [
        issue.code for issue in capability.issues
    ]


def test_missing_guide_universe_degrades_to_water_cell_without_count_drift() -> None:
    patches = _minimal_3d_patches()
    universes = next(p for p in patches if isinstance(p, UniversesPatch))
    universes.universes = [
        univ for univ in universes.universes
        if univ.kind != "guide_tube"
    ]
    universes.universes.append(UniverseSpecPatch(
        universe_id="water_cell",
        kind="water_cell",
        cells=[CellLayerPatch(
            id="water", role="coolant", material_id="water", region_kind="background",
        )],
    ))

    result = assemble_simulation_plan_from_patches(patches)

    assert result.ok is True
    assert result.plan is not None
    lattice = result.plan.complex_model.lattices[0]
    assert lattice.universe_pattern[2][2] == "water_cell"
    assert lattice.universe_pattern[2][5] == "water_cell"
    assert result.summary["actual_pin_counts"]["guide_tube"] == 2
    assert "assembly.pin_map.special_universe_degraded_to_water_cell" in [
        issue.code for issue in result.issues
    ]


# ---------------------------------------------------------------------------
# 2. Missing required patch fails cleanly
# ---------------------------------------------------------------------------


def test_missing_required_patch_fails() -> None:
    patches = _minimal_3d_patches()
    # Remove pin_map
    patches = [p for p in patches if getattr(p, "patch_type", "") != "pin_map"]
    result = assemble_simulation_plan_from_patches(patches)
    assert result.ok is False
    codes = [i.code for i in result.issues]
    assert "assembly.missing_patch" in codes


# ---------------------------------------------------------------------------
# 3. Pin map expansion 17x17 count
# ---------------------------------------------------------------------------


def test_pin_map_expansion_17x17_count() -> None:
    pin_map = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
    )
    grid = expand_pin_map(pin_map)
    assert len(grid) == 17
    assert all(len(row) == 17 for row in grid)
    total = sum(len(row) for row in grid)
    assert total == 289


# ---------------------------------------------------------------------------
# 4. Pin map special coordinates replace default
# ---------------------------------------------------------------------------


def test_pin_map_special_coords_replace() -> None:
    pin_map = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        guide_tube_coords=[(2, 2), (2, 5), (3, 3)],
        coordinate_convention=CoordinateConvention(index_base=0),
    )
    universe_ids = {
        "fuel_pin": "fuel_pin",
        "guide_tube": "gt",
    }
    grid = expand_pin_map(pin_map, universe_ids=universe_ids)
    assert grid[2][2] == "gt"
    assert grid[2][5] == "gt"
    assert grid[3][3] == "gt"  # pyrex host position is guide tube in base lattice
    assert grid[0][0] == "fuel_pin"


# ---------------------------------------------------------------------------
# 5. Pin map coordinate convention 1-indexed normalized
# ---------------------------------------------------------------------------


def test_pin_map_1_indexed_convention() -> None:
    pin_map = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        guide_tube_coords=[(1, 1)],
        coordinate_convention=CoordinateConvention(index_base=1),
    )
    universe_ids = {"fuel_pin": "fuel_pin", "guide_tube": "gt"}
    grid = expand_pin_map(pin_map, universe_ids=universe_ids)
    assert grid[0][0] == "gt"  # (1,1) 1-indexed -> (0,0) 0-indexed
    assert grid[0][1] == "fuel_pin"


# ---------------------------------------------------------------------------
# 6. Pin map overlap blocked before assembly
# ---------------------------------------------------------------------------


def test_pin_map_overlap_blocked() -> None:
    # With localized inserts separated from base paths, overlapping guide_tube
    # and instrument_tube coords should still raise an error.
    pin_map = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        guide_tube_coords=[(5, 5)],
        instrument_tube_coords=[(5, 5)],
        coordinate_convention=CoordinateConvention(index_base=0),
    )
    universe_ids = {"fuel_pin": "fuel_pin", "guide_tube": "gt", "instrument_tube": "it"}
    with pytest.raises(ValueError, match="overlap|assigned"):
        expand_pin_map(pin_map, universe_ids=universe_ids)


# ---------------------------------------------------------------------------
# 7. Assembled axial layers preserve z ranges
# ---------------------------------------------------------------------------


def test_axial_layers_preserve_z_ranges() -> None:
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    layers = result.plan.complex_model.core.axial_layers
    fuel_layer = next(l for l in layers if l.id == "active_fuel")
    assert fuel_layer.z_min_cm == 0.0
    assert fuel_layer.z_max_cm == 100.0
    assert fuel_layer.fill.type == "lattice"
    assert fuel_layer.fill.id == "assembly_lattice"


def test_axial_insert_coords_keep_guide_tube_base_for_unseen_model() -> None:
    patches = _minimal_3d_patches()
    materials = next(p for p in patches if getattr(p, "patch_type", "") == "materials")
    materials.materials.append(
        MaterialSpecPatch(
            material_id="pyrex", name="Pyrex", role="absorber", density_g_cm3=2.23,
        )
    )
    universes = next(p for p in patches if getattr(p, "patch_type", "") == "universes")
    universes.universes.append(
        UniverseSpecPatch(universe_id="pyrex_rod", kind="pyrex_rod", cells=[
            CellLayerPatch(id="pyrex", role="absorber", material_id="pyrex", region_kind="cylinder"),
        ])
    )
    pin_map = next(p for p in patches if getattr(p, "patch_type", "") == "pin_map")
    pin_map.guide_tube_coords = [(3, 6)]
    pin_map.localized_insert_intents = [
        LocalizedInsertIntentPatchItem(
            insert_id="pyrex_1",
            insert_kind="pyrex_rod",
            host_kind="guide_tube",
            insert_universe_id="pyrex_rod",
            coordinates=[(3, 6)],
            z_min_cm=15.0,
            z_max_cm=376.0,
            application_mode="nested_component_override",
            component_role="absorber",
            preserve_component_roles=["tube_wall"],
        ),
    ]
    pin_map.coordinate_convention = CoordinateConvention(index_base=1)
    axial_layers = next(p for p in patches if getattr(p, "patch_type", "") == "axial_layers")
    axial_layers.layers[0].loading_id = "base_loading"
    axial_layers.lattice_loadings = [
        LatticeLoadingPatchItem(
            loading_id="base_loading",
            base_lattice_id="assembly_lattice",
            derived_lattice_id="assembly_lattice",
            overrides={},
            purpose="base lattice",
        )
    ]

    result = assemble_simulation_plan_from_patches(patches)

    assert result.ok is True
    model = result.plan.complex_model
    lattice = model.lattices[0]
    # Base lattice has guide tube at insert position
    assert lattice.universe_pattern[2][5] == "gt"
    # Loading was derived
    assert any(l.id == "localized_insert_pyrex_1" for l in model.lattice_loadings)


def test_existing_llm_insert_loading_is_normalized_and_pruned() -> None:
    patches = _minimal_3d_patches()
    materials = next(p for p in patches if getattr(p, "patch_type", "") == "materials")
    materials.materials.extend([
        MaterialSpecPatch(
            material_id="pyrex", name="Pyrex", role="absorber", density_g_cm3=2.23,
        ),
        MaterialSpecPatch(
            material_id="steel", name="Steel", role="structure", density_g_cm3=7.9,
        ),
    ])
    universes = next(p for p in patches if getattr(p, "patch_type", "") == "universes")
    universes.universes.extend([
        UniverseSpecPatch(universe_id="pyrex_rod", kind="pyrex_rod", cells=[
            CellLayerPatch(id="pyrex", role="poison", material_id="pyrex", region_kind="cylinder"),
        ]),
        UniverseSpecPatch(universe_id="thimble_plug", kind="thimble_plug", cells=[
            CellLayerPatch(id="plug", role="plug", material_id="steel", region_kind="cylinder"),
        ]),
    ])
    pin_map = next(p for p in patches if getattr(p, "patch_type", "") == "pin_map")
    pin_map.guide_tube_coords = [(3, 6), (3, 9), (6, 6), (9, 3)]
    pin_map.localized_insert_intents = [
        LocalizedInsertIntentPatchItem(
            insert_id="pyrex_group",
            insert_kind="pyrex_rod",
            host_kind="guide_tube",
            insert_universe_id="pyrex_rod",
            coordinates=[(3, 6)],
            z_min_cm=15.0,
            z_max_cm=376.0,
            component_role="absorber",
            preserve_component_roles=["tube_wall"],
        ),
        LocalizedInsertIntentPatchItem(
            insert_id="thimble_group",
            insert_kind="thimble_plug",
            host_kind="guide_tube",
            insert_universe_id="thimble_plug",
            coordinates=[(3, 9), (6, 6), (9, 3)],
            z_min_cm=383.0,
            z_max_cm=394.0,
            component_role="plug",
            preserve_component_roles=["tube_wall"],
        ),
    ]
    pin_map.coordinate_convention = CoordinateConvention(index_base=1)
    axial_layers = next(p for p in patches if getattr(p, "patch_type", "") == "axial_layers")
    axial_layers.layers[0].loading_id = "loading_3B"
    axial_layers.layers.append(AxialLayerPatchItem(
        layer_id="upper_plenum",
        role="upper_plenum",
        z_min_cm=377.0,
        z_max_cm=397.0,
        fill_type="lattice",
        fill_id="assembly_lattice",
        loading_id="loading_3B",
    ))
    axial_layers.lattice_loadings = [
        LatticeLoadingPatchItem(
            loading_id="loading_3B",
            base_lattice_id="assembly_lattice",
            derived_lattice_id="assembly_lattice_3B",
            overrides={},
            purpose="base loading",
        )
    ]

    result = assemble_simulation_plan_from_patches(patches)

    assert result.ok is True
    model = result.plan.complex_model
    # Both insert loadings should be derived
    loading_ids = [l.id for l in model.lattice_loadings]
    assert "localized_insert_pyrex_group" in loading_ids
    assert "localized_insert_thimble_group" in loading_ids
    # Base lattice has guide tubes at all insert positions
    lattice = model.lattices[0]
    assert lattice.universe_pattern[2][5] == "gt"
    assert lattice.universe_pattern[2][8] == "gt"
    assert lattice.universe_pattern[5][5] == "gt"
    assert lattice.universe_pattern[8][2] == "gt"


# ---------------------------------------------------------------------------
# 8. Assembled overlays preserve Level 1 mode
# ---------------------------------------------------------------------------


def test_overlays_preserve_level1_mode() -> None:
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    overlays = result.plan.complex_model.core.axial_overlays
    assert len(overlays) == 1
    ov = overlays[0]
    assert ov.geometry_mode == "homogenized_open_region"
    assert ov.through_path_preserved is True


# ---------------------------------------------------------------------------
# 9. Settings patch does not block assembly
# ---------------------------------------------------------------------------


def test_settings_patch_does_not_block() -> None:
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    assert result.plan is not None
    # Cross sections is runtime; assembly should succeed without a path
    assert result.plan.complex_model is not None


# ---------------------------------------------------------------------------
# 14. PlanBuildState assembly lifecycle
# ---------------------------------------------------------------------------


def test_plan_build_state_assembly_lifecycle() -> None:
    from openmc_agent.plan_builder.patches import parse_patch_content

    state = PlanBuildState(
        state_id="test_assembly",
        requirement_text="3D assembly",
    )
    patches = _minimal_3d_patches()
    for i, patch in enumerate(patches):
        ptype = patch.patch_type
        envelope = PlanPatchEnvelope(
            patch_id=f"patch_{i}_{ptype}",
            patch_type=ptype,
            content=patch.model_dump(mode="json"),
            source="fixture",
        )
        parsed = parse_patch_content(ptype, patch.model_dump(mode="json"))
        val_result = validate_patch(parsed)
        assert val_result.ok, f"patch {ptype} failed validation: {[i.code for i in val_result.issues if i.severity == 'error']}"
        add_validated_patch_to_state(state, envelope, parsed, val_result)

    # All patches should be valid
    assert len(state.get_valid_patches()) == len(patches)

    # Assemble
    state = assemble_state_if_ready(state)
    assert state.assembled_plan is not None
    event_types = [e.event_type for e in state.build_log]
    assert "planning.assembly_started" in event_types
    assert "planning.assembly_completed" in event_types


def test_plan_build_state_assembly_failed_no_patches() -> None:
    state = PlanBuildState(
        state_id="test_empty",
        requirement_text="test",
    )
    state = assemble_state_if_ready(state)
    assert state.assembled_plan is None
    event_types = [e.event_type for e in state.build_log]
    assert "planning.assembly_failed" in event_types


# ---------------------------------------------------------------------------
# 15. Pin map expansion with out-of-bounds raises
# ---------------------------------------------------------------------------


def test_pin_map_out_of_bounds_raises() -> None:
    pin_map = PinMapPatch(
        lattice_size=(5, 5),
        default_universe_id="fuel",
        guide_tube_coords=[(10, 10)],
        coordinate_convention=CoordinateConvention(index_base=0),
    )
    with pytest.raises(ValueError, match="out of bounds"):
        expand_pin_map(pin_map, universe_ids={"guide_tube": "gt"})


def test_assembler_summary_includes_actual_pin_counts() -> None:
    result = assemble_simulation_plan_from_patches(_minimal_3d_patches())
    assert result.ok is True
    assert result.summary["actual_pin_counts"]["fuel_pin"] == 287
    assert result.summary["actual_pin_counts"]["guide_tube"] == 2


def test_assembler_canonicalizes_overlay_material_alias() -> None:
    patches = _minimal_3d_patches()
    materials = next(p for p in patches if isinstance(p, MaterialsPatch))
    materials.materials.append(
        MaterialSpecPatch(
            material_id="zircaloy4",
            name="Zircaloy-4",
            role="cladding",
            composition={"Zr": 1.0},
            composition_status="approximate",
            warnings=["Zircaloy-4 approximated as pure Zr"],
        )
    )
    overlays = next(p for p in patches if isinstance(p, AxialOverlaysPatch))
    overlays.overlays[0].material_id = "grid_zircaloy4"

    result = assemble_simulation_plan_from_patches(patches)
    assert result.ok is True
    overlay = result.plan.complex_model.core.axial_overlays[0]
    assert overlay.material_id == "zircaloy4"
    assert result.summary["material_aliases_applied"] == {
        "grid_zircaloy4": "zircaloy4"
    }


def test_assembler_canonicalizes_axial_layer_material_variant_suffix() -> None:
    patches = _minimal_3d_patches()
    materials = next(p for p in patches if isinstance(p, MaterialsPatch))
    materials.materials.append(
        MaterialSpecPatch(
            material_id="borated_water_3B",
            name="Borated Water",
            role="coolant",
            density_g_cm3=0.743,
        )
    )
    layers = next(p for p in patches if isinstance(p, AxialLayersPatch))
    layers.layers.append(
        AxialLayerPatchItem(
            layer_id="lower_moderator_buffer",
            role="reflector",
            z_min_cm=-55.0,
            z_max_cm=-5.0,
            fill_type="material",
            fill_id="borated_water",
        )
    )

    result = assemble_simulation_plan_from_patches(patches)
    assert result.ok is True
    layer = next(
        l for l in result.plan.complex_model.core.axial_layers
        if l.id == "lower_moderator_buffer"
    )
    assert layer.fill.id == "borated_water_3B"
    assert result.summary["material_aliases_applied"]["borated_water"] == "borated_water_3B"
