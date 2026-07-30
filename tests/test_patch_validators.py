"""Tests for patch validators (Phase 2)."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.patches import (
    AxialLayerPatchItem,
    AxialLayersPatch,
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
    CellLayerPatch,
    FactsPatch,
    LocalizedInsertIntentPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
    PinMapPatch,
    SettingsPatch,
    UniverseSpecPatch,
    UniversesPatch,
    parse_patch_content,
)
from openmc_agent.plan_builder.state import (
    PlanBuildState,
    PlanPatchEnvelope,
    add_validated_patch_to_state,
)
from openmc_agent.plan_builder.validators import (
    PatchValidationContext,
    PatchValidationResult,
    validate_patch,
)
from openmc_agent.plan_builder.material_resolution import (
    infer_material_aliases,
    resolve_material_id,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _codes(result: PatchValidationResult) -> list[str]:
    return [i.code for i in result.issues]


# ---------------------------------------------------------------------------
# 1. FactsPatch valid minimal
# ---------------------------------------------------------------------------


def test_facts_valid_minimal() -> None:
    patch = FactsPatch(
        benchmark_id="VERA3",
        selected_variant="3B",
        lattice_size=(17, 17),
        has_axial_geometry=True,
        has_special_pin_map=True,
        active_fuel_region_cm=(0.0, 365.76),
        expected_pyrex_count=16,
    )
    result = validate_patch(patch)
    assert result.ok is True
    assert not any(i.severity == "error" for i in result.issues)


# ---------------------------------------------------------------------------
# 2. FactsPatch invalid z range
# ---------------------------------------------------------------------------


def test_facts_invalid_z_range() -> None:
    patch = FactsPatch(active_fuel_region_cm=(300.0, 100.0))
    result = validate_patch(patch)
    assert "patch.schema_invalid" in _codes(result)
    assert result.ok is False


def test_facts_negative_count() -> None:
    patch = FactsPatch(expected_pin_count=-5)
    result = validate_patch(patch)
    assert result.ok is False


def test_facts_missing_facts_info() -> None:
    patch = FactsPatch(missing_facts=["coolant density"])
    result = validate_patch(patch)
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "patch.missing_required_field" in info_codes
    assert result.ok is True


# ---------------------------------------------------------------------------
# 3. MaterialsPatch unique IDs
# ---------------------------------------------------------------------------


def test_materials_duplicate_ids() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(material_id="fuel", name="UO2", role="fuel"),
        MaterialSpecPatch(material_id="fuel", name="UO2", role="fuel"),
    ])
    result = validate_patch(patch)
    assert "patch.duplicate_id" in _codes(result)
    assert result.ok is False


def test_materials_invalid_density() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(material_id="fuel", name="UO2", role="fuel", density_g_cm3=-1.0),
    ])
    result = validate_patch(patch)
    assert "patch.materials.invalid_density" in _codes(result)
    assert result.ok is False


# ---------------------------------------------------------------------------
# 4. Alloy pure-element approximation warning / error
# ---------------------------------------------------------------------------


def test_alloy_confirmed_pure_element_is_error() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(
            material_id="clad",
            name="Zircaloy-4",
            role="cladding",
            composition={"Zr": 1.0},
            composition_status="confirmed",
        ),
    ])
    result = validate_patch(patch)
    assert "patch.materials.alloy_reduced_to_pure_element" in _codes(result)
    error = next(i for i in result.issues if "alloy_reduced" in i.code)
    assert error.severity == "error"
    assert result.ok is False


def test_alloy_approximate_with_warning_is_ok() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(
            material_id="clad",
            name="Zircaloy-4",
            role="cladding",
            composition={"Zr": 1.0},
            composition_status="approximate",
            warnings=["Zircaloy-4 approximated as pure Zr"],
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is True


def test_alloy_approximate_without_warning_is_warning() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(
            material_id="clad",
            name="SS-304",
            role="structural",
            composition={"Fe": 1.0},
            composition_status="approximate",
        ),
    ])
    result = validate_patch(patch)
    assert "patch.materials.alloy_reduced_to_pure_element" in _codes(result)
    w = next(i for i in result.issues if "alloy_reduced" in i.code)
    assert w.severity == "warning"
    assert result.ok is True


def test_alloy_inconel_confirmed_pure_ni() -> None:
    patch = MaterialsPatch(materials=[
        MaterialSpecPatch(
            material_id="inconel",
            name="Inconel-718",
            role="structural",
            composition={"Ni": 1.0},
            composition_status="confirmed",
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is False


# ---------------------------------------------------------------------------
# 5. UniversesPatch guide tube wall missing
# ---------------------------------------------------------------------------


def test_guide_tube_wall_missing() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(
            universe_id="gt_univ",
            kind="guide_tube",
            cells=[
                CellLayerPatch(id="water", role="coolant", region_kind="cylinder"),
            ],
        ),
    ])
    result = validate_patch(patch)
    assert "patch.universes.guide_tube_wall_missing" in _codes(result)


# ---------------------------------------------------------------------------
# 6. UniversesPatch valid guide tube
# ---------------------------------------------------------------------------


def test_guide_tube_valid() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(
            universe_id="gt_univ",
            kind="guide_tube",
            cells=[
                CellLayerPatch(id="inner_water", role="coolant", region_kind="cylinder", r_max_cm=0.5),
                CellLayerPatch(id="wall", role="cladding", region_kind="annulus", r_min_cm=0.5, r_max_cm=0.6),
                CellLayerPatch(id="outer_water", role="background", region_kind="background"),
            ],
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is True


def test_universe_duplicate_id() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(universe_id="u1", kind="fuel_pin", cells=[CellLayerPatch(id="c1", role="fuel")]),
        UniverseSpecPatch(universe_id="u1", kind="fuel_pin", cells=[CellLayerPatch(id="c1", role="fuel")]),
    ])
    result = validate_patch(patch)
    assert "patch.universes.duplicate_universe_id" in _codes(result)
    assert result.ok is False


def test_universe_empty() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(universe_id="u1", kind="fuel_pin", cells=[]),
    ])
    result = validate_patch(patch)
    assert "patch.universes.empty_universe" in _codes(result)
    assert result.ok is False


def test_universe_invalid_radius() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(universe_id="u1", kind="fuel_pin", cells=[
            CellLayerPatch(id="c1", role="fuel", r_min_cm=0.5, r_max_cm=0.3),
        ]),
    ])
    result = validate_patch(patch)
    assert "patch.universes.invalid_radius_order" in _codes(result)


def test_pyrex_material_missing() -> None:
    patch = UniversesPatch(universes=[
        UniverseSpecPatch(universe_id="px", kind="pyrex_rod", cells=[
            CellLayerPatch(id="c1", role="filler"),
        ]),
    ])
    result = validate_patch(patch)
    assert "patch.universes.pyrex_material_missing" in _codes(result)


# ---------------------------------------------------------------------------
# 7. PinMapPatch coordinate out of bounds
# ---------------------------------------------------------------------------


def test_pin_map_coord_out_of_bounds() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel",
        guide_tube_coords=[(17, 0)],  # 0-indexed: row 17 is out of bounds
    )
    result = validate_patch(patch)
    assert "patch.pin_map.coord_out_of_bounds" in _codes(result)
    assert result.ok is False


# ---------------------------------------------------------------------------
# 8. PinMapPatch overlap
# ---------------------------------------------------------------------------


def test_pin_map_overlap() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel",
        guide_tube_coords=[(5, 5)],
        pyrex_rod_coords=[(5, 5)],
    )
    result = validate_patch(patch)
    # Overlap is detected and reported (warning, not error)
    # The assembler's derive_localized_insert_loadings handles normalization.
    codes = _codes(result)
    assert "patch.pin_map.coord_overlap_detected" in codes
    assert result.ok is True  # detected, not blocking
    # Verify the patch was NOT modified in-place by the validator
    assert patch.guide_tube_coords == [(5, 5)]  # unchanged
    assert patch.pyrex_rod_coords == [(5, 5)]  # unchanged


# ---------------------------------------------------------------------------
# 9. PinMapPatch VERA3 count mismatch
# ---------------------------------------------------------------------------


def test_pin_map_count_mismatch() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel",
        pyrex_rod_coords=[(i, i) for i in range(15)],  # 15 instead of 16
    )
    context = PatchValidationContext(
        benchmark_id="VERA3",
        selected_variant="3B",
        expected_counts={"expected_pyrex_rod_count": 16},
    )
    result = validate_patch(patch, context)
    assert "patch.pin_map.count_mismatch" in _codes(result)


def test_modern_pin_map_total_pin_count_and_segmented_insert_counts() -> None:
    pyrex_coords = [(3, 6), (4, 4)]
    thimble_coords = [(3, 9)]
    patch = PinMapPatch(
        lattice_size=(9, 9),
        default_universe_id="fuel",
        coordinate_convention={"index_base": 1},
        guide_tube_coords=[(3, 6), (4, 4), (3, 9)],
        localized_insert_intents=[
            LocalizedInsertIntentPatchItem(
                insert_id="pyrex_poison",
                insert_kind="pyrex_rod",
                insert_universe_id="pyrex_poison",
                coordinates=pyrex_coords,
            ),
            LocalizedInsertIntentPatchItem(
                insert_id="pyrex_plenum",
                insert_kind="pyrex_rod",
                insert_universe_id="pyrex_plenum",
                coordinates=pyrex_coords,
            ),
            LocalizedInsertIntentPatchItem(
                insert_id="thimble_plug",
                insert_kind="thimble_plug",
                insert_universe_id="thimble_plug",
                coordinates=thimble_coords,
            ),
        ],
    )
    context = PatchValidationContext(
        benchmark_id="VERA3",
        selected_variant="3B",
        expected_counts={
            "expected_pin_count": 81,
            "expected_guide_tube_count": 3,
            "expected_pyrex_count": 2,
            "expected_thimble_plug_count": 1,
        },
        strict_benchmark=True,
    )

    result = validate_patch(patch, context)

    assert "patch.pin_map.count_mismatch" not in _codes(result)


def test_pin_map_count_mismatch_strict_is_error() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel",
        guide_tube_coords=[(1, 1)] * 20,  # 20 instead of 24
    )
    context = PatchValidationContext(
        benchmark_id="VERA3",
        expected_counts={"expected_guide_tube_count": 24},
        strict_benchmark=True,
    )
    result = validate_patch(patch, context)
    issue = next(i for i in result.issues if "count_mismatch" in i.code)
    assert issue.severity == "error"
    assert result.ok is False


def test_pin_map_valid() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel",
        guide_tube_coords=[(2, 2), (2, 5)],
        instrument_tube_coords=[(9, 9)],
    )
    result = validate_patch(patch)
    assert result.ok is True


def test_pin_map_partial_expected_counts_warns_not_error() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        instrument_tube_coords=[(0, 0)],
        pyrex_rod_coords=[(i, 1) for i in range(16)],
        thimble_plug_coords=[(i, 2) for i in range(8)],
    )
    context = PatchValidationContext(
        expected_counts={"fuel_pin": 264},
        strict_benchmark=True,
    )
    result = validate_patch(patch, context)
    assert result.ok is True
    assert "patch.pin_map.expected_counts_partial" in _codes(result)
    assert "patch.pin_map.expected_counts_sum_mismatch" not in _codes(result)
    assert "patch.pin_map.count_mismatch" not in _codes(result)


def test_pin_map_complete_expected_counts_sum_mismatch_fails() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
    )
    context = PatchValidationContext(
        expected_counts={"fuel_pin": 264},
        expected_counts_complete=True,
    )
    result = validate_patch(patch, context)
    assert result.ok is False
    assert "patch.pin_map.expected_counts_sum_mismatch" in _codes(result)


def test_pin_map_reference_expected_counts_complete_ok() -> None:
    patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        instrument_tube_coords=[(0, 0)],
        pyrex_rod_coords=[(i, 1) for i in range(16)],
        thimble_plug_coords=[(i, 2) for i in range(8)],
    )
    context = PatchValidationContext(
        reference_expected_counts={
            "fuel_pin": 264,
            "pyrex_rod": 16,
            "thimble_plug": 8,
            "instrument_tube": 1,
            "guide_tube": 0,
        },
        strict_benchmark=True,
    )
    result = validate_patch(patch, context)
    assert result.ok is True
    assert "patch.pin_map.expected_counts_sum_mismatch" not in _codes(result)


# ---------------------------------------------------------------------------
# 10. AxialLayersPatch active fuel missing
# ---------------------------------------------------------------------------


def test_axial_layers_active_fuel_missing() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="plenum", role="upper_plenum", z_min_cm=0.0, z_max_cm=10.0),
    ])
    result = validate_patch(patch)
    assert "patch.axial_layers.active_fuel_missing" in _codes(result)


# ---------------------------------------------------------------------------
# 11. AxialLayersPatch overlap
# ---------------------------------------------------------------------------


def test_axial_layers_overlap() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=0.0, z_max_cm=200.0, fill_type="lattice", fill_id="lat1"),
        AxialLayerPatchItem(layer_id="l2", role="custom", z_min_cm=150.0, z_max_cm=300.0),
    ])
    result = validate_patch(patch)
    assert "patch.axial_layers.overlap" in _codes(result)
    assert result.ok is False


def test_axial_layers_invalid_range() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=100.0, z_max_cm=50.0, fill_type="lattice", fill_id="lat1"),
    ])
    result = validate_patch(patch)
    assert "patch.axial_layers.invalid_range" in _codes(result)


def test_axial_layers_fill_missing() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=0.0, z_max_cm=200.0, fill_type="material"),
    ])
    result = validate_patch(patch)
    assert "patch.axial_layers.fill_missing" in _codes(result)


def test_axial_layers_rejects_incomplete_unknown_layer() -> None:
    """A schema-valid placeholder layer must not be accepted for assembly."""
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(
            layer_id="lower_buffer",
            role="lower_moderator_buffer",
            z_min_cm=-55.0,
            z_max_cm=None,
            fill_type="unknown",
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is False
    assert "patch.axial_layers.invalid_range" in _codes(result)
    assert "patch.axial_layers.fill_unknown" in _codes(result)


def test_axial_layers_rejects_unattached_lattice_loading() -> None:
    """Declared localized-insert loading must be referenced by a layer."""
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(
            layer_id="fuel",
            role="active_fuel",
            z_min_cm=0.0,
            z_max_cm=100.0,
            fill_type="lattice",
            fill_id="assembly_lattice",
        ),
    ], lattice_loadings=[{
        "loading_id": "rcca_loading",
        "base_lattice_id": "assembly_lattice",
    }])
    result = validate_patch(patch)
    assert result.ok is False
    assert "patch.axial_layers.loading_unattached" in _codes(result)


# ---------------------------------------------------------------------------
# 12. AxialLayersPatch default z=-1..1 flagged for 3D benchmark
# ---------------------------------------------------------------------------


def test_axial_layers_default_unit_slab_for_benchmark() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=-1.0, z_max_cm=1.0, fill_type="lattice", fill_id="lat1"),
    ])
    context = PatchValidationContext(benchmark_id="VERA3", selected_variant="3A")
    result = validate_patch(patch, context)
    assert "patch.axial_layers.default_unit_slab" in _codes(result)
    assert result.ok is False


def test_axial_layers_valid() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=0.0, z_max_cm=200.0, fill_type="lattice", fill_id="lat1"),
        AxialLayerPatchItem(layer_id="l2", role="upper_plenum", z_min_cm=200.0, z_max_cm=300.0),
    ])
    result = validate_patch(patch)
    assert result.ok is True


def test_axial_layers_duplicate_id() -> None:
    patch = AxialLayersPatch(layers=[
        AxialLayerPatchItem(layer_id="l1", role="active_fuel", z_min_cm=0.0, z_max_cm=200.0, fill_type="lattice", fill_id="lat1"),
        AxialLayerPatchItem(layer_id="l1", role="upper_plenum", z_min_cm=200.0, z_max_cm=300.0),
    ])
    result = validate_patch(patch)
    assert "patch.duplicate_id" in _codes(result)


def test_axial_layers_empty() -> None:
    patch = AxialLayersPatch(layers=[])
    result = validate_patch(patch)
    assert "patch.axial_layers.empty" in _codes(result)
    assert result.ok is False


# ---------------------------------------------------------------------------
# 13. AxialOverlaysPatch homogenized overlay valid
# ---------------------------------------------------------------------------


def test_overlay_homogenized_valid() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            material_id="grid_material",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is True


# ---------------------------------------------------------------------------
# 14. AxialOverlaysPatch missing through path
# ---------------------------------------------------------------------------


def test_overlay_through_path_derived_for_none() -> None:
    """through_path_preserved is deterministically derived to True for modes
    that inherently preserve through-paths (mass_conserving_outer_frame,
    homogenized_open_region) when the field is absent (None)."""
    for mode in ("mass_conserving_outer_frame", "homogenized_open_region"):
        ov = AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            material_id="grid_material",
            geometry_mode=mode,
        )
        assert ov.through_path_preserved is True, f"{mode} should derive True"
    # skeleton mode does NOT derive
    ov_skel = AxialOverlayPatchItem(
        overlay_id="grid1", overlay_kind="spacer_grid", geometry_mode="skeleton",
    )
    assert ov_skel.through_path_preserved is None


def test_overlay_through_path_explicit_false_is_contradiction() -> None:
    """Explicit through_path_preserved=False is a semantic contradiction for
    mass_conserving_outer_frame and homogenized_open_region."""
    from openmc_agent.plan_builder.validators import validate_patch
    for mode in ("mass_conserving_outer_frame", "homogenized_open_region"):
        patch = AxialOverlaysPatch(overlays=[
            AxialOverlayPatchItem(
                overlay_id="grid1",
                overlay_kind="spacer_grid",
                z_min_cm=10.0,
                z_max_cm=12.0,
                target_lattice_id="assembly_lattice",
                material_id="grid_material",
                total_mass_g=500.0,
                geometry_mode=mode,
                through_path_preserved=False,
            ),
        ])
        result = validate_patch(patch)
        codes = _codes(result)
        assert "patch.axial_overlays.mode_semantic_contradiction" in codes, (
            f"{mode} + explicit False should produce mode_semantic_contradiction, "
            f"got {codes}"
        )
        assert result.ok is False


def test_overlay_target_missing() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            material_id="grid_material",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    result = validate_patch(patch)
    assert "patch.axial_overlays.target_missing" in _codes(result)


def test_overlay_material_missing() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    result = validate_patch(patch)
    assert "patch.axial_overlays.material_missing" in _codes(result)


def test_overlay_material_alias_resolves() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            material_id="grid_zircaloy4",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    context = PatchValidationContext(
        known_material_ids=["zircaloy4", "inconel718"],
    )
    result = validate_patch(patch, context)
    assert result.ok is True
    assert "patch.axial_overlays.material_alias_resolved" in _codes(result)
    resolved = resolve_material_id("grid_zircaloy4", {"zircaloy4", "inconel718"})
    assert resolved.ok is True
    assert resolved.resolved_id == "zircaloy4"


def test_overlay_material_alias_inferred_from_unique_material_name() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            material_id="zircaloy4",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    context = PatchValidationContext(
        known_material_ids=["mat_zirc"],
        material_summaries=[
            {"material_id": "mat_zirc", "name": "Zircaloy-4 Cladding"},
        ],
    )
    result = validate_patch(patch, context)
    assert result.ok is True
    assert "patch.axial_overlays.material_alias_resolved" in _codes(result)

    aliases = infer_material_aliases(context.material_summaries, context.known_material_ids)
    assert aliases["zircaloy4"] == "mat_zirc"
    assert aliases["zircaloy_4"] == "mat_zirc"

    resolved = resolve_material_id(
        "zircaloy-4",
        {"mat_zirc"},
        aliases,
    )
    assert resolved.ok is True
    assert resolved.resolved_id == "mat_zirc"


def test_overlay_material_alias_inference_drops_ambiguous_names() -> None:
    aliases = infer_material_aliases(
        [
            {"material_id": "mat_zirc_a", "name": "Zircaloy-4 Cladding"},
            {"material_id": "mat_zirc_b", "name": "Zircaloy-4 Guide Tube"},
        ],
        ["mat_zirc_a", "mat_zirc_b"],
    )
    assert "zircaloy4" not in aliases


def test_overlay_unresolved_material_still_fails() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            target_lattice_id="assembly_lattice",
            material_id="unknown_grid_material",
            geometry_mode="homogenized_open_region",
            through_path_preserved=True,
        ),
    ])
    context = PatchValidationContext(known_material_ids=["zircaloy4", "inconel718"])
    result = validate_patch(patch, context)
    assert result.ok is False
    assert "patch.axial_overlays.material_missing" in _codes(result)


def test_overlay_invalid_range_non_skeleton() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            geometry_mode="homogenized_open_region",
            target_lattice_id="lat",
            material_id="m",
            through_path_preserved=True,
        ),
    ])
    result = validate_patch(patch)
    assert "patch.axial_overlays.invalid_range" in _codes(result)


# ---------------------------------------------------------------------------
# 15. AxialOverlaysPatch volume_fraction_calibrated missing volume fraction
# ---------------------------------------------------------------------------


def test_overlay_volume_fraction_missing() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="grid1",
            overlay_kind="spacer_grid",
            z_min_cm=10.0,
            z_max_cm=12.0,
            geometry_mode="volume_fraction_calibrated",
        ),
    ])
    result = validate_patch(patch)
    assert "patch.axial_overlays.volume_fraction_missing" in _codes(result)
    assert result.ok is False


def test_overlay_duplicate_id() -> None:
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(overlay_id="g1", overlay_kind="spacer_grid"),
        AxialOverlayPatchItem(overlay_id="g1", overlay_kind="spacer_grid"),
    ])
    result = validate_patch(patch)
    assert "patch.axial_overlays.duplicate_overlay_id" in _codes(result)


# ---------------------------------------------------------------------------
# 16. SettingsPatch runtime xsec not blocking
# ---------------------------------------------------------------------------


def test_settings_runtime_xsec_not_blocking() -> None:
    patch = SettingsPatch()
    result = validate_patch(patch)
    assert result.ok is True
    info_codes = [i.code for i in result.issues if i.severity == "info"]
    assert "patch.settings.cross_sections_runtime_only" in info_codes
    assert "patch.settings.tallies_not_required_for_smoke" in info_codes


def test_settings_plot_not_full_assembly_warning() -> None:
    patch = SettingsPatch(plot_strategy="quarter_assembly")
    context = PatchValidationContext(benchmark_id="VERA3")
    result = validate_patch(patch, context)
    assert "patch.settings.plot_not_full_assembly" in _codes(result)
    assert result.ok is True  # warning, not error


# ---------------------------------------------------------------------------
# 18. PlanBuildState validated patch lifecycle
# ---------------------------------------------------------------------------


def test_plan_build_state_validated_patch_lifecycle() -> None:
    state = PlanBuildState(
        state_id="test_lifecycle",
        requirement_text="VERA3 3B benchmark",
    )
    pin_map = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        guide_tube_coords=[(2, 2), (2, 5)],
    )
    envelope = PlanPatchEnvelope(
        patch_id="patch_pin_map_01",
        patch_type="pin_map",
        content=pin_map.model_dump(mode="json"),
        source="fixture",
    )
    result = validate_patch(pin_map)
    assert result.ok is True

    state = add_validated_patch_to_state(state, envelope, pin_map, result)

    # Patch status should be valid
    assert state.patches["patch_pin_map_01"].status == "valid"
    assert state.patch_status["patch_pin_map_01"] == "valid"

    # Build log should have parse + validate events
    event_types = [e.event_type for e in state.build_log]
    assert "planning.patch_parsed" in event_types
    assert "planning.patch_validated" in event_types

    # get_valid_patches should return it
    valid = state.get_valid_patches("pin_map")
    assert len(valid) == 1
    assert valid[0].patch_id == "patch_pin_map_01"


def test_plan_build_state_invalid_patch_lifecycle() -> None:
    state = PlanBuildState(
        state_id="test_invalid",
        requirement_text="test",
    )
    bad_patch = PinMapPatch(
        lattice_size=(17, 17),
        default_universe_id="fuel_pin",
        guide_tube_coords=[(99, 99)],  # out of bounds
    )
    envelope = PlanPatchEnvelope(
        patch_id="patch_pin_map_bad",
        patch_type="pin_map",
        content=bad_patch.model_dump(mode="json"),
    )
    result = validate_patch(bad_patch)
    assert result.ok is False

    state = add_validated_patch_to_state(state, envelope, bad_patch, result)

    assert state.patches["patch_pin_map_bad"].status == "invalid"
    event_types = [e.event_type for e in state.build_log]
    assert "planning.patch_invalid" in event_types
    assert state.get_valid_patches("pin_map") == []
