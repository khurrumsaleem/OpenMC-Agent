"""Tests for axial segment fill-mode and whole-plane contract (P2-FULLCORE-2D-A)."""

from __future__ import annotations

from openmc_agent.plan_builder.hierarchical_assembler import (
    AxialSegment,
    compile_global_axial_segments,
)
from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    AxialLayerPatchItem,
    AxialLayersPatch,
    BasePathAxialProfilePatchItem,
    BasePathStateBindingPatchItem,
    CoreLayoutPatch,
    FactsPatch,
)
from openmc_agent.plan_builder.axial_state_materializer import (
    materialize_concrete_axial_states,
)
from openmc_agent.schemas import LatticeSpec


def _make_simple_catalog() -> AssemblyCatalogPatch:
    """Build a simple catalog with one assembly type, no inserts."""
    return AssemblyCatalogPatch(assembly_types=[
        AssemblyTypePatchItem(
            assembly_type_id="fuel_a",
            name="fuel assembly",
            role="fuel",
            pin_map=AssemblyPinMapPatchItem(
                lattice_size=(3, 3),
                default_universe_id="fuel_pin",
                guide_tube_coords=[(1, 1)],
            ),
        ),
    ])


def _make_simple_layout() -> CoreLayoutPatch:
    return CoreLayoutPatch(
        shape=(1, 1),
        assembly_pitch_cm=3.78,
        assembly_pattern=[["fuel_a"]],
        boundary="reflective",
    )


def _make_base_pin_lattice() -> dict[str, LatticeSpec]:
    return {
        "fuel_a": LatticeSpec(
            id="assembly_lattice__fuel_a",
            name="pin lattice fuel_a",
            kind="rect",
            pitch_cm=(1.26, 1.26),
            outer_universe_id="moderator_outer",
            universe_pattern=[["fuel_pin", "fuel_pin", "fuel_pin"],
                              ["fuel_pin", "guide_tube", "fuel_pin"],
                              ["fuel_pin", "fuel_pin", "fuel_pin"]],
            shape=(3, 3),
        ),
    }


class TestAxialSegmentFillMode:
    """Test AxialSegment fill_mode field and defaults."""

    def test_default_fill_mode_is_detailed_core(self):
        seg = AxialSegment(segment_id="s0", z_min_cm=0.0, z_max_cm=1.0)
        assert seg.fill_mode == "detailed_core"

    def test_fill_mode_whole_plane_material(self):
        seg = AxialSegment(
            segment_id="s0", z_min_cm=0.0, z_max_cm=5.0,
            fill_mode="whole_plane_material",
            base_fill_id="lower_nozzle_mix",
        )
        assert seg.fill_mode == "whole_plane_material"
        assert seg.base_fill_id == "lower_nozzle_mix"

    def test_base_layer_fields_exist(self):
        seg = AxialSegment(
            segment_id="s0", z_min_cm=-55.0, z_max_cm=-5.0,
            base_axial_layer_id="layer_1",
            base_role="lower_moderator_buffer",
            fill_mode="whole_plane_material",
            base_fill_id="water",
        )
        assert seg.base_axial_layer_id == "layer_1"
        assert seg.base_role == "lower_moderator_buffer"
        assert seg.base_fill_id == "water"
        assert seg.detailed_path_state == {}
        assert seg.active_overlay_ids == []


class TestCompileGlobalAxialSegmentsFillMode:
    """Test compile_global_axial_segments with base_axial_layers."""

    def test_whole_plane_material_classification(self):
        """A layer with fill_type=material should produce whole_plane_material segments."""
        facts = FactsPatch(axial_domain_cm=(-5.0, 10.0))
        catalog = _make_simple_catalog()
        layers = [
            AxialLayerPatchItem(
                layer_id="nozzle", role="lower_nozzle",
                z_min_cm=-5.0, z_max_cm=5.0,
                fill_type="material", fill_id="nozzle_mix",
            ),
            AxialLayerPatchItem(
                layer_id="fuel", role="active_fuel",
                z_min_cm=5.0, z_max_cm=10.0,
                fill_type="lattice", fill_id="core_lattice",
            ),
        ]

        segments = compile_global_axial_segments(
            facts, catalog, base_axial_layers=layers,
        )

        assert len(segments) == 2
        assert segments[0].fill_mode == "whole_plane_material"
        assert segments[0].base_fill_id == "nozzle_mix"
        assert segments[0].base_axial_layer_id == "nozzle"
        assert segments[0].base_role == "lower_nozzle"

        assert segments[1].fill_mode == "detailed_core"
        assert segments[1].base_axial_layer_id == "fuel"

    def test_whole_plane_universe_classification(self):
        """A layer with fill_type=universe should produce whole_plane_universe segments."""
        facts = FactsPatch(axial_domain_cm=(0.0, 10.0))
        catalog = _make_simple_catalog()
        layers = [
            AxialLayerPatchItem(
                layer_id="reflector", role="reflector",
                z_min_cm=0.0, z_max_cm=5.0,
                fill_type="universe", fill_id="reflector_universe",
            ),
            AxialLayerPatchItem(
                layer_id="fuel", role="active_fuel",
                z_min_cm=5.0, z_max_cm=10.0,
                fill_type="lattice", fill_id="core_lattice",
            ),
        ]

        segments = compile_global_axial_segments(
            facts, catalog, base_axial_layers=layers,
        )

        assert segments[0].fill_mode == "whole_plane_universe"
        assert segments[0].base_fill_id == "reflector_universe"

    def test_void_classification(self):
        """A layer with fill_type=void should produce void segments."""
        facts = FactsPatch(axial_domain_cm=(0.0, 5.0))
        catalog = _make_simple_catalog()
        layers = [
            AxialLayerPatchItem(
                layer_id="void_layer", role="custom",
                z_min_cm=0.0, z_max_cm=5.0,
                fill_type="void", fill_id=None,
            ),
        ]

        segments = compile_global_axial_segments(
            facts, catalog, base_axial_layers=layers,
        )

        assert len(segments) == 1
        assert segments[0].fill_mode == "void"

    def test_negative_z_domain(self):
        """Segments with negative z coordinates should be handled correctly."""
        facts = FactsPatch(axial_domain_cm=(-55.0, 0.0))
        catalog = _make_simple_catalog()
        layers = [
            AxialLayerPatchItem(
                layer_id="buffer", role="lower_moderator_buffer",
                z_min_cm=-55.0, z_max_cm=-5.0,
                fill_type="material", fill_id="water",
            ),
            AxialLayerPatchItem(
                layer_id="plate", role="lower_core_plate",
                z_min_cm=-5.0, z_max_cm=0.0,
                fill_type="material", fill_id="core_plate_mix",
            ),
        ]

        segments = compile_global_axial_segments(
            facts, catalog, base_axial_layers=layers,
        )

        assert len(segments) == 2
        assert segments[0].z_min_cm == -55.0
        assert segments[0].z_max_cm == -5.0
        assert segments[0].fill_mode == "whole_plane_material"
        assert segments[1].z_min_cm == -5.0
        assert segments[1].z_max_cm == 0.0

    def test_backward_compat_axial_layer_boundaries(self):
        """The old axial_layer_boundaries parameter should still work."""
        facts = FactsPatch(axial_domain_cm=(0.0, 10.0))
        catalog = _make_simple_catalog()

        segments = compile_global_axial_segments(
            facts, catalog,
            axial_layer_boundaries=[(0.0, 5.0), (5.0, 10.0)],
        )

        assert len(segments) == 2
        # Backward compat: all default to detailed_core
        for seg in segments:
            assert seg.fill_mode == "detailed_core"


class TestMaterializerWholePlane:
    """Test materializer handles whole-plane segments correctly."""

    def test_whole_plane_segment_no_core_lattice(self):
        """Whole-plane material segments should not generate core lattices."""
        catalog = _make_simple_catalog()
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0", z_min_cm=-5.0, z_max_cm=0.0,
                fill_mode="whole_plane_material",
                base_fill_id="nozzle_mix",
                base_role="lower_nozzle",
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}

        result = materialize_concrete_axial_states(
            catalog, layout, segments, base_lats, base_uvs,
        )

        assert len(result.axial_layers) == 1
        assert result.axial_layers[0].fill.type == "material"
        assert result.axial_layers[0].fill.id == "nozzle_mix"
        assert len(result.segment_core_lattices) == 0
        assert len(result.derived_pin_lattices) == 0

    def test_detailed_core_segment_generates_core_lattice(self):
        """Detailed-core segments should generate core lattice fills."""
        catalog = _make_simple_catalog()
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0", z_min_cm=0.0, z_max_cm=10.0,
                fill_mode="detailed_core",
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}

        result = materialize_concrete_axial_states(
            catalog, layout, segments, base_lats, base_uvs,
        )

        assert len(result.axial_layers) == 1
        assert result.axial_layers[0].fill.type == "lattice"
        # No active inserts → reuse base core lattice
        assert result.axial_layers[0].fill.id == "core_lattice"

    def test_base_path_state_generates_segment_core_lattice(self):
        """Fuel-path state changes must not be dropped when no insert/grid is active."""
        catalog = _make_simple_catalog()
        catalog.assembly_types[0].base_path_profile_id = "fuel_path"
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0",
                z_min_cm=10.0,
                z_max_cm=20.0,
                fill_mode="detailed_core",
                base_role="upper_plenum",
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}
        profiles = {
            "fuel_path": BasePathAxialProfilePatchItem(
                profile_id="fuel_path",
                state_bindings=[
                    BasePathStateBindingPatchItem(
                        axial_role="upper_plenum",
                        source_universe_ids=["fuel_pin"],
                        replacement_universe_id="fuel_plenum",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    )
                ],
            )
        }

        result = materialize_concrete_axial_states(
            catalog,
            layout,
            segments,
            base_lats,
            base_uvs,
            base_path_profiles=profiles,
        )

        assert result.axial_layers[0].fill.type == "lattice"
        assert result.axial_layers[0].fill.id != "core_lattice"
        assert len(result.segment_core_lattices) == 1
        assert len(result.derived_pin_lattices) == 1
        assert result.derived_pin_lattices[0].universe_pattern == [
            ["fuel_plenum", "fuel_plenum", "fuel_plenum"],
            ["fuel_plenum", "guide_tube", "fuel_plenum"],
            ["fuel_plenum", "fuel_plenum", "fuel_plenum"],
        ]
        assert result.segment_index[0]["has_derived_lattices"] is True

    def test_base_path_preserves_guide_tube_positions_even_if_source_ids_include_them(self):
        """Preserve roles protect guide tubes from accidental fuel-path replacement."""
        catalog = _make_simple_catalog()
        catalog.assembly_types[0].base_path_profile_id = "fuel_path"
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0",
                z_min_cm=20.0,
                z_max_cm=30.0,
                fill_mode="detailed_core",
                base_role="upper_plenum",
            ),
        ]
        profiles = {
            "fuel_path": BasePathAxialProfilePatchItem(
                profile_id="fuel_path",
                state_bindings=[
                    BasePathStateBindingPatchItem(
                        axial_role="upper_plenum",
                        source_universe_ids=["fuel_pin", "guide_tube"],
                        replacement_universe_id="fuel_plenum",
                        preserve_path_roles=["guide_tube"],
                    )
                ],
            )
        }

        result = materialize_concrete_axial_states(
            catalog,
            layout,
            segments,
            _make_base_pin_lattice(),
            {"fuel_a": "assembly_universe__fuel_a"},
            base_path_profiles=profiles,
        )

        assert result.derived_pin_lattices[0].universe_pattern[1][1] == "guide_tube"

    def test_mixed_whole_plane_and_detailed(self):
        """A mix of whole-plane and detailed segments should work."""
        catalog = _make_simple_catalog()
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0", z_min_cm=-5.0, z_max_cm=0.0,
                fill_mode="whole_plane_material",
                base_fill_id="nozzle_mix",
                base_role="lower_nozzle",
            ),
            AxialSegment(
                segment_id="s1", z_min_cm=0.0, z_max_cm=10.0,
                fill_mode="detailed_core",
            ),
            AxialSegment(
                segment_id="s2", z_min_cm=10.0, z_max_cm=15.0,
                fill_mode="whole_plane_material",
                base_fill_id="upper_nozzle_mix",
                base_role="upper_nozzle",
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}

        result = materialize_concrete_axial_states(
            catalog, layout, segments, base_lats, base_uvs,
        )

        assert len(result.axial_layers) == 3
        assert result.axial_layers[0].fill.type == "material"
        assert result.axial_layers[1].fill.type == "lattice"
        assert result.axial_layers[2].fill.type == "material"

    def test_void_segment(self):
        """Void segments should produce void fill."""
        catalog = _make_simple_catalog()
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0", z_min_cm=0.0, z_max_cm=5.0,
                fill_mode="void",
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}

        result = materialize_concrete_axial_states(
            catalog, layout, segments, base_lats, base_uvs,
        )

        assert len(result.axial_layers) == 1
        assert result.axial_layers[0].fill.type == "void"

    def test_whole_plane_missing_fill_id_reports_issue(self):
        """Missing fill_id for whole-plane segment should report an issue."""
        catalog = _make_simple_catalog()
        layout = _make_simple_layout()
        segments = [
            AxialSegment(
                segment_id="s0", z_min_cm=0.0, z_max_cm=5.0,
                fill_mode="whole_plane_material",
                base_fill_id=None,
            ),
        ]
        base_lats = _make_base_pin_lattice()
        base_uvs = {"fuel_a": "assembly_universe__fuel_a"}

        result = materialize_concrete_axial_states(
            catalog, layout, segments, base_lats, base_uvs,
        )

        issue_codes = [i.code for i in result.issues]
        assert "fullcore.whole_plane_fill_missing" in issue_codes
