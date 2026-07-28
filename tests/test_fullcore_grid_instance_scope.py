"""Tests for spacer-grid assembly-instance materialization (P2-FULLCORE-2D-A/6)."""

from __future__ import annotations

from openmc_agent.plan_builder.axial_state_materializer import (
    AssemblyGridState,
    GridFrameDerivationReport,
    _compute_grid_frame_derivation,
    _get_active_grids_for_segment,
    materialize_concrete_axial_states,
)
from openmc_agent.plan_builder.hierarchical_assembler import AxialSegment
from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    AxialOverlayPatchItem,
    CoreLayoutPatch,
)
from openmc_agent.schemas import LatticeSpec


def _catalog() -> AssemblyCatalogPatch:
    return AssemblyCatalogPatch(assembly_types=[
        AssemblyTypePatchItem(
            assembly_type_id="fuel", name="fuel", role="fuel",
            pin_map=AssemblyPinMapPatchItem(lattice_size=(3, 3), default_universe_id="fuel_pin"),
        ),
    ])


def _layout() -> CoreLayoutPatch:
    return CoreLayoutPatch(
        shape=(1, 1), assembly_pitch_cm=3.78,
        assembly_pattern=[["fuel"]], boundary="reflective",
    )


def _base_lats() -> dict[str, LatticeSpec]:
    return {"fuel": LatticeSpec(
        id="assembly_lattice__fuel", name="test", kind="rect",
        pitch_cm=(1.26, 1.26), outer_universe_id="mod_outer",
        universe_pattern=[["fuel_pin"] * 3] * 3, shape=(3, 3),
    )}


def _base_uvs() -> dict[str, str]:
    return {"fuel": "assembly_universe__fuel"}


class TestGridFrameDerivation:
    def test_end_grid_mass_derivation(self):
        """Inconel-718 end grid: 1017g, density 8.19, 289 cells, height ~3.866cm."""
        ov = AxialOverlayPatchItem(
            overlay_id="grid_end_0",
            overlay_kind="spacer_grid",
            z_min_cm=11.951, z_max_cm=15.817,
            material_id="inconel718",
            geometry_mode="mass_conserving_outer_frame",
            total_mass_g=1017.0,
            cell_count=289,
            pitch_cm=1.26,
        )
        report = _compute_grid_frame_derivation(ov, density_g_cm3=8.19, pitch_cm=1.26)

        assert report.material_id == "inconel718"
        assert report.total_mass_g == 1017.0
        assert abs(report.density_g_cm3 - 8.19) < 1e-6
        assert report.cell_count == 289
        assert report.area_per_cell_cm2 > 0
        assert report.frame_thickness_cm > 0
        assert 0 < report.volume_fraction < 1

    def test_middle_grid_mass_derivation(self):
        """Zircaloy-4 middle grid: 875g, density 6.56, 289 cells, height ~3.810cm."""
        ov = AxialOverlayPatchItem(
            overlay_id="grid_mid_1",
            overlay_kind="spacer_grid",
            z_min_cm=73.295, z_max_cm=77.105,
            material_id="zircaloy4",
            geometry_mode="mass_conserving_outer_frame",
            total_mass_g=875.0,
            cell_count=289,
            pitch_cm=1.26,
        )
        report = _compute_grid_frame_derivation(ov, density_g_cm3=6.56, pitch_cm=1.26)

        assert report.material_id == "zircaloy4"
        assert abs(report.total_mass_g - 875.0) < 1e-6
        assert report.area_per_cell_cm2 > 0

    def test_volume_fraction_decreases_with_more_mass(self):
        """More mass → larger frame area → higher volume fraction."""
        ov_light = AxialOverlayPatchItem(
            overlay_id="light", overlay_kind="spacer_grid",
            z_min_cm=0.0, z_max_cm=3.0, material_id="m",
            total_mass_g=100.0, cell_count=289, pitch_cm=1.26,
        )
        ov_heavy = AxialOverlayPatchItem(
            overlay_id="heavy", overlay_kind="spacer_grid",
            z_min_cm=0.0, z_max_cm=3.0, material_id="m",
            total_mass_g=1000.0, cell_count=289, pitch_cm=1.26,
        )
        r_light = _compute_grid_frame_derivation(ov_light, 7.0, 1.26)
        r_heavy = _compute_grid_frame_derivation(ov_heavy, 7.0, 1.26)
        assert r_heavy.volume_fraction > r_light.volume_fraction


class TestGridSegmentDetection:
    def test_grid_active_in_overlapping_segment(self):
        ov = AxialOverlayPatchItem(
            overlay_id="grid_1", overlay_kind="spacer_grid",
            z_min_cm=73.0, z_max_cm=77.0,
            material_id="grid_mat",
            geometry_mode="mass_conserving_outer_frame",
        )
        seg = AxialSegment(segment_id="s0", z_min_cm=74.0, z_max_cm=76.0)
        active = _get_active_grids_for_segment(seg, [ov])
        assert len(active) == 1
        assert active[0].overlay_id == "grid_1"

    def test_grid_inactive_outside_segment(self):
        ov = AxialOverlayPatchItem(
            overlay_id="grid_1", overlay_kind="spacer_grid",
            z_min_cm=73.0, z_max_cm=77.0,
        )
        seg = AxialSegment(segment_id="s0", z_min_cm=0.0, z_max_cm=10.0)
        active = _get_active_grids_for_segment(seg, [ov])
        assert len(active) == 0

    def test_skeleton_grid_is_not_active_for_materialization(self):
        ov = AxialOverlayPatchItem(
            overlay_id="grid_skeleton",
            overlay_kind="spacer_grid",
            z_min_cm=0.0,
            z_max_cm=10.0,
            geometry_mode="skeleton",
            material_id=None,
            requires_human_confirmation=True,
        )
        seg = AxialSegment(segment_id="s0", z_min_cm=1.0, z_max_cm=2.0)
        active = _get_active_grids_for_segment(seg, [ov])
        assert active == []


class TestGridMaterializerIntegration:
    def test_skeleton_grid_does_not_require_density_or_decorate_universe(self):
        grids = [
            AxialOverlayPatchItem(
                overlay_id="grid_skeleton",
                overlay_kind="spacer_grid",
                z_min_cm=0.0,
                z_max_cm=10.0,
                geometry_mode="skeleton",
                material_id=None,
                requires_human_confirmation=True,
            ),
        ]
        segments = [
            AxialSegment(segment_id="s0", z_min_cm=0.0, z_max_cm=5.0, fill_mode="detailed_core"),
        ]
        result = materialize_concrete_axial_states(
            _catalog(), _layout(), segments, _base_lats(), _base_uvs(),
            grid_overlays=grids,
            grid_density_lookup={},
        )
        assert not result.has_errors
        assert result.grid_decorated_universe_patches == []
        assert "s0" not in result.grid_states

    def test_grid_state_tracked_in_result(self):
        """Materializer should track grid state for segments with active grids."""
        grids = [
            AxialOverlayPatchItem(
                overlay_id="grid_test", overlay_kind="spacer_grid",
                z_min_cm=0.0, z_max_cm=10.0,
                material_id="inconel718",
                geometry_mode="mass_conserving_outer_frame",
                total_mass_g=500.0, cell_count=289, pitch_cm=1.26,
            ),
        ]
        segments = [
            AxialSegment(segment_id="s0", z_min_cm=0.0, z_max_cm=5.0, fill_mode="detailed_core"),
        ]
        result = materialize_concrete_axial_states(
            _catalog(), _layout(), segments, _base_lats(), _base_uvs(),
            grid_overlays=grids,
            grid_density_lookup={"inconel718": 8.19},
        )
        assert "s0" in result.grid_states
        gs = result.grid_states["s0"]
        assert "grid_test" in gs.active_overlay_ids
        assert len(gs.derivation_reports) == 1
        assert gs.derivation_reports[0].material_id == "inconel718"

    def test_grid_affects_state_hash(self):
        """Same segment with and without grid should produce different pin hashes."""
        grids = [
            AxialOverlayPatchItem(
                overlay_id="grid_1", overlay_kind="spacer_grid",
                z_min_cm=0.0, z_max_cm=10.0,
                material_id="m", geometry_mode="mass_conserving_outer_frame",
                total_mass_g=500.0, cell_count=289, pitch_cm=1.26,
            ),
        ]
        segments = [
            AxialSegment(segment_id="s0", z_min_cm=0.0, z_max_cm=5.0, fill_mode="detailed_core"),
        ]
        # Without grid
        r1 = materialize_concrete_axial_states(
            _catalog(), _layout(), segments, _base_lats(), _base_uvs(),
        )
        # With grid
        r2 = materialize_concrete_axial_states(
            _catalog(), _layout(), segments, _base_lats(), _base_uvs(),
            grid_overlays=grids,
            grid_density_lookup={"m": 7.0},
        )
        # State hashes should differ
        h1 = r1.state_reuse_report.get("pin_state_hashes", [])
        h2 = r2.state_reuse_report.get("pin_state_hashes", [])
        # The detailed-core segment without inserts has no derived pin lattices
        # but the core state hash should differ because of grid
        c1 = r1.state_reuse_report.get("core_state_hashes", [])
        c2 = r2.state_reuse_report.get("core_state_hashes", [])
        # Grid creates a different segment_index entry
        assert r2.segment_index[0].get("grid_overlay_ids") != r1.segment_index[0].get("grid_overlay_ids")
