"""Tests for multi-assembly plot derivation and lattice bounds selection.

Reactor-neutral: uses synthetic models, not VERA4-specific data.
"""

from __future__ import annotations

import pytest

from openmc_agent.geometry_bounds import compute_geometry_bounds
from openmc_agent.plan_builder.assembler import _derive_multi_assembly_plots
from openmc_agent.schemas import (
    AxialLayerSpec,
    AxialOverlaySpec,
    ComplexModelSpec,
    CoreSpec,
    FillRefSpec,
    LatticeSpec,
    PlotSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mat_layer(name: str, z_min: float, z_max: float) -> AxialLayerSpec:
    return AxialLayerSpec(
        id=name, name=name, z_min_cm=z_min, z_max_cm=z_max,
        fill=FillRefSpec(type="material", id="struct_mat"),
    )


def _lat_layer(name: str, z_min: float, z_max: float) -> AxialLayerSpec:
    return AxialLayerSpec(
        id=name, name=name, z_min_cm=z_min, z_max_cm=z_max,
        fill=FillRefSpec(type="lattice", id="assembly_lattice"),
    )


def _overlay(oid: str, z_min: float, z_max: float) -> AxialOverlaySpec:
    return AxialOverlaySpec(
        id=oid, overlay_kind="spacer_grid", z_min_cm=z_min, z_max_cm=z_max,
        material_id="grid_mat", geometry_mode="mass_conserving_outer_frame",
        total_mass_g=500.0, target_lattice_id="assembly_lattice",
    )


# ---------------------------------------------------------------------------
# Test: _lattice_bounds picks largest lattice
# ---------------------------------------------------------------------------

def test_lattice_bounds_picks_largest() -> None:
    """compute_geometry_bounds should pick the core lattice (largest area),
    not the first assembly-level lattice."""
    core_lat = LatticeSpec(
        id="core_lattice", name="core_lattice", kind="rect",
        pitch_cm=(21.5, 21.5),
        universe_pattern=[["a", "b", "a"], ["b", "a", "b"], ["a", "b", "a"]],
        lower_left_cm=(-32.25, -32.25),
    )
    asm_lat = LatticeSpec(
        id="assembly_lat", name="assembly_lat", kind="rect",
        pitch_cm=(1.26, 1.26),
        universe_pattern=[["u1"] * 17 for _ in range(17)],
    )
    model = ComplexModelSpec(
        name="test", kind="core",
        materials=[], cells=[], surfaces=[], regions=[],
        universes=[], lattices=[asm_lat, core_lat],  # assembly first!
        core=CoreSpec(
            id="core", name="core",
            axial_layers=[_lat_layer("fuel", 0.0, 400.0)],
            axial_overlays=[],
        ),
    )
    gb = compute_geometry_bounds(model)
    assert gb is not None
    # Core lattice: 3*21.5 = 64.5 cm wide
    assert abs(gb.lattice_width[0] - 64.5) < 0.01
    assert abs(gb.lattice_width[1] - 64.5) < 0.01
    # NOT the assembly lattice (21.42 cm)
    assert gb.lattice_width[0] > 50.0


def test_lattice_bounds_single_assembly() -> None:
    """For single-assembly models (one lattice), behavior unchanged."""
    asm_lat = LatticeSpec(
        id="assembly_lat", name="assembly_lat", kind="rect",
        pitch_cm=(1.26, 1.26),
        universe_pattern=[["u1"] * 17 for _ in range(17)],
    )
    model = ComplexModelSpec(
        name="test", kind="assembly",
        materials=[], cells=[], surfaces=[], regions=[],
        universes=[], lattices=[asm_lat],
        core=CoreSpec(
            id="core", name="core",
            axial_layers=[_lat_layer("fuel", 0.0, 100.0)],
            axial_overlays=[],
        ),
    )
    gb = compute_geometry_bounds(model)
    assert gb is not None
    assert abs(gb.lattice_width[0] - 21.42) < 0.01


# ---------------------------------------------------------------------------
# Test: multi-height XY plots
# ---------------------------------------------------------------------------

def test_multi_assembly_plots_has_multiple_xy_heights() -> None:
    """Should produce ≥3 XY plots at different z heights."""
    layers = [
        _mat_layer("lower_nozzle", 0.0, 10.0),
        _lat_layer("fuel_bottom", 10.0, 100.0),
        _lat_layer("fuel_mid", 100.0, 200.0),
        _lat_layer("fuel_top", 200.0, 300.0),
        _mat_layer("upper_nozzle", 300.0, 350.0),
    ]
    overlays = [
        _overlay("grid1", 50.0, 54.0),
        _overlay("grid2", 120.0, 124.0),
        _overlay("grid3", 200.0, 204.0),
    ]
    plots = _derive_multi_assembly_plots(layers, overlays, core_width=64.5)

    xy_plots = [p for p in plots if p.basis == "xy"]
    assert len(xy_plots) >= 3, f"Expected ≥3 XY plots, got {len(xy_plots)}"

    # z heights must be distinct
    z_values = [p.origin[2] for p in xy_plots]
    assert len(set(round(z, 1) for z in z_values)) == len(z_values), \
        f"Z heights not distinct: {z_values}"

    # One should be near active fuel mid (~155 cm)
    assert any(100 < z < 200 for z in z_values), \
        f"No XY plot near active fuel mid: {z_values}"


def test_multi_assembly_plots_includes_grid_height() -> None:
    """When overlays exist, one XY plot should be at grid mid-elevation."""
    layers = [
        _mat_layer("nozzle", 0.0, 10.0),
        _lat_layer("fuel", 10.0, 300.0),
        _mat_layer("upper", 300.0, 350.0),
    ]
    overlays = [
        _overlay("g1", 50.0, 54.0),
        _overlay("g2", 150.0, 154.0),  # middle overlay
        _overlay("g3", 250.0, 254.0),
    ]
    plots = _derive_multi_assembly_plots(layers, overlays, core_width=64.5)
    # Grid mid-elevation should be ~152 cm
    grid_plot = next(
        (p for p in plots if "grid" in p.filename), None,
    )
    assert grid_plot is not None
    assert 150 <= grid_plot.origin[2] <= 155


def test_multi_assembly_plots_no_overlays() -> None:
    """Without overlays, still produces active fuel + structural plots."""
    layers = [
        _mat_layer("lower", 0.0, 10.0),
        _lat_layer("fuel", 10.0, 300.0),
        _mat_layer("upper", 300.0, 350.0),
    ]
    plots = _derive_multi_assembly_plots(layers, [], core_width=64.5)
    xy_plots = [p for p in plots if p.basis == "xy"]
    assert len(xy_plots) >= 2  # at least active fuel + structural
    assert not any("grid" in p.filename for p in plots)


def test_multi_assembly_plots_xz_height_adaptive() -> None:
    """XZ plot axial height must match actual geometry, not hardcoded 400."""
    layers = [
        _mat_layer("lower", -10.0, 0.0),
        _lat_layer("fuel", 0.0, 200.0),
        _mat_layer("upper", 200.0, 250.0),
    ]
    plots = _derive_multi_assembly_plots(layers, [], core_width=64.5)
    xz_plot = next(p for p in plots if p.basis == "xz")
    expected_height = 250.0 - (-10.0)  # 260.0
    assert abs(xz_plot.width_cm[1] - expected_height) < 0.01
    assert xz_plot.width_cm[1] != 400.0  # NOT hardcoded


def test_multi_assembly_plots_core_width() -> None:
    """All plots should use the provided core_width."""
    layers = [_lat_layer("fuel", 0.0, 100.0)]
    plots = _derive_multi_assembly_plots(layers, [], core_width=80.0)
    for p in plots:
        if p.basis == "xy":
            assert abs(p.width_cm[0] - 80.0) < 0.01
        elif p.basis == "xz":
            assert abs(p.width_cm[0] - 80.0) < 0.01


def test_multi_assembly_plots_filenames_distinct() -> None:
    """Each plot should have a unique filename."""
    layers = [
        _mat_layer("lower", 0.0, 10.0),
        _lat_layer("fuel", 10.0, 300.0),
        _mat_layer("upper", 300.0, 350.0),
    ]
    overlays = [_overlay("g1", 100.0, 104.0)]
    plots = _derive_multi_assembly_plots(layers, overlays, core_width=64.5)
    filenames = [p.filename for p in plots]
    assert len(filenames) == len(set(filenames)), \
        f"Duplicate filenames: {filenames}"


def test_multi_assembly_plots_include_localized_insert_slice_and_zoom() -> None:
    """Concrete segment metadata should add a slice through active inserts.

    This prevents the default active-fuel midplane plot from missing rods or
    other inserts whose axial occupancy covers only part of the fuel height.
    """
    layers = [
        _mat_layer("lower", 0.0, 10.0),
        _lat_layer("fuel_lower", 10.0, 200.0),
        _lat_layer("fuel_inserted", 200.0, 300.0),
        _mat_layer("upper", 300.0, 350.0),
    ]
    segments = [
        {
            "segment_id": "seg_inserted",
            "z_min_cm": 220.0,
            "z_max_cm": 260.0,
            "active_types": ["inserted_assembly"],
        }
    ]
    plots = _derive_multi_assembly_plots(
        layers,
        [],
        core_width=64.5,
        assembly_pitch_cm=21.5,
        axial_segment_index=segments,
    )

    full = next(p for p in plots if p.filename == "full_core_xy_localized_insert.png")
    zoom = next(p for p in plots if p.filename == "center_assembly_xy_localized_insert.png")
    assert full.origin[2] == pytest.approx(240.0)
    assert full.pixels == (2400, 2400)
    assert zoom.width_cm == (21.5, 21.5)
    assert zoom.origin[2] == pytest.approx(240.0)
