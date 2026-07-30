"""Tests for the pluggable renderer registry, skeleton fallback, and assembly checks."""

from pathlib import Path

import pytest

pytestmark = pytest.mark.openmc
openmc = pytest.importorskip(
    "openmc", reason="OpenMC is required for this integration test"
)

from openmc_agent.renderers import RENDERERS, choose_renderer, list_renderers
from openmc_agent.renderers.assembly import RectAssemblyRenderer
from openmc_agent.renderers.base import BaseRenderer
from openmc_agent.renderers.pin_cell import PinCellRenderer
from openmc_agent.renderers.skeleton import SkeletonRenderer
from openmc_agent.schemas import (
    AssemblySpec,
    AxialLayerSpec,
    CellSpec,
    ComplexMaterialSpec,
    ComplexModelSpec,
    CoreBoundarySpec,
    CoreSpec,
    ExecutionCheckSpec,
    GeometrySpec,
    LatticeLoadingSpec,
    LatticeSpec,
    MaterialSpec,
    NuclideSpec,
    PinCellSpec,
    PlotSpec,
    RenderCapabilityReport,
    RunSettingsSpec,
    SimulationPlan,
    SimulationSpec,
    SurfaceSpec,
    UniverseSpec,
)


def _pin_cell_plan() -> SimulationPlan:
    fuel = MaterialSpec(
        name="UO2 fuel",
        density_unit="g/cm3",
        density_value=10.4,
        composition=[NuclideSpec(name="U235", percent=4.95), NuclideSpec(name="O16", percent=2.0)],
    )
    moderator = MaterialSpec(
        name="Water",
        density_unit="g/cm3",
        density_value=1.0,
        composition=[NuclideSpec(name="H1", percent=2.0), NuclideSpec(name="O16", percent=1.0)],
    )
    spec = SimulationSpec(
        name="UO2 pin-cell",
        pin_cell=PinCellSpec(
            fuel=fuel,
            moderator=moderator,
            geometry=GeometrySpec(fuel_radius_cm=0.41, pitch_cm=1.26),
        ),
    )
    return SimulationPlan(
        model_spec=spec,
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.26, 1.26), filename="pin.png")],
        execution_check=ExecutionCheckSpec(
            settings=RunSettingsSpec(batches=5, inactive=1, particles=100)
        ),
    )


def _assembly_plan(*, materials, universe_pattern=None, shape=None, boundary="reflective") -> SimulationPlan:
    pattern = universe_pattern or [["pin", "pin"], ["pin", "pin"]]
    model = ComplexModelSpec(
        name="test assembly",
        kind="assembly",
        materials=materials,
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[UniverseSpec(id="pin", name="pin", cell_ids=["fuel_cell"])],
        lattices=[
            LatticeSpec(
                id="assembly_lattice",
                name="lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                shape=shape,
                universe_pattern=pattern,
            )
        ],
        assemblies=[
            AssemblySpec(
                id="assembly",
                name="root",
                lattice_id="assembly_lattice",
                boundary=boundary,
            )
        ],
        settings=RunSettingsSpec(batches=8, inactive=2, particles=80),
    )
    return SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(2.52, 2.52), filename="assembly.png")],
        execution_check=ExecutionCheckSpec(
            settings=RunSettingsSpec(batches=4, inactive=1, particles=20)
        ),
    )


def _complete_fuel() -> ComplexMaterialSpec:
    return ComplexMaterialSpec(
        id="fuel",
        name="UO2 fuel",
        density_unit="g/cm3",
        density_value=10.4,
        composition=[NuclideSpec(name="U235", percent=4.95), NuclideSpec(name="O16", percent=2.0)],
    )


# -- registry plumbing ----------------------------------------------------


def test_registry_orders_skeleton_last() -> None:
    names = [renderer.name for renderer in list_renderers()]
    assert names[-1] == "skeleton"
    assert "pin_cell" in names and "assembly" in names


def test_all_renderers_implement_interface() -> None:
    for renderer in RENDERERS:
        assert isinstance(renderer, BaseRenderer)
        assert renderer.name
        assert hasattr(renderer, "can_render")
        assert hasattr(renderer, "render")


# -- pin cell backward compatibility --------------------------------------


def test_existing_pin_cell_still_works(tmp_path: Path) -> None:
    plan = _pin_cell_plan()
    renderer, capability = choose_renderer(plan)
    assert isinstance(renderer, PinCellRenderer)
    assert capability.renderability == "runnable"
    assert capability.is_executable is True

    result = renderer.render(plan, tmp_path)
    assert result.renderability == "runnable"
    assert "model.export_to_xml()" in result.script
    assert "openmc.run()" not in result.script
    assert (tmp_path / "model.py").exists()


# -- pin cell mixed percent_type handling ---------------------------------


def _mixed_pin_cell_plan(
    *,
    chemical_formula: str | None = None,
    enrichment_percent: float | None = None,
) -> SimulationPlan:
    """VERA-style pin cell whose fuel mixes wo (U isotopes) and ao (O16)."""
    fuel = MaterialSpec(
        name="UO2 Fuel",
        density_unit="g/cm3",
        density_value=10.257,
        composition=[
            NuclideSpec(name="U234", percent=0.0263, percent_type="wo"),
            NuclideSpec(name="U235", percent=3.1, percent_type="wo"),
            NuclideSpec(name="U238", percent=96.8594, percent_type="wo"),
            NuclideSpec(name="O16", percent=2.0, percent_type="ao"),
        ],
        chemical_formula=chemical_formula,
        enrichment_percent=enrichment_percent,
    )
    moderator = MaterialSpec(
        name="Water",
        density_unit="g/cm3",
        density_value=0.743,
        composition=[
            NuclideSpec(name="H1", percent=2.0),
            NuclideSpec(name="O16", percent=1.0),
        ],
    )
    spec = SimulationSpec(
        name="VERA pin-cell",
        pin_cell=PinCellSpec(
            fuel=fuel,
            moderator=moderator,
            geometry=GeometrySpec(fuel_radius_cm=0.4096, pitch_cm=1.26),
        ),
    )
    return SimulationPlan(
        model_spec=spec,
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.26, 1.26), filename="pin.png")],
        execution_check=ExecutionCheckSpec(
            settings=RunSettingsSpec(batches=5, inactive=1, particles=100)
        ),
    )


def test_pin_cell_mixed_percent_without_formula_is_blocked(tmp_path: Path) -> None:
    """Mixed ao/wo without chemical_formula must not reach the renderer."""
    from openmc_agent.validator import validate_simulation_spec

    plan = _mixed_pin_cell_plan()  # no chemical_formula
    report = validate_simulation_spec(plan.model_spec)
    assert not report.is_valid
    assert any("mixes atom and weight percents" in error for error in report.errors)
    assert any(
        issue.code == "material.pin_cell.mixed_percent_no_formula"
        for issue in report.issues
    )

    renderer, _capability = choose_renderer(plan)
    # The pin-cell renderer is skipped; skeleton fallback handles it.
    assert not isinstance(renderer, PinCellRenderer)


def test_pin_cell_mixed_percent_uses_formula_fallback(tmp_path: Path) -> None:
    """With chemical_formula, the renderer emits add_elements_from_formula."""
    from openmc_agent.validator import validate_simulation_spec

    plan = _mixed_pin_cell_plan(chemical_formula="UO2", enrichment_percent=3.1)
    report = validate_simulation_spec(plan.model_spec)
    assert report.is_valid  # warning only, not an error
    assert any("chemical_formula fallback" in warning for warning in report.warnings)

    renderer, capability = choose_renderer(plan)
    assert isinstance(renderer, PinCellRenderer)
    assert capability.is_executable

    result = renderer.render(plan, tmp_path)
    assert "add_elements_from_formula('UO2'" in result.script
    assert "enrichment=3.1" in result.script
    # The fuel (material_0) goes through the formula fallback, so it must not
    # emit any add_nuclide call (moderator still does, which is fine).
    assert "material_0.add_nuclide" not in result.script


def test_pin_cell_formula_fallback_infers_enrichment_from_composition(
    tmp_path: Path,
) -> None:
    """U235 wt% in composition is used when enrichment_percent is null."""
    plan = _mixed_pin_cell_plan(chemical_formula="UO2")  # enrichment_percent None
    renderer, _capability = choose_renderer(plan)
    assert isinstance(renderer, PinCellRenderer)
    result = renderer.render(plan, tmp_path)
    assert "add_elements_from_formula('UO2'" in result.script
    assert "enrichment=3.1" in result.script


def test_build_openmc_material_raises_on_mixed_without_formula() -> None:
    """Runtime build path defends against mixed percents like the codegen path."""
    from openmc_agent.executor import build_openmc_material

    spec = MaterialSpec(
        name="bad fuel",
        density_unit="g/cm3",
        density_value=10.0,
        composition=[
            NuclideSpec(name="U235", percent=3.1, percent_type="wo"),
            NuclideSpec(name="O16", percent=2.0, percent_type="ao"),
        ],
    )
    with pytest.raises(ValueError, match="mixes atom and weight percents"):
        build_openmc_material(spec)


# -- skeleton for incomplete assembly -------------------------------------


def test_skeleton_renderer_for_incomplete_assembly(tmp_path: Path) -> None:
    plan = _assembly_plan(
        materials=[
            ComplexMaterialSpec(
                id="fuel",
                name="UO2 fuel",
                chemical_formula="UO2",
                requires_human_confirmation=["density", "enrichment"],
            )
        ]
    )
    renderer, capability = choose_renderer(plan)
    # Either the assembly renderer's skeleton mode or the generic skeleton fallback is acceptable.
    assert renderer is not None
    assert capability.renderability == "skeleton"
    assert capability.is_executable is False

    result = renderer.render(plan, tmp_path)
    assert result.renderability == "skeleton"
    assert "NOT EXECUTABLE" in result.script
    assert "TODO" in result.script
    # Skeleton never calls export_to_xml or openmc.run.
    assert "openmc.run()" not in result.script
    assert (tmp_path / "model.py").exists()
    assert (tmp_path / "capability_report.json").exists()
    assert (tmp_path / "TODO.md").exists()


def test_skeleton_renderer_for_unrecognized_kind(tmp_path: Path) -> None:
    model = ComplexModelSpec(
        name="mixed thing",
        kind="mixed",
        materials=[_complete_fuel()],
        requires_human_confirmation=["unknown subsystem layout"],
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.0, 1.0), filename="m.png")],
    )
    renderer, capability = choose_renderer(plan)
    assert isinstance(renderer, SkeletonRenderer)
    assert capability.renderability == "skeleton"
    result = renderer.render(plan, tmp_path)
    assert "NOT EXECUTABLE" in result.script


def test_core_renderer_rejects_mixed_percent_material_without_formula() -> None:
    model = ComplexModelSpec(
        name="bad core",
        kind="core",
        materials=[
            ComplexMaterialSpec(
                id="fuel",
                name="fuel",
                density_unit="g/cm3",
                density_value=10.0,
                composition=[
                    NuclideSpec(name="U235", percent=3.3, percent_type="wo"),
                    NuclideSpec(name="O16", percent=2.0, percent_type="ao"),
                ],
            )
        ],
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[UniverseSpec(id="pin", name="pin", cell_ids=["fuel_cell"])],
        lattices=[
            LatticeSpec(
                id="core_lattice",
                name="core lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                universe_pattern=[["pin"]],
            )
        ],
        core=CoreSpec(id="core", name="core", lattice_id="core_lattice"),
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.26, 1.26), filename="core_xy.png")],
    )

    _renderer, capability = choose_renderer(plan)

    assert capability.renderability == "skeleton"
    assert capability.supported_renderer == "core"
    assert any("mixes atom and weight percents" in reason for reason in capability.reasons)


def test_core_renderer_can_render_auto_materialized_missing_cells() -> None:
    model = ComplexModelSpec(
        name="repairable missing cells core",
        kind="core",
        materials=[
            ComplexMaterialSpec(
                id="uo2",
                name="UO2",
                density_unit="g/cm3",
                density_value=10.0,
                chemical_formula="UO2",
                enrichment_percent=3.3,
            ),
            ComplexMaterialSpec(
                id="water",
                name="water",
                density_unit="g/cm3",
                density_value=0.997,
                chemical_formula="H2O",
            ),
        ],
        cells=[],
        universes=[
            UniverseSpec(id="pin_uo2", name="pin", cell_ids=["pin_uo2_fuel_cell", "pin_uo2_mod_cell"]),
            UniverseSpec(id="water_universe", name="water", cell_ids=["water_cell"]),
        ],
        lattices=[
            LatticeSpec(
                id="core_lattice",
                name="core lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                universe_pattern=[["pin_uo2", "water_universe"]],
            )
        ],
        core=CoreSpec(id="core", name="core", lattice_id="core_lattice"),
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(2.52, 1.26), filename="core_xy.png")],
    )

    _renderer, capability = choose_renderer(plan)

    assert capability.supported_renderer == "core"
    assert capability.renderability != "skeleton"
    assert not any("references missing cells" in reason for reason in capability.reasons)


def test_core_renderer_blocks_lattice_expected_count_mismatch() -> None:
    model = ComplexModelSpec(
        name="pin count core",
        kind="core",
        materials=[_complete_fuel()],
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[
            UniverseSpec(id="pin_a", name="pin A", cell_ids=["fuel_cell"]),
            UniverseSpec(id="pin_b", name="pin B", cell_ids=["fuel_cell"]),
        ],
        lattices=[
            LatticeSpec(
                id="mox_assembly_lattice",
                name="MOX assembly lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                universe_pattern=[["pin_a", "pin_a"], ["pin_a", "pin_b"]],
                expected_counts={"pin_a": 2, "pin_b": 2},
            )
        ],
        core=CoreSpec(id="core", name="core", lattice_id="mox_assembly_lattice"),
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(2.52, 2.52), filename="core_xy.png")],
    )

    _renderer, capability = choose_renderer(plan)

    assert capability.supported_renderer == "core"
    assert capability.renderability == "skeleton"
    assert any(issue.code == "lattice.pin_count_mismatch" for issue in capability.issues)
    assert any("pin counts do not match expected_counts" in reason for reason in capability.reasons)


def test_core_renderer_reports_lattice_loading_reference_errors() -> None:
    model = ComplexModelSpec(
        name="bad loading core",
        kind="core",
        materials=[_complete_fuel()],
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[UniverseSpec(id="pin", name="pin", cell_ids=["fuel_cell"])],
        lattices=[
            LatticeSpec(
                id="core_lattice",
                name="core lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                universe_pattern=[["pin"]],
            )
        ],
        lattice_loadings=[
            LatticeLoadingSpec(
                id="bad_loading",
                base_lattice_id="missing_lattice",
                overrides={"missing_universe": [(3, 0)]},
            )
        ],
        core=CoreSpec(
            id="core",
            name="core",
            lattice_id="core_lattice",
            axial_layers=[
                AxialLayerSpec(
                    id="fuel",
                    name="fuel",
                    z_min_cm=0.0,
                    z_max_cm=1.0,
                    fill={"type": "lattice", "id": "bad_loading_lattice"},
                    loading_id="bad_loading",
                )
            ],
        ),
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.26, 1.26), filename="core_xy.png")],
    )

    _renderer, capability = choose_renderer(plan)

    assert capability.renderability == "skeleton"
    codes = {issue.code for issue in capability.issues}
    assert "lattice_loading.base_ref_missing" in codes


def _axial_assembly_plan(*, bad_grid_material_slab: bool = False) -> SimulationPlan:
    materials = [
        _complete_fuel(),
        ComplexMaterialSpec(
            id="moderator",
            name="water",
            density_unit="g/cm3",
            density_value=0.743,
            chemical_formula="H2O",
        ),
        ComplexMaterialSpec(
            id="grid_material",
            name="generic spacer grid alloy",
            density_unit="g/cm3",
            density_value=7.9,
            composition=[NuclideSpec(name="Fe56", percent=1.0)],
        ),
    ]
    cells = [
        CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel"),
        CellSpec(id="moderator_cell", name="moderator", fill_type="material", fill_id="moderator"),
        CellSpec(id="grid_cell", name="spacer grid", fill_type="material", fill_id="grid_material"),
    ]
    universes = [
        UniverseSpec(id="fuel_pin", name="fuel pin", cell_ids=["fuel_cell"]),
        UniverseSpec(id="moderator_pin", name="moderator pin", cell_ids=["moderator_cell"]),
        UniverseSpec(id="grid_overlay_pin", name="grid overlay pin", cell_ids=["grid_cell"]),
    ]
    lattices = [
        LatticeSpec(
            id="active_lattice",
            name="active assembly lattice",
            kind="rect",
            pitch_cm=(1.26, 1.26),
            universe_pattern=[["fuel_pin", "moderator_pin"], ["moderator_pin", "fuel_pin"]],
        )
    ]
    loading_id = None if bad_grid_material_slab else "grid_loading"
    lattice_loadings = [] if bad_grid_material_slab else [
        LatticeLoadingSpec(
            id="grid_loading",
            base_lattice_id="active_lattice",
            derived_lattice_id="grid_lattice",
            overrides={"grid_overlay_pin": [(0, 0), (0, 1), (1, 0), (1, 1)]},
        )
    ]
    grid_fill = (
        {"type": "material", "id": "grid_material"}
        if bad_grid_material_slab
        else {"type": "lattice", "id": "grid_lattice"}
    )
    model = ComplexModelSpec(
        name="generic 3D assembly",
        kind="assembly",
        materials=materials,
        cells=cells,
        universes=universes,
        lattices=lattices,
        lattice_loadings=lattice_loadings,
        assemblies=[
            AssemblySpec(
                id="assembly",
                name="root",
                lattice_id="active_lattice",
                boundary="reflective",
            )
        ],
        core=CoreSpec(
            id="axial_assembly",
            name="axial assembly root",
            lattice_id="active_lattice",
            boundary="mixed",
            boundary_conditions=CoreBoundarySpec(
                xmin="reflective",
                xmax="reflective",
                ymin="reflective",
                ymax="reflective",
                zmin="vacuum",
                zmax="vacuum",
            ),
            axial_layers=[
                AxialLayerSpec(
                    id="lower_active",
                    name="lower active",
                    z_min_cm=0.0,
                    z_max_cm=10.0,
                    fill={"type": "lattice", "id": "active_lattice"},
                ),
                AxialLayerSpec(
                    id="spacer_grid",
                    name="spacer grid",
                    z_min_cm=10.0,
                    z_max_cm=10.5,
                    fill=grid_fill,
                    loading_id=loading_id,
                ),
                AxialLayerSpec(
                    id="upper_active",
                    name="upper active",
                    z_min_cm=10.5,
                    z_max_cm=20.0,
                    fill={"type": "lattice", "id": "active_lattice"},
                ),
            ],
        ),
        settings=RunSettingsSpec(batches=8, inactive=2, particles=80),
    )
    return SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xz", width_cm=(2.52, 20.0), filename="axial.png")],
        execution_check=ExecutionCheckSpec(settings=RunSettingsSpec(batches=4, inactive=1, particles=20)),
    )


def test_rect_assembly_with_axial_layers_renders_3d_root(tmp_path: Path) -> None:
    plan = _axial_assembly_plan()
    renderer, capability = choose_renderer(plan)

    assert isinstance(renderer, RectAssemblyRenderer)
    assert capability.supported_renderer == "assembly"
    assert capability.renderability in {"exportable", "runnable"}
    assert "axial_layers" in capability.executable_subsystems
    assert "lattice_loadings" in capability.executable_subsystems

    result = renderer.render(plan, tmp_path)
    script = result.script

    assert "Generated OpenMC axial assembly model" in script
    assert "assembly_z_min = 0.0" in script
    assert "assembly_z_max = 20.0" in script
    assert "assembly_zmin = openmc.ZPlane(z0=assembly_z_min, boundary_type='vacuum')" in script
    assert "assembly_xmin = openmc.XPlane(x0=assembly_x_min, boundary_type='reflective')" in script
    assert "lattice_grid_lattice = openmc.RectLattice(name='derived from active_lattice')" in script
    assert "root_cell_spacer_grid = openmc.Cell(name='spacer grid', fill=lattices['grid_lattice']" in script
    assert "(-1.0)" not in script


def test_axial_spacer_grid_material_slab_is_not_exportable() -> None:
    plan = _axial_assembly_plan(bad_grid_material_slab=True)

    capability = RectAssemblyRenderer().can_render(plan)

    assert capability.renderability == "skeleton"
    assert any(
        issue.code == "assembly3d.spacer_grid_material_slab"
        for issue in capability.issues
    )


# -- assembly validation checks -------------------------------------------


def test_rect_assembly_lattice_shape_validation() -> None:
    plan = _assembly_plan(
        materials=[_complete_fuel()],
        universe_pattern=[["pin", "pin"], ["pin", "pin"]],
        shape=(15, 15),
    )
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    assert capability.renderability != "runnable"
    assert any("shape" in reason and "15" in reason for reason in capability.reasons)


def test_rect_assembly_missing_lattice_pattern_becomes_skeleton(tmp_path: Path) -> None:
    model = ComplexModelSpec(
        name="incomplete assembly",
        kind="assembly",
        materials=[_complete_fuel()],
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[UniverseSpec(id="pin", name="pin", cell_ids=["fuel_cell"])],
        lattices=[
            LatticeSpec(
                id="assembly_lattice",
                name="lattice",
                kind="rect",
                pitch_cm=(1.26, 1.26),
                shape=(17, 17),
                universe_pattern=None,
            )
        ],
        assemblies=[AssemblySpec(id="assembly", name="root", lattice_id="assembly_lattice")],
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        model_spec=None,
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(21.42, 21.42), filename="assembly.png")],
    )

    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    result = renderer.render(plan, tmp_path)

    assert capability.renderability == "skeleton"
    assert any("requires universe_pattern" in reason for reason in capability.reasons)
    assert "universe_pattern missing" in result.script
    assert "lattice assembly_lattice: rect lattice universe_pattern is missing" in (
        tmp_path / "TODO.md"
    ).read_text(encoding="utf-8")


def test_rect_assembly_missing_universe_reference() -> None:
    plan = _assembly_plan(
        materials=[_complete_fuel()],
        universe_pattern=[["pin", "ghost"], ["pin", "pin"]],
    )
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    assert capability.renderability not in {"exportable", "runnable"}
    assert any("missing universes" in reason and "ghost" in reason for reason in capability.reasons)


def test_rect_assembly_material_completeness_missing_density() -> None:
    plan = _assembly_plan(
        materials=[
            ComplexMaterialSpec(
                id="fuel",
                name="UO2 fuel",
                chemical_formula="UO2",
                requires_human_confirmation=["density"],
            )
        ]
    )
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    assert capability.is_executable is False
    assert capability.renderability == "skeleton"
    assert any("missing density" in reason for reason in capability.reasons)


def test_rect_assembly_material_completeness_missing_composition() -> None:
    plan = _assembly_plan(
        materials=[
            ComplexMaterialSpec(
                id="fuel",
                name="UO2 fuel",
                density_unit="g/cm3",
                density_value=10.4,
                requires_human_confirmation=["composition"],
            )
        ]
    )
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    assert capability.renderability == "skeleton"
    assert any(
        "missing composition or chemical_formula" in reason for reason in capability.reasons
    )


def test_rect_assembly_complete_is_exportable_or_runnable() -> None:
    plan = _assembly_plan(materials=[_complete_fuel()])
    renderer, capability = choose_renderer(plan)
    assert isinstance(renderer, RectAssemblyRenderer)
    # Default execution_check settings (20 particles, 4 batches) are within smoke limits.
    assert capability.renderability in {"exportable", "runnable"}
    assert capability.is_executable is True


def test_hex_lattice_is_diagnosed_but_stays_skeleton() -> None:
    model = ComplexModelSpec(
        name="hex assembly",
        kind="assembly",
        materials=[_complete_fuel()],
        cells=[CellSpec(id="fuel_cell", name="fuel", fill_type="material", fill_id="fuel")],
        universes=[UniverseSpec(id="pin", name="pin", cell_ids=["fuel_cell"])],
        lattices=[
            LatticeSpec(
                id="hex_lat",
                name="hex lattice",
                kind="hex",
                pitch_cm=(1.26, 1.26),
                rings=[],
            )
        ],
        assemblies=[AssemblySpec(id="assembly", name="root", lattice_id="hex_lat")],
    )
    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        complex_model=model,
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(2.0, 2.0), filename="hex.png")],
    )

    _renderer, capability = choose_renderer(plan)

    assert capability.renderability == "skeleton"
    assert capability.is_executable is False
    codes = {issue.code for issue in capability.issues}
    assert "lattice.hex.renderer_unsupported" in codes
    assert "lattice.hex.rings_missing" in codes
    renderer_issue = next(
        issue for issue in capability.issues if issue.code == "lattice.hex.renderer_unsupported"
    )
    assert renderer_issue.route_hint == "capability_downgrade"
    assert "HexLattice" in renderer_issue.grep_patterns


def test_rect_assembly_warns_oversized_cylinder() -> None:
    plan = _assembly_plan(materials=[_complete_fuel()])
    plan.complex_model.surfaces = [
        SurfaceSpec(id="fuel_outer", kind="zcylinder", parameters={"r": 0.8}),
    ]
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    # pitch=1.26 -> pitch/2=0.63; r=0.8 >= 0.63. Oversized cylinders are
    # downgraded to a warning (geometry imperfection, not a structural break)
    # so the model can still render with the caveat recorded in warnings.
    assert any("pitch" in w.lower() for w in capability.warnings)


def test_rect_assembly_rejects_nonpositive_cylinder_radius() -> None:
    plan = _assembly_plan(materials=[_complete_fuel()])
    plan.complex_model.surfaces = [
        SurfaceSpec(id="fuel_outer", kind="zcylinder", parameters={"r": -0.1}),
    ]
    renderer = RectAssemblyRenderer()
    capability = renderer.can_render(plan)
    assert capability.renderability != "runnable"
    assert any("radius must be positive" in reason for reason in capability.reasons)


# -- renderer authoring stub ---------------------------------------------


def test_renderer_authoring_agent_reports_not_implemented() -> None:
    from openmc_agent.renderer_authoring import (
        AUTHORING_NOT_IMPLEMENTED,
        RendererAuthoringAgent,
    )

    plan = _pin_cell_plan()
    capability = RenderCapabilityReport(renderability="none", supported_renderer="none")
    candidate = RendererAuthoringAgent().propose_renderer(plan, capability)
    assert candidate.status == AUTHORING_NOT_IMPLEMENTED
    assert candidate.implemented is False
    # Safety policy is documented even though codegen is disabled.
    assert "subprocess" in candidate.safety_constraints.forbidden_modules
    assert any(
        call.startswith("subprocess") for call in candidate.safety_constraints.forbidden_calls
    )


def test_renderer_authoring_validator_rejects_forbidden_calls() -> None:
    from openmc_agent.renderer_authoring.validator import validate_renderer_source

    bad = "import os\nos.system('rm -rf /')\n"
    result = validate_renderer_source(bad)
    assert result.is_valid is False
    assert result.forbidden_modules_seen
