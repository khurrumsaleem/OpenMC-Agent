import pytest

pytestmark = pytest.mark.openmc
openmc = pytest.importorskip(
    "openmc", reason="OpenMC is required for this integration test"
)
from pathlib import Path
from types import SimpleNamespace

from openmc_agent.schemas import (
    AxialLayerSpec,
    CellSpec,
    ComplexMaterialSpec,
    ComplexModelSpec,
    CoreSpec,
    ExecutionCheckSpec,
    GeometrySpec,
    LatticeSpec,
    MaterialSpec,
    NuclideSpec,
    PinCellSpec,
    PlotSpec,
    RenderCapabilityReport,
    RunSettingsSpec,
    SimulationPlan,
    SimulationSpec,
    UniverseSpec,
)
from openmc_agent.tools import (
    export_xml,
    parse_openmc_output,
    run_geometry_plots,
    run_smoke_test,
)


def _plan(particles: int = 100, batches: int = 5) -> SimulationPlan:
    fuel = MaterialSpec(
        name="UO2 fuel",
        density_unit="g/cm3",
        density_value=10.4,
        composition=[
            NuclideSpec(name="U235", percent=4.95),
            NuclideSpec(name="U238", percent=95.05),
            NuclideSpec(name="O16", percent=200.0),
        ],
    )
    moderator = MaterialSpec(
        name="Water moderator",
        density_unit="g/cm3",
        density_value=1.0,
        composition=[
            NuclideSpec(name="H1", percent=2.0),
            NuclideSpec(name="O16", percent=1.0),
        ],
    )
    return SimulationPlan(
        model_spec=SimulationSpec(
            name="Tool test pin-cell",
            pin_cell=PinCellSpec(
                fuel=fuel,
                moderator=moderator,
                geometry=GeometrySpec(fuel_radius_cm=0.41, pitch_cm=1.26),
            ),
            settings=RunSettingsSpec(batches=50, inactive=10, particles=1000),
        ),
        plot_specs=[
            PlotSpec(
                basis="xy",
                origin=(0.0, 0.0, 0.0),
                width_cm=(1.26, 1.26),
                pixels=(200, 200),
                filename="geometry_xy.png",
            )
        ],
        execution_check=ExecutionCheckSpec(
            settings=RunSettingsSpec(
                batches=batches,
                inactive=1,
                particles=particles,
            )
        ),
    )


def test_export_xml_returns_subprocess_output_and_artifacts(tmp_path: Path, monkeypatch) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('export')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[-1] == "model.py"
        assert kwargs["cwd"] == tmp_path
        (tmp_path / "materials.xml").write_text("<materials />", encoding="utf-8")
        (tmp_path / "geometry.xml").write_text("<geometry />", encoding="utf-8")
        (tmp_path / "settings.xml").write_text("<settings />", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="exported", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is True
    assert result.returncode == 0
    assert result.stdout == "exported"
    assert str(tmp_path / "materials.xml") in result.artifacts


def test_export_xml_fails_when_required_xml_artifacts_are_missing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('skeleton')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[-1] == "model.py"
        return SimpleNamespace(returncode=0, stdout="skeleton", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is False
    assert result.returncode == 0
    assert result.artifacts == []
    assert "materials.xml" in result.error


def test_export_xml_fails_on_dangling_lattice_universe_reference(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('export')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[-1] == "model.py"
        (tmp_path / "materials.xml").write_text("<materials />", encoding="utf-8")
        (tmp_path / "settings.xml").write_text("<settings />", encoding="utf-8")
        (tmp_path / "geometry.xml").write_text(
            """<geometry>
  <cell id="1" universe="1" />
  <lattice id="7">
    <universes>1 99</universes>
  </lattice>
</geometry>
""",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="exported", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is False
    assert result.returncode == 0
    assert "lattice 7" in result.error
    assert "99" in result.error


def test_export_xml_fails_on_cell_fill_without_universe_or_lattice(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('export')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        assert command[-1] == "model.py"
        (tmp_path / "materials.xml").write_text("<materials />", encoding="utf-8")
        (tmp_path / "settings.xml").write_text("<settings />", encoding="utf-8")
        (tmp_path / "geometry.xml").write_text(
            """<geometry>
  <cell id="1" universe="1" />
  <cell id="18" fill="12" universe="1" />
</geometry>
""",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="exported", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is False
    assert "cell 18" in result.error
    assert "fill 12" in result.error


def test_run_geometry_plots_preserves_failure_stderr(tmp_path: Path, monkeypatch) -> None:
    def fake_run(command, **kwargs):
        assert command == ["openmc", "-p"]
        return SimpleNamespace(returncode=2, stdout="", stderr="Plot error")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = run_geometry_plots(tmp_path)

    assert result.ok is False
    assert result.returncode == 2
    assert result.stderr == "Plot error"


def test_run_smoke_test_rejects_settings_above_safety_limits(tmp_path: Path) -> None:
    result = run_smoke_test(
        tmp_path,
        _plan(particles=100000, batches=5),
        max_particles=1000,
        max_batches=20,
    )

    assert result.ok is False
    assert result.returncode is None
    assert "exceed safety limits" in result.error
    assert not (tmp_path / "smoke_model.py").exists()


def test_run_smoke_test_writes_smoke_script_and_runs_openmc(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "smoke_model.py":
            return SimpleNamespace(returncode=0, stdout="xml ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="k-effective 1.000 +/- 0.010", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = run_smoke_test(tmp_path, _plan(particles=120, batches=6))

    assert result.ok is True
    assert calls[0][-1] == "smoke_model.py"
    assert calls[1] == ["openmc"]
    assert "settings.particles = 120" in (tmp_path / "smoke_model.py").read_text(
        encoding="utf-8"
    )
    assert "k-effective" in result.stdout


def test_run_smoke_test_refreshes_core_renderer_capability(tmp_path: Path, monkeypatch) -> None:
    calls: list[list[str]] = []

    def fake_run(command, **kwargs):
        calls.append(command)
        if command[-1] == "smoke_model.py":
            return SimpleNamespace(returncode=0, stdout="xml ok", stderr="")
        return SimpleNamespace(returncode=0, stdout="Combined k-effective = 1.000 +/- 0.010", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    plan = SimulationPlan(
        schema_version="simulation_plan.v2",
        complex_model=ComplexModelSpec(
            name="core smoke test",
            kind="core",
            materials=[
                ComplexMaterialSpec(
                    id="fuel",
                    name="fuel",
                    density_unit="g/cm3",
                    density_value=10.0,
                    chemical_formula="UO2",
                    enrichment_percent=3.3,
                )
            ],
            cells=[CellSpec(id="pin_cell", name="pin", fill_type="material", fill_id="fuel")],
            universes=[UniverseSpec(id="pin", name="pin", cell_ids=["pin_cell"])],
            lattices=[
                LatticeSpec(
                    id="core_lattice",
                    name="core lattice",
                    kind="rect",
                    pitch_cm=(1.26, 1.26),
                    universe_pattern=[["pin"]],
                )
            ],
            core=CoreSpec(
                id="core",
                name="core",
                lattice_id="core_lattice",
                axial_layers=[
                    AxialLayerSpec(
                        id="active",
                        name="active",
                        z_min_cm=0.0,
                        z_max_cm=10.0,
                        fill={"type": "lattice", "id": "core_lattice"},
                    )
                ],
            ),
        ),
        capability_report=RenderCapabilityReport(is_executable=False, supported_renderer="none"),
        plot_specs=[PlotSpec(basis="xy", width_cm=(1.26, 1.26), filename="core_xy.png")],
        execution_check=ExecutionCheckSpec(
            settings=RunSettingsSpec(batches=6, inactive=1, particles=120)
        ),
    )

    result = run_smoke_test(tmp_path, plan)

    assert result.ok is True
    assert calls[0][-1] == "smoke_model.py"
    assert calls[1] == ["openmc"]
    script = (tmp_path / "smoke_model.py").read_text(encoding="utf-8")
    assert "openmc.RectLattice" in script
    assert "settings.particles = 120" in script
    assert "Combined k-effective" in result.stdout


def test_parse_openmc_output_extracts_common_diagnostics() -> None:
    report = parse_openmc_output(
        stdout="WARNING: After particle 1 crossed surface, it could not be located in any cell",
        stderr="ERROR: No cross_sections.xml was specified\nTraceback (most recent call last)",
    )

    assert report.is_valid is False
    text = " ".join(report.errors + report.warnings)
    assert "cross section" in text
    assert "undefined region" in text
    assert "Python traceback" in text


def test_parse_openmc_output_maps_cross_sections_missing_to_stable_code() -> None:
    report = parse_openmc_output(
        stdout="",
        stderr="ERROR: No cross_sections.xml was specified",
    )

    assert report.is_valid is False
    issue = report.issues[0]
    assert issue.code == "runtime.cross_sections_missing"
    assert issue.route_hint == "ask_expert"
    assert issue.requires_human_confirmation is True
    assert any("OPENMC_CROSS_SECTIONS" in pattern for pattern in issue.grep_patterns)


def test_parse_openmc_output_maps_geometry_overlap_to_reflect_plan() -> None:
    report = parse_openmc_output(
        stdout="",
        stderr="ERROR: Overlap detected between cells 10 and 11",
    )

    assert report.is_valid is False
    issue = report.issues[0]
    assert issue.code == "runtime.geometry_overlap"
    assert issue.route_hint == "reflect_plan"
    assert issue.requires_retrieval is True


def test_parse_openmc_output_maps_unknown_runtime_error() -> None:
    report = parse_openmc_output(stdout="", stderr="ERROR: unexpected OpenMC failure")

    assert report.is_valid is False
    assert report.issues[0].code == "runtime.openmc_unknown_error"


def test_parse_openmc_output_accepts_successful_cross_section_reads() -> None:
    report = parse_openmc_output(
        stdout=(
            "Reading cross sections XML file...\n"
            "Reading U235 from /home/wbd/openmc_data/endfb-vii.1-hdf5/neutron/U235.h5\n"
            "Reading c_H_in_H2O from /home/wbd/openmc_data/endfb-vii.1-hdf5/neutron/c_H_in_H2O.h5\n"
            "Creating state point statepoint.15.h5...\n"
            "Combined k-effective = 1.00000 +/- 0.01000\n"
        ),
        stderr="",
    )

    assert report.is_valid is True
    assert report.errors == []


def test_export_xml_dangling_lattice_reference_has_issue_and_auto_repair_route(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('export')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        (tmp_path / "materials.xml").write_text("<materials />", encoding="utf-8")
        (tmp_path / "settings.xml").write_text("<settings />", encoding="utf-8")
        (tmp_path / "geometry.xml").write_text(
            """<geometry>
  <cell id="1" universe="1" />
  <lattice id="7">
    <universes>2</universes>
  </lattice>
</geometry>
""",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="exported", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is False
    assert result.issues[0].code == "export_xml.dangling_lattice_universe"
    assert result.issues[0].route_hint == "auto_repair"
    assert "2" in result.issues[0].grep_patterns


def test_export_xml_dangling_cell_fill_has_structured_issue(
    tmp_path: Path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "model.py"
    model_path.write_text("print('export')\n", encoding="utf-8")

    def fake_run(command, **kwargs):
        (tmp_path / "materials.xml").write_text("<materials />", encoding="utf-8")
        (tmp_path / "settings.xml").write_text("<settings />", encoding="utf-8")
        (tmp_path / "geometry.xml").write_text(
            """<geometry>
  <cell id="1" universe="1" fill="99" />
</geometry>
""",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="exported", stderr="")

    monkeypatch.setattr("openmc_agent.tools.subprocess.run", fake_run)

    result = export_xml(model_path)

    assert result.ok is False
    assert result.issues[0].code == "export_xml.dangling_cell_fill"
    assert result.issues[0].route_hint in {"reflect_plan", "manual_review"}
    assert "99" in result.issues[0].grep_patterns
