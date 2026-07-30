"""Reactor-neutral contract tests for deterministic material species resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from openmc_agent.executor import _expand_compound_composition, _render_complex_material_definition
from openmc_agent.material_species import (
    classify_species_name,
    parse_empirical_formula,
    preflight_cross_sections,
    resolve_material_species,
)
from openmc_agent.plan_builder.patches import MaterialsPatch
from openmc_agent.plan_builder.assembler import assemble_simulation_plan_from_patches
from openmc_agent.plan_builder.validators import PatchValidationContext, validate_patch
from openmc_agent.schemas import ComplexMaterialSpec, NuclideSpec


def _pyrex_resolution():
    return resolve_material_species(
        material_id="glass", role="absorber", composition={}, composition_basis="weight_frac",
        compound_components=[
            {"formula": "B2O3", "fraction": 12.5, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"},
            {"formula": "SiO2", "fraction": 87.5, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"},
        ],
    )


def test_simple_empirical_formula_boundary() -> None:
    assert parse_empirical_formula("B2O3") == [("B", 2), ("O", 3)]
    assert parse_empirical_formula("SiO2") == [("Si", 1), ("O", 2)]
    assert classify_species_name("B10") == "nuclide"
    assert classify_species_name("U235") == "nuclide"
    assert classify_species_name("Am-242m") == "nuclide"
    assert classify_species_name("B") == "element"
    with pytest.raises(ValueError, match="unsupported_compound_formula"):
        parse_empirical_formula("Ca(OH)2")


def test_pyrex_weight_resolution_conserves_mass_and_merges_oxygen() -> None:
    resolution = _pyrex_resolution()
    assert resolution.ok
    assert set(resolution.resolved_elements) == {"B", "Si", "O"}
    assert resolution.resolved_elements["B"] == pytest.approx(3.882, abs=0.01)
    assert resolution.resolved_elements["Si"] == pytest.approx(40.901, abs=0.02)
    assert resolution.resolved_elements["O"] == pytest.approx(55.217, abs=0.02)
    assert resolution.mass_balance_before == pytest.approx(100.0)
    assert resolution.mass_balance_after == pytest.approx(100.0)
    assert len([s for s in resolution.species if s.name == "O"]) == 1


def test_mixed_fraction_basis_degrades_to_audited_warning() -> None:
    resolution = resolve_material_species(
        material_id="mixed_glass",
        role="absorber",
        composition={"B": 0.1},
        composition_basis="weight_frac",
        compound_components=[
            {
                "formula": "SiO2",
                "fraction": 0.9,
                "fraction_basis": "atom_frac",
                "isotope_policy": "natural_elements",
            }
        ],
    )

    assert resolution.ok
    assert "material_resolution.mixed_fraction_basis" not in resolution.errors
    assert "material_resolution.mixed_fraction_basis_fallback" in resolution.warnings
    assert resolution.fraction_basis in {"weight_frac", "atom_frac"}


def test_unsupported_formula_and_fissile_formula_fail_closed() -> None:
    unsupported = resolve_material_species(
        material_id="lime", role="structural", composition={}, composition_basis="weight_frac",
        compound_components=[{"formula": "Ca(OH)2", "fraction": 1, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"}],
    )
    assert "materials.unsupported_compound_formula" in unsupported.errors
    fuel = resolve_material_species(
        material_id="fuel", role="fuel", composition={}, composition_basis="weight_frac",
        compound_components=[{"formula": "UO2", "fraction": 100, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"}],
    )
    assert "materials.fissile_compound_isotope_policy_missing" in fuel.errors
    assert "materials.fissile_compound_would_erase_enrichment" in fuel.errors


def test_validator_rejects_formula_in_transport_composition_and_accepts_typed_components() -> None:
    invalid = MaterialsPatch.model_validate({"patch_type": "materials", "materials": [{
        "material_id": "glass", "name": "glass", "role": "absorber", "density_g_cm3": 2.25,
        "composition": {"B2O3": 12.5}, "composition_basis": "weight_frac",
    }]})
    result = validate_patch(invalid, PatchValidationContext())
    assert "materials.compound_in_transport_composition" in [issue.code for issue in result.issues]

    valid = MaterialsPatch.model_validate({"patch_type": "materials", "materials": [{
        "material_id": "glass", "name": "glass", "role": "absorber", "density_g_cm3": 2.25,
        "composition": {}, "composition_basis": "weight_frac",
        "compound_components": [
            {"formula": "B2O3", "fraction": 12.5, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"},
            {"formula": "SiO2", "fraction": 87.5, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"},
        ],
    }]})
    assert validate_patch(valid, PatchValidationContext()).ok


def test_renderer_uses_canonical_species_and_legacy_is_audited() -> None:
    resolution = _pyrex_resolution()
    spec = ComplexMaterialSpec(
        id="glass", name="glass", density_value=2.25, density_unit="g/cm3",
        composition=[NuclideSpec(name=s.name, percent=s.fraction, percent_type="wo", kind=s.kind) for s in resolution.species],
    )
    rendered = _render_complex_material_definition(spec)
    assert "add_nuclide('B2O3'" not in rendered
    assert "add_nuclide('SiO2'" not in rendered
    assert "add_element('B'" in rendered
    legacy = resolve_material_species(
        material_id="legacy", role="absorber", composition={"B2O3": 100.0},
        composition_basis="weight_frac", legacy_compatibility=True,
    )
    assert legacy.ok
    assert legacy.normalization_events


def test_cross_section_preflight_catches_missing_nuclide_before_openmc(tmp_path: Path) -> None:
    xs = tmp_path / "cross_sections.xml"
    xs.write_text('<cross_sections><library materials="B10 B11 O16 Si28" /></cross_sections>', encoding="utf-8")
    resolution = resolve_material_species(
        material_id="bad", role="absorber", composition={"Xe999": 1.0}, composition_basis="atom_frac",
    )
    errors = preflight_cross_sections([resolution], str(xs))
    assert errors[0]["code"] == "runtime.nuclide_not_in_cross_sections"
    assert errors[0]["species_name"] == "Xe999"


def test_vera4_pyrex_typed_resolution_and_fuel_vector_regression() -> None:
    """The VERA4 fixture keeps fuel nuclides while Pyrex resolves to B/Si/O."""
    from scripts.vera4_base_fixture import build_all_vera4_patches

    result = assemble_simulation_plan_from_patches(build_all_vera4_patches())
    assert result.ok
    assert result.plan is not None
    pyrex = next(m for m in result.plan.complex_model.materials if m.id == "pyrex_glass")
    assert {c.name for c in pyrex.composition} == {"B", "Si", "O"}
    assert all(c.kind == "element" for c in pyrex.composition)
    fuel = next(m for m in result.plan.complex_model.materials if m.id == "fuel_r1")
    assert {"U234", "U235", "U236", "U238", "O16"} <= {c.name for c in fuel.composition}
    report = result.material_species_resolution_report
    assert report is not None
    entry = next(item for item in report["materials"] if item["material_id"] == "pyrex_glass")
    assert entry["mass_balance_before"] == pytest.approx(100.0)
    assert entry["mass_balance_after"] == pytest.approx(100.0)


def test_legacy_tuple_and_model_component_emission_share_canonical_digest() -> None:
    """All direct/mixture renderer adapters use the same canonical resolver."""
    model_entries = _expand_compound_composition([
        NuclideSpec(name="B2O3", percent=12.5, percent_type="wo", kind="nuclide"),
        NuclideSpec(name="SiO2", percent=87.5, percent_type="wo", kind="nuclide"),
    ])
    tuple_entries = _expand_compound_composition([
        ("B2O3", 12.5, "wo"), ("SiO2", 87.5, "wo"),
    ])
    model_digest = [(entry.name, round(entry.percent, 12), entry.kind) for entry in model_entries]
    tuple_digest = [(name, round(percent, 12), "element") for name, percent, _ in tuple_entries]
    assert model_digest == tuple_digest
