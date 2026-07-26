"""Tests for material validation and invariants."""

from __future__ import annotations

from openmc_agent.material_normalization import normalize_material_semantics
from openmc_agent.material_validation import (
    compute_material_invariants,
    validate_normalized_material,
)
from openmc_agent.schemas import (
    CompositionValueBasis,
    MaterialSpec,
    NormalizationStatus,
    NuclideSpec,
)
from openmc_agent.plan_builder.patches import MaterialsPatch
from openmc_agent.plan_builder.validators import validate_patch


def _n(name, pct, pt="ao"):
    return NuclideSpec(name=name, percent=pct, percent_type=pt)


def test_uo2_invariants_after_normalization():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    new_fuel, _ = normalize_material_semantics(fuel)
    invariants = compute_material_invariants(new_fuel)
    checks = {c.name: c for c in invariants}
    assert checks["o_u_ratio"].passed
    assert checks["uranium_isotope_sum"].passed


def test_o_u_ratio_fails_without_normalization():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.ATOM_FRACTION,
    )
    invariants = compute_material_invariants(fuel)
    o_u_check = next(c for c in invariants if c.name == "o_u_ratio")
    assert not o_u_check.passed


def test_water_invariants():
    water = MaterialSpec(
        name="water", density_unit="g/cm3", density_value=0.743,
        composition=[_n("H1", 0.666), _n("O16", 0.333)],
        composition_basis=CompositionValueBasis.ATOM_FRACTION,
    )
    invariants = compute_material_invariants(water)
    h_o = next(c for c in invariants if c.name == "h_o_ratio")
    assert h_o.passed


def test_negative_fraction_detected():
    mat = MaterialSpec(
        name="bad", density_unit="g/cm3", density_value=1.0,
        composition=[_n("U235", 0.001), _n("U238", 99.999)],
        composition_basis=CompositionValueBasis.ATOM_FRACTION,
    )
    # Bypass Pydantic validation to simulate a negative fraction.
    mat.composition[0] = mat.composition[0].model_copy(update={"percent": -1.0})
    invariants = compute_material_invariants(mat)
    neg = next(c for c in invariants if c.name == "no_negative_fractions")
    assert not neg.passed


def test_duplicate_nuclides_detected():
    mat = MaterialSpec(
        name="bad", density_unit="g/cm3", density_value=1.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381)],
        composition_basis=CompositionValueBasis.ATOM_FRACTION,
    )
    # Simulate a duplicate by appending a copy.
    mat = mat.model_copy(update={
        "composition": [
            _n("U235", 2.619), _n("U235", 3.0), _n("U238", 97.381),
        ]
    })
    invariants = compute_material_invariants(mat)
    dup = next(c for c in invariants if c.name == "no_duplicate_nuclides")
    assert not dup.passed


def test_ambiguous_material_blocks_render():
    mat = MaterialSpec(
        name="mystery", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.UNKNOWN,
        normalization_status=NormalizationStatus.AMBIGUOUS,
    )
    result = validate_normalized_material(mat)
    assert not result.render_ready
    assert result.blockers


def test_normalized_material_passes_validation():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    new_fuel, _ = normalize_material_semantics(fuel)
    result = validate_normalized_material(new_fuel)
    assert result.render_ready


def test_materials_patch_rejects_stoichiometric_ratio_declared_as_atom_frac():
    patch = MaterialsPatch.model_validate({
        "patch_type": "materials",
        "materials": [{
            "material_id": "water",
            "name": "water",
            "role": "coolant",
            "density_g_cm3": 0.743,
            "composition": {"H1": 2.0, "O16": 1.0},
            "composition_basis": "atom_frac",
            "composition_status": "confirmed",
        }],
    })
    result = validate_patch(patch)
    assert any(
        issue.code == "materials.composition_fraction_sum_invalid"
        and issue.severity == "error"
        for issue in result.issues
    )


def test_materials_patch_accepts_fraction_sum():
    patch = MaterialsPatch.model_validate({
        "patch_type": "materials",
        "materials": [{
            "material_id": "water",
            "name": "water",
            "role": "coolant",
            "density_g_cm3": 0.743,
            "composition": {"H1": 0.6666667, "O16": 0.3333333},
            "composition_basis": "atom_frac",
            "composition_status": "confirmed",
        }],
    })
    result = validate_patch(patch)
    assert not any(issue.code == "materials.composition_fraction_sum_invalid" for issue in result.issues)


def test_materials_patch_rejects_percent_style_sum_under_fraction_basis():
    patch = MaterialsPatch.model_validate({
        "patch_type": "materials",
        "materials": [{
            "material_id": "steel",
            "name": "SS304",
            "role": "structural",
            "density_g_cm3": 7.9,
            "composition": {"Fe": 69.5, "Cr": 19.0, "Ni": 9.5, "Mn": 2.0},
            "composition_basis": "weight_frac",
            "composition_status": "approximate",
        }],
    })
    result = validate_patch(patch)
    assert any(
        issue.code == "materials.composition_fraction_sum_invalid"
        and issue.expected == "0 < sum <= 1.0"
        for issue in result.issues
    )


def test_materials_patch_accepts_partial_fraction_vector():
    patch = MaterialsPatch.model_validate({
        "patch_type": "materials",
        "materials": [{
            "material_id": "trace",
            "name": "trace",
            "role": "structural",
            "density_g_cm3": 1.0,
            "composition": {"B10": 0.0001, "B11": 0.0004},
            "composition_basis": "atom_frac",
            "composition_status": "approximate",
        }],
    })
    result = validate_patch(patch)
    assert not any(issue.code == "materials.composition_fraction_sum_invalid" for issue in result.issues)
