"""Tests for material normalization."""

from __future__ import annotations

from openmc_agent.material_normalization import (
    NORMALIZATION_CONTRACT_VERSION,
    normalize_material_semantics,
)
from openmc_agent.schemas import (
    CompositionValueBasis,
    MaterialSpec,
    NormalizationStatus,
    NuclideSpec,
)


def _n(name, pct, pt="ao"):
    return NuclideSpec(name=name, percent=pct, percent_type=pt)


def test_stoichiometric_normalized_to_atom_fraction():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    new_fuel, result = normalize_material_semantics(fuel)
    assert result.normalization_status == NormalizationStatus.DETERMINISTICALLY_NORMALIZED
    o16 = [c for c in new_fuel.composition if c.name == "O16"][0]
    assert o16.percent == 200.0


def test_atom_fraction_not_modified():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 200.0)],
        composition_basis=CompositionValueBasis.ATOM_FRACTION,
    )
    new_fuel, result = normalize_material_semantics(fuel)
    assert result.normalization_status == NormalizationStatus.NOT_REQUIRED
    assert all(
        r.percent == c.percent
        for r, c in zip(new_fuel.composition, fuel.composition)
    )


def test_stoichiometric_already_expanded_oxygen_not_expanded_again():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 200.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    new_fuel, result = normalize_material_semantics(fuel)
    assert result.normalization_status == NormalizationStatus.DETERMINISTICALLY_NORMALIZED
    o16 = [c for c in new_fuel.composition if c.name == "O16"][0]
    assert o16.percent == 200.0


def test_ppm_normalized_to_atom_fraction():
    water = MaterialSpec(
        name="borated_water", density_unit="g/cm3", density_value=0.743,
        composition=[_n("B10", 0.001066), _n("B11", 0.004), _n("H1", 0.666), _n("O16", 0.329)],
        composition_basis=CompositionValueBasis.PPM_BY_WEIGHT,
    )
    new_water, result = normalize_material_semantics(water)
    assert result.normalization_status == NormalizationStatus.DETERMINISTICALLY_NORMALIZED
    b10 = [c for c in new_water.composition if c.name == "B10"][0]
    assert b10.percent < 0.0005


def test_normalization_does_not_modify_original():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    normalize_material_semantics(fuel)
    assert fuel.composition[2].percent == 2.0


def test_normalization_records_provenance():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    new_fuel, result = normalize_material_semantics(fuel)
    assert len(result.operations) == 1
    assert result.operations[0].operation == "uo2_stoichiometric_expansion"
    assert new_fuel.normalization_version == NORMALIZATION_CONTRACT_VERSION
    assert new_fuel.original_composition is not None
    assert new_fuel.normalized_composition is not None


def test_normalization_hash_changes():
    fuel = MaterialSpec(
        name="fuel", density_unit="g/cm3", density_value=10.0,
        composition=[_n("U235", 2.619), _n("U238", 97.381), _n("O16", 2.0)],
        composition_basis=CompositionValueBasis.STOICHIOMETRIC_RATIO,
    )
    _, result = normalize_material_semantics(fuel)
    assert result.original_hash != result.normalized_hash


def test_weight_fraction_passes_through():
    steel = MaterialSpec(
        name="steel", density_unit="g/cm3", density_value=7.8,
        composition=[_n("Fe56", 70.0, "wo"), _n("Cr52", 18.0, "wo"), _n("Ni58", 12.0, "wo")],
        composition_basis=CompositionValueBasis.WEIGHT_FRACTION,
    )
    _, result = normalize_material_semantics(steel)
    assert result.normalization_status == NormalizationStatus.NOT_REQUIRED
