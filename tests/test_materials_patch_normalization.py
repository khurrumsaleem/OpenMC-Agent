"""Production MaterialsPatch deterministic normalization tests."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.materials_patch_normalization import (
    extract_soluble_boron_requirement,
    normalize_materials_patch_content,
)


def _boron_mass_fraction(atom: dict[str, float]) -> float:
    weights = {"H1": 1.00784, "O16": 15.994, "B10": 10.012937, "B11": 11.009305}
    total = sum(atom[name] * weights[name] for name in weights)
    boron = atom["B10"] * weights["B10"] + atom["B11"] * weights["B11"]
    return boron / total


def test_extracts_soluble_boron_ppm_and_isotope_split() -> None:
    req = extract_soluble_boron_requirement(
        "coolant: soluble boron 1360 ppm by mass\n"
        "natural boron B-10 isotope fraction 19.9 at%"
    )
    assert req is not None
    assert req.mass_fraction == pytest.approx(0.00136)
    assert req.b10_atom_fraction == pytest.approx(0.199)


def test_repairs_low_boron_coolant_atom_fraction_from_source_mass_ppm() -> None:
    patch = {
        "patch_type": "materials",
        "materials": [{
            "material_id": "coolant",
            "name": "Borated water",
            "role": "coolant",
            "density_g_cm3": 0.743,
            "composition": {
                "H1": 0.666633333,
                "O16": 0.333316667,
                "B10": 0.00001,
                "B11": 0.00004,
            },
            "composition_basis": "atom_frac",
            "composition_status": "confirmed",
        }],
    }
    result = normalize_materials_patch_content(
        patch,
        requirement_text=(
            "soluble boron 1360 ppm by mass\n"
            "natural boron B-10 isotope fraction 19.9 at%"
        ),
    )
    assert result.changed
    fixed = result.content["materials"][0]["composition"]
    assert _boron_mass_fraction(fixed) == pytest.approx(0.00136, rel=5e-4)
    assert result.operations[0]["operation"] == "coolant_boron_mass_ppm_atom_fraction_repair"


def test_does_not_modify_materials_without_source_boron_requirement() -> None:
    patch = {
        "patch_type": "materials",
        "materials": [{
            "material_id": "coolant",
            "name": "Water",
            "role": "coolant",
            "composition": {"H1": 0.6667, "O16": 0.3333},
            "composition_basis": "atom_frac",
        }],
    }
    result = normalize_materials_patch_content(patch, requirement_text="plain water coolant")
    assert not result.changed
    assert result.content == patch
