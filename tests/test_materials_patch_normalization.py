"""Production MaterialsPatch deterministic normalization tests."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.materials_patch_normalization import (
    extract_soluble_boron_requirement,
    normalize_materials_patch_content,
    normalize_materials_patches_in_state,
)
from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.state import PlanBuildState


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


def test_repairs_schema_surface_enum_and_element_component_drift() -> None:
    patch = {
        "patch_type": "materials",
        "materials": [
            {
                "material_id": "fuel",
                "name": "fuel",
                "role": "fuel",
                "density_g_cm3": 10.0,
                "composition_basis": "unknown",
                "compound_components": [{
                    "formula": "UO2",
                    "fraction": 1.0,
                    "fraction_basis": "stoichiometric_ratio",
                    "isotope_policy": "explicit",
                    "isotope_overrides": {"U235": 2.0},
                }],
            },
            {
                "material_id": "ss304",
                "name": "SS304",
                "role": "structural",
                "density_g_cm3": 8.0,
                "composition_basis": "weight_frac",
                "compound_components": [
                    {"formula": "Fe", "fraction": 0.70, "fraction_basis": "weight_frac"},
                    {"formula": "Cr", "fraction": 0.19, "fraction_basis": "weight_frac"},
                    {"formula": "Ni", "fraction": 0.10, "fraction_basis": "weight_frac"},
                ],
            },
        ],
    }

    result = normalize_materials_patch_content(patch, requirement_text="plain materials")

    assert result.changed
    fuel_component = result.content["materials"][0]["compound_components"][0]
    assert fuel_component["fraction_basis"] == "atom_frac"
    assert fuel_component["isotope_policy"] == "explicit_isotopes"
    assert fuel_component["isotope_overrides"]["U235"] == {"fraction": 2.0}
    steel = result.content["materials"][1]
    assert steel["compound_components"] == []
    assert steel["composition"] == {"Fe": 0.70, "Cr": 0.19, "Ni": 0.10}
    parse_patch_content("materials", result.content)


def test_redundant_compound_component_does_not_double_composition() -> None:
    """When the LLM declares an element in BOTH composition and compound_components
    (a common redundancy), the compound component must be dropped, not added —
    otherwise the fraction doubles (e.g. AIC 0.80+0.15+0.05 summing to 2.0)."""
    patch = {
        "patch_type": "materials",
        "materials": [
            {
                "material_id": "aic",
                "name": "AIC",
                "role": "poison",
                "density_g_cm3": 10.2,
                "composition_basis": "weight_frac",
                "composition": {"Ag": 0.8, "In": 0.15, "Cd": 0.05},
                "compound_components": [
                    {"formula": "Ag", "fraction": 0.8, "fraction_basis": "weight_frac"},
                    {"formula": "In", "fraction": 0.15, "fraction_basis": "weight_frac"},
                    {"formula": "Cd", "fraction": 0.05, "fraction_basis": "weight_frac"},
                ],
            },
        ],
    }

    result = normalize_materials_patch_content(patch, requirement_text="plain materials")

    aic = result.content["materials"][0]
    # Composition unchanged (not doubled).
    assert aic["composition"] == {"Ag": 0.8, "In": 0.15, "Cd": 0.05}
    # Redundant compound components dropped.
    assert aic["compound_components"] == []
    ops = [op["operation"] for op in result.operations]
    assert ops.count("compound_element_component_redundant_dropped") == 3
    assert "compound_element_component_to_composition_repair" not in ops
    parse_patch_content("materials", result.content)


def test_multiple_compound_components_same_species_still_accumulate() -> None:
    """Two compound components contributing to the same species (with no prior
    composition entry) must still accumulate; only pre-existing composition
    entries suppress the move."""
    patch = {
        "patch_type": "materials",
        "materials": [
            {
                "material_id": "mix",
                "name": "mix",
                "role": "structural",
                "density_g_cm3": 8.0,
                "compound_components": [
                    {"formula": "Fe", "fraction": 0.40, "fraction_basis": "weight_frac"},
                    {"formula": "Fe", "fraction": 0.30, "fraction_basis": "weight_frac"},
                ],
            },
        ],
    }

    result = normalize_materials_patch_content(patch, requirement_text="plain materials")
    mix = result.content["materials"][0]
    assert mix["composition"] == {"Fe": 0.70}


def test_fuel_source_variant_id_canonicalized_to_facts_variant_id() -> None:
    """The LLM often emits a short fuel variant label (e.g. "3B") where Facts
    declared a canonical variant id (e.g. "fuel_3B"); universe qualification
    requires the material source_variant_id to match the universe's
    fuel_variant_id exactly. The normalizer must canonicalize via the Facts
    variant ids."""
    from openmc_agent.plan_builder.state import PlanPatchEnvelope, PlanBuildState

    patch = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel_3b", "name": "Fuel 3B", "role": "fuel",
             "density_g_cm3": 10.257, "source_variant_id": "3B"},
            {"material_id": "coolant", "name": "Water", "role": "coolant",
             "density_g_cm3": 0.743},
        ],
    }
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content={"patch_type": "facts",
                 "fuel_variant_requirements": [{"variant_id": "fuel_3B"}]},
        status="valid",
    ))

    result = normalize_materials_patch_content(patch, state=state)

    fuel = result.content["materials"][0]
    assert fuel["source_variant_id"] == "fuel_3B"
    ops = [op["operation"] for op in result.operations]
    assert "fuel_source_variant_id_canonicalized" in ops


def test_fuel_source_variant_id_token_order_canonicalized_to_facts_variant_id() -> None:
    from openmc_agent.plan_builder.state import PlanPatchEnvelope, PlanBuildState

    patch = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel_3b", "name": "Fuel 3B", "role": "fuel",
             "density_g_cm3": 10.257, "source_variant_id": "fuel_3b"},
        ],
    }
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content={"patch_type": "facts",
                 "fuel_variant_requirements": [{"variant_id": "3B_fuel"}]},
        status="valid",
    ))

    result = normalize_materials_patch_content(patch, state=state)

    assert result.content["materials"][0]["source_variant_id"] == "3B_fuel"
    assert "fuel_source_variant_id_canonicalized" in [
        op["operation"] for op in result.operations
    ]


def test_fuel_source_variant_id_state_label_canonicalized_to_facts_variant_id() -> None:
    from openmc_agent.plan_builder.state import PlanPatchEnvelope, PlanBuildState

    patch = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel_3b", "name": "Fuel 3B", "role": "fuel",
             "density_g_cm3": 10.257, "source_variant_id": "state_3b"},
        ],
    }
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content={"patch_type": "facts",
                 "fuel_variant_requirements": [{"variant_id": "3B_fuel"}]},
        status="valid",
    ))

    result = normalize_materials_patch_content(patch, state=state)

    assert result.content["materials"][0]["source_variant_id"] == "3B_fuel"
    assert "fuel_source_variant_id_canonicalized" in [
        op["operation"] for op in result.operations
    ]


def test_fuel_source_variant_id_ambiguous_not_canonicalized() -> None:
    """When the short label matches multiple Facts variants, do not guess."""
    from openmc_agent.plan_builder.state import PlanPatchEnvelope, PlanBuildState

    patch = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel", "name": "Fuel", "role": "fuel",
             "density_g_cm3": 10.0, "source_variant_id": "3"},
        ],
    }
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content={"patch_type": "facts",
                 "fuel_variant_requirements": [{"variant_id": "fuel_3A"}, {"variant_id": "fuel_3B"}]},
        status="valid",
    ))

    result = normalize_materials_patch_content(patch, state=state)
    assert result.content["materials"][0]["source_variant_id"] == "3"
    assert "fuel_source_variant_id_canonicalized" not in [op["operation"] for op in result.operations]


def test_normalizes_existing_assembled_plan_materials_in_seed_state() -> None:
    state = PlanBuildState(
        state_id="seed",
        requirement_text=(
            "coolant soluble boron 1360 ppm by mass\n"
            "natural boron B-10 isotope fraction 19.9 at%"
        ),
    )
    state.assembled_plan = {
        "complex_model": {
            "materials": [{
                "id": "water",
                "name": "Borated water coolant 1300ppmB",
                "density_value": 0.743,
                "density_unit": "g/cm3",
                "composition_basis": "atom_fraction",
                "composition": [
                    {"name": "B10", "percent": 0.000143732, "percent_type": "ao", "kind": "nuclide"},
                    {"name": "B11", "percent": 0.00057854, "percent_type": "ao", "kind": "nuclide"},
                    {"name": "H1", "percent": 0.666172, "percent_type": "ao", "kind": "nuclide"},
                    {"name": "O16", "percent": 0.333103, "percent_type": "ao", "kind": "nuclide"},
                ],
            }],
        },
    }

    ops = normalize_materials_patches_in_state(state)

    assert ops
    material = state.assembled_plan["complex_model"]["materials"][0]
    assert "warnings" not in material
    assert "source_note" not in material
    comp = {
        item["name"]: item["percent"]
        for item in material["composition"]
    }
    assert _boron_mass_fraction(comp) == pytest.approx(0.00136, rel=5e-4)
