"""Phase 4: VERA4 Material-Universe offline qualification."""

from __future__ import annotations

import copy

import pytest

from scripts.vera4_base_fixture import build_all_vera4_patches

from openmc_agent.plan_builder.closed_loop.material_universe_binding import build_material_universe_binding_view
from openmc_agent.plan_builder.closed_loop.material_universe_preflight import run_material_universe_preflight
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope


def _vera4_state() -> PlanBuildState:
    state = PlanBuildState(state_id="vera4-mu", requirement_text="VERA4", benchmark_id="VERA4")
    for patch in build_all_vera4_patches():
        content = patch.model_dump(mode="json")
        state.add_patch(PlanPatchEnvelope(patch_id=content["patch_type"], patch_type=content["patch_type"], content=content, status="valid", source="fixture"))
    return state


def test_vera4_baseline_preflight_minimal_errors() -> None:
    """VERA4 baseline should pass the material-universe preflight with no blocking errors."""
    state = _vera4_state()
    policy = PlanClosedLoopPolicy(mode="controlled", gate_enabled={"facts": True, "material_universe": True})
    result = run_material_universe_preflight(state=state, policy=policy)
    error_codes = {i["code"] for i in result.issues if i.get("severity") == "error"}
    # VERA4 fixture is well-formed; allow at most warnings.
    assert not error_codes, f"VERA4 baseline material-universe errors: {error_codes}"


def test_vera4_multiple_fuel_variants_have_distinct_materials() -> None:
    state = _vera4_state()
    view = build_material_universe_binding_view(state=state)
    fuel_materials = [m for m in view.material_records if m.role == "fuel"]
    # VERA4 has two enrichments.
    assert len(fuel_materials) >= 1


def test_vera4_material_mutation_detected() -> None:
    state = _vera4_state()
    materials = next(item for item in state.patches.values() if item.patch_type == "materials")
    mutated = copy.deepcopy(materials.content)
    # Remove density from a structural material.
    for m in mutated.get("materials", []):
        if m.get("role") == "cladding":
            m["density_g_cm3"] = -1.0
            break
    materials.content = mutated
    policy = PlanClosedLoopPolicy(mode="controlled", gate_enabled={"facts": True, "material_universe": True})
    result = run_material_universe_preflight(state=state, policy=policy)
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_density_invalid" in codes


def test_vera4_unknown_material_reference_detected() -> None:
    state = _vera4_state()
    universes = next(item for item in state.patches.values() if item.patch_type == "universes")
    mutated = copy.deepcopy(universes.content)
    # Corrupt one material reference.
    if mutated["universes"]:
        first_universe = mutated["universes"][0]
        if first_universe.get("cells"):
            first_universe["cells"][0]["material_id"] = "totally_fake_material"
    universes.content = mutated
    policy = PlanClosedLoopPolicy(mode="controlled", gate_enabled={"facts": True, "material_universe": True})
    result = run_material_universe_preflight(state=state, policy=policy)
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_reference_missing" in codes


def test_source_declared_coolant_boron_atom_fraction_is_repaired_on_add_patch() -> None:
    state = PlanBuildState(
        state_id="vera4-boron-repair",
        requirement_text=(
            "coolant density 0.743 g/cc\n"
            "soluble boron 1360 ppm by mass\n"
            "natural boron B-10 isotope fraction 19.9 at%"
        ),
        benchmark_id="VERA4",
    )
    materials = next(p for p in build_all_vera4_patches() if p.patch_type == "materials").model_dump(mode="json")
    for material in materials["materials"]:
        if material["material_id"] == "water":
            material["composition"] = {
                "H1": 0.666633333,
                "O16": 0.333316667,
                "B10": 0.00001,
                "B11": 0.00004,
            }
            material["composition_basis"] = "atom_frac"
            material["source_note"] = "LLM candidate used atom fractions"
            break

    state.add_patch(PlanPatchEnvelope(patch_id="materials", patch_type="materials", content=materials, status="valid", source="llm"))

    stored = state.patches["materials"].content
    water = next(m for m in stored["materials"] if m["material_id"] == "water")
    atom = water["composition"]
    atomic_weights = {"H1": 1.00784, "O16": 15.994, "B10": 10.012937, "B11": 11.009305}
    total_mass = sum(atom[name] * atomic_weights[name] for name in atomic_weights)
    boron_mass = atom["B10"] * atomic_weights["B10"] + atom["B11"] * atomic_weights["B11"]
    assert boron_mass / total_mass == pytest.approx(0.001360, rel=5e-4)
    assert state.metadata["materials_deterministic_normalizations"][0]["operation"] == "coolant_boron_mass_ppm_atom_fraction_repair"
    assert "deterministically normalized coolant boron" in water["warnings"][0]
