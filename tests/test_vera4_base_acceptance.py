"""Tests for VERA4 base acceptance checks (P2-FULLCORE-2D-A).

Uses the deterministic fixture to verify the acceptance module works.
Does NOT run real OpenMC transport (that's the smoke test).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from vera4_base_fixture import build_all_vera4_patches, build_vera4_materials
from openmc_agent.plan_builder.assembler import assemble_simulation_plan_from_patches
from openmc_agent.campaign_eval.vera4_base_acceptance import (
    AcceptanceResult,
    check_plan_level,
    check_geometry_level,
    run_full_acceptance,
)


@pytest.fixture
def assembled_plan():
    patches = build_all_vera4_patches()
    result = assemble_simulation_plan_from_patches(patches, strict=False)
    assert result.ok
    return result.plan


class TestPlanLevelAcceptance:
    def test_borated_water_matches_1360_ppm_mass_fraction(self):
        materials = build_vera4_materials()
        water = next(m for m in materials.materials if m.material_id == "water")
        assert water.density_g_cm3 == pytest.approx(0.743)
        atom = water.composition
        atomic_weights = {"H1": 1.00784, "O16": 15.994, "B10": 10.0129, "B11": 11.0093}
        total_mass = sum(atom[name] * atomic_weights[name] for name in atomic_weights)
        boron_mass = atom["B10"] * atomic_weights["B10"] + atom["B11"] * atomic_weights["B11"]
        assert boron_mass / total_mass == pytest.approx(0.001360, rel=5e-4)

    def test_b4c_density_matches_input_table(self):
        materials = build_vera4_materials()
        b4c = next(m for m in materials.materials if m.material_id == "rcca_b4c_mat")
        assert b4c.density_g_cm3 == pytest.approx(1.76)

    def test_all_key_universes_present(self, assembled_plan):
        checks = check_plan_level(assembled_plan)
        uv_checks = [c for c in checks if c.code.startswith("plan.universe.")]
        assert all(c.passed for c in uv_checks)

    def test_all_key_materials_present(self, assembled_plan):
        checks = check_plan_level(assembled_plan)
        mat_checks = [c for c in checks if c.code.startswith("plan.material.")]
        assert all(c.passed for c in mat_checks)

    def test_axial_layers_have_mixed_fills(self, assembled_plan):
        checks = check_plan_level(assembled_plan)
        material_check = next(c for c in checks if c.code == "plan.axial_layers.has_material_fill")
        lattice_check = next(c for c in checks if c.code == "plan.axial_layers.has_lattice_fill")
        assert material_check.passed
        assert lattice_check.passed

    def test_domain_coverage(self, assembled_plan):
        checks = check_plan_level(assembled_plan)
        min_check = next(c for c in checks if c.code == "plan.axial_layers.domain_min")
        max_check = next(c for c in checks if c.code == "plan.axial_layers.domain_max")
        assert min_check.passed
        assert max_check.passed


class TestGeometryLevelAcceptance:
    def test_no_broken_cell_refs(self, assembled_plan):
        checks = check_geometry_level(assembled_plan)
        cell_check = next(c for c in checks if c.code == "geo.cell_refs_ok")
        assert cell_check.passed

    def test_no_broken_lattice_refs(self, assembled_plan):
        checks = check_geometry_level(assembled_plan)
        lat_check = next(c for c in checks if c.code == "geo.lattice_refs_ok")
        assert lat_check.passed

    def test_mixture_refs_ok(self, assembled_plan):
        checks = check_geometry_level(assembled_plan)
        mix_check = next(c for c in checks if c.code == "geo.mixture_refs_ok")
        assert mix_check.passed

    def test_rcca_reachable(self, assembled_plan):
        checks = check_geometry_level(assembled_plan)
        rcca_check = next(c for c in checks if c.code == "geo.rcca_reachable")
        assert rcca_check.passed

    def test_pyrex_reachable(self, assembled_plan):
        checks = check_geometry_level(assembled_plan)
        pyrex_check = next(c for c in checks if c.code == "geo.pyrex_reachable")
        assert pyrex_check.passed


class TestFullAcceptanceNoRuntime:
    def test_full_acceptance_passes_without_runtime(self, assembled_plan):
        """Acceptance should pass all A-D checks even without runtime data."""
        result = run_full_acceptance(assembled_plan)
        failed = [c for c in result.checks if not c.passed]
        assert len(failed) == 0, f"Failed checks: {[c.code for c in failed]}"

    def test_full_acceptance_with_fake_smoke(self, assembled_plan):
        """Acceptance with fake smoke data should pass E checks."""
        result = run_full_acceptance(
            assembled_plan,
            smoke_result={
                "returncode": 0,
                "keff": 1.27,
                "keff_std": 0.01,
                "lost_particles": 0,
            },
        )
        rt_checks = [c for c in result.checks if c.level == "E"]
        assert all(c.passed for c in rt_checks)
