"""Tests for MaterialGenerationRequirementSet + validation."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.material_requirements import (
    MaterialGenerationRequirement,
    MaterialGenerationRequirementSet,
    MaterialValidationReport,
    extract_material_requirements_from_facts,
    extract_material_requirements_from_inventory,
    validate_materials_against_requirement_set,
)
from openmc_agent.plan_investigation.geometry_inventory import (
    GeometryComponentInventory,
    MaterialRoleRequirement,
    RadialProfileRequirement,
)


def _inventory_with_roles(roles):
    """Build a minimal inventory carrying the supplied material roles."""
    role_reqs = []
    for role in roles:
        role_reqs.append(MaterialRoleRequirement(
            requirement_id=f"mrole_{role}",
            role=role,
        ))
    # Bypass compile_ by constructing directly; we test extraction below.
    return GeometryComponentInventory(
        inventory_id="inv_test",
        requirement_hash="rh",
        source_index_hash="sih",
        ledger_hash="lh",
        material_role_requirements=tuple(role_reqs),
    )


def test_extract_one_requirement_per_distinct_role() -> None:
    inv = _inventory_with_roles(["fuel", "coolant", "structural"])
    req_set = extract_material_requirements_from_inventory(inv)
    assert len(req_set.requirements) == 3
    roles = set(req_set.roles)
    assert roles == {"fuel", "coolant", "structural"}


def test_extract_dedupes_same_role_across_profiles() -> None:
    """Two profiles declaring 'coolant' produce ONE coolant requirement."""
    inv = GeometryComponentInventory(
        inventory_id="inv_test",
        requirement_hash="rh",
        source_index_hash="sih",
        ledger_hash="lh",
        material_role_requirements=(
            MaterialRoleRequirement(requirement_id="m1", role="coolant", required_by_profile_ids=("p1",)),
            MaterialRoleRequirement(requirement_id="m2", role="coolant", required_by_profile_ids=("p2",)),
        ),
    )
    req_set = extract_material_requirements_from_inventory(inv)
    coolant_reqs = [r for r in req_set.requirements if r.role == "coolant"]
    assert len(coolant_reqs) == 1
    # The deduped requirement references both profiles.
    assert set(coolant_reqs[0].required_by_profile_ids) == {"p1", "p2"}


def test_extract_preserves_fuel_variant_binding() -> None:
    """Each fuel variant gets its own requirement."""
    inv = GeometryComponentInventory(
        inventory_id="inv_test",
        requirement_hash="rh",
        source_index_hash="sih",
        ledger_hash="lh",
        material_role_requirements=(
            MaterialRoleRequirement(requirement_id="m1", role="fuel", fuel_variant_id="v1"),
            MaterialRoleRequirement(requirement_id="m2", role="fuel", fuel_variant_id="v2"),
        ),
    )
    req_set = extract_material_requirements_from_inventory(inv)
    fuel_reqs = [r for r in req_set.requirements if r.role == "fuel"]
    assert len(fuel_reqs) == 2
    variant_ids = {r.source_variant_id for r in fuel_reqs}
    assert variant_ids == {"v1", "v2"}


def test_extract_does_not_merge_poison_and_absorber() -> None:
    inv = _inventory_with_roles(["poison", "absorber"])
    req_set = extract_material_requirements_from_inventory(inv)
    roles = set(req_set.roles)
    assert "poison" in roles
    assert "absorber" in roles
    assert len(req_set.requirements) == 2


def test_validate_materials_covers_all_requirements() -> None:
    inv = _inventory_with_roles(["fuel", "coolant"])
    req_set = extract_material_requirements_from_inventory(inv)

    class _Mat:
        def __init__(self, mid, role):
            self.material_id = mid
            self.role = role

    class _Patch:
        materials = [_Mat("m_fuel", "fuel"), _Mat("m_cool", "coolant")]

    report = validate_materials_against_requirement_set(
        materials_patch=_Patch, requirement_set=req_set
    )
    assert report.ok
    assert len(report.covered_requirement_ids) == 2
    assert len(report.uncovered_requirement_ids) == 0


def test_validate_materials_reports_uncovered_role() -> None:
    inv = _inventory_with_roles(["fuel", "coolant", "absorber"])
    req_set = extract_material_requirements_from_inventory(inv)

    class _Mat:
        def __init__(self, mid, role):
            self.material_id = mid
            self.role = role

    class _Patch:
        materials = [_Mat("m_fuel", "fuel"), _Mat("m_cool", "coolant")]  # no absorber

    report = validate_materials_against_requirement_set(
        materials_patch=_Patch, requirement_set=req_set
    )
    assert not report.ok
    assert len(report.uncovered_requirement_ids) == 1
    assert any("absorber" in w for w in report.warnings)


def test_validate_materials_reports_unmatched_material() -> None:
    """A material whose role is not declared by Inventory is flagged."""
    inv = _inventory_with_roles(["fuel"])
    req_set = extract_material_requirements_from_inventory(inv)

    class _Mat:
        def __init__(self, mid, role):
            self.material_id = mid
            self.role = role

    class _Patch:
        materials = [_Mat("m_fuel", "fuel"), _Mat("m_unknown", "custom_role")]

    report = validate_materials_against_requirement_set(
        materials_patch=_Patch, requirement_set=req_set
    )
    # Coverage is OK (fuel covered) but unmatched material reported.
    assert len(report.covered_requirement_ids) == 1
    assert len(report.unmatched_material_ids) == 1
    assert "m_unknown" in report.unmatched_material_ids


def test_requirement_set_hash_deterministic() -> None:
    inv = _inventory_with_roles(["fuel", "coolant"])
    a = extract_material_requirements_from_inventory(inv)
    b = extract_material_requirements_from_inventory(inv)
    assert a.requirement_set_hash == b.requirement_set_hash


def test_requirement_set_hash_changes_with_roles() -> None:
    inv_a = _inventory_with_roles(["fuel"])
    inv_b = _inventory_with_roles(["fuel", "coolant"])
    a = extract_material_requirements_from_inventory(inv_a)
    b = extract_material_requirements_from_inventory(inv_b)
    assert a.requirement_set_hash != b.requirement_set_hash


def test_homogenized_material_without_evidence_is_needs_confirmation() -> None:
    """A homogenized role that came in as 'needs_library' status stays
    as needs_library (no fabricated composition)."""
    inv = GeometryComponentInventory(
        inventory_id="inv_test",
        requirement_hash="rh",
        source_index_hash="sih",
        ledger_hash="lh",
        material_role_requirements=(
            MaterialRoleRequirement(
                requirement_id="m1", role="structural_coolant_homogenized",
                status="needs_library",
            ),
        ),
    )
    req_set = extract_material_requirements_from_inventory(inv)
    assert req_set.requirements[0].resolution_status == "needs_library"
    # Assumptions are NOT allowed for unresolved materials.
    assert req_set.requirements[0].assumptions_allowed is False


# ---------------------------------------------------------------------------
# Facts-fallback extractor
# ---------------------------------------------------------------------------


class _Variant:
    def __init__(self, variant_id, source_label=None):
        self.variant_id = variant_id
        self.source_label = source_label


class _Facts:
    def __init__(self, *, fuel_variants=None, material_roles=None):
        self.fuel_variant_requirements = fuel_variants or []
        self.material_roles = material_roles or []


def test_facts_fallback_one_requirement_per_fuel_variant_and_role() -> None:
    facts = _Facts(
        fuel_variants=[
            _Variant("region1_2.11", "Region 1 UO2"),
            _Variant("region2_2.619", "Region 2 UO2"),
        ],
        material_roles=["Zircaloy-4_cladding", "SS304_structural", "helium"],
    )
    req_set = extract_material_requirements_from_facts(facts)

    roles = [r.role for r in req_set.requirements]
    # Two fuel requirements (one per variant) + three non-fuel roles.
    assert roles.count("fuel_UO2") == 2
    assert "Zircaloy-4_cladding" in roles
    assert "SS304_structural" in roles
    assert "helium" in roles
    fuel = [r for r in req_set.requirements if r.role == "fuel_UO2"]
    assert {r.source_variant_id for r in fuel} == {"region1_2.11", "region2_2.619"}
    assert req_set.metadata["source"] == "facts_fallback"


def test_facts_fallback_skips_fuel_roles_when_variants_declared() -> None:
    """When fuel variants are declared, generic/UO2 material roles that
    represent fuel must not double-count alongside the per-variant fuel
    requirements."""
    facts = _Facts(
        fuel_variants=[_Variant("3B_uo2")],
        material_roles=["UO2_fuel_3B", "fuel_UO2", "Zircaloy-4_cladding", "borated_water"],
    )
    req_set = extract_material_requirements_from_facts(facts)
    roles = [r.role for r in req_set.requirements]
    assert roles.count("fuel_UO2") == 1
    assert "UO2_fuel_3B" not in roles
    assert "Zircaloy-4_cladding" in roles


def test_facts_fallback_keeps_single_fuel_role_when_no_variants() -> None:
    facts = _Facts(material_roles=["fuel_UO2", "Zircaloy-4_cladding"])
    req_set = extract_material_requirements_from_facts(facts)
    roles = [r.role for r in req_set.requirements]
    assert "fuel_UO2" in roles
    assert "Zircaloy-4_cladding" in roles


def test_facts_fallback_empty_when_nothing_declared() -> None:
    req_set = extract_material_requirements_from_facts(_Facts())
    assert req_set.requirements == ()
