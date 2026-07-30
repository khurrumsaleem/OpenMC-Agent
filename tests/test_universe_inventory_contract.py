"""Tests for the inventory-first universe contract closure (Phase 8C Step 2F-1).

Covers:
1. Material-role preflight: universe requiring a role not in accepted
   materials → deterministic ``unavailable_material_role`` blocker (v6
   reproduction).
2. Inventory-driven requirement conversion: no implicit:* requirements,
   correct kind/cell-role/material-role mapping.
3. Role → material_id binding and radial layer contracts in fragment prompt.
4. Schema-repair prompt on second attempt.
5. Legacy path (no inventory set) still produces implicit:* requirements.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmc_agent.plan_builder.patches import (
    FactsPatch,
    FuelVariantRequirementPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
)
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseGenerationRequirementSet,
    build_manifest_from_requirements,
    convert_inventory_to_generation_requirements,
    extract_universe_requirements,
)
from openmc_agent.plan_builder.universe_patch_pipeline import (
    _build_fragment_prompt,
    _build_role_binding_map,
    _build_schema_repair_prompt,
    _preflight_material_role_coverage,
    generate_universes_patch,
)
from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseManifest,
    UniverseManifestItem,
)
from openmc_agent.plan_investigation.inventory_universe_requirements import (
    InventoryUniverseRequirement,
    InventoryUniverseRequirementSet,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _materials_with(roles: list[str]) -> MaterialsPatch:
    """Build a MaterialsPatch with one material per role."""
    mats = []
    for role in roles:
        mats.append(MaterialSpecPatch(
            material_id=f"m_{role}", name=role, role=role,
            density_g_cm3=5.0,
            composition={"X": 100.0},
            composition_basis="weight_frac",
            composition_status="approximate",
        ))
    return MaterialsPatch(patch_type="materials", materials=mats)


def _facts_with_axial() -> FactsPatch:
    return FactsPatch(
        patch_type="facts",
        benchmark_id=None,
        geometry_type="single_assembly",
        lattice_size=(17, 17),
        pin_pitch_cm=1.26,
        has_axial_geometry=True,
        active_fuel_region_cm=(0.0, 100.0),
        fuel_variant_requirements=[
            FuelVariantRequirementPatchItem(
                variant_id="v1", source_label="v1",
                enrichment_wt_percent=2.0, density_g_cm3=10.257,
            )
        ],
    )


def _state_with_accepted(facts: FactsPatch, materials: MaterialsPatch) -> PlanBuildState:
    state = PlanBuildState(state_id="test_inv_contract", requirement_text="reactor-neutral")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content=facts.model_dump(mode="json"),
        source="fixture", status="valid",
    ))
    state.add_patch(PlanPatchEnvelope(
        patch_id="materials", patch_type="materials",
        content=materials.model_dump(mode="json"),
        source="fixture", status="valid",
    ))
    return state


class _FakeLLM:
    """Minimal fake LLM that records prompts and returns a fixed string."""

    def __init__(self, response: str = '{"patch_type": "universes", "universes": []}'):
        self._response = response
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self._response


def _inventory_set(
    requirements: list[InventoryUniverseRequirement] | None = None,
) -> InventoryUniverseRequirementSet:
    """Build a minimal InventoryUniverseRequirementSet."""
    if requirements is None:
        requirements = [
            InventoryUniverseRequirement(
                requirement_id="ureq_fuel",
                geometry_profile_id="profile_fuel_v1",
                profile_kind="active_fuel_pin",
                component_kind="fuel_pin",
                fuel_variant_id="v1",
                required_cell_roles=("fuel",),
                required_material_roles=("fuel",),
                source_claim_ids=("claim_1",),
            ),
        ]
    return InventoryUniverseRequirementSet(
        requirements=tuple(requirements),
        inventory_hash="inv_hash_001",
        material_requirement_set_hash="mat_hash_001",
    )


# ---------------------------------------------------------------------------
# 1. Material-role preflight (v6 reproduction)
# ---------------------------------------------------------------------------


class TestMaterialRolePreflight:
    """Universe requiring a role not in accepted materials → block before LLM."""

    def test_missing_role_blocks_deterministically(self):
        """v6 reproduction: implicit_gas_gap needs coolant + structural,
        but materials only has fuel → unavailable_material_role."""
        facts = _facts_with_axial()
        materials = _materials_with(["fuel"])  # no coolant, no structural
        state = _state_with_accepted(facts, materials)
        llm = _FakeLLM()

        result = generate_universes_patch(
            requirement="test",
            state=state,
            llm_client=llm,
            mode="fragmented",
        )
        assert not result.ok
        codes = [i["code"] for i in result.issues]
        assert "patch_generation.unavailable_material_role" in codes
        # No LLM call should have been made.
        assert len(llm.prompts) == 0

    def test_all_roles_covered_passes_preflight(self):
        """When all roles are covered, preflight does not block."""
        facts = _facts_with_axial()
        materials = _materials_with(["fuel", "coolant", "structural", "moderator"])
        state = _state_with_accepted(facts, materials)

        # Build the manifest from the legacy path.
        from openmc_agent.plan_builder.patches import parse_patch_content
        facts_obj = parse_patch_content("facts", state.patches["facts"].content)
        materials_obj = parse_patch_content("materials", state.patches["materials"].content)
        req_set = extract_universe_requirements(facts=facts_obj, materials=materials_obj)

        from openmc_agent.plan_builder.universe_fragment_generation import (
            build_manifest_from_requirements,
        )
        manifest = build_manifest_from_requirements(req_set, known_material_ids=set())

        material_roles_by_id = {m.material_id: m.role for m in materials_obj.materials}
        issues = _preflight_material_role_coverage(manifest, material_roles_by_id)
        assert issues == []

    def test_preflight_reports_specific_role_and_universe(self):
        """The issue metadata identifies which universe and role is missing."""
        manifest = UniverseManifest(
            manifest_id="test",
            items=[
                UniverseManifestItem(
                    universe_id="u_gas_gap",
                    kind="custom",
                    required_material_roles=["coolant", "structural"],
                ),
            ],
            generation_order=["u_gas_gap"],
            expected_universe_count=1,
        )
        issues = _preflight_material_role_coverage(manifest, {"m_fuel": "fuel"})
        assert len(issues) == 2
        roles_reported = {i["metadata"]["required_role"] for i in issues}
        assert roles_reported == {"coolant", "structural"}

    def test_preflight_accepts_absorber_material_for_poison_requirement(self):
        manifest = UniverseManifest(
            manifest_id="test",
            items=[
                UniverseManifestItem(
                    universe_id="u_pyrex",
                    kind="pyrex_rod",
                    required_material_roles=["poison"],
                ),
            ],
            generation_order=["u_pyrex"],
            expected_universe_count=1,
        )

        issues = _preflight_material_role_coverage(
            manifest, {"pyrex": "absorber"},
        )

        assert issues == []
        assert all(i["metadata"]["universe_id"] == "u_gas_gap" for i in issues)


# ---------------------------------------------------------------------------
# 2. Inventory-driven requirement conversion
# ---------------------------------------------------------------------------


class TestInventoryConversion:
    """convert_inventory_to_generation_requirements produces correct output."""

    def test_no_implicit_ids(self):
        inv_set = _inventory_set()
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        for req in gen_set.requirements:
            assert not req.requirement_id.startswith("implicit:")
            assert not req.universe_id.startswith("implicit:")

    def test_kind_mapping(self):
        """component_kind → universe kind mapping."""
        inv_set = _inventory_set([
            InventoryUniverseRequirement(
                requirement_id="ureq_gt",
                geometry_profile_id="profile_gt",
                profile_kind="guide_tube",
                component_kind="guide_tube",
                required_cell_roles=("wall", "coolant"),
                required_material_roles=("structural", "coolant"),
                source_claim_ids=("c1",),
            ),
            InventoryUniverseRequirement(
                requirement_id="ureq_endplug",
                geometry_profile_id="profile_ep",
                profile_kind="fuel_rod_end_plug",
                component_kind="end_plug",
                required_cell_roles=("structural",),
                required_material_roles=("structural",),
                source_claim_ids=("c2",),
            ),
        ])
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        kinds = {r.universe_id: r.kind for r in gen_set.requirements}
        assert kinds["u_guide_tube"] == "guide_tube"
        assert kinds["profile_ep"] == "custom"

    def test_accepts_dict_input(self):
        """The adapter accepts dict (as stored in state.metadata)."""
        inv_set = _inventory_set()
        dump = inv_set.model_dump(mode="json")
        gen_set = convert_inventory_to_generation_requirements(dump)
        assert len(gen_set.requirements) == 1
        assert gen_set.requirements[0].universe_id == "u_fuel_v1"

    def test_fuel_variant_universe_id(self):
        """Fuel pin profiles get a clean u_fuel_<variant> id."""
        inv_set = _inventory_set([
            InventoryUniverseRequirement(
                requirement_id="ureq_fuel_v2",
                geometry_profile_id="profile_fuel_v2",
                profile_kind="active_fuel_pin",
                component_kind="fuel_pin",
                fuel_variant_id="v2",
                required_cell_roles=("fuel",),
                required_material_roles=("fuel",),
                source_claim_ids=("c",),
            ),
        ])
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        assert gen_set.requirements[0].universe_id == "u_fuel_v2"

    def test_metadata_has_requirement_source_inventory(self):
        inv_set = _inventory_set()
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        assert gen_set.metadata.get("requirement_source") == "inventory"

    def test_required_layer_roles_preserved_from_inventory(self):
        inv_set = _inventory_set([
            InventoryUniverseRequirement(
                requirement_id="ureq_insert",
                geometry_profile_id="localized_insert_profile",
                profile_kind="pyrex_rod",
                component_kind="pyrex_rod",
                required_cell_roles=("poison",),
                required_material_roles=("poison",),
                required_layer_roles=(
                    "center_gas",
                    "inner_tube",
                    "inner_gap",
                    "poison",
                    "outer_gap",
                    "outer_clad",
                    "outer_coolant",
                ),
                source_claim_ids=("claim_insert",),
            ),
        ])
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        assert gen_set.requirements[0].required_layer_roles == [
            "center_gas",
            "inner_tube",
            "inner_gap",
            "poison",
            "outer_gap",
            "outer_clad",
            "outer_coolant",
        ]

        manifest = build_manifest_from_requirements(gen_set)
        assert manifest.items[0].required_layer_roles == gen_set.requirements[0].required_layer_roles
        assert manifest.items[0].contract_hash

    def test_distinct_input_hash_from_legacy(self):
        """Inventory-driven requirements have a different input hash from legacy."""
        inv_set = _inventory_set()
        gen_set = convert_inventory_to_generation_requirements(inv_set)
        legacy_set = extract_universe_requirements(facts=None, materials=None)
        assert gen_set.input_hash != legacy_set.input_hash


# ---------------------------------------------------------------------------
# 3. Role → material_id binding in prompt
# ---------------------------------------------------------------------------


class TestRoleBindingPrompt:
    """The fragment prompt includes role→material_id mapping."""

    def test_role_binding_map_built_correctly(self):
        item = UniverseManifestItem(
            universe_id="u_fuel_v1", kind="fuel_pin",
            required_material_roles=["fuel"],
        )
        binding = _build_role_binding_map(item, {"m_fuel": "fuel", "m_water": "coolant"})
        assert binding == {"fuel": ["m_fuel"]}

    def test_role_binding_filters_fuel_materials_by_variant(self):
        item = UniverseManifestItem(
            universe_id="u_fuel_v2",
            kind="fuel_pin",
            required_material_roles=["fuel", "coolant"],
            fuel_variant_id="v2",
        )
        binding = _build_role_binding_map(
            item,
            {
                "m_fuel_v1": "fuel",
                "m_fuel_v2": "fuel",
                "m_water": "coolant",
            },
            {
                "m_fuel_v1": "v1",
                "m_fuel_v2": "v2",
                "m_water": None,
            },
        )
        assert binding == {"coolant": ["m_water"], "fuel": ["m_fuel_v2"]}

    def test_fragment_prompt_includes_binding(self):
        item = UniverseManifestItem(
            universe_id="u_fuel_v1", kind="fuel_pin",
            required_material_roles=["fuel"],
        )
        prompt = _build_fragment_prompt(
            item, requirement="test", material_summary="  - m_fuel (role=fuel)",
            role_binding={"fuel": ["m_fuel"]},
        )
        assert "Role → material bindings" in prompt
        assert "fuel → m_fuel" in prompt

    def test_fragment_prompt_includes_required_layer_roles(self):
        item = UniverseManifestItem(
            universe_id="localized_insert_profile",
            kind="pyrex_rod",
            required_material_roles=["poison"],
            required_layer_roles=[
                "center_gas",
                "inner_tube",
                "inner_gap",
                "poison",
                "outer_gap",
                "outer_clad",
                "outer_coolant",
            ],
        )
        prompt = _build_fragment_prompt(
            item,
            requirement="source says annular insert with gaps",
            material_summary="  - m_poison (role=poison)",
            role_binding={"poison": ["m_poison"]},
        )
        assert "Required radial/structural layer roles in source order" in prompt
        assert "center_gas, inner_tube, inner_gap, poison, outer_gap, outer_clad, outer_coolant" in prompt
        assert "Create one cell per required layer role" in prompt

    def test_fragment_prompt_includes_fuel_variant_material_constraint(self):
        item = UniverseManifestItem(
            universe_id="u_fuel_v1",
            kind="fuel_pin",
            required_material_roles=["fuel"],
            fuel_variant_id="v1",
        )
        prompt = _build_fragment_prompt(
            item,
            requirement="test",
            material_summary="  - m_fuel_v1 (role=fuel, source_variant_id=v1)",
            role_binding={"fuel": ["m_fuel_v1"]},
        )
        assert "source_variant_id matches this fuel variant" in prompt
        assert "do not mix fuel materials from multiple variants" in prompt

    def test_fragment_prompt_without_binding(self):
        """When no role_binding is passed, the section is omitted."""
        item = UniverseManifestItem(
            universe_id="u_test", kind="custom",
        )
        prompt = _build_fragment_prompt(
            item, requirement="test", material_summary="  - m_x",
        )
        assert "Role → material bindings" not in prompt


# ---------------------------------------------------------------------------
# 4. Schema-repair prompt on second attempt
# ---------------------------------------------------------------------------


class TestSchemaRepairPrompt:
    """The second attempt uses a focused schema-repair prompt."""

    def test_repair_prompt_shows_errors(self):
        item = UniverseManifestItem(
            universe_id="u_fuel_v1", kind="fuel_pin",
            required_cell_roles=["fuel"],
            required_material_roles=["fuel"],
        )
        prompt = _build_schema_repair_prompt(
            item, requirement="test", material_summary="  - m_fuel",
            role_binding={"fuel": ["m_fuel"]},
            prior_failures=["qualification.role_material_mismatch: role mismatch"],
        )
        assert "SCHEMA REPAIR" in prompt
        assert "ERRORS TO FIX" in prompt
        assert "qualification.role_material_mismatch" in prompt

    def test_repair_prompt_includes_binding(self):
        item = UniverseManifestItem(
            universe_id="u_test", kind="fuel_pin",
            required_material_roles=["fuel"],
        )
        prompt = _build_schema_repair_prompt(
            item, requirement="test", material_summary="  - m_fuel",
            role_binding={"fuel": ["m_fuel"]},
        )
        assert "fuel → m_fuel" in prompt

    def test_repair_prompt_includes_required_layer_roles(self):
        item = UniverseManifestItem(
            universe_id="localized_insert_profile",
            kind="pyrex_rod",
            required_material_roles=["poison"],
            required_layer_roles=["center_gas", "poison", "outer_gap"],
        )
        prompt = _build_schema_repair_prompt(
            item,
            requirement="source says annular insert with gaps",
            material_summary="  - m_poison",
            role_binding={"poison": ["m_poison"]},
            prior_failures=["patch.universes.pyrex_annular_poison_missing"],
        )
        assert "each MUST be represented by a separate cell" in prompt
        assert "Do NOT collapse hollow centers" in prompt

    def test_repair_prompt_includes_json_schema(self):
        item = UniverseManifestItem(
            universe_id="u_test", kind="fuel_pin",
            required_cell_roles=["fuel"],
            required_material_roles=["fuel"],
        )
        prompt = _build_schema_repair_prompt(
            item, requirement="test", material_summary="  - m_fuel",
            role_binding={"fuel": ["m_fuel"]},
        )
        assert "```json" in prompt
        assert "patch_type" in prompt
        assert "u_test" in prompt


# ---------------------------------------------------------------------------
# 5. Inventory-driven pipeline integration
# ---------------------------------------------------------------------------


class TestInventoryDrivenPipeline:
    """When inventory_universe_requirement_set is passed, no implicit:* appears."""

    def test_inventory_set_used_over_legacy(self):
        """Passing inventory_universe_requirement_set skips legacy extraction."""
        facts = _facts_with_axial()  # legacy would emit implicit:gas_gap etc.
        materials = _materials_with(["fuel"])  # minimal
        state = _state_with_accepted(facts, materials)
        llm = _FakeLLM()

        inv_set = _inventory_set()
        result = generate_universes_patch(
            requirement="test",
            state=state,
            llm_client=llm,
            mode="fragmented",
            inventory_universe_requirement_set=inv_set,
        )
        # The inventory set only has a fuel requirement, and we have a fuel material.
        # So preflight should pass and we reach the LLM.
        # (The LLM returns empty universes, so the pipeline will fail at fragment
        # qualification, but the point is: no unavailable_material_role blocker.)
        issue_codes = [i.get("code", "") for i in result.issues]
        assert "patch_generation.unavailable_material_role" not in issue_codes

    def test_inventory_set_with_unsatisfied_role_blocks(self):
        """Inventory requirement needs 'coolant' but materials don't have it."""
        facts = _facts_with_axial()
        materials = _materials_with(["fuel"])
        state = _state_with_accepted(facts, materials)

        inv_set = _inventory_set([
            InventoryUniverseRequirement(
                requirement_id="ureq_coolant",
                geometry_profile_id="profile_water",
                profile_kind="moderator_only",
                component_kind="water_pin",
                required_cell_roles=("coolant",),
                required_material_roles=("coolant",),
                source_claim_ids=("c1",),
            ),
        ])
        result = generate_universes_patch(
            requirement="test",
            state=state,
            llm_client=_FakeLLM(),
            mode="fragmented",
            inventory_universe_requirement_set=inv_set,
        )
        assert not result.ok
        codes = [i["code"] for i in result.issues]
        assert "patch_generation.unavailable_material_role" in codes

    def test_fragment_failure_reports_layer_contract_and_qualification_codes(self):
        """A failed annular insert fragment keeps enough metadata for offline diagnosis."""
        facts = _facts_with_axial()
        materials = _materials_with(["poison"])
        state = _state_with_accepted(facts, materials)

        inv_set = _inventory_set([
            InventoryUniverseRequirement(
                requirement_id="ureq_insert",
                geometry_profile_id="localized_insert_profile",
                profile_kind="pyrex_rod",
                component_kind="pyrex_rod",
                required_material_roles=("poison",),
                required_layer_roles=(
                    "center_gas",
                    "inner_tube",
                    "inner_gap",
                    "poison",
                    "outer_gap",
                    "outer_clad",
                    "outer_coolant",
                ),
                source_claim_ids=("claim_insert",),
            ),
        ])

        class _SolidInsertLLM:
            def __call__(self, prompt: str) -> str:
                return json.dumps({
                    "patch_type": "universes",
                    "universes": [{
                        "universe_id": "localized_insert_profile",
                        "kind": "pyrex_rod",
                        "cells": [{
                            "id": "solid_poison",
                            "role": "pyrex",
                            "material_id": "m_poison",
                            "region_kind": "cylinder",
                            "r_min_cm": 0.0,
                            "r_max_cm": 0.4,
                        }],
                    }],
                })

        result = generate_universes_patch(
            requirement="test",
            state=state,
            llm_client=_SolidInsertLLM(),
            mode="fragmented",
            inventory_universe_requirement_set=inv_set,
        )

        assert not result.ok
        issue = result.issues[0]
        assert issue["code"] == "patch_generation.fragment_failed"
        assert issue["metadata"]["required_layer_roles"] == [
            "center_gas",
            "inner_tube",
            "inner_gap",
            "poison",
            "outer_gap",
            "outer_clad",
            "outer_coolant",
        ]
        qual_codes = {
            q["code"]
            for q in issue["metadata"]["last_qualification_issues"]
        }
        assert "patch.universes.pyrex_annular_poison_missing" in qual_codes


# ---------------------------------------------------------------------------
# 6. Legacy fallback (no inventory set)
# ---------------------------------------------------------------------------


class TestLegacyFallback:
    """When no inventory set is passed, legacy implicit:* requirements appear."""

    def test_legacy_path_still_emits_implicit(self):
        facts = _facts_with_axial()
        materials = _materials_with(["fuel", "coolant", "structural", "moderator"])
        state = _state_with_accepted(facts, materials)

        # Without inventory_universe_requirement_set, the legacy path is used.
        from openmc_agent.plan_builder.patches import parse_patch_content
        facts_obj = parse_patch_content("facts", state.patches["facts"].content)
        materials_obj = parse_patch_content("materials", state.patches["materials"].content)
        req_set = extract_universe_requirements(facts=facts_obj, materials=materials_obj)
        req_ids = [r.requirement_id for r in req_set.requirements]
        implicit_ids = [r for r in req_ids if r.startswith("implicit:")]
        assert len(implicit_ids) > 0  # gas_gap, end_plug, water_pin
