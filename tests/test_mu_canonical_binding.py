"""Tests for MU canonical binding fixes (Phase 8C Step 3).

Tests that the v9 false-positive findings are resolved:
1. Universe metadata stamping (geometry_profile_id, source_requirement_ids)
2. Dict-vs-model access in validate_materials_against_requirement_set
3. Localized insert universe matching via metadata.localized_insert_requirement_id
4. UniverseSpecPatch.metadata field acceptance
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmc_agent.plan_builder.patches import (
    FactsPatch,
    FuelVariantRequirementPatchItem,
    LocalizedInsertPlacementRequirementPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
    UniverseSpecPatch,
    UniversesPatch,
    parse_patch_content,
)
from openmc_agent.plan_builder.material_requirements import (
    MaterialGenerationRequirement,
    MaterialGenerationRequirementSet,
    validate_materials_against_requirement_set,
)


# ---------------------------------------------------------------------------
# 1. UniverseSpecPatch accepts metadata
# ---------------------------------------------------------------------------


class TestUniverseMetadataField:
    """UniverseSpecPatch now has an optional metadata field."""

    def test_metadata_accepted(self):
        u = UniverseSpecPatch(
            universe_id="u_test", kind="fuel_pin",
            cells=[{"id": "c1", "role": "fuel", "material_id": "m1",
                     "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
            metadata={"geometry_profile_id": "profile_abc", "component_kind": "fuel_pin"},
        )
        assert u.metadata["geometry_profile_id"] == "profile_abc"

    def test_metadata_defaults_empty(self):
        u = UniverseSpecPatch(
            universe_id="u_test", kind="fuel_pin",
            cells=[{"id": "c1", "role": "fuel", "material_id": "m1",
                     "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
        )
        assert u.metadata == {}

    def test_metadata_in_parsed_patch(self):
        """UniversesPatch.model_validate preserves metadata."""
        content = {
            "patch_type": "universes",
            "universes": [{
                "universe_id": "u_fuel_v1",
                "kind": "fuel_pin",
                "cells": [{"id": "c1", "role": "fuel", "material_id": "m_fuel",
                           "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
                "metadata": {
                    "geometry_profile_id": "profile_xyz",
                    "source_requirement_ids": ["ureq_1"],
                    "localized_insert_requirement_id": "insert_a",
                },
            }],
        }
        patch = parse_patch_content("universes", content)
        assert patch.universes[0].metadata["geometry_profile_id"] == "profile_xyz"
        assert patch.universes[0].metadata["localized_insert_requirement_id"] == "insert_a"


# ---------------------------------------------------------------------------
# 2. Dict-vs-model access in validate_materials_against_requirement_set
# ---------------------------------------------------------------------------


class TestDictModelAccess:
    """validate_materials_against_requirement_set handles dict materials."""

    def _req_set(self, roles: list[str]) -> MaterialGenerationRequirementSet:
        reqs = [
            MaterialGenerationRequirement(
                requirement_id=f"mreq_{role}",
                role=role,
            )
            for role in roles
        ]
        return MaterialGenerationRequirementSet(requirements=reqs)

    def test_dict_materials_covered(self):
        """Materials as dicts (from patch content) are properly checked."""
        materials_patch = {
            "patch_type": "materials",
            "materials": [
                {"material_id": "m_fuel", "role": "fuel", "density_g_cm3": 10.0,
                 "composition": {"U235": 100.0}, "composition_basis": "weight_frac",
                 "composition_status": "approximate"},
                {"material_id": "m_cool", "role": "coolant", "density_g_cm3": 0.7,
                 "composition": {"H": 1.0}, "composition_basis": "weight_frac",
                 "composition_status": "approximate"},
            ],
        }
        req_set = self._req_set(["fuel", "coolant"])
        report = validate_materials_against_requirement_set(
            materials_patch=materials_patch, requirement_set=req_set,
        )
        assert len(report.uncovered_requirement_ids) == 0
        assert len(report.covered_requirement_ids) == 2

    def test_dict_materials_uncovered_role(self):
        """Missing role in dict materials is correctly reported."""
        materials_patch = {
            "patch_type": "materials",
            "materials": [
                {"material_id": "m_fuel", "role": "fuel", "density_g_cm3": 10.0,
                 "composition": {"U235": 100.0}, "composition_basis": "weight_frac",
                 "composition_status": "approximate"},
            ],
        }
        req_set = self._req_set(["fuel", "coolant"])
        report = validate_materials_against_requirement_set(
            materials_patch=materials_patch, requirement_set=req_set,
        )
        assert len(report.uncovered_requirement_ids) == 1
        assert "mreq_coolant" in report.uncovered_requirement_ids

    def test_model_materials_still_work(self):
        """Materials as Pydantic models still work correctly."""
        materials_patch = MaterialsPatch(
            patch_type="materials",
            materials=[
                MaterialSpecPatch(
                    material_id="m_fuel", name="fuel", role="fuel",
                    density_g_cm3=10.0, composition={"U235": 100.0},
                    composition_basis="weight_frac", composition_status="approximate",
                ),
            ],
        )
        req_set = self._req_set(["fuel"])
        report = validate_materials_against_requirement_set(
            materials_patch=materials_patch, requirement_set=req_set,
        )
        assert len(report.uncovered_requirement_ids) == 0


# ---------------------------------------------------------------------------
# 3. Localized insert matching via metadata
# ---------------------------------------------------------------------------


class TestLocalizedInsertProfileFallback:
    """The preflight resolves localized insert universes via metadata."""

    def test_profile_covered_via_metadata(self):
        """A universe with metadata.localized_insert_requirement_id
        covers all expected_insert_universe_ids for that insert."""
        from openmc_agent.plan_builder.closed_loop.material_universe_preflight import (
            _universe_metadata,
        )
        u = UniverseSpecPatch(
            universe_id="profile_abc", kind="pyrex_rod",
            cells=[{"id": "c1", "role": "poison", "material_id": "m_poison",
                     "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
            metadata={"localized_insert_requirement_id": "pyrex_e"},
        )
        assert _universe_metadata(u).get("localized_insert_requirement_id") == "pyrex_e"

    def test_dict_universe_metadata(self):
        """_universe_metadata handles plain dicts."""
        from openmc_agent.plan_builder.closed_loop.material_universe_preflight import (
            _universe_metadata,
        )
        d = {"universe_id": "u1", "metadata": {"geometry_profile_id": "p1"}}
        assert _universe_metadata(d).get("geometry_profile_id") == "p1"

    def test_no_metadata_returns_empty(self):
        from openmc_agent.plan_builder.closed_loop.material_universe_preflight import (
            _universe_metadata,
        )
        u = UniverseSpecPatch(
            universe_id="u1", kind="fuel_pin",
            cells=[{"id": "c1", "role": "fuel", "material_id": "m1",
                     "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
        )
        assert _universe_metadata(u) == {}


# ---------------------------------------------------------------------------
# 4. Fragment pipeline stamps metadata on merged patch
# ---------------------------------------------------------------------------


class TestFragmentPipelineMetadataStamping:
    """The fragmented universe pipeline stamps metadata on generated universes."""

    def test_metadata_stamped_on_merged_universes(self):
        """End-to-end: inventory-driven pipeline stamps geometry_profile_id."""
        from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
        from openmc_agent.plan_builder.universe_patch_pipeline import generate_universes_patch
        from openmc_agent.plan_investigation.inventory_universe_requirements import (
            InventoryUniverseRequirement,
            InventoryUniverseRequirementSet,
        )

        state = PlanBuildState(state_id="test_meta_stamp", requirement_text="test")
        state.add_patch(PlanPatchEnvelope(
            patch_id="facts", patch_type="facts",
            content=FactsPatch(
                patch_type="facts", benchmark_id=None,
                geometry_type="single_assembly", lattice_size=(17, 17),
                pin_pitch_cm=1.26, has_axial_geometry=False,
                active_fuel_region_cm=(0.0, 100.0),
                fuel_variant_requirements=[
                    FuelVariantRequirementPatchItem(
                        variant_id="v1", source_label="v1",
                        enrichment_wt_percent=2.0, density_g_cm3=10.257,
                    )
                ],
            ).model_dump(mode="json"),
            source="fixture", status="valid",
        ))
        state.add_patch(PlanPatchEnvelope(
            patch_id="materials", patch_type="materials",
            content=MaterialsPatch(
                patch_type="materials",
                materials=[
                    MaterialSpecPatch(
                        material_id="m_fuel", name="fuel", role="fuel",
                        density_g_cm3=10.0, composition={"U235": 100.0},
                        composition_basis="weight_frac", composition_status="approximate",
                    ),
                ],
            ).model_dump(mode="json"),
            source="fixture", status="valid",
        ))

        inv_set = InventoryUniverseRequirementSet(
            requirements=(
                InventoryUniverseRequirement(
                    requirement_id="ureq_fuel",
                    geometry_profile_id="profile_fuel_v1",
                    profile_kind="active_fuel_pin",
                    component_kind="fuel_pin",
                    fuel_variant_id="v1",
                    required_cell_roles=("fuel",),
                    required_material_roles=("fuel",),
                    source_claim_ids=("c1",),
                ),
            ),
            inventory_hash="ihash",
            material_requirement_set_hash="mhash",
        )

        class _FakeLLM:
            def __call__(self, prompt: str) -> str:
                return json.dumps({
                    "patch_type": "universes",
                    "universes": [{
                        "universe_id": "u_fuel_v1",
                        "kind": "fuel_pin",
                        "cells": [{"id": "c1", "role": "fuel", "material_id": "m_fuel",
                                   "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
                    }],
                })

        result = generate_universes_patch(
            requirement="test", state=state, llm_client=_FakeLLM(),
            mode="fragmented",
            inventory_universe_requirement_set=inv_set,
        )
        assert result.ok
        assert result.envelope is not None
        universes = result.envelope.content.get("universes", [])
        assert len(universes) == 1
        meta = universes[0].get("metadata", {})
        assert meta.get("geometry_profile_id") == "profile_fuel_v1"
        assert "source_requirement_ids" in meta

    def test_fragmented_universes_materialize_localized_insert_expected_ids(self):
        """Metadata-only localized insert coverage is expanded into exact universe IDs."""
        from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
        from openmc_agent.plan_builder.universe_patch_pipeline import generate_universes_patch
        from openmc_agent.plan_investigation.inventory_universe_requirements import (
            InventoryUniverseRequirement,
            InventoryUniverseRequirementSet,
        )

        state = PlanBuildState(state_id="test_insert_aliases", requirement_text="test")
        state.add_patch(PlanPatchEnvelope(
            patch_id="facts", patch_type="facts",
            content=FactsPatch(
                patch_type="facts", benchmark_id=None,
                geometry_type="single_assembly", lattice_size=(17, 17),
                pin_pitch_cm=1.26, has_axial_geometry=True,
                active_fuel_region_cm=(0.0, 100.0),
                localized_insert_requirements=[
                    LocalizedInsertPlacementRequirementPatchItem(
                        requirement_id="insert_a",
                        insert_kind="pyrex_rod",
                        assembly_type_ids=["A"],
                        expected_coordinate_count_per_assembly=1,
                        host_kind="guide_tube",
                        required_profile_id="insert_profile",
                        required_segment_roles=["poison", "plenum"],
                        expected_insert_universe_ids=["insert_poison", "insert_plenum"],
                        control_state_id="base",
                    )
                ],
            ).model_dump(mode="json"),
            source="fixture", status="valid",
        ))
        state.add_patch(PlanPatchEnvelope(
            patch_id="materials", patch_type="materials",
            content=MaterialsPatch(
                patch_type="materials",
                materials=[
                    MaterialSpecPatch(
                        material_id="m_poison", name="poison", role="poison",
                        density_g_cm3=2.0, composition={"B10": 0.2, "O16": 0.8},
                        composition_basis="atom_frac", composition_status="approximate",
                    ),
                ],
            ).model_dump(mode="json"),
            source="fixture", status="valid",
        ))

        inv_set = InventoryUniverseRequirementSet(
            requirements=(
                InventoryUniverseRequirement(
                    requirement_id="ureq_insert",
                    geometry_profile_id="profile_insert",
                    profile_kind="pyrex_rod",
                    component_kind="pyrex_rod",
                    required_cell_roles=("poison",),
                    required_material_roles=("poison",),
                    localized_insert_requirement_ids=("insert_a",),
                    source_claim_ids=("c_insert",),
                ),
            ),
            inventory_hash="ihash_insert",
            material_requirement_set_hash="mhash_insert",
        )

        class _FakeLLM:
            def __call__(self, prompt: str) -> str:
                return json.dumps({
                    "patch_type": "universes",
                    "universes": [{
                        "universe_id": "profile_insert",
                        "kind": "pyrex_rod",
                        "cells": [{"id": "poison", "role": "poison", "material_id": "m_poison",
                                   "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4}],
                    }],
                })

        result = generate_universes_patch(
            requirement="test", state=state, llm_client=_FakeLLM(),
            mode="fragmented",
            inventory_universe_requirement_set=inv_set,
        )

        assert result.ok
        assert result.envelope is not None
        by_id = {
            universe["universe_id"]: universe
            for universe in result.envelope.content.get("universes", [])
        }
        assert {"profile_insert", "insert_poison", "insert_plenum"} <= set(by_id)
        assert by_id["insert_poison"]["metadata"]["alias_of_universe_id"] == "profile_insert"
        assert by_id["insert_plenum"]["metadata"]["localized_insert_requirement_id"] == "insert_a"
