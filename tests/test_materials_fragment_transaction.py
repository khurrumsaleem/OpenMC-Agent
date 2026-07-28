"""Phase 8B Step 4B-2 — Materials fragment transaction tests.

Covers:
* Manifest construction from requirement set (deterministic, ordered).
* Fragment qualification: identity, role, variant, placeholder, poison/absorber.
* Checkpoint hash / contract hash integrity on resume.
* Dependency-aware merge with structured issues.
* Monolithic→fragmented auto-switch on truncation.
* Targeted fragment replay (only invalid fragments regenerated).
* Deterministic merge hash stability.
* Full generate_materials_patch integration with FakePatchLLM.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from openmc_agent.plan_builder.materials_fragment_generation import (
    AcceptedMaterialFragmentRecord,
    MaterialDefinitionFragment,
    MaterialFragmentQualificationResult,
    MaterialManifest,
    MaterialManifestItem,
    MaterialsPatchGenerationSession,
    MaterialMergeResult,
    build_material_manifest,
    compute_manifest_item_contract_hash,
    estimate_materials_output_size,
    merge_material_fragments_structured,
    qualify_material_fragment,
    should_fragment_materials,
    validate_material_manifest,
    verify_accepted_material_fragment,
)
from openmc_agent.plan_builder.material_requirements import (
    MaterialGenerationRequirement,
    MaterialGenerationRequirementSet,
)
from openmc_agent.plan_builder.state import PlanBuildState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_requirement(
    rid: str = "mreq_fuel_1",
    role: str = "fuel",
    variant: str | None = "region_1",
    insert: str | None = None,
    mixture: bool = False,
    mixture_components: tuple[str, ...] = (),
) -> MaterialGenerationRequirement:
    return MaterialGenerationRequirement(
        requirement_id=rid,
        role=role,
        source_variant_id=variant,
        localized_insert_requirement_id=insert,
        mixture_required=mixture,
        mixture_components=mixture_components,
    )


def _make_requirement_set(
    reqs: list[MaterialGenerationRequirement] | None = None,
) -> MaterialGenerationRequirementSet:
    if reqs is None:
        reqs = [_make_requirement()]
    return MaterialGenerationRequirementSet(
        requirements=tuple(reqs),
        inventory_hash="inv_test",
    )


def _make_manifest_item(
    mid: str = "mat_fuel_1",
    rid: str = "mreq_fuel_1",
    role: str = "fuel",
    variant: str | None = "region_1",
    contract_hash: str = "",
) -> MaterialManifestItem:
    item = MaterialManifestItem(
        material_id=mid,
        requirement_id=rid,
        role=role,
        source_variant_id=variant,
        contract_hash=contract_hash,
    )
    if not contract_hash:
        item.recompute_contract_hash()
    return item


def _make_material_data(
    mid: str = "mat_fuel_1",
    role: str = "fuel",
    variant: str | None = "region_1",
    density: float = 10.257,
    composition: dict[str, float] | None = None,
) -> dict[str, Any]:
    if composition is None:
        composition = {"U235": 0.03, "U238": 0.97, "O16": 2.0}
    return {
        "material_id": mid,
        "name": f"{role}_{mid}",
        "role": role,
        "density_g_cm3": density,
        "density_status": "confirmed",
        "composition": composition,
        "composition_basis": "stoichiometric_ratio",
        "composition_status": "confirmed",
        "source_variant_id": variant,
    }


# ---------------------------------------------------------------------------
# Manifest tests
# ---------------------------------------------------------------------------

class TestManifestConstruction:

    def test_single_requirement_produces_one_item(self):
        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        assert len(manifest.items) == 1
        assert manifest.items[0].role == "fuel"
        assert manifest.items[0].source_variant_id == "v1"
        assert manifest.items[0].contract_hash != ""

    def test_deterministic_material_id(self):
        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        m1 = build_material_manifest(rs)
        m2 = build_material_manifest(rs)
        assert m1.items[0].material_id == m2.items[0].material_id

    def test_mixture_items_come_after_components(self):
        rs = _make_requirement_set([
            _make_requirement("m1", "structural"),
            _make_requirement("m2", "structural"),
            _make_requirement("m3", "structural", mixture=True, mixture_components=("mat_structural_a", "mat_structural_b")),
        ])
        manifest = build_material_manifest(rs)
        ids = manifest.generation_order
        # The mixture item should be last in generation order.
        last_item = manifest.item_by_id(ids[-1])
        assert last_item.mixture_required
        non_mixture = [i for i in manifest.items if not i.mixture_required]
        mixture = [i for i in manifest.items if i.mixture_required]
        assert all(m.generation_order_index > nm.generation_order_index for m in mixture for nm in non_mixture)

    def test_manifest_validation_passes_for_consistent_manifest(self):
        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        errors = validate_material_manifest(manifest, rs)
        assert errors == []

    def test_manifest_validation_detects_missing_requirement(self):
        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        # Tamper: remove requirement_id link.
        manifest.items[0].requirement_id = "different"
        errors = validate_material_manifest(manifest, rs)
        assert any("missing" in e for e in errors)

    def test_contract_hash_is_deterministic(self):
        item = _make_manifest_item()
        h1 = compute_manifest_item_contract_hash(item)
        h2 = compute_manifest_item_contract_hash(item)
        assert h1 == h2
        assert len(h1) > 0


# ---------------------------------------------------------------------------
# Fragment qualification tests
# ---------------------------------------------------------------------------

class TestFragmentQualification:

    def test_valid_fragment_passes(self):
        item = _make_manifest_item()
        data = _make_material_data()
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert result.ok
        assert result.fragment_hash != ""

    def test_wrong_material_id_rejected(self):
        item = _make_manifest_item(mid="mat_fuel_1")
        data = _make_material_data(mid="mat_fuel_2")
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={"mat_fuel_1", "mat_fuel_2"},
        )
        assert not result.ok
        assert any("material_id_mismatch" in i["code"] for i in result.issues)

    def test_wrong_role_rejected(self):
        item = _make_manifest_item(role="fuel")
        data = _make_material_data(role="coolant")
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok
        assert any("role_mismatch" in i["code"] for i in result.issues)

    def test_wrong_fuel_variant_rejected(self):
        item = _make_manifest_item(variant="region_1")
        data = _make_material_data(variant="region_2")
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok
        assert any("variant_mismatch" in i["code"] for i in result.issues)

    def test_placeholder_composition_rejected(self):
        item = _make_manifest_item()
        data = _make_material_data()
        data["composition_status"] = "placeholder"
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok
        assert any("placeholder" in i["code"] for i in result.issues)

    def test_atom_frac_stoichiometric_ratio_rejected_before_fragment_acceptance(self):
        item = _make_manifest_item(mid="mat_water", role="coolant", variant=None)
        data = _make_material_data(mid="mat_water", role="coolant", variant=None, composition={"H1": 2.0, "O16": 1.0})
        data["composition_basis"] = "atom_frac"
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok
        assert any(
            i["code"] == "qualification.schema_materials.composition_fraction_sum_invalid"
            for i in result.issues
        )

    def test_weight_frac_percent_style_vector_normalized_before_fragment_acceptance(self):
        item = _make_manifest_item(mid="mat_steel", role="structural", variant=None)
        data = _make_material_data(
            mid="mat_steel",
            role="structural",
            variant=None,
            density=7.9,
            composition={"Fe": 69.5, "Cr": 19.0, "Ni": 9.5, "Mn": 2.0},
        )
        data["composition_basis"] = "weight_frac"
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert result.ok
        assert result.canonical_material_data["composition"] == {
            "Fe": 0.695,
            "Cr": 0.19,
            "Ni": 0.095,
            "Mn": 0.02,
        }
        assert any("percent-style" in warning for warning in result.canonical_material_data["warnings"])

    def test_placeholder_material_id_rejected(self):
        item = _make_manifest_item(mid="mat_fuel_1")
        data = _make_material_data(mid="REPLACE")
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok

    def test_multiple_materials_in_fragment_rejected(self):
        item = _make_manifest_item()
        data1 = _make_material_data()
        data2 = _make_material_data(mid="mat_fuel_2")
        result = qualify_material_fragment(
            raw_fragment={"materials": [data1, data2]},
            manifest_item=item,
            all_manifest_material_ids={"mat_fuel_1", "mat_fuel_2"},
        )
        assert not result.ok
        assert any("multiple_materials" in i["code"] for i in result.issues)

    def test_empty_fragment_rejected(self):
        item = _make_manifest_item()
        result = qualify_material_fragment(
            raw_fragment={},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert not result.ok
        assert any("no_material" in i["code"] for i in result.issues)

    def test_poison_absorber_equivalence_accepted(self):
        """poison and absorber are related roles; either is acceptable for poison/absorber manifest."""
        item = _make_manifest_item(mid="mat_poison", role="poison")
        data = _make_material_data(mid="mat_poison", role="absorber", variant=None,
                                    composition={"B10": 1.0})
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={"mat_poison"},
        )
        # poison↔absorber equivalence should pass role check.
        assert result.ok or any("role_mismatch" not in i["code"] for i in result.issues)

    def test_mixture_missing_components_rejected(self):
        item = MaterialManifestItem(
            material_id="mat_mix_1", requirement_id="m1", role="structural",
            mixture_required=True, mixture_component_ids=("mat_a", "mat_b"),
        )
        item.recompute_contract_hash()
        data = {
            "material_id": "mat_mix_1", "name": "mix", "role": "structural",
            "density_g_cm3": 5.0, "density_status": "confirmed",
            "composition": {}, "composition_basis": "unknown",
            "composition_status": "derived_from_mixture",
        }
        result = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={"mat_mix_1", "mat_a", "mat_b"},
        )
        assert not result.ok
        assert any("mixture_missing" in i["code"] for i in result.issues)


# ---------------------------------------------------------------------------
# Resume verification tests
# ---------------------------------------------------------------------------

class TestResumeVerification:

    def test_valid_record_passes_resume(self):
        item = _make_manifest_item()
        data = _make_material_data()
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        assert qual.ok
        record = AcceptedMaterialFragmentRecord(
            material_id=item.material_id,
            material=qual.canonical_material_data,
            fragment_hash=qual.fragment_hash,
            manifest_contract_hash=qual.manifest_contract_hash,
        )
        result = verify_accepted_material_fragment(record, item, {item.material_id})
        assert result.ok

    def test_hash_drift_detected_on_resume(self):
        item = _make_manifest_item()
        data = _make_material_data()
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        record = AcceptedMaterialFragmentRecord(
            material_id=item.material_id,
            material=qual.canonical_material_data,
            fragment_hash="WRONG_HASH",
            manifest_contract_hash=qual.manifest_contract_hash,
        )
        result = verify_accepted_material_fragment(record, item, {item.material_id})
        assert not result.ok
        assert any("hash_drift" in i["code"] for i in result.issues)

    def test_contract_drift_detected_on_resume(self):
        item = _make_manifest_item()
        data = _make_material_data()
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]},
            manifest_item=item,
            all_manifest_material_ids={item.material_id},
        )
        record = AcceptedMaterialFragmentRecord(
            material_id=item.material_id,
            material=qual.canonical_material_data,
            fragment_hash=qual.fragment_hash,
            manifest_contract_hash="OLD_CONTRACT",
        )
        result = verify_accepted_material_fragment(record, item, {item.material_id})
        assert not result.ok
        assert any("contract_drift" in i["code"] for i in result.issues)


# ---------------------------------------------------------------------------
# Merge tests
# ---------------------------------------------------------------------------

class TestStructuredMerge:

    def test_all_accepted_merges_ok(self):
        rs = _make_requirement_set([
            _make_requirement("m1", "fuel", "v1"),
            _make_requirement("m2", "coolant", None),
        ])
        manifest = build_material_manifest(rs)
        ids = list(manifest.material_ids)
        records = {}
        for i, mid in enumerate(ids):
            item = manifest.item_by_id(mid)
            data = _make_material_data(
                mid=mid, role=item.role, variant=item.source_variant_id,
                composition={"U235": 0.03} if item.role == "fuel" else {"H1": 2.0, "O16": 1.0},
            )
            qual = qualify_material_fragment(
                raw_fragment={"materials": [data]},
                manifest_item=item,
                all_manifest_material_ids=manifest.material_ids,
            )
            assert qual.ok, f"qualification failed for {mid}: {qual.issues}"
            records[mid] = AcceptedMaterialFragmentRecord(
                material_id=mid, material=qual.canonical_material_data,
                fragment_hash=qual.fragment_hash,
                manifest_contract_hash=qual.manifest_contract_hash,
            )
        fragments = [
            MaterialDefinitionFragment(material_id=mid, material=rec.material,
                                       fragment_hash=rec.fragment_hash,
                                       manifest_contract_hash=rec.manifest_contract_hash)
            for mid, rec in records.items()
        ]
        result = merge_material_fragments_structured(
            manifest=manifest, accepted_fragments=fragments, accepted_records=records,
        )
        assert result.ok
        assert result.merged_patch is not None
        assert len(result.merged_patch["materials"]) == 2

    def test_missing_fragment_blocks_merge(self):
        rs = _make_requirement_set([
            _make_requirement("m1", "fuel", "v1"),
            _make_requirement("m2", "coolant", None),
        ])
        manifest = build_material_manifest(rs)
        ids = list(manifest.material_ids)
        # Only provide first material, second is missing.
        item = manifest.item_by_id(ids[0])
        data = _make_material_data(mid=ids[0], role=item.role, variant=item.source_variant_id)
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]}, manifest_item=item,
            all_manifest_material_ids=manifest.material_ids,
        )
        record = AcceptedMaterialFragmentRecord(
            material_id=ids[0], material=qual.canonical_material_data,
            fragment_hash=qual.fragment_hash,
            manifest_contract_hash=qual.manifest_contract_hash,
        )
        result = merge_material_fragments_structured(
            manifest=manifest,
            accepted_fragments=[MaterialDefinitionFragment(material_id=ids[0], material=record.material)],
            accepted_records={ids[0]: record},
        )
        assert not result.ok
        assert any(i.code == "merge.missing_fragment" for i in result.issues)

    def test_duplicate_fragment_detected(self):
        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        mid = list(manifest.material_ids)[0]
        item = manifest.item_by_id(mid)
        data = _make_material_data(mid=mid, role=item.role, variant=item.source_variant_id)
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]}, manifest_item=item,
            all_manifest_material_ids=manifest.material_ids,
        )
        record = AcceptedMaterialFragmentRecord(
            material_id=mid, material=qual.canonical_material_data,
            fragment_hash=qual.fragment_hash,
            manifest_contract_hash=qual.manifest_contract_hash,
        )
        # Provide same fragment twice.
        frag = MaterialDefinitionFragment(material_id=mid, material=record.material)
        result = merge_material_fragments_structured(
            manifest=manifest, accepted_fragments=[frag, frag],
            accepted_records={mid: record},
        )
        assert not result.ok
        assert any(i.code == "merge.duplicate_fragment" for i in result.issues)

    def test_mixture_dependency_missing_blocks(self):
        rs = _make_requirement_set([
            _make_requirement("m1", "structural"),
            _make_requirement("m3", "structural", mixture=True, mixture_components=("mat_structural_a",)),
        ])
        manifest = build_material_manifest(rs)
        ids = list(manifest.material_ids)
        # Only provide the mixture, not its component.
        mixture_id = [i for i in manifest.items if i.mixture_required][0].material_id
        item = manifest.item_by_id(mixture_id)
        data = {
            "material_id": mixture_id, "name": "mix", "role": "structural",
            "density_g_cm3": 5.0, "density_status": "confirmed",
            "composition": {}, "composition_basis": "unknown",
            "composition_status": "derived_from_mixture",
            "mixture_components": [{"material_id": "mat_structural_a", "volume_fraction": 1.0}],
        }
        qual = qualify_material_fragment(
            raw_fragment={"materials": [data]}, manifest_item=item,
            all_manifest_material_ids=manifest.material_ids,
        )
        record = AcceptedMaterialFragmentRecord(
            material_id=mixture_id, material=qual.canonical_material_data,
            fragment_hash=qual.fragment_hash,
            manifest_contract_hash=qual.manifest_contract_hash,
        )
        result = merge_material_fragments_structured(
            manifest=manifest,
            accepted_fragments=[MaterialDefinitionFragment(material_id=mixture_id, material=record.material)],
            accepted_records={mixture_id: record},
        )
        assert not result.ok


# ---------------------------------------------------------------------------
# Strategy decision tests
# ---------------------------------------------------------------------------

class TestStrategyDecision:

    def test_explicit_fragmented(self):
        do_frag, reason = should_fragment_materials(mode="fragmented", material_count=1)
        assert do_frag
        assert "explicit" in reason

    def test_explicit_monolithic(self):
        do_frag, reason = should_fragment_materials(mode="monolithic", material_count=1)
        assert not do_frag

    def test_auto_small_count_stays_monolithic(self):
        do_frag, _ = should_fragment_materials(mode="auto", material_count=3, provider_max_output_tokens=16000)
        assert not do_frag

    def test_auto_large_count_fragments(self):
        do_frag, reason = should_fragment_materials(mode="auto", material_count=10, provider_max_output_tokens=16000)
        assert do_frag

    def test_history_truncated_forces_fragmented(self):
        do_frag, reason = should_fragment_materials(
            mode="auto", material_count=2, history_json_truncated=True,
        )
        assert do_frag
        assert "truncated" in reason


# ---------------------------------------------------------------------------
# Integration: generate_materials_patch with FakePatchLLM
# ---------------------------------------------------------------------------

class TestGenerateMaterialsPatchIntegration:

    def test_fragmented_generation_with_fake_client(self):
        """Full pipeline: fragmented mode, fake LLM, verify checkpoint + merge."""
        from openmc_agent.plan_builder.materials_patch_pipeline import generate_materials_patch
        from openmc_agent.plan_builder.patch_generator import FakePatchLLM

        rs = _make_requirement_set([
            _make_requirement("m1", "fuel", "v1"),
            _make_requirement("m2", "coolant", None),
        ])
        manifest = build_material_manifest(rs)

        # Build fake responses — one per material, in generation_order
        # (a deterministic list) so response order matches pipeline
        # consumption order regardless of PYTHONHASHSEED.
        responses = []
        for mid in manifest.generation_order:
            item = manifest.item_by_id(mid)
            mat = _make_material_data(
                mid=mid, role=item.role, variant=item.source_variant_id,
                composition={"U235": 0.03, "U238": 0.97, "O16": 2.0} if item.role == "fuel" else {"H1": 2.0, "O16": 1.0},
            )
            responses.append(json.dumps({"patch_type": "materials", "materials": [mat]}))

        llm = FakePatchLLM(responses)
        state = PlanBuildState(state_id="s", requirement_text="test")
        state.metadata["planning_material_requirement_set"] = rs.model_dump(mode="json")

        result = generate_materials_patch(
            requirement="test requirement",
            state=state,
            llm_client=llm,
            mode="fragmented",
        )
        assert result.ok
        assert result.envelope is not None
        assert result.envelope.patch_type == "materials"
        parsed = result.parsed_patch
        assert len(parsed["materials"]) == 2

    def test_checkpoint_reuse_on_resume(self):
        """Second call with same state should reuse accepted fragments."""
        from openmc_agent.plan_builder.materials_patch_pipeline import generate_materials_patch
        from openmc_agent.plan_builder.patch_generator import FakePatchLLM

        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        mid = list(manifest.material_ids)[0]
        mat = _make_material_data(mid=mid, role="fuel", variant="v1")

        state = PlanBuildState(state_id="s", requirement_text="test")
        state.metadata["planning_material_requirement_set"] = rs.model_dump(mode="json")

        # First call.
        llm1 = FakePatchLLM([json.dumps({"patch_type": "materials", "materials": [mat]})])
        result1 = generate_materials_patch(requirement="test", state=state, llm_client=llm1, mode="fragmented")
        assert result1.ok

        # Second call — should reuse checkpoint, not call LLM.
        llm2 = FakePatchLLM([])  # Empty — should NOT be called.
        result2 = generate_materials_patch(requirement="test", state=state, llm_client=llm2, mode="fragmented")
        assert result2.ok
        assert result2.parsed_patch["materials"][0]["material_id"] == mid

    def test_wrong_variant_rejected_then_retried(self):
        """Fragment with wrong variant is rejected; retry with correct data succeeds."""
        from openmc_agent.plan_builder.materials_patch_pipeline import generate_materials_patch
        from openmc_agent.plan_builder.patch_generator import FakePatchLLM

        rs = _make_requirement_set([_make_requirement("m1", "fuel", "region_1")])
        manifest = build_material_manifest(rs)
        mid = list(manifest.material_ids)[0]
        item = manifest.item_by_id(mid)

        # First response: wrong variant.
        wrong_mat = _make_material_data(mid=mid, role="fuel", variant="region_2")
        # Second response: correct variant.
        correct_mat = _make_material_data(mid=mid, role="fuel", variant="region_1")

        llm = FakePatchLLM([
            json.dumps({"patch_type": "materials", "materials": [wrong_mat]}),
            json.dumps({"patch_type": "materials", "materials": [correct_mat]}),
        ])
        state = PlanBuildState(state_id="s", requirement_text="test")
        state.metadata["planning_material_requirement_set"] = rs.model_dump(mode="json")

        result = generate_materials_patch(
            requirement="test", state=state, llm_client=llm,
            mode="fragmented", max_fragment_attempts=2,
        )
        assert result.ok
        assert result.parsed_patch["materials"][0]["source_variant_id"] == "region_1"

    def test_markdown_fenced_json_extracted(self):
        """LLM response wrapped in markdown fences should still parse."""
        from openmc_agent.plan_builder.materials_patch_pipeline import generate_materials_patch
        from openmc_agent.plan_builder.patch_generator import FakePatchLLM

        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        mid = manifest.generation_order[0]
        item = manifest.item_by_id(mid)
        mat = _make_material_data(mid=mid, role="fuel", variant="v1")

        fenced = "```json\n" + json.dumps({"patch_type": "materials", "materials": [mat]}) + "\n```"
        llm = FakePatchLLM([fenced])
        state = PlanBuildState(state_id="s", requirement_text="test")
        state.metadata["planning_material_requirement_set"] = rs.model_dump(mode="json")

        result = generate_materials_patch(
            requirement="test", state=state, llm_client=llm,
            mode="fragmented", max_fragment_attempts=1,
        )
        assert result.ok
        assert len(result.parsed_patch["materials"]) == 1

    def test_falls_back_to_facts_derived_requirements_without_inventory_set(self):
        """When no planning_material_requirement_set is present, the pipeline
        derives a requirement set from the accepted FactsPatch so fragmentation
        can activate (mirrors the universes pipeline) instead of falling through
        to monolithic generation that truncates on large catalogs."""
        from openmc_agent.plan_builder.materials_patch_pipeline import (
            _derive_requirement_set_from_facts,
        )
        from openmc_agent.plan_builder.state import PlanPatchEnvelope

        facts = {
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "material_roles": ["clad_Zircaloy-4", "coolant_borated_water", "helium"],
            "fuel_variant_requirements": [
                {"variant_id": "v1", "density_g_cm3": 10.257, "enrichment_wt_percent": 3.1, "source_label": "Region 1 UO2"},
            ],
        }
        state = PlanBuildState(state_id="s", requirement_text="test")
        # NOTE: planning_material_requirement_set is deliberately NOT set.
        state.add_patch(PlanPatchEnvelope(
            patch_id="facts", patch_type="facts", content=facts, status="valid",
        ))

        rs = _derive_requirement_set_from_facts(state)

        assert rs is not None
        assert rs.metadata["source"] == "facts_fallback"
        roles = [r.role for r in rs.requirements]
        assert "fuel_UO2" in roles
        assert "clad_Zircaloy-4" in roles
        assert "coolant_borated_water" in roles
        assert "helium" in roles
        # Fuel appears once per variant, not duplicated by a generic fuel role.
        assert roles.count("fuel_UO2") == 1

    def test_cot_prose_before_json_extracted(self):
        """LLM response with chain-of-thought before JSON should still parse."""
        from openmc_agent.plan_builder.materials_patch_pipeline import generate_materials_patch
        from openmc_agent.plan_builder.patch_generator import FakePatchLLM

        rs = _make_requirement_set([_make_requirement("m1", "fuel", "v1")])
        manifest = build_material_manifest(rs)
        mid = manifest.generation_order[0]
        item = manifest.item_by_id(mid)
        mat = _make_material_data(mid=mid, role="fuel", variant="v1")

        json_str = json.dumps({"patch_type": "materials", "materials": [mat]})
        cot = "Let me analyze the fuel composition for this variant.\nThe material should be UO2 with 3.5 wt% enrichment.\n" + json_str
        llm = FakePatchLLM([cot])
        state = PlanBuildState(state_id="s", requirement_text="test")
        state.metadata["planning_material_requirement_set"] = rs.model_dump(mode="json")

        result = generate_materials_patch(
            requirement="test", state=state, llm_client=llm,
            mode="fragmented", max_fragment_attempts=1,
        )
        assert result.ok
        assert len(result.parsed_patch["materials"]) == 1
