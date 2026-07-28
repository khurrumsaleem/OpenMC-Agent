"""Offline campaign-closure regression suite.

Each test loads a sanitized fixture extracted from a real failed LLM run and
replays it through the deterministic production gate that closed the failure
(no LLM calls). Together these tests prove the cumulative fix set closes the
reactive canary chain offline, providing a permanent regression shield that
decouples from the ephemeral ``runs/`` directory.

Closed failures
---------------

* VERA4 v1  - Facts split-review chunk-local unsupported findings -> whole-source reconciliation.
* VERA3B v14 - multi-segment insert profile contract -> required_segment_roles accepted.
* VERA4 v3  - materials JSON dropped object opener -> structural repair.
* VERA4 v5  - materials redundant compound components -> de-dup (no double-count).
* VERA3B v12 - pyrex_rod fragment qualification -> annular-insert oracle.
* VERA3B v13 - pyrex plenum alias -> absorber replaced with gas.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from openmc_agent.plan_builder.closed_loop.annular_insert_universe_oracle import (
    propose_annular_insert_universe,
)
from openmc_agent.plan_builder.closed_loop.facts_consistency import (
    run_facts_consistency_preflight,
)
from openmc_agent.plan_builder.closed_loop.facts_reviewer import (
    reconcile_facts_findings_against_whole_source,
)
from openmc_agent.plan_builder.closed_loop.models import (
    PlanFindingCategory,
    PlanFindingSeverity,
    PlanGateId,
    PlanReviewFinding,
)
from openmc_agent.plan_builder.materials_patch_normalization import (
    normalize_materials_patch_content,
)
from openmc_agent.plan_builder.patch_generator import parse_llm_patch_json
from openmc_agent.plan_builder.patches import (
    MaterialSpecPatch,
    parse_patch_content,
)
from openmc_agent.plan_builder.planning_scope import planning_feature_contract
from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseDefinitionFragment,
    UniverseManifestItem,
)
from openmc_agent.plan_builder.universe_fragment_qualification import (
    qualify_universe_fragment,
)
from openmc_agent.plan_builder.universe_patch_pipeline import (
    materialize_localized_insert_universe_aliases,
)
from openmc_agent.plan_builder.validators import (
    PatchValidationContext,
    validate_patch,
)

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "offline_closure"

FIXTURE_NAMES = [
    "vera4_v1_facts_chunk_reconciliation.json",
    "vera3b_v14_facts_profile_id_preflight.json",
    "vera4_v3_materials_dropped_opener.json",
    "vera4_v5_materials_double_count.json",
    "vera3b_v12_pyrex_oracle.json",
    "vera3b_v13_plenum_alias.json",
]


def _load(name: str) -> dict[str, Any]:
    return json.loads((FIXTURE_DIR / name).read_text())


@pytest.mark.parametrize("name", FIXTURE_NAMES)
def test_fixture_present_and_loadable(name: str) -> None:
    data = _load(name)
    assert data.get("scenario"), f"{name} missing scenario"
    assert data.get("source_run"), f"{name} missing source_run"


# ---------------------------------------------------------------------------
# F1: Facts whole-source reconciliation (VERA4 v1)
# ---------------------------------------------------------------------------


def test_facts_chunk_local_unsupported_findings_reconciled() -> None:
    data = _load("vera4_v1_facts_chunk_reconciliation.json")
    severity_map = {"error": PlanFindingSeverity.ERROR, "warning": PlanFindingSeverity.WARNING}
    category_map = {
        "unsupported_inference": PlanFindingCategory.UNSUPPORTED_INFERENCE,
        "source_coverage": PlanFindingCategory.SOURCE_COVERAGE,
    }
    findings = [
        PlanReviewFinding(
            gate_id=PlanGateId.FACTS,
            code=item["code"],
            severity=severity_map[item["severity"]],
            category=category_map[item["category"]],
            message=item["message"],
            confidence=float(item.get("confidence", 0.9)),
            affected_json_paths=item.get("affected_json_paths", []),
            metadata={"current_value": item.get("current_value")},
        )
        for item in data["findings"]
    ]
    assert any(f.severity is PlanFindingSeverity.ERROR for f in findings)

    reconciled, audit = reconcile_facts_findings_against_whole_source(
        findings, data["requirement_excerpt"]
    )

    # Both chunk-local findings are downgraded because their cited sections
    # (§14 thimble, §15.1 RCCA) and values are present in the whole source.
    assert all(f.severity is PlanFindingSeverity.WARNING for f in reconciled)
    assert len(audit) == len(findings)
    assert {item["finding_code"] for item in audit} == {
        "UNSUPPORTED_RCCA_DATA",
        "THIMBLE_PLUG_GEOMETRY_MISSING",
    }


# ---------------------------------------------------------------------------
# F2: Multi-segment insert profile contract (VERA3B v14)
# ---------------------------------------------------------------------------


def test_segment_roles_satisfy_profile_contract_offline() -> None:
    data = _load("vera3b_v14_facts_profile_id_preflight.json")
    contract = planning_feature_contract(
        {"feature_summary": data["feature_summary"]}
    )
    result = run_facts_consistency_preflight(
        feature_contract=contract, facts_patch=data["facts_patch"]
    )
    codes = {item["code"] for item in result.issues}
    assert "facts.localized_insert_profile_contract_missing" not in codes


# ---------------------------------------------------------------------------
# F3: Materials JSON dropped-object-opener repair (VERA4 v3)
# ---------------------------------------------------------------------------


def test_materials_dropped_opener_repaired_offline() -> None:
    data = _load("vera4_v3_materials_dropped_opener.json")
    obj = parse_llm_patch_json(data["raw_llm_output"], "materials")
    assert obj.get("patch_type") == "materials"
    assert len(obj.get("materials", [])) == data["expected_material_count"]


# ---------------------------------------------------------------------------
# F4: Materials redundant compound double-count de-dup (VERA4 v5)
# ---------------------------------------------------------------------------


def test_materials_redundant_compound_dedup_offline() -> None:
    data = _load("vera4_v5_materials_double_count.json")
    obj = parse_llm_patch_json(data["raw_llm_output"], "materials")
    normalized = normalize_materials_patch_content(obj)
    parsed = parse_patch_content("materials", normalized.content)
    result = validate_patch(parsed, context=PatchValidationContext())

    errors = [i for i in result.issues if i.severity == "error"]
    assert errors == [], [(i.code, i.message) for i in errors]

    aic = next(m for m in parsed.materials if m.material_id == data["expected_aic_material_id"])
    total = sum(v for v in (aic.composition or {}).values() if isinstance(v, (int, float)))
    assert abs(total - data["expected_aic_sum"]) < 1e-9
    # Confirm the redundant-drop operation fired (proves the fix path, not a clean LLM output).
    ops = [op.get("operation") for op in normalized.operations]
    assert "compound_element_component_redundant_dropped" in ops


# ---------------------------------------------------------------------------
# F5: Annular-insert oracle rescues pyrex fragment (VERA3B v12)
# ---------------------------------------------------------------------------


def _manifest_item(spec: dict[str, Any]) -> UniverseManifestItem:
    item = UniverseManifestItem(
        universe_id=spec["universe_id"],
        kind=spec["kind"],
        required_cell_roles=list(spec.get("required_cell_roles", [])),
        required_material_roles=list(spec.get("required_material_roles", [])),
        required_material_ids=[],
        protected_through_path_roles=[],
        source_requirement_ids=[f"req:{spec['universe_id']}"],
    )
    item.recompute_contract_hash()
    return item


def test_pyrex_oracle_constructs_qualifying_universe_offline() -> None:
    data = _load("vera3b_v12_pyrex_oracle.json")
    materials = [
        MaterialSpecPatch(material_id=m["material_id"], name=m["name"], role=m["role"])
        for m in data["materials"]
    ]
    item = _manifest_item(data["manifest_item"])

    proposal = propose_annular_insert_universe(
        manifest_item=item,
        requirement=data["requirement_excerpt"],
        materials=materials,
    )
    assert proposal.ok, proposal.reason

    fragment = UniverseDefinitionFragment(
        universe_id=item.universe_id,
        universe=proposal.universe_data,
        manifest_contract_hash=item.contract_hash,
    )
    qualification = qualify_universe_fragment(
        manifest_item=item,
        fragment=fragment,
        known_material_ids={m.material_id for m in materials},
        material_roles_by_id={m.material_id: m.role for m in materials},
        material_source_variants_by_id={},
    )
    assert qualification.ok, [
        (i.code, i.message) for i in qualification.issues if i.severity == "error"
    ]
    # The poison cell is an annulus with r_min > 0 (the original failure mode).
    poison = [c for c in qualification.canonical_universe_data["cells"] if c["role"] == "poison"]
    assert poison and poison[0]["region_kind"] == "annulus" and poison[0]["r_min_cm"] > 0.0


# ---------------------------------------------------------------------------
# F6: Gas-plenum alias replaces absorber (VERA3B v13)
# ---------------------------------------------------------------------------


def test_pyrex_plenum_alias_replaces_absorber_offline() -> None:
    data = _load("vera3b_v13_plenum_alias.json")
    patch = {"patch_type": "universes", "universes": [data["source_universe"]]}
    facts_obj = SimpleNamespace(
        localized_insert_requirements=[
            SimpleNamespace(
                requirement_id=req_id,
                expected_insert_universe_ids=ids,
            )
            for req_id, ids in data["facts_expected_ids"].items()
        ]
    )

    issues = materialize_localized_insert_universe_aliases(patch, facts_obj=facts_obj)
    assert issues == []

    by_id = {u["universe_id"]: u for u in patch["universes"]}
    assert "pyrex_poison_segment" in by_id
    assert "pyrex_plenum_segment" in by_id

    plenum = by_id["pyrex_plenum_segment"]
    # Plenum must not carry the absorber material; former absorber is now gas.
    assert all(c["material_id"] != "pyrex_glass" for c in plenum["cells"])
    assert any(
        c["role"] == "gas_gap" and c["material_id"] == "helium" and c["r_min_cm"] == 0.241
        for c in plenum["cells"]
    )
    assert plenum["metadata"]["gas_plenum_transform"] is True

    # The assembled patch must validate cleanly (no pyrex_plenum_contains_poison).
    parsed = parse_patch_content("universes", patch)
    result = validate_patch(
        parsed,
        context=PatchValidationContext(
            known_material_ids=data["known_material_ids"],
            known_universe_ids=list(by_id),
        ),
    )
    errors = [i for i in result.issues if i.severity == "error"]
    assert errors == []
