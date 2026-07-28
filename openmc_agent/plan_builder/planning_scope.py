"""Canonical, reactor-neutral planning scope and task-plan resolution.

This module deliberately owns the one scope decision consumed by task planning
and assembly.  It never guesses a core scope from a 1x1 layout or a single
assembly type.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel
from .closed_loop.fingerprints import canonical_json_dumps
import hashlib

PlanningScopeValue = Literal["single_pin", "single_assembly", "multi_assembly_core", "full_core", "unknown"]


def _hash(value: Any) -> str:
    return hashlib.sha256(canonical_json_dumps(value).encode("utf-8")).hexdigest()


class PlanningScopeEvidence(AgentBaseModel):
    evidence_id: str
    source: Literal["feature_detector", "facts_patch", "task_plan", "existing_patch_family", "human_confirmation"]
    value: PlanningScopeValue
    confidence: Literal["high", "medium", "low"]
    reason: str
    source_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PlanningFeatureContract(AgentBaseModel):
    multi_assembly_core: bool = False
    core_lattice: bool = False
    has_spacer_grid: bool = False
    has_special_pin_map: bool = False
    has_localized_insert: bool = False
    has_multi_segment_localized_insert: bool = False
    has_control_state: bool = False
    has_multiple_fuel_variants: bool = False
    has_axial_geometry: bool = False
    evidence: dict[str, Any] = Field(default_factory=dict)
    confidence_by_feature: dict[str, str] = Field(default_factory=dict)
    contract_hash: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.contract_hash:
            self.contract_hash = _hash(self.model_dump(mode="json", exclude={"contract_hash"}))


class ResolvedPlanningScope(AgentBaseModel):
    value: PlanningScopeValue = "unknown"
    status: Literal["resolved", "conflict", "ambiguous"] = "ambiguous"
    evidence: list[PlanningScopeEvidence] = Field(default_factory=list)
    conflicting_values: list[PlanningScopeValue] = Field(default_factory=list)
    resolution_source: str = ""
    canonical_hash: str = ""
    requires_human_confirmation: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.canonical_hash:
            self.canonical_hash = _hash(self.model_dump(mode="json", exclude={"canonical_hash"}))


class CanonicalPatchTaskPlan(AgentBaseModel):
    planning_scope: ResolvedPlanningScope
    required_patch_types: list[str] = Field(default_factory=list)
    optional_patch_types: list[str] = Field(default_factory=list)
    ordered_patch_types: list[str] = Field(default_factory=list)
    feature_contract_hash: str
    facts_patch_hash: str
    plan_hash: str = ""
    reasons_by_patch_type: dict[str, list[str]] = Field(default_factory=dict)
    excluded_patch_types: list[str] = Field(default_factory=list)
    generated_at_stage: str = "facts_accepted"
    metadata: dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if not self.plan_hash:
            self.plan_hash = _hash(self.model_dump(mode="json", exclude={"plan_hash"}))


class ProfileRequirementDecision(AgentBaseModel):
    status: Literal["required", "not_required", "ambiguous"]
    reasons: list[str] = Field(default_factory=list)
    source_requirement_ids: list[str] = Field(default_factory=list)


def planning_feature_contract(decision: dict[str, Any] | None) -> PlanningFeatureContract:
    summary = (decision or {}).get("feature_summary", {}) if isinstance(decision, dict) else {}
    terms = summary.get("special_pin_terms", []) or []
    text = " ".join(str(x).lower() for x in terms)
    return PlanningFeatureContract(
        multi_assembly_core=bool(summary.get("multi_assembly_core")), core_lattice=bool(summary.get("core_lattice")),
        has_spacer_grid=bool(summary.get("has_spacer_grid")), has_special_pin_map=bool(summary.get("has_special_pin_map")),
        has_localized_insert=bool(summary.get("has_localized_insert")) or any(x in text for x in ("control", "rod", "absorber", "pyrex", "thimble")),
        has_multi_segment_localized_insert=bool(summary.get("has_multi_segment_localized_insert")),
        has_control_state=bool(summary.get("has_control_state")),
        # A benchmark may list multiple operating states (e.g. 3A/3B), while a
        # user run selects exactly one state.  Do not equate "has benchmark
        # variants" with "this selected model needs multiple fuel variants";
        # require the explicit multi-fuel signal instead.
        has_multiple_fuel_variants=bool(summary.get("has_multiple_fuel_variants")),
        has_axial_geometry=bool(summary.get("has_axial_geometry")), evidence={"feature_summary": summary},
        confidence_by_feature={key: "high" for key, value in summary.items() if isinstance(value, bool) and value},
    )


def resolve_planning_scope(*, planning_mode_decision: dict[str, Any] | None, facts_patch: dict[str, Any] | None, existing_valid_patch_types: list[str], confirmed_facts: dict[str, Any] | None) -> ResolvedPlanningScope:
    contract = planning_feature_contract(planning_mode_decision)
    evidence: list[PlanningScopeEvidence] = []
    if contract.multi_assembly_core or contract.core_lattice:
        evidence.append(PlanningScopeEvidence(evidence_id="feature_core", source="feature_detector", value="multi_assembly_core", confidence="high", reason="planning feature contract requires a core-level assembly family"))
    facts = facts_patch or {}
    scope = facts.get("model_scope")
    if scope in {"single_pin", "single_assembly", "multi_assembly_core", "full_core"}:
        evidence.append(PlanningScopeEvidence(evidence_id="facts_scope", source="facts_patch", value=scope, confidence="high", reason="FactsPatch model_scope"))
    if isinstance(facts.get("assembly_count"), int) and facts["assembly_count"] > 1:
        evidence.append(PlanningScopeEvidence(evidence_id="facts_count", source="facts_patch", value="multi_assembly_core", confidence="high", reason="assembly_count > 1"))
    by_type = facts.get("assembly_type_counts")
    if isinstance(by_type, dict) and sum(v for v in by_type.values() if isinstance(v, int)) > 1:
        evidence.append(PlanningScopeEvidence(evidence_id="facts_types", source="facts_patch", value="multi_assembly_core", confidence="high", reason="assembly_type_counts total > 1"))
    families = set(existing_valid_patch_types)
    if {"assembly_catalog", "core_layout"} <= families:
        evidence.append(PlanningScopeEvidence(evidence_id="patch_family_core", source="existing_patch_family", value="multi_assembly_core", confidence="high", reason="assembly_catalog plus core_layout exist"))
    if "pin_map" in families and not ({"assembly_catalog", "core_layout"} <= families):
        evidence.append(PlanningScopeEvidence(evidence_id="patch_family_single", source="existing_patch_family", value="single_assembly", confidence="medium", reason="only top-level pin_map family exists"))
    confirmed = (confirmed_facts or {}).get("planning_scope") if isinstance(confirmed_facts, dict) else None
    if confirmed in {"single_pin", "single_assembly", "multi_assembly_core", "full_core"}:
        evidence.append(PlanningScopeEvidence(evidence_id="human_scope", source="human_confirmation", value=confirmed, confidence="high", reason="confirmed planning scope"))
    high = {item.value for item in evidence if item.confidence == "high"}
    if len(high) > 1:
        return ResolvedPlanningScope(value="unknown", status="conflict", evidence=evidence, conflicting_values=sorted(high), resolution_source="high_confidence_conflict")
    if high:
        value = next(iter(high))
        return ResolvedPlanningScope(value=value, status="resolved", evidence=evidence, resolution_source="high_confidence_evidence")
    return ResolvedPlanningScope(value="unknown", status="ambiguous", evidence=evidence, conflicting_values=sorted({item.value for item in evidence}), resolution_source="insufficient_evidence", requires_human_confirmation=True)


def requires_localized_insert_profiles(*, feature_contract: PlanningFeatureContract, accepted_facts: dict[str, Any]) -> ProfileRequirementDecision:
    requirements = accepted_facts.get("localized_insert_requirements", []) if isinstance(accepted_facts, dict) else []
    ids = [str(item.get("requirement_id")) for item in requirements if isinstance(item, dict) and item.get("requirement_id")]
    if any(isinstance(item, dict) and (item.get("required_profile_id") or item.get("required_segment_roles")) for item in requirements):
        return ProfileRequirementDecision(status="required", reasons=["accepted Facts declares a profile or segment role"], source_requirement_ids=ids)
    if feature_contract.has_multi_segment_localized_insert:
        return ProfileRequirementDecision(status="ambiguous", reasons=["feature contract requires a multi-segment insert but Facts lacks profile details"], source_requirement_ids=ids)
    return ProfileRequirementDecision(status="not_required", reasons=["no source contract requires an axial insert profile"], source_requirement_ids=ids)


def build_canonical_task_plan(*, scope: ResolvedPlanningScope, contract: PlanningFeatureContract, facts_patch: dict[str, Any], feature_order: list[str]) -> CanonicalPatchTaskPlan:
    facts_hash = _hash(facts_patch)
    profile = requires_localized_insert_profiles(feature_contract=contract, accepted_facts=facts_patch)
    required = ["facts", "materials", "universes"]
    reasons: dict[str, list[str]] = {item: ["canonical base dependency"] for item in required}
    if scope.value in {"multi_assembly_core", "full_core"}:
        required += ["assembly_catalog", "core_layout"]
        reasons.update({"assembly_catalog": ["canonical multi-assembly patch family"], "core_layout": ["canonical multi-assembly patch family"]})
    elif scope.value in {"single_assembly", "single_pin"}:
        required.append("pin_map")
        reasons["pin_map"] = ["canonical single-assembly patch family"]
    if profile.status == "required":
        required.append("localized_insert_profiles")
        reasons["localized_insert_profiles"] = [f"required because {', '.join(profile.source_requirement_ids) or 'accepted insert contract'} declares a profile"]
    if contract.has_axial_geometry:
        required.append("axial_layers"); reasons["axial_layers"] = ["feature contract has axial geometry"]
    if contract.has_spacer_grid:
        required.append("axial_overlays"); reasons["axial_overlays"] = ["feature contract has spacer grids"]
    required.append("settings"); reasons["settings"] = ["execution configuration"]
    ordered = [item for item in feature_order if item in required]
    return CanonicalPatchTaskPlan(planning_scope=scope, required_patch_types=required, ordered_patch_types=ordered, feature_contract_hash=contract.contract_hash, facts_patch_hash=facts_hash, reasons_by_patch_type=reasons, excluded_patch_types=[item for item in feature_order if item not in required], metadata={"profile_decision": profile.model_dump(mode="json")})
