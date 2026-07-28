"""Deterministic feature-to-Facts contract checks, before the LLM critic."""
from __future__ import annotations
from typing import Any
from openmc_agent.schemas import AgentBaseModel
from openmc_agent.plan_builder.control_state_normalization import (
    canonicalize_control_state_id,
)
from openmc_agent.plan_builder.planning_scope import PlanningFeatureContract, ResolvedPlanningScope, resolve_planning_scope

class FactsConsistencyResult(AgentBaseModel):
    issues: list[dict[str, Any]] = []
    scope: ResolvedPlanningScope
    @property
    def ok(self) -> bool: return not any(item.get("severity") == "error" for item in self.issues)

def _issue(code: str, path: str, *, repairable: bool = True, human: bool = False, **metadata: Any) -> dict[str, Any]:
    return {"code": code, "severity": "error", "blocking": True, "path": path, "owner_patch_type": "facts", "repairable_by_llm": repairable, "requires_human": human, **metadata}

def run_facts_consistency_preflight(*, feature_contract: PlanningFeatureContract, facts_patch: dict[str, Any], confirmed_facts: dict[str, Any] | None = None, existing_valid_patch_types: list[str] | None = None) -> FactsConsistencyResult:
    scope = resolve_planning_scope(planning_mode_decision={"feature_summary": feature_contract.evidence.get("feature_summary", {})}, facts_patch=facts_patch, existing_valid_patch_types=existing_valid_patch_types or [], confirmed_facts=confirmed_facts)
    issues: list[dict[str, Any]] = []
    if scope.status == "conflict":
        issues.append(_issue("facts.model_scope_conflicts_with_planning_features", "/model_scope", expected_scope="multi_assembly_core", facts_scope=facts_patch.get("model_scope"), feature_contract_hash=feature_contract.contract_hash))
    if feature_contract.multi_assembly_core and facts_patch.get("model_scope") in {"single_pin", "single_assembly"}:
        if not any(item["code"] == "facts.model_scope_conflicts_with_planning_features" for item in issues):
            issues.append(_issue("facts.model_scope_conflicts_with_planning_features", "/model_scope"))
    if feature_contract.multi_assembly_core and (facts_patch.get("assembly_count") is None or facts_patch.get("core_lattice_size") is None or not facts_patch.get("assembly_type_counts")):
        issues.append(_issue("facts.multi_assembly_contract_incomplete", "/assembly_count"))
    if feature_contract.has_spacer_grid and not facts_patch.get("has_spacer_grids"):
        issues.append(_issue("facts.spacer_grid_contract_missing", "/has_spacer_grids"))
    requirements = facts_patch.get("localized_insert_requirements", []) or []
    if feature_contract.has_localized_insert and not requirements:
        issues.append(_issue("facts.localized_insert_contract_missing", "/localized_insert_requirements"))
    if feature_contract.has_multi_segment_localized_insert and not any(
        isinstance(x, dict) and (x.get("required_profile_id") or x.get("required_segment_roles"))
        for x in requirements
    ):
        issues.append(_issue("facts.localized_insert_profile_contract_missing", "/localized_insert_requirements"))
    if feature_contract.has_control_state and requirements and not any(
        isinstance(x, dict)
        and "control_state_id" in x
        and canonicalize_control_state_id(x.get("control_state_id")) is not None
        for x in requirements
    ):
        issues.append(_issue("facts.control_state_contract_missing", "/localized_insert_requirements"))
    variants = facts_patch.get("fuel_variant_requirements", []) or []
    if feature_contract.has_multiple_fuel_variants and len(variants) < 2:
        issues.append(_issue("facts.fuel_variant_contract_missing", "/fuel_variant_requirements"))
    count, by_type, lattice = facts_patch.get("assembly_count"), facts_patch.get("assembly_type_counts"), facts_patch.get("core_lattice_size")
    if isinstance(count, int) and isinstance(by_type, dict) and by_type and sum(v for v in by_type.values() if isinstance(v, int)) != count:
        issues.append(_issue("facts.assembly_count_inconsistent", "/assembly_type_counts"))
    if isinstance(count, int) and isinstance(lattice, (list, tuple)) and len(lattice) == 2 and all(isinstance(x, int) for x in lattice) and lattice[0] * lattice[1] != count:
        issues.append(_issue("facts.core_lattice_size_inconsistent", "/core_lattice_size"))
    return FactsConsistencyResult(issues=issues, scope=scope)
