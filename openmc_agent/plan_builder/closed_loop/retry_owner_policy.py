"""Deterministic owner/action registry for executable retry requests.

Phase 8B Step 1 unification:

1. **Priority order**:
   a. existing Facts / Placement / Axial / Assembled policy (unchanged)
   b. Inventory canonical owner map (``owner_for_inventory_finding_code``)
   c. Material-Universe canonical issue policy (``material_universe_issue_owner``)
   d. legacy local sets (non-overlapping codes only)
   e. unknown → ``None`` (fail closed)

2. **Special route codes** (``source_claim_missing``, ``hash_mismatch``,
   ``conflict_unresolved``, ``component_unresolved``) return
   :class:`SpecialRetryRoute` instead of :class:`RetryOwnerPolicy` so
   the controller never creates a fake patch retry for them.

3. **No duplicate code sets**.  The canonical registries in
   ``inventory_preflight`` and ``material_universe_issue_policy``
   are the single source of truth.
"""

from __future__ import annotations

from typing import Any

from openmc_agent.schemas import AgentBaseModel

from .models import PlanGateId
from .retry_models import PlanRetryAction, SpecialRetryAction, SpecialRetryRoute


class RetryOwnerPolicy(AgentBaseModel):
    owner_patch_types: list[str]
    preferred_action: PlanRetryAction
    fallback_action: PlanRetryAction = PlanRetryAction.FAIL_CLOSED
    invalidated_dependents: bool = True
    gates_to_invalidate: list[PlanGateId] = []
    required_acceptance_checks: list[str] = []
    requires_human_when_ambiguous: bool = True
    max_attempts: int = 2
    supported_modes: list[str] = ["controlled", "advisory"]
    protected_json_paths: list[str] = ["/patch_type"]


_MATERIAL_UNIVERSE_FACTS_CODES = {
    "material_universe.required_fuel_variant_missing",
}

_FACTS_CODES = {
    "facts.model_scope_conflicts_with_planning_features",
    "facts.multi_assembly_contract_incomplete",
    "facts.localized_insert_contract_missing",
    "facts.localized_insert_profile_contract_missing",
    "facts.control_state_contract_missing",
    "facts.fuel_variant_contract_missing",
    "facts.assembly_count_inconsistent",
    "facts.core_lattice_size_inconsistent",
    "assembly.model_scope_patch_family_conflict",
} | _MATERIAL_UNIVERSE_FACTS_CODES

# Legacy MU codes that overlap with the canonical
# material_universe_issue_policy.  These are preserved here for
# backward compat but the canonical registry takes priority.
_LEGACY_MATERIAL_CODES = {
    "materials.composition_fraction_sum_invalid",
    "qualification.schema_materials.composition_fraction_sum_invalid",
    "materials.execution_density_required",
    "assembly.unresolved_material_reference",
    "materials.compound_in_transport_composition",
    "materials.unsupported_compound_formula",
    "materials.unresolved_species",
    "material_universe.material_duplicate",
    "material_universe.material_density_invalid",
    "material_universe.material_density_missing",
    "material_universe.transport_species_invalid",
    "material_universe.compound_in_transport_composition",
    "material_universe.compound_fraction_basis_missing",
    "material_universe.compound_isotope_unresolved",
    "material_universe.fissile_isotope_policy_missing",
    "material_universe.alloy_reduced_without_disclosure",
    "material_universe.required_material_missing",
    "material_universe.material_requirement_missing",
    "material_universe.required_density_missing",
    "material_universe.required_fuel_variant_material_missing",
    "material_universe.fuel_variant_material_duplicate",
    "material_universe.placeholder_material_unresolved",
    "material_universe.enrichment_contract_mismatch",
}
_LEGACY_UNIVERSE_CODES = {
    "patch_generation.merge_failed",
    "localized_insert.required_universe_missing",
    "patch.pin_map.default_universe_missing",
    "assembly_catalog.universe_missing",
    "assembly.unresolved_universe_reference",
    "profile.segment_universe_missing",
    "required_fuel_universe_missing",
    "material_universe.universe_duplicate",
    "material_universe.universe_empty",
    "material_universe.required_universe_missing",
    "material_universe.cell_duplicate",
    "material_universe.invalid_radial_order",
    "material_universe.radial_gap",
    "material_universe.radial_overlap",
    "material_universe.background_missing",
    "material_universe.localized_insert_universe_missing",
    "material_universe.protected_path_missing",
    "material_universe.material_reference_missing",
    "material_universe.material_role_mismatch",
    "material_universe.multiple_variants_in_one_universe",
    "material_universe.fuel_variant_material_unreachable",
    "material_universe.fuel_variant_collapsed",
    "material_universe.variant_identity_mismatch",
}
_TASK_PLAN_CODES = {
    "planning.required_patch_omitted",
    "planning.mixed_patch_family",
    "planning.stale_canonical_task_plan",
    "planning.task_plan_hash_mismatch",
}
_PLACEMENT_CODES = {
    "localized_insert.required_placement_missing",
    "localized_insert.required_assembly_type_missing",
    "localized_insert.required_profile_missing",
    "localized_insert.required_profile_unused",
    "localized_insert.coordinate_count_mismatch",
    "localized_insert.coordinates_not_in_host_path",
    "localized_insert.coordinate_duplicate",
    "localized_insert.instrument_path_misused",
    "localized_insert.anchor_mismatch",
    "localized_insert.control_state_mismatch",
    "localized_insert.core_multiplicity_mismatch",
    "localized_insert.unexpected_assembly_scope",
}


def _resolve_placement_owner(issue: dict[str, Any], scope: str | None, code: str = "") -> list[str]:
    """Pick exactly one placement owner for a given canonical scope.

    The two assembly-family representations (top-level ``pin_map`` vs
    ``assembly_catalog`` + ``core_layout``) are mutually exclusive.  Returning
    both at once would ask the producer to regenerate two patches that can
    never coexist, so we fail closed when the scope is ambiguous.
    """
    declared = str(issue.get("owner_patch_type") or "")
    if declared in {"localized_insert_profiles", "core_layout"}:
        return [declared]
    effective_code = code or str(issue.get("code") or "")
    if effective_code:
        from .placement_issue_policy import placement_issue_owner
        owner_dict = placement_issue_owner(effective_code, canonical_scope=scope)
        owners = owner_dict.get("owner_patch_types", [])
        if owners and not ({"pin_map", "assembly_catalog"}.issubset(set(owners))):
            return owners
    if declared in {"pin_map", "assembly_catalog"}:
        if scope in {"single_assembly"} and declared == "pin_map":
            return ["pin_map"]
        if scope in {"multi_assembly", "full_core"} and declared == "assembly_catalog":
            return ["assembly_catalog"]
        return [declared]
    if scope in {"single_assembly"}:
        return ["pin_map"]
    if scope in {"multi_assembly", "full_core"}:
        return ["assembly_catalog"]
    return []


# ---------------------------------------------------------------------------
# Inventory special-route codes that MUST NOT enter patch retry
# ---------------------------------------------------------------------------

_INVENTORY_SPECIAL_ROUTES: dict[str, SpecialRetryRoute] = {}


def _ensure_special_routes() -> dict[str, SpecialRetryRoute]:
    if not _INVENTORY_SPECIAL_ROUTES:
        _INVENTORY_SPECIAL_ROUTES.update({
            "inventory.source_claim_missing": SpecialRetryRoute(
                action=SpecialRetryAction.RETRIEVE_EVIDENCE,
                message="source evidence missing; route to evidence retrieval",
            ),
            "inventory.source_span_invalid": SpecialRetryRoute(
                action=SpecialRetryAction.RETRIEVE_EVIDENCE,
                message="source span invalid; route to evidence retrieval",
            ),
            "inventory.conflict_unresolved": SpecialRetryRoute(
                action=SpecialRetryAction.ASK_HUMAN,
                message="unresolved inventory conflict requires human resolution",
                requires_human=True,
            ),
            "inventory.component_unresolved": SpecialRetryRoute(
                action=SpecialRetryAction.ASK_HUMAN,
                message="unresolved component requires human resolution or engineering choice",
                requires_human=True,
            ),
            "inventory.hash_mismatch": SpecialRetryRoute(
                action=SpecialRetryAction.INVENTORY_REBUILD,
                message="inventory hash mismatch; deterministic rebuild required",
            ),
        })
    return _INVENTORY_SPECIAL_ROUTES


# ---------------------------------------------------------------------------
# Adapter: inventory finding owner → RetryOwnerPolicy
# ---------------------------------------------------------------------------


def _inventory_owner_to_retry_policy(code: str, issue: dict[str, Any] | None) -> RetryOwnerPolicy | SpecialRetryRoute | None:
    """Translate an ``owner_for_inventory_finding_code`` result.

    Returns ``RetryOwnerPolicy`` for patch-owned inventory codes,
    ``SpecialRetryRoute`` for non-patch routes, and ``None`` for
    unknown codes.
    """
    from openmc_agent.plan_investigation.inventory_preflight import (
        owner_for_inventory_finding_code,
    )
    owner = owner_for_inventory_finding_code(code)
    if owner is None:
        return None
    if owner == "plan_investigation":
        route = _ensure_special_routes().get(code)
        if route is not None:
            return route
        return SpecialRetryRoute(
            action=SpecialRetryAction.RETRIEVE_EVIDENCE,
            message=f"unhandled plan_investigation route for {code}",
        )
    if owner == "plan_investigation_or_human":
        route = _ensure_special_routes().get(code)
        if route is not None:
            return route
        return SpecialRetryRoute(
            action=SpecialRetryAction.ASK_HUMAN,
            message=f"unhandled plan_investigation_or_human route for {code}",
            requires_human=True,
        )
    if owner == "inventory_rebuild":
        route = _ensure_special_routes().get(code)
        if route is not None:
            return route
        return SpecialRetryRoute(
            action=SpecialRetryAction.INVENTORY_REBUILD,
            message=f"inventory rebuild required for {code}",
        )
    if owner == "materials":
        return RetryOwnerPolicy(
            owner_patch_types=["materials"],
            preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT],
            required_acceptance_checks=[
                "materials_schema", "role_coverage", "fuel_variant_binding",
                "composition", "density", "provenance", "material_universe_preflight",
            ],
            protected_json_paths=["/patch_type", "/materials/*/material_id", "/materials/*/role"],
        )
    if owner == "universes":
        return RetryOwnerPolicy(
            owner_patch_types=["universes"],
            preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE],
            required_acceptance_checks=[
                "universes_schema", "material_references", "profile_binding",
                "source_requirement_binding", "fragment_manifest",
                "inventory_preflight", "material_universe_preflight",
            ],
            protected_json_paths=["/patch_type", "/universes/*/universe_id"],
        )
    # Unknown owner string — return None so the caller fails closed.
    return None


# ---------------------------------------------------------------------------
# Adapter: material_universe issue owner → RetryOwnerPolicy
# ---------------------------------------------------------------------------


def _mu_owner_to_retry_policy(code: str, issue: dict[str, Any] | None) -> RetryOwnerPolicy | None:
    """Translate ``material_universe_issue_owner`` into ``RetryOwnerPolicy``.

    Facts-dependency MU codes route to ``facts`` owner.  Materials-owned
    and universes-owned codes route to the corresponding patch type.
    Unknown codes return ``None``.
    """
    from .material_universe_issue_policy import material_universe_issue_owner
    mu_owner = material_universe_issue_owner(code, issue)
    if not mu_owner:
        return None
    dep = mu_owner.get("dependency_patch_type")
    if dep == "facts":
        return RetryOwnerPolicy(
            owner_patch_types=["facts"],
            preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.FACTS, PlanGateId.PLACEMENT],
            required_acceptance_checks=[
                "facts_schema", "facts_consistency", "resolved_scope",
                "source_critical_feature_coverage", "facts_critic",
                "canonical_task_plan",
            ],
        )
    mu_owners = mu_owner.get("owner_patch_types", [])
    if not mu_owners:
        return None
    owner_type = mu_owners[0]
    if owner_type == "materials":
        return RetryOwnerPolicy(
            owner_patch_types=["materials"],
            preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT],
            required_acceptance_checks=[
                "materials_schema", "material_species", "composition_basis",
                "fuel_variant_identity", "density_policy", "material_readiness",
                "material_universe_preflight",
            ],
            protected_json_paths=["/patch_type", "/materials/*/material_id", "/materials/*/role"],
        )
    if owner_type == "universes":
        return RetryOwnerPolicy(
            owner_patch_types=["universes"],
            preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.PLACEMENT],
            required_acceptance_checks=[
                "universes_schema", "material_references", "required_universe_ids",
                "cell_geometry_local", "through_path", "fuel_variant_reachability",
                "profile_references", "placement_preflight", "material_universe_preflight",
            ],
            protected_json_paths=["/patch_type", "/universes/*/fuel_variant_id"],
        )
    return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def retry_owner_policy(
    code: str,
    issue: dict[str, Any] | None = None,
    *,
    canonical_scope: str | None = None,
) -> RetryOwnerPolicy | SpecialRetryRoute | None:
    """Return the deterministic retry policy for ``code``.

    Priority:
    1. existing Facts / Placement / Axial / Assembled sets (unchanged)
    2. Inventory canonical owner map (``owner_for_inventory_finding_code``)
    3. Material-Universe canonical issue policy (``material_universe_issue_owner``)
    4. legacy local sets
    5. unknown → ``None``
    """
    # 1. Facts
    if code in _FACTS_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["facts"], preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.FACTS, PlanGateId.PLACEMENT],
            required_acceptance_checks=["facts_schema", "facts_consistency", "resolved_scope", "source_critical_feature_coverage", "facts_critic", "canonical_task_plan"],
        )
    # 2. Task plan
    if code in _TASK_PLAN_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["planning_task_plan"], preferred_action=PlanRetryAction.RECOMPUTE_TASK_PLAN,
            gates_to_invalidate=[PlanGateId.PLACEMENT],
            required_acceptance_checks=["canonical_task_plan", "patch_family"],
        )
    # 3. Placement
    if code in _PLACEMENT_CODES:
        owner_types = _resolve_placement_owner(issue or {}, canonical_scope, code=code)
        if not owner_types:
            return None
        return RetryOwnerPolicy(
            owner_patch_types=owner_types, preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.PLACEMENT],
            required_acceptance_checks=["placement_preflight", "placement_critic", "placement_contract_coverage"],
            protected_json_paths=["/patch_type", "/facts", "/materials", "/universes"],
        )
    # 4. Axial
    if code.startswith("patch.axial_") or code.startswith("patch.base_path_axial_profiles"):
        owner = str((issue or {}).get("patch_type") or "axial_overlays")
        if owner in {"axial_layers", "axial_overlays", "base_path_axial_profiles"}:
            return RetryOwnerPolicy(
                owner_patch_types=[owner], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
                gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
                required_acceptance_checks=["patch_schema", "patch_validation"],
            )
    if code.startswith("axial."):
        from .axial_geometry_issue_policy import axial_geometry_issue_owner
        return axial_geometry_issue_owner(code, issue)
    if code.startswith("assembled."):
        from .assembled_plan_issue_policy import assembled_plan_issue_owner
        return assembled_plan_issue_owner(code, issue)

    # 5. Inventory codes (canonical owner map)
    if code.startswith("inventory.") or code.startswith("manifest."):
        result = _inventory_owner_to_retry_policy(code, issue)
        if result is not None:
            return result

    # 6. Material-Universe codes (canonical issue policy)
    if code.startswith("material_universe."):
        result = _mu_owner_to_retry_policy(code, issue)
        if result is not None:
            return result

    # 7. Legacy local sets (non-overlapping codes)
    if code in _LEGACY_MATERIAL_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["materials"], preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT],
            required_acceptance_checks=["materials_schema", "material_species", "composition_basis", "fuel_variant_identity", "density_policy", "material_readiness"],
            protected_json_paths=["/patch_type", "/materials/*/material_id", "/materials/*/role", "/materials/*/fuel_enrichment"],
        )
    if code in _LEGACY_UNIVERSE_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["universes"], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.PLACEMENT],
            required_acceptance_checks=["universes_schema", "material_references", "required_universe_ids", "cell_geometry_local", "through_path", "fuel_variant_reachability", "profile_references", "placement_preflight"],
            protected_json_paths=["/patch_type", "/universes/*/fuel_variant_id"],
        )

    # 8. Unknown code — fail closed
    return None


def registered_retry_issue_codes() -> set[str]:
    """Return all registered issue codes across all registries.

    This set is authoritative for completeness tests.  New codes added
    to any canonical registry will automatically appear here.
    """
    result = (
        _FACTS_CODES
        | _LEGACY_MATERIAL_CODES
        | _LEGACY_UNIVERSE_CODES
        | _TASK_PLAN_CODES
        | _PLACEMENT_CODES
    )
    try:
        from openmc_agent.plan_investigation.inventory_preflight import PREFLIGHT_ISSUE_CODES
        result = result | set(PREFLIGHT_ISSUE_CODES)
    except ImportError:
        pass
    try:
        from .material_universe_issue_policy import registered_material_universe_issue_codes as _mu_codes
        result = result | _mu_codes()
    except ImportError:
        pass
    return result
