"""Deterministic owner policy for Axial Geometry Gate issue codes.

The Critic never decides owner or action.  This module maps issue codes to
the correct Phase-3B owner for retry routing.
"""

from __future__ import annotations

from typing import Any

from .models import PlanGateId
from .retry_models import PlanRetryAction
from .retry_owner_policy import RetryOwnerPolicy


# Codes whose root cause is an incomplete source axial contract in Facts.
_FACTS_DEPENDENCY_CODES = {
    "axial.domain_missing",
    "axial.domain_invalid",
    "axial.localized_insert_profile_contract_missing",
    "axial.localized_insert_control_state_missing",
}

# Codes whose root cause is a missing material property.
_MATERIALS_DEPENDENCY_CODES = {
    "axial.overlay_material_missing",
    "axial.material_reference_missing",
}

# Codes whose root cause is a missing Universe structure.
_UNIVERSES_DEPENDENCY_CODES = {
    "axial.universe_reference_missing",
    "axial.derived_overlay_universe_incomplete",
}

# Codes whose root cause is in a Placement-owned patch.
_PLACEMENT_DEPENDENCY_CODES = {
    "axial.localized_insert_profile_missing",
    "axial.localized_insert_anchor_missing",
    "axial.localized_insert_extent_outside_domain",
}

# Codes owned by axial_layers patch.
_AXIAL_LAYERS_CODES = {
    "axial.layer_duplicate",
    "axial.layer_interval_invalid",
    "axial.layer_zero_thickness",
    "axial.layer_interval_outside_domain",
    "axial.layer_gap",
    "axial.layer_overlap",
    "axial.layer_default_placeholder",
    "axial.fill_missing",
    "axial.fill_unknown",
    "axial.lattice_reference_missing",
    "axial.loading_reference_missing",
    "axial.loading_unattached",
    "axial.loading_base_lattice_missing",
    "axial.loading_interval_missing",
    "axial.active_fuel_region_not_covered",
    "axial.active_fuel_layer_outside_contract",
    "axial.localized_insert_no_host_layer_overlap",
}

# Codes owned by axial_overlays patch.
_AXIAL_OVERLAYS_CODES = {
    "axial.overlay_duplicate",
    "axial.overlay_interval_invalid",
    "axial.overlay_outside_domain",
    "axial.overlay_overlap_conflict",
    "axial.overlay_target_lattice_missing",
    "axial.overlay_geometry_mode_unsupported",
    "axial.overlay_not_structurally_renderable",
    "axial.overlay_open_region_missing",
    "axial.overlay_through_path_not_preserved",
    "axial.overlay_duplicate_physical_band",
    "axial.overlay_source_count_mismatch",
    "axial.overlay_density_required",
    "axial.through_path_fuel_interrupted",
    "axial.through_path_guide_tube_interrupted",
    "axial.through_path_instrument_tube_interrupted",
    "axial.grid_replaced_entire_lattice",
    "axial.overlay_consumes_protected_cell",
}

# Codes owned by base_path_axial_profiles patch.
_BASE_PATH_CODES = {
    "axial.base_path_profile_missing",
    "axial.base_path_segment_interval_invalid",
    "axial.base_path_segment_overlap",
    "axial.base_path_segment_gap",
    "axial.base_path_segment_order_invalid",
    "axial.base_path_reference_missing",
    "axial.base_path_extent_mismatch",
    "axial.base_path_clipping_unapproved",
}

# Codes owned by the canonical task plan.
_TASK_PLAN_CODES = {
    "axial.required_patch_omitted",
}


def axial_geometry_issue_owner(code: str, issue: dict[str, Any] | None = None) -> RetryOwnerPolicy | None:
    """Return the deterministic owner policy for an axial geometry issue code."""
    if code in _FACTS_DEPENDENCY_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["facts"], preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.FACTS, PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT, PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["facts_schema", "facts_consistency", "canonical_task_plan"],
        )
    if code in _MATERIALS_DEPENDENCY_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["materials"], preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT, PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["materials_schema", "material_species", "density_policy", "material_readiness"],
        )
    if code in _UNIVERSES_DEPENDENCY_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["universes"], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.MATERIAL_UNIVERSE, PlanGateId.PLACEMENT, PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["universes_schema", "material_references", "required_universe_ids"],
        )
    if code in _PLACEMENT_DEPENDENCY_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["localized_insert_profiles"], preferred_action=PlanRetryAction.REVISE_OWNER_PATCH,
            fallback_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.PLACEMENT, PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["placement_preflight", "placement_critic"],
        )
    if code in _AXIAL_LAYERS_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["axial_layers"], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["patch_schema", "patch_validation", "axial_preflight"],
        )
    if code in _AXIAL_OVERLAYS_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["axial_overlays"], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["patch_schema", "patch_validation", "axial_preflight"],
        )
    if code in _BASE_PATH_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["base_path_axial_profiles"], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["patch_schema", "patch_validation", "axial_preflight"],
        )
    if code in _TASK_PLAN_CODES:
        return RetryOwnerPolicy(
            owner_patch_types=["planning_task_plan"], preferred_action=PlanRetryAction.RECOMPUTE_TASK_PLAN,
            gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["canonical_task_plan", "patch_family"],
        )
    # Default: try the declared owner patch type.
    declared = str((issue or {}).get("owner_patch_type") or "")
    if declared in {"axial_layers", "axial_overlays", "base_path_axial_profiles"}:
        return RetryOwnerPolicy(
            owner_patch_types=[declared], preferred_action=PlanRetryAction.REGENERATE_OWNER_PATCH,
            gates_to_invalidate=[PlanGateId.AXIAL_GEOMETRY],
            required_acceptance_checks=["patch_schema", "patch_validation", "axial_preflight"],
        )
    return None


_AXIAL_GEOMETRY_CODES = (
    _FACTS_DEPENDENCY_CODES | _MATERIALS_DEPENDENCY_CODES | _UNIVERSES_DEPENDENCY_CODES
    | _PLACEMENT_DEPENDENCY_CODES | _AXIAL_LAYERS_CODES | _AXIAL_OVERLAYS_CODES
    | _BASE_PATH_CODES | _TASK_PLAN_CODES
)


__all__ = ["axial_geometry_issue_owner", "_AXIAL_GEOMETRY_CODES"]
