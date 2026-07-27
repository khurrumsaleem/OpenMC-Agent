"""Incremental executor + dependency-aware local retry router (Phase 5).

Reads a :class:`PlanBuildState`, generates patches one at a time in dependency
order (facts → materials → universes → pin_map → axial_layers → overlays →
settings → assembly), propagates context between patches, retries failures
locally, and finally assembles a complete SimulationPlan — all without
touching the graph workflow or OpenMC.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .assembler import assemble_simulation_plan_from_patches
from .dependency_graph import DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH
from .patches import (
    AxialLayersPatch,
    AxialOverlaysPatch,
    FactsPatch,
    MaterialsPatch,
    PinMapPatch,
    SettingsPatch,
    UniversesPatch,
    parse_patch_content,
)
from .patch_generator import (
    PatchGenerationContext,
    RetryPatchGenerationContext,
    generate_patch,
)
from .llm_adapter import LARGE_PATCH_MAX_TOKENS
from .closed_loop.fingerprints import compute_candidate_hash
from .validators import validate_patch
from .scoped_counts import resolve_expected_counts_for_pin_map
from .reference_patches import (
    REFERENCE_PATCH_TYPES,
    build_reference_patch,
    load_benchmark_reference,
)
from .state import (
    EVENT_ASSEMBLY_COMPLETED,
    EVENT_ASSEMBLY_FAILED,
    EVENT_ASSEMBLY_STARTED,
    EVENT_PATCH_GENERATED,
    EVENT_PATCH_GENERATION_FAILED,
    PlanBuildState,
    PlanPatchEnvelope,
    EVENT_CLOSED_LOOP_INITIALIZED,
    EVENT_GATE_INITIALIZED,
    EVENT_CLOSED_LOOP_ARTIFACT_WRITTEN,
    EVENT_CLOSED_LOOP_ARTIFACT_WARNING,
    EVENT_GATE_TRANSITIONED,
    add_validated_patch_to_state,
    assemble_state_if_ready,
)


# ---------------------------------------------------------------------------
# New event codes
# ---------------------------------------------------------------------------

EVENT_INCREMENTAL_EXECUTION_STARTED: str = "planning.incremental_execution_started"
EVENT_INCREMENTAL_EXECUTION_COMPLETED: str = "planning.incremental_execution_completed"
EVENT_INCREMENTAL_EXECUTION_FAILED: str = "planning.incremental_execution_failed"
EVENT_PATCH_SKIPPED_ALREADY_VALID: str = "planning.patch_skipped_already_valid"
EVENT_PATCH_DEPENDENCY_CONTEXT_BUILT: str = "planning.patch_dependency_context_built"
EVENT_PATCH_RETRY_ROUTED: str = "planning.patch_retry_routed"
EVENT_DETERMINISTIC_SETTINGS_CREATED: str = "planning.deterministic_settings_patch_created"
EVENT_INCREMENTAL_RESUME_STARTED: str = "planning.incremental_resume_started"
EVENT_INCREMENTAL_RESUME_COMPLETED: str = "planning.incremental_resume_completed"
EVENT_PATCH_SKIPPED_FROM_RESUME: str = "planning.patch_skipped_from_resume_state"
EVENT_REFERENCE_PATCH_LOADED: str = "reference_patch.loaded"
EVENT_REFERENCE_PATCH_GENERATED: str = "reference_patch.generated"
EVENT_REFERENCE_PATCH_FALLBACK: str = "reference_patch.fallback_after_llm_failure"
EVENT_REFERENCE_PATCH_VALIDATION_FAILED: str = "reference_patch.validation_failed"
EVENT_REFERENCE_COUNTS_APPLIED: str = "patch.pin_map.reference_counts_applied"
EVENT_PATCH_PLAN_VALIDATION_REPAIR_STARTED: str = "planning.plan_validation_repair_started"


# Compatibility alias.  New logic must use the typed graph rather than
# maintaining a second hand-written dependency table.
_PATCH_DEPENDENTS: dict[str, tuple[str, ...]] = {
    patch_type: tuple(DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH.dependents_of(patch_type))
    for patch_type in DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH._ORDER
}


def _expand_patch_repair_targets(patch_types: list[str]) -> list[str]:
    """Return patch types plus downstream dependents in canonical order."""
    return DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH.transitive_dependents(patch_types)


# ---------------------------------------------------------------------------
# Result / issue models
# ---------------------------------------------------------------------------


class IncrementalExecutionIssue(AgentBaseModel):
    code: str
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    patch_type: str | None = None
    patch_id: str | None = None
    path: str | None = None


class IncrementalExecutionResult(AgentBaseModel):
    ok: bool = False
    state: PlanBuildState
    assembled_plan: dict[str, Any] | None = None
    issues: list[IncrementalExecutionIssue] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    plan_loop_outcome: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Retry router
# ---------------------------------------------------------------------------


class RetryDecision(AgentBaseModel):
    action: Literal[
        "retry_same_patch",
        "retry_dependency_patch",
        "create_skeleton_patch",
        "fail",
    ]
    patch_type: str
    dependency_patch_type: str | None = None
    reason: str = ""


# Codes that signal an unresolved reference to a *different* patch type.
_REFERENCE_CODES: dict[str, str] = {
    "patch.axial_overlays.target_missing": "pin_map",
    "patch.axial_overlays.material_missing": "materials",
    "patch.pin_map.default_universe_missing": "universes",
    "assembly.unresolved_material_reference": "materials",
    "assembly.unresolved_universe_reference": "universes",
    # P2-FULLCORE-1: scoped count mismatches route to responsible patches.
    "assembly_catalog.universe_missing": "universes",
    "assembly_catalog.local_count_mismatch": "assembly_catalog",
    "assembly_catalog.pin_map_invalid": "assembly_catalog",
    "assembly_catalog.duplicate_type_id": "assembly_catalog",
    "assembly_catalog.empty": "assembly_catalog",
    "core_layout.assembly_type_missing": "assembly_catalog",
    "core_layout.shape_mismatch": "core_layout",
    "core_layout.row_length_mismatch": "core_layout",
    "core_layout.multiplicity_mismatch": "core_layout",
    "core_layout.pitch_invalid": "core_layout",
    "core_layout.boundary_missing": "core_layout",
    "core_layout.pattern_incomplete": "core_layout",
    # P2-FULLCORE-2D-B: fuel variant source contract routing
    "materials.required_fuel_variant_missing": "facts",
    "materials.fuel_variant_source_id_missing": "facts",
    "materials.fuel_variant_duplicate_material": "materials",
    "universes.fuel_material_unreachable": "materials",
    "universes.multiple_fuel_variants_in_universe": "universes",
    "universes.fuel_variant_collapsed": "universes",
    "assembly_catalog.fuel_variant_missing": "facts",
    "assembly_catalog.fuel_variant_source_mismatch": "facts",
    "assembly_catalog.fuel_material_mismatch": "universes",
    "assembly_catalog.distinct_fuel_variants_collapsed": "assembly_catalog",
}

# Codes that signal scoped-count issues.
_SCOPED_COUNT_FACT_CODES: frozenset[str] = frozenset({
    "facts.count_scope_ambiguous",
    "counts.homogeneous_derivation_unproven",
})

_SCOPED_COUNT_CATALOG_CODES: frozenset[str] = frozenset({
    "counts.assembly_type_mismatch",
})

_SCOPED_COUNT_LAYOUT_CODES: frozenset[str] = frozenset({
    "counts.core_total_mismatch",
    "counts.scope_mismatch",
})

_MATERIAL_SPECIES_RETRY_CODES: frozenset[str] = frozenset({
    "materials.compound_in_transport_composition",
    "materials.unsupported_compound_formula",
    "materials.compound_fraction_basis_missing",
    "materials.compound_isotope_policy_missing",
    "materials.fissile_compound_isotope_policy_missing",
    "materials.fissile_compound_would_erase_enrichment",
    "materials.species_name_invalid",
    "materials.unresolved_species",
})


def route_retry(
    *,
    failed_patch_type: str,
    issues: list[dict[str, Any]],
    state: PlanBuildState,
) -> RetryDecision:
    """Decide what to do after a patch generation or validation failure."""
    error_codes = [i.get("code", "") for i in issues if i.get("severity") == "error"]
    if not error_codes:
        # Only warnings/info — treat as retry-same for completeness.
        return RetryDecision(
            action="retry_same_patch",
            patch_type=failed_patch_type,
            reason="non-error issues detected; retrying for completeness",
        )

    if any(code in _MATERIAL_SPECIES_RETRY_CODES for code in error_codes):
        return RetryDecision(
            action="retry_same_patch",
            patch_type="materials",
            reason=f"material species contract error(s): {error_codes[:3]}",
        )

    # Check for unresolved-reference codes that point to a dependency patch.
    for code in error_codes:
        dep = _REFERENCE_CODES.get(code)
        if dep is not None:
            dep_envelope = next(
                (e for e in state.patches.values()
                 if e.patch_type == dep and e.status == "valid"),
                None,
            )
            if dep_envelope is None:
                return RetryDecision(
                    action="retry_dependency_patch",
                    patch_type=failed_patch_type,
                    dependency_patch_type=dep,
                    reason=f"issue {code} references missing dependency patch {dep!r}",
                )
            # Dependency is valid but current patch still can't resolve it —
            # retry current patch with enriched context.
            return RetryDecision(
                action="retry_same_patch",
                patch_type=failed_patch_type,
                reason=f"issue {code}: dependency {dep!r} is valid; retry current patch",
            )

    # JSON parse / schema / local validation errors → retry same patch.
    local_prefixes = (
        "patch_generation.json_parse_error",
        "patch_generation.schema_error",
        "patch.pin_map.",
        "patch.axial_layers.",
        "patch.axial_overlays.",
        "patch.materials.",
        "patch.universes.",
        "patch.assembly_catalog.",
        "patch.core_layout.",
        "patch.schema_invalid",
        "patch.duplicate_id",
        "patch_retry.",
    )
    if any(code.startswith(local_prefixes) for code in error_codes):
        return RetryDecision(
            action="retry_same_patch",
            patch_type=failed_patch_type,
            reason=f"local validation error(s): {error_codes[:3]}",
        )

    # P2-FULLCORE-1: Scoped count mismatches route to responsible patches.
    for code in error_codes:
        if code in _SCOPED_COUNT_FACT_CODES:
            return RetryDecision(
                action="retry_dependency_patch",
                patch_type=failed_patch_type,
                dependency_patch_type="facts",
                reason=f"scoped count issue {code} requires facts retry",
            )
        if code in _SCOPED_COUNT_CATALOG_CODES:
            return RetryDecision(
                action="retry_same_patch",
                patch_type="assembly_catalog",
                reason=f"assembly-type count mismatch: {code}",
            )
        if code in _SCOPED_COUNT_LAYOUT_CODES:
            return RetryDecision(
                action="retry_same_patch",
                patch_type="core_layout",
                reason=f"core-level count mismatch: {code}",
            )

    return RetryDecision(
        action="fail",
        patch_type=failed_patch_type,
        reason=f"unroutable error codes: {error_codes[:3]}",
    )


# ---------------------------------------------------------------------------
# Dependency graph
# ---------------------------------------------------------------------------

_DEFAULT_ORDER: tuple[str, ...] = (
    "facts",
    "materials",
    "universes",
    "localized_insert_profiles",
    "base_path_axial_profiles",
    "pin_map",
    "assembly_catalog",
    "axial_layers",
    "axial_overlays",
    "core_layout",
    "settings",
)

_DEPENDENCIES: dict[str, list[str]] = {
    patch_type: DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH.dependencies_of(patch_type)
    for patch_type in _DEFAULT_ORDER
}


def default_patch_task_order(state: PlanBuildState) -> list[str]:
    """Return the default patch generation order based on state features."""
    if state.canonical_task_plan is not None:
        return list(state.canonical_task_plan.ordered_patch_types)
    order = list(_DEFAULT_ORDER)
    is_multi = _state_is_multi_assembly(state)
    # Remove axial_overlays if spacer grids are not expected.
    has_spacer = _state_has_feature(state, "has_spacer_grid")
    if not has_spacer:
        order = [t for t in order if t != "axial_overlays"]
    # Remove localized_insert_profiles if no multi-segment inserts.
    has_profiles = _state_requires_localized_insert_profiles(state)
    if not has_profiles:
        order = [t for t in order if t != "localized_insert_profiles"]
    # Remove base_path_axial_profiles unless explicitly requested.
    has_base_path = _state_has_feature(state, "has_base_path_profiles")
    if not has_base_path:
        order = [t for t in order if t != "base_path_axial_profiles"]
    if is_multi:
        # Multi-assembly core: use assembly_catalog + core_layout path.
        # Remove top-level pin_map (each assembly type has its own pin map
        # inside the assembly_catalog patch).
        order = [t for t in order if t != "pin_map"]
    else:
        # Single-assembly path: remove assembly_catalog and core_layout.
        order = [t for t in order if t not in ("assembly_catalog", "core_layout")]
        # Remove pin_map if no special pin map.
        has_special = _state_has_feature(state, "has_special_pin_map")
        has_large = state.metadata.get("planning_mode_decision", {}).get(
            "feature_summary", {}
        ).get("large_lattice_dimension") is not None
        if not has_special and not has_large:
            order = [t for t in order if t != "pin_map"]
    return order


def required_patch_types_for_state(state: PlanBuildState) -> list[str]:
    """Return the minimal required patch types for this state.

    For single-assembly models, the path is:
        facts, materials, universes, pin_map, axial_layers, settings,
        (axial_overlays if spacer grids).

    For multi-assembly core models, the path is:
        facts, materials, universes, assembly_catalog, axial_layers,
        (axial_overlays if spacer grids), core_layout, settings.

    The top-level pin_map is NOT required for multi-assembly cores.
    """
    if state.canonical_task_plan is not None:
        return list(state.canonical_task_plan.required_patch_types)
    is_multi = _state_is_multi_assembly(state)
    has_spacer = _state_has_feature(state, "has_spacer_grid")
    has_profiles = _state_requires_localized_insert_profiles(state)
    has_special = _state_has_feature(state, "has_special_pin_map")
    has_large = bool(_state_has_feature(state, "large_lattice_dimension"))
    has_benchmark_variant = _state_has_feature(state, "has_benchmark_variant")

    if is_multi:
        required = [
            "facts", "materials", "universes",
            "assembly_catalog", "axial_layers",
        ]
        if has_spacer:
            required.append("axial_overlays")
        required.append("core_layout")
        required.append("settings")
    else:
        required = ["facts", "materials", "universes", "axial_layers", "settings"]
        if has_spacer:
            required.append("axial_overlays")
        if has_special or has_large or has_benchmark_variant:
            required.append("pin_map")

    if has_profiles:
        required.append("localized_insert_profiles")

    return [t for t in _DEFAULT_ORDER if t in required]


def _order_for_controlled_gate_barriers(
    order: list[str],
    *,
    placement_controlled: bool,
    material_universe_controlled: bool,
    axial_geometry_controlled: bool,
) -> list[str]:
    """Reorder patch generation around controlled review barriers.

    The default historical order is renderer-oriented.  Controlled gates need
    milestone-oriented ordering: generate exactly the upstream patch family
    needed for the next gate, run that gate, then continue.  In particular the
    Placement Gate consumes placement patches (pin map / assembly catalog /
    core layout) and must not wait behind axial geometry patches such as
    ``axial_overlays``.
    """

    if not (placement_controlled or material_universe_controlled or axial_geometry_controlled):
        return order

    if not placement_controlled:
        mu_types = {
            "localized_insert_profiles",
            "base_path_axial_profiles",
            "pin_map",
            "assembly_catalog",
            "axial_layers",
            "axial_overlays",
            "core_layout",
        } if material_universe_controlled else set()
        early = [item for item in order if item in {"facts", "materials", "universes"}]
        gated = [item for item in order if item in mu_types]
        late = [item for item in order if item not in set(early) | set(gated)]
        return early + gated + late

    early_types = {"facts", "materials", "universes"}
    placement_types = {
        "localized_insert_profiles",
        "pin_map",
        "assembly_catalog",
        "core_layout",
    } if placement_controlled else set()
    axial_types = {
        "base_path_axial_profiles",
        "axial_layers",
        "axial_overlays",
    } if axial_geometry_controlled or placement_controlled else set()

    buckets: list[set[str]] = [
        early_types,
        placement_types,
        axial_types,
    ]
    seen: set[str] = set()
    result: list[str] = []
    for bucket in buckets:
        for item in order:
            if item in bucket and item not in seen:
                result.append(item)
                seen.add(item)
    for item in order:
        if item not in seen:
            result.append(item)
            seen.add(item)
    return result


def _state_is_multi_assembly(state: PlanBuildState) -> bool:
    """Check whether the state describes a multi-assembly core model."""
    if state.resolved_planning_scope is not None:
        return state.resolved_planning_scope.value in ("multi_assembly_core", "full_core")
    pmd = state.metadata.get("planning_mode_decision", {})
    fs = pmd.get("feature_summary", {})
    if fs.get("multi_assembly_core") or fs.get("core_lattice"):
        return True
    model_scope = state.extracted_facts.get("model_scope", "")
    if model_scope in ("multi_assembly_core", "full_core"):
        return True
    facts_scope = _accepted_facts_content(state).get("model_scope")
    if facts_scope in ("multi_assembly_core", "full_core"):
        return True
    assembly_count = state.extracted_facts.get("assembly_count")
    if isinstance(assembly_count, int) and assembly_count > 1:
        return True
    facts_count = _accepted_facts_content(state).get("assembly_count")
    if isinstance(facts_count, int) and facts_count > 1:
        return True
    return False


def _state_has_feature(state: PlanBuildState, feature: str) -> bool:
    pmd = state.metadata.get("planning_mode_decision", {})
    fs = pmd.get("feature_summary", {})
    if fs.get(feature):
        return True
    # Also check extracted_facts.
    return bool(state.extracted_facts.get(feature))


def _accepted_facts_content(state: PlanBuildState) -> dict[str, Any]:
    facts = [
        envelope.content
        for envelope in state.patches.values()
        if envelope.patch_type == "facts" and envelope.status == "valid"
    ]
    return facts[-1] if facts else {}


def _state_requires_localized_insert_profiles(state: PlanBuildState) -> bool:
    """Return whether accepted Facts require localized insert profiles."""

    if _state_has_feature(state, "has_localized_insert_profiles"):
        return True
    facts = _accepted_facts_content(state)
    requirements = facts.get("localized_insert_requirements", [])
    if not isinstance(requirements, list):
        return False
    return any(
        isinstance(item, dict)
        and (item.get("required_profile_id") or item.get("required_segment_roles"))
        for item in requirements
    )


def _sync_controlled_order_after_facts(
    *,
    state: PlanBuildState,
    order: list[str],
    required: list[str],
    placement_controlled: bool,
    material_universe_controlled: bool,
    axial_geometry_controlled: bool,
) -> None:
    """Refresh controlled execution order once accepted Facts are available."""

    refreshed_required = required_patch_types_for_state(state)
    for patch_type in refreshed_required:
        if patch_type not in order:
            order.append(patch_type)
    selected = set(order)
    graph_ordered = [
        patch_type
        for patch_type in DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH._ORDER
        if patch_type in selected
    ]
    graph_ordered.extend(patch_type for patch_type in order if patch_type not in graph_ordered)
    order[:] = _order_for_controlled_gate_barriers(
        graph_ordered,
        placement_controlled=placement_controlled,
        material_universe_controlled=material_universe_controlled,
        axial_geometry_controlled=axial_geometry_controlled,
    )
    required[:] = refreshed_required


def _retry_outcome_requires_downstream_resume(outcome: Any) -> bool:
    """Return whether a committed retry owner requires downstream rebuild.

    ``execute_plan_retry_loop`` can return ``resolved`` when the owner commit
    already removes the selected issue from its local issue set.  That still
    changes workflow state and invalidates dependent patches, so the executor
    must resume downstream just like the explicit ``resumed`` status.
    """

    status = getattr(getattr(outcome, "status", None), "value", getattr(outcome, "status", ""))
    return bool(getattr(outcome, "workflow_behavior_changed", False)) and status in {
        "resumed",
        "resolved",
        "partially_resolved",
    }


def _placement_dependency_required_ids(
    issues: list[dict[str, Any]],
    dependency_code: str,
    *,
    state: PlanBuildState | None = None,
) -> list[str]:
    """Collect concrete IDs required by a Placement dependency issue set.

    Placement preflight can report the same root cause once per contract row.
    The retry request must carry every missing ID; otherwise the upstream
    producer is asked to regenerate a dependency without a target and can fail
    without making progress.
    """
    requirement_ids: list[str] = []
    for item in issues:
        if item.get("code") != dependency_code:
            continue
        req_id = item.get("requirement_id")
        if isinstance(req_id, str) and req_id:
            requirement_ids.append(req_id)

    required: list[str] = []
    for item in issues:
        if item.get("code") != dependency_code:
            continue
        item_required: list[str] = []
        for key in ("expected", "missing_ids", "required_ids"):
            value = item.get(key)
            values = value if isinstance(value, list) else [value]
            for candidate in values:
                if isinstance(candidate, str) and candidate:
                    item_required.append(candidate)
        # Backward-compatible fallback for old artifacts where the validator
        # embedded the universe ID only in the message text.
        if not item_required:
            message = str(item.get("message") or "")
            marker = "Required universe '"
            if marker in message:
                tail = message.split(marker, 1)[1]
                item_required.append(tail.split("'", 1)[0])
        required.extend(item_required)

    # Placement preflight may report a localized insert requirement as soon as
    # the first missing universe is detected.  For an upstream Universes retry,
    # the owner needs the whole source-backed contract row, not just the first
    # missing segment.  Enrich from accepted Facts using the requirement IDs
    # carried by the deterministic preflight issues.
    if state is not None and requirement_ids:
        facts_env = next(
            (
                item for item in state.patches.values()
                if item.patch_type == "facts" and item.status == "valid"
            ),
            None,
        )
        rows = (
            facts_env.content.get("localized_insert_requirements", [])
            if facts_env is not None and isinstance(facts_env.content, dict)
            else []
        )
        wanted = set(requirement_ids)
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict):
                    continue
                if row.get("requirement_id") not in wanted:
                    continue
                values = row.get("expected_insert_universe_ids", [])
                values = values if isinstance(values, list) else [values]
                for candidate in values:
                    if isinstance(candidate, str) and candidate:
                        required.append(candidate)
    return list(dict.fromkeys(required))


# ---------------------------------------------------------------------------
# Context propagation
# ---------------------------------------------------------------------------


def build_generation_context_from_state(
    state: PlanBuildState,
    patch_type: str,
    *,
    few_shot_case_ids: list[str] | None = None,
) -> PatchGenerationContext:
    """Build a :class:`PatchGenerationContext` enriched from all valid patches."""
    ctx = PatchGenerationContext(
        benchmark_id=state.benchmark_id,
        selected_variant=state.selected_variant,
        confirmed_facts=dict(state.confirmed_facts),
        extracted_facts=dict(state.extracted_facts),
        strict_benchmark=False,
        few_shot_case_ids=list(few_shot_case_ids or []),
    )

    known_material_ids: list[str] = []
    known_universe_ids: list[str] = []
    known_cell_ids: list[str] = []
    cell_owner_universe_ids: dict[str, list[str]] = {}
    material_roles_by_id: dict[str, str] = {}
    known_overlay_summaries: list[dict[str, Any]] = []
    expected_counts: dict[str, int] = {}
    fuel_variant_requirements: list[dict[str, Any]] = []
    material_summaries: list[dict[str, Any]] = []
    universe_summaries: list[dict[str, Any]] = []
    assembly_fuel_binding_summaries: list[dict[str, Any]] = []
    localized_insert_requirements: list[dict[str, Any]] = []
    localized_insert_universe_summaries: list[dict[str, Any]] = []
    assembly_insert_binding_summaries: list[dict[str, Any]] = []
    reference_expected_counts: dict[str, int] = {
        str(k): int(v)
        for k, v in state.metadata.get("reference_expected_counts", {}).items()
        if isinstance(v, int)
    }
    active_fuel: tuple[float, float] | None = None
    axial_domain: tuple[float, float] | None = None

    for env in state.patches.values():
        if env.status != "valid":
            continue
        ctx.validated_patch_summaries.setdefault(
            env.patch_type,
            {"status": "valid", "patch_id": env.patch_id},
        )
        content = env.content
        ptype = env.patch_type

        if ptype == "facts":
            ctx.benchmark_id = content.get("benchmark_id") or ctx.benchmark_id
            ctx.selected_variant = content.get("selected_variant") or ctx.selected_variant
            for key in (
                "expected_pin_count",
                "expected_guide_tube_count",
                "expected_instrument_tube_count",
                "expected_pyrex_count",
                "expected_thimble_plug_count",
                "expected_spacer_grid_count",
            ):
                val = content.get(key)
                if isinstance(val, int):
                    expected_counts[key] = val
            afr = content.get("active_fuel_region_cm")
            if isinstance(afr, list) and len(afr) == 2:
                active_fuel = (afr[0], afr[1])
            ad = content.get("axial_domain_cm")
            if isinstance(ad, list) and len(ad) == 2:
                axial_domain = (ad[0], ad[1])
            # Propagate feature flags.
            for flag in ("has_spacer_grids", "has_special_pin_map", "has_axial_geometry"):
                if content.get(flag):
                    ctx.extracted_facts[flag] = True
            ctx.strict_benchmark = bool(content.get("benchmark_id"))
            fvr = content.get("fuel_variant_requirements")
            if isinstance(fvr, list):
                for item in fvr:
                    if isinstance(item, dict) and item.get("variant_id"):
                        fuel_variant_requirements.append({
                            k: item.get(k)
                            for k in (
                                "variant_id", "source_label",
                                "enrichment_wt_percent", "density_g_cm3",
                                "assembly_type_ids", "expected_assembly_count",
                                "source_note",
                            )
                        })
            # P2-FULLCORE-2C-C: extract localized insert placement requirements.
            lir = content.get("localized_insert_requirements")
            if isinstance(lir, list):
                for item in lir:
                    if isinstance(item, dict) and item.get("requirement_id"):
                        localized_insert_requirements.append({
                            k: item.get(k)
                            for k in (
                                "requirement_id", "insert_kind",
                                "assembly_type_ids",
                                "expected_coordinate_count_per_assembly",
                                "expected_assembly_instance_count",
                                "host_kind", "required_profile_id",
                                "required_segment_roles",
                                "expected_insert_universe_ids",
                                "anchor_z_cm", "control_state_id",
                                "required_in_detailed_domain",
                                "source_note",
                            )
                        })
            # P2-FULLCORE-1: propagate multi-assembly fields.
            model_scope = content.get("model_scope")
            if isinstance(model_scope, str):
                ctx.model_scope = model_scope
                ctx.extracted_facts["model_scope"] = model_scope
            ac = content.get("assembly_count")
            if isinstance(ac, int):
                ctx.assembly_count = ac
                ctx.extracted_facts["assembly_count"] = ac
            cls = content.get("core_lattice_size")
            if isinstance(cls, list) and len(cls) == 2:
                ctx.core_lattice_size = (cls[0], cls[1])
            atc = content.get("assembly_type_counts")
            if isinstance(atc, dict):
                ctx.assembly_type_counts = {
                    str(k): int(v) for k, v in atc.items() if isinstance(v, (int, float))
                }
            apc = content.get("assembly_pitch_cm")
            if isinstance(apc, (int, float)):
                ctx.assembly_pitch_cm = float(apc)
            sec = content.get("scoped_expected_counts")
            if isinstance(sec, list):
                ctx.scoped_expected_counts = sec
                # Multi-assembly cores: the legacy flat ``expected_*_count``
                # fields hold core_total values, but the pin_map patch
                # describes a single (superposed) assembly lattice. Resolve
                # scoped counts to per-assembly scope so the validator
                # compares like-for-like instead of flagging a false mismatch
                # (e.g. fuel_pin 264 per-assembly vs 2376 core-total).
                resolved = resolve_expected_counts_for_pin_map(
                    sec,
                    model_scope=model_scope if isinstance(model_scope, str) else "single_assembly",
                    assembly_count=ac if isinstance(ac, int) else None,
                    assembly_type_counts=ctx.assembly_type_counts,
                )
                if resolved:
                    expected_counts.clear()
                    expected_counts.update(resolved)

        elif ptype == "materials":
            for mat in content.get("materials", []):
                mid = mat.get("material_id")
                if isinstance(mid, str):
                    known_material_ids.append(mid)
                role = mat.get("role")
                if isinstance(mid, str) and isinstance(role, str):
                    material_roles_by_id[mid] = role
                material_summaries.append({
                    "material_id": mid,
                    "role": role,
                    "source_variant_id": mat.get("source_variant_id"),
                    "density_g_cm3": mat.get("density_g_cm3"),
                    "composition_basis": mat.get("composition_basis"),
                    "composition_status": mat.get("composition_status"),
                })

        elif ptype == "universes":
            for univ in content.get("universes", []):
                uid = univ.get("universe_id")
                if isinstance(uid, str):
                    known_universe_ids.append(uid)
                cell_material_ids: list[str] = []
                cell_roles: list[str] = []
                for cell in univ.get("cells", []):
                    cid = cell.get("id")
                    if isinstance(cid, str) and isinstance(uid, str):
                        if cid not in known_cell_ids:
                            known_cell_ids.append(cid)
                        cell_owner_universe_ids.setdefault(cid, []).append(uid)
                    crole = cell.get("role")
                    if isinstance(crole, str):
                        cell_roles.append(crole)
                    cmat = cell.get("material_id")
                    if isinstance(cmat, str):
                        cell_material_ids.append(cmat)
                fuel_mats = [
                    cm for cm, cr in zip(cell_material_ids, cell_roles)
                    if cr == "fuel"
                ]
                fuel_variants = [
                    material_summaries[idx]["source_variant_id"]
                    for idx, ms in enumerate(material_summaries)
                    if ms["material_id"] in fuel_mats and ms.get("source_variant_id")
                ]
                universe_summaries.append({
                    "universe_id": uid,
                    "kind": univ.get("kind"),
                    "material_ids": cell_material_ids,
                    "fuel_material_ids": fuel_mats,
                    "fuel_variant_ids": list(dict.fromkeys(fuel_variants)),
                    "cell_roles": cell_roles,
                    "is_active_fuel_capable": len(fuel_mats) > 0,
                })

        elif ptype == "pin_map":
            for group in (
                "guide_tube_coords",
                "instrument_tube_coords",
                "pyrex_rod_coords",
                "thimble_plug_coords",
            ):
                coords = content.get(group, [])
                if isinstance(coords, list):
                    label = group.replace("_coords", "_count")
                    expected_counts[f"expected_{label}"] = len(coords)

        elif ptype == "axial_layers":
            ad = content.get("axial_domain_cm")
            if isinstance(ad, list) and len(ad) == 2 and axial_domain is None:
                axial_domain = (ad[0], ad[1])
            for layer in content.get("layers", []):
                if layer.get("role") == "active_fuel":
                    z_min = layer.get("z_min_cm")
                    z_max = layer.get("z_max_cm")
                    if isinstance(z_min, (int, float)) and isinstance(z_max, (int, float)):
                        active_fuel = (z_min, z_max)

        elif ptype == "axial_overlays":
            for ov in content.get("overlays", []):
                tl = ov.get("target_lattice_id")
                if isinstance(tl, str) and tl not in ctx.known_lattice_ids:
                    ctx.known_lattice_ids.append(tl)
                known_overlay_summaries.append({
                    k: ov.get(k)
                    for k in (
                        "overlay_id", "overlay_kind", "z_min_cm", "z_max_cm",
                        "target_lattice_id", "material_id", "geometry_mode",
                        "through_path_preserved",
                    )
                })

        elif ptype == "assembly_catalog":
            for atype in content.get("assembly_types", []):
                tid = atype.get("assembly_type_id")
                if isinstance(tid, str):
                    ctx.known_assembly_type_ids.append(tid)
                pm = atype.get("pin_map", {})
                default_uv = pm.get("default_universe_id")
                resolved_fuel_mats = [
                    us["fuel_material_ids"]
                    for us in universe_summaries
                    if us["universe_id"] == default_uv
                ]
                resolved_fuel_mids = (
                    resolved_fuel_mats[0] if resolved_fuel_mats else []
                )
                resolved_fuel_vids: list[str] = []
                for fmid in resolved_fuel_mids:
                    for ms in material_summaries:
                        if ms["material_id"] == fmid and ms.get("source_variant_id"):
                            resolved_fuel_vids.append(ms["source_variant_id"])
                assembly_fuel_binding_summaries.append({
                    "assembly_type_id": tid,
                    "fuel_variant_id": atype.get("fuel_variant_id"),
                    "default_universe_id": default_uv,
                    "resolved_fuel_material_ids": resolved_fuel_mids,
                    "resolved_fuel_variant_ids": list(dict.fromkeys(resolved_fuel_vids)),
                })

        elif ptype == "localized_insert_profiles":
            for prof in content.get("profiles", []):
                pid = prof.get("profile_id")
                if isinstance(pid, str):
                    ctx.known_insert_profile_ids.append(pid)
                ctx.insert_profile_summaries.append({
                    k: prof.get(k)
                    for k in ("profile_id", "anchor_kind", "anchor_z_cm")
                })
            ctx.movable_insert_facts = {
                "has_localized_insert_profiles": True,
            }

    ctx.expected_counts = expected_counts
    ctx.reference_expected_counts = reference_expected_counts
    ctx.expected_counts_complete = bool(state.metadata.get("expected_counts_complete", False))
    ctx.known_material_ids = list(dict.fromkeys(known_material_ids))
    ctx.material_aliases = {
        str(k): str(v)
        for k, v in state.metadata.get("material_aliases", {}).items()
        if isinstance(k, str) and isinstance(v, str)
    }
    ctx.known_universe_ids = list(dict.fromkeys(known_universe_ids))
    ctx.active_fuel_region_cm = active_fuel
    ctx.axial_domain_cm = axial_domain
    ctx.known_cell_ids = list(dict.fromkeys(known_cell_ids))
    ctx.cell_owner_universe_ids = cell_owner_universe_ids
    ctx.material_roles_by_id = material_roles_by_id
    ctx.known_overlay_summaries = known_overlay_summaries
    ctx.has_spacer_grids = bool(
        ctx.extracted_facts.get("has_spacer_grids")
        or any(s.get("overlay_kind") == "spacer_grid" for s in known_overlay_summaries)
    )
    ctx.expected_spacer_grid_count = expected_counts.get("expected_spacer_grid_count")
    ctx.fuel_variant_requirements = fuel_variant_requirements
    ctx.material_summaries = material_summaries
    ctx.universe_summaries = universe_summaries
    ctx.assembly_fuel_binding_summaries = assembly_fuel_binding_summaries
    ctx.localized_insert_requirements = localized_insert_requirements
    ctx.localized_insert_universe_summaries = localized_insert_universe_summaries
    ctx.assembly_insert_binding_summaries = assembly_insert_binding_summaries

    state.add_event(
        event_type=EVENT_PATCH_DEPENDENCY_CONTEXT_BUILT,
        message=f"context built for {patch_type} ({len(ctx.validated_patch_summaries)} valid patches)",
        data={
            "patch_type": patch_type,
            "known_material_count": len(ctx.known_material_ids),
            "known_universe_count": len(ctx.known_universe_ids),
            "expected_count_keys": list(ctx.expected_counts.keys()),
            "reference_expected_count_keys": list(ctx.reference_expected_counts.keys()),
        },
    )
    # Phase 8A Step 5: inject inventory-driven evidence payloads into
    # the patch generation context.  Returns an empty list when no
    # inventory is compiled (off mode) so legacy prompts stay
    # byte-identical.
    inventory_constraints = _inventory_planning_constraints_for_patch_type(state, patch_type)
    if inventory_constraints:
        # Phase 8A Step 6: derived planning constraints are stored in
        # the dedicated ``planning_constraints`` field (NOT the
        # ``investigation_evidence`` field).  This separates source-
        # backed EvidenceClaims from inventory-derived constraints in
        # the prompt so the LLM and reviewers can distinguish them.
        ctx.planning_constraints = [
            c.to_prompt_dict() if hasattr(c, "to_prompt_dict") else dict(c)
            for c in inventory_constraints
        ]
    return ctx


def _patch_generation_context_fingerprint(
    state: PlanBuildState,
    patch_type: str,
) -> str:
    """Fingerprint the validated upstream contract for one patch candidate.

    Candidate reuse is only unsafe when the same patch is requested against
    the same accepted upstream content.  Including the canonical task plan
    avoids treating a post-Facts-revision retry as a duplicate of an older
    task plan.
    """
    upstream_hashes = {
        envelope.patch_type: compute_candidate_hash(
            target_patch_type=envelope.patch_type,
            candidate_patch=envelope.content,
        )
        for envelope in state.patches.values()
        if envelope.status == "valid" and envelope.patch_type != patch_type
    }
    task_plan_hash = (
        state.canonical_task_plan.plan_hash
        if state.canonical_task_plan is not None
        else None
    )
    return compute_candidate_hash(
        target_patch_type=patch_type,
        candidate_patch={
            "upstream_patch_hashes": upstream_hashes,
            "canonical_task_plan_hash": task_plan_hash,
        },
    )


# ---------------------------------------------------------------------------
# Deterministic settings fallback
# ---------------------------------------------------------------------------


def build_deterministic_settings_patch(state: PlanBuildState) -> SettingsPatch:
    """Return a default SettingsPatch without calling the LLM."""
    return SettingsPatch(
        source_strategy="active_fuel_box",
        source_requires_fissionable_constraint=True,
        plot_strategy="full_assembly",
        cross_sections_runtime_required=True,
        tallies_required_for_smoke_test=False,
        assumptions=["cross sections resolved at runtime via OPENMC_CROSS_SECTIONS"],
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_valid_patch(state: PlanBuildState, patch_type: str) -> bool:
    return any(
        e.patch_type == patch_type and e.status == "valid"
        for e in state.patches.values()
    )


def _add_envelope(
    state: PlanBuildState,
    patch_type: str,
    content: dict[str, Any],
    source: str = "llm",
) -> PlanPatchEnvelope:
    import hashlib
    digest = hashlib.md5(
        str(content).encode("utf-8"), usedforsecurity=False
    ).hexdigest()[:8]
    envelope = PlanPatchEnvelope(
        patch_id=f"patch_{patch_type}_{digest}",
        patch_type=patch_type,
        content=content,
        source=source,  # type: ignore[arg-type]
        status="valid",
    )
    state.add_patch(envelope)
    return envelope


def _extract_reference_expected_counts(reference_data: dict[str, Any] | None) -> dict[str, int]:
    """Extract complete role counts from a reference facts patch."""
    if not reference_data:
        return {}
    patches = reference_data.get("patches", [])
    if not isinstance(patches, list):
        return {}
    mapping = {
        "expected_pin_count": "fuel_pin",
        "expected_guide_tube_count": "guide_tube",
        "expected_instrument_tube_count": "instrument_tube",
        "expected_pyrex_count": "pyrex_rod",
        "expected_thimble_plug_count": "thimble_plug",
    }
    for entry in patches:
        if not isinstance(entry, dict) or entry.get("patch_type") != "facts":
            continue
        counts: dict[str, int] = {}
        for fact_key, role in mapping.items():
            value = entry.get(fact_key)
            if isinstance(value, int):
                counts[role] = value
        return counts
    return {}


def _record_reference_metadata(
    state: PlanBuildState,
    reference_data: dict[str, Any] | None,
) -> None:
    counts = _extract_reference_expected_counts(reference_data)
    if counts:
        if state.metadata.get("reference_expected_counts") != counts:
            state.add_event(
                event_type=EVENT_REFERENCE_COUNTS_APPLIED,
                message="reference expected pin counts applied",
                data={"expected_counts": counts},
            )
        state.metadata["reference_expected_counts"] = counts
        state.metadata["expected_counts_complete"] = True
    if reference_data:
        state.metadata["reference_match_status"] = str(
            reference_data.get("_reference_match_status") or "matched"
        )
        ref_path = reference_data.get("_reference_path")
        if isinstance(ref_path, str):
            state.metadata["reference_path"] = ref_path
        aliases = reference_data.get("material_aliases")
        if isinstance(aliases, dict):
            state.metadata["material_aliases"] = {
                str(k): str(v)
                for k, v in aliases.items()
                if isinstance(k, str) and isinstance(v, str)
            }


def _validation_context_for_state(state: PlanBuildState, patch_type: str) -> Any:
    from .patch_generator import _to_validation_context

    return _to_validation_context(build_generation_context_from_state(state, patch_type))


def _latest_assembly_summary(state: PlanBuildState) -> dict[str, Any]:
    for event in reversed(state.build_log):
        if event.event_type in (EVENT_ASSEMBLY_COMPLETED, EVENT_ASSEMBLY_FAILED):
            summary = event.data.get("summary", event.data)
            return summary if isinstance(summary, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# Phase 8A Step 5: Geometry Component Inventory compilation + injection
# ---------------------------------------------------------------------------


def _maybe_compile_geometry_inventory(
    *,
    state: PlanBuildState,
    facts_env: Any,
    requirement: str,
    artifact_writer: Any | None = None,
) -> dict[str, Any]:
    """Compile the :class:`GeometryComponentInventory` from accepted Facts.

    Called right after the Facts Gate accepts.  Stores the serialized
    inventory on ``state.metadata['planning_geometry_inventory']`` so
    the Materials and Universes patch generators can consume it via
    :func:`_inventory_planning_constraints_for_patch_type`.

    Returns a typed status dict:

    * ``compiled`` (bool): True iff the inventory + requirement sets
      were produced without exception.
    * ``inventory_hash`` (str): hash of the compiled inventory.
    * ``error`` (str | None): exception message when compilation
      crashed.
    * ``failure_code`` (str): one of the stable codes below.

    Phase 8A Step 6 (P0-2 fix): the previous implementation only
    logged an event on exception and returned, allowing the run to
    continue with Materials/Universes via the legacy prompt path in
    controlled mode.  The caller now inspects ``compiled`` and blocks
    the run when ``compiled=False`` in controlled mode.
    """

    try:
        from openmc_agent.plan_investigation.evidence_ledger import (
            PlanningEvidenceLedger,
        )
        from openmc_agent.plan_investigation.geometry_inventory import (
            compile_geometry_component_inventory,
        )
        from openmc_agent.plan_investigation.runner import (
            build_investigation_ledger,
            build_investigation_source_index,
        )
    except ImportError:
        return {
            "compiled": False,
            "inventory_hash": "",
            "error": "plan_investigation package unavailable",
            "failure_code": "planning.inventory.compile_module_unavailable",
        }
    state.add_event(
        "planning.geometry_inventory_compilation_started",
        "geometry component inventory compilation starting",
        {},
    )
    try:
        source_index = build_investigation_source_index(requirement)
        ledger = build_investigation_ledger(
            requirement_text=requirement,
            source_indexes={source_index.document.source_id: source_index},
        )
        existing_ledger_dump = state.metadata.get("planning_evidence_ledger")
        if existing_ledger_dump:
            ledger = PlanningEvidenceLedger.model_validate(existing_ledger_dump)

        from .patches import parse_patch_content
        facts_patch = parse_patch_content("facts", facts_env.content)

        inventory = compile_geometry_component_inventory(
            accepted_facts=facts_patch,
            evidence_ledger=ledger,
            source_indexes={source_index.document.source_id: source_index},
            facts_accepted=True,
        )
        state.metadata["planning_geometry_inventory"] = inventory.model_dump(mode="json")
        state.metadata["planning_geometry_inventory_hash"] = inventory.inventory_hash
        from openmc_agent.plan_builder.material_requirements import (
            extract_material_requirements_from_inventory,
        )
        mreq_set = extract_material_requirements_from_inventory(
            inventory, accepted_facts=facts_patch
        )
        state.metadata["planning_material_requirement_set"] = mreq_set.model_dump(
            mode="json"
        )
        from openmc_agent.plan_investigation.inventory_universe_requirements import (
            extract_universe_requirements_from_inventory,
        )
        ureq_set = extract_universe_requirements_from_inventory(
            inventory,
            material_requirement_set=mreq_set,
            accepted_facts=facts_patch,
        )
        state.metadata["planning_universe_requirement_set"] = ureq_set.model_dump(
            mode="json"
        )
        from openmc_agent.plan_builder.requirement_skeletons import (
            compile_material_requirement_skeleton,
            compile_universe_requirement_skeleton,
        )
        material_skeleton = compile_material_requirement_skeleton(
            inventory=inventory, accepted_facts=facts_patch, evidence_ledger=ledger,
        )
        universe_skeleton = compile_universe_requirement_skeleton(
            inventory=inventory, accepted_facts=facts_patch, evidence_ledger=ledger,
        )
        state.metadata["planning_material_requirement_skeleton"] = material_skeleton.model_dump(mode="json")
        state.metadata["planning_universe_requirement_skeleton"] = universe_skeleton.model_dump(mode="json")
        state.add_event(
            "planning.geometry_inventory_compiled",
            "geometry component inventory compiled",
            {
                "inventory_hash": inventory.inventory_hash[:12],
                "radial_profile_count": len(inventory.radial_profiles),
                "axial_region_count": len(inventory.axial_regions),
                "material_role_count": len(inventory.material_role_requirements),
                "unresolved_component_count": len(inventory.unresolved_components),
            },
        )
        if artifact_writer is not None:
            try:
                artifact_writer._write(
                    "geometry_component_inventory.json",
                    inventory.model_dump(mode="json"),
                )
                artifact_writer._write("material_requirement_skeleton.json", material_skeleton)
                artifact_writer._write("universe_requirement_skeleton.json", universe_skeleton)
            except Exception:
                pass
        return {
            "compiled": True,
            "inventory_hash": inventory.inventory_hash,
            "error": None,
            "failure_code": "",
        }
    except Exception as exc:
        state.add_event(
            "planning.geometry_inventory_blocked",
            f"geometry inventory compilation failed: {type(exc).__name__}: {exc}",
            {"error": str(exc)[:200]},
        )
        return {
            "compiled": False,
            "inventory_hash": "",
            "error": f"{type(exc).__name__}: {exc}",
            "failure_code": "planning.inventory.compilation_failed",
        }


def _maybe_execute_research_for_gate(
    *,
    gate_id: str,
    stage: Any,
    state: PlanBuildState,
    error_findings: list[Any],
    input_hash: str,
    shared_source_index: Any,
    shared_ledger: Any,
    requirement: str,
    artifact_writer: Any | None = None,
    plan_investigation_output_dir: str | None = None,
    plan_loop_output_dir: str | None = None,
    plan_investigation_client: Any = None,
) -> bool:
    """Execute a PlanResearchRequest for source-coverage findings.

    Returns True when new evidence was committed and the gate should
    reopen; False when no actionable evidence was found (caller falls
    through to the existing retry/block path).

    Phase 8A Step 6B (Section 14): the research executor reuses the
    existing read-only investigation tools (query_evidence_ledger +
    search_source_index).  No shell, no network search, no repository
    grep.  Budget knobs enforce bounded execution.
    """
    from .closed_loop.models import PlanStageStatus

    if not error_findings:
        return False
    if shared_source_index is None or shared_ledger is None:
        return False
    try:
        from openmc_agent.plan_investigation.research_router import (
            route_findings_to_research,
            aggregate_action,
        )
        from openmc_agent.plan_investigation.research_models import (
            PlanResearchRequest,
            PlanResearchStatus,
        )
        from openmc_agent.plan_investigation.research_executor import (
            execute_plan_research_request,
            ResearchExecutorConfig,
        )
        from openmc_agent.plan_builder.closed_loop.models import PlanReviewAction
    except ImportError:
        return False
    decisions = route_findings_to_research(
        gate_id=gate_id, findings=error_findings,
    )
    if not any(d.action is PlanReviewAction.RETRIEVE_EVIDENCE for d in decisions):
        return False
    # Build a single PlanResearchRequest covering all retrievable findings.
    targets = tuple(
        target for d in decisions if d.action is PlanReviewAction.RETRIEVE_EVIDENCE
        for target in d.targets
    )
    if not targets:
        return False
    # No-progress guards: track seen fingerprints + delta hashes on state.
    seen_fp = set(state.metadata.setdefault("plan_research_attempts_by_fingerprint", []))
    seen_deltas = set(state.metadata.setdefault("plan_research_candidate_hashes", []))
    round_num = int(state.metadata.get("plan_research_rounds", 0)) + 1
    state.metadata["plan_research_rounds"] = round_num
    request = PlanResearchRequest(
        gate_id=gate_id,
        finding_ids=tuple(f.finding_id for f in error_findings),
        issue_codes=tuple(f.code for f in error_findings),
        targets=targets,
        ledger_hash_before=shared_ledger.ledger_hash,
        gate_input_hash=input_hash,
        research_round=round_num,
    )
    state.add_event(
        "planning.research_requested",
        f"research request {request.request_id} for gate {gate_id}",
        {
            "request_id": request.request_id,
            "target_count": len(targets),
            "finding_count": len(error_findings),
            "round": round_num,
        },
    )
    out_dir = (
        Path(plan_investigation_output_dir)
        if plan_investigation_output_dir is not None
        else (Path(plan_loop_output_dir) if plan_loop_output_dir else None)
    )
    result = execute_plan_research_request(
        request=request,
        source_index=shared_source_index,
        ledger=shared_ledger,
        config=ResearchExecutorConfig(),
        seen_request_fingerprints=seen_fp,
        seen_delta_hashes=seen_deltas,
        add_event=lambda evt, msg, data: state.add_event(evt, msg, data),
        artifact_output_dir=out_dir,
    )
    # Record the result on state for auditor + resume fingerprint.
    state.metadata.setdefault("plan_research_results", []).append(
        result.model_dump(mode="json")
    )
    state.metadata.setdefault("plan_research_attempts_by_fingerprint", []).append(
        request.request_fingerprint
    )
    if result.evidence_delta and not result.evidence_delta.is_empty:
        state.metadata.setdefault("plan_research_candidate_hashes", []).append(
            result.evidence_delta.delta_hash
        )
    if artifact_writer is not None:
        try:
            artifact_writer._write(
                f"research_result_{request.request_id}.json",
                result.model_dump(mode="json"),
            )
        except Exception:
            pass
    # Determine whether to reopen the gate.
    if result.status == PlanResearchStatus.EVIDENCE_ADDED:
        # New evidence committed → reopen the gate so the owner patch
        # gets regenerated with the new context.
        stage.status = PlanStageStatus.PENDING
        stage.completed_at = None
        state.add_event(
            "planning.research_evidence_added",
            f"research added evidence; gate {gate_id} reopened",
            {
                "request_id": request.request_id,
                "added_claim_count": len(result.evidence_delta.added_claim_ids if result.evidence_delta else ()),
                "added_span_count": len(result.evidence_delta.added_source_span_ids if result.evidence_delta else ()),
            },
        )
        return True
    # Phase 8A Step 7: when candidate spans were found but no evidence
    # committed yet, run the LLM synthesis path.  If the synthesis
    # produces accepted claims that change the Ledger hash, reopen the
    # gate.  Otherwise fall through to the existing retry/block path.
    if (
        result.status == PlanResearchStatus.CANDIDATE_SPANS_FOUND
        and plan_investigation_client is not None
    ):
        from openmc_agent.plan_investigation.research_synthesis import (
            run_research_evidence_synthesis,
        )
        accepted_facts_obj, geometry_inventory_obj = _load_inventory_context_objects(state)
        material_req_obj = _load_material_requirement_set(state)
        universe_req_obj = _load_universe_requirement_set(state)
        # Locate accepted Materials + Universes patches.
        materials_env = next(
            (item for item in state.patches.values()
             if item.patch_type == "materials" and item.status == "valid"),
            None,
        )
        universes_env = next(
            (item for item in state.patches.values()
             if item.patch_type == "universes" and item.status == "valid"),
            None,
        )
        delta = run_research_evidence_synthesis(
            request=request,
            research_result=result,
            ledger=shared_ledger,
            source_index=shared_source_index,
            gate_findings=error_findings,
            geometry_inventory=geometry_inventory_obj,
            material_requirement_set=material_req_obj,
            universe_requirement_set=universe_req_obj,
            materials_patch=(materials_env.content if materials_env else None),
            universes_patch=(universes_env.content if universes_env else None),
            llm_client=plan_investigation_client,
            add_event=lambda evt, msg, data: state.add_event(evt, msg, data),
        )
        if delta is not None and delta.ledger_hash_after != delta.ledger_hash_before:
            # Real evidence committed → reopen the gate.
            state.metadata.setdefault("planning_evidence_deltas", []).append(
                delta.model_dump(mode="json")
            )
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event(
                "planning.research_evidence_committed",
                f"LLM synthesis committed {len(delta.added_claim_ids)} claims; gate {gate_id} reopened",
                {
                    "request_id": request.request_id,
                    "added_claim_count": len(delta.added_claim_ids),
                    "ledger_hash_before": delta.ledger_hash_before[:12],
                    "ledger_hash_after": delta.ledger_hash_after[:12],
                },
            )
            return True
    if result.status == PlanResearchStatus.NO_PROGRESS:
        state.add_event(
            "planning.research_no_progress",
            f"research request {request.request_id} made no progress",
            {"request_id": request.request_id},
        )
    elif result.status == PlanResearchStatus.NO_EVIDENCE_FOUND:
        state.add_event(
            "planning.research_no_evidence",
            f"research request {request.request_id} found no evidence",
            {
                "request_id": request.request_id,
                "absence_count": len(result.absence_records),
            },
        )
        # Phase 8A Step 7: when research finds no evidence, the
        # blocking findings are Patch-content issues (not source-evidence
        # gaps).  Invalidate the affected owner patches so they get
        # regenerated with the planning_constraints context.  This is
        # the REVISE_CURRENT_PATCH action — the Materials/Universes
        # patches will be rebuilt with explicit role/profile bindings
        # from the inventory requirement sets.
        owners = set()
        for finding in error_findings:
            for ptype in (
                getattr(finding, "affected_patch_types", None)
                or (finding.get("affected_patch_types") if isinstance(finding, dict) else [])
                or []
            ):
                if ptype in {"materials", "universes"}:
                    owners.add(ptype)
        if owners:
            invalidated = []
            for ptype in owners:
                # Remove the existing valid patch so it gets regenerated.
                to_remove = [
                    pid for pid, p in state.patches.items()
                    if p.patch_type == ptype and p.status == "valid"
                ]
                for pid in to_remove:
                    state.patches[pid].status = "invalid"
                    invalidated.append(pid)
            # Reopen the MU gate.
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event(
                "planning.research_patch_revision_scheduled",
                f"research found no evidence; invalidating {len(invalidated)} patch(es) for revision: {sorted(owners)}",
                {
                    "invalidated_patch_ids": invalidated,
                    "owner_patch_types": sorted(owners),
                    "gate_id": gate_id,
                },
            )
            return True
    return False


def _load_inventory_context_objects(
    state: PlanBuildState,
) -> tuple[Any, Any]:
    """Load typed Facts + GeometryComponentInventory from state.metadata.

    Returns ``(accepted_facts, geometry_inventory)`` so the Materials /
    Universes investigation baseline resolver can read fuel variants
    and radial profiles.  Both may be ``None`` when the inventory has
    not been compiled yet (off mode / pre-Facts acceptance).
    """

    inv_dump = state.metadata.get("planning_geometry_inventory")
    if not inv_dump:
        return None, None
    try:
        from openmc_agent.plan_investigation.geometry_inventory import (
            GeometryComponentInventory,
        )
        inventory = GeometryComponentInventory.model_validate(inv_dump)
    except Exception:
        inventory = None
    facts_env = next(
        (item for item in state.patches.values()
         if item.patch_type == "facts" and item.status == "valid"),
        None,
    )
    accepted_facts = None
    if facts_env is not None:
        try:
            from .patches import parse_patch_content
            accepted_facts = parse_patch_content("facts", facts_env.content)
        except Exception:
            accepted_facts = None
    return accepted_facts, inventory


def _load_inventory_context_objects(
    state: PlanBuildState,
) -> tuple[Any, Any]:
    """Load typed Facts + GeometryComponentInventory from state.metadata.

    Returns ``(accepted_facts, geometry_inventory)`` so the Materials /
    Universes investigation baseline resolver can read fuel variants
    and radial profiles.  Both may be ``None`` when the inventory has
    not been compiled yet (off mode / pre-Facts acceptance).
    """

    inv_dump = state.metadata.get("planning_geometry_inventory")
    if not inv_dump:
        return None, None
    try:
        from openmc_agent.plan_investigation.geometry_inventory import (
            GeometryComponentInventory,
        )
        inventory = GeometryComponentInventory.model_validate(inv_dump)
    except Exception:
        inventory = None
    facts_env = next(
        (item for item in state.patches.values()
         if item.patch_type == "facts" and item.status == "valid"),
        None,
    )
    accepted_facts = None
    if facts_env is not None:
        try:
            from .patches import parse_patch_content
            accepted_facts = parse_patch_content("facts", facts_env.content)
        except Exception:
            accepted_facts = None
    return accepted_facts, inventory


def _load_material_requirement_set(state: PlanBuildState) -> Any:
    """Load the MaterialGenerationRequirementSet from state.metadata."""

    dump = state.metadata.get("planning_material_requirement_set")
    if not dump:
        return None
    try:
        from openmc_agent.plan_builder.material_requirements import (
            MaterialGenerationRequirementSet,
        )
        return MaterialGenerationRequirementSet.model_validate(dump)
    except Exception:
        return None


def _load_universe_requirement_set(state: PlanBuildState) -> Any:
    """Load the InventoryUniverseRequirementSet from state.metadata."""

    dump = state.metadata.get("planning_universe_requirement_set")
    if not dump:
        return None
    try:
        from openmc_agent.plan_investigation.inventory_universe_requirements import (
            InventoryUniverseRequirementSet,
        )
        return InventoryUniverseRequirementSet.model_validate(dump)
    except Exception:
        return None


def _inventory_planning_constraints_for_patch_type(
    state: PlanBuildState,
    patch_type: str,
) -> list[Any]:
    """Build typed planning constraints from the inventory requirement sets.

    Returns an empty list when no inventory is present (off mode) so
    the patch prompt stays byte-identical to legacy.

    Phase 8A Step 6 (P0-4 fix): the previous implementation produced
    fake "EvidenceClaim" payloads with ``status="explicit"`` and
    ``source_spans=[]`` whose ``claim_id`` was actually a
    ``requirement_id``.  Those payloads were indistinguishable from
    real source-backed claims in the prompt renderer.  The new
    payloads are :class:`EvidenceConstraintPayload` instances with
    ``derivation_status=deterministically_derived``.
    """

    inv_dump = state.metadata.get("planning_geometry_inventory")
    if not inv_dump:
        return []
    try:
        from openmc_agent.plan_investigation.inventory_constraints import (
            build_inventory_constraints_for_patch_type,
        )
    except ImportError:
        return []
    mreq_dump = state.metadata.get("planning_material_requirement_set") or {}
    ureq_dump = state.metadata.get("planning_universe_requirement_set") or {}
    ledger_hash = ""
    try:
        ledger_dump = state.metadata.get("planning_evidence_ledger") or {}
        ledger_hash = ledger_dump.get("ledger_hash", "") if isinstance(ledger_dump, dict) else ""
    except Exception:
        ledger_hash = ""
    constraints = build_inventory_constraints_for_patch_type(
        patch_type=patch_type,
        inventory_dump=inv_dump,
        material_requirement_set_dump=mreq_dump,
        universe_requirement_set_dump=ureq_dump,
        ledger_hash=ledger_hash,
    )
    skeleton_key = (
        "planning_material_requirement_skeleton"
        if patch_type == "materials"
        else "planning_universe_requirement_skeleton"
    )
    skeleton = state.metadata.get(skeleton_key) or {}
    for requirement in skeleton.get("requirements", []) if isinstance(skeleton, dict) else []:
        constraints.append({
            "constraint_type": f"{patch_type}_requirement_skeleton",
            "derivation_status": "deterministically_derived",
            "requirement": requirement,
            "immutable_fields": (
                ["material_id", "role", "source_variant"]
                if patch_type == "materials"
                else ["universe_id", "component_kind", "geometry_profile", "source_requirement_ids"]
            ),
        })
    return constraints


def _inventory_evidence_payloads_for_patch_type(
    state: PlanBuildState,
    patch_type: str,
) -> list[dict[str, Any]]:
    """Backward-compat shim.

    Phase 8A Step 6 deprecated the legacy "fake EvidenceClaim" payload
    path; callers should consume ``planning_constraints`` via
    :func:`_inventory_planning_constraints_for_patch_type` instead.
    This shim returns ``[]`` always so any stray caller falls back to
    the legacy (no-payload) prompt without misrepresenting derived
    requirements as source-backed claims.
    """

    return []


def _maybe_run_inventory_preflight(
    *,
    state: PlanBuildState,
    materials_env: Any,
    universes_env: Any,
    artifact_writer: Any | None = None,
    controlled_mode: bool = False,
) -> list[dict[str, Any]]:
    """Run the inventory-driven Material-Universe preflight.

    Returns a list of issue dicts (same shape as the existing
    ``run_material_universe_preflight`` output).  Returns an empty list
    when no inventory is compiled (off mode).

    Phase 8A Step 6 (P0-3 fix): the previous implementation caught
    every exception and returned ``[]``, which the gate treated as
    "no deterministic finding".  In ``controlled_mode`` we now emit a
    blocking finding with code
    ``material_universe.inventory_preflight_exception`` (or
    ``material_universe.inventory_preflight_not_executed`` when the
    preflight was skipped despite an inventory being present) so the
    gate fails closed.  The finding owner is taken from the explicit
    :data:`INVENTORY_FINDING_OWNER_MAP` (no more string-contains
    matching with a default of ``materials``).
    """

    result = _run_typed_inventory_preflight(
        state=state,
        materials_env=materials_env,
        universes_env=universes_env,
        artifact_writer=artifact_writer,
    )
    if not result.has_blocking_deterministic_finding:
        # Convert report findings to dict shape with explicit owner.
        return _inventory_preflight_findings_to_dicts(result.report)
    if not result.executed:
        # Preflight did not run (inventory present but report missing).
        # In controlled mode this is blocking; in advisory it is a
        # warning only.
        if controlled_mode:
            return [{
                "code": result.failure_code or "material_universe.inventory_preflight_not_executed",
                "severity": "error",
                "message": "inventory preflight did not execute (controlled mode requires it)",
                "owner_patch_type": "plan_investigation",
            }]
        state.add_event(
            "planning.inventory_material_universe_preflight_skipped",
            "inventory preflight did not execute (advisory mode)",
            {"failure_code": result.failure_code},
        )
        return []
    if result.execution_error is not None:
        # Preflight crashed.
        if controlled_mode:
            return [{
                "code": result.failure_code or "material_universe.inventory_preflight_exception",
                "severity": "error",
                "message": f"inventory preflight crashed: {result.execution_error[:160]}",
                "owner_patch_type": "plan_investigation",
            }]
        return []
    # Preflight ran and produced error-severity findings.
    return _inventory_preflight_findings_to_dicts(result.report)


def _run_typed_inventory_preflight(
    *,
    state: PlanBuildState,
    materials_env: Any,
    universes_env: Any,
    artifact_writer: Any | None = None,
) -> "InventoryPreflightExecutionResult":
    """Typed wrapper that never raises and never silently returns []."""

    from openmc_agent.plan_investigation.inventory_preflight import (
        InventoryPreflightExecutionResult,
        PREFLIGHT_EXCEPTION_CODE,
        PREFLIGHT_NOT_EXECUTED_CODE,
    )
    inv_dump = state.metadata.get("planning_geometry_inventory")
    if not inv_dump:
        # No inventory present (off mode): preflight did not run and
        # has nothing to report.  ``inventory_present=False`` ensures
        # ``has_blocking_deterministic_finding`` is False so the gate
        # does not spuriously block in off mode.
        return InventoryPreflightExecutionResult(
            executed=False,
            inventory_present=False,
            failure_code="",  # silent in off mode
        )
    try:
        from openmc_agent.plan_investigation.geometry_inventory import (
            GeometryComponentInventory,
        )
        from openmc_agent.plan_investigation.inventory_preflight import (
            run_geometry_inventory_material_universe_preflight,
        )
        from openmc_agent.plan_investigation.inventory_universe_requirements import (
            InventoryUniverseRequirementSet,
        )
        from openmc_agent.plan_builder.material_requirements import (
            MaterialGenerationRequirementSet,
        )
        inventory = GeometryComponentInventory.model_validate(inv_dump)
        mreq_set = MaterialGenerationRequirementSet.model_validate(
            state.metadata.get("planning_material_requirement_set") or {}
        )
        ureq_set = InventoryUniverseRequirementSet.model_validate(
            state.metadata.get("planning_universe_requirement_set") or {}
        )
        materials_patch_obj = materials_env.content if materials_env is not None else None
        universes_patch_obj = universes_env.content if universes_env is not None else None
        known_material_ids = [
            m.get("material_id") for m in (materials_patch_obj or {}).get("materials", [])
        ] if isinstance(materials_patch_obj, dict) else []
        known_material_ids = [mid for mid in known_material_ids if mid]
        report = run_geometry_inventory_material_universe_preflight(
            inventory=inventory,
            material_requirement_set=mreq_set,
            universe_requirement_set=ureq_set,
            materials_patch=materials_patch_obj,
            universes_patch=universes_patch_obj,
            known_material_ids=known_material_ids,
        )
        if artifact_writer is not None:
            try:
                artifact_writer._write(
                    "inventory_material_universe_preflight.json",
                    report.to_dict(),
                )
            except Exception:
                pass
        state.add_event(
            "planning.inventory_material_universe_preflight_completed",
            "inventory material-universe preflight completed",
            {
                "passed": report.passed,
                "error_count": report.error_count,
                "warning_count": report.warning_count,
            },
        )
        return InventoryPreflightExecutionResult(
            executed=True,
            report=report,
        )
    except Exception as exc:
        state.add_event(
            "planning.inventory_material_universe_preflight_failed",
            f"inventory preflight failed: {type(exc).__name__}: {exc}",
            {"error": str(exc)[:200]},
        )
        return InventoryPreflightExecutionResult(
            executed=False,
            execution_error=f"{type(exc).__name__}: {exc}",
            failure_code=PREFLIGHT_EXCEPTION_CODE,
        )


def _inventory_preflight_findings_to_dicts(report: Any) -> list[dict[str, Any]]:
    """Convert preflight findings to dict shape with EXPLICIT owner."""

    from openmc_agent.plan_investigation.inventory_preflight import (
        owner_for_inventory_finding_code,
    )
    if report is None:
        return []
    out: list[dict[str, Any]] = []
    for finding in report.findings:
        code = finding.code
        owner = owner_for_inventory_finding_code(code)
        if owner is None:
            # Unknown code: fail closed by routing to plan_investigation
            # rather than the legacy default of "materials".
            owner = "plan_investigation"
        out.append({
            "code": code,
            "severity": finding.severity,
            "message": finding.message,
            "owner_patch_type": owner,
        })
    return out


# ---------------------------------------------------------------------------
# Main executor
# ---------------------------------------------------------------------------


def run_incremental_planning(
    *,
    requirement: str,
    state: PlanBuildState,
    llm_client: Any,
    max_patch_attempts: int = 2,
    strict: bool = True,
    task_order: list[str] | None = None,
    reference_patch_policy: str = "off",
    reference_path: str | Path | None = None,
    few_shot_case_ids: list[str] | None = None,
    material_policy: Any = None,
    plan_loop_policy: Any = None,
    plan_loop_output_dir: str | Path | None = None,
    plan_reviewer_client: Any = None,
    plan_repair_client: Any = None,
    universes_generation_mode: str = "auto",
    universe_fragment_max_tokens: int | None = None,
    large_patch_safe_output_ratio: float = 0.6,
    strict_structured_patch_output: bool = False,
    plan_investigation_config: Any = None,
    plan_investigation_client: Any = None,
    plan_investigation_registry: Any = None,
    plan_investigation_policy_registry: Any = None,
    plan_investigation_output_dir: str | Path | None = None,
    checkpoint_callback: Any = None,
    facts_action_callback: Any = None,
) -> IncrementalExecutionResult:
    """Run the full incremental planning pipeline.

    Parameters
    ----------
    reference_patch_policy
        Controls when reference patches are used for structural patches:
        ``"off"`` (LLM only), ``"reference_only_for_structural"``
        (structural patches from reference, LLM for facts/materials/universes),
        ``"fallback_after_llm_failure"`` (try LLM, then reference),
        ``"prefer_reference_for_structural"`` (use reference when available,
        otherwise continue with input-driven LLM patch generation).
    reference_path
        Explicit path to reference file.  If None, tries benchmark lookup.
    material_policy
        Optional material composition policy forwarded to the assembler.
        Accepts the enum, a string value, or None (assembler default).
    """
    from .closed_loop.artifacts import PlanLoopArtifactWriter
    from .closed_loop.controller import (
        build_advisory_outcome,
        check_stage_budget,
        enter_review_cycle,
        guard_blocked_stage,
        initialize_plan_loop_state,
        record_candidate,
        record_decision,
        record_findings,
        record_no_progress,
        transition_stage,
    )
    from .closed_loop.facts_evidence import build_facts_evidence_packs
    from .closed_loop.facts_review_prompts import build_facts_review_prompt
    from .closed_loop.facts_reviewer import run_facts_review
    from .closed_loop.models import PlanClosedLoopPolicy, PlanFindingCategory, PlanFindingSeverity, PlanGateId, PlanLoopMode, PlanReviewAction, PlanReviewDecision, PlanReviewFinding, PlanStageStatus
    from .closed_loop.policy import canonical_gate_order, compute_allowed_actions
    from .closed_loop.fingerprints import compute_issue_fingerprint

    policy = PlanClosedLoopPolicy.model_validate(plan_loop_policy or {})
    # Advisory is explicitly useful as an observability mode.  Its default
    # effective registry is all gates while the model default remains inert.
    if policy.mode is PlanLoopMode.ADVISORY and not any(policy.gate_enabled.values()):
        policy = policy.model_copy(update={"gate_enabled": {gate: True for gate in canonical_gate_order()}})
    if policy.mode is PlanLoopMode.CONTROLLED and not any(policy.gate_enabled.values()):
        policy = policy.model_copy(update={"gate_enabled": {PlanGateId.FACTS: True}})
    if policy.mode is PlanLoopMode.CONTROLLED and policy.gate_enabled.get(PlanGateId.PLACEMENT, False) and not policy.gate_enabled.get(PlanGateId.FACTS, False):
        issue = IncrementalExecutionIssue(code="planning.closed_loop.invalid_gate_configuration", severity="error", message="controlled placement requires the facts gate")
        return IncrementalExecutionResult(ok=False, state=state, issues=[issue], summary={"issue_codes": [issue.code]}, plan_loop_outcome={"status": "blocked", "detail": issue.message, "additional_llm_calls_used": 0})
    # Phase-4: Material-Universe gate requires Facts to be accepted first.
    if policy.mode is PlanLoopMode.CONTROLLED and policy.gate_enabled.get(PlanGateId.MATERIAL_UNIVERSE, False) and not policy.gate_enabled.get(PlanGateId.FACTS, False):
        issue = IncrementalExecutionIssue(code="planning.closed_loop.invalid_gate_configuration", severity="error", message="controlled material-universe requires the facts gate")
        return IncrementalExecutionResult(ok=False, state=state, issues=[issue], summary={"issue_codes": [issue.code]}, plan_loop_outcome={"status": "blocked", "detail": issue.message, "additional_llm_calls_used": 0})
    supported_controlled = {
        PlanGateId.FACTS,
        PlanGateId.MATERIAL_UNIVERSE,
        PlanGateId.PLACEMENT,
        PlanGateId.AXIAL_GEOMETRY,
        PlanGateId.ASSEMBLED_PLAN,
    }
    unsupported_controlled = [gate for gate, enabled in policy.gate_enabled.items() if enabled and gate not in supported_controlled]
    if policy.mode is PlanLoopMode.CONTROLLED and unsupported_controlled:
        issue = IncrementalExecutionIssue(
            code="planning.closed_loop.gate_not_implemented", severity="error",
            message="controlled mode currently implements only facts, material_universe, placement, axial_geometry and assembled_plan gates",
        )
        state.add_event(
            event_type="planning.closed_loop.controlled_not_implemented",
            message=issue.message,
            data={"mode": policy.mode.value, "gates": [gate.value for gate in unsupported_controlled]},
        )
        return IncrementalExecutionResult(
            ok=False, state=state, issues=[issue],
            summary={"issue_codes": [issue.code], "gate_not_implemented": True},
            plan_loop_outcome={"status": "blocked", "detail": issue.message, "additional_llm_calls_used": 0},
        )

    issues: list[IncrementalExecutionIssue] = []
    reference_data: dict[str, Any] | None = None
    reference_patches_used: list[str] = []

    # Note: reference loading is deferred until after facts patch is generated,
    # so benchmark_id can be extracted from FactsPatch content (LLM output).
    # This keeps the system benchmark-agnostic — no hardcoded text matching.

    state.add_event(
        event_type=EVENT_INCREMENTAL_EXECUTION_STARTED,
        message="incremental planning execution started",
        data={
            "max_patch_attempts": max_patch_attempts,
            "reference_patch_policy": reference_patch_policy,
            "reference_available": reference_data is not None,
        },
    )

    order = task_order or default_patch_task_order(state)
    # Only controlled Placement Gate gets a barrier.  The historical off and
    # advisory orders must remain byte-for-byte behaviourally identical.
    placement_controlled = policy.mode is PlanLoopMode.CONTROLLED and policy.gate_enabled.get(PlanGateId.PLACEMENT, False)
    material_universe_controlled = policy.mode is PlanLoopMode.CONTROLLED and policy.gate_enabled.get(PlanGateId.MATERIAL_UNIVERSE, False)
    axial_geometry_controlled = policy.mode is PlanLoopMode.CONTROLLED and policy.gate_enabled.get(PlanGateId.AXIAL_GEOMETRY, False) and policy.axial_geometry_review_mode == "controlled"
    if (placement_controlled or material_universe_controlled or axial_geometry_controlled) and task_order is None:
        # Patches that depend on Materials/Universes must wait until the
        # Material-Universe Gate is accepted, but Placement-owned patches must
        # still be generated before Axial-owned patches so the Placement gate
        # can stop/review at its real milestone boundary.
        order = _order_for_controlled_gate_barriers(
            order,
            placement_controlled=placement_controlled,
            material_universe_controlled=material_universe_controlled,
            axial_geometry_controlled=axial_geometry_controlled,
        )
    required = required_patch_types_for_state(state)
    advisory_enabled = policy.mode is PlanLoopMode.ADVISORY
    artifact_writer = PlanLoopArtifactWriter(plan_loop_output_dir, policy.artifact_subdir)

    def _persist_checkpoint(boundary: str) -> None:
        # Phase 8C Step 3B: persist a sanitized snapshot at accepted
        # boundaries.  No-op when no callback was supplied (legacy path).
        if checkpoint_callback is None:
            return
        checkpoint_callback(boundary, state)

    def _write_advisory_artifacts() -> dict[str, Any] | None:
        if not advisory_enabled:
            return None
        for stage in state.plan_loop_stages.values():
            if stage.status is PlanStageStatus.PENDING and (
                not stage.patch_types or all(_has_valid_patch(state, patch_type) for patch_type in stage.patch_types)
            ):
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["review_not_implemented"] = True
                state.add_event(
                    EVENT_GATE_TRANSITIONED, "plan closed-loop advisory gate skipped without review",
                    {"stage_id": stage.stage_id, "from": "pending", "to": "skipped", "review_not_implemented": True},
                )
        outcome = build_advisory_outcome(state, policy)
        summary = {
            "mode": policy.mode.value,
            "foundation_version": policy.contract_version,
            "enabled_gates": [gate.value for gate in canonical_gate_order() if policy.gate_enabled.get(gate, False)],
            "stage_count": len(state.plan_loop_stages),
            "additional_llm_calls": state.plan_loop_additional_llm_calls,
            "reviewer_calls": sum(int(item.get("reviewer_calls", 0)) for item in state.facts_review_history),
            "repair_calls": 0,
            "human_interrupts": 0,
            "workflow_behavior_changed": False,
        }
        facts_stage = state.plan_loop_stages.get("plan_gate_facts")
        placement_stage = state.plan_loop_stages.get("plan_gate_placement")
        facts_history = state.facts_review_history[-1] if state.facts_review_history else {}
        placement_history = state.placement_review_history[-1] if state.placement_review_history else {}
        summary.update({
            "facts_review_executed": bool(state.facts_review_history),
            "facts_stage_status": facts_stage.status.value if facts_stage else None,
            "facts_reviewer_calls": int(facts_history.get("reviewer_calls", 0)),
            "facts_review_schema_retries": int(facts_history.get("schema_retries", 0)),
            "facts_review_coverage_complete": bool(facts_history.get("coverage_complete", False)),
            "facts_revision_calls": sum(1 for item in state.facts_revision_history if "proposal" in item),
            "confirmed_fact_count": len(state.plan_confirmed_fact_records),
            "budget_remaining": max(0, policy.max_total_additional_llm_calls - state.plan_loop_additional_llm_calls),
            "placement_gate_applicable": bool(placement_stage and placement_stage.metadata.get("reason") != "not_applicable"),
            "placement_stage_status": placement_stage.status.value if placement_stage else None,
            "placement_reviewer_calls": int(placement_history.get("reviewer_calls", 0)),
            "placement_review_schema_retries": int(placement_history.get("schema_retries", 0)),
            "placement_revision_calls": len(state.placement_revision_history),
            "placement_dependency_requests": len(state.placement_dependency_requests),
        })
        writes = [
            artifact_writer.write_plan_loop_policy(policy),
            artifact_writer.write_plan_loop_state(state),
            artifact_writer.write_plan_loop_summary(summary),
            artifact_writer.write_gate_registry(),
        ]
        for path in writes:
            if path:
                state.plan_loop_artifacts.append(path)
                state.add_event(EVENT_CLOSED_LOOP_ARTIFACT_WRITTEN, "plan closed-loop artifact written", {"path": path})
            elif plan_loop_output_dir is not None:
                state.add_event(EVENT_CLOSED_LOOP_ARTIFACT_WARNING, "plan closed-loop artifact write failed", {})
        return outcome.model_dump(mode="json")

    if advisory_enabled:
        created = initialize_plan_loop_state(state, policy, required)
        state.add_event(
            EVENT_CLOSED_LOOP_INITIALIZED, "plan closed-loop foundation initialized",
            {"mode": policy.mode.value, "foundation_only": True},
        )
        for stage in created:
            state.add_event(EVENT_GATE_INITIALIZED, "plan closed-loop gate initialized", {"gate_id": stage.gate_id.value, "stage_id": stage.stage_id})
    elif policy.mode is PlanLoopMode.CONTROLLED:
        created = initialize_plan_loop_state(state, policy, required)
        state.add_event(EVENT_CLOSED_LOOP_INITIALIZED, "controlled facts gate initialized", {"mode": policy.mode.value})

    facts_stage_for_resume = state.plan_loop_stages.get("plan_gate_facts")
    if facts_stage_for_resume is not None and facts_stage_for_resume.status is PlanStageStatus.AWAITING_HUMAN and state.plan_human_answers:
        from .closed_loop.facts_human import consume_facts_answer
        consumed = []
        for question_id, question in list(state.plan_human_questions.items()):
            if any(record.question_id == question_id for record in state.plan_confirmed_fact_records.values()):
                continue
            answer = state.plan_human_answers.get(question_id)
            if answer is None:
                continue
            record = consume_facts_answer(question=question, answer=answer, round_index=facts_stage_for_resume.human_round_count + 1)
            state.plan_confirmed_fact_records[record.fact_id] = record
            # Keep the legacy namespaced view for old prompts while preserving
            # the typed RFC6901 path in a replayable record list.
            state.confirmed_facts.setdefault("plan_closed_loop", {}).setdefault("facts", {})[record.json_path] = record.value
            state.confirmed_facts.setdefault("plan_closed_loop_records", []).append(record.model_dump(mode="json"))
            consumed.append(record)
        if consumed:
            facts_stage_for_resume.human_round_count += 1
            transition_stage(facts_stage_for_resume, PlanStageStatus.REPAIRING)
            state.invalidate_patch_types(_expand_patch_repair_targets(["facts"]), reason="facts human confirmation", issues=[{"code": "facts.human_confirmation"}])
            artifact_writer._write("facts_human_answers.json", list(state.plan_human_answers.values()))
            artifact_writer._write("facts_confirmed_fact_records.json", consumed)
            state.add_event("planning.facts_human_answer_consumed", "typed facts confirmation consumed", {"count": len(consumed)})

    placement_stage_for_resume = state.plan_loop_stages.get("plan_gate_placement")
    if placement_stage_for_resume is not None and placement_stage_for_resume.status is PlanStageStatus.AWAITING_HUMAN and state.plan_human_answers:
        from .closed_loop.placement_human import consume_placement_answer
        consumed = []
        for question_id, question in list(state.plan_human_questions.items()):
            if question.gate_id is not PlanGateId.PLACEMENT or question_id not in state.plan_human_answers:
                continue
            if any(record.question_id == question_id for record in state.plan_confirmed_plan_fact_records.values()):
                continue
            record = consume_placement_answer(question=question, answer=state.plan_human_answers[question_id], round_index=placement_stage_for_resume.human_round_count + 1)
            state.plan_confirmed_plan_fact_records[record.fact_id] = record
            state.confirmed_facts.setdefault("plan_closed_loop", {}).setdefault("placement", {})[record.json_path] = record.value
            state.confirmed_facts.setdefault("plan_closed_loop_records", []).append(record.model_dump(mode="json"))
            consumed.append(record)
        if consumed:
            placement_stage_for_resume.human_round_count += 1
            transition_stage(placement_stage_for_resume, PlanStageStatus.REPAIRING)
            targets = _expand_patch_repair_targets(sorted({ptype for record in consumed for ptype in record.affected_patch_types}))
            state.invalidate_patch_types(targets, reason="placement human confirmation", issues=[{"code": "placement.human_confirmation"}])
            artifact_writer._write("placement_human_answers.json", [state.plan_human_answers[item.question_id] for item in consumed])
            artifact_writer._write("placement_confirmed_fact_records.json", consumed)
            state.add_event("planning.placement_human_answer_consumed", "typed placement confirmation consumed", {"count": len(consumed), "invalidated_patch_types": targets})

    def _facts_stage():
        return state.plan_loop_stages.get("plan_gate_facts")

    def _require_accepted_facts_gate(*, next_patch_type: str | None = None) -> IncrementalExecutionIssue | None:
        """Prevent a controlled run from resuming below a terminal Facts gate.

        A graph retry restores valid envelopes from ``PlanBuildState``.  Merely
        skipping a valid FactsPatch is unsafe when the persisted Facts stage is
        blocked (for example because the evidence source was not fully
        reviewable): it would let Materials and all downstream patches bypass
        the barrier that controlled mode promises.  A pending stage is the only
        resumable case with a valid facts envelope; run the real gate again in
        that case.  Terminal stages are never transitioned implicitly.
        """
        if (
            policy.mode is not PlanLoopMode.CONTROLLED
            or not policy.gate_enabled.get(PlanGateId.FACTS, False)
            or next_patch_type == "facts"
        ):
            return None
        stage = _facts_stage()
        if stage is not None and stage.status is PlanStageStatus.ACCEPTED:
            return None
        if (
            stage is not None
            and stage.status is PlanStageStatus.PENDING
            and _has_valid_patch(state, "facts")
        ):
            replay_issue = _run_facts_gate()
            if replay_issue is not None:
                return replay_issue
            if stage.status is PlanStageStatus.ACCEPTED:
                return None

        if stage is not None and stage.status is PlanStageStatus.AWAITING_HUMAN:
            return IncrementalExecutionIssue(
                code="planning.facts_awaiting_human",
                severity="error",
                message="controlled planning is awaiting a Facts Gate human confirmation",
                patch_type="facts",
            )
        status = stage.status.value if stage is not None else "uninitialized"
        return IncrementalExecutionIssue(
            code="planning.facts_gate_not_accepted",
            severity="error",
            message=f"controlled planning requires an accepted Facts Gate before {next_patch_type or 'assembly'} (current status={status})",
            patch_type="facts",
        )

    def _run_facts_gate() -> IncrementalExecutionIssue | None:
        if policy.mode is PlanLoopMode.OFF:
            return None
        stage = _facts_stage()
        if stage is None:
            return None
        if _reuse_frozen_accepted_gate(PlanGateId.FACTS):
            return None
        initial_decision_result: dict[str, Any] | None = None
        candidate_validation_rounds: list[dict[str, Any]] = []
        candidate_commit_result: dict[str, Any] = {
            "attempted": False,
            "committed": False,
        }

        def _write_facts_gate_result(
            terminal_reason: str,
            *,
            failure_code: str | None = None,
        ) -> None:
            payload = {
                "gate_id": "facts",
                "initial_decision": initial_decision_result,
                "candidate_validation": {
                    "attempted": bool(candidate_validation_rounds),
                    "rounds": candidate_validation_rounds,
                    "final": candidate_validation_rounds[-1] if candidate_validation_rounds else None,
                },
                "candidate_commit": candidate_commit_result,
                "final_gate_status": {
                    "status": stage.status.value,
                    "accepted": stage.status is PlanStageStatus.ACCEPTED,
                    "terminal_reason": terminal_reason,
                    "failure_code": failure_code,
                },
            }
            path = artifact_writer.write_facts_gate_result(payload)
            if path and path not in state.plan_loop_artifacts:
                state.plan_loop_artifacts.append(path)
        # Phase 8C Step 1: accepted_input_hash replay protection.
        # The Facts gate was the only gate that did not save
        # accepted_input_hash (audit defect).  When the input is unchanged
        # we short-circuit; when it changes we reopen the gate.
        from .closed_loop.facts_evidence import facts_gate_input_hash
        facts_input_hash = facts_gate_input_hash(state, policy=policy)
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") == facts_input_hash:
            return None
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") != facts_input_hash:
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event(
                "planning.facts_input_hash_changed",
                "facts accepted input hash changed; gate reopened",
                {"old": stage.metadata.get("accepted_input_hash"), "new": facts_input_hash},
            )
        if stage.status is PlanStageStatus.PENDING:
            transition_stage(stage, PlanStageStatus.PROPOSING)
        if stage.status is PlanStageStatus.PROPOSING:
            transition_stage(stage, PlanStageStatus.VALIDATING)
            stage.validation_count += 1
        transition_stage(stage, PlanStageStatus.REVIEWING)
        stage.review_count += 1
        budget = check_stage_budget(state, stage, policy)
        if budget is not None:
            transition_stage(stage, PlanStageStatus.BLOCKED)
            state.add_event("planning.closed_loop_budget_exhausted", "facts gate budget exhausted before review", {"budget": budget})
            _write_facts_gate_result("budget_exhausted", failure_code="planning.closed_loop.budget_exhausted")
            return IncrementalExecutionIssue(code="planning.closed_loop.budget_exhausted", severity="error", message=f"facts gate budget exhausted: {budget}", patch_type="facts")
        state.add_event("planning.facts_gate_started", "facts evidence review gate started", {})
        facts_env = next((item for item in state.patches.values() if item.patch_type == "facts" and item.status == "valid"), None)
        if facts_env is None:
            _write_facts_gate_result("missing_patch", failure_code="planning.facts_gate_missing_patch")
            return IncrementalExecutionIssue(code="planning.facts_gate_missing_patch", severity="error", message="facts patch unavailable for review", patch_type="facts")
        # Feature/Facts reconciliation happens before any independent reviewer.
        # It is deterministic and therefore cannot be bypassed by a fluent but
        # incomplete critic response.
        from .planning_scope import planning_feature_contract, build_canonical_task_plan
        from .closed_loop.facts_consistency import run_facts_consistency_preflight
        contract = planning_feature_contract(state.metadata.get("planning_mode_decision"))
        existing_types = [item.patch_type for item in state.get_valid_patches()]
        consistency = run_facts_consistency_preflight(
            feature_contract=contract, facts_patch=facts_env.content,
            confirmed_facts=state.confirmed_facts, existing_valid_patch_types=existing_types,
        )
        state.planning_feature_contract = contract
        state.resolved_planning_scope = consistency.scope
        state.metadata["planning_feature_contract"] = contract.model_dump(mode="json")
        state.metadata["resolved_planning_scope"] = consistency.scope.model_dump(mode="json")
        state.metadata["facts_consistency_issues"] = consistency.issues
        state.metadata["expected_patch_family"] = {"scope": consistency.scope.value, "required": "assembly_catalog+core_layout" if consistency.scope.value in {"multi_assembly_core", "full_core"} else "pin_map"}
        # Phase 8B Step 3: deterministic evidence↔Facts consistency check.
        # Compares evidence ledger claims against FactsPatch field values
        # so the gate can catch scope/insert/grid/fuel conflicts without
        # relying on the LLM reviewer.
        from .closed_loop.facts_evidence_consistency import check_facts_evidence_consistency
        evidence_ledger_dump = state.metadata.get("planning_evidence_ledger")
        evidence_consistency_findings: list[PlanReviewFinding] = []
        if evidence_ledger_dump:
            try:
                from openmc_agent.plan_investigation.evidence_ledger import PlanningEvidenceLedger
                _ledger = PlanningEvidenceLedger.model_validate(evidence_ledger_dump)
                _ev_result = check_facts_evidence_consistency(
                    facts_patch=facts_env.content,
                    evidence_claims=_ledger.claims,
                )
                state.metadata["facts_evidence_consistency"] = _ev_result.to_issue_dicts()
                for f in _ev_result.findings:
                    evidence_consistency_findings.append(
                        PlanReviewFinding(
                            gate_id=PlanGateId.FACTS,
                            code=f.code,
                            severity=PlanFindingSeverity.ERROR,
                            category=PlanFindingCategory.CROSS_PATCH_MISMATCH,
                            message=f.message,
                            affected_patch_types=["facts"],
                            affected_json_paths=[f.path] if f.path else [],
                            repairable_by_llm=f.repairable_by_llm,
                            requires_human=f.requires_human,
                            confidence=1.0,
                            metadata={
                                "deterministic": True,
                                "evidence_consistency": True,
                                "evidence_claim_ids": list(f.evidence_claim_ids),
                                "expected_value": f.expected_value,
                                "actual_value": f.actual_value,
                                "repair_kind": "evidence_backed_repair",
                            },
                        )
                    )
                if _ev_result.findings:
                    state.add_event(
                        "planning.facts_evidence_consistency_failed",
                        "evidence↔Facts consistency check found issues",
                        {"codes": [f.code for f in _ev_result.findings]},
                    )
            except Exception:
                state.add_event(
                    "planning.facts_evidence_consistency_error",
                    "evidence↔Facts consistency check raised; skipping",
                    {},
                )
        for filename, payload in (
            ("facts_feature_contract.json", contract),
            ("planning_scope_evidence.json", {"evidence": consistency.scope.evidence}),
            ("resolved_planning_scope.json", consistency.scope),
            ("facts_consistency_preflight.json", consistency),
        ):
            path = artifact_writer._write(filename, payload)
            if path: state.plan_loop_artifacts.append(path)
        state.add_event("planning.feature_contract_built", "planning feature contract built", {"hash": contract.contract_hash})
        if consistency.scope.status == "conflict":
            state.add_event("planning.scope_conflict_detected", "Facts scope conflicts with planning features", {"scope": consistency.scope.model_dump(mode="json")})
        state.add_event("planning.facts_consistency_preflight_started", "feature-to-Facts consistency preflight started", {})
        if consistency.issues:
            state.add_event("planning.facts_consistency_preflight_failed", "feature-to-Facts consistency preflight found blocking issues", {"codes": [item["code"] for item in consistency.issues]})
        else:
            state.add_event("planning.facts_consistency_preflight_passed", "feature-to-Facts consistency preflight passed", {})
        packs = build_facts_evidence_packs(
            requirement_text=requirement, facts_patch=facts_env.content,
            confirmed_facts=state.confirmed_facts, planning_metadata=state.metadata, policy=policy,
        )
        for index, pack in enumerate(packs):
            path = artifact_writer._write(f"facts_evidence_pack_{index:03d}.json", pack)
            if path:
                state.plan_loop_artifacts.append(path)
            prompt_path = artifact_writer.write_text(f"facts_review_prompt_{index:03d}.txt", build_facts_review_prompt(pack))
            if prompt_path:
                state.plan_loop_artifacts.append(prompt_path)
        state.add_event("planning.facts_evidence_built", "facts evidence packs built", {"pack_count": len(packs)})
        if len(requirement) > policy.max_facts_review_source_chars or any(pack.metadata.get("source_truncated") for pack in packs):
            code = "facts.review_source_too_large"
            state.add_event("planning.facts_review_failed", code, {})
            if policy.mode is PlanLoopMode.CONTROLLED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                _write_facts_gate_result("review_source_too_large", failure_code=code)
                return IncrementalExecutionIssue(code=code, severity="error", message="source exceeds facts review coverage budget", patch_type="facts")
        if plan_reviewer_client is None:
            state.add_event("planning.facts_reviewer_unavailable", "facts reviewer client unavailable", {})
            if policy.mode is PlanLoopMode.CONTROLLED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                _write_facts_gate_result("reviewer_unavailable", failure_code="planning.facts_reviewer_unavailable")
                return IncrementalExecutionIssue(code="planning.facts_reviewer_unavailable", severity="error", message="controlled facts review requires a reviewer", patch_type="facts")
            transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
            _write_facts_gate_result("reviewer_unavailable")
            return None
        state.add_event("planning.facts_review_started", "independent facts critic called", {"pack_count": len(packs)})
        review = run_facts_review(evidence_packs=packs, reviewer_client=plan_reviewer_client, state=state, policy=policy)
        for index, raw_output in enumerate(review.raw_outputs):
            path = artifact_writer._write(f"facts_review_raw_{index:03d}.json", {"raw": raw_output})
            if path:
                state.plan_loop_artifacts.append(path)
        for index, output in enumerate(review.outputs):
            path = artifact_writer._write(f"facts_review_normalized_{index:03d}.json", output)
            if path:
                state.plan_loop_artifacts.append(path)
        findings_path = artifact_writer._write("facts_review_findings.json", review.findings)
        if findings_path:
            state.plan_loop_artifacts.append(findings_path)
        state.facts_review_history.append(review.model_dump(mode="json"))
        consistency_findings = [
            PlanReviewFinding(
                gate_id=PlanGateId.FACTS, code=str(item["code"]),
                severity=PlanFindingSeverity.ERROR, category=PlanFindingCategory.CROSS_PATCH_MISMATCH,
                message=f"deterministic Facts contract violation: {item['code']}",
                affected_patch_types=["facts"], affected_json_paths=[str(item.get("path", "/"))],
                repairable_by_llm=bool(item.get("repairable_by_llm", True)),
                requires_human=bool(item.get("requires_human", False)), confidence=1.0,
                metadata={
                    "deterministic": True,
                    "evidence_hashes": [contract.contract_hash],
                    # Phase 8C Step 1: lossless metadata.
                    # The inline lift used to drop everything except code +
                    # message.  Carry the deterministic metadata through to
                    # the retry request builder so targeted repair can use
                    # json_path, expected_value, actual_value, slot_ids,
                    # source_claim_ids, source_span_ids, derivation_codes.
                    "expected_value": item.get("expected_value"),
                    "actual_value": item.get("actual_value"),
                    "slot_ids": list(item.get("slot_ids", []) or []),
                    "source_claim_ids": list(item.get("source_claim_ids", []) or []),
                    "source_span_ids": list(item.get("source_span_ids", []) or []),
                    "derivation_codes": list(item.get("derivation_codes", []) or []),
                    "repair_kind": str(item.get("repair_kind", "") or ""),
                },
            ) for item in consistency.issues
        ]
        # Phase 8C Step 2: lift the contract preflight issues (produced
        # after the skeleton merge) into PlanReviewFinding so the gate's
        # action-decision policy can route them to REVISE_CURRENT_PATCH.
        contract_preflight_issues = list(state.metadata.get("facts_contract_preflight_issues", []) or [])
        contract_preflight_findings = [
            PlanReviewFinding(
                gate_id=PlanGateId.FACTS, code=str(item.get("code", "facts_contract.unknown")),
                severity=PlanFindingSeverity.ERROR if item.get("severity") == "error" else PlanFindingSeverity.WARNING,
                category=PlanFindingCategory.CROSS_PATCH_MISMATCH,
                message=str(item.get("message", item.get("code", "facts_contract violation"))),
                affected_patch_types=["facts"],
                affected_json_paths=[str(item.get("path", "/"))],
                repairable_by_llm=True,
                requires_human=bool(item.get("requires_human", False)),
                confidence=1.0,
                metadata={
                    "deterministic": True,
                    "contract_preflight": True,
                    "evidence_hashes": [contract.contract_hash],
                    "expected_value": item.get("expected"),
                    "actual_value": item.get("actual"),
                    "slot_ids": list(item.get("slot_ids", []) or []),
                    "source_claim_ids": list(item.get("source_claim_ids", []) or []),
                    "derivation_codes": list(item.get("derivation_codes", []) or []),
                    "repair_kind": str(item.get("repair_kind", "contract_backed_repair") or "contract_backed_repair"),
                },
            )
            for item in contract_preflight_issues
            if item.get("severity") == "error"
        ]
        all_findings = consistency_findings + contract_preflight_findings + evidence_consistency_findings + list(review.findings)
        record_findings(state, stage, all_findings)
        if not review.ok:
            state.add_event("planning.facts_review_failed", review.error or review.failure_code or "facts_review.schema_invalid", {})
            if policy.mode is PlanLoopMode.CONTROLLED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                _write_facts_gate_result("initial_review_unusable", failure_code=review.failure_code or "facts_review.schema_invalid")
                return IncrementalExecutionIssue(code=review.failure_code or "facts_review.schema_invalid", severity="error", message="facts review output was unusable", patch_type="facts")
        review_deterministic_issues = list(consistency.issues) + list(contract_preflight_issues) + [f.model_dump(mode="json") for f in evidence_consistency_findings if hasattr(f, "model_dump")] + ([{"code": "facts_review.coverage_incomplete", "severity": "error", "blocking": True}]
                                       if not review.coverage_complete else [])
        actions = compute_allowed_actions(policy=policy, stage_state=stage, findings=all_findings, deterministic_issues=review_deterministic_issues, additional_llm_calls_used=state.plan_loop_additional_llm_calls)
        action = actions[0] if actions else PlanReviewAction.FAIL_CLOSED
        decision = PlanReviewDecision(
            decision_id=f"facts_decision_{len(state.plan_review_decisions):03d}", gate_id=PlanGateId.FACTS,
            action=action, target_patch_types=["facts"] if action in {PlanReviewAction.REVISE_CURRENT_PATCH, PlanReviewAction.RETRY_DEPENDENCY} else [],
            finding_ids=[item.finding_id for item in all_findings], rationale="deterministic facts-gate action policy",
            allowed_actions_snapshot=actions or [PlanReviewAction.FAIL_CLOSED], decided_by="deterministic",
        )
        record_decision(state, stage, decision)
        initial_decision_result = decision.model_dump(mode="json")
        decision_path = artifact_writer._write("facts_review_decision.json", decision)
        if decision_path:
            state.plan_loop_artifacts.append(decision_path)
        state.add_event("planning.facts_review_completed", "facts critic result normalized", {"finding_count": len(review.findings), "rejected_count": len(review.rejected)})
        if policy.mode is PlanLoopMode.ADVISORY:
            transition_stage(stage, PlanStageStatus.REVIEWED if review.ok and review.coverage_complete else PlanStageStatus.REVIEW_FAILED)
            state.add_event("planning.facts_review_advisory_completed", "facts review recorded without plan mutation", {"review_success": review.ok and review.coverage_complete})
            _write_facts_gate_result("advisory_review_completed")
            return None
        if action is PlanReviewAction.APPROVE:
            # Acceptance is bound to the reconciled scope and a post-Facts
            # task plan; subsequent executor iterations never reuse the
            # provisional feature-only order.
            state.canonical_task_plan = build_canonical_task_plan(
                scope=consistency.scope, contract=contract, facts_patch=facts_env.content,
                feature_order=list(_DEFAULT_ORDER),
            )
            state.metadata["canonical_task_plan"] = state.canonical_task_plan.model_dump(mode="json")
            path = artifact_writer._write("canonical_task_plan.json", state.canonical_task_plan)
            if path: state.plan_loop_artifacts.append(path)
            state.add_event("planning.canonical_task_plan_built", "canonical patch task plan built after Facts acceptance", {"plan_hash": state.canonical_task_plan.plan_hash, "scope": consistency.scope.value})
            transition_stage(stage, PlanStageStatus.ACCEPTED)
            stage.metadata["accepted_input_hash"] = facts_input_hash
            state.add_event("planning.facts_gate_accepted", "facts gate accepted", {})
            # Phase 8A Step 5: compile the GeometryComponentInventory
            # immediately after Facts accepts, so the Materials and
            # Universes patch generators can consume a deterministic
            # inventory rather than guessing from the requirement text.
            # Only fires when plan investigation is enabled; legacy
            # off-mode behaviour is unchanged.
            if _investigation_config_resolved is not None and not _investigation_config_resolved.is_off:
                inventory_status = _maybe_compile_geometry_inventory(
                    state=state,
                    facts_env=facts_env,
                    requirement=requirement,
                    artifact_writer=artifact_writer,
                )
                # Phase 8A Step 6 (P0-2): in controlled mode, an
                # inventory compilation failure must block the run.
                # Advisory mode logs the failure but keeps going so
                # the legacy prompt path is exercised.
                if not inventory_status.get("compiled"):
                    if _investigation_config_resolved.is_controlled:
                        transition_stage(stage, PlanStageStatus.BLOCKED)
                        _write_facts_gate_result(
                            "inventory_compilation_failed",
                            failure_code=inventory_status.get("failure_code") or "planning.inventory.compilation_failed",
                        )
                        return IncrementalExecutionIssue(
                            code=inventory_status.get("failure_code") or "planning.inventory.compilation_failed",
                            severity="error",
                            message=(
                                f"controlled mode: geometry inventory compilation failed: "
                                f"{inventory_status.get('error', 'unknown')}"
                            ),
                            patch_type="facts",
                        )
                    state.add_event(
                        "planning.inventory_compilation_failed_advisory",
                        "advisory mode: inventory compilation failed, continuing with legacy path",
                        {"error": inventory_status.get("error", "")[:160]},
                    )
            _write_facts_gate_result("initial_decision_approved")
            _persist_checkpoint("gate:facts")
            return None
        if action is PlanReviewAction.ASK_HUMAN:
            from .closed_loop.facts_human import build_facts_human_question
            transition_stage(stage, PlanStageStatus.AWAITING_HUMAN)
            for finding in review.findings:
                if finding.requires_human:
                    question = build_facts_human_question(finding)
                    if question.question_id not in state.plan_human_answers:
                        state.plan_human_questions[question.question_id] = question
            artifact_writer._write("facts_human_questions.json", list(state.plan_human_questions.values()))
            state.add_event("planning.facts_awaiting_human", "facts ambiguity requires typed confirmation", {"question_count": len(state.plan_human_questions)})
            _write_facts_gate_result("awaiting_human", failure_code="planning.facts_awaiting_human")
            return IncrementalExecutionIssue(code="planning.facts_awaiting_human", severity="error", message="facts gate awaiting human confirmation", patch_type="facts")
        if action is PlanReviewAction.REVISE_CURRENT_PATCH and plan_repair_client is not None:
            from .closed_loop.facts_revision import (
                MAX_FACTS_REVISION_CLOSURE_ROUNDS,
                allowed_paths_for_findings,
                evaluate_facts_revision,
                facts_revision_closure_fingerprint,
                normalize_facts_revision,
            )
            from .closed_loop.facts_revision_prompts import build_facts_revision_prompt
            from .closed_loop.models import FactsRevisionProposal
            from .closed_loop.review_io import StructuredReviewCallSpec, run_structured_review_call
            from openmc_agent.structured_output import canonical_payload_hash

            transition_stage(stage, PlanStageStatus.REPAIRING)
            candidate_facts = facts_env.content
            closure_findings = [item for item in all_findings if item.severity.value == "error"]
            latest_unresolved_findings = list(closure_findings)
            closure_failure_code = "planning.facts_revision.incomplete_closure"
            closure_rounds_completed = 0

            for closure_round in range(MAX_FACTS_REVISION_CLOSURE_ROUNDS):
                if closure_round:
                    transition_stage(stage, PlanStageStatus.REPAIRING)
                stage.repair_count += 1
                closure_rounds_completed = closure_round + 1
                issue_fingerprint = facts_revision_closure_fingerprint(closure_findings)
                stage.issue_fingerprint = issue_fingerprint
                if state.plan_loop_issue_attempts_by_fingerprint.get(issue_fingerprint, 0) >= policy.max_attempts_per_issue_fingerprint:
                    closure_failure_code = "planning.facts_revision.attempt_budget_exhausted"
                    break

                allowed_paths = allowed_paths_for_findings(closure_findings)
                revision_input = {
                    "facts_patch": candidate_facts,
                    "findings": [item.model_dump(mode="json") for item in closure_findings],
                    "evidence": [item.model_dump(mode="json") for pack in packs for item in pack.source_excerpts],
                    "allowed_paths": allowed_paths,
                    "confirmed_facts": state.confirmed_facts,
                }
                prompt = build_facts_revision_prompt(
                    facts_patch=candidate_facts,
                    findings=revision_input["findings"],
                    evidence=revision_input["evidence"],
                    allowed_paths=allowed_paths,
                    confirmed_facts=state.confirmed_facts,
                )
                prompt_path = artifact_writer.write_text(
                    f"facts_revision_prompt_{closure_round:03d}.txt", prompt,
                )
                if prompt_path:
                    state.plan_loop_artifacts.append(prompt_path)

                revision_call = run_structured_review_call(
                    client=plan_repair_client,
                    initial_prompt=prompt,
                    retry_prompt_builder=lambda raw, error: (
                        "Output only one JSON FactsRevisionProposal object. Do not emit prose, "
                        "reasoning, or markdown fences. Preserve the immutable input payload and "
                        "correct only the output schema.\n\n" + prompt
                    ),
                    output_model=FactsRevisionProposal,
                    call_spec=StructuredReviewCallSpec(
                        role_id="facts_revision",
                        gate_id=PlanGateId.FACTS,
                        schema_name="FactsRevisionProposal",
                        json_schema=FactsRevisionProposal.model_json_schema(),
                        artifact_prefix="facts_revision",
                        input_payload_hash=canonical_payload_hash(revision_input),
                    ),
                    state=state,
                    stage=stage,
                    policy=policy,
                )
                telemetry = {
                    "round": closure_round,
                    "issue_fingerprint": issue_fingerprint,
                    "input_payload_hash": revision_call.input_payload_hash,
                    "call_count": revision_call.call_count,
                    "schema_retry_count": revision_call.schema_retry_count,
                    "error_code": revision_call.error_code,
                    "attempts": [
                        item.model_dump(mode="json", exclude={"raw_text"})
                        for item in revision_call.attempts
                    ],
                }
                if not revision_call.ok or revision_call.parsed_output is None:
                    state.facts_revision_history.append(telemetry)
                    closure_failure_code = revision_call.error_code or "planning.facts_revision.schema_invalid"
                    break
                try:
                    proposal = normalize_facts_revision(revision_call.parsed_output)
                except Exception as exc:
                    telemetry["normalization_error"] = str(exc)
                    state.facts_revision_history.append(telemetry)
                    closure_failure_code = "planning.facts_revision.schema_invalid"
                    break

                evaluation = evaluate_facts_revision(
                    facts_patch=candidate_facts,
                    proposal=proposal,
                    findings=closure_findings,
                    confirmed_facts=state.confirmed_facts,
                    prior_candidate_hashes=state.plan_loop_candidate_hashes_by_fingerprint.get(issue_fingerprint, []),
                )
                proposal_path = artifact_writer._write(
                    f"facts_revision_proposal_{closure_round:03d}.json", proposal,
                )
                candidate_path = artifact_writer._write(
                    f"facts_revision_candidate_{closure_round:03d}.json", evaluation.candidate or {},
                )
                evaluation_path = artifact_writer._write(
                    f"facts_revision_evaluation_{closure_round:03d}.json", evaluation,
                )
                for path in (proposal_path, candidate_path, evaluation_path):
                    if path:
                        state.plan_loop_artifacts.append(path)
                if evaluation.candidate_hash:
                    duplicate = record_no_progress(state, stage, issue_fingerprint, evaluation.candidate_hash)
                    if duplicate:
                        evaluation.accepted = False
                        evaluation.reasons.append("facts_revision.no_progress")
                telemetry["proposal_id"] = proposal.proposal_id
                telemetry["evaluation"] = evaluation.model_dump(mode="json")
                state.facts_revision_history.append(telemetry)
                if not evaluation.accepted or evaluation.candidate is None:
                    closure_failure_code = (
                        "planning.facts_revision.no_progress"
                        if "facts_revision.no_progress" in evaluation.reasons
                        else "planning.facts_revision.incomplete_closure"
                    )
                    break

                # The candidate remains isolated until a full rereview clears
                # every error finding. A valid candidate with residual findings
                # becomes the next round's immutable input, never a durable patch.
                transition_stage(stage, PlanStageStatus.VALIDATING)
                stage.validation_count += 1
                transition_stage(stage, PlanStageStatus.REVIEWING)
                stage.review_count += 1
                rereview_packs = build_facts_evidence_packs(
                    requirement_text=requirement,
                    facts_patch=evaluation.candidate,
                    confirmed_facts=state.confirmed_facts,
                    planning_metadata=state.metadata,
                    policy=policy,
                )
                for index, pack in enumerate(rereview_packs):
                    pack_path = artifact_writer._write(
                        f"facts_rereview_evidence_pack_{closure_round:03d}_{index:03d}.json",
                        pack,
                    )
                    prompt_path = artifact_writer.write_text(
                        f"facts_rereview_prompt_{closure_round:03d}_{index:03d}.txt",
                        build_facts_review_prompt(pack),
                    )
                    for path in (pack_path, prompt_path):
                        if path:
                            state.plan_loop_artifacts.append(path)
                rereview = run_facts_review(
                    evidence_packs=rereview_packs,
                    reviewer_client=plan_reviewer_client,
                    state=state,
                    policy=policy,
                )
                for index, raw_output in enumerate(rereview.raw_outputs):
                    path = artifact_writer._write(
                        f"facts_rereview_raw_{closure_round:03d}_{index:03d}.json",
                        {"raw": raw_output},
                    )
                    if path:
                        state.plan_loop_artifacts.append(path)
                for index, output in enumerate(rereview.outputs):
                    path = artifact_writer._write(
                        f"facts_rereview_normalized_{closure_round:03d}_{index:03d}.json",
                        output,
                    )
                    if path:
                        state.plan_loop_artifacts.append(path)
                candidate_consistency = run_facts_consistency_preflight(
                    feature_contract=contract,
                    facts_patch=evaluation.candidate,
                    confirmed_facts=state.confirmed_facts,
                    existing_valid_patch_types=[item.patch_type for item in state.get_valid_patches()],
                )
                candidate_consistency_ok = not any(
                    item.get("severity") == "error" for item in candidate_consistency.issues
                )
                rereview_passing = (
                    rereview.ok
                    and rereview.coverage_complete
                    and not any(item.severity.value == "error" for item in rereview.findings)
                )
                rereview_schema_failed = (
                    not rereview.ok and "schema_invalid" in (rereview.failure_code or "")
                )
                accept_candidate = rereview_passing or (
                    rereview_schema_failed and candidate_consistency_ok
                )
                validation_result = {
                    "round": closure_round,
                    "proposal_id": proposal.proposal_id,
                    "candidate_hash": evaluation.candidate_hash,
                    "revision_evaluation_accepted": evaluation.accepted,
                    "candidate_consistency_ok": candidate_consistency_ok,
                    "rereview_ok": rereview.ok,
                    "rereview_coverage_complete": rereview.coverage_complete,
                    "rereview_error_finding_count": sum(
                        item.severity.value == "error" for item in rereview.findings
                    ),
                    "rereview_failure_code": rereview.failure_code,
                    "accepted_for_commit": accept_candidate,
                }
                candidate_validation_rounds.append(validation_result)
                for filename, payload in (
                    (f"facts_rereview_findings_{closure_round:03d}.json", rereview.findings),
                    (f"facts_rereview_candidate_consistency_{closure_round:03d}.json", candidate_consistency),
                    (f"facts_rereview_result_{closure_round:03d}.json", validation_result),
                ):
                    path = artifact_writer._write(filename, payload)
                    if path:
                        state.plan_loop_artifacts.append(path)
                state.facts_revision_history.append({
                    "round": closure_round,
                    "proposal_id": proposal.proposal_id,
                    "rereview": rereview.model_dump(mode="json", exclude={"raw_outputs"}),
                    "candidate_consistency_ok": candidate_consistency_ok,
                    "committed": accept_candidate,
                })
                record_findings(state, stage, rereview.findings)
                if accept_candidate:
                    candidate_commit_result.update({
                        "attempted": True,
                        "committed": False,
                        "round": closure_round,
                        "proposal_id": proposal.proposal_id,
                        "candidate_hash": evaluation.candidate_hash,
                    })
                    before_path = artifact_writer._write("facts_patch_before.json", facts_env.content)
                    after_path = artifact_writer._write("facts_patch_after.json", evaluation.candidate)
                    for path in (before_path, after_path):
                        if path:
                            state.plan_loop_artifacts.append(path)
                    state.invalidate_patch_types(
                        _expand_patch_repair_targets(["facts"]),
                        reason="accepted facts revision",
                        issues=[{"code": "facts_revision.accepted", "proposal_id": proposal.proposal_id}],
                    )
                    repaired = PlanPatchEnvelope(
                        patch_id=f"{facts_env.patch_id}_repair_{stage.repair_count}",
                        patch_type="facts",
                        content=evaluation.candidate,
                        source="repair",
                        status="valid",
                        metadata={
                            "proposal_id": proposal.proposal_id,
                            "candidate_hash": evaluation.candidate_hash,
                            "closure_round": closure_round,
                            "closure_input_hash": revision_call.input_payload_hash,
                        },
                    )
                    state.add_patch(repaired)
                    candidate_commit_result["committed"] = True
                    candidate_commit_result["patch_id"] = repaired.patch_id
                    transition_stage(stage, PlanStageStatus.ACCEPTED)
                    stage.metadata["accepted_input_hash"] = facts_gate_input_hash(state, policy=policy)
                    stage.metadata["facts_revision_closure"] = {
                        "rounds": closure_round + 1,
                        "final_candidate_hash": evaluation.candidate_hash,
                        "input_payload_hash": revision_call.input_payload_hash,
                    }
                    state.add_event(
                        "planning.facts_revision_accepted",
                        "facts revision atomically committed after clone rereview",
                        {
                            "proposal_id": proposal.proposal_id,
                            "candidate_hash": evaluation.candidate_hash,
                            "closure_round": closure_round,
                            "rereview_schema_failed": rereview_schema_failed,
                            "candidate_consistency_ok": candidate_consistency_ok,
                        },
                    )
                    if _investigation_config_resolved is not None and not _investigation_config_resolved.is_off:
                        inventory_status = _maybe_compile_geometry_inventory(
                            state=state,
                            facts_env=repaired,
                            requirement=requirement,
                            artifact_writer=artifact_writer,
                        )
                        if not inventory_status.get("compiled") and _investigation_config_resolved.is_controlled:
                            transition_stage(stage, PlanStageStatus.BLOCKED)
                            _write_facts_gate_result(
                                "candidate_committed_inventory_compilation_failed",
                                failure_code=inventory_status.get("failure_code") or "planning.inventory.compilation_failed",
                            )
                            return IncrementalExecutionIssue(
                                code=inventory_status.get("failure_code") or "planning.inventory.compilation_failed",
                                severity="error",
                                message=(
                                    "controlled mode: geometry inventory compilation failed after facts revision: "
                                    f"{inventory_status.get('error', 'unknown')}"
                                ),
                                patch_type="facts",
                            )
                    _write_facts_gate_result("candidate_committed_and_accepted")
                    _persist_checkpoint("gate:facts")
                    return None

                if not rereview.ok:
                    closure_failure_code = rereview.failure_code or "planning.facts_revision.rereview_unusable"
                    break
                remaining = [
                    item for item in rereview.findings if item.severity.value == "error"
                ]
                latest_unresolved_findings = list(remaining)
                if not remaining:
                    closure_failure_code = "planning.facts_revision.incomplete_closure"
                    break
                if any(item.requires_human or not item.repairable_by_llm for item in remaining):
                    closure_failure_code = "planning.facts_revision.unresolved_requires_human"
                    break
                next_fingerprint = facts_revision_closure_fingerprint(remaining)
                if next_fingerprint == issue_fingerprint:
                    stage.no_progress_count += 1
                    state.plan_loop_no_progress_events.append({
                        "stage_id": stage.stage_id,
                        "issue_fingerprint": issue_fingerprint,
                        "candidate_hash": evaluation.candidate_hash,
                        "count": stage.no_progress_count,
                    })
                    state.add_event(
                        "planning.no_progress_detected",
                        "facts rereview retained the same unresolved finding set",
                        {"stage_id": stage.stage_id, "issue_fingerprint": issue_fingerprint},
                    )
                    closure_failure_code = "planning.facts_revision.no_progress"
                    break
                candidate_facts = evaluation.candidate
                resolved_count = len(closure_findings) - len(remaining)
                closure_findings = remaining
                state.add_event(
                    "planning.facts_revision_closure_continues",
                    "facts revision cleared a subset of findings; continuing with remaining closure set",
                    {
                        "round": closure_round,
                        "resolved_count": resolved_count,
                        "remaining_count": len(remaining),
                        "candidate_hash": evaluation.candidate_hash,
                    },
                )

            transition_stage(stage, PlanStageStatus.BLOCKED)
            stage.metadata["facts_revision_closure"] = {
                "rounds": closure_rounds_completed,
                "failure_code": closure_failure_code,
                "unresolved_finding_ids": [item.finding_id for item in latest_unresolved_findings],
            }
            state.add_event(
                "planning.facts_revision_closure_blocked",
                "facts revision closure did not reach a clean rereview",
                {"failure_code": closure_failure_code, "rounds": closure_rounds_completed},
            )
            _write_facts_gate_result("revision_closure_blocked", failure_code=closure_failure_code)
            return IncrementalExecutionIssue(
                code=closure_failure_code,
                severity="error",
                message="facts revision closure did not pass clone/review acceptance",
                patch_type="facts",
            )
        transition_stage(stage, PlanStageStatus.BLOCKED)
        state.add_event("planning.facts_gate_blocked", "facts gate blocked by deterministic policy", {"action": action.value})
        _write_facts_gate_result("initial_decision_blocked", failure_code="planning.facts_gate_blocked")
        return IncrementalExecutionIssue(code="planning.facts_gate_blocked", severity="error", message=f"facts gate action={action.value}", patch_type="facts")

    repair_request = state.metadata.pop("plan_validation_repair", None)
    if isinstance(repair_request, dict):
        requested_targets = [
            str(patch_type)
            for patch_type in repair_request.get("target_patch_types", [])
            if isinstance(patch_type, str)
        ]
        repair_targets = [
            patch_type for patch_type in _expand_patch_repair_targets(requested_targets)
            if patch_type in order
        ]
        repair_issues = [
            issue for issue in repair_request.get("issues", [])
            if isinstance(issue, dict)
        ]
        if repair_targets:
            state.add_event(
                event_type=EVENT_PATCH_PLAN_VALIDATION_REPAIR_STARTED,
                message=(
                    "plan-level validation repair requested for patch type(s): "
                    f"{repair_targets}"
                ),
                data={
                    "requested_patch_types": requested_targets,
                    "expanded_patch_types": repair_targets,
                    "issue_codes": [
                        issue.get("code") for issue in repair_issues if issue.get("code")
                    ],
                },
            )
            state.invalidate_patch_types(
                repair_targets,
                reason="plan-level validation failure",
                issues=repair_issues,
            )
            required = sorted(set(required) | set(repair_targets), key=order.index)

    dependency_repair = state.metadata.pop("incremental_dependency_repair", None)
    if isinstance(dependency_repair, dict):
        missing_universe_ids = [
            str(universe_id)
            for universe_id in dependency_repair.get("missing_universe_ids", [])
            if isinstance(universe_id, str)
        ]
        if missing_universe_ids:
            requirement = (
                f"{requirement}\n\n"
                "=== Incremental dependency correction required ===\n"
                "The prior axial-layers patch references the following replacement "
                "universe IDs, but the universes patch did not define them:\n"
                + "\n".join(f"- {universe_id}" for universe_id in missing_universe_ids)
                + "\nRegenerate the universes patch to define every listed universe "
                "with input-supported concentric cells, then regenerate downstream "
                "patches. Preserve valid upstream facts and materials; do not replace "
                "a missing profile universe with an unrelated base universe."
            )

    def _sync_benchmark_from_facts() -> None:
        """Extract benchmark_id/variant from the valid FactsPatch content.

        This is benchmark-agnostic: the identification comes from the
        LLM-generated FactsPatch (which extracts it from the requirement
        text), NOT from hardcoded text matching in production code.
        """
        facts_env = next(
            (e for e in state.patches.values()
             if e.patch_type == "facts" and e.status == "valid"),
            None,
        )
        if facts_env is None:
            return
        content = facts_env.content
        bid = content.get("benchmark_id")
        var = content.get("selected_variant")
        if bid and not state.benchmark_id:
            state.benchmark_id = bid
        if var and not state.selected_variant:
            state.selected_variant = var

    def _build_failure_summary(
        pt: str,
        error_codes: list[str],
        attempt_count: int,
        *,
        no_progress: bool = False,
    ) -> dict[str, Any]:
        valid_types = sorted({
            e.patch_type for e in state.patches.values() if e.status == "valid"
        })
        invalid_types = sorted({
            e.patch_type for e in state.patches.values()
            if e.status != "valid"
        })
        if pt not in invalid_types:
            invalid_types = sorted(set(invalid_types) | {pt})
        return {
            "failed_patch_type": pt,
            "failed_stage": "patch_generation",
            "attempt_count": attempt_count,
            "issue_codes": error_codes,
            "valid_patch_types": valid_types,
            "invalid_patch_types": invalid_types,
            "next_recommended_action": (
                "stop_no_progress" if no_progress else "resume_from_failed_patch"
            ),
            "patch_generation_exhausted": (
                "patch_generation.max_attempts_exceeded" in error_codes
                or no_progress
            ),
            "no_progress": no_progress,
            "monolithic_fallback_attempted": False,
            "reference_patches_used": reference_patches_used,
            "actual_pin_counts": _latest_assembly_summary(state).get("actual_pin_counts", {}),
            "material_aliases_applied": _latest_assembly_summary(state).get("material_aliases_applied", {}),
            "reference_match_status": state.metadata.get(
                "reference_match_status",
                "off" if reference_patch_policy == "off" else "unavailable",
            ),
            "reference_path": state.metadata.get("reference_path"),
        }

    def _fail_reference_only(
        *,
        pt: str,
        code: str,
        message: str,
        detail_codes: list[str] | None = None,
    ) -> IncrementalExecutionResult:
        issue_codes = [code] + list(detail_codes or [])
        issues.append(IncrementalExecutionIssue(
            code=code,
            severity="error",
            message=message,
            patch_type=pt,
        ))
        state.metadata.setdefault("reference_match_status", "unavailable")
        state.add_event(
            event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
            message=message,
            data={"failed_patch_type": pt, "error_codes": issue_codes},
        )
        return IncrementalExecutionResult(
            ok=False,
            state=state,
            issues=issues,
            summary=_build_failure_summary(pt, issue_codes, 0),
            plan_loop_outcome=_write_advisory_artifacts(),
        )

    def _placement_stage():
        return state.plan_loop_stages.get("plan_gate_placement")

    def _material_universe_stage():
        return state.plan_loop_stages.get("plan_gate_material_universe")

    def _require_accepted_material_universe_gate(
        *,
        next_patch_type: str | None = None,
    ) -> IncrementalExecutionIssue | None:
        """Prevent controlled execution from resuming below a non-accepted MU gate."""
        if (
            policy.mode is not PlanLoopMode.CONTROLLED
            or not policy.gate_enabled.get(PlanGateId.MATERIAL_UNIVERSE, False)
            or next_patch_type in {"facts", "materials", "universes"}
        ):
            return None
        stage = _material_universe_stage()
        if stage is not None and stage.status is PlanStageStatus.ACCEPTED:
            return None
        if stage is not None and stage.status is PlanStageStatus.PENDING:
            replay_issue = _run_material_universe_gate(finalize_non_applicable=True)
            if replay_issue is not None:
                return replay_issue
            if stage.status is PlanStageStatus.ACCEPTED:
                return None
        status = stage.status.value if stage is not None else "uninitialized"
        return IncrementalExecutionIssue(
            code="planning.material_universe_gate_not_accepted",
            severity="error",
            message=(
                "controlled planning requires an accepted Material-Universe Gate "
                f"before {next_patch_type or 'assembly'} (current status={status})"
            ),
            patch_type="universes",
        )

    def _axial_geometry_stage():
        return state.plan_loop_stages.get("plan_gate_axial_geometry")

    def _assembled_plan_stage():
        return state.plan_loop_stages.get("plan_gate_assembled_plan")

    def _stop_after_gate_target() -> PlanGateId | None:
        raw = getattr(policy, "stop_after_gate", None)
        if raw is None:
            return None
        if isinstance(raw, PlanGateId):
            return raw
        try:
            return PlanGateId(str(raw))
        except ValueError:
            return None

    def _stage_for_gate(gate_id: PlanGateId):
        return state.plan_loop_stages.get(f"plan_gate_{gate_id.value}")

    def _target_seed_gate() -> PlanGateId | None:
        if not state.metadata.get("accepted_plan_build_state_seed"):
            return None
        raw = state.metadata.get("target_gate_seed_gate")
        if raw is None:
            return None
        try:
            return PlanGateId(str(raw))
        except ValueError:
            return None

    def _target_seed_freezes_gate(gate_id: PlanGateId) -> bool:
        """Accepted target-gate seeds reuse upstream accepted boundaries.

        A target review is not authorized to relitigate earlier gates; if an
        upstream accepted gate would otherwise see a hash drift or reviewer
        drift, that becomes a target-gate blocker/classification item rather
        than an upstream re-review inside this run.
        """
        target = _target_seed_gate()
        if target is None:
            return False
        order = [
            PlanGateId.FACTS,
            PlanGateId.MATERIAL_UNIVERSE,
            PlanGateId.PLACEMENT,
            PlanGateId.AXIAL_GEOMETRY,
            PlanGateId.ASSEMBLED_PLAN,
        ]
        if gate_id not in order or target not in order:
            return False
        return order.index(gate_id) < order.index(target)

    def _reuse_frozen_accepted_gate(gate_id: PlanGateId) -> bool:
        stage = _stage_for_gate(gate_id)
        if (
            stage is not None
            and stage.status is PlanStageStatus.ACCEPTED
            and _target_seed_freezes_gate(gate_id)
        ):
            state.add_event(
                f"planning.{gate_id.value}_gate_reused_from_target_seed",
                "accepted upstream gate reused by target-gate seed run",
                {"gate_id": gate_id.value, "target_gate": _target_seed_gate().value if _target_seed_gate() else None},
            )
            return True
        return False

    def _maybe_stop_after_gate(gate_id: PlanGateId) -> IncrementalExecutionResult | None:
        """Return a milestone success once the requested gate has accepted.

        ``--stop-after-gate`` is used for target live-review/canary runs.  The
        cumulative gate prefix still enforces upstream barriers, but once the
        requested gate accepts the executor must not continue into downstream
        patch generation because later patch failures would contaminate the
        gate milestone result.
        """
        if policy.mode is not PlanLoopMode.CONTROLLED:
            return None
        if _stop_after_gate_target() is not gate_id:
            return None
        stage = _stage_for_gate(gate_id)
        if stage is None or stage.status is not PlanStageStatus.ACCEPTED:
            return None
        valid_patch_types = sorted({
            e.patch_type for e in state.patches.values() if e.status == "valid"
        })
        state.metadata["stopped_after_gate"] = gate_id.value
        state.add_event(
            event_type=EVENT_INCREMENTAL_EXECUTION_COMPLETED,
            message=f"incremental planning stopped after {gate_id.value} gate accepted",
            data={
                "gate_id": gate_id.value,
                "valid_patch_count": len(state.get_valid_patches()),
                "valid_patch_types": valid_patch_types,
            },
        )
        return IncrementalExecutionResult(
            ok=True,
            state=state,
            assembled_plan=None,
            issues=issues,
            summary={
                "stopped_after_gate": gate_id.value,
                "valid_patch_count": len(state.get_valid_patches()),
                "valid_patch_types": valid_patch_types,
                "assembled": False,
                "reference_patches_used": reference_patches_used,
                "reference_match_status": state.metadata.get(
                    "reference_match_status",
                    "off" if reference_patch_policy == "off" else "unavailable",
                ),
                "reference_path": state.metadata.get("reference_path"),
                "actual_pin_counts": _latest_assembly_summary(state).get("actual_pin_counts", {}),
                "lattice_loading_count": _latest_assembly_summary(state).get("lattice_loading_count", 0),
                "material_aliases_applied": _latest_assembly_summary(state).get("material_aliases_applied", {}),
                "material_composition_policy": _latest_assembly_summary(state).get(
                    "material_composition_policy", "default"
                ),
                "material_composition_report_present": state.material_composition_report is not None,
            },
            plan_loop_outcome={
                "status": "stopped_after_gate",
                "active_gate_id": gate_id.value,
                "active_stage_id": stage.stage_id,
                "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                "detail": f"stopped after {gate_id.value} gate accepted",
            },
        )

    def _run_assembled_plan_gate(
        *, finalize_non_applicable: bool = False,
    ) -> IncrementalExecutionIssue | None:
        """Run the Final / Assembled Plan Gate after assembly succeeds."""
        if policy.mode is PlanLoopMode.OFF or not policy.gate_enabled.get(PlanGateId.ASSEMBLED_PLAN, False):
            return None
        if policy.assembled_plan_review_mode == "off":
            return None
        from .closed_loop.assembled_plan_evidence import (
            build_assembled_plan_evidence_pack,
            assembled_plan_gate_applicable,
            assembled_plan_gate_ready,
            assembled_plan_gate_input_hash,
        )
        from .closed_loop.assembled_plan_preflight import run_assembled_plan_preflight
        from .closed_loop.assembled_plan_reviewer import run_assembled_plan_review

        stage = _assembled_plan_stage()
        if stage is None:
            return None
        # Controlled barrier: all upstream gates must be accepted.
        if policy.mode is PlanLoopMode.CONTROLLED:
            if stage.status is PlanStageStatus.BLOCKED:
                return None
            for gate_key, stage_key, label in [
                (PlanGateId.FACTS, "plan_gate_facts", "Facts"),
                (PlanGateId.MATERIAL_UNIVERSE, "plan_gate_material_universe", "Material-Universe"),
                (PlanGateId.PLACEMENT, "plan_gate_placement", "Placement"),
                (PlanGateId.AXIAL_GEOMETRY, "plan_gate_axial_geometry", "Axial-Geometry"),
            ]:
                if not policy.gate_enabled.get(gate_key, False):
                    continue
                upstream = state.plan_loop_stages.get(stage_key)
                if upstream is not None and upstream.status is not PlanStageStatus.ACCEPTED:
                    transition_stage(stage, PlanStageStatus.BLOCKED)
                    return IncrementalExecutionIssue(code=f"planning.assembled_plan_requires_accepted_{gate_key.value}", severity="error", message=f"controlled assembled-plan gate requires accepted {label} Gate", patch_type="facts")
        # Build SimulationPlan from assembled_plan dict.
        plan_obj = None
        if state.assembled_plan is not None:
            try:
                from openmc_agent.schemas import SimulationPlan
                plan_obj = SimulationPlan.model_validate(state.assembled_plan) if isinstance(state.assembled_plan, dict) else state.assembled_plan
            except Exception:
                plan_obj = None
        if plan_obj is None:
            if not assembled_plan_gate_applicable(state):
                if finalize_non_applicable and stage.status is PlanStageStatus.PENDING:
                    transition_stage(stage, PlanStageStatus.SKIPPED)
                    stage.metadata["reason"] = "not_applicable"
                    state.add_event("planning.assembled_plan_gate_not_applicable", "assembled-plan gate not applicable", {})
                return None
            return None
        if not assembled_plan_gate_ready(state):
            return None
        if stage.status is PlanStageStatus.SKIPPED and stage.metadata.get("reason") == "not_applicable":
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.metadata.pop("reason", None)
            state.add_event("planning.assembled_plan_gate_reopened", "stale not_applicable assembled-plan stage reopened", {})
        input_hash = assembled_plan_gate_input_hash(state, policy=policy)
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") == input_hash:
            return None
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") != input_hash:
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event("planning.assembled_plan_input_hash_changed", "assembled-plan accepted input hash changed; gate reopened", {})
        state.add_event("planning.assembled_plan_preflight_started", "assembled-plan deterministic preflight started", {})
        preflight = run_assembled_plan_preflight(state=state, policy=policy, plan=plan_obj)
        artifact_writer._write("assembled_plan_preflight.json", preflight)
        if preflight.binding_view is not None:
            artifact_writer._write("assembled_plan_binding_view.json", preflight.binding_view)
        state.add_event("planning.assembled_plan_preflight_completed", "assembled-plan deterministic preflight completed", {"issue_count": len(preflight.issues), "blocking": sum(1 for i in preflight.issues if i.get("severity") == "error")})
        pack = build_assembled_plan_evidence_pack(state=state, policy=policy, plan=plan_obj, deterministic_issues=preflight.issues)
        artifact_writer._write("assembled_plan_evidence_pack.json", pack)
        artifact_writer._write("assembled_plan_contract_matrix.json", pack.contract_matrix)
        enter_review_cycle(stage, state)
        if policy.mode is PlanLoopMode.ADVISORY:
            if plan_reviewer_client is not None:
                review = run_assembled_plan_review(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
                state.facts_review_history.append({"assembled_plan_review": review.model_dump(mode="json")})
                if review.coverage_complete and not review.failure_code:
                    transition_stage(stage, PlanStageStatus.REVIEWED)
                    state.add_event("planning.assembled_plan_gate_reviewed", "assembled-plan gate reviewed without mutation", {})
                else:
                    transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                    state.add_event("planning.assembled_plan_review_failed", "assembled-plan advisory review failed", {"failure_code": review.failure_code})
            else:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["review_not_implemented"] = True
            return None
        # Controlled mode.
        all_findings: list[Any] = []
        if plan_reviewer_client is not None:
            review = run_assembled_plan_review(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
            all_findings = list(review.findings)
            state.facts_review_history.append({"assembled_plan_review": review.model_dump(mode="json")})
            if review.failure_code and not review.coverage_complete:
                transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                state.add_event("planning.assembled_plan_review_failed", "assembled-plan review failed coverage", {"failure_code": review.failure_code})
                return IncrementalExecutionIssue(code="planning.assembled_plan_review_failed", severity="error", message=f"assembled-plan review failed: {review.failure_code}", patch_type="facts")
        from .closed_loop.models import PlanFindingSeverity as _Sev, PlanReviewFinding as _Finding
        det_findings = [_Finding(gate_id=PlanGateId.ASSEMBLED_PLAN, code=str(item["code"]), severity=_Sev(item.get("severity", "error")), category="cross_patch_mismatch", message=str(item.get("message", "")), confidence=1.0, affected_patch_types=["facts", "materials", "universes", "axial_layers", "axial_overlays"]) for item in preflight.issues]
        all_findings = list({f.finding_id: f for f in (det_findings + all_findings)}.values())
        for finding in all_findings:
            state.plan_review_findings[finding.finding_id] = finding
            if finding.finding_id not in stage.finding_ids:
                stage.finding_ids.append(finding.finding_id)
        error_findings = [f for f in all_findings if f.severity is _Sev.ERROR]
        if not error_findings:
            transition_stage(stage, PlanStageStatus.ACCEPTED)
            stage.metadata["accepted_input_hash"] = input_hash
            state.add_event("planning.assembled_plan_gate_accepted", "assembled-plan gate accepted", {"input_hash": input_hash})
            _persist_checkpoint("gate:assembled_plan")
            return None
        human_required = any(f.requires_human for f in error_findings)
        if human_required and policy.enable_human_gate:
            transition_stage(stage, PlanStageStatus.AWAITING_HUMAN)
            state.add_event("planning.assembled_plan_human_question_created", "assembled-plan ambiguity requires typed confirmation", {})
            return IncrementalExecutionIssue(code="planning.assembled_plan_awaiting_human", severity="error", message="assembled-plan gate awaiting human confirmation", patch_type="facts")
        from .closed_loop.retry_controller import normalize_retry_request
        from .closed_loop.retry_models import RetryTriggerOrigin
        for finding in error_findings:
            typed = normalize_retry_request(
                {"code": finding.code, "issue_codes": [finding.code], "required_ids": finding.metadata.get("required_ids", []), "reason": finding.message},
                state=state, origin=RetryTriggerOrigin.ASSEMBLED_PLAN_GATE,
            )
            if typed is not None:
                artifact_writer._write(f"assembled_plan_retry_request_{typed.request_id[:12]}.json", typed)
                state.add_event("planning.assembled_plan_retry_requested", "assembled-plan blocking finding routed to Phase-3B retry", {"request_id": typed.request_id, "owner_patch_types": typed.owner_patch_types})
        transition_stage(stage, PlanStageStatus.BLOCKED)
        return IncrementalExecutionIssue(code="planning.assembled_plan_gate_blocked", severity="error", message=f"assembled-plan gate blocked by {len(error_findings)} finding(s)", patch_type="facts")

    def _run_axial_geometry_gate(
        *, finalize_non_applicable: bool = False,
    ) -> IncrementalExecutionIssue | None:
        """Run the Axial-Geometry Gate once axial patches are valid."""
        if policy.mode is PlanLoopMode.OFF or not policy.gate_enabled.get(PlanGateId.AXIAL_GEOMETRY, False):
            return None
        if policy.axial_geometry_review_mode == "off":
            return None
        from .closed_loop.axial_geometry_evidence import (
            build_axial_geometry_evidence_pack,
            axial_geometry_gate_applicable,
            axial_geometry_gate_ready,
            axial_geometry_gate_input_hash,
        )
        from .closed_loop.axial_geometry_preflight import run_axial_geometry_preflight
        from .closed_loop.axial_geometry_reviewer import run_axial_geometry_review

        stage = _axial_geometry_stage()
        if stage is None:
            return None
        if _reuse_frozen_accepted_gate(PlanGateId.AXIAL_GEOMETRY):
            return None
        # Controlled barrier: Facts, Material-Universe, and Placement must be accepted first.
        if policy.mode is PlanLoopMode.CONTROLLED:
            if stage.status is PlanStageStatus.BLOCKED:
                return None
            facts_stage = _facts_stage()
            if facts_stage is None or facts_stage.status is not PlanStageStatus.ACCEPTED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.axial_geometry_requires_accepted_facts", severity="error", message="controlled axial-geometry gate requires accepted Facts Gate", patch_type="axial_layers")
            # Upstream gates only become *ready* once their patches have
            # been generated.  When PENDING (patches not yet available) we
            # DEFER (return None).  Once ready but not accepted, it is a
            # real blocker and we transition to BLOCKED.
            from .closed_loop.material_universe_evidence import (
                material_universe_gate_applicable,
                material_universe_gate_ready,
            )
            from .closed_loop.placement_evidence import (
                placement_gate_applicable,
                placement_gate_ready,
            )
            mu_stage = _material_universe_stage()
            if (
                mu_stage is not None
                and material_universe_gate_applicable(state)
                and material_universe_gate_ready(state)
                and mu_stage.status is not PlanStageStatus.ACCEPTED
            ):
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.axial_geometry_requires_accepted_material_universe", severity="error", message="controlled axial-geometry gate requires accepted Material-Universe Gate", patch_type="axial_layers")
            placement_stage = _placement_stage()
            if (
                placement_stage is not None
                and placement_gate_applicable(state)
                and placement_gate_ready(state)
                and placement_stage.status is not PlanStageStatus.ACCEPTED
            ):
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.axial_geometry_requires_accepted_placement", severity="error", message="controlled axial-geometry gate requires accepted Placement Gate", patch_type="axial_layers")
        applicable = axial_geometry_gate_applicable(state)
        state.add_event("planning.axial_geometry_gate_applicability_checked", "axial-geometry gate applicability checked", {"applicable": applicable})
        if not applicable:
            if finalize_non_applicable and stage.status is PlanStageStatus.PENDING:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["reason"] = "not_applicable"
                state.add_event("planning.axial_geometry_gate_not_applicable", "axial-geometry gate not applicable for this task plan", {})
            return None
        if stage.status is PlanStageStatus.SKIPPED and stage.metadata.get("reason") == "not_applicable":
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.metadata.pop("reason", None)
            state.add_event("planning.axial_geometry_gate_reopened", "stale not_applicable axial-geometry stage reopened", {})
        if not axial_geometry_gate_ready(state):
            return None
        input_hash = axial_geometry_gate_input_hash(state, policy=policy)
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") == input_hash:
            return None
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") != input_hash:
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event("planning.axial_geometry_input_hash_changed", "axial-geometry accepted input hash changed; gate reopened", {"old": stage.metadata.get("accepted_input_hash"), "new": input_hash})
        state.add_event("planning.axial_geometry_preflight_started", "axial-geometry deterministic preflight started", {})
        preflight = run_axial_geometry_preflight(state=state, policy=policy)
        artifact_writer._write("axial_geometry_preflight.json", preflight)
        artifact_writer._write("axial_geometry_binding_view.json", preflight.binding_view)
        state.add_event("planning.axial_geometry_preflight_completed", "axial-geometry deterministic preflight completed", {"issue_count": len(preflight.issues), "blocking": sum(1 for i in preflight.issues if i.get("severity") == "error")})
        pack = build_axial_geometry_evidence_pack(state=state, policy=policy, deterministic_issues=preflight.issues)
        artifact_writer._write("axial_geometry_evidence_pack.json", pack)
        artifact_writer._write("axial_geometry_contract_matrix.json", pack.contract_matrix)
        enter_review_cycle(stage, state)
        if policy.mode is PlanLoopMode.ADVISORY:
            if plan_reviewer_client is not None:
                review = run_axial_geometry_review(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
                state.facts_review_history.append({"axial_geometry_review": review.model_dump(mode="json")})
                if review.coverage_complete and not review.failure_code:
                    transition_stage(stage, PlanStageStatus.REVIEWED)
                    state.add_event("planning.axial_geometry_gate_reviewed", "axial-geometry gate reviewed without mutation", {})
                else:
                    transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                    state.add_event("planning.axial_geometry_review_failed", "axial-geometry advisory review failed", {"failure_code": review.failure_code})
            else:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["review_not_implemented"] = True
            return None
        # Controlled mode.
        all_findings: list[Any] = []
        if plan_reviewer_client is not None:
            review = run_axial_geometry_review(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
            all_findings = list(review.findings)
            state.facts_review_history.append({"axial_geometry_review": review.model_dump(mode="json")})
            if review.failure_code and not review.coverage_complete:
                transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                state.add_event("planning.axial_geometry_review_failed", "axial-geometry review failed coverage", {"failure_code": review.failure_code})
                return IncrementalExecutionIssue(code="planning.axial_geometry_review_failed", severity="error", message=f"axial-geometry review failed: {review.failure_code}", patch_type="axial_layers")
        from .closed_loop.models import PlanFindingSeverity as _Sev, PlanReviewFinding as _Finding
        det_findings = [_Finding(gate_id=PlanGateId.AXIAL_GEOMETRY, code=str(item["code"]), severity=_Sev(item.get("severity", "error")), category="cross_patch_mismatch", message=str(item.get("message", "")), confidence=1.0, affected_patch_types=[item.get("owner_patch_type", "axial_layers")] if item.get("owner_patch_type") else ["axial_layers", "axial_overlays"]) for item in preflight.issues]
        all_findings = list({f.finding_id: f for f in (det_findings + all_findings)}.values())
        for finding in all_findings:
            state.plan_review_findings[finding.finding_id] = finding
            if finding.finding_id not in stage.finding_ids:
                stage.finding_ids.append(finding.finding_id)
        error_findings = [f for f in all_findings if f.severity is _Sev.ERROR]
        if not error_findings:
            transition_stage(stage, PlanStageStatus.ACCEPTED)
            stage.metadata["accepted_input_hash"] = input_hash
            state.add_event("planning.axial_geometry_gate_accepted", "axial-geometry gate accepted", {"input_hash": input_hash})
            _persist_checkpoint("gate:axial_geometry")
            return None
        human_required = any(f.requires_human for f in error_findings)
        if human_required and policy.enable_human_gate:
            transition_stage(stage, PlanStageStatus.AWAITING_HUMAN)
            state.add_event("planning.axial_geometry_human_question_created", "axial-geometry ambiguity requires typed confirmation", {})
            return IncrementalExecutionIssue(code="planning.axial_geometry_awaiting_human", severity="error", message="axial-geometry gate awaiting human confirmation", patch_type="axial_layers")
        from .closed_loop.retry_controller import normalize_retry_request
        from .closed_loop.retry_models import RetryTriggerOrigin
        for finding in error_findings:
            typed = normalize_retry_request(
                {"code": finding.code, "issue_codes": [finding.code], "required_ids": finding.metadata.get("required_ids", []), "reason": finding.message, "layer_id": finding.metadata.get("layer_id"), "overlay_id": finding.metadata.get("overlay_id"), "loading_id": finding.metadata.get("loading_id"), "profile_id": finding.metadata.get("profile_id")},
                state=state, origin=RetryTriggerOrigin.AXIAL_GEOMETRY_GATE,
            )
            if typed is not None:
                artifact_writer._write(f"axial_geometry_retry_request_{typed.request_id[:12]}.json", typed)
                state.add_event("planning.axial_geometry_retry_requested", "axial-geometry blocking finding routed to Phase-3B retry", {"request_id": typed.request_id, "owner_patch_types": typed.owner_patch_types})
        transition_stage(stage, PlanStageStatus.BLOCKED)
        return IncrementalExecutionIssue(code="planning.axial_geometry_gate_blocked", severity="error", message=f"axial-geometry gate blocked by {len(error_findings)} finding(s)", patch_type="axial_layers")

    def _run_material_universe_gate(
        *, finalize_non_applicable: bool = False,
    ) -> IncrementalExecutionIssue | None:
        """Run the Material-Universe Gate once Materials and Universes are valid."""
        if policy.mode is PlanLoopMode.OFF or not policy.gate_enabled.get(PlanGateId.MATERIAL_UNIVERSE, False):
            return None
        if policy.material_universe_review_mode == "off":
            return None
        from .closed_loop.material_universe_evidence import (
            build_material_universe_evidence_pack,
            material_universe_gate_applicable,
            material_universe_gate_input_hash,
            material_universe_gate_ready,
        )
        from .closed_loop.material_universe_preflight import run_material_universe_preflight
        from .closed_loop.material_universe_reviewer import run_material_universe_review
        if getattr(policy, "material_universe_review_split", False):
            from .closed_loop.material_universe_review_split import run_material_universe_review_split
        else:
            run_material_universe_review_split = None

        stage = _material_universe_stage()
        if stage is None:
            return None
        if _reuse_frozen_accepted_gate(PlanGateId.MATERIAL_UNIVERSE):
            return None
        if guard_blocked_stage(stage):
            return None
        if stage.status is PlanStageStatus.BLOCKED:
            invalidated_by = list(stage.metadata.pop("invalidated_by_patch_types"))
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.updated_at = None
            state.add_event(
                "planning.material_universe_gate_reopened",
                "material-universe owner patch changed after Phase-3B retry",
                {"patch_types": invalidated_by},
            )
        # Controlled barrier: Facts must be accepted first.
        if policy.mode is PlanLoopMode.CONTROLLED:
            facts_stage = _facts_stage()
            if facts_stage is None or facts_stage.status is not PlanStageStatus.ACCEPTED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.material_universe_requires_accepted_facts", severity="error", message="controlled material-universe gate requires accepted Facts Gate", patch_type="materials")
        applicable = material_universe_gate_applicable(state)
        state.add_event("planning.material_universe_gate_applicability_checked", "material-universe gate applicability checked", {"applicable": applicable})
        if not applicable:
            if finalize_non_applicable and stage.status is PlanStageStatus.PENDING:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["reason"] = "not_applicable"
                state.add_event("planning.material_universe_gate_not_applicable", "material-universe gate not applicable for this task plan", {})
            return None
        # Reopen a stale not_applicable checkpoint if inputs became applicable.
        if stage.status is PlanStageStatus.SKIPPED and stage.metadata.get("reason") == "not_applicable":
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.metadata.pop("reason", None)
            state.add_event("planning.material_universe_gate_reopened", "stale not_applicable material-universe stage reopened", {})
        if not material_universe_gate_ready(state):
            return None
        # Skip if already accepted with same input hash.
        input_hash = material_universe_gate_input_hash(state, policy=policy)
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") == input_hash:
            return None
        # Input hash changed → reopen accepted gate.
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") != input_hash:
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event("planning.material_universe_input_hash_changed", "material-universe accepted input hash changed; gate reopened", {"old": stage.metadata.get("accepted_input_hash"), "new": input_hash})
        # Build species report (reuse the executor's resolver if available).
        species_report: dict[str, Any] = {}
        materials_env = next((item for item in state.patches.values() if item.patch_type == "materials" and item.status == "valid"), None)
        if materials_env is not None:
            try:
                from openmc_agent.material_species import resolve_material_species_report
                species_report = resolve_material_species_report(materials_env.content) or {}
            except Exception:
                species_report = {}
        state.add_event("planning.material_universe_preflight_started", "material-universe deterministic preflight started", {})
        preflight = run_material_universe_preflight(state=state, policy=policy, species_report=species_report)
        artifact_writer._write("material_universe_preflight.json", preflight)
        artifact_writer._write("material_universe_binding_view.json", preflight.binding_view)
        artifact_writer._write("material_universe_contract_matrix.json", preflight.binding_view and build_material_universe_evidence_pack(state=state, policy=policy, species_report=species_report, deterministic_issues=preflight.issues).contract_matrix)
        state.add_event("planning.material_universe_preflight_completed", "material-universe deterministic preflight completed", {"issue_count": len(preflight.issues), "blocking": sum(1 for i in preflight.issues if i.get("severity") == "error")})
        # Phase 8A Step 5: run the inventory-driven preflight alongside
        # the existing deterministic preflight.  When a geometry
        # component inventory is present (compiled after Facts Gate
        # accepted), its findings merge into the gate's finding ledger.
        # When no inventory is present (off mode), this is a no-op.
        inventory_preflight_issues = _maybe_run_inventory_preflight(
            state=state,
            materials_env=materials_env,
            universes_env=next((item for item in state.patches.values() if item.patch_type == "universes" and item.status == "valid"), None),
            artifact_writer=artifact_writer,
            controlled_mode=(policy.mode is PlanLoopMode.CONTROLLED),
        )
        # Deterministic contract blocking: when the preflight has
        # blocking errors (material-reference, profile-coverage, variant,
        # localized-insert), skip the reviewer but continue through the
        # deterministic finding and owner-retry routing below.  This avoids
        # wasting LLM calls without losing targeted Materials/Universes
        # regeneration.
        _blocking_preflight = [i for i in preflight.issues if i.get("severity") == "error"]
        # Build evidence pack (needed for review even in advisory).
        pack = build_material_universe_evidence_pack(state=state, policy=policy, species_report=species_report, deterministic_issues=preflight.issues)
        artifact_writer._write("material_universe_evidence_pack.json", pack)
        if not (_blocking_preflight and policy.mode is PlanLoopMode.CONTROLLED):
            enter_review_cycle(stage, state)
        # In advisory mode, run the critic if a reviewer is available; never mutate.
        if policy.mode is PlanLoopMode.ADVISORY:
            if plan_reviewer_client is not None:
                review = (run_material_universe_review_split or run_material_universe_review)(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
                state.facts_review_history.append({"material_universe_review": review.model_dump(mode="json")})
                if review.coverage_complete and not review.failure_code:
                    transition_stage(stage, PlanStageStatus.REVIEWED)
                    state.add_event("planning.material_universe_gate_reviewed", "material-universe gate reviewed without mutation", {})
                else:
                    transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                    state.add_event("planning.material_universe_review_failed", "material-universe advisory review failed", {"failure_code": review.failure_code})
            else:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["review_not_implemented"] = True
            return None
        # Controlled mode.
        # If deterministic preflight has no blocking issues and no reviewer, accept.
        all_findings: list[Any] = []
        if _blocking_preflight:
            state.add_event(
                "planning.material_universe_contract_preflight_failed",
                "material-universe gate blocked by deterministic preflight errors; reviewer skipped",
                {
                    "blocking_error_count": len(_blocking_preflight),
                    "error_codes": sorted({i.get("code", "?") for i in _blocking_preflight}),
                },
            )
        elif plan_reviewer_client is not None:
            review = (run_material_universe_review_split or run_material_universe_review)(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
            all_findings = list(review.findings)
            state.facts_review_history.append({"material_universe_review": review.model_dump(mode="json")})
            if review.failure_code and not review.coverage_complete:
                transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                state.add_event("planning.material_universe_review_failed", "material-universe review failed coverage", {"failure_code": review.failure_code})
                return IncrementalExecutionIssue(code="planning.material_universe_review_failed", severity="error", message=f"material-universe review failed: {review.failure_code}", patch_type="materials")
        # Combine deterministic + critic findings.
        from .closed_loop.models import PlanFindingSeverity as _Sev, PlanReviewFinding as _Finding
        det_findings = [_Finding(
            gate_id=PlanGateId.MATERIAL_UNIVERSE,
            code=str(item["code"]), severity=_Sev(item.get("severity", "error")),
            category="cross_patch_mismatch", message=str(item.get("message", "")),
            confidence=1.0,
            affected_patch_types=[item.get("owner_patch_type", "materials")] if item.get("owner_patch_type") else ["materials", "universes"],
            affected_json_paths=list(item.get("affected_json_paths", []) or []),
            metadata={
                key: item[key]
                for key in (
                    "required_ids",
                    "requirement_id",
                    "universe_id",
                    "cell_id",
                    "material_id",
                    "expected_variant_id",
                    "actual_variant_id",
                )
                if key in item
            },
        ) for item in preflight.issues]
        # Phase 8A Step 5: append inventory-driven preflight findings.
        inventory_det_findings = [_Finding(gate_id=PlanGateId.MATERIAL_UNIVERSE, code=str(item["code"]), severity=_Sev(item.get("severity", "error")), category="cross_patch_mismatch", message=str(item.get("message", "")), confidence=1.0, affected_patch_types=[item.get("owner_patch_type", "materials")] if item.get("owner_patch_type") else ["materials", "universes"]) for item in inventory_preflight_issues]
        det_findings = det_findings + inventory_det_findings
        all_findings = list({f.finding_id: f for f in (det_findings + all_findings)}.values())
        for finding in all_findings:
            state.plan_review_findings[finding.finding_id] = finding
            if finding.finding_id not in stage.finding_ids:
                stage.finding_ids.append(finding.finding_id)
        error_findings = [f for f in all_findings if f.severity is _Sev.ERROR]
        if not error_findings:
            transition_stage(stage, PlanStageStatus.ACCEPTED)
            stage.metadata["accepted_input_hash"] = input_hash
            state.add_event("planning.material_universe_gate_accepted", "material-universe gate accepted", {"input_hash": input_hash})
            _persist_checkpoint("gate:material_universe")
            return None
        # Route blocking findings through Phase-3B retry.
        human_required = any(f.requires_human for f in error_findings)
        if human_required and policy.enable_human_gate:
            transition_stage(stage, PlanStageStatus.AWAITING_HUMAN)
            state.add_event("planning.material_universe_human_question_created", "material-universe ambiguity requires typed confirmation", {})
            return IncrementalExecutionIssue(code="planning.material_universe_awaiting_human", severity="error", message="material-universe gate awaiting human confirmation", patch_type="materials")
        # Phase 8A Step 6B: RETRIEVE_EVIDENCE research path.  When the
        # deterministic router sees source-coverage findings (e.g.
        # inventory.material_role_uncovered) and research is enabled,
        # execute a bounded PlanResearchRequest against the existing
        # SourceIndex.  New evidence is committed to the shared Ledger
        # so the next patch-generation pass picks it up via the
        # planning_constraints / investigation_evidence fields.
        if (
            _investigation_config_resolved is not None
            and not _investigation_config_resolved.is_off
            and _maybe_execute_research_for_gate(
                gate_id="material_universe",
                stage=stage,
                state=state,
                error_findings=error_findings,
                input_hash=input_hash,
                shared_source_index=_investigation_shared_source_index,
                shared_ledger=_investigation_shared_ledger,
                requirement=requirement,
                artifact_writer=artifact_writer,
                plan_investigation_output_dir=plan_investigation_output_dir,
                plan_loop_output_dir=plan_loop_output_dir,
                plan_investigation_client=plan_investigation_client,
            )
        ):
            # Research produced new evidence → reopen the gate so the
            # next pass regenerates the owner patch with the new
            # evidence context.  We return None so the outer for-loop
            # continues to the next patch_type; when Materials /
            # Universes are regenerated they will pick up the new
            # claims via PatchGenerationContext.investigation_evidence.
            return None
        # Build typed retry requests for blocking findings (upstream priority).
        from .closed_loop.retry_controller import normalize_retry_request
        from .closed_loop.retry_models import RetryTriggerOrigin
        for finding in error_findings:
            typed = normalize_retry_request(
                {
                    "code": finding.code,
                    "issue_codes": [finding.code],
                    "required_ids": finding.metadata.get("required_ids", []),
                    "affected_json_paths": finding.affected_json_paths,
                    "reason": finding.message,
                    "material_id": finding.metadata.get("material_id"),
                    "universe_id": finding.metadata.get("universe_id"),
                },
                state=state, origin=RetryTriggerOrigin.MATERIAL_UNIVERSE_GATE,
            )
            if typed is not None:
                artifact_writer._write(f"material_universe_retry_request_{typed.request_id[:12]}.json", typed)
                state.add_event("planning.material_universe_retry_requested", "material-universe blocking finding routed to Phase-3B retry", {"request_id": typed.request_id, "owner_patch_types": typed.owner_patch_types})
        transition_stage(stage, PlanStageStatus.BLOCKED)
        owner_patch_types = [
            owner
            for finding in error_findings
            for owner in finding.affected_patch_types
            if owner in {"facts", "materials", "universes"}
        ]
        failed_owner = owner_patch_types[0] if owner_patch_types else "materials"
        code = (
            "planning.material_universe.contract_preflight_failed"
            if _blocking_preflight
            else "planning.material_universe_gate_blocked"
        )
        return IncrementalExecutionIssue(
            code=code,
            severity="error",
            message=f"material-universe gate blocked by {len(error_findings)} finding(s)",
            patch_type=failed_owner,
        )

    def _run_placement_gate(
        *, finalize_non_applicable: bool = False,
    ) -> IncrementalExecutionIssue | None:
        """Run read-only/admission Placement Gate once its inputs are valid."""
        if policy.mode is PlanLoopMode.OFF or not policy.gate_enabled.get(PlanGateId.PLACEMENT, False):
            return None
        from .closed_loop.placement_evidence import (
            build_placement_evidence_pack, placement_gate_applicable,
            placement_gate_input_hash, placement_gate_ready,
        )
        from .closed_loop.placement_preflight import run_placement_preflight
        from .closed_loop.placement_reviewer import run_placement_review
        from .closed_loop.placement_review_prompts import build_placement_review_prompt
        from .closed_loop.placement_issue_policy import placement_issue_owner

        stage = _placement_stage()
        if stage is None:
            return None
        if _reuse_frozen_accepted_gate(PlanGateId.PLACEMENT):
            return None
        if policy.mode is PlanLoopMode.CONTROLLED:
            facts_stage = _facts_stage()
            if facts_stage is None or facts_stage.status is not PlanStageStatus.ACCEPTED:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.placement_requires_accepted_facts", severity="error", message="controlled placement gate requires accepted Facts Gate", patch_type="placement")
        applicable = placement_gate_applicable(state)
        # A placement gate can only be declared not applicable after all
        # candidate producer patches have had a chance to run.  In particular,
        # a profile or assembly intent generated after Facts may make the gate
        # applicable.  ``skipped`` is terminal in the protocol, so marking it
        # during the per-patch loop used to cause an illegal
        # skipped -> reviewing transition later in the same execution.
        if stage.status is PlanStageStatus.SKIPPED:
            if applicable and stage.metadata.get("reason") == "not_applicable":
                stage.status = PlanStageStatus.PENDING
                stage.completed_at = None
                stage.updated_at = None
                stage.metadata.pop("reason", None)
                state.add_event(
                    "planning.placement_gate_reopened",
                    "placement inputs became applicable after an earlier skip",
                    {"stage_id": stage.stage_id},
                )
            else:
                return None
        if not applicable:
            # Empty foundation-only runs have no facts patch at all.  Leave
            # their pending stage for the Phase-0 advisory artifact writer so
            # historic "review_not_implemented" semantics remain intact.
            if not _has_valid_patch(state, "facts") or not finalize_non_applicable:
                return None
            if stage.status is PlanStageStatus.PENDING:
                transition_stage(stage, PlanStageStatus.SKIPPED)
                stage.metadata["reason"] = "not_applicable"
                state.add_event(
                    "planning.placement_gate_not_applicable",
                    "placement gate not applicable after task-plan completion",
                    {},
                )
            return None
        if not placement_gate_ready(state):
            return None
        input_hash = placement_gate_input_hash(state)
        if stage.status in {PlanStageStatus.REVIEWED, PlanStageStatus.REVIEW_FAILED}:
            if stage.metadata.get("reviewed_input_hash") == input_hash:
                return None
            # Advisory reviews are terminal per input snapshot, not for the
            # lifetime of a run.  A changed placement patch begins a new,
            # auditable review round.
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            stage.updated_at = None
            state.add_event(
                "planning.placement_input_hash_changed",
                "reviewed placement input changed; starting a new review round",
                {"previous": stage.metadata.get("reviewed_input_hash"), "current": input_hash},
            )
        if stage.status is PlanStageStatus.ACCEPTED and stage.metadata.get("accepted_input_hash") == input_hash:
            return None
        if stage.status is PlanStageStatus.ACCEPTED:
            # A dependent valid patch was changed on resume; accepted evidence
            # cannot be reused.  This is the sole safe same-stage reset.
            stage.status = PlanStageStatus.PENDING
            stage.completed_at = None
            state.add_event("planning.placement_input_hash_changed", "accepted placement input changed", {"previous": stage.metadata.get("accepted_input_hash"), "current": input_hash})
        if stage.status in {PlanStageStatus.BLOCKED, PlanStageStatus.AWAITING_HUMAN}:
            return None
        if stage.status is PlanStageStatus.REPAIRING:
            transition_stage(stage, PlanStageStatus.VALIDATING)
            stage.validation_count += 1
        if stage.status is PlanStageStatus.PENDING:
            transition_stage(stage, PlanStageStatus.PROPOSING)
        if stage.status is PlanStageStatus.PROPOSING:
            transition_stage(stage, PlanStageStatus.VALIDATING)
            stage.validation_count += 1
        preflight = run_placement_preflight(state=state)
        state.add_event("planning.placement_preflight_completed", "placement deterministic preflight completed", {"issue_count": len(preflight["issues"])})
        pack = build_placement_evidence_pack(state=state, policy=policy, deterministic_issues=preflight["issues"])
        placement_scope_kind = ""
        if isinstance(preflight.get("binding_view"), dict):
            placement_scope_kind = str(preflight.get("binding_view", {}).get("scope_kind") or "")
        stage.metadata.update({"reviewed_input_hash": pack.input_hash, "review_model": getattr(plan_reviewer_client, "model_name", None)})
        for name, value in (("placement_binding_view.json", preflight.get("binding_view")), ("placement_contract_matrix.json", pack.contract_matrix), ("placement_evidence_pack.json", pack), ("placement_preflight.json", preflight)):
            path = artifact_writer._write(name, value)
            if path:
                state.plan_loop_artifacts.append(path)
        prompt_path = artifact_writer.write_text(f"placement_review_prompt_{stage.review_count:03d}.txt", build_placement_review_prompt(pack))
        if prompt_path:
            state.plan_loop_artifacts.append(prompt_path)
        enter_review_cycle(stage, state)
        state.add_event("planning.placement_review_started", "independent placement critic called", {"input_hash": pack.input_hash})
        if plan_reviewer_client is None:
            if policy.mode is PlanLoopMode.ADVISORY:
                transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                state.add_event("planning.placement_review_failed", "placement reviewer unavailable", {})
                return None
            transition_stage(stage, PlanStageStatus.BLOCKED)
            return IncrementalExecutionIssue(code="planning.placement_reviewer_unavailable", severity="error", message="controlled placement review requires a reviewer", patch_type="placement")
        review = run_placement_review(evidence_pack=pack, reviewer_client=plan_reviewer_client, state=state, policy=policy)
        state.placement_review_history.append(review.model_dump(mode="json"))
        for index, attempt in enumerate(review.attempts):
            # Providers may prepend private reasoning to an otherwise useful
            # JSON object.  Preserve replayable JSON only; never persist that
            # prose in artifacts.
            raw_value = str(attempt.get("raw_text", "")) if attempt.get("extraction_strategy") in {"json", "dict"} else "[non-JSON provider text redacted; see extraction metadata]"
            raw_path = artifact_writer.write_text(f"placement_review_raw_{index:03d}.txt", raw_value)
            extraction_path = artifact_writer._write(f"placement_review_extraction_{index:03d}.json", {key: value for key, value in attempt.items() if key != "raw_text"})
            for path in (raw_path, extraction_path):
                if path:
                    state.plan_loop_artifacts.append(path)
        if review.output is not None:
            path = artifact_writer._write("placement_review_normalized_000.json", review.output)
            if path:
                state.plan_loop_artifacts.append(path)
        path = artifact_writer._write("placement_review_findings.json", review.findings)
        if path:
            state.plan_loop_artifacts.append(path)
        record_findings(state, stage, review.findings)
        deterministic_findings = [
            PlanReviewFinding(gate_id=PlanGateId.PLACEMENT, code=item["code"], severity=PlanFindingSeverity.ERROR if item.get("severity") == "error" else PlanFindingSeverity.WARNING,
                              category=PlanFindingCategory.PLACEMENT_GAP, message=item.get("message", item["code"]), source_evidence=[],
                              affected_patch_types=placement_issue_owner(item["code"], canonical_scope=placement_scope_kind).get("owner_patch_types", []), affected_json_paths=[],
                              repairable_by_llm=bool(placement_issue_owner(item["code"], canonical_scope=placement_scope_kind).get("owner_patch_types")), requires_human=False, confidence=1.0,
                              metadata={"deterministic": True, "requirement_id": item.get("requirement_id")})
            for item in preflight["issues"]
        ]
        record_findings(state, stage, deterministic_findings)
        all_findings = deterministic_findings + review.findings
        if not review.ok:
            state.add_event("planning.placement_review_failed", review.error_code or "placement_review.schema_invalid", {})
            if policy.mode is PlanLoopMode.ADVISORY:
                transition_stage(stage, PlanStageStatus.REVIEW_FAILED)
                return None
            transition_stage(stage, PlanStageStatus.BLOCKED)
            return IncrementalExecutionIssue(code=review.error_code or "placement_review.schema_invalid", severity="error", message="placement review result unavailable", patch_type="placement")
        actions = compute_allowed_actions(policy=policy, stage_state=stage, findings=all_findings, deterministic_issues=preflight["issues"], additional_llm_calls_used=state.plan_loop_additional_llm_calls)
        action = actions[0] if actions else PlanReviewAction.FAIL_CLOSED
        dependency = next((item for item in preflight["issues"] if placement_issue_owner(item["code"]).get("dependency_patch_type")), None)
        if dependency is not None:
            action = PlanReviewAction.RETRY_DEPENDENCY
        decision = PlanReviewDecision(
            decision_id=f"placement_decision_{len(state.plan_review_decisions):03d}", gate_id=PlanGateId.PLACEMENT, action=action,
            target_patch_types=([placement_issue_owner(dependency["code"]).get("dependency_patch_type")] if dependency else (sorted({ptype for finding in all_findings for ptype in finding.affected_patch_types}) if action is PlanReviewAction.REVISE_CURRENT_PATCH else [])),
            finding_ids=[item.finding_id for item in all_findings], rationale="deterministic placement-gate action policy", allowed_actions_snapshot=list(dict.fromkeys(actions + [action])), decided_by="deterministic",
            metadata={"input_hash": pack.input_hash},
        )
        record_decision(state, stage, decision)
        path = artifact_writer._write("placement_review_decision.json", decision)
        if path:
            state.plan_loop_artifacts.append(path)
        state.add_event("planning.placement_review_completed", "placement critic normalized", {"finding_count": len(review.findings), "preflight_issue_count": len(preflight["issues"])})
        if policy.mode is PlanLoopMode.ADVISORY:
            transition_stage(stage, PlanStageStatus.REVIEWED)
            state.add_event("planning.placement_gate_reviewed", "placement gate recorded without mutation", {})
            return None
        if action is PlanReviewAction.APPROVE:
            transition_stage(stage, PlanStageStatus.ACCEPTED)
            stage.metadata["accepted_input_hash"] = pack.input_hash
            state.add_event("planning.placement_gate_accepted", "placement gate accepted", {"input_hash": pack.input_hash})
            _persist_checkpoint("gate:placement")
            return None
        if action is PlanReviewAction.ASK_HUMAN:
            from .closed_loop.placement_human import build_placement_human_question
            transition_stage(stage, PlanStageStatus.AWAITING_HUMAN)
            for finding in all_findings:
                if finding.requires_human:
                    question = build_placement_human_question(finding, input_hash=pack.input_hash)
                    if question.question_id not in state.plan_human_answers:
                        state.plan_human_questions[question.question_id] = question
            artifact_writer._write("placement_human_questions.json", [question for question in state.plan_human_questions.values() if question.gate_id is PlanGateId.PLACEMENT])
            state.add_event("planning.placement_human_question_created", "placement ambiguity requires typed confirmation", {"input_hash": pack.input_hash})
            return IncrementalExecutionIssue(code="planning.placement_awaiting_human", severity="error", message="placement gate awaiting human confirmation", patch_type="placement")
        if action is PlanReviewAction.RETRY_DEPENDENCY:
            dependency_code = str(dependency["code"])
            dependency_issues = [
                item for item in preflight["issues"]
                if item.get("code") == dependency_code
            ]
            requirement_ids = list(dict.fromkeys(
                str(item.get("requirement_id"))
                for item in dependency_issues
                if item.get("requirement_id")
            ))
            expected_ids = _placement_dependency_required_ids(
                preflight["issues"], dependency_code, state=state
            )
            request = {
                "request_id": f"placement_dependency_{len(state.placement_dependency_requests):03d}",
                "gate_id": "placement",
                "dependency_patch_type": placement_issue_owner(dependency_code).get("dependency_patch_type", "universes"),
                "issue_codes": [dependency_code],
                "finding_ids": [item.finding_id for item in all_findings],
                "required_ids": expected_ids,
                "requirement_id": dependency.get("requirement_id"),
                "requirement_ids": requirement_ids,
                "gate_input_hash": pack.input_hash,
                "reason": "placement gate dependency requires a prior gate",
                "downstream_patch_types": ["localized_insert_profiles", "pin_map", "assembly_catalog", "core_layout"],
            }
            state.placement_dependency_requests.append(request)
            artifact_writer._write("placement_dependency_retry_request.json", request)
            # Phase-3 turns the legacy Placement request into a typed owner
            # retry.  Advisory remains read-only; controlled mode regenerates
            # just the Python-selected dependency in a clone before commit.
            from .closed_loop.retry_controller import execute_plan_retry_loop
            from .closed_loop.retry_models import RetryTriggerOrigin

            from .closed_loop.retry_request_builders import build_retry_request_from_placement_dependency

            typed_request = build_retry_request_from_placement_dependency(
                dependency_patch_type=request["dependency_patch_type"],
                issue_codes=request["issue_codes"],
                finding_ids=request["finding_ids"],
                required_ids=request["required_ids"],
                reason=request["reason"],
                state=state,
                downstream_patch_types=request["downstream_patch_types"],
                gate_input_hash=pack.input_hash,
                consumer_ids=requirement_ids,
            )
            if typed_request is None:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                return IncrementalExecutionIssue(code="planning.retry.unsupported_request", severity="error", message="placement dependency is not registered for retry", patch_type="placement")
            artifact_writer.write_retry_artifact(f"retry_request_{len(state.plan_retry_rounds):03d}.json", typed_request)
            if policy.mode is PlanLoopMode.ADVISORY:
                outcome = execute_plan_retry_loop(state=state, policy=policy)
                artifact_writer.write_retry_artifact("retry_outcome.json", outcome)
                state.add_event("planning.placement_dependency_retry_requested", "placement dependency retry plan recorded in advisory mode", {"request_id": typed_request.request_id})
                return None

            def _produce_owner_candidate(retry_request: Any, _execution_plan: Any, clone_state: PlanBuildState) -> dict[str, dict[str, Any]]:
                candidates: dict[str, dict[str, Any]] = {}
                for owner_patch_type in retry_request.owner_patch_types:
                    if owner_patch_type == "planning_task_plan":
                        continue
                    base_context = build_generation_context_from_state(
                        clone_state, owner_patch_type, few_shot_case_ids=few_shot_case_ids
                    )
                    retry_context = RetryPatchGenerationContext(
                        base_context=base_context,
                        retry_request_id=getattr(retry_request, "request_id", ""),
                        reason_code=getattr(retry_request, "reason_code", ""),
                        source_issue_codes=list(getattr(retry_request, "source_issue_codes", [])),
                        required_ids=sorted({
                            item for target in retry_request.targets
                            for item in getattr(target, "required_ids", [])
                        }),
                        required_properties=sorted({
                            item for target in retry_request.targets
                            for item in getattr(target, "required_properties", [])
                        }),
                        affected_json_paths=sorted({
                            item for target in retry_request.targets
                            for item in getattr(target, "affected_json_paths", [])
                        }),
                        protected_invariants=sorted({
                            item for target in retry_request.targets
                            for item in getattr(target, "protected_json_paths", [])
                        }),
                        prior_candidate_hashes=list(
                            state.plan_retry_candidate_hashes_by_fingerprint.get(
                                getattr(retry_request, "request_fingerprint", ""), []
                            )
                        ),
                    )
                    generated = generate_patch(
                        patch_type=owner_patch_type,
                        requirement=requirement,
                        state=clone_state,
                        context=retry_context,
                        llm_client=llm_client,
                        max_attempts=max_patch_attempts,
                        max_tokens=LARGE_PATCH_MAX_TOKENS.get(owner_patch_type),
                    )
                    if not generated.ok or generated.parsed_patch is None:
                        issue_codes = [
                            str(issue.get("code"))
                            for issue in getattr(generated, "issues", [])
                            if isinstance(issue, dict) and issue.get("code")
                        ]
                        state.add_event(
                            "planning.retry_owner_generation_failed",
                            f"retry owner generation failed: {owner_patch_type}",
                            {
                                "owner_patch_type": owner_patch_type,
                                "issue_codes": issue_codes,
                                "issues": getattr(generated, "issues", []),
                            },
                        )
                        suffix = f": {', '.join(issue_codes)}" if issue_codes else ""
                        raise ValueError(f"retry owner generation failed: {owner_patch_type}{suffix}")
                    candidates[owner_patch_type] = generated.parsed_patch
                return candidates

            def _validate_owner_candidate(retry_request: Any, _execution_plan: Any, clone_state: PlanBuildState) -> list[dict[str, Any]]:
                # Universe retries must satisfy the requested IDs before any
                # downstream profile/catalog can be rebuilt.  Other schema
                # checks have already run in the retry controller.
                requested_ids = {item for target in retry_request.targets for item in target.required_ids}
                if "universes" in retry_request.owner_patch_types and requested_ids:
                    env = next((item for item in clone_state.patches.values() if item.patch_type == "universes" and item.status == "valid"), None)
                    found = {str(item.get("universe_id")) for item in (env.content.get("universes", []) if env else []) if isinstance(item, dict)}
                    missing = sorted(requested_ids - found)
                    if missing:
                        return [{"code": "retry.required_universe_ids_missing", "severity": "error", "missing_ids": missing}]
                return []

            outcome = execute_plan_retry_loop(
                state=state, policy=policy,
                candidate_producer=_produce_owner_candidate,
                candidate_validator=_validate_owner_candidate,
            )
            artifact_writer.write_retry_artifact(f"retry_execution_plan_{len(state.plan_retry_rounds):03d}.json", state.plan_retry_execution_plans)
            artifact_writer.write_retry_artifact("retry_outcome.json", outcome)
            if _retry_outcome_requires_downstream_resume(outcome):
                # Resume only invalidated downstream tasks.  The recursive
                # call observes committed owner envelopes and skips them; it
                # never clears the state or enters a monolithic fallback.
                state.add_event("planning.retry_downstream_resume_started", "resuming incremental generation after dependency owner commit", {"request_id": typed_request.request_id})
                depth = int(state.metadata.get("phase3_retry_resume_depth", 0))
                if depth >= policy.max_retry_rounds:
                    transition_stage(stage, PlanStageStatus.BLOCKED)
                    return IncrementalExecutionIssue(code="planning.retry_budget_exhausted", severity="error", message="retry resume budget exhausted", patch_type="placement")
                state.metadata["phase3_retry_resume_depth"] = depth + 1
                # Phase 8A Step 6 (P0-7): recursive resume MUST forward every
                # configuration knob the outer call received, otherwise the
                # controlled contract (investigation mode, fragmented
                # universes, strict structured output, budget) is silently
                # lost on the downstream rebuild.
                resumed = run_incremental_planning(
                    requirement=requirement, state=state, llm_client=llm_client,
                    max_patch_attempts=max_patch_attempts, strict=strict,
                    task_order=None, reference_patch_policy=reference_patch_policy,
                    reference_path=reference_path, few_shot_case_ids=few_shot_case_ids,
                    material_policy=material_policy, plan_loop_policy=policy,
                    plan_loop_output_dir=plan_loop_output_dir,
                    plan_reviewer_client=plan_reviewer_client,
                    plan_repair_client=plan_repair_client,
                    universes_generation_mode=universes_generation_mode,
                    universe_fragment_max_tokens=universe_fragment_max_tokens,
                    large_patch_safe_output_ratio=large_patch_safe_output_ratio,
                    strict_structured_patch_output=strict_structured_patch_output,
                    plan_investigation_config=plan_investigation_config,
                    plan_investigation_client=plan_investigation_client,
                    plan_investigation_registry=plan_investigation_registry,
                    plan_investigation_policy_registry=plan_investigation_policy_registry,
                    plan_investigation_output_dir=plan_investigation_output_dir,
                )
                state.metadata["phase3_retry_resume_depth"] = depth
                if resumed.ok:
                    state.add_event("planning.retry_downstream_resume_completed", "incremental downstream rebuild completed", {"request_id": typed_request.request_id})
                    return None
                return IncrementalExecutionIssue(code="planning.retry_downstream_rebuild_failed", severity="error", message="owner committed but downstream rebuild remains blocked", patch_type="placement")
            transition_stage(stage, PlanStageStatus.BLOCKED)
            state.add_event("planning.placement_dependency_retry_requested", "placement dependency retry did not resolve", {"request_id": typed_request.request_id, "outcome": outcome.status.value})
            return IncrementalExecutionIssue(code=f"planning.retry.{outcome.status.value}", severity="error", message=outcome.detail, patch_type="placement")
        if action is PlanReviewAction.REVISE_CURRENT_PATCH and plan_repair_client is not None:
            from .closed_loop.fingerprints import compute_issue_fingerprint
            from .closed_loop.placement_evidence import build_placement_evidence_pack
            from .closed_loop.placement_revision import (
                allowed_paths_for_placement_findings, commit_placement_revision,
                evaluate_placement_revision, normalize_placement_revision,
            )
            from .closed_loop.placement_revision_prompts import build_placement_revision_prompt
            from .closed_loop.models import PlacementRevisionProposal
            transition_stage(stage, PlanStageStatus.REPAIRING)
            stage.repair_count += 1
            blocking = [item for item in all_findings if item.severity is PlanFindingSeverity.ERROR and item.repairable_by_llm and not item.requires_human]
            issue_fingerprint = compute_issue_fingerprint(gate_id="placement", code="placement.blocking_set", affected_patch_type="placement", actual=sorted(item.finding_id for item in blocking))
            stage.issue_fingerprint = issue_fingerprint
            if state.plan_loop_issue_attempts_by_fingerprint.get(issue_fingerprint, 0) >= policy.max_attempts_per_issue_fingerprint or state.plan_loop_additional_llm_calls >= policy.max_total_additional_llm_calls:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                state.add_event("planning.placement_budget_exhausted", "placement repair budget exhausted", {})
                return IncrementalExecutionIssue(code="planning.closed_loop.issue_attempt_budget_exhausted", severity="error", message="placement repair budget exhausted", patch_type="placement")
            current = {}
            missing_owner_patch_types: list[str] = []
            for patch_type in sorted({ptype for finding in blocking for ptype in finding.affected_patch_types}):
                env = next((item for item in state.patches.values() if item.patch_type == patch_type and item.status == "valid"), None)
                if env is None:
                    missing_owner_patch_types.append(patch_type)
                    continue
                current[patch_type] = env.content
            if missing_owner_patch_types:
                transition_stage(stage, PlanStageStatus.BLOCKED)
                state.add_event(
                    "planning.placement_repair_owner_missing",
                    "placement repair owner patch is not present in the current state",
                    {"missing_patch_types": missing_owner_patch_types, "issue_fingerprint": issue_fingerprint},
                )
                return IncrementalExecutionIssue(
                    code="planning.placement_repair_owner_missing",
                    severity="error",
                    message="placement repair requires a missing owner patch",
                    patch_type=missing_owner_patch_types[0],
                )
            prompt = build_placement_revision_prompt(
                patches=current, findings=[item.model_dump(mode="json") for item in blocking], evidence_pack=pack.model_dump(mode="json"),
                allowed_paths=allowed_paths_for_placement_findings(blocking), confirmed_records=[item.model_dump(mode="json") for item in state.plan_confirmed_plan_fact_records.values()],
            )
            prompt_path = artifact_writer.write_text(f"placement_revision_prompt_{stage.repair_count - 1:03d}.txt", prompt)
            if prompt_path:
                state.plan_loop_artifacts.append(prompt_path)
            state.add_event("planning.placement_revision_started", "transactional placement revision started", {"finding_count": len(blocking)})
            try:
                if hasattr(plan_repair_client, "generate_patch_json"):
                    raw_proposal = plan_repair_client.generate_patch_json(prompt=prompt, patch_type="placement_revision", json_schema=PlacementRevisionProposal.model_json_schema(), temperature=0)
                else:
                    raw_proposal = plan_repair_client(prompt)
                state.plan_loop_additional_llm_calls += 1
                raw_path = artifact_writer.write_text(f"placement_revision_raw_{stage.repair_count - 1:03d}.txt", raw_proposal if isinstance(raw_proposal, str) else json.dumps(raw_proposal, ensure_ascii=False))
                if raw_path:
                    state.plan_loop_artifacts.append(raw_path)
                proposal = normalize_placement_revision(raw_proposal)
                proposal_path = artifact_writer._write(f"placement_revision_proposal_{stage.repair_count - 1:03d}.json", proposal)
                if proposal_path:
                    state.plan_loop_artifacts.append(proposal_path)
                evaluation = evaluate_placement_revision(state=state, proposal=proposal, findings=blocking, prior_candidate_hashes=state.plan_loop_candidate_hashes_by_fingerprint.get(issue_fingerprint, []))
                evaluation_path = artifact_writer._write(f"placement_revision_evaluation_{stage.repair_count - 1:03d}.json", evaluation)
                if evaluation_path:
                    state.plan_loop_artifacts.append(evaluation_path)
                if evaluation.candidate_hash:
                    duplicate = record_no_progress(state, stage, issue_fingerprint, evaluation.candidate_hash)
                    if duplicate or not evaluation.accepted:
                        state.placement_revision_history.append({"proposal": proposal.model_dump(mode="json"), "evaluation": evaluation.model_dump(mode="json")})
                        transition_stage(stage, PlanStageStatus.BLOCKED)
                        state.add_event("planning.placement_revision_rejected", "placement revision failed clone preflight", {"reasons": evaluation.reasons})
                        return IncrementalExecutionIssue(code="planning.placement_revision_rejected", severity="error", message="placement candidate failed clone evaluation", patch_type="placement")
                if not evaluation.accepted or evaluation.clone_state is None:
                    transition_stage(stage, PlanStageStatus.BLOCKED)
                    return IncrementalExecutionIssue(code="planning.placement_revision_rejected", severity="error", message="placement candidate rejected", patch_type="placement")
                clone = PlanBuildState.model_validate(evaluation.clone_state)
                clone_preflight = run_placement_preflight(state=clone)
                clone_pack = build_placement_evidence_pack(state=clone, policy=policy, deterministic_issues=clone_preflight["issues"])
                before_calls = clone.plan_loop_additional_llm_calls
                rereview = run_placement_review(evidence_pack=clone_pack, reviewer_client=plan_reviewer_client, state=clone, policy=policy)
                state.plan_loop_additional_llm_calls += clone.plan_loop_additional_llm_calls - before_calls
                after_errors = [item for item in clone_preflight["issues"] if item.get("severity") == "error"] + [item for item in rereview.findings if item.severity is PlanFindingSeverity.ERROR]
                if not rereview.ok or after_errors:
                    state.placement_revision_history.append({"proposal": proposal.model_dump(mode="json"), "evaluation": evaluation.model_dump(mode="json"), "rereview": rereview.model_dump(mode="json")})
                    transition_stage(stage, PlanStageStatus.BLOCKED)
                    state.add_event("planning.placement_revision_rejected", "placement clone re-review did not clear blocking findings", {})
                    return IncrementalExecutionIssue(code="planning.placement_revision_rereview_failed", severity="error", message="placement candidate failed independent re-review", patch_type="placement")
                changed = commit_placement_revision(state=state, evaluated=evaluation, proposal_id=proposal.proposal_id)
                state.placement_revision_history.append({"proposal": proposal.model_dump(mode="json"), "evaluation": evaluation.model_dump(mode="json"), "rereview": rereview.model_dump(mode="json"), "committed_patch_ids": changed})
                transition_stage(stage, PlanStageStatus.VALIDATING)
                transition_stage(stage, PlanStageStatus.REVIEWING)
                transition_stage(stage, PlanStageStatus.ACCEPTED)
                stage.metadata["accepted_input_hash"] = clone_pack.input_hash
                state.add_event("planning.placement_revision_rereviewed", "placement candidate independently re-reviewed", {})
                return None
            except Exception as exc:
                state.placement_revision_history.append({"error": str(exc)})
                transition_stage(stage, PlanStageStatus.BLOCKED)
                state.add_event("planning.placement_revision_rejected", "placement revision proposal rejected", {"error": str(exc)})
                return IncrementalExecutionIssue(code="planning.placement_revision_rejected", severity="error", message="placement revision proposal failed", patch_type="placement")
        transition_stage(stage, PlanStageStatus.BLOCKED)
        state.add_event("planning.placement_gate_blocked", "placement gate blocked", {"action": action.value})
        return IncrementalExecutionIssue(code="planning.placement_gate_blocked", severity="error", message=f"placement gate action={action.value}; revision is not available for this candidate", patch_type="placement")

    # Phase 8A Step 4: optional Facts investigation stage.  Setup happens
    # once, before the per-patch loop, so the shared SourceIndex + Ledger
    # + session cache live across the whole incremental run.  The actual
    # investigation runs only on the Facts patch_type (below).
    _investigation_session_cache: Any = None
    _investigation_shared_source_index: Any = None
    _investigation_shared_ledger: Any = None
    _investigation_config_resolved: Any = None
    _investigation_feature_contract: Any = getattr(state, "planning_feature_contract", None)
    if _investigation_feature_contract is None:
        from .planning_scope import planning_feature_contract as _build_investigation_feature_contract
        _investigation_feature_contract = _build_investigation_feature_contract(
            state.metadata.get("planning_mode_decision")
        )
    if plan_investigation_config is not None or plan_investigation_client is not None:
        from openmc_agent.plan_investigation.executor_injection import (
            InvestigationSessionCache,
            run_facts_investigation_stage,
        )
        from openmc_agent.plan_investigation.runner import (
            PlanInvestigationConfig as _ResolvedInvestigationConfig,
            get_investigation_config as _get_investigation_config,
            set_investigation_config as _set_investigation_config,
        )
        if isinstance(plan_investigation_config, dict):
            _investigation_config_resolved = _ResolvedInvestigationConfig.model_validate(
                plan_investigation_config
            )
        elif isinstance(plan_investigation_config, _ResolvedInvestigationConfig):
            _investigation_config_resolved = plan_investigation_config
        else:
            try:
                _investigation_config_resolved = _get_investigation_config(state)
            except Exception:
                _investigation_config_resolved = _ResolvedInvestigationConfig()
        # Persist the resolved config on state so downstream code can see
        # what mode the run is in (read-only; the executor never modifies
        # state.investigation-related slots directly elsewhere).
        if _investigation_config_resolved is not None and not _investigation_config_resolved.is_off:
            _set_investigation_config(state, _investigation_config_resolved)
            _investigation_session_cache = InvestigationSessionCache()
            # Phase 8A Step 6 (P0-1): build the shared SourceIndex + Ledger
            # ONCE here so Facts / Materials / Universes investigations +
            # the research executor all see the same canonical ledger.
            # Previously these stayed None and each investigation built
            # its own (silently dropping the shared-ledger contract).
            from openmc_agent.plan_investigation.runner import (
                build_investigation_ledger as _build_ledger,
                build_investigation_source_index as _build_idx,
            )
            _investigation_shared_source_index = _build_idx(requirement)
            _investigation_shared_ledger = _build_ledger(
                requirement_text=requirement,
                source_indexes={_investigation_shared_source_index.document.source_id: _investigation_shared_source_index},
            )

    for patch_type in order:
        stop_target = _stop_after_gate_target()
        if stop_target is not None:
            stopped_after_gate = _maybe_stop_after_gate(stop_target)
            if stopped_after_gate is not None:
                return stopped_after_gate
        facts_barrier_issue = _require_accepted_facts_gate(next_patch_type=patch_type)
        if facts_barrier_issue is not None:
            issues.append(facts_barrier_issue)
            state.add_event(
                EVENT_INCREMENTAL_EXECUTION_FAILED,
                "controlled Facts Gate blocked downstream patch generation",
                {"issue_code": facts_barrier_issue.code, "next_patch_type": patch_type},
            )
            status = "awaiting_human" if facts_barrier_issue.code == "planning.facts_awaiting_human" else "blocked"
            return IncrementalExecutionResult(
                ok=False,
                state=state,
                issues=issues,
                summary=_build_failure_summary("facts", [facts_barrier_issue.code], 0),
                plan_loop_outcome={
                    "status": status,
                    "active_gate_id": "facts",
                    "active_stage_id": "plan_gate_facts",
                    "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                    "detail": facts_barrier_issue.message,
                },
            )
        stopped_after_gate = _maybe_stop_after_gate(PlanGateId.FACTS)
        if stopped_after_gate is not None:
            return stopped_after_gate
        material_universe_barrier_issue = _require_accepted_material_universe_gate(
            next_patch_type=patch_type
        )
        if material_universe_barrier_issue is not None:
            issues.append(material_universe_barrier_issue)
            state.add_event(
                EVENT_INCREMENTAL_EXECUTION_FAILED,
                "controlled Material-Universe Gate blocked downstream patch generation",
                {
                    "issue_code": material_universe_barrier_issue.code,
                    "next_patch_type": patch_type,
                },
            )
            return IncrementalExecutionResult(
                ok=False,
                state=state,
                issues=issues,
                summary=_build_failure_summary(
                    material_universe_barrier_issue.patch_type or "universes",
                    [material_universe_barrier_issue.code],
                    0,
                ),
                plan_loop_outcome={
                    "status": "blocked",
                    "active_gate_id": "material_universe",
                    "active_stage_id": "plan_gate_material_universe",
                    "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                    "detail": material_universe_barrier_issue.message,
                },
            )
        # Skip if already valid.
        if _has_valid_patch(state, patch_type):
            state.add_event(
                event_type=EVENT_PATCH_SKIPPED_ALREADY_VALID,
                message=f"{patch_type} already valid, skipping",
                data={"patch_type": patch_type},
            )
            continue

        is_structural = patch_type in REFERENCE_PATCH_TYPES
        strict_reference_only = reference_patch_policy == "reference_only_for_structural"
        prefer_reference = reference_patch_policy == "prefer_reference_for_structural"

        # Lazy-load reference after facts patch has set benchmark_id.
        if (
            is_structural
            and reference_data is None
            and reference_patch_policy != "off"
            and state.benchmark_id is not None
        ):
            reference_data = load_benchmark_reference(
                benchmark_id=state.benchmark_id,
                variant=state.selected_variant,
                reference_path=reference_path,
                # Don't pass llm_client here — it would consume patch
                # generation responses. LLM matching should be done
                # separately if needed, not in the patch loop.
            )
            if reference_data is not None:
                _record_reference_metadata(state, reference_data)
                state.add_event(
                    event_type=EVENT_REFERENCE_PATCH_LOADED,
                    message=f"benchmark reference loaded for {state.benchmark_id}/{state.selected_variant}",
                    data={"policy": reference_patch_policy},
                )

        # Deterministic settings fallback remains available outside strict
        # reference-first structural policies.
        if (
            patch_type == "settings"
            and not strict_reference_only
            and not (prefer_reference and reference_data is not None)
        ):
            settings_patch = build_deterministic_settings_patch(state)
            content = settings_patch.model_dump(mode="json")
            _add_envelope(state, "settings", content, source="deterministic")
            state.add_event(
                event_type=EVENT_DETERMINISTIC_SETTINGS_CREATED,
                message="deterministic settings patch created",
                data={"source_strategy": settings_patch.source_strategy},
            )
            continue

        use_reference_first = (
            is_structural
            and reference_data is not None
            and reference_patch_policy in (
                "reference_only_for_structural",
                "prefer_reference_for_structural",
            )
        )

        if use_reference_first:
            ref_patch = build_reference_patch(
                patch_type=patch_type,
                reference=reference_data,
                variant=state.selected_variant,
            )
            if ref_patch is not None:
                val_result = validate_patch(
                    ref_patch,
                    _validation_context_for_state(state, patch_type),
                )
                if val_result.ok:
                    content = ref_patch.model_dump(mode="json")
                    _add_envelope(state, patch_type, content, source="fixture")
                    reference_patches_used.append(patch_type)
                    state.add_event(
                        event_type=EVENT_REFERENCE_PATCH_GENERATED,
                        message=f"{patch_type} patch from reference (valid)",
                        data={"patch_type": patch_type},
                    )
                    continue
                else:
                    issue_codes = [
                        i.code for i in val_result.issues if i.severity == "error"
                    ]
                    state.metadata["reference_match_status"] = "validation_failed"
                    state.add_event(
                        event_type=EVENT_REFERENCE_PATCH_VALIDATION_FAILED,
                        message=f"{patch_type} reference patch failed validation",
                        data={
                            "patch_type": patch_type,
                            "issue_codes": issue_codes,
                        },
                    )
                    if strict_reference_only:
                        return _fail_reference_only(
                            pt=patch_type,
                            code="reference_patch.validation_failed",
                            message=f"{patch_type} reference patch failed validation",
                            detail_codes=issue_codes,
                        )
            if strict_reference_only:
                return _fail_reference_only(
                    pt=patch_type,
                    code="reference_patch.required_unavailable",
                    message=f"{patch_type} reference patch is required but unavailable",
                )
            # Reference not available or failed in prefer mode → fall through to LLM.

        if (
            is_structural
            and reference_data is None
            and strict_reference_only
        ):
            return _fail_reference_only(
                pt=patch_type,
                code="reference_patch.required_unavailable",
                message=f"{patch_type} reference patch is required but unavailable",
            )

        # Build context from valid patches.
        ctx = build_generation_context_from_state(
            state, patch_type, few_shot_case_ids=few_shot_case_ids
        )
        candidate_context_fingerprint = _patch_generation_context_fingerprint(
            state, patch_type
        )
        prior_candidate_hashes = list(
            state.patch_generation_candidate_hashes_by_context.get(
                candidate_context_fingerprint, []
            )
        )
        generation_context: PatchGenerationContext | RetryPatchGenerationContext = ctx
        if prior_candidate_hashes:
            generation_context = RetryPatchGenerationContext(
                base_context=ctx,
                reason_code="patch_generation_resume",
                prior_candidate_hashes=prior_candidate_hashes,
            )
        if patch_type == "facts" and policy.mode is not PlanLoopMode.OFF:
            facts_stage = _facts_stage()
            if facts_stage is not None and facts_stage.status in {PlanStageStatus.PENDING, PlanStageStatus.REPAIRING}:
                transition_stage(facts_stage, PlanStageStatus.PROPOSING)
                facts_stage.attempt_count += 1

        # Phase 8A Step 4: optional Facts investigation stage runs BEFORE
        # the Facts patch LLM is invoked.  In controlled mode a blocked
        # investigation short-circuits the whole run with a stable
        # disposition.  In advisory mode failures are non-blocking but
        # are still recorded so callers cannot misrepresent them.
        if (
            patch_type == "facts"
            and _investigation_config_resolved is not None
            and not _investigation_config_resolved.is_off
        ):
            from openmc_agent.plan_investigation.executor_injection import (
                inject_investigation_evidence_into_context,
                run_facts_investigation_stage,
                BLOCK_CODE_FACTS_BLOCKED,
            )
            investigation_outcome = run_facts_investigation_stage(
                requirement=requirement,
                state=state,
                config=_investigation_config_resolved,
                llm_client=plan_investigation_client,
                registry=plan_investigation_registry,
                policy_registry=plan_investigation_policy_registry,
                session_cache=_investigation_session_cache,
                shared_source_index=_investigation_shared_source_index,
                shared_ledger=_investigation_shared_ledger,
                feature_contract=_investigation_feature_contract,
                artifact_output_dir=(
                    Path(plan_investigation_output_dir)
                    if plan_investigation_output_dir is not None
                    else (Path(plan_loop_output_dir) if plan_loop_output_dir else None)
                ),
                add_event=lambda evt, msg, data: state.add_event(evt, msg, data),
                action_callback=facts_action_callback,
            )
            # Step 4 only runs the Facts investigation; the shared
            # SourceIndex + Ledger caches are reserved for Step 5+ when
            # Materials/Universes investigations reuse them.
            if investigation_outcome.should_block_facts_patch:
                issues.append(
                    IncrementalExecutionIssue(
                        code="planning.investigation_facts_blocked",
                        severity="error",
                        message=(
                            f"controlled Facts investigation blocked: "
                            f"{investigation_outcome.block_code or BLOCK_CODE_FACTS_BLOCKED}"
                        ),
                        patch_type="facts",
                    )
                )
                return IncrementalExecutionResult(
                    ok=False,
                    state=state,
                    issues=issues,
                    summary={
                        "issue_codes": ["planning.investigation_facts_blocked"],
                        "investigation_block_code": investigation_outcome.block_code,
                        "investigation_session_id": investigation_outcome.session_id,
                        "investigation_evidence_claim_count": len(
                            investigation_outcome.evidence_claim_ids
                        ),
                    },
                    plan_loop_outcome={
                        "status": "blocked",
                        "active_gate_id": "facts",
                        "active_stage_id": "plan_gate_facts",
                        "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                        "detail": (
                            f"BLOCKED_BY_INVESTIGATION:facts "
                            f"({investigation_outcome.block_code})"
                        ),
                    },
                )
            if investigation_outcome.evidence_payloads:
                generation_context = inject_investigation_evidence_into_context(
                    generation_context, investigation_outcome
                )
                state.add_event(
                    "planning.investigation_evidence_injected",
                    "Facts patch prompt received investigation evidence",
                    {
                        "claim_count": len(investigation_outcome.evidence_claim_ids),
                        "evidence_context_hash": investigation_outcome.evidence_context_hash[:12],
                        "cache_reused": investigation_outcome.cache_reused,
                    },
                )

                # Phase-8B Step 3: compile the FactsRequirementSkeleton from
                # investigation evidence + feature contract.  Both the
                # prompt injection (via facts_evidence_contract) and the
                # post-generation merge + preflight (via
                # facts_requirement_skeleton) use this.
                from .facts_requirement_skeleton import (
                    compile_facts_requirement_skeleton,
                )
                from .facts_evidence_contract import (
                    compile_facts_evidence_contract,
                )
                from .planning_scope import (
                    planning_feature_contract as _build_feature_contract,
                )

                feature_contract_obj = _build_feature_contract(
                    state.metadata.get("planning_mode_decision")
                )
                skel_result = compile_facts_requirement_skeleton(
                    requirement_text=requirement,
                    feature_contract=feature_contract_obj,
                    evidence_ledger=_investigation_shared_ledger,
                    source_index=_investigation_shared_source_index,
                    confirmed_facts=state.confirmed_facts,
                )
                if skel_result.ok and skel_result.skeleton is not None:
                    ev_contract = compile_facts_evidence_contract(
                        skel_result.skeleton
                    )
                    # Set on the unwrapped context so the prompt builder
                    # and post-generation merge can both access them.
                    ctx.facts_requirement_skeleton = (
                        skel_result.skeleton.model_dump(mode="json")
                    )
                    ctx.facts_evidence_contract = (
                        ev_contract.model_dump(mode="json")
                    )
                    state.add_event(
                        "planning.facts_skeleton_compiled",
                        "FactsRequirementSkeleton compiled from investigation evidence",
                        {
                            "skeleton_hash": skel_result.skeleton.skeleton_hash[:12],
                            "contract_hash": ev_contract.evidence_contract_hash[:12],
                            "fuel_variant_count": len(skel_result.skeleton.fuel_variant_slots),
                            "localized_insert_count": len(skel_result.skeleton.localized_insert_slots),
                        },
                    )
                else:
                    state.add_event(
                        "planning.facts_skeleton_compilation_failed",
                        "FactsRequirementSkeleton compilation did not produce a skeleton",
                        {"warnings": skel_result.warnings, "errors": skel_result.errors},
                    )

        # Phase 8A Step 6 (P0-1): Materials and Universes investigations.
        # Mirror the Facts investigation path: run BEFORE the patch LLM,
        # block in controlled mode on failure, inject evidence into the
        # generation context.  The Materials/Universes investigations
        # reuse the SAME shared SourceIndex + Ledger that the Facts
        # investigation populated so claims accumulate across patch
        # types (single canonical ledger per incremental run).
        if (
            patch_type in {"materials", "universes"}
            and _investigation_config_resolved is not None
            and not _investigation_config_resolved.is_off
        ):
            from openmc_agent.plan_investigation.executor_injection import (
                BLOCK_CODE_MATERIALS_BLOCKED,
                BLOCK_CODE_UNIVERSES_BLOCKED,
                inject_investigation_evidence_into_context,
                run_patch_investigation_stage,
            )
            # Build the typed inventory context from accepted Facts +
            # compiled inventory + requirement sets so the baseline
            # resolver can read fuel variants / radial profiles.
            accepted_facts_obj, geometry_inventory_obj = _load_inventory_context_objects(state)
            material_req_obj = _load_material_requirement_set(state)
            universe_req_obj = _load_universe_requirement_set(state)
            investigation_outcome = run_patch_investigation_stage(
                patch_type=patch_type,
                requirement=requirement,
                state=state,
                config=_investigation_config_resolved,
                llm_client=plan_investigation_client,
                registry=plan_investigation_registry,
                policy_registry=plan_investigation_policy_registry,
                session_cache=_investigation_session_cache,
                shared_source_index=_investigation_shared_source_index,
                shared_ledger=_investigation_shared_ledger,
                accepted_facts=accepted_facts_obj,
                geometry_inventory=geometry_inventory_obj,
                material_requirement_set=material_req_obj,
                universe_requirement_set=universe_req_obj,
                feature_contract=_investigation_feature_contract,
                artifact_output_dir=(
                    Path(plan_investigation_output_dir)
                    if plan_investigation_output_dir is not None
                    else (Path(plan_loop_output_dir) if plan_loop_output_dir else None)
                ),
                add_event=lambda evt, msg, data: state.add_event(evt, msg, data),
                action_callback=facts_action_callback,
            )
            if investigation_outcome.blocked:
                block_code = investigation_outcome.block_code or (
                    BLOCK_CODE_MATERIALS_BLOCKED if patch_type == "materials" else BLOCK_CODE_UNIVERSES_BLOCKED
                )
                issues.append(
                    IncrementalExecutionIssue(
                        code=f"planning.investigation_{patch_type}_blocked",
                        severity="error",
                        message=(
                            f"controlled {patch_type} investigation blocked: {block_code}"
                        ),
                        patch_type=patch_type,
                    )
                )
                return IncrementalExecutionResult(
                    ok=False,
                    state=state,
                    issues=issues,
                    summary={
                        "issue_codes": [f"planning.investigation_{patch_type}_blocked"],
                        "investigation_block_code": block_code,
                        "investigation_session_id": investigation_outcome.session_id,
                        "investigation_evidence_claim_count": len(
                            investigation_outcome.evidence_claim_ids
                        ),
                    },
                    plan_loop_outcome={
                        "status": "blocked",
                        "active_gate_id": (
                            "material_universe" if patch_type in {"materials", "universes"} else "facts"
                        ),
                        "active_stage_id": f"plan_gate_material_universe",
                        "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                        "detail": (
                            f"BLOCKED_BY_INVESTIGATION:{patch_type} ({block_code})"
                        ),
                    },
                )
            if investigation_outcome.evidence_payloads:
                generation_context = inject_investigation_evidence_into_context(
                    generation_context, investigation_outcome
                )
                state.add_event(
                    f"planning.investigation_{patch_type}_evidence_injected",
                    f"{patch_type} patch prompt received investigation evidence",
                    {
                        "claim_count": len(investigation_outcome.evidence_claim_ids),
                        "evidence_context_hash": investigation_outcome.evidence_context_hash[:12],
                        "cache_reused": investigation_outcome.cache_reused,
                    },
                )

        # Generate patch with retry. By default we use the provider's output
        # token default (e.g. DeepSeek ~8192), which is larger than any safe
        # universal per-patch cap. Large multi-assembly / full-core patches
        # (universes, assembly_catalog, core_layout) exceed that default and
        # get truncated mid-JSON (observed: a VERA4 11-universe catalog cut at
        # ~6500 tokens), so for those we pass an explicit larger budget from
        # LARGE_PATCH_MAX_TOKENS; other patch types keep the provider default.
        # For thinking-mode providers (ds:), reasoning_effort is capped in the
        # client instead — see DSChatClient.adjust_payload.
        # P0-LARGE-STRUCTURED-PATCH: for universes and materials, use the
        # fragmented pipeline which auto-decides monolithic vs fragmented
        # based on estimated size and truncation history.
        if patch_type == "materials" and getattr(state, 'metadata', {}).get("planning_material_requirement_set"):
            from .materials_patch_pipeline import generate_materials_patch
            result = generate_materials_patch(
                requirement=requirement,
                state=state,
                llm_client=llm_client,
                mode="auto",
                max_tokens=LARGE_PATCH_MAX_TOKENS.get(patch_type),
                safe_output_ratio=large_patch_safe_output_ratio,
            )
        elif patch_type == "universes" and universes_generation_mode != "off":
            from .universe_patch_pipeline import generate_universes_patch
            # In controlled/inventory mode, pass the compiled inventory
            # universe requirement set so the pipeline consumes only
            # source-backed requirements (no legacy implicit:* fallback).
            _inv_universe_req = state.metadata.get("planning_universe_requirement_set")
            result = generate_universes_patch(
                requirement=requirement,
                state=state,
                llm_client=llm_client,
                mode=universes_generation_mode,
                max_tokens=universe_fragment_max_tokens or LARGE_PATCH_MAX_TOKENS.get(patch_type),
                safe_output_ratio=large_patch_safe_output_ratio,
                strict_structured=strict_structured_patch_output,
                inventory_universe_requirement_set=_inv_universe_req,
            )
        else:
            result = generate_patch(
                patch_type=patch_type,
                requirement=requirement,
                state=state,
                context=generation_context,
                llm_client=llm_client,
                max_attempts=max_patch_attempts,
                max_tokens=LARGE_PATCH_MAX_TOKENS.get(patch_type),
            )

        if result.ok and result.envelope is not None:
            # Phase-8B Step 3: merge skeleton locked fields into the
            # generated Facts patch, then run preflight.
            if patch_type == "facts" and ctx.facts_requirement_skeleton is not None:
                from .facts_requirement_skeleton import (
                    FactsRequirementSkeleton,
                )
                from .facts_evidence_contract import (
                    FactsContentProposal,
                    merge_facts_content_into_skeleton,
                    run_facts_skeleton_preflight,
                )

                skeleton_model = FactsRequirementSkeleton.model_validate(
                    ctx.facts_requirement_skeleton
                )
                proposal = FactsContentProposal(
                    proposal_id="facts_post_gen_merge",
                    resolved_fields=dict(result.envelope.content),
                )
                merge_result = merge_facts_content_into_skeleton(
                    skeleton=skeleton_model,
                    proposal=proposal,
                    evidence_ledger=_investigation_shared_ledger,
                )
                if merge_result.ok and merge_result.merged is not None:
                    result.envelope.content = merge_result.merged.patch
                    state.add_event(
                        "planning.facts_skeleton_merge_applied",
                        "Facts skeleton locked fields merged into generated patch",
                        {
                            "patch_hash": merge_result.merged.patch_hash[:12],
                            "warning_count": len(merge_result.warnings),
                        },
                    )
                else:
                    state.add_event(
                        "planning.facts_skeleton_merge_failed",
                        "Facts skeleton merge did not produce merged patch",
                        {"errors": merge_result.errors},
                    )
                # Run preflight on the (possibly merged) content.
                preflight = run_facts_skeleton_preflight(
                    skeleton_model, result.envelope.content
                )
                # Phase 8C Step 2: stash the contract preflight issues on
                # state metadata so _run_facts_gate can promote them to
                # PlanReviewFinding and route to REVISE_CURRENT_PATCH.
                # Without this lift the preflight findings were logged as
                # events only and the gate never saw them.
                state.metadata["facts_contract_preflight_issues"] = list(preflight.issues)
                state.metadata["facts_contract_preflight_ok"] = bool(preflight.ok)
                if not preflight.ok:
                    state.add_event(
                        "planning.facts_skeleton_preflight_issues",
                        "Facts skeleton preflight flagged blocking issues",
                        {
                            "issue_count": len(preflight.issues),
                            "issue_codes": [i.get("code") for i in preflight.issues],
                            "blocking_count": sum(1 for i in preflight.issues if i.get("severity") == "error"),
                        },
                    )
            state.add_patch(result.envelope)
            if result.envelope.status == "valid" and patch_type == "materials":
                _persist_checkpoint("patch:materials")
            if result.envelope.status == "valid" and patch_type == "universes":
                _persist_checkpoint("patch:universes")
            if patch_type == "facts":
                facts_gate_issue = _run_facts_gate()
                if facts_gate_issue is not None:
                    issues.append(facts_gate_issue)
                    state.add_event(
                        event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
                        message="facts gate blocked downstream patch generation",
                        data={"issue_code": facts_gate_issue.code},
                    )
                    return IncrementalExecutionResult(
                        ok=False, state=state, issues=issues,
                        summary=_build_failure_summary("facts", [facts_gate_issue.code], len(result.attempts)),
                        plan_loop_outcome={
                            "status": "awaiting_human" if facts_gate_issue.code == "planning.facts_awaiting_human" else "blocked",
                            "active_gate_id": "facts",
                            "active_stage_id": "plan_gate_facts",
                            "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                            "detail": facts_gate_issue.message,
                        },
                    )
                stopped_after_gate = _maybe_stop_after_gate(PlanGateId.FACTS)
                if stopped_after_gate is not None:
                    return stopped_after_gate
                if policy.mode is PlanLoopMode.CONTROLLED and task_order is None:
                    previous_order = list(order)
                    _sync_controlled_order_after_facts(
                        state=state,
                        order=order,
                        required=required,
                        placement_controlled=placement_controlled,
                        material_universe_controlled=material_universe_controlled,
                        axial_geometry_controlled=axial_geometry_controlled,
                    )
                    if order != previous_order:
                        state.add_event(
                            "planning.task_order_refreshed_after_facts",
                            "controlled patch order refreshed after accepted Facts",
                            {
                                "previous_order": previous_order,
                                "order": order,
                                "required": required,
                            },
                        )
                if (policy.mode is PlanLoopMode.CONTROLLED and state.canonical_task_plan is not None
                        and task_order is None and required != ["facts"]):
                    # Mutate the iterated list in-place: the remaining work is
                    # now authoritative canonical work, not the provisional
                    # detector-only order calculated before Facts existed.
                    order[:] = list(state.canonical_task_plan.ordered_patch_types)
                    required[:] = list(state.canonical_task_plan.required_patch_types)
                    _sync_controlled_order_after_facts(
                        state=state,
                        order=order,
                        required=required,
                        placement_controlled=placement_controlled,
                        material_universe_controlled=material_universe_controlled,
                        axial_geometry_controlled=axial_geometry_controlled,
                    )
                    state.add_event("planning.task_plan_reconciled", "provisional task order replaced by canonical task plan", {"plan_hash": state.canonical_task_plan.plan_hash, "order": order})
            # Phase-4: Material-Universe Gate runs after Materials and
            # Universes become valid, before any downstream patch that
            # consumes them.
            material_universe_gate_issue = _run_material_universe_gate()
            if material_universe_gate_issue is not None:
                issues.append(material_universe_gate_issue)
                state.add_event(
                    event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
                    message="material-universe gate blocked downstream patch generation",
                    data={"issue_code": material_universe_gate_issue.code},
                )
                return IncrementalExecutionResult(
                    ok=False, state=state, issues=issues,
                    summary=_build_failure_summary(
                        material_universe_gate_issue.patch_type or "materials",
                        [material_universe_gate_issue.code],
                        len(result.attempts),
                    ),
                    plan_loop_outcome={
                        "status": "blocked", "active_gate_id": "material_universe", "active_stage_id": "plan_gate_material_universe",
                        "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": material_universe_gate_issue.message,
                    },
                )
            stopped_after_gate = _maybe_stop_after_gate(PlanGateId.MATERIAL_UNIVERSE)
            if stopped_after_gate is not None:
                return stopped_after_gate
            # Placement is evaluated at the first point all of its scoped
            # inputs become valid.  In controlled mode the reordered task list
            # makes this a barrier before axial patch generation; advisory
            # retains historical order and remains read-only.
            placement_gate_issue = _run_placement_gate()
            if placement_gate_issue is not None:
                issues.append(placement_gate_issue)
                state.add_event(
                    event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
                    message="placement gate blocked downstream patch generation",
                    data={"issue_code": placement_gate_issue.code},
                )
                return IncrementalExecutionResult(
                    ok=False, state=state, issues=issues,
                    summary=_build_failure_summary("placement", [placement_gate_issue.code], len(result.attempts)),
                    plan_loop_outcome={
                        "status": "blocked", "active_gate_id": "placement", "active_stage_id": "plan_gate_placement",
                        "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": placement_gate_issue.message,
                    },
                )
            stopped_after_gate = _maybe_stop_after_gate(PlanGateId.PLACEMENT)
            if stopped_after_gate is not None:
                return stopped_after_gate
            # Phase-5: Axial-Geometry Gate runs after axial patches become valid.
            axial_geometry_gate_issue = _run_axial_geometry_gate()
            if axial_geometry_gate_issue is not None:
                issues.append(axial_geometry_gate_issue)
                state.add_event(
                    event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
                    message="axial-geometry gate blocked downstream patch generation",
                    data={"issue_code": axial_geometry_gate_issue.code},
                )
                return IncrementalExecutionResult(
                    ok=False, state=state, issues=issues,
                    summary=_build_failure_summary("axial_layers", [axial_geometry_gate_issue.code], len(result.attempts)),
                    plan_loop_outcome={
                        "status": "blocked", "active_gate_id": "axial_geometry", "active_stage_id": "plan_gate_axial_geometry",
                        "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": axial_geometry_gate_issue.message,
                    },
                )
            stopped_after_gate = _maybe_stop_after_gate(PlanGateId.AXIAL_GEOMETRY)
            if stopped_after_gate is not None:
                return stopped_after_gate
            # Phase 7D: extract benchmark_id from FactsPatch for reference loading.
            _sync_benchmark_from_facts()

            # Phase 7D+: one-time LLM semantic benchmark matching after facts.
            # This runs ONCE, right after facts is generated, so the LLM call
            # doesn't interfere with subsequent patch generation responses.
            if (
                patch_type == "facts"
                and reference_patch_policy != "off"
                and reference_data is None
                and state.benchmark_id is not None
            ):
                import re
                # Only attempt matching for plausible benchmark identifiers
                # (avoids wasting an LLM call on test placeholders like "T").
                alpha_len = len(re.sub(r"[^a-zA-Z]", "", state.benchmark_id))
                if alpha_len >= 4:
                    # Try exact match first (no LLM).
                    reference_data = load_benchmark_reference(
                        benchmark_id=state.benchmark_id,
                        variant=state.selected_variant,
                        reference_path=reference_path,
                    )
                    if reference_data is not None:
                        _record_reference_metadata(state, reference_data)
                    # If exact match failed, try LLM semantic matching.
                    if reference_data is None:
                        try:
                            reference_data = load_benchmark_reference(
                                benchmark_id=state.benchmark_id,
                                variant=state.selected_variant,
                                reference_path=reference_path,
                                llm_client=llm_client,
                            )
                            if reference_data is not None:
                                _record_reference_metadata(state, reference_data)
                        except Exception:
                            pass
                    if reference_data is not None:
                        state.add_event(
                            event_type=EVENT_REFERENCE_PATCH_LOADED,
                            message=(
                                f"benchmark reference matched for "
                                f"{state.benchmark_id}/{state.selected_variant}"
                            ),
                            data={"policy": reference_patch_policy},
                        )

            state.add_event(
                event_type=EVENT_PATCH_GENERATED,
                message=f"{patch_type} generated and validated",
                data={
                    "patch_id": result.envelope.patch_id,
                    "attempts": len(result.attempts),
                },
            )
        else:
            # Try reference fallback if policy allows.
            error_codes = [
                i.get("code", "") for i in result.issues
                if i.get("severity") == "error"
            ]

            # Lazy-load reference for fallback (benchmark_id may have been
            # extracted from FactsPatch after initial load attempt).
            if (
                is_structural
                and reference_data is None
                and reference_patch_policy == "fallback_after_llm_failure"
                and state.benchmark_id is not None
            ):
                reference_data = load_benchmark_reference(
                    benchmark_id=state.benchmark_id,
                    variant=state.selected_variant,
                    reference_path=reference_path,
                )
                if reference_data is not None:
                    _record_reference_metadata(state, reference_data)

            if (
                is_structural
                and reference_data is not None
                and reference_patch_policy == "fallback_after_llm_failure"
            ):
                ref_patch = build_reference_patch(
                    patch_type=patch_type,
                    reference=reference_data,
                    variant=state.selected_variant,
                )
                if ref_patch is not None:
                    val_result = validate_patch(
                        ref_patch,
                        _validation_context_for_state(state, patch_type),
                    )
                    if val_result.ok:
                        content = ref_patch.model_dump(mode="json")
                        _add_envelope(state, patch_type, content, source="fixture")
                        reference_patches_used.append(patch_type)
                        state.add_event(
                            event_type=EVENT_REFERENCE_PATCH_FALLBACK,
                            message=f"{patch_type} reference fallback after LLM failure",
                            data={"patch_type": patch_type, "llm_error_codes": error_codes},
                        )
                        continue

            # All retries exhausted.
            attempt_count = len(result.attempts)

            # Phase 7D+: save raw attempt data for diagnosis.
            patch_attempts = state.metadata.setdefault("patch_attempt_artifacts", {})
            for att in result.attempts:
                att_key = f"{patch_type}_attempt_{att.attempt_index + 1}"
                patch_attempts[att_key] = {
                    "patch_type": patch_type,
                    "attempt_index": att.attempt_index,
                    "raw_chars": att.raw_chars,
                    "raw_text": (att.raw_text or "")[:50000],
                    "prompt_text": (att.prompt_text or "")[:60000],
                    "issues": att.issues,
                    "output_mode_used": att.output_mode_used,
                    "candidate_hash": att.candidate_hash,
                    "semantic_normalizations": att.semantic_normalizations,
                    "error": att.error,
                }

            candidate_hashes = [
                attempt.candidate_hash
                for attempt in result.attempts
                if attempt.candidate_hash
            ]
            if candidate_hashes:
                ledger = state.patch_generation_candidate_hashes_by_context.setdefault(
                    candidate_context_fingerprint, []
                )
                for candidate_hash in candidate_hashes:
                    if candidate_hash not in ledger:
                        ledger.append(candidate_hash)
                state.patch_generation_attempts_by_context[
                    candidate_context_fingerprint
                ] = state.patch_generation_attempts_by_context.get(
                    candidate_context_fingerprint, 0
                ) + 1

            decision = route_retry(
                failed_patch_type=patch_type,
                issues=result.issues,
                state=state,
            )
            state.add_event(
                event_type=EVENT_PATCH_RETRY_ROUTED,
                message=f"{patch_type} retry routed: {decision.action}",
                data={
                    "patch_type": patch_type,
                    "action": decision.action,
                    "reason": decision.reason,
                    "error_codes": error_codes,
                },
            )

            missing_universe_ids = sorted({
                str(issue.get("actual"))
                for issue in result.issues
                if issue.get("code") == "lattice_transform.replacement_universe_missing"
                and isinstance(issue.get("actual"), str)
            })
            if patch_type == "axial_layers" and missing_universe_ids:
                dependency_issues = [
                    issue for issue in result.issues
                    if issue.get("code") == "lattice_transform.replacement_universe_missing"
                ]
                state.metadata["plan_validation_repair"] = {
                    # The missing IDs are owned by UniversesPatch, but the
                    # failing axial patch must also be regenerated after that
                    # owner changes.  This is a bounded dependency replay,
                    # not a request to restart the whole plan.
                    "target_patch_types": ["universes", "axial_layers"],
                    "issues": dependency_issues,
                }
                state.metadata["incremental_dependency_repair"] = {
                    "missing_universe_ids": missing_universe_ids,
                }
                state.add_event(
                    event_type="planning.axial_dependency_repair_scheduled",
                    message=(
                        "axial-layers replacement universes are missing; scheduling "
                        "targeted universe regeneration"
                    ),
                    data={"missing_universe_ids": missing_universe_ids},
                )

            issues.append(IncrementalExecutionIssue(
                code="incremental.patch_generation_failed",
                severity="error",
                message=f"{patch_type} generation failed: {error_codes}",
                patch_type=patch_type,
            ))
            state.add_event(
                event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
                message=f"execution stopped: {patch_type} generation failed",
                data={"failed_patch_type": patch_type, "error_codes": error_codes},
            )
            no_progress = any(
                issue.get("code") == "patch_generation.no_progress_duplicate_candidate"
                for issue in result.issues
            )
            if no_progress:
                state.add_event(
                    event_type="planning.patch_generation_no_progress",
                    message=(
                        f"{patch_type} repeated a rejected candidate; "
                        "stopping targeted generation"
                    ),
                    data={
                        "patch_type": patch_type,
                        "candidate_context_fingerprint": candidate_context_fingerprint,
                        "candidate_hashes": candidate_hashes,
                    },
                )
            return IncrementalExecutionResult(
                ok=False,
                state=state,
                issues=issues,
                summary=_build_failure_summary(
                    patch_type,
                    error_codes,
                    attempt_count,
                    no_progress=no_progress,
                ),
                plan_loop_outcome=_write_advisory_artifacts(),
            )

    # Check required patches.
    missing = [pt for pt in required if not _has_valid_patch(state, pt)]
    if missing:
        for pt in missing:
            issues.append(IncrementalExecutionIssue(
                code="assembly.missing_patch",
                severity="error",
                message=f"required {pt} patch is missing",
                patch_type=pt,
            ))
        state.add_event(
            event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
            message=f"missing required patches: {missing}",
            data={"missing": missing},
        )
        return IncrementalExecutionResult(
            ok=False,
            state=state,
            issues=issues,
            summary={
                "missing_patches": missing,
                "failed_patch_type": missing[0] if missing else None,
                "issue_codes": [i.code for i in issues],
                "reference_match_status": state.metadata.get(
                    "reference_match_status",
                    "off" if reference_patch_policy == "off" else "unavailable",
                ),
                "reference_path": state.metadata.get("reference_path"),
                "reference_patches_used": reference_patches_used,
            },
            plan_loop_outcome=_write_advisory_artifacts(),
        )

    # Resume paths may skip every already-valid patch, so give each gate one
    # final opportunity before assembly.  The input-hash guards make these
    # checks idempotent.
    facts_gate_issue = _require_accepted_facts_gate()
    if facts_gate_issue is not None:
        issues.append(facts_gate_issue)
        status = "awaiting_human" if facts_gate_issue.code == "planning.facts_awaiting_human" else "blocked"
        return IncrementalExecutionResult(
            ok=False,
            state=state,
            issues=issues,
            summary=_build_failure_summary("facts", [facts_gate_issue.code], 0),
            plan_loop_outcome={
                "status": status,
                "active_gate_id": "facts",
                "active_stage_id": "plan_gate_facts",
                "additional_llm_calls_used": state.plan_loop_additional_llm_calls,
                "detail": facts_gate_issue.message,
            },
        )
    material_universe_gate_issue = _require_accepted_material_universe_gate()
    if material_universe_gate_issue is not None:
        issues.append(material_universe_gate_issue)
        return IncrementalExecutionResult(
            ok=False, state=state, issues=issues,
            summary=_build_failure_summary(
                material_universe_gate_issue.patch_type or "materials",
                [material_universe_gate_issue.code],
                0,
            ),
            plan_loop_outcome={"status": "blocked", "active_gate_id": "material_universe", "active_stage_id": "plan_gate_material_universe", "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": material_universe_gate_issue.message},
        )
    placement_gate_issue = _run_placement_gate(finalize_non_applicable=True)
    if placement_gate_issue is not None:
        issues.append(placement_gate_issue)
        return IncrementalExecutionResult(
            ok=False, state=state, issues=issues,
            summary=_build_failure_summary("placement", [placement_gate_issue.code], 0),
            plan_loop_outcome={"status": "blocked", "active_gate_id": "placement", "active_stage_id": "plan_gate_placement", "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": placement_gate_issue.message},
        )

    # Phase-5: Axial-Geometry Gate gets a final opportunity before assembly.
    axial_geometry_gate_issue = _run_axial_geometry_gate(finalize_non_applicable=True)
    if axial_geometry_gate_issue is not None:
        issues.append(axial_geometry_gate_issue)
        return IncrementalExecutionResult(
            ok=False, state=state, issues=issues,
            summary=_build_failure_summary("axial_layers", [axial_geometry_gate_issue.code], 0),
            plan_loop_outcome={"status": "blocked", "active_gate_id": "axial_geometry", "active_stage_id": "plan_gate_axial_geometry", "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": axial_geometry_gate_issue.message},
        )

    # Assembly readiness is deterministic and intentionally precedes the
    # renderer/assembler.  It aggregates all mass-derived grid failures by
    # material so an executor never retries eight overlays as eight causes.
    materials_env = next((item for item in state.patches.values() if item.patch_type == "materials" and item.status == "valid"), None)
    overlays_env = next((item for item in state.patches.values() if item.patch_type == "axial_overlays" and item.status == "valid"), None)
    if materials_env is not None and overlays_env is not None:
        from .material_execution_readiness import validate_material_execution_readiness
        readiness = validate_material_execution_readiness(materials_patch=materials_env.content, axial_overlays_patch=overlays_env.content, policy=str(state.metadata.get("structural_density_policy", "source_only")))
        readiness_path = artifact_writer._write("material_execution_readiness.json", readiness)
        if readiness_path: state.plan_loop_artifacts.append(readiness_path)
        requirements_path = artifact_writer._write("material_execution_requirements.json", readiness.requirements)
        if requirements_path: state.plan_loop_artifacts.append(requirements_path)
        state.add_event("planning.material_execution_preflight_started", "material execution-readiness preflight started", {})
        if readiness.issues:
            readiness_issues = [item.model_dump(mode="json") for item in readiness.issues]
            state.validation_issues.extend(readiness_issues)
            state.add_event("planning.material_density_required", "material density required by mass-derived geometry", {"material_ids": [item.material_id for item in readiness.issues]})
            from .root_cause_classifier import classify_planning_root_causes, record_targeted_retry_attempt
            from .closed_loop.fingerprints import compute_candidate_hash
            from .closed_loop.retry_request_builders import build_retry_request_from_material_readiness
            material_hash = compute_candidate_hash(target_patch_type="materials", candidate_patch=materials_env.content)
            causes = classify_planning_root_causes(readiness_issues, {"materials": material_hash})
            retry_records = [record_targeted_retry_attempt(state, cause) for cause in causes]
            # Phase-3B: also register a typed ExecutablePlanRetryRequest per
            # distinct material so the retry loop can drive a real owner
            # producer with exact required IDs and properties.
            typed_requests = []
            for issue in readiness.issues:
                typed = build_retry_request_from_material_readiness(
                    material_id=issue.material_id,
                    consumer_ids=issue.affected_consumer_ids,
                    required_property=issue.required_property,
                    state=state,
                )
                if typed is not None:
                    typed_requests.append(typed)
            if typed_requests:
                artifact_writer._write("retry_material_readiness_requests.json", typed_requests)
            artifact_writer._write("root_cause_bundle_000.json", causes)
            artifact_writer._write("targeted_retry_trace.json", retry_records)
            if any(record["no_progress"] for record in retry_records):
                state.add_event("planning.retry_no_progress", "same owner candidate repeated for the same root cause", {"records": retry_records})
                artifact_writer._write("no_progress_report.json", retry_records)
            if policy.mode is PlanLoopMode.CONTROLLED:
                state.add_event("planning.assembly_readiness_failed", "assembly blocked before materialization by material readiness", {"codes": [item.code for item in readiness.issues]})
                return IncrementalExecutionResult(ok=False, state=state, issues=[IncrementalExecutionIssue(code=item.code, severity="error", message=f"material {item.material_id} lacks density required by overlays", patch_type="materials") for item in readiness.issues], summary=_build_failure_summary("materials", [item.code for item in readiness.issues], 0), plan_loop_outcome={"status": "blocked", "detail": "material execution readiness failed", "additional_llm_calls_used": state.plan_loop_additional_llm_calls})

    # Assemble.
    assemble_kwargs: dict[str, Any] = {"strict": strict, "resolved_planning_scope": state.resolved_planning_scope}
    if material_policy is not None:
        assemble_kwargs["material_policy"] = material_policy
    validation_issue_count_before_assembly = len(state.validation_issues)
    state = assemble_state_if_ready(state, **assemble_kwargs)
    if state.assembled_plan is not None:
        # Phase-6: Final / Assembled Plan Gate.
        assembled_plan_gate_issue = _run_assembled_plan_gate()
        if assembled_plan_gate_issue is not None:
            issues.append(assembled_plan_gate_issue)
            return IncrementalExecutionResult(
                ok=False, state=state, issues=issues,
                summary=_build_failure_summary("assembled_plan", [assembled_plan_gate_issue.code], 0),
                plan_loop_outcome={"status": "blocked", "active_gate_id": "assembled_plan", "active_stage_id": "plan_gate_assembled_plan", "additional_llm_calls_used": state.plan_loop_additional_llm_calls, "detail": assembled_plan_gate_issue.message},
            )
        state.add_event(
            event_type=EVENT_INCREMENTAL_EXECUTION_COMPLETED,
            message="incremental planning completed, plan assembled",
            data={
                "patch_count": len(state.patches),
                "valid_patch_count": len(state.get_valid_patches()),
                "material_composition_report_present": state.material_composition_report is not None,
            },
        )
        return IncrementalExecutionResult(
            ok=True,
            state=state,
            assembled_plan=state.assembled_plan,
            issues=issues,
            summary={
                "valid_patch_count": len(state.get_valid_patches()),
                "assembled": True,
                "reference_patches_used": reference_patches_used,
                "reference_match_status": state.metadata.get(
                    "reference_match_status",
                    "off" if reference_patch_policy == "off" else "unavailable",
                ),
                "reference_path": state.metadata.get("reference_path"),
                "valid_patch_types": sorted({
                    e.patch_type for e in state.patches.values() if e.status == "valid"
                }),
                "actual_pin_counts": _latest_assembly_summary(state).get("actual_pin_counts", {}),
                "lattice_loading_count": _latest_assembly_summary(state).get("lattice_loading_count", 0),
                "material_aliases_applied": _latest_assembly_summary(state).get("material_aliases_applied", {}),
                "material_composition_policy": _latest_assembly_summary(state).get(
                    "material_composition_policy", "default"
                ),
                "material_composition_report_present": state.material_composition_report is not None,
            },
            plan_loop_outcome=_write_advisory_artifacts(),
        )
    else:
        # ``assemble_state_if_ready`` preserves the assembler's structured
        # diagnostics in ``validation_issues``.  Surface the diagnostics in
        # the executor result as well: callers otherwise only see the generic
        # ``incremental.assembly_failed`` wrapper and cannot route a local
        # repair to the owning patch.
        detailed_assembly_issues: list[IncrementalExecutionIssue] = []
        for raw_issue in state.validation_issues[validation_issue_count_before_assembly:]:
            if not isinstance(raw_issue, dict):
                continue
            code = str(raw_issue.get("code") or "assembly.unknown")
            severity = str(raw_issue.get("severity") or "error")
            if severity not in {"error", "warning", "info"}:
                severity = "error"
            detailed_assembly_issues.append(IncrementalExecutionIssue(
                code=code,
                severity=severity,  # type: ignore[arg-type]
                message=str(raw_issue.get("message") or code),
                patch_type=raw_issue.get("patch_type"),
                patch_id=raw_issue.get("patch_id"),
                path=raw_issue.get("path"),
            ))
        known_issue_codes = {issue.code for issue in issues}
        for detail in detailed_assembly_issues:
            if detail.code not in known_issue_codes:
                issues.append(detail)
                known_issue_codes.add(detail.code)
        issues.append(IncrementalExecutionIssue(
            code="incremental.assembly_failed",
            severity="error",
            message="assembly failed after all patches generated",
        ))
        state.add_event(
            event_type=EVENT_INCREMENTAL_EXECUTION_FAILED,
            message="assembly failed",
            data={},
        )
        return IncrementalExecutionResult(
            ok=False,
            state=state,
            issues=issues,
            summary={
                "assembled": False,
                "failed_stage": "assembly",
                "issue_codes": [i.code for i in issues],
                "reference_match_status": state.metadata.get(
                    "reference_match_status",
                    "off" if reference_patch_policy == "off" else "unavailable",
                ),
                "reference_path": state.metadata.get("reference_path"),
                "reference_patches_used": reference_patches_used,
                "actual_pin_counts": _latest_assembly_summary(state).get("actual_pin_counts", {}),
                "lattice_loading_count": _latest_assembly_summary(state).get("lattice_loading_count", 0),
                "material_aliases_applied": _latest_assembly_summary(state).get("material_aliases_applied", {}),
            },
            plan_loop_outcome=_write_advisory_artifacts(),
        )


__all__ = [
    "IncrementalExecutionIssue",
    "IncrementalExecutionResult",
    "RetryDecision",
    "run_incremental_planning",
    "route_retry",
    "default_patch_task_order",
    "required_patch_types_for_state",
    "build_generation_context_from_state",
    "build_deterministic_settings_patch",
    "EVENT_INCREMENTAL_EXECUTION_STARTED",
    "EVENT_INCREMENTAL_EXECUTION_COMPLETED",
    "EVENT_INCREMENTAL_EXECUTION_FAILED",
    "EVENT_INCREMENTAL_RESUME_STARTED",
    "EVENT_INCREMENTAL_RESUME_COMPLETED",
    "EVENT_PATCH_SKIPPED_ALREADY_VALID",
    "EVENT_PATCH_SKIPPED_FROM_RESUME",
    "EVENT_PATCH_DEPENDENCY_CONTEXT_BUILT",
    "EVENT_PATCH_RETRY_ROUTED",
    "EVENT_DETERMINISTIC_SETTINGS_CREATED",
    "EVENT_REFERENCE_PATCH_LOADED",
    "EVENT_REFERENCE_PATCH_GENERATED",
    "EVENT_REFERENCE_PATCH_FALLBACK",
    "EVENT_REFERENCE_PATCH_VALIDATION_FAILED",
]
