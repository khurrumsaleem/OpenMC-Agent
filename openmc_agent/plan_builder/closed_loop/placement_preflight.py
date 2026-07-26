"""Deterministic placement checks; the critic never recomputes these facts."""

from __future__ import annotations

from typing import Any

from openmc_agent.plan_builder.required_placement_validator import validate_required_localized_insert_placements

from .placement_evidence import build_placement_binding_view, placement_gate_ready


def _issue(code: str, message: str, requirement_id: str | None = None, *, expected: Any = None, actual: Any = None, severity: str = "error") -> dict[str, Any]:
    return {"code": code, "severity": severity, "blocking": severity == "error", "message": message, "requirement_id": requirement_id, "expected": expected, "actual": actual}


def _roles_satisfy(expected: set[str], actual: set[str]) -> bool:
    """Allow representation-equivalent structural role labels.

    A source can call a terminal structural segment ``end_structure`` while a
    patch schema calls it ``upper_endplug``.  This is a generic role-family
    equivalence, not a reactor or benchmark rule.
    """
    for requested in expected:
        if requested in actual:
            continue
        tokens = set(requested.split("_"))
        if "end" in tokens and any("end" in value and ("plug" in value or "structure" in value) for value in actual):
            continue
        return False
    return True


def validate_placement_binding_view(view: Any) -> list[dict[str, Any]]:
    """Validate invariant binding facts for both plan scope representations."""
    issues: list[dict[str, Any]] = []
    profiles = {item.profile_id: item for item in view.profiles}
    universe_ids = {item.universe_id for item in view.universes}
    for requirement in view.requirements:
        scopes = [scope for scope in view.assembly_scopes if not requirement.assembly_type_ids or scope.assembly_type_id in requirement.assembly_type_ids]
        if not scopes:
            issues.append(_issue("localized_insert.required_assembly_type_missing", "required assembly scope is absent", requirement.requirement_id, expected=requirement.assembly_type_ids))
            continue
        if requirement.expected_assembly_instance_count is not None and view.scope_kind == "multi_assembly":
            actual = sum(scope.multiplicity or 0 for scope in scopes)
            if actual != requirement.expected_assembly_instance_count:
                issues.append(_issue("localized_insert.core_multiplicity_mismatch", "core instance multiplicity differs from accepted Facts contract", requirement.requirement_id, expected=requirement.expected_assembly_instance_count, actual=actual))
        for scope in scopes:
            intents = [item for item in scope.localized_insert_intents if item.get("insert_kind") == requirement.insert_kind]
            if not intents:
                issues.append(_issue("localized_insert.required_placement_missing", "required localized insert intent is absent", requirement.requirement_id, expected=requirement.insert_kind, actual=[]))
                continue
            for intent in intents:
                coords = [tuple(item) for item in intent.get("coordinates", [])]
                if requirement.expected_coordinate_count is not None and len(coords) != requirement.expected_coordinate_count:
                    issues.append(_issue("localized_insert.coordinate_count_mismatch", "localized intent coordinate count differs from contract", requirement.requirement_id, expected=requirement.expected_coordinate_count, actual=len(coords)))
                if len(set(coords)) != len(coords):
                    issues.append(_issue("localized_insert.coordinate_duplicate", "localized intent has duplicate coordinates", requirement.requirement_id))
                host = scope.guide_tube_coords if requirement.host_kind == "guide_tube" else scope.instrument_tube_coords if requirement.host_kind == "instrument_tube" else []
                if host and not set(coords).issubset(set(host)):
                    issues.append(_issue("localized_insert.coordinates_not_in_host_path", "localized intent coordinates are outside the required host path", requirement.requirement_id))
                if requirement.insert_kind == "control_rod" and set(coords) & set(scope.instrument_tube_coords):
                    issues.append(_issue("localized_insert.instrument_path_misused", "control insert overlaps instrument path", requirement.requirement_id))
                if requirement.required_profile_id:
                    profile_id = intent.get("axial_profile_id")
                    if profile_id != requirement.required_profile_id or profile_id not in profiles:
                        issues.append(_issue("localized_insert.required_profile_missing", "required profile is absent or not used by intent", requirement.requirement_id, expected=requirement.required_profile_id, actual=profile_id))
                    else:
                        profile = profiles[profile_id]
                        segment_universes = {segment.get("universe_id") for segment in profile.segments}
                        segment_roles = {segment.get("role") for segment in profile.segments}
                        if requirement.required_segment_roles and not _roles_satisfy(set(requirement.required_segment_roles), segment_roles):
                            issues.append(_issue("localized_insert.required_profile_unused", "profile lacks required segment role", requirement.requirement_id))
                        missing = set(requirement.expected_universe_ids) - universe_ids
                        if missing or not segment_universes.issubset(universe_ids):
                            issues.append(_issue("localized_insert.required_universe_missing", "required localized insert universe is absent", requirement.requirement_id, expected=sorted(requirement.expected_universe_ids), actual=sorted(universe_ids)))
                elif intent.get("insert_universe_id") not in universe_ids:
                    issues.append(_issue("localized_insert.required_universe_missing", "intent references an unavailable universe", requirement.requirement_id, actual=intent.get("insert_universe_id")))
                if requirement.anchor_z_cm is not None and intent.get("anchor_z_cm") != requirement.anchor_z_cm:
                    issues.append(_issue("localized_insert.anchor_mismatch", "intent anchor differs from accepted Facts contract", requirement.requirement_id, expected=requirement.anchor_z_cm, actual=intent.get("anchor_z_cm")))
                # Static inserts may carry a source-state label in Facts, but
                # only movable control inserts need the intent to bind a
                # runtime control state.  Keep this aligned with
                # required_placement_validator to avoid deterministic false
                # positives for stationary Pyrex / thimble plug placements.
                if (
                    requirement.control_state_id is not None
                    and requirement.insert_kind == "control_rod"
                    and intent.get("control_state_id") != requirement.control_state_id
                ):
                    issues.append(_issue("localized_insert.control_state_mismatch", "intent control state differs from accepted Facts contract", requirement.requirement_id, expected=requirement.control_state_id, actual=intent.get("control_state_id")))
    return issues


def run_placement_preflight(*, state: Any) -> dict[str, Any]:
    """Return JSON-safe deterministic issues and never invoke an LLM."""
    if not placement_gate_ready(state):
        return {"ok": False, "issues": [_issue("placement.required_patch_missing", "placement gate inputs are not ready")], "binding_view": None}
    view = build_placement_binding_view(state=state)
    issues = validate_placement_binding_view(view)
    # Preserve the established catalog validator as a second, backwards
    # compatible check.  Single-assembly validation is supplied by the view.
    if view.scope_kind == "multi_assembly":
        from openmc_agent.plan_builder.patches import parse_patch_content
        valid = {item.patch_type: item for item in state.patches.values() if item.status == "valid"}
        result = validate_required_localized_insert_placements(
            parse_patch_content("facts", valid["facts"].content), parse_patch_content("universes", valid["universes"].content),
            parse_patch_content("localized_insert_profiles", valid["localized_insert_profiles"].content) if "localized_insert_profiles" in valid else None,
            parse_patch_content("assembly_catalog", valid["assembly_catalog"].content), parse_patch_content("core_layout", valid["core_layout"].content),
        )
        for item in result.issues:
            payload = _issue(item.code, item.message, item.requirement_id, expected=item.expected, actual=item.actual, severity=item.severity)
            if not any(existing["code"] == payload["code"] and existing.get("requirement_id") == payload.get("requirement_id") for existing in issues):
                issues.append(payload)
    return {"ok": not any(item["severity"] == "error" for item in issues), "issues": issues, "binding_view": view.model_dump(mode="json")}
