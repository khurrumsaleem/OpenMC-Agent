"""Cross-patch validator for source-driven localized insert placement.

Checks that every ``localized_insert_requirement`` declared in FactsPatch
is fulfilled by matching intents in the assembly_catalog, that profiles
are correctly bound, and that the core layout multiplicity is consistent.

This validator is reactor-neutral: it does NOT hardcode any reactor-specific
names, coordinates, or values. All specifics come from the patches themselves.

Issue codes (all ``severity="error"`` unless noted):

    localized_insert.required_placement_missing
    localized_insert.required_assembly_type_missing
    localized_insert.required_profile_missing
    localized_insert.required_profile_unused
    localized_insert.required_universe_missing
    localized_insert.coordinate_count_mismatch
    localized_insert.coordinates_not_in_host_path
    localized_insert.coordinate_duplicate
    localized_insert.instrument_path_misused
    localized_insert.anchor_mismatch
    localized_insert.control_state_mismatch
    localized_insert.no_absorber_overlap_with_domain
    localized_insert.core_multiplicity_mismatch
    localized_insert.unexpected_assembly_scope
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyTypePatchItem,
    CoreLayoutPatch,
    FactsPatch,
    LocalizedInsertIntentPatchItem,
    LocalizedInsertPlacementRequirementPatchItem,
    LocalizedInsertProfilesPatch,
    UniversesPatch,
    normalized_coords,
)
from openmc_agent.plan_builder.localized_insert_profiles import (
    resolve_profile_for_intent,
)


@dataclass
class PlacementValidationIssue:
    code: str
    severity: str = "error"
    message: str = ""
    requirement_id: str | None = None
    assembly_type_id: str | None = None
    expected: Any = None
    actual: Any = None


@dataclass
class PlacementValidationResult:
    ok: bool = True
    issues: list[PlacementValidationIssue] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.ok = not any(i.severity == "error" for i in self.issues)


def _insert_matches_requirement(
    intent: LocalizedInsertIntentPatchItem,
    req: LocalizedInsertPlacementRequirementPatchItem,
) -> bool:
    """Check if an intent matches a requirement's insert_kind."""
    return intent.insert_kind == req.insert_kind


def _find_matching_intents(
    assembly_type: AssemblyTypePatchItem,
    req: LocalizedInsertPlacementRequirementPatchItem,
) -> list[LocalizedInsertIntentPatchItem]:
    """Find all intents in an assembly type that match a requirement."""
    return [
        intent
        for intent in assembly_type.pin_map.localized_insert_intents
        if _insert_matches_requirement(intent, req)
    ]


def validate_required_localized_insert_placements(
    facts_patch: FactsPatch | None,
    universes_patch: UniversesPatch | None,
    profiles_patch: LocalizedInsertProfilesPatch | None,
    assembly_catalog_patch: AssemblyCatalogPatch | None,
    core_layout_patch: CoreLayoutPatch | None,
    *,
    axial_domain_cm: tuple[float, float] | None = None,
) -> PlacementValidationResult:
    """Validate that all source-driven placement requirements are satisfied.

    Parameters
    ----------
    facts_patch
        Must contain ``localized_insert_requirements``. If None or empty,
        returns ok=True (no requirements to check).
    universes_patch
        Used to verify universe IDs exist.
    profiles_patch
        Used to verify profiles exist and segments reference valid universes.
    assembly_catalog_patch
        Checked for matching localized_insert_intents.
    core_layout_patch
        Used to verify assembly instance counts and multiplicity.
    axial_domain_cm
        Optional detailed-domain bounds for overlap checking.
    """
    issues: list[PlacementValidationIssue] = []

    if facts_patch is None:
        return PlacementValidationResult(ok=True, issues=issues)

    requirements = facts_patch.localized_insert_requirements
    if not requirements:
        return PlacementValidationResult(ok=True, issues=issues)

    # Build lookup tables
    catalog_map: dict[str, AssemblyTypePatchItem] = {}
    if assembly_catalog_patch is not None:
        for at in assembly_catalog_patch.assembly_types:
            catalog_map[at.assembly_type_id] = at

    known_universe_ids: set[str] = set()
    if universes_patch is not None:
        known_universe_ids = {u.universe_id for u in universes_patch.universes}

    profile_map = {}
    if profiles_patch is not None:
        for p in profiles_patch.profiles:
            profile_map[p.profile_id] = p

    # Core layout instance counts
    layout_counts: dict[str, int] = {}
    if core_layout_patch is not None:
        for row in core_layout_patch.assembly_pattern:
            for tid in row:
                layout_counts[tid] = layout_counts.get(tid, 0) + 1

    domain_z_min, domain_z_max = (None, None)
    if axial_domain_cm is not None:
        domain_z_min, domain_z_max = axial_domain_cm

    for req in requirements:
        # 1. Target assembly types exist
        missing_types = [
            tid for tid in req.assembly_type_ids if tid not in catalog_map
        ]
        if missing_types:
            issues.append(PlacementValidationIssue(
                code="localized_insert.required_assembly_type_missing",
                message=f"Required assembly type(s) {missing_types} not found in catalog",
                requirement_id=req.requirement_id,
                expected=req.assembly_type_ids,
                actual=list(catalog_map.keys()),
            ))
            continue

        # 2. Core layout instance count
        if req.expected_assembly_instance_count is not None and core_layout_patch is not None:
            actual_instances = sum(
                layout_counts.get(tid, 0) for tid in req.assembly_type_ids
            )
            if actual_instances != req.expected_assembly_instance_count:
                issues.append(PlacementValidationIssue(
                    code="localized_insert.core_multiplicity_mismatch",
                    message=(
                        f"Expected {req.expected_assembly_instance_count} instances "
                        f"of {req.assembly_type_ids}, found {actual_instances} in core layout"
                    ),
                    requirement_id=req.requirement_id,
                    expected=req.expected_assembly_instance_count,
                    actual=actual_instances,
                ))

        for tid in req.assembly_type_ids:
            at = catalog_map[tid]
            matching_intents = _find_matching_intents(at, req)

            # 3. Matching intent exists
            if not matching_intents:
                issues.append(PlacementValidationIssue(
                    code="localized_insert.required_placement_missing",
                    message=(
                        f"Assembly type '{tid}' has no localized_insert_intent "
                        f"with insert_kind='{req.insert_kind}'"
                    ),
                    requirement_id=req.requirement_id,
                    assembly_type_id=tid,
                    expected=f"intent with insert_kind={req.insert_kind}",
                    actual=[i.insert_id for i in at.pin_map.localized_insert_intents],
                ))
                continue

            # Check each matching intent
            for intent in matching_intents:
                # 4. Coordinate count
                coord_count = len(intent.coordinates)
                if (
                    req.expected_coordinate_count_per_assembly is not None
                    and coord_count != req.expected_coordinate_count_per_assembly
                ):
                    issues.append(PlacementValidationIssue(
                        code="localized_insert.coordinate_count_mismatch",
                        message=(
                            f"Intent '{intent.insert_id}' in '{tid}' has {coord_count} "
                            f"coordinates, expected {req.expected_coordinate_count_per_assembly}"
                        ),
                        requirement_id=req.requirement_id,
                        assembly_type_id=tid,
                        expected=req.expected_coordinate_count_per_assembly,
                        actual=coord_count,
                    ))

                # 5. Coordinates are subset of host path
                host_coords: set[tuple[int, int]] = set()
                if req.host_kind == "guide_tube":
                    host_coords = set(at.pin_map.guide_tube_coords)
                elif req.host_kind == "instrument_tube":
                    host_coords = set(at.pin_map.instrument_tube_coords)

                intent_coord_set = set(intent.coordinates)

                # 6. No duplicate coordinates
                if len(intent_coord_set) < coord_count:
                    issues.append(PlacementValidationIssue(
                        code="localized_insert.coordinate_duplicate",
                        message=f"Intent '{intent.insert_id}' has duplicate coordinates",
                        requirement_id=req.requirement_id,
                        assembly_type_id=tid,
                    ))

                # Check out-of-bounds
                nrows, ncols = at.pin_map.lattice_size
                for r, c in intent.coordinates:
                    if r < 0 or r >= nrows or c < 0 or c >= ncols:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.coordinate_duplicate",
                            message=(
                                f"Intent '{intent.insert_id}' coordinate ({r},{c}) "
                                f"is out of bounds for lattice_size [{nrows},{ncols}]"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                        ))

                # Check subset of host path
                if host_coords:
                    not_in_host = intent_coord_set - host_coords
                    if not_in_host:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.coordinates_not_in_host_path",
                            message=(
                                f"Intent '{intent.insert_id}' coordinates {not_in_host} "
                                f"are not in {req.host_kind} coordinates"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                        ))

                # 7. Instrument tube coords not in RCCA path
                if req.insert_kind == "control_rod" and at.pin_map.instrument_tube_coords:
                    inst_set = set(at.pin_map.instrument_tube_coords)
                    overlap = intent_coord_set & inst_set
                    if overlap:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.instrument_path_misused",
                            message=(
                                f"Control rod intent '{intent.insert_id}' overlaps "
                                f"instrument tube coordinates {overlap}"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                        ))

                # 8-9. Profile reference
                if req.required_profile_id is not None:
                    if intent.axial_profile_id is None:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.required_profile_missing",
                            message=(
                                f"Intent '{intent.insert_id}' must reference "
                                f"profile '{req.required_profile_id}' but has no axial_profile_id"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=req.required_profile_id,
                            actual=None,
                        ))
                    elif intent.axial_profile_id != req.required_profile_id:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.required_profile_missing",
                            message=(
                                f"Intent '{intent.insert_id}' references profile "
                                f"'{intent.axial_profile_id}', expected '{req.required_profile_id}'"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=req.required_profile_id,
                            actual=intent.axial_profile_id,
                        ))
                    elif intent.axial_profile_id not in profile_map:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.required_profile_missing",
                            message=(
                                f"Intent '{intent.insert_id}' references profile "
                                f"'{intent.axial_profile_id}' which does not exist"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                        ))

                # 10. Profile segment universes exist
                if intent.axial_profile_id and intent.axial_profile_id in profile_map:
                    profile = profile_map[intent.axial_profile_id]
                    for seg in profile.segments:
                        if seg.universe_id not in known_universe_ids:
                            issues.append(PlacementValidationIssue(
                                code="localized_insert.required_universe_missing",
                                message=(
                                    f"Profile '{profile.profile_id}' segment '{seg.segment_id}' "
                                    f"references universe '{seg.universe_id}' which does not exist"
                                ),
                                requirement_id=req.requirement_id,
                                assembly_type_id=tid,
                                expected=seg.universe_id,
                                actual=list(known_universe_ids),
                            ))

                # 10b. Expected insert universe IDs exist
                for uid in req.expected_insert_universe_ids:
                    if uid not in known_universe_ids:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.required_universe_missing",
                            message=(
                                f"Required universe '{uid}' for requirement "
                                f"'{req.requirement_id}' does not exist"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=uid,
                            actual=list(known_universe_ids),
                        ))

                # 11. Anchor check
                if req.anchor_z_cm is not None:
                    if intent.anchor_z_cm is None:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.anchor_mismatch",
                            message=(
                                f"Intent '{intent.insert_id}' has no anchor_z_cm, "
                                f"expected {req.anchor_z_cm}"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=req.anchor_z_cm,
                            actual=None,
                        ))
                    elif abs(intent.anchor_z_cm - req.anchor_z_cm) > 1e-3:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.anchor_mismatch",
                            message=(
                                f"Intent '{intent.insert_id}' anchor_z_cm={intent.anchor_z_cm}, "
                                f"expected {req.anchor_z_cm}"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=req.anchor_z_cm,
                            actual=intent.anchor_z_cm,
                        ))

                # 11b. Control state check
                # A control-state binding is meaningful for movable inserts.
                # Static inserts may carry a source-state label on the requirement
                # without needing the intent to duplicate it.
                if req.control_state_id is not None and req.insert_kind == "control_rod":
                    if intent.control_state_id != req.control_state_id:
                        issues.append(PlacementValidationIssue(
                            code="localized_insert.control_state_mismatch",
                            message=(
                                f"Intent '{intent.insert_id}' control_state_id="
                                f"'{intent.control_state_id}', expected '{req.control_state_id}'"
                            ),
                            requirement_id=req.requirement_id,
                            assembly_type_id=tid,
                            expected=req.control_state_id,
                            actual=intent.control_state_id,
                        ))

                # 12-13. Profile resolves and overlaps domain
                if (
                    intent.axial_profile_id
                    and intent.axial_profile_id in profile_map
                    and domain_z_min is not None
                    and domain_z_max is not None
                ):
                    resolved = resolve_profile_for_intent(
                        intent, profile_map, tid
                    )
                    if resolved and resolved.resolved_segments:
                        has_absorber_overlap = False
                        for seg in resolved.resolved_segments:
                            seg_role = seg.role.lower()
                            is_absorber = any(
                                token in seg_role
                                for token in ("absorber", "aic", "b4c", "poison")
                            )
                            if is_absorber:
                                overlap_min = max(seg.absolute_z_min_cm, domain_z_min)
                                overlap_max = min(seg.absolute_z_max_cm, domain_z_max)
                                if overlap_max > overlap_min:
                                    has_absorber_overlap = True

                        if req.required_in_detailed_domain and not has_absorber_overlap:
                            # Check if any required segment role is an absorber
                            absorber_roles = [
                                r for r in req.required_segment_roles
                                if any(t in r.lower() for t in ("absorber", "aic", "b4c", "poison"))
                            ]
                            if absorber_roles:
                                issues.append(PlacementValidationIssue(
                                    code="localized_insert.no_absorber_overlap_with_domain",
                                    message=(
                                        f"Profile '{intent.axial_profile_id}' has no absorber "
                                        f"segment overlapping domain [{domain_z_min}, {domain_z_max}]"
                                    ),
                                    requirement_id=req.requirement_id,
                                    assembly_type_id=tid,
                                ))

        # 14. Required placement not applied to wrong assembly types
        if assembly_catalog_patch is not None:
            for at in assembly_catalog_patch.assembly_types:
                if at.assembly_type_id in req.assembly_type_ids:
                    continue
                wrong_intents = [
                    i for i in at.pin_map.localized_insert_intents
                    if i.insert_kind == req.insert_kind
                    and req.required_profile_id is not None
                    and i.axial_profile_id == req.required_profile_id
                ]
                if wrong_intents:
                    issues.append(PlacementValidationIssue(
                        code="localized_insert.unexpected_assembly_scope",
                        message=(
                            f"Assembly type '{at.assembly_type_id}' has intent "
                            f"'{wrong_intents[0].insert_id}' with insert_kind="
                            f"'{req.insert_kind}' and profile '{req.required_profile_id}' "
                            f"but is not in requirement assembly_type_ids"
                        ),
                        requirement_id=req.requirement_id,
                        assembly_type_id=at.assembly_type_id,
                    ))

    # 15. Required profile is used by at least one intent
    for req in requirements:
        if req.required_profile_id and req.required_profile_id in profile_map:
            profile_used = False
            if assembly_catalog_patch is not None:
                for at in assembly_catalog_patch.assembly_types:
                    for intent in at.pin_map.localized_insert_intents:
                        if intent.axial_profile_id == req.required_profile_id:
                            profile_used = True
                            break
            if not profile_used:
                issues.append(PlacementValidationIssue(
                    code="localized_insert.required_profile_unused",
                    message=(
                        f"Required profile '{req.required_profile_id}' is defined "
                        f"but not referenced by any intent"
                    ),
                    requirement_id=req.requirement_id,
                ))

    return PlacementValidationResult(issues=issues)
