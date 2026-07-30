"""Deterministic, reactor-neutral placement binding evidence.

This module is intentionally about source-contract-to-patch binding, not
assembled OpenMC root reachability.  It consumes only valid incremental patch
envelopes and represents single-assembly and multi-assembly paths uniformly.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from openmc_agent.plan_builder.patches import parse_patch_content

from .fingerprints import compute_candidate_hash, compute_evidence_pack_hash
from .models import (
    PlanClosedLoopPolicy, PlanEvidenceItem, PlacementAssemblyScopeView,
    PlacementBindingView, PlacementContractMatrix, PlacementContractRow,
    PlacementCoreInstanceView, PlacementEvidencePack, PlacementProfileView,
    PlacementRequirementView, PlacementUniverseView,
)


PLACEMENT_PATCH_TYPES = ("facts", "universes", "localized_insert_profiles", "pin_map", "assembly_catalog", "core_layout")


def _valid(state: Any, patch_type: str) -> Any | None:
    matches = [patch for patch in state.patches.values() if patch.patch_type == patch_type and patch.status == "valid"]
    if len(matches) > 1:
        raise ValueError(f"placement.multiple_valid_envelopes:{patch_type}")
    return matches[0] if matches else None


def _patches(state: Any) -> dict[str, Any]:
    return {ptype: env for ptype in PLACEMENT_PATCH_TYPES if (env := _valid(state, ptype)) is not None}


def _facts_content(state: Any) -> dict[str, Any]:
    env = _valid(state, "facts")
    return env.content if env is not None else {}


def placement_gate_applicable(state: Any) -> bool:
    facts = _facts_content(state)
    if facts.get("localized_insert_requirements"):
        return True
    for patch_type in ("pin_map", "assembly_catalog", "localized_insert_profiles"):
        env = _valid(state, patch_type)
        if env is None:
            continue
        content = env.content
        if patch_type == "pin_map" and content.get("localized_insert_intents"):
            return True
        if patch_type == "assembly_catalog" and any(item.get("pin_map", {}).get("localized_insert_intents") for item in content.get("assembly_types", [])):
            return True
        if patch_type == "localized_insert_profiles" and content.get("profiles"):
            return True
    return False


def placement_gate_required_patch_types(state: Any) -> list[str]:
    facts = _facts_content(state)
    scope = facts.get("model_scope")
    multi = scope in {"multi_assembly_core", "full_core"} or _valid(state, "assembly_catalog") is not None
    required = ["facts", "universes"]
    if multi:
        required.extend(["assembly_catalog", "core_layout"])
    else:
        required.append("pin_map")
    needs_profile = any(item.get("required_profile_id") for item in facts.get("localized_insert_requirements", []))
    if not needs_profile:
        pin = _valid(state, "pin_map")
        catalog = _valid(state, "assembly_catalog")
        needs_profile = bool(pin and any(item.get("axial_profile_id") for item in pin.content.get("localized_insert_intents", [])))
        needs_profile = needs_profile or bool(catalog and any(intent.get("axial_profile_id") for item in catalog.content.get("assembly_types", []) for intent in item.get("pin_map", {}).get("localized_insert_intents", [])))
    if needs_profile:
        required.append("localized_insert_profiles")
    return required


def placement_gate_ready(state: Any) -> bool:
    return placement_gate_applicable(state) and all(_valid(state, patch_type) is not None for patch_type in placement_gate_required_patch_types(state))


def build_placement_binding_view(*, state: Any) -> PlacementBindingView:
    patches = _patches(state)
    facts = parse_patch_content("facts", patches["facts"].content) if "facts" in patches else None
    multi = bool(facts and facts.model_scope in {"multi_assembly_core", "full_core"}) or "assembly_catalog" in patches
    requirements = [PlacementRequirementView(
        requirement_id=item.requirement_id, insert_kind=item.insert_kind,
        assembly_type_ids=list(item.assembly_type_ids), expected_coordinate_count=item.expected_coordinate_count_per_assembly,
        expected_assembly_instance_count=item.expected_assembly_instance_count, host_kind=item.host_kind,
        required_profile_id=item.required_profile_id, required_segment_roles=list(item.required_segment_roles),
        expected_universe_ids=list(item.expected_insert_universe_ids), anchor_z_cm=item.anchor_z_cm,
        control_state_id=item.control_state_id, required_in_detailed_domain=item.required_in_detailed_domain,
        requires_human_confirmation=item.requires_human_confirmation,
    ) for item in (facts.localized_insert_requirements if facts else [])]
    scopes: list[PlacementAssemblyScopeView] = []
    conventions: dict[str, Any] = {}
    if multi and "assembly_catalog" in patches:
        catalog = parse_patch_content("assembly_catalog", patches["assembly_catalog"].content)
        counts = Counter()
        if "core_layout" in patches:
            layout = parse_patch_content("core_layout", patches["core_layout"].content)
            for r, row in enumerate(layout.assembly_pattern):
                for c, assembly_type_id in enumerate(row):
                    counts[assembly_type_id] += 1
        for item in catalog.assembly_types:
            pin = item.pin_map
            path = f"/assembly_types/{item.assembly_type_id}/pin_map"
            scopes.append(PlacementAssemblyScopeView(
                scope_id=item.assembly_type_id, source_patch_type="assembly_catalog", source_json_path=path,
                assembly_type_id=item.assembly_type_id, multiplicity=counts.get(item.assembly_type_id, item.multiplicity_hint),
                lattice_size=pin.lattice_size, coordinate_convention=pin.coordinate_convention.model_dump(mode="json"),
                guide_tube_coords=list(pin.guide_tube_coords), instrument_tube_coords=list(pin.instrument_tube_coords),
                localized_insert_intents=[intent.model_dump(mode="json") for intent in pin.localized_insert_intents],
            ))
            conventions[item.assembly_type_id] = pin.coordinate_convention.model_dump(mode="json")
    elif "pin_map" in patches:
        pin = parse_patch_content("pin_map", patches["pin_map"].content)
        scopes.append(PlacementAssemblyScopeView(
            scope_id="single_assembly", source_patch_type="pin_map", source_json_path="/",
            assembly_type_id="single_assembly", multiplicity=1, lattice_size=pin.lattice_size,
            coordinate_convention=pin.coordinate_convention.model_dump(mode="json"),
            guide_tube_coords=list(pin.guide_tube_coords), instrument_tube_coords=list(pin.instrument_tube_coords),
            localized_insert_intents=[intent.model_dump(mode="json") for intent in pin.localized_insert_intents],
        ))
        conventions["single_assembly"] = pin.coordinate_convention.model_dump(mode="json")
    profiles: list[PlacementProfileView] = []
    if "localized_insert_profiles" in patches:
        parsed = parse_patch_content("localized_insert_profiles", patches["localized_insert_profiles"].content)
        profiles = [PlacementProfileView(profile_id=item.profile_id, anchor_kind=item.anchor_kind, anchor_z_cm=item.anchor_z_cm, segments=[segment.model_dump(mode="json") for segment in item.segments]) for item in parsed.profiles]
    universes = []
    if "universes" in patches:
        parsed = parse_patch_content("universes", patches["universes"].content)
        universes = [PlacementUniverseView(universe_id=item.universe_id, kind=item.kind) for item in parsed.universes]
    core_instances: list[PlacementCoreInstanceView] = []
    if "core_layout" in patches:
        layout = parse_patch_content("core_layout", patches["core_layout"].content)
        core_instances = [PlacementCoreInstanceView(assembly_type_id=item, coordinate=(r, c)) for r, row in enumerate(layout.assembly_pattern) for c, item in enumerate(row)]
    return PlacementBindingView(scope_kind="multi_assembly" if multi else "single_assembly", requirements=requirements, assembly_scopes=scopes, profiles=profiles, universes=universes, core_instances=core_instances, coordinate_conventions=conventions)


def _status(issue_codes: list[str], requirement: PlacementRequirementView) -> str:
    if requirement.requires_human_confirmation:
        return "ambiguous"
    return "fail" if issue_codes else "pass"


def _segment_bounds(profile: PlacementProfileView) -> tuple[float | None, float | None]:
    starts = [
        float(segment["relative_z_min_cm"])
        for segment in profile.segments
        if segment.get("relative_z_min_cm") is not None
    ]
    ends = [
        float(segment["relative_z_max_cm"])
        for segment in profile.segments
        if segment.get("relative_z_max_cm") is not None
    ]
    if not starts or not ends:
        return None, None
    return min(starts), max(ends)


def _profile_resolution_summary(
    *,
    scope: PlacementAssemblyScopeView,
    intent: dict[str, Any],
    profile: PlacementProfileView,
) -> dict[str, Any]:
    rel_min, rel_max = _segment_bounds(profile)
    roles = sorted({
        str(segment.get("role"))
        for segment in profile.segments
        if segment.get("role")
    })
    universe_ids = sorted({
        str(segment.get("universe_id"))
        for segment in profile.segments
        if segment.get("universe_id")
    })
    anchor_kind = str(profile.anchor_kind or "absolute")
    anchor = intent.get("anchor_z_cm")
    anchor_source = "intent.anchor_z_cm"
    if anchor is None:
        anchor = profile.anchor_z_cm
        anchor_source = "profile.anchor_z_cm"
    if anchor is None and anchor_kind == "bottom" and intent.get("z_min_cm") is not None and rel_min is not None and abs(rel_min) < 1e-6:
        anchor = intent.get("z_min_cm")
        anchor_source = "intent.z_min_cm"

    resolved_interval = None
    if rel_min is not None and rel_max is not None:
        if anchor_kind == "absolute":
            resolved_interval = [rel_min, rel_max]
        elif anchor is not None:
            anchor_value = float(anchor)
            if anchor_kind == "bottom":
                resolved_interval = [anchor_value + rel_min, anchor_value + rel_max]
            elif anchor_kind == "top":
                resolved_interval = [anchor_value - rel_max, anchor_value - rel_min]
            elif anchor_kind == "center":
                center = (rel_min + rel_max) / 2.0
                resolved_interval = [anchor_value + rel_min - center, anchor_value + rel_max - center]

    return {
        "scope_id": scope.scope_id,
        "intent_id": intent.get("insert_id"),
        "profile_id": profile.profile_id,
        "anchor_kind": anchor_kind,
        "anchor_z_cm": anchor,
        "anchor_source": anchor_source if anchor is not None else "unresolved",
        "intent_envelope_cm": [intent.get("z_min_cm"), intent.get("z_max_cm")],
        "profile_relative_interval_cm": [rel_min, rel_max],
        "resolved_profile_interval_cm": resolved_interval,
        "coordinate_frame_status": "resolved" if resolved_interval is not None else "unresolved_anchor_or_source_frame",
        "roles": roles,
        "universe_ids": universe_ids,
        "coordinate_frame_note": (
            "profile relative_z_* values are profile-frame coordinates; compare "
            "against intent z_min_cm/z_max_cm only after anchor translation"
        ),
    }


def build_placement_contract_matrix(view: PlacementBindingView, issues: list[dict[str, Any]] | None = None) -> PlacementContractMatrix:
    by_requirement: dict[str, list[dict[str, Any]]] = {}
    for issue in issues or []:
        if issue.get("requirement_id"):
            by_requirement.setdefault(str(issue["requirement_id"]), []).append(issue)
    profiles = {profile.profile_id: profile for profile in view.profiles}
    universes = {universe.universe_id for universe in view.universes}
    rows: list[PlacementContractRow] = []
    for req in view.requirements:
        matching_scopes = [scope for scope in view.assembly_scopes if not req.assembly_type_ids or scope.assembly_type_id in req.assembly_type_ids]
        if not matching_scopes and view.scope_kind == "single_assembly" and view.assembly_scopes:
            matching_scopes = list(view.assembly_scopes)
        intents = [(scope, intent) for scope in matching_scopes for intent in scope.localized_insert_intents if intent.get("insert_kind") == req.insert_kind]
        actual_profiles = sorted({str(intent.get("axial_profile_id")) for _, intent in intents if intent.get("axial_profile_id")})
        referenced = set()
        roles = set()
        profile_resolution: list[dict[str, Any]] = []
        if req.required_profile_id and req.required_profile_id in profiles:
            referenced.update(segment.get("universe_id") for segment in profiles[req.required_profile_id].segments if segment.get("universe_id"))
            roles.update(segment.get("role") for segment in profiles[req.required_profile_id].segments if segment.get("role"))
        for scope, intent in intents:
            profile_id = intent.get("axial_profile_id")
            profile = profiles.get(str(profile_id)) if profile_id else None
            if profile is None:
                referenced.add(intent.get("insert_universe_id"))
                continue
            referenced.update(segment.get("universe_id") for segment in profile.segments if segment.get("universe_id"))
            roles.update(segment.get("role") for segment in profile.segments if segment.get("role"))
            profile_resolution.append(_profile_resolution_summary(scope=scope, intent=intent, profile=profile))
        found_issues = by_requirement.get(req.requirement_id, [])
        codes = sorted({str(issue.get("code")) for issue in found_issues})
        rows.append(PlacementContractRow(
            requirement_id=req.requirement_id, insert_kind=req.insert_kind, source_scope=view.scope_kind,
            expected_assembly_type_ids=list(req.assembly_type_ids), actual_assembly_type_ids=[scope.assembly_type_id or "single_assembly" for scope in matching_scopes],
            expected_instance_count=req.expected_assembly_instance_count,
            actual_instance_count=sum((scope.multiplicity or 0) for scope in matching_scopes) if view.scope_kind == "multi_assembly" else 1,
            expected_coordinate_count=req.expected_coordinate_count,
            actual_coordinate_counts={scope.scope_id: len(intent.get("coordinates", [])) for scope, intent in intents},
            host_kind=req.host_kind, host_coordinate_counts={scope.scope_id: len(scope.guide_tube_coords if req.host_kind == "guide_tube" else scope.instrument_tube_coords) for scope in matching_scopes},
            matching_intent_ids=[str(intent.get("insert_id")) for _, intent in intents], required_profile_id=req.required_profile_id,
            actual_profile_ids=actual_profiles, required_segment_roles=list(req.required_segment_roles), actual_segment_roles=sorted(str(role) for role in roles),
            actual_profile_resolution=profile_resolution,
            expected_universe_ids=list(req.expected_universe_ids), referenced_universe_ids=sorted(str(value) for value in referenced if value),
            missing_universe_ids=sorted(set(req.expected_universe_ids) - universes), anchor_expected=req.anchor_z_cm,
            anchor_actual={scope.scope_id: intent.get("anchor_z_cm") for scope, intent in intents},
            control_state_expected=req.control_state_id, control_state_actual={scope.scope_id: intent.get("control_state_id") for scope, intent in intents},
            coordinate_convention_status="pass" if len({str(scope.coordinate_convention) for scope in matching_scopes}) <= 1 else "ambiguous",
            static_binding_status=_status(codes, req), issue_codes=codes,
        ))
    matrix = PlacementContractMatrix(rows=rows)
    matrix.input_hash = compute_evidence_pack_hash(matrix)
    return matrix


def placement_gate_input_hash(state: Any) -> str:
    view = build_placement_binding_view(state=state)
    matrix = build_placement_contract_matrix(view)
    patch_hashes = {ptype: compute_candidate_hash(target_patch_type=ptype, candidate_patch=env.content) for ptype, env in _patches(state).items()}
    return compute_evidence_pack_hash({"patch_hashes": patch_hashes, "matrix": matrix, "coordinate_conventions": view.coordinate_conventions})


def build_placement_evidence_pack(*, state: Any, policy: PlanClosedLoopPolicy, deterministic_issues: list[dict[str, Any]] | None = None) -> PlacementEvidencePack:
    patches = _patches(state)
    view = build_placement_binding_view(state=state)
    matrix = build_placement_contract_matrix(view, deterministic_issues)
    patch_hashes = {ptype: compute_candidate_hash(target_patch_type=ptype, candidate_patch=env.content) for ptype, env in sorted(patches.items())}
    items: list[PlanEvidenceItem] = []
    index = 1
    def add(kind: str, prefix: str, patch_type: str | None, path: str | None, label: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        nonlocal index
        canonical_hash = compute_evidence_pack_hash({"kind": kind, "patch_type": patch_type, "path": path, "value": value})
        items.append(PlanEvidenceItem(ref_id=f"{prefix}{index:03d}", evidence_kind=kind, patch_type=patch_type, json_path=path, label=label, value=value, canonical_hash=canonical_hash, metadata=metadata or {}))
        index += 1
    for requirement in view.requirements:
        add("accepted_fact_contract", "F", "facts", "/localized_insert_requirements", f"accepted placement requirement {requirement.requirement_id}", requirement.model_dump(mode="json"), {"evidence_origin": "accepted_facts_contract"})
    for scope in view.assembly_scopes:
        add("patch_fragment", "A", scope.source_patch_type, scope.source_json_path, f"placement scope {scope.scope_id}", scope.model_dump(mode="json"))
    if "core_layout" in patches:
        add("patch_fragment", "C", "core_layout", "/core_layout", "core layout", patches["core_layout"].content)
    for profile in view.profiles:
        add("patch_fragment", "P", "localized_insert_profiles", f"/profiles/{profile.profile_id}", f"localized profile {profile.profile_id}", profile.model_dump(mode="json"))
    for universe in view.universes:
        add("patch_fragment", "U", "universes", f"/universes/{universe.universe_id}", f"placement universe {universe.universe_id}", universe.model_dump(mode="json"))
    for row in matrix.rows:
        add("contract_matrix_row", "D", None, None, f"contract matrix {row.requirement_id}", row.model_dump(mode="json"))
    pack = PlacementEvidencePack(
        placement_scope_kind=view.scope_kind, evidence_items=items, contract_matrix=matrix,
        deterministic_issues=list(deterministic_issues or []), relevant_patch_hashes=patch_hashes,
        required_patch_types=placement_gate_required_patch_types(state),
        optional_patch_types=[ptype for ptype in ("localized_insert_profiles",) if ptype not in placement_gate_required_patch_types(state)],
        accepted_facts_hash=patch_hashes.get("facts", ""), coordinate_convention_summary=view.coordinate_conventions,
    )
    pack.input_hash = compute_evidence_pack_hash(pack)
    return pack
