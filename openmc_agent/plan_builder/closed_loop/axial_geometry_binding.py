"""Axial Geometry static binding view.

Builds a deterministic, reactor-neutral view of the accepted upstream
contracts (Facts / Material-Universe / Placement) -> axial patches
(base_path_axial_profiles / axial_layers / axial_overlays) edge.

This is *static* coverage and occupancy analysis — it does not claim that
the assembled root universe is reachable, only that finite z-intervals,
references, and through-path structures are internally consistent.
"""

from __future__ import annotations

from typing import Any

from openmc_agent.plan_builder.patches import parse_patch_content

from .fingerprints import compute_candidate_hash
from .models import (
    AxialDerivedSegment,
    AxialGeometryBindingView,
    AxialLayerRecord,
    AxialOverlayRecord,
    AxialReferenceEdge,
    BasePathProfileRecord,
    BasePathSegmentRecord,
    LatticeLoadingRecord,
    LocalizedInsertAxialRecord,
    SourceAxialContract,
    ThroughPathRecord,
)


_AXIAL_PATCH_TYPES = ("base_path_axial_profiles", "axial_layers", "axial_overlays")
_Z_TOL = 1e-6


def _valid(state: Any, patch_type: str) -> Any | None:
    matches = [item for item in state.patches.values() if item.patch_type == patch_type and item.status == "valid"]
    if len(matches) > 1:
        raise ValueError(f"axial_geometry.multiple_valid_envelopes:{patch_type}")
    return matches[0] if matches else None


def _hash(state: Any, patch_type: str) -> str:
    env = _valid(state, patch_type)
    return compute_candidate_hash(target_patch_type=patch_type, candidate_patch=env.content) if env else ""


def _accepted_hash(state: Any, stage_key: str) -> str:
    stage = state.plan_loop_stages.get(stage_key)
    if stage is None:
        return ""
    return str(stage.metadata.get("accepted_input_hash", ""))


def _extract_source_axial_contracts(facts: Any) -> list[SourceAxialContract]:
    contracts: list[SourceAxialContract] = []
    if facts is None:
        return contracts
    fc = getattr(facts, "planning_feature_contract", None) or facts
    axial_domain = getattr(fc, "axial_domain_cm", None)
    active_fuel = getattr(fc, "active_fuel_region_cm", None)
    roles: list[str] = []
    if getattr(fc, "has_axial_geometry", False):
        roles.append("axial_geometry")
    if getattr(fc, "has_spacer_grids", False):
        roles.append("spacer_grid")
    grid_count = getattr(fc, "expected_spacer_grid_count", None)
    contracts.append(SourceAxialContract(
        requirement_id="source:axial_domain",
        axial_domain_cm=tuple(axial_domain) if axial_domain else None,
        active_fuel_region_cm=tuple(active_fuel) if active_fuel else None,
        required_axial_roles=roles,
        spacer_grid_intervals=[],
        finite_axial_model=bool(axial_domain is not None),
        allows_homogenization=False,
        allows_clipping=False,
        metadata={"expected_spacer_grid_count": grid_count},
    ))
    return contracts


def _build_base_path_profiles(profiles_patch: Any) -> list[BasePathProfileRecord]:
    records: list[BasePathProfileRecord] = []
    if profiles_patch is None:
        return records
    for profile in getattr(profiles_patch, "profiles", []):
        segments: list[BasePathSegmentRecord] = []
        for binding in getattr(profile, "state_bindings", []):
            role = getattr(binding, "axial_role", "")
            uid = getattr(binding, "replacement_universe_id", None) or next(iter(getattr(binding, "source_universe_ids", []) or []), None)
            segments.append(BasePathSegmentRecord(
                segment_id=f"{profile.profile_id}:{role}",
                universe_id=uid,
                role=role,
            ))
        records.append(BasePathProfileRecord(
            profile_id=profile.profile_id,
            source_requirement_id=getattr(profile, "source_note", ""),
            anchor_kind="base_path",
            segment_ids=[s.segment_id for s in segments],
            segments=segments,
            referenced_universe_ids=sorted({s.universe_id for s in segments if s.universe_id}),
            coverage_status="unverified",
        ))
    return records


def _build_axial_layer_records(layers_patch: Any) -> list[AxialLayerRecord]:
    records: list[AxialLayerRecord] = []
    if layers_patch is None:
        return records
    for layer in getattr(layers_patch, "layers", []):
        z_min = getattr(layer, "z_min_cm", None)
        z_max = getattr(layer, "z_max_cm", None)
        thickness = None
        if z_min is not None and z_max is not None:
            thickness = round(z_max - z_min, 6)
        loading_ids = list(getattr(layer, "loading_ids", []) or [])
        lid = getattr(layer, "loading_id", None)
        if lid and lid not in loading_ids:
            loading_ids.insert(0, lid)
        records.append(AxialLayerRecord(
            layer_id=layer.layer_id,
            role=getattr(layer, "role", ""),
            z_min_cm=z_min,
            z_max_cm=z_max,
            thickness_cm=thickness,
            fill_type=getattr(layer, "fill_type", "unknown"),
            fill_id=getattr(layer, "fill_id", None),
            loading_ids=loading_ids,
            coverage_status="unverified",
        ))
    return records


def _build_loading_records(layers_patch: Any) -> list[LatticeLoadingRecord]:
    records: list[LatticeLoadingRecord] = []
    if layers_patch is None:
        return records
    layer_index: dict[str, AxialLayerRecord] = {r.layer_id: r for r in _build_axial_layer_records(layers_patch)}
    seen: set[str] = set()
    for layer in getattr(layers_patch, "layers", []):
        lids = list(getattr(layer, "loading_ids", []) or [])
        lid = getattr(layer, "loading_id", None)
        if lid and lid not in lids:
            lids.append(lid)
        for loading_id in lids:
            if loading_id in seen:
                continue
            seen.add(loading_id)
    for loading in getattr(layers_patch, "lattice_loadings", []):
        loading_id = loading.loading_id
        seen.discard(loading_id)
        transformations = list(getattr(loading, "transformations", []) or [])
        replacement_universes = sorted({
            getattr(t, "replacement_universe_id", None)
            for t in transformations
            if getattr(t, "replacement_universe_id", None)
        })
        attached = sorted({
            layer.layer_id for layer in getattr(layers_patch, "layers", [])
            if loading_id in (list(getattr(layer, "loading_ids", []) or []) or [getattr(layer, "loading_id", None)])
        })
        records.append(LatticeLoadingRecord(
            loading_id=loading_id,
            base_lattice_id=getattr(loading, "base_lattice_id", None),
            attached_layer_ids=attached,
            transformation_ids=[getattr(t, "operation_id", f"op{i}") for i, t in enumerate(transformations)],
            referenced_universe_ids=replacement_universes,
            attachment_status="attached" if attached else "unattached",
        ))
    for orphan_id in sorted(seen):
        records.append(LatticeLoadingRecord(
            loading_id=orphan_id,
            attached_layer_ids=[],
            attachment_status="unattached",
        ))
    return records


def _build_overlay_records(overlays_patch: Any) -> list[AxialOverlayRecord]:
    records: list[AxialOverlayRecord] = []
    if overlays_patch is None:
        return records
    for overlay in getattr(overlays_patch, "overlays", []):
        z_min = getattr(overlay, "z_min_cm", None)
        z_max = getattr(overlay, "z_max_cm", None)
        thickness = None
        if z_min is not None and z_max is not None:
            thickness = round(z_max - z_min, 6)
        needs_density = getattr(overlay, "geometry_mode", "") == "mass_conserving_outer_frame"
        density = getattr(overlay, "effective_density_g_cm3", None)
        total_mass_g = getattr(overlay, "total_mass_g", None)
        has_mass_conserving_basis = (
            (density is not None and density > 0)
            or (total_mass_g is not None and total_mass_g > 0)
        )
        records.append(AxialOverlayRecord(
            overlay_id=overlay.overlay_id,
            overlay_kind=getattr(overlay, "overlay_kind", ""),
            z_min_cm=z_min,
            z_max_cm=z_max,
            thickness_cm=thickness,
            target_lattice_id=getattr(overlay, "target_lattice_id", None),
            material_id=getattr(overlay, "material_id", None),
            geometry_mode=getattr(overlay, "geometry_mode", ""),
            required_density=density if needs_density else None,
            density_status="pass" if not needs_density or has_mass_conserving_basis else ("fail" if needs_density else "not_applicable"),
            structural_renderability="unverified",
            preserved_through_path_ids=[overlay.overlay_id] if getattr(overlay, "through_path_preserved", True) else [],
        ))
    return records


def _build_localized_insert_axial_records(
    facts: Any, localized_profiles_patch: Any, layers_patch: Any,
) -> list[LocalizedInsertAxialRecord]:
    records: list[LocalizedInsertAxialRecord] = []
    if facts is None:
        return records
    axial_domain = getattr(facts, "axial_domain_cm", None) or (0.0, 0.0)
    profiles_by_id = {
        str(getattr(profile, "profile_id", "")): profile
        for profile in (getattr(localized_profiles_patch, "profiles", []) if localized_profiles_patch else [])
        if getattr(profile, "profile_id", "")
    }
    for req in getattr(facts, "localized_insert_requirements", []):
        rid = req.requirement_id
        profile_id = str(getattr(req, "required_profile_id", None) or "")
        profile = profiles_by_id.get(profile_id) if profile_id else None
        anchor_z: float | None = getattr(req, "anchor_z_cm", None)
        if anchor_z is None and profile is not None:
            anchor_z = getattr(profile, "anchor_z_cm", None)
        control_state_id = str(getattr(req, "control_state_id", None) or "")
        translated: tuple[float, float] | None = None
        profile_extent: tuple[float, float] | None = None
        segment_roles: list[str] = []
        segment_universe_ids: list[str] = []
        if profile is not None:
            segments = list(getattr(profile, "segments", []) or [])
            if segments:
                starts = [float(getattr(segment, "relative_z_min_cm")) for segment in segments]
                ends = [float(getattr(segment, "relative_z_max_cm")) for segment in segments]
                profile_extent = (min(starts), max(ends))
                segment_roles = [str(getattr(segment, "role", "")) for segment in segments if getattr(segment, "role", "")]
                segment_universe_ids = [str(getattr(segment, "universe_id", "")) for segment in segments if getattr(segment, "universe_id", "")]
        if anchor_z is not None:
            if profile_extent is None:
                translated = (anchor_z, anchor_z)
            else:
                anchor_kind = str(getattr(profile, "anchor_kind", "bottom")) if profile is not None else "bottom"
                if anchor_kind == "top":
                    translated = (anchor_z - profile_extent[1], anchor_z - profile_extent[0])
                elif anchor_kind == "center":
                    half = (profile_extent[1] - profile_extent[0]) / 2.0
                    translated = (anchor_z - half, anchor_z + half)
                else:
                    translated = (anchor_z + profile_extent[0], anchor_z + profile_extent[1])
        host_layers: list[str] = []
        overlaps: list[tuple[float, float]] = []
        if layers_patch and translated:
            for layer in getattr(layers_patch, "layers", []):
                z_min = getattr(layer, "z_min_cm", None)
                z_max = getattr(layer, "z_max_cm", None)
                if z_min is not None and z_max is not None:
                    if _intervals_overlap(translated, (z_min, z_max), _Z_TOL):
                        host_layers.append(layer.layer_id)
                        overlaps.append((max(float(z_min), translated[0]), min(float(z_max), translated[1])))
        records.append(LocalizedInsertAxialRecord(
            requirement_id=rid,
            profile_id=profile_id,
            control_state_id=control_state_id,
            anchor_z_cm=anchor_z,
            profile_relative_extent=profile_extent,
            translated_absolute_extent=translated,
            segment_roles=segment_roles or ([req.insert_kind] if hasattr(req, "insert_kind") else []),
            segment_universe_ids=segment_universe_ids,
            host_layer_ids=host_layers,
            overlapping_layer_intervals=overlaps,
            clipping="reachable" if host_layers else ("outside_domain" if translated else "pending"),
            coverage_status="pass" if host_layers else "fail",
        ))
    return records


def _build_through_path_records(
    layers: list[AxialLayerRecord], overlays: list[AxialOverlayRecord],
) -> list[ThroughPathRecord]:
    records: list[ThroughPathRecord] = []
    for layer in layers:
        if layer.fill_type == "lattice":
            records.append(ThroughPathRecord(
                through_path_id=f"tp:{layer.layer_id}",
                path_kind="base_lattice",
                preserved=True,
                issue_codes=[],
            ))
        elif layer.fill_type in {"material", "universe"} and layer.role in {"spacer_grid", "grid", "structural_slab"}:
            records.append(ThroughPathRecord(
                through_path_id=f"tp:{layer.layer_id}",
                path_kind="material_slab",
                preserved=False,
                issue_codes=["axial.through_path_fuel_interrupted"],
            ))
    for overlay in overlays:
        if overlay.geometry_mode in {"homogenized_open_region", "mass_conserving_outer_frame", "skeleton"}:
            records.append(ThroughPathRecord(
                through_path_id=f"tp:{overlay.overlay_id}",
                path_kind=overlay.geometry_mode,
                overlay_band_ids=[overlay.overlay_id],
                preserved=bool(overlay.preserved_through_path_ids),
                issue_codes=[] if overlay.preserved_through_path_ids else ["axial.overlay_through_path_not_preserved"],
            ))
    return records


def _intervals_overlap(a: tuple[float, float], b: tuple[float, float], tol: float = 0.0) -> bool:
    return a[0] < b[1] - tol and b[0] < a[1] - tol


def derive_axial_geometry_segments(
    *,
    axial_domain: tuple[float, float] | None,
    layers: list[AxialLayerRecord],
    overlays: list[AxialOverlayRecord],
    profiles: list[BasePathProfileRecord],
    inserts: list[LocalizedInsertAxialRecord],
    tol: float = _Z_TOL,
) -> list[AxialDerivedSegment]:
    """Merge all axial boundaries into sorted, finite, non-zero segments.

    Reuses the boundary-merge concept from ``compute_axial_segments`` but
    operates on the binding-view records so it is independent of the
    assembled SimulationPlan.  Zero-thickness segments are never emitted.
    """
    boundaries: set[float] = set()
    if axial_domain:
        boundaries.add(float(axial_domain[0]))
        boundaries.add(float(axial_domain[1]))
    for layer in layers:
        if layer.z_min_cm is not None:
            boundaries.add(float(layer.z_min_cm))
        if layer.z_max_cm is not None:
            boundaries.add(float(layer.z_max_cm))
    for overlay in overlays:
        if overlay.z_min_cm is not None:
            boundaries.add(float(overlay.z_min_cm))
        if overlay.z_max_cm is not None:
            boundaries.add(float(overlay.z_max_cm))
    for insert in inserts:
        if insert.translated_absolute_extent:
            boundaries.add(float(insert.translated_absolute_extent[0]))
            boundaries.add(float(insert.translated_absolute_extent[1]))
    sorted_bounds = sorted(boundaries)
    segments: list[AxialDerivedSegment] = []
    for i in range(len(sorted_bounds) - 1):
        z0, z1 = sorted_bounds[i], sorted_bounds[i + 1]
        if z1 - z0 <= tol:
            continue
        active_layers = [l.layer_id for l in layers if l.z_min_cm is not None and l.z_max_cm is not None and z0 >= l.z_min_cm - tol and z1 <= l.z_max_cm + tol]
        active_overlays = [o.overlay_id for o in overlays if o.z_min_cm is not None and o.z_max_cm is not None and z0 >= o.z_min_cm - tol and z1 <= o.z_max_cm + tol]
        base_layer = next((l for l in layers if l.layer_id in active_layers), None)
        segments.append(AxialDerivedSegment(
            segment_id=f"seg:{i}",
            z_min_cm=round(z0, 6),
            z_max_cm=round(z1, 6),
            active_layer_ids=active_layers,
            active_overlay_ids=active_overlays,
            base_fill_type=base_layer.fill_type if base_layer else "",
            base_fill_id=base_layer.fill_id if base_layer else None,
        ))
    return segments


def build_axial_geometry_binding_view(*, state: Any) -> AxialGeometryBindingView:
    """Construct the AxialGeometryBindingView from the current PlanBuildState."""
    facts_env = _valid(state, "facts")
    facts = facts_env.content if facts_env is not None else None
    facts_obj = facts
    if facts is not None and not hasattr(facts, "localized_insert_requirements"):
        facts_obj = parse_patch_content("facts", facts)

    profiles_patch = _valid(state, "base_path_axial_profiles")
    profiles_content = profiles_patch.content if profiles_patch is not None else None
    localized_profiles_patch = _valid(state, "localized_insert_profiles")
    localized_profiles_content = localized_profiles_patch.content if localized_profiles_patch is not None else None
    layers_patch = _valid(state, "axial_layers")
    layers_content = layers_patch.content if layers_patch is not None else None
    overlays_patch = _valid(state, "axial_overlays")
    overlays_content = overlays_patch.content if overlays_patch is not None else None

    profiles_obj = parse_patch_content("base_path_axial_profiles", profiles_content) if profiles_content else None
    localized_profiles_obj = parse_patch_content("localized_insert_profiles", localized_profiles_content) if localized_profiles_content else None
    layers_obj = parse_patch_content("axial_layers", layers_content) if layers_content else None
    overlays_obj = parse_patch_content("axial_overlays", overlays_content) if overlays_content else None

    source_contracts = _extract_source_axial_contracts(facts_obj)
    axial_domain = source_contracts[0].axial_domain_cm if source_contracts else None
    active_fuel = source_contracts[0].active_fuel_region_cm if source_contracts else None

    profile_records = _build_base_path_profiles(profiles_obj)
    layer_records = _build_axial_layer_records(layers_obj)
    loading_records = _build_loading_records(layers_obj)
    overlay_records = _build_overlay_records(overlays_obj)
    insert_records = _build_localized_insert_axial_records(facts_obj, localized_profiles_obj, layers_obj)
    through_path_records = _build_through_path_records(layer_records, overlay_records)
    derived_segments = derive_axial_geometry_segments(
        axial_domain=axial_domain, layers=layer_records, overlays=overlay_records,
        profiles=profile_records, inserts=insert_records,
    )

    _scope_obj = getattr(state, "resolved_planning_scope", None)
    if _scope_obj is None:
        scope = "single_assembly"
    elif isinstance(_scope_obj, str):
        scope = _scope_obj
    else:
        # ResolvedPlanningScope or similar: extract a string representation.
        scope = str(getattr(_scope_obj, "model_scope", None) or getattr(_scope_obj, "status", None) or "unknown")
    feature_contract = getattr(facts_obj, "planning_feature_contract", None)
    fc_hash = compute_candidate_hash(target_patch_type="facts", candidate_patch={"feature_contract": feature_contract.model_dump(mode="json") if feature_contract else {}}) if feature_contract else ""
    ctp_hash = getattr(state, "canonical_task_plan", None)
    ctp_hash_str = ctp_hash.plan_hash if ctp_hash is not None else ""

    unresolved: list[str] = []
    for layer in layer_records:
        if layer.fill_id and layer.fill_type in {"material", "universe", "lattice"}:
            unresolved.append(f"{layer.fill_type}:{layer.fill_id}")

    return AxialGeometryBindingView(
        planning_scope=scope,
        axial_domain_cm=axial_domain,
        active_fuel_region_cm=active_fuel,
        facts_patch_hash=_hash(state, "facts"),
        materials_patch_hash=_hash(state, "materials"),
        universes_patch_hash=_hash(state, "universes"),
        placement_input_hash="",
        material_universe_accepted_hash=_accepted_hash(state, "plan_gate_material_universe"),
        placement_accepted_hash=_accepted_hash(state, "plan_gate_placement"),
        base_path_profiles_hash=_hash(state, "base_path_axial_profiles"),
        axial_layers_hash=_hash(state, "axial_layers"),
        axial_overlays_hash=_hash(state, "axial_overlays"),
        feature_contract_hash=fc_hash,
        canonical_task_plan_hash=ctp_hash_str,
        source_axial_contracts=source_contracts,
        base_path_profile_records=profile_records,
        axial_layer_records=layer_records,
        lattice_loading_records=loading_records,
        axial_overlay_records=overlay_records,
        localized_insert_axial_records=insert_records,
        through_path_records=through_path_records,
        derived_segments=derived_segments,
        unresolved_references=sorted(set(unresolved)),
    )


__all__ = [
    "build_axial_geometry_binding_view",
    "derive_axial_geometry_segments",
]
