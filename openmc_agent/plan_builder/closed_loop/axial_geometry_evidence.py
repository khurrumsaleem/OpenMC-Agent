"""Axial Geometry evidence pack, contract matrix, and gate applicability.

Mirrors the Material-Universe evidence module but for the
accepted-upstream -> axial-patches static edge.
"""

from __future__ import annotations

from typing import Any

from .axial_geometry_binding import (
    _hash,
    _valid,
    build_axial_geometry_binding_view,
    derive_axial_geometry_segments,
)
from .fingerprints import compute_candidate_hash, compute_evidence_pack_hash
from .models import (
    AxialGeometryBindingView,
    AxialGeometryContractMatrix,
    AxialGeometryContractRow,
    AxialGeometryEvidencePack,
    PlanClosedLoopPolicy,
    PlanEvidenceItem,
    PlanGateId,
    PlanLoopMode,
    PlanReviewAction,
)


def axial_geometry_gate_applicable(state: Any) -> bool:
    """Gate applies when the task plan requires axial patches or Facts declares axial geometry."""
    if state.canonical_task_plan is not None:
        ordered = set(state.canonical_task_plan.ordered_patch_types)
        if ordered & {"base_path_axial_profiles", "axial_layers", "axial_overlays"}:
            return True
    else:
        for task in getattr(state, "component_tasks", []) or []:
            if getattr(task, "patch_type", "") in {"base_path_axial_profiles", "axial_layers", "axial_overlays"}:
                return True
        for ptype in ("base_path_axial_profiles", "axial_layers", "axial_overlays"):
            if _valid(state, ptype) is not None:
                return True
    facts_env = _valid(state, "facts")
    if facts_env is not None:
        facts = facts_env.content
        fc = getattr(facts, "planning_feature_contract", None) or facts
        if getattr(fc, "has_axial_geometry", False) or getattr(fc, "axial_domain_cm", None):
            return True
    return False


def axial_geometry_gate_ready(state: Any) -> bool:
    """Controlled/advisory review requires valid axial patches and upstream acceptance."""
    if not axial_geometry_gate_applicable(state):
        return False
    for patch_type in ("facts",):
        if _valid(state, patch_type) is None:
            return False
    for patch_type in ("axial_layers",):
        if _valid(state, patch_type) is None:
            return False
    return True


def axial_geometry_gate_input_hash(state: Any, *, policy: PlanClosedLoopPolicy | None = None) -> str:
    """Bind the gate input to every input that should invalidate the accepted hash."""
    view = build_axial_geometry_binding_view(state=state)
    payload: dict[str, Any] = {
        "facts_patch_hash": view.facts_patch_hash,
        "materials_patch_hash": view.materials_patch_hash,
        "universes_patch_hash": view.universes_patch_hash,
        "base_path_profiles_hash": view.base_path_profiles_hash,
        "axial_layers_hash": view.axial_layers_hash,
        "axial_overlays_hash": view.axial_overlays_hash,
        "material_universe_accepted_hash": view.material_universe_accepted_hash,
        "placement_accepted_hash": view.placement_accepted_hash,
        "feature_contract_hash": view.feature_contract_hash,
        "canonical_task_plan_hash": view.canonical_task_plan_hash,
        "axial_domain_cm": view.axial_domain_cm,
        "active_fuel_region_cm": view.active_fuel_region_cm,
        "confirmed_records": [str(item) for item in (state.plan_confirmed_plan_fact_records.keys() if hasattr(state, "plan_confirmed_plan_fact_records") else [])],
    }
    if policy is not None:
        payload["axial_geometry_review_mode"] = policy.axial_geometry_review_mode
        payload["structural_density_policy"] = str(getattr(state, "metadata", {}).get("structural_density_policy", "source_only"))
        payload["review_schema_version"] = "1"
    return compute_evidence_pack_hash(payload)


def build_axial_geometry_contract_matrix(view: AxialGeometryBindingView, issues: list[dict[str, Any]] | None = None) -> AxialGeometryContractMatrix:
    """Construct the nine-kind contract matrix from the binding view."""
    by_key: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for issue in issues or []:
        kind = str(issue.get("row_kind", "source_domain_coverage"))
        key = str(issue.get("row_key", ""))
        by_key.setdefault((kind, key), []).append(issue)
    rows: list[AxialGeometryContractRow] = []

    def _codes(kind: str, key: str) -> list[str]:
        return sorted({str(item.get("code")) for item in by_key.get((kind, key), [])})

    # 1. Source domain coverage rows.
    domain = view.axial_domain_cm
    for contract in view.source_axial_contracts:
        rid = contract.requirement_id
        actual_intervals: list[tuple[float, float]] = []
        if domain:
            actual_intervals.append(domain)
        gaps: list[tuple[float, float]] = []
        if domain and contract.axial_domain_cm:
            if domain[0] > contract.axial_domain_cm[0] + 1e-6:
                gaps.append((contract.axial_domain_cm[0], domain[0]))
            if domain[1] < contract.axial_domain_cm[1] - 1e-6:
                gaps.append((domain[1], contract.axial_domain_cm[1]))
        coverage = "pass" if domain and not gaps else ("fail" if not domain else "ambiguous")
        rows.append(AxialGeometryContractRow(
            row_id=f"sdc:{rid}",
            row_kind="source_domain_coverage",
            requirement_id=rid,
            expected_interval=contract.axial_domain_cm,
            actual_intervals=actual_intervals,
            gaps=gaps,
            boundary_status="covered" if not gaps else "incomplete",
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("source_domain_coverage", rid),
        ))
    # 2. Active fuel coverage rows.
    if view.active_fuel_region_cm:
        af = view.active_fuel_region_cm
        fuel_layers = [l for l in view.axial_layer_records if l.role in {"active_fuel", "fuel"}]
        actual_fuel = [(l.z_min_cm, l.z_max_cm) for l in fuel_layers if l.z_min_cm is not None and l.z_max_cm is not None]
        af_gaps = _compute_gaps(af, actual_fuel)
        coverage = "pass" if not af_gaps else "fail"
        rows.append(AxialGeometryContractRow(
            row_id="afc:active_fuel",
            row_kind="active_fuel_coverage",
            expected_interval=af,
            actual_intervals=actual_fuel,
            gaps=af_gaps,
            boundary_status="covered" if not af_gaps else "incomplete",
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("active_fuel_coverage", "active_fuel"),
        ))
    # 3. Layer fill binding rows.
    for layer in view.axial_layer_records:
        coverage = "pass"
        if layer.fill_type in {"material", "universe", "lattice"} and not layer.fill_id:
            coverage = "fail"
        elif layer.fill_type == "unknown":
            coverage = "ambiguous"
        rows.append(AxialGeometryContractRow(
            row_id=f"lfb:{layer.layer_id}",
            row_kind="layer_fill_binding",
            layer_id=layer.layer_id,
            material_id=layer.resolved_material_id or "",
            universe_id=layer.resolved_universe_id or "",
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("layer_fill_binding", layer.layer_id),
        ))
    # 4. Loading attachment rows.
    for loading in view.lattice_loading_records:
        coverage = "pass" if loading.attachment_status == "attached" else "fail"
        rows.append(AxialGeometryContractRow(
            row_id=f"lat:{loading.loading_id}",
            row_kind="loading_attachment",
            loading_id=loading.loading_id,
            lattice_id=loading.base_lattice_id or "",
            actual_count=len(loading.attached_layer_ids),
            boundary_status=loading.attachment_status,
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("loading_attachment", loading.loading_id),
        ))
    # 5. Overlay binding rows.
    for overlay in view.axial_overlay_records:
        coverage = "pass"
        if not overlay.target_lattice_id:
            coverage = "fail"
        elif not overlay.material_id and overlay.geometry_mode != "skeleton":
            coverage = "fail"
        elif overlay.density_status == "fail":
            coverage = "fail"
        rows.append(AxialGeometryContractRow(
            row_id=f"ovb:{overlay.overlay_id}",
            row_kind="overlay_binding",
            overlay_id=overlay.overlay_id,
            material_id=overlay.material_id or "",
            lattice_id=overlay.target_lattice_id or "",
            expected_interval=(overlay.z_min_cm, overlay.z_max_cm) if overlay.z_min_cm is not None and overlay.z_max_cm is not None else None,
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("overlay_binding", overlay.overlay_id),
        ))
    # 6. Base-path profile coverage rows.
    for profile in view.base_path_profile_records:
        coverage = "pass" if profile.segments else "ambiguous"
        rows.append(AxialGeometryContractRow(
            row_id=f"bpc:{profile.profile_id}",
            row_kind="base_path_profile_coverage",
            profile_id=profile.profile_id,
            requirement_id=profile.source_requirement_id,
            actual_count=len(profile.segments),
            boundary_status=profile.coverage_status,
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("base_path_profile_coverage", profile.profile_id),
        ))
    # 7. Localized insert axial occupancy rows.
    for insert in view.localized_insert_axial_records:
        rows.append(AxialGeometryContractRow(
            row_id=f"lia:{insert.requirement_id}",
            row_kind="localized_insert_axial_occupancy",
            insert_requirement_id=insert.requirement_id,
            profile_id=insert.profile_id,
            expected_interval=insert.translated_absolute_extent,
            boundary_status=insert.clipping,
            coverage_status=insert.coverage_status,  # type: ignore[arg-type]
            issue_codes=_codes("localized_insert_axial_occupancy", insert.requirement_id),
        ))
    # 8. Through-path preservation rows.
    for tp in view.through_path_records:
        coverage = "pass" if tp.preserved else "fail"
        rows.append(AxialGeometryContractRow(
            row_id=f"tpp:{tp.through_path_id}",
            row_kind="through_path_preservation",
            through_path_id=tp.through_path_id,
            coverage_status=coverage,  # type: ignore[arg-type]
            issue_codes=_codes("through_path_preservation", tp.through_path_id) + list(tp.issue_codes),
        ))
    # 9. Spacer-grid structural count row.
    grid_overlays = [o for o in view.axial_overlay_records if o.overlay_kind == "spacer_grid"]
    expected_count = None
    for c in view.source_axial_contracts:
        if c.metadata.get("expected_spacer_grid_count"):
            expected_count = int(c.metadata["expected_spacer_grid_count"])
            break
    coverage_sg = "pass"
    if expected_count is not None and len(grid_overlays) != expected_count:
        coverage_sg = "fail"
    elif not grid_overlays:
        coverage_sg = "ambiguous"
    rows.append(AxialGeometryContractRow(
        row_id="sgc:spacer_grids",
        row_kind="spacer_grid_structural_count",
        expected_count=expected_count,
        actual_count=len(grid_overlays),
        coverage_status=coverage_sg,  # type: ignore[arg-type]
        issue_codes=_codes("spacer_grid_structural_count", "spacer_grids"),
    ))
    matrix = AxialGeometryContractMatrix(rows=rows)
    matrix.input_hash = compute_evidence_pack_hash(matrix)
    return matrix


def _compute_gaps(domain: tuple[float, float], intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Compute uncovered gaps within ``domain`` given covering ``intervals``."""
    if not intervals:
        return [domain]
    merged = sorted(intervals)
    gaps: list[tuple[float, float]] = []
    cursor = domain[0]
    for z0, z1 in merged:
        if z0 > cursor + 1e-6:
            gaps.append((cursor, z0))
        cursor = max(cursor, z1)
    if cursor < domain[1] - 1e-6:
        gaps.append((cursor, domain[1]))
    return gaps


def build_axial_geometry_evidence_pack(
    *, state: Any, policy: PlanClosedLoopPolicy, deterministic_issues: list[dict[str, Any]] | None = None,
) -> AxialGeometryEvidencePack:
    view = build_axial_geometry_binding_view(state=state)
    matrix = build_axial_geometry_contract_matrix(view, deterministic_issues)
    patch_hashes: dict[str, str] = {}
    for ptype in ("facts", "base_path_axial_profiles", "axial_layers", "axial_overlays"):
        env = _valid(state, ptype)
        if env is not None:
            patch_hashes[ptype] = compute_candidate_hash(target_patch_type=ptype, candidate_patch=env.content)
    items: list[PlanEvidenceItem] = []
    index = 1

    def _add(kind: str, prefix: str, patch_type: str | None, path: str | None, label: str, value: Any, metadata: dict[str, Any] | None = None) -> None:
        nonlocal index
        canonical_hash = compute_evidence_pack_hash({"kind": kind, "patch_type": patch_type, "path": path, "value": value})
        items.append(PlanEvidenceItem(ref_id=f"{prefix}{index:03d}", evidence_kind=kind, patch_type=patch_type, json_path=path, label=label, value=value, canonical_hash=canonical_hash, metadata=metadata or {}))
        index += 1

    for contract in view.source_axial_contracts:
        _add("accepted_fact_contract", "F", "facts", "/source_axial_contracts", f"source axial contract {contract.requirement_id}", contract.model_dump(mode="json"))
    if view.material_universe_accepted_hash:
        _add("accepted_fact_contract", "M", None, None, "Material-Universe accepted hash", {"accepted_input_hash": view.material_universe_accepted_hash})
    if view.placement_accepted_hash:
        _add("accepted_fact_contract", "P", None, None, "Placement accepted hash", {"accepted_input_hash": view.placement_accepted_hash})
    for profile in view.base_path_profile_records:
        _add("patch_fragment", "B", "base_path_axial_profiles", f"/profiles/{profile.profile_id}", f"profile {profile.profile_id}", profile.model_dump(mode="json"))
    for layer in view.axial_layer_records:
        _add("patch_fragment", "A", "axial_layers", f"/layers/{layer.layer_id}", f"layer {layer.layer_id}", layer.model_dump(mode="json"))
    for loading in view.lattice_loading_records:
        _add("patch_fragment", "L", "axial_layers", f"/lattice_loadings/{loading.loading_id}", f"loading {loading.loading_id}", loading.model_dump(mode="json"))
    for overlay in view.axial_overlay_records:
        _add("patch_fragment", "O", "axial_overlays", f"/overlays/{overlay.overlay_id}", f"overlay {overlay.overlay_id}", overlay.model_dump(mode="json"))
    for insert in view.localized_insert_axial_records:
        _add("patch_fragment", "I", "facts", f"/localized_insert_requirements/{insert.requirement_id}", f"insert {insert.requirement_id}", insert.model_dump(mode="json"))
    for tp in view.through_path_records:
        _add("patch_fragment", "T", None, f"/through_path/{tp.through_path_id}", f"through-path {tp.through_path_id}", tp.model_dump(mode="json"))
    for seg in view.derived_segments:
        _add("patch_fragment", "G", None, f"/derived_segments/{seg.segment_id}", f"segment {seg.segment_id}", seg.model_dump(mode="json"))
    for issue in deterministic_issues or []:
        _add("deterministic_issue", "D", None, None, f"deterministic issue {issue.get('code', '')}", issue)
    pack = AxialGeometryEvidencePack(
        binding_view=view,
        contract_matrix=matrix,
        deterministic_issues=list(deterministic_issues or []),
        relevant_patch_hashes=patch_hashes,
        accepted_facts_hash=patch_hashes.get("facts", ""),
        accepted_material_universe_hash=view.material_universe_accepted_hash,
        accepted_placement_hash=view.placement_accepted_hash,
        evidence_items=items,
        confirmed_records=[item.model_dump(mode="json") for item in getattr(state, "plan_confirmed_plan_fact_records", {}).values()],
        allowed_actions=list(_allowed_review_actions(policy)),
    )
    pack.input_hash = axial_geometry_gate_input_hash(state, policy=policy)
    pack.evidence_pack_id = pack.input_hash
    return pack


def _allowed_review_actions(policy: PlanClosedLoopPolicy) -> list[PlanReviewAction]:
    if policy.mode is PlanLoopMode.OFF:
        return []
    return [PlanReviewAction.APPROVE, PlanReviewAction.REVISE_CURRENT_PATCH, PlanReviewAction.RETRY_DEPENDENCY, PlanReviewAction.ASK_HUMAN, PlanReviewAction.FAIL_CLOSED]


__all__ = [
    "axial_geometry_gate_applicable",
    "axial_geometry_gate_ready",
    "axial_geometry_gate_input_hash",
    "build_axial_geometry_contract_matrix",
    "build_axial_geometry_evidence_pack",
]
