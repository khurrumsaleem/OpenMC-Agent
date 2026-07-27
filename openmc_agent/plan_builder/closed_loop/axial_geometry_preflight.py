"""Deterministic Axial Geometry preflight.

Produces the canonical set of cross-patch issues that the Critic is *not*
allowed to recompute.  Reuses single-patch validators, the axial overlay
geometry tools, and the assembly3D structural guard wherever possible.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.validators import PatchValidationContext, validate_patch
from openmc_agent.schemas import AgentBaseModel

from .axial_geometry_binding import (
    _valid,
    build_axial_geometry_binding_view,
)
from .axial_geometry_evidence import (
    axial_geometry_gate_input_hash,
    build_axial_geometry_contract_matrix,
)
from .models import AxialGeometryBindingView, PlanClosedLoopPolicy


class AxialGeometryPreflightResult(AgentBaseModel):
    ok: bool = False
    binding_view: AxialGeometryBindingView | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    input_hash: str = ""

    @property
    def blocking_issues(self) -> list[dict[str, Any]]:
        return [item for item in self.issues if item.get("severity") == "error"]


def _issue(code: str, message: str, *, severity: str = "error", row_kind: str = "source_domain_coverage", row_key: str = "", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "severity": severity, "blocking": severity == "error", "message": message, "row_kind": row_kind, "row_key": row_key}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


_Z_TOL = 1e-6


def _collect_local_validation_issues(state: Any, patch_types: tuple[str, ...]) -> list[dict[str, Any]]:
    """Run single-patch validators and collect their issues."""
    issues: list[dict[str, Any]] = []
    known_material_ids: set[str] = set()
    known_universe_ids: set[str] = set()
    known_lattice_ids = _collect_known_lattice_ids(state)
    for ptype in ("materials", "universes"):
        env = _valid(state, ptype)
        if env is None:
            continue
        content = env.content
        if ptype == "materials":
            for m in getattr(content, "materials", []):
                known_material_ids.add(m.material_id)
        elif ptype == "universes":
            for u in getattr(content, "universes", []):
                known_universe_ids.add(u.universe_id)
    for ptype in patch_types:
        env = _valid(state, ptype)
        if env is None:
            continue
        ctx = PatchValidationContext(
            known_material_ids=known_material_ids,
            known_universe_ids=known_universe_ids,
            known_lattice_ids=sorted(known_lattice_ids),
        )
        result = validate_patch(env.content, context=ctx)
        for item in result.issues:
            code = str(item.code)
            mapped = _map_local_code(code, ptype)
            if mapped is None:
                continue
            issues.append(_issue(
                mapped, str(item.message),
                severity=str(item.severity),
                row_kind=_row_kind_for_code(mapped),
                row_key=str(result.patch_id or ""),
                owner_patch_type=ptype,
            ))
    return issues


_LOCAL_CODE_MAP: dict[str, str] = {
    "patch.axial_layers.duplicate_id": "axial.layer_duplicate",
    "patch.axial_layers.invalid_range": "axial.layer_interval_invalid",
    "patch.axial_layers.fill_missing": "axial.fill_missing",
    "patch.axial_layers.fill_unknown": "axial.fill_unknown",
    "patch.axial_layers.fill_ref_missing": "axial.material_reference_missing",
    "patch.axial_layers.overlap": "axial.layer_overlap",
    "patch.axial_layers.active_fuel_missing": "axial.active_fuel_region_not_covered",
    "patch.axial_layers.loading_unattached": "axial.loading_unattached",
    "patch.axial_layers.default_unit_slab": "axial.layer_default_placeholder",
    "patch.axial_layers.loading_transformation_replacement_universe_missing": "axial.universe_reference_missing",
    "patch.axial_layers.loading_transformation_source_universe_missing": "axial.universe_reference_missing",
    "patch.axial_layers.loading_transformation_coordinates_not_owned": "axial.loading_reference_missing",
    "patch.axial_overlays.duplicate_overlay_id": "axial.overlay_duplicate",
    "patch.axial_overlays.invalid_range": "axial.overlay_interval_invalid",
    "patch.axial_overlays.material_missing": "axial.overlay_material_missing",
    "patch.axial_overlays.target_missing": "axial.overlay_target_lattice_missing",
    "patch.axial_overlays.mode_semantic_contradiction": "axial.overlay_through_path_not_preserved",
    "patch.axial_overlays.total_mass_missing": "axial.overlay_density_required",
    "patch.axial_overlays.volume_fraction_missing": "axial.overlay_geometry_mode_unsupported",
}


def _map_local_code(code: str, patch_type: str) -> str | None:
    return _LOCAL_CODE_MAP.get(code)


def _row_kind_for_code(code: str) -> str:
    if "domain" in code or "active_fuel" in code:
        return "source_domain_coverage" if "domain" in code and "active" not in code else "active_fuel_coverage"
    if "layer" in code and "fill" in code:
        return "layer_fill_binding"
    if "loading" in code:
        return "loading_attachment"
    if "overlay" in code:
        return "overlay_binding"
    if "base_path" in code or "profile" in code:
        return "base_path_profile_coverage"
    if "localized_insert" in code or "insert" in code:
        return "localized_insert_axial_occupancy"
    if "through_path" in code or "grid_replaced" in code:
        return "through_path_preservation"
    return "source_domain_coverage"


def _collect_domain_issues(view: AxialGeometryBindingView) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    domain = view.axial_domain_cm
    if not domain:
        issues.append(_issue("axial.domain_missing", "no finite axial_domain_cm declared", row_kind="source_domain_coverage", row_key="domain", owner_patch_type="facts"))
        return issues
    z0, z1 = domain
    if z1 <= z0:
        issues.append(_issue("axial.domain_invalid", f"axial domain reversed: ({z0}, {z1})", row_kind="source_domain_coverage", row_key="domain", owner_patch_type="facts"))
    layers = [l for l in view.axial_layer_records if l.z_min_cm is not None and l.z_max_cm is not None]
    for layer in layers:
        if layer.z_min_cm >= layer.z_max_cm - _Z_TOL:
            issues.append(_issue("axial.layer_zero_thickness", f"layer {layer.layer_id} has zero or negative thickness", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="axial_layers"))
        if layer.z_min_cm < z0 - _Z_TOL or layer.z_max_cm > z1 + _Z_TOL:
            issues.append(_issue("axial.layer_interval_outside_domain", f"layer {layer.layer_id} interval ({layer.z_min_cm}, {layer.z_max_cm}) outside domain ({z0}, {z1})", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="axial_layers", severity="warning"))
    # Check for gaps and overlaps among base layers (non-overlay).
    sorted_layers = sorted(layers, key=lambda l: l.z_min_cm)
    for i in range(len(sorted_layers) - 1):
        cur = sorted_layers[i]
        nxt = sorted_layers[i + 1]
        gap = nxt.z_min_cm - cur.z_max_cm  # type: ignore[union-attr]
        if gap > _Z_TOL:
            issues.append(_issue("axial.layer_gap", f"gap of {gap:.4f} cm between layers {cur.layer_id} and {nxt.layer_id}", row_kind="source_domain_coverage", row_key=f"{cur.layer_id}:{nxt.layer_id}", owner_patch_type="axial_layers", severity="warning"))
        elif gap < -_Z_TOL:
            issues.append(_issue("axial.layer_overlap", f"overlap of {-gap:.4f} cm between layers {cur.layer_id} and {nxt.layer_id}", row_kind="source_domain_coverage", row_key=f"{cur.layer_id}:{nxt.layer_id}", owner_patch_type="axial_layers"))
    # Active fuel coverage.
    if view.active_fuel_region_cm:
        af = view.active_fuel_region_cm
        fuel_layers = [l for l in layers if l.role in {"active_fuel", "fuel"}]
        fuel_intervals = [(l.z_min_cm, l.z_max_cm) for l in fuel_layers]  # type: ignore[misc]
        covered = _interval_union(fuel_intervals)
        for f0, f1 in fuel_intervals:
            if f0 < af[0] - _Z_TOL or f1 > af[1] + _Z_TOL:
                issues.append(_issue("axial.active_fuel_layer_outside_contract", f"fuel layer interval ({f0}, {f1}) exceeds active fuel region {af}", row_kind="active_fuel_coverage", row_key="active_fuel", owner_patch_type="axial_layers", severity="warning"))
        af_gaps = _compute_coverage_gaps(af, covered)
        for g0, g1 in af_gaps:
            issues.append(_issue("axial.active_fuel_region_not_covered", f"active fuel region gap ({g0:.4f}, {g1:.4f})", row_kind="active_fuel_coverage", row_key="active_fuel", owner_patch_type="axial_layers"))
    return issues


def _collect_fill_reference_issues(view: AxialGeometryBindingView, state: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    material_ids: set[str] = set()
    universe_ids: set[str] = set()
    lattice_ids = _collect_known_lattice_ids(state)
    for ptype, attr in (("materials", "materials"), ("universes", "universes")):
        env = _valid(state, ptype)
        if env is None:
            continue
        content = env.content
        if isinstance(content, dict):
            content = parse_patch_content(ptype, content)
        for item in getattr(content, attr, []):
            if ptype == "materials":
                material_ids.add(item.material_id)
            else:
                universe_ids.add(item.universe_id)
    layers_env = _valid(state, "axial_layers")
    layers_content = layers_env.content if layers_env is not None else None
    if isinstance(layers_content, dict):
        layers_content = parse_patch_content("axial_layers", layers_content)
    if layers_content is not None:
        for loading in getattr(layers_content, "lattice_loadings", []):
            if loading.base_lattice_id:
                lattice_ids.add(loading.base_lattice_id)
            if loading.derived_lattice_id:
                lattice_ids.add(loading.derived_lattice_id)
    for layer in view.axial_layer_records:
        if layer.fill_type in {"material", "universe", "lattice"} and not layer.fill_id:
            issues.append(_issue("axial.fill_missing", f"layer {layer.layer_id} fill_type={layer.fill_type} but no fill_id", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="axial_layers"))
            continue
        if layer.fill_type == "unknown":
            issues.append(_issue("axial.fill_unknown", f"layer {layer.layer_id} fill_type=unknown", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="axial_layers", severity="warning"))
        if layer.fill_type == "material" and layer.fill_id and layer.fill_id not in material_ids:
            issues.append(_issue("axial.material_reference_missing", f"layer {layer.layer_id} references material {layer.fill_id} not in materials patch", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="materials"))
        if layer.fill_type == "universe" and layer.fill_id and layer.fill_id not in universe_ids:
            issues.append(_issue("axial.universe_reference_missing", f"layer {layer.layer_id} references universe {layer.fill_id} not in universes patch", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="universes"))
        if layer.fill_type == "lattice" and layer.fill_id and layer.fill_id not in lattice_ids:
            issues.append(_issue("axial.lattice_reference_missing", f"layer {layer.layer_id} references lattice {layer.fill_id} not defined", row_kind="layer_fill_binding", row_key=layer.layer_id, owner_patch_type="axial_layers"))
    # Loading references.
    declared_loadings: set[str] = set()
    if layers_content is not None:
        declared_loadings = {l.loading_id for l in getattr(layers_content, "lattice_loadings", [])}
    for loading in view.lattice_loading_records:
        if loading.loading_id not in declared_loadings and not loading.attached_layer_ids:
            issues.append(_issue("axial.loading_reference_missing", f"loading {loading.loading_id} is not declared and not attached", row_kind="loading_attachment", row_key=loading.loading_id, owner_patch_type="axial_layers", severity="warning"))
        if loading.attachment_status == "unattached":
            issues.append(_issue("axial.loading_unattached", f"loading {loading.loading_id} declared but not attached to any layer", row_kind="loading_attachment", row_key=loading.loading_id, owner_patch_type="axial_layers"))
        if not loading.base_lattice_id:
            issues.append(_issue("axial.loading_base_lattice_missing", f"loading {loading.loading_id} has no base_lattice_id", row_kind="loading_attachment", row_key=loading.loading_id, owner_patch_type="axial_layers"))
    return issues


def _collect_overlay_issues(view: AxialGeometryBindingView, state: Any) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    domain = view.axial_domain_cm
    lattice_ids = _collect_known_lattice_ids(state)
    layers_env = _valid(state, "axial_layers")
    layers_content = layers_env.content if layers_env is not None else None
    if isinstance(layers_content, dict):
        layers_content = parse_patch_content("axial_layers", layers_content)
    if layers_content is not None:
        for loading in getattr(layers_content, "lattice_loadings", []):
            if loading.base_lattice_id:
                lattice_ids.add(loading.base_lattice_id)
            if loading.derived_lattice_id:
                lattice_ids.add(loading.derived_lattice_id)
    material_ids: set[str] = set()
    mat_env = _valid(state, "materials")
    mat_content = mat_env.content if mat_env is not None else None
    if isinstance(mat_content, dict):
        mat_content = parse_patch_content("materials", mat_content)
    if mat_content is not None:
        material_ids = {m.material_id for m in getattr(mat_content, "materials", [])}
    for overlay in view.axial_overlay_records:
        if overlay.z_min_cm is not None and overlay.z_max_cm is not None:
            if overlay.z_min_cm >= overlay.z_max_cm - _Z_TOL:
                issues.append(_issue("axial.overlay_interval_invalid", f"overlay {overlay.overlay_id} interval invalid", row_kind="overlay_binding", row_key=overlay.overlay_id, owner_patch_type="axial_overlays"))
            if domain and (overlay.z_min_cm < domain[0] - _Z_TOL or overlay.z_max_cm > domain[1] + _Z_TOL):
                issues.append(_issue("axial.overlay_outside_domain", f"overlay {overlay.overlay_id} outside domain", row_kind="overlay_binding", row_key=overlay.overlay_id, owner_patch_type="axial_overlays", severity="warning"))
        if overlay.target_lattice_id and overlay.target_lattice_id not in lattice_ids:
            issues.append(_issue("axial.overlay_target_lattice_missing", f"overlay {overlay.overlay_id} targets lattice {overlay.target_lattice_id} not defined", row_kind="overlay_binding", row_key=overlay.overlay_id, owner_patch_type="axial_layers"))
        if overlay.material_id and overlay.material_id not in material_ids:
            issues.append(_issue("axial.overlay_material_missing", f"overlay {overlay.overlay_id} references material {overlay.material_id} not in materials patch", row_kind="overlay_binding", row_key=overlay.overlay_id, owner_patch_type="materials"))
        if overlay.density_status == "fail":
            issues.append(_issue("axial.overlay_density_required", f"overlay {overlay.overlay_id} requires mass-conserving basis but none provided", row_kind="overlay_binding", row_key=overlay.overlay_id, owner_patch_type="axial_overlays"))
    # Overlay z-overlaps.
    overlays_with_z = [(o.overlay_id, o.z_min_cm, o.z_max_cm) for o in view.axial_overlay_records if o.z_min_cm is not None and o.z_max_cm is not None]
    for i in range(len(overlays_with_z)):
        for j in range(i + 1, len(overlays_with_z)):
            a_id, a0, a1 = overlays_with_z[i]
            b_id, b0, b1 = overlays_with_z[j]
            if a0 < b1 - _Z_TOL and b0 < a1 - _Z_TOL:
                issues.append(_issue("axial.overlay_overlap_conflict", f"overlays {a_id} and {b_id} overlap in z", row_kind="overlay_binding", row_key=f"{a_id}:{b_id}", owner_patch_type="axial_overlays", severity="warning"))
    # Source grid count.
    expected_grids = None
    for c in view.source_axial_contracts:
        if c.metadata.get("expected_spacer_grid_count"):
            expected_grids = int(c.metadata["expected_spacer_grid_count"])
            break
    if expected_grids is not None:
        actual_grids = len([o for o in view.axial_overlay_records if o.overlay_kind == "spacer_grid"])
        if actual_grids != expected_grids:
            issues.append(_issue("axial.overlay_source_count_mismatch", f"expected {expected_grids} spacer grids, found {actual_grids}", row_kind="spacer_grid_structural_count", row_key="spacer_grids", owner_patch_type="axial_overlays", severity="warning", expected=expected_grids, actual=actual_grids))
    return issues


def _collect_known_lattice_ids(state: Any) -> set[str]:
    """Collect concrete and abstract lattice IDs available to axial patches.

    In full-core planning, Placement may define an assembly family through
    ``assembly_catalog``/``core_layout`` rather than a single concrete
    ``assembly_lattice`` object.  Axial patches are allowed to bind to that
    assembly-family alias; the assembled/rendering path later expands it to
    concrete per-assembly-type lattices.
    """
    lattice_ids: set[str] = set()

    layers_env = _valid(state, "axial_layers")
    layers_content = layers_env.content if layers_env is not None else None
    if isinstance(layers_content, dict):
        layers_content = parse_patch_content("axial_layers", layers_content)
    if layers_content is not None:
        for loading in getattr(layers_content, "lattice_loadings", []):
            if loading.base_lattice_id:
                lattice_ids.add(loading.base_lattice_id)
            if loading.derived_lattice_id:
                lattice_ids.add(loading.derived_lattice_id)

    assembly_env = _valid(state, "assembly_catalog")
    assembly_content = assembly_env.content if assembly_env is not None else None
    if isinstance(assembly_content, dict):
        assembly_content = parse_patch_content("assembly_catalog", assembly_content)
    assembly_type_ids: list[str] = []
    if assembly_content is not None:
        for assembly_type in getattr(assembly_content, "assembly_types", []) or []:
            assembly_type_id = getattr(assembly_type, "assembly_type_id", None)
            if assembly_type_id:
                assembly_type_ids.append(str(assembly_type_id))

    core_env = _valid(state, "core_layout")
    core_content = core_env.content if core_env is not None else None
    if isinstance(core_content, dict):
        core_content = parse_patch_content("core_layout", core_content)
    if core_content is not None:
        core_lattice_id = getattr(core_content, "core_lattice_id", None)
        if core_lattice_id:
            lattice_ids.add(str(core_lattice_id))
        pattern = getattr(core_content, "assembly_pattern", None) or []
        for row in pattern:
            if isinstance(row, (list, tuple)):
                for assembly_type_id in row:
                    if assembly_type_id:
                        assembly_type_ids.append(str(assembly_type_id))

    if assembly_type_ids:
        lattice_ids.add("assembly_lattice")
        for assembly_type_id in assembly_type_ids:
            lattice_ids.add(f"assembly_lattice__{assembly_type_id}")

    return lattice_ids


def _collect_through_path_issues(view: AxialGeometryBindingView) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for tp in view.through_path_records:
        for code in tp.issue_codes:
            issues.append(_issue(code, f"through-path {tp.through_path_id} ({tp.path_kind}) interrupted", row_kind="through_path_preservation", row_key=tp.through_path_id, owner_patch_type="axial_overlays" if tp.overlay_band_ids else "axial_layers"))
    return issues


def _collect_localized_insert_issues(view: AxialGeometryBindingView) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    domain = view.axial_domain_cm
    for insert in view.localized_insert_axial_records:
        if not insert.profile_id:
            issues.append(_issue("axial.localized_insert_profile_missing", f"insert {insert.requirement_id} has no profile", row_kind="localized_insert_axial_occupancy", row_key=insert.requirement_id, owner_patch_type="localized_insert_profiles"))
        if insert.anchor_z_cm is None:
            issues.append(_issue("axial.localized_insert_anchor_missing", f"insert {insert.requirement_id} has no anchor", row_kind="localized_insert_axial_occupancy", row_key=insert.requirement_id, owner_patch_type="localized_insert_profiles", severity="warning"))
        if not insert.host_layer_ids:
            issues.append(_issue("axial.localized_insert_no_host_layer_overlap", f"insert {insert.requirement_id} does not overlap any host layer", row_kind="localized_insert_axial_occupancy", row_key=insert.requirement_id, owner_patch_type="axial_layers"))
        if domain and insert.translated_absolute_extent:
            ta = insert.translated_absolute_extent
            if ta[0] < domain[0] - _Z_TOL or ta[1] > domain[1] + _Z_TOL:
                issues.append(_issue("axial.localized_insert_extent_outside_domain", f"insert {insert.requirement_id} translated extent outside domain", row_kind="localized_insert_axial_occupancy", row_key=insert.requirement_id, owner_patch_type="localized_insert_profiles", severity="warning"))
    return issues


def _interval_union(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    merged = sorted(intervals)
    result = [merged[0]]
    for z0, z1 in merged[1:]:
        if z0 <= result[-1][1] + _Z_TOL:
            result[-1] = (result[-1][0], max(result[-1][1], z1))
        else:
            result.append((z0, z1))
    return result


def _compute_coverage_gaps(domain: tuple[float, float], covered: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not covered:
        return [domain]
    gaps: list[tuple[float, float]] = []
    cursor = domain[0]
    for z0, z1 in covered:
        if z0 > cursor + _Z_TOL:
            gaps.append((cursor, z0))
        cursor = max(cursor, z1)
    if cursor < domain[1] - _Z_TOL:
        gaps.append((cursor, domain[1]))
    return gaps


def run_axial_geometry_preflight(*, state: Any, policy: PlanClosedLoopPolicy) -> AxialGeometryPreflightResult:
    """Run deterministic preflight, reusing single-patch validators and geometry tools."""
    view = build_axial_geometry_binding_view(state=state)
    issues: list[dict[str, Any]] = []
    issues.extend(_collect_local_validation_issues(state, ("base_path_axial_profiles", "axial_layers", "axial_overlays")))
    issues.extend(_collect_domain_issues(view))
    issues.extend(_collect_fill_reference_issues(view, state))
    issues.extend(_collect_overlay_issues(view, state))
    issues.extend(_collect_through_path_issues(view))
    issues.extend(_collect_localized_insert_issues(view))
    # Deduplicate by (code, row_key) keeping highest severity.
    seen: dict[tuple[str, str], dict[str, Any]] = {}
    for issue in issues:
        key = (issue["code"], issue.get("row_key", ""))
        if key not in seen or (issue.get("severity") == "error" and seen[key].get("severity") != "error"):
            seen[key] = issue
    issues = list(seen.values())
    matrix = build_axial_geometry_contract_matrix(view, issues)
    input_hash = axial_geometry_gate_input_hash(state, policy=policy)
    blocking = [i for i in issues if i.get("severity") == "error"]
    return AxialGeometryPreflightResult(
        ok=len(blocking) == 0,
        binding_view=view,
        issues=issues,
        summary={
            "total": len(issues),
            "blocking": len(blocking),
            "warnings": len([i for i in issues if i.get("severity") == "warning"]),
            "layer_count": len(view.axial_layer_records),
            "overlay_count": len(view.axial_overlay_records),
            "loading_count": len(view.lattice_loading_records),
            "profile_count": len(view.base_path_profile_records),
            "insert_count": len(view.localized_insert_axial_records),
            "derived_segment_count": len(view.derived_segments),
            "matrix_rows": len(matrix.rows),
        },
        input_hash=input_hash,
    )


__all__ = ["AxialGeometryPreflightResult", "run_axial_geometry_preflight"]
