"""Deterministic Material-Universe preflight.

Produces the canonical set of cross-patch issues that the Critic is *not*
allowed to recompute.  Reuses single-patch validators, the material species
resolver, and radial profile validation wherever possible.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from openmc_agent.material_species import classify_species_name
from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.validators import validate_patch
from openmc_agent.schemas import AgentBaseModel

from .fingerprints import compute_evidence_pack_hash
from .material_universe_binding import _valid, build_material_universe_binding_view
from .material_universe_evidence import build_material_universe_contract_matrix, material_universe_gate_input_hash
from .models import MaterialUniverseBindingView, PlanClosedLoopPolicy


class MaterialUniversePreflightResult(AgentBaseModel):
    ok: bool = False
    binding_view: MaterialUniverseBindingView | None = None
    issues: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    input_hash: str = ""

    @property
    def blocking_issues(self) -> list[dict[str, Any]]:
        return [item for item in self.issues if item.get("severity") == "error"]


def _issue(code: str, message: str, *, severity: str = "error", row_kind: str = "source_material_coverage", row_key: str = "", **extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"code": code, "severity": severity, "blocking": severity == "error", "message": message, "row_kind": row_kind, "row_key": row_key}
    payload.update({k: v for k, v in extra.items() if v is not None})
    return payload


def _universe_metadata(universe: Any) -> dict[str, Any]:
    """Extract metadata dict from a universe (handles both model and dict)."""
    if isinstance(universe, dict):
        return universe.get("metadata") or {}
    meta = getattr(universe, "metadata", None)
    if isinstance(meta, dict):
        return meta
    return {}


def _universe_fuel_variant_id(universe: Any) -> str | None:
    direct = getattr(universe, "fuel_variant_id", None)
    if direct:
        return str(direct)
    meta = _universe_metadata(universe)
    value = meta.get("fuel_variant_id")
    return str(value) if value else None


def _metadata_covers_localized_insert(meta: dict[str, Any], req_id: str) -> bool:
    if not req_id:
        return False
    if meta.get("localized_insert_requirement_id") == req_id:
        return True
    values = meta.get("localized_insert_requirement_ids") or []
    if isinstance(values, str):
        values = [values]
    return req_id in {str(item) for item in values}


def _collect_materials_issues(materials_patch: Any, view: MaterialUniverseBindingView, species_report: dict[str, Any]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for material in materials_patch.materials:
        mid = material.material_id
        if mid in seen_ids:
            issues.append(_issue("material_universe.material_duplicate", f"duplicate material_id {mid}", row_kind="source_material_coverage", row_key=mid, material_id=mid))
            continue
        seen_ids.add(mid)
        if material.density_g_cm3 is not None and material.density_g_cm3 <= 0:
                issues.append(_issue("material_universe.material_density_invalid", f"material {mid} density <= 0", row_kind="source_material_coverage", row_key=mid, material_id=mid, density=material.density_g_cm3))
        if material.density_g_cm3 is None and material.role in {"fuel", "cladding", "coolant", "absorber", "poison", "structural"}:
            issues.append(_issue("material_universe.material_density_missing", f"material {mid} has no density", row_kind="source_material_coverage", row_key=mid, material_id=mid))
        # Compound formula must not live in transport composition.
        if material.compound_components:
            for element, fraction in material.composition.items():
                # A transport-composition key must be a nuclide/element symbol
                # (e.g. B10, U235, O16, Si), not a compound formula (B2O3).
                # The earlier heuristic (digit + all-upper + len>2) misflagged
                # nuclides like B10/B11/O16; classify_species_name distinguishes
                # nuclides/elements from compounds.
                if classify_species_name(element) == "compound":
                    issues.append(_issue("material_universe.compound_in_transport_composition", f"material {mid} has compound formula {element} in transport composition", row_kind="source_material_coverage", row_key=mid, material_id=mid, compound=element))
                    break
            if not material.composition_basis or material.composition_basis == "unknown":
                issues.append(_issue("material_universe.compound_fraction_basis_missing", f"material {mid} has compound_components but no composition_basis", row_kind="source_material_coverage", row_key=mid, material_id=mid))
            for component in material.compound_components:
                isotope_policy = getattr(component, "isotope_policy", None)
                if isotope_policy in {None, "requires_confirmation"}:
                    issues.append(_issue("material_universe.compound_isotope_unresolved", f"material {mid} compound {component.formula} has unresolved isotope policy", row_kind="source_material_coverage", row_key=mid, material_id=mid, compound=component.formula))
        # Fissile isotope policy: fuel material with composition U235 but no explicit policy warning.
        if material.role == "fuel" and material.composition:
            u235 = material.composition.get("U235")
            if u235 is not None and u235 > 0 and not material.source_variant_id:
                issues.append(_issue("material_universe.fissile_isotope_policy_missing", f"fuel material {mid} has U235 but no source_variant_id", row_kind="source_material_coverage", row_key=mid, material_id=mid, severity="warning"))
        # Approximate composition must not be marked confirmed.
        if material.composition_status == "confirmed" and material.source_variant_id is None and material.role in {"structural", "cladding"}:
            # Commercial alloy with no source disclosure cannot be confirmed.
            if not material.composition:
                issues.append(_issue("material_universe.alloy_reduced_without_disclosure", f"structural material {mid} marked confirmed without composition disclosure", row_kind="source_material_coverage", row_key=mid, material_id=mid, severity="warning"))
        if material.composition_status == "placeholder":
            issues.append(_issue("material_universe.placeholder_material_unresolved", f"material {mid} is still a placeholder", row_kind="source_material_coverage", row_key=mid, material_id=mid))
        # Species resolver warnings become deterministic issues.
        resolver_entry = species_report.get("materials", {}).get(mid, {}) if isinstance(species_report, dict) else {}
        for warning in resolver_entry.get("warnings", []):
            if "error" in str(warning).lower():
                issues.append(_issue("material_universe.transport_species_invalid", f"material {mid} resolver: {warning}", row_kind="source_material_coverage", row_key=mid, material_id=mid))
    # Source-required materials missing.
    for contract in view.required_material_contracts:
        if contract.get("implicit"):
            continue
        role = contract.get("expected_role")
        variant = contract.get("expected_variant_id")
        if variant:
            # Fuel variant: check by source_variant_id.
            found = any(
                m.role == "fuel" and m.source_variant_id == variant
                for m in materials_patch.materials
            )
            if not found:
                issues.append(_issue("material_universe.required_fuel_variant_material_missing", f"fuel variant {variant} has no material", row_kind="source_material_coverage", row_key=contract["requirement_id"], requirement_id=contract["requirement_id"]))
        elif role:
            # "poison" and "absorber" are semantically equivalent for
            # material role matching — both describe neutron-absorbing
            # materials.  LLMs may use either term depending on training
            # data, and the physics is identical.
            equivalent_roles = {role}
            if role in {"poison", "absorber"}:
                equivalent_roles = {"poison", "absorber"}
            found = any(m.role in equivalent_roles for m in materials_patch.materials)
            if not found and role in {"absorber", "poison"}:
                issues.append(_issue("material_universe.required_material_missing", f"required material role {role} not found", row_kind="source_material_coverage", row_key=contract["requirement_id"], requirement_id=contract["requirement_id"], expected_role=role))
    # Duplicate fuel variant materials (two materials same source_variant_id).
    variant_counts: dict[str, int] = {}
    for m in materials_patch.materials:
        if m.role == "fuel" and m.source_variant_id:
            variant_counts[m.source_variant_id] = variant_counts.get(m.source_variant_id, 0) + 1
    for variant_id, count in variant_counts.items():
        if count > 1:
            issues.append(_issue("material_universe.fuel_variant_material_duplicate", f"source_variant_id {variant_id} used by {count} materials", row_kind="source_material_coverage", row_key=variant_id, source_variant_id=variant_id))
    return issues


def _collect_universes_issues(universes_patch: Any, view: MaterialUniverseBindingView) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    material_ids = {m.material_id for m in view.material_records}
    material_variant_by_id = {
        m.material_id: m.source_variant_id
        for m in view.material_records
        if m.role == "fuel" and m.source_variant_id
    }
    for universe in universes_patch.universes:
        uid = universe.universe_id
        if uid in seen_ids:
            issues.append(_issue("material_universe.universe_duplicate", f"duplicate universe_id {uid}", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid))
            continue
        seen_ids.add(uid)
        if not universe.cells:
            issues.append(_issue("material_universe.universe_empty", f"universe {uid} has no cells", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid))
            continue
        seen_cell_ids: set[str] = set()
        radii: list[tuple[str, float | None, float | None]] = []
        for cell in universe.cells:
            if cell.id in seen_cell_ids:
                issues.append(_issue("material_universe.cell_duplicate", f"duplicate cell_id {cell.id} in universe {uid}", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid, cell_id=cell.id))
                continue
            seen_cell_ids.add(cell.id)
            if cell.material_id and cell.material_id not in material_ids:
                issues.append(_issue("material_universe.material_reference_missing", f"universe {uid} cell {cell.id} references unknown material {cell.material_id}", row_kind="material_to_cell_binding", row_key=f"{uid}:{cell.id}", universe_id=uid, cell_id=cell.id, material_id=cell.material_id))
            declared_variant = _universe_fuel_variant_id(universe)
            material_variant = material_variant_by_id.get(cell.material_id or "")
            if (
                universe.kind == "fuel_pin"
                and cell.role == "fuel"
                and declared_variant
                and material_variant
                and material_variant != declared_variant
            ):
                issues.append(_issue(
                    "material_universe.fuel_variant_material_mismatch",
                    (
                        f"fuel universe {uid} declares fuel_variant_id={declared_variant!r} "
                        f"but fuel cell {cell.id!r} references material variant {material_variant!r}"
                    ),
                    row_kind="fuel_variant_identity",
                    row_key=uid,
                    universe_id=uid,
                    cell_id=cell.id,
                    material_id=cell.material_id,
                    expected_variant_id=declared_variant,
                    actual_variant_id=material_variant,
                    owner_patch_type="universes",
                ))
            if cell.r_min_cm is not None and cell.r_max_cm is not None:
                radii.append((cell.id, cell.r_min_cm, cell.r_max_cm))
                if cell.r_min_cm > cell.r_max_cm:
                    issues.append(_issue("material_universe.invalid_radial_order", f"cell {cell.id} r_min > r_max", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid, cell_id=cell.id, severity="error"))
        # Radial gap/overlap detection (concentric cells).
        sorted_radii = sorted([r for r in radii if r[1] is not None and r[2] is not None], key=lambda item: item[1] or 0.0)
        for i in range(1, len(sorted_radii)):
            prev_id, _, prev_max = sorted_radii[i - 1]
            cur_id, cur_min, _ = sorted_radii[i]
            if cur_min is None or prev_max is None:
                continue
            if cur_min > prev_max + 1e-6:
                issues.append(_issue("material_universe.radial_gap", f"radial gap between {prev_id} and {cur_id}", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid, severity="warning"))
            elif cur_min < prev_max - 1e-6:
                issues.append(_issue("material_universe.radial_overlap", f"radial overlap between {prev_id} and {cur_id}", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid))
    # Cell role / material role mismatches from the binding view.
    for binding in view.cell_material_bindings:
        for code in binding.issue_codes:
            issues.append(_issue(code, f"binding {binding.binding_id} failed", row_kind="material_to_cell_binding", row_key=binding.binding_id, universe_id=binding.universe_id, cell_id=binding.cell_id, material_id=binding.material_id))
    return issues


def _collect_fuel_variant_issues(view: MaterialUniverseBindingView) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    for variant in view.fuel_variant_bindings:
        for code in variant.issue_codes:
            issues.append(_issue(code, f"fuel variant {variant.variant_id} failed", row_kind="fuel_variant_identity", row_key=variant.variant_id, variant_id=variant.variant_id, material_id=variant.material_id))
        # Enrichment contract mismatch (deterministic numeric comparison).
        if variant.source_enrichment_wt_percent is not None and variant.material_enrichment_wt_percent is not None:
            if abs(variant.source_enrichment_wt_percent - variant.material_enrichment_wt_percent) > 0.01:
                issues.append(_issue("material_universe.enrichment_contract_mismatch", f"variant {variant.variant_id} source={variant.source_enrichment_wt_percent} material={variant.material_enrichment_wt_percent}", row_kind="fuel_variant_identity", row_key=variant.variant_id, variant_id=variant.variant_id, severity="warning"))
    return issues


def _collect_required_universe_issues(state: Any, universes_patch: Any, view: MaterialUniverseBindingView) -> list[dict[str, Any]]:
    """Check source-declared insert universes and variant identity locally.

    A localized insert may declare ``expected_insert_universe_ids`` with
    specific IDs.  When the generated universes use *different* IDs but
    cover the same ``required_segment_roles`` through their cells, we
    accept that as coverage instead of reporting a false-positive
    missing-universe error.
    """
    issues: list[dict[str, Any]] = []
    universe_by_id = {item.universe_id: item for item in universes_patch.universes}
    facts_env = _valid(state, "facts")
    facts = facts_env.content if facts_env is not None and isinstance(facts_env.content, dict) else {}
    inventory = getattr(state, "metadata", {}).get("planning_geometry_inventory") or {}
    protected_profiles = {
        str(profile.get("profile_id"))
        for profile in inventory.get("radial_profiles", [])
        if profile.get("protected_through_path_roles")
    }
    insert_profiles_by_requirement: dict[str, set[str]] = {}
    for binding in inventory.get("localized_insert_profiles", []) or []:
        req_id = str(binding.get("insert_requirement_id") or "")
        profile_id = str(binding.get("profile_id") or "")
        if req_id and profile_id:
            insert_profiles_by_requirement.setdefault(req_id, set()).add(profile_id)
    # Pre-compute cell-role coverage for each universe.
    universe_cell_roles: dict[str, set[str]] = {}
    for univ in universes_patch.universes:
        roles = set()
        for cell in univ.cells:
            r = getattr(cell, "role", "") or ""
            if r:
                roles.add(r)
        universe_cell_roles[univ.universe_id] = roles

    # Map insert_kind to acceptable universe kind.
    _insert_kind_to_universe_kind = {
        "control_rod": "control_rod",
        "absorber_insert": "control_rod",
        "pyrex_rod": "pyrex_rod",
        "thimble_plug": "thimble_plug",
        "instrumentation_insert": "instrument_tube",
    }

    for req in facts.get("localized_insert_requirements", []) or []:
        req_id = str(req.get("requirement_id", ""))
        expected = [str(item) for item in req.get("expected_insert_universe_ids", []) or []]
        required_profile = str(req.get("required_profile_id", ""))
        insert_kind = str(req.get("insert_kind", ""))
        required_segment_roles = set(req.get("required_segment_roles", []) or [])

        # Build set of candidate universes of the matching insert kind.
        matching_kind = _insert_kind_to_universe_kind.get(insert_kind, insert_kind)
        kind_matched_universes = [
            u for u in universes_patch.universes
            if u.kind == matching_kind or (insert_kind and insert_kind in u.kind)
        ]
        # Aggregate all cell roles across all kind-matched universes.
        aggregate_cell_roles: set[str] = set()
        for u in kind_matched_universes:
            aggregate_cell_roles |= universe_cell_roles.get(u.universe_id, set())

        # Profile-based fallback: if any universe was generated from an
        # inventory profile bound to this localized insert requirement,
        # consider all expected IDs covered.  The universe carries
        # metadata.localized_insert_requirement_id (stamped by the
        # fragmented pipeline from the manifest item).
        profile_covered = any(
            _metadata_covers_localized_insert(_universe_metadata(u), req_id)
            for u in universes_patch.universes
        )
        if not profile_covered and req_id in insert_profiles_by_requirement:
            required_profile_ids = insert_profiles_by_requirement[req_id]
            profile_covered = any(
                str(_universe_metadata(u).get("geometry_profile_id") or "") in required_profile_ids
                for u in universes_patch.universes
            )

        for uid in expected:
            universe = universe_by_id.get(uid)
            if universe is not None:
                # Exact ID match — check protected path if applicable.
                if required_profile in protected_profiles and not any(bool(cell.protected_through_path) for cell in universe.cells):
                    issues.append(_issue("material_universe.protected_path_missing", f"localized insert universe {uid} has no protected through-path cell", row_kind="required_universe_material_structure", row_key=uid, universe_id=uid, requirement_id=req_id))
                continue
            # Exact ID not found — check segment-role coverage as fallback.
            if required_segment_roles and required_segment_roles.issubset(aggregate_cell_roles):
                continue
            # Profile-based fallback: the universe was generated from an
            # inventory profile bound to this insert requirement.
            if profile_covered:
                continue
            # Neither exact ID, role coverage, nor profile binding — report missing.
            issues.append(_issue(
                "material_universe.localized_insert_universe_missing",
                f"localized insert {req_id} requires missing universe {uid}",
                row_kind="required_universe_material_structure", row_key=uid,
                universe_id=uid, requirement_id=req_id, required_ids=[uid],
                affected_json_paths=[f"/universes/{uid}"],
            ))
    # A fuel Universe may carry one and only one variant.  Do this per
    # universe, rather than comparing all active variants globally.
    for universe in universes_patch.universes:
        if universe.kind != "fuel_pin":
            continue
        fuel_variant_id = _universe_fuel_variant_id(universe)
        variants = {fuel_variant_id} if fuel_variant_id else set()
        cell_variants = {str(cell.source_variant_id) for cell in universe.cells if getattr(cell, "source_variant_id", None)}
        variants |= cell_variants
        if len(variants) > 1:
            issues.append(_issue("material_universe.fuel_variant_collapsed", f"fuel universe {universe.universe_id} contains multiple variants", row_kind="fuel_variant_identity", row_key=universe.universe_id, universe_id=universe.universe_id, variant_ids=sorted(variants)))
    return issues


def _collect_requirement_skeleton_issues(state: Any, materials_patch: Any, universes_patch: Any) -> list[dict[str, Any]]:
    """Validate generated patches against the post-Facts typed skeletons."""
    issues: list[dict[str, Any]] = []
    material_data = getattr(state, "metadata", {}).get("planning_material_requirement_skeleton")
    universe_data = getattr(state, "metadata", {}).get("planning_universe_requirement_skeleton")
    if isinstance(material_data, dict):
        materials = list(materials_patch.materials)
        for req in material_data.get("requirements", []):
            role = str(req.get("role", ""))
            variant = req.get("source_variant")
            matches = [m for m in materials if m.role == role and (not variant or m.source_variant_id == variant)]
            if not matches:
                issues.append(_issue("material_universe.material_requirement_missing", f"material requirement {req.get('requirement_id', '')} is not covered", row_kind="source_material_coverage", row_key=str(req.get("requirement_id", "")), requirement_id=req.get("requirement_id"), expected_role=role, expected_variant=variant))
            elif req.get("required_density") is not None and matches[0].density_g_cm3 is None:
                issues.append(_issue("material_universe.required_density_missing", f"material requirement {req.get('requirement_id', '')} lacks required density", row_kind="source_material_coverage", row_key=str(req.get("requirement_id", "")), requirement_id=req.get("requirement_id"), material_id=matches[0].material_id))
    if isinstance(universe_data, dict):
        universes = list(universes_patch.universes)
        for req in universe_data.get("requirements", []):
            profile = str(req.get("geometry_profile", ""))
            matches = []
            for universe in universes:
                u_meta = _universe_metadata(universe)
                if u_meta.get("geometry_profile_id") == profile or profile in (u_meta.get("source_requirement_ids") or []):
                    matches.append(universe)
            if not matches:
                issues.append(_issue("material_universe.required_universe_missing", f"universe requirement {profile} is not covered", row_kind="required_universe_material_structure", row_key=profile, geometry_profile_id=profile, component_kind=req.get("component_kind")))
    return issues


def _collect_background_issues(view: MaterialUniverseBindingView) -> list[dict[str, Any]]:
    """Background cell presence for fuel/guide-tube universes."""
    issues: list[dict[str, Any]] = []
    for universe in view.universe_records:
        if universe.kind in {"fuel_pin", "guide_tube", "instrument_tube"} and universe.background_cell_id is None:
            issues.append(_issue("material_universe.background_missing", f"universe {universe.universe_id} ({universe.kind}) has no background cell", row_kind="required_universe_material_structure", row_key=universe.universe_id, universe_id=universe.universe_id, severity="warning"))
    return issues


def run_material_universe_preflight(*, state: Any, policy: PlanClosedLoopPolicy, species_report: dict[str, Any] | None = None) -> MaterialUniversePreflightResult:
    """Run every deterministic check; never invoke an LLM."""
    if not material_universe_gate_applicable(state):
        return MaterialUniversePreflightResult(ok=True, summary={"applicable": False}, input_hash="")
    materials_env = _valid(state, "materials")
    universes_env = _valid(state, "universes")
    if materials_env is None or universes_env is None:
        return MaterialUniversePreflightResult(ok=False, issues=[_issue("material_universe.required_patch_missing", "materials or universes patch missing")], summary={"applicable": True, "ready": False})
    view = build_material_universe_binding_view(state=state, species_report=species_report)
    issues: list[dict[str, Any]] = []
    materials_parsed = None
    universes_parsed = None
    # Reuse single-patch validators.
    try:
        materials_parsed = parse_patch_content("materials", materials_env.content)
        materials_validation = validate_patch(materials_parsed)
        issues.extend({"code": i.code, "severity": i.severity, "message": i.message, "source_validator": True, "row_kind": "source_material_coverage", "row_key": ""} for i in materials_validation.issues if i.severity == "error")
    except Exception as exc:
        issues.append(_issue("material_universe.materials_schema_invalid", f"materials parse failed: {exc}"))
    try:
        universes_parsed = parse_patch_content("universes", universes_env.content)
        universes_validation = validate_patch(universes_parsed)
        issues.extend({"code": i.code, "severity": i.severity, "message": i.message, "source_validator": True, "row_kind": "required_universe_material_structure", "row_key": ""} for i in universes_validation.issues if i.severity == "error")
    except Exception as exc:
        issues.append(_issue("material_universe.universes_schema_invalid", f"universes parse failed: {exc}"))
    # Cross-patch checks run whenever both patches parsed successfully, even
    # if the single-patch validator produced errors.  The two check sets are
    # complementary, not mutually exclusive.
    if materials_parsed is not None and universes_parsed is not None:
        issues.extend(_collect_materials_issues(materials_parsed, view, species_report or {}))
        issues.extend(_collect_universes_issues(universes_parsed, view))
        issues.extend(_collect_fuel_variant_issues(view))
        issues.extend(_collect_required_universe_issues(state, universes_parsed, view))
        issues.extend(_collect_requirement_skeleton_issues(state, materials_parsed, universes_parsed))
        issues.extend(_collect_background_issues(view))
    # Deduplicate by (code, row_key).
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in issues:
        key = (str(item.get("code", "")), str(item.get("row_key", "")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    input_hash = material_universe_gate_input_hash(state, species_report=species_report, policy=policy)
    matrix = build_material_universe_contract_matrix(view, deduped)
    summary = {
        "applicable": True,
        "ready": True,
        "material_count": len(view.material_records),
        "universe_count": len(view.universe_records),
        "cell_binding_count": len(view.cell_material_bindings),
        "fuel_variant_contract_count": len(view.fuel_variant_bindings),
        "issue_count": len(deduped),
        "blocking_issue_count": sum(1 for item in deduped if item.get("severity") == "error"),
        "matrix_row_count": len(matrix.rows),
    }
    ok = not any(item.get("severity") == "error" for item in deduped)
    return MaterialUniversePreflightResult(ok=ok, binding_view=view, issues=deduped, summary=summary, input_hash=input_hash)


from .material_universe_evidence import material_universe_gate_applicable  # noqa: E402

__all__ = ["MaterialUniversePreflightResult", "run_material_universe_preflight"]
