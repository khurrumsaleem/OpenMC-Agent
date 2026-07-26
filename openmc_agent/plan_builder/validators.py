"""Patch validators for incremental plan building (Phase 2).

Each validator checks a single patch in isolation (plus optional lightweight
cross-references via :class:`PatchValidationContext`).  Validators never
assemble patches into a full ``SimulationPlan``; they only flag structural /
semantic issues so a future local retry router can target the exact patch
that needs regeneration.

Design constraints
------------------
* **No OpenMC, no renderer.**
* **Severity spectrum:** ``error`` blocks assembly; ``warning`` allows it but
  records a concern; ``info`` is purely advisory.
* **Approximate compositions are warnings, not errors** — the plan structure
  can still be built; material fidelity is a separate concern.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from openmc_agent.schemas import AgentBaseModel
from openmc_agent.radial_profile_validation import validate_concentric_radial_profile
from openmc_agent.material_species import (
    FISSILE_COMPOUND_ELEMENTS,
    classify_species_name,
    parse_empirical_formula,
)

from .material_resolution import resolve_material_id
from .patches import (
    AxialLayerPatchItem,
    AxialLayersPatch,
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    CoordinateConvention,
    CoreLayoutPatch,
    FactsPatch,
    LatticeTransformationPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
    PinMapPatch,
    ScopedExpectedCount,
    SettingsPatch,
    UniverseSpecPatch,
    UniversesPatch,
    LocalizedInsertProfilesPatch,
    normalized_coords,
)


# ---------------------------------------------------------------------------
# Result models
# ---------------------------------------------------------------------------


class PatchValidationIssue(AgentBaseModel):
    """A single validation issue found in a patch."""

    code: str
    severity: Literal["error", "warning", "info"] = "warning"
    message: str
    path: str | None = None
    expected: Any | None = None
    actual: Any | None = None


class PatchValidationResult(AgentBaseModel):
    """Result of validating a single patch."""

    patch_id: str | None = None
    patch_type: str
    ok: bool = True
    issues: list[PatchValidationIssue] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class PatchValidationContext(AgentBaseModel):
    """Cross-reference context for patch validation.

    Allows validators to check references (material IDs, universe IDs, lattice
    IDs) across patches without assembling a full SimulationPlan.
    """

    benchmark_id: str | None = None
    selected_variant: str | None = None
    expected_counts: dict[str, int] = Field(default_factory=dict)
    expected_counts_complete: bool = False
    reference_expected_counts: dict[str, int] = Field(default_factory=dict)
    expected_material_roles: list[str] = Field(default_factory=list)
    known_material_ids: list[str] = Field(default_factory=list)
    material_aliases: dict[str, str] = Field(default_factory=dict)
    known_universe_ids: list[str] = Field(default_factory=list)
    known_lattice_ids: list[str] = Field(default_factory=list)
    axial_domain_cm: tuple[float, float] | None = None
    active_fuel_region_cm: tuple[float, float] | None = None
    strict_benchmark: bool = False
    known_cell_ids: list[str] = Field(default_factory=list)
    cell_owner_universe_ids: dict[str, list[str]] = Field(default_factory=dict)
    material_roles_by_id: dict[str, str] = Field(default_factory=dict)
    known_overlay_summaries: list[dict[str, Any]] = Field(default_factory=list)
    has_spacer_grids: bool = False
    expected_spacer_grid_count: int | None = None
    # P2-FULLCORE-1: multi-assembly context fields
    model_scope: str | None = None
    assembly_count: int | None = None
    core_lattice_size: tuple[int, int] | None = None
    assembly_type_counts: dict[str, int] = Field(default_factory=dict)
    known_assembly_type_ids: list[str] = Field(default_factory=list)
    scoped_expected_counts: list[dict[str, Any]] = Field(default_factory=list)
    assembly_pitch_cm: float | None = None
    # P2-FULLCORE-2C-A: localized insert profile context
    known_insert_profile_ids: list[str] = Field(default_factory=list)
    insert_profile_summaries: list[dict[str, Any]] = Field(default_factory=list)
    movable_insert_facts: dict[str, Any] = Field(default_factory=dict)
    # P2-FULLCORE-2D-B: fuel variant source contract
    fuel_variant_requirements: list[dict[str, Any]] = Field(default_factory=list)
    material_summaries: list[dict[str, Any]] = Field(default_factory=list)
    universe_summaries: list[dict[str, Any]] = Field(default_factory=list)
    assembly_fuel_binding_summaries: list[dict[str, Any]] = Field(default_factory=list)
    # P2-FULLCORE-2C-C: localized insert placement contract
    localized_insert_requirements: list[dict[str, Any]] = Field(default_factory=list)
    localized_insert_universe_summaries: list[dict[str, Any]] = Field(default_factory=list)
    assembly_insert_binding_summaries: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Component-profile roles (mirror of assembly3d_guard._COMPONENT_PROFILE_ROLES)
# ---------------------------------------------------------------------------

_PATCH_COMPONENT_PROFILE_ROLES: frozenset[str] = frozenset({
    "lower_end_plug",
    "upper_end_plug",
    "lower_plenum",
    "upper_plenum",
    "gas_gap",
    "shoulder_gap",
    "lower_shoulder_gap",
    "upper_shoulder_gap",
})

_PATCH_STRUCTURAL_SLAB_ROLES: frozenset[str] = frozenset({
    "lower_nozzle",
    "upper_nozzle",
    "core_plate",
})

_MODERATOR_MATERIAL_ROLES: frozenset[str] = frozenset({
    "coolant",
    "moderator",
    "water",
})


# Weak evidence tokens for grid transformation detection.
_GRID_OPERATION_TOKENS: frozenset[str] = frozenset({
    "grid", "spacer", "top_grid", "replace_water_with_grid",
    "replace_water_with_top_grid", "spacer_grid",
})


def _is_likely_grid_transformation(
    t: LatticeTransformationPatchItem,
    has_spacer_grids: bool,
    overlay_summaries: list[dict[str, Any]],
) -> bool:
    """Heuristic: does this transformation likely express a spacer grid?

    Weak evidence (operation_id / purpose text containing grid tokens) is used
    only when combined with ``has_spacer_grids=True`` or existing spacer_grid
    overlays.  This is intentionally conservative: it must not fire on ordinary
    missing-universe errors that happen to contain the letter 'grid'.
    """
    text = " ".join([
        t.operation_id or "",
        t.purpose or "",
    ]).lower()
    has_grid_token = any(tok in text for tok in _GRID_OPERATION_TOKENS)
    if not has_grid_token:
        return False
    # Require corroborating evidence from the plan structure.
    if has_spacer_grids:
        return True
    has_grid_overlay = any(
        s.get("overlay_kind") == "spacer_grid" for s in overlay_summaries
    )
    return has_grid_overlay


# ---------------------------------------------------------------------------
# Alloy detection helpers
# ---------------------------------------------------------------------------

# (name_pattern, primary_element, display_name)
_ALLOY_SIGNATURES: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (re.compile(r"zircaloy[- ]?4", re.IGNORECASE), "Zr", "Zircaloy-4"),
    (re.compile(r"zircaloy", re.IGNORECASE), "Zr", "Zircaloy"),
    (re.compile(r"\bss[- ]?304\b", re.IGNORECASE), "Fe", "SS-304"),
    (re.compile(r"stainless.*304", re.IGNORECASE), "Fe", "SS-304"),
    (re.compile(r"inconel[- ]?718", re.IGNORECASE), "Ni", "Inconel-718"),
    (re.compile(r"inconel", re.IGNORECASE), "Ni", "Inconel"),
)


def _is_pure_element(composition: dict[str, float]) -> tuple[bool, str | None]:
    """Return (is_pure, element_symbol) for a 1-key composition."""
    if len(composition) == 1:
        return True, next(iter(composition))
    return False, None


def _detect_alloy_reduction(
    mat: MaterialSpecPatch,
) -> PatchValidationIssue | None:
    """Check if an alloy material is silently reduced to its primary element."""
    for pattern, primary_elem, display_name in _ALLOY_SIGNATURES:
        if not pattern.search(mat.name):
            continue
        is_pure, elem = _is_pure_element(mat.composition)
        if not is_pure or elem is None:
            continue
        if str(elem).strip().upper() != primary_elem.upper():
            continue
        # The alloy is reduced to pure primary element.
        if mat.composition_status == "confirmed":
            return PatchValidationIssue(
                code="patch.materials.alloy_reduced_to_pure_element",
                severity="error",
                message=(
                    f"material {mat.material_id!r} named {mat.name!r} "
                    f"({display_name}) has composition reduced to pure "
                    f"{primary_elem} but composition_status='confirmed'. "
                    f"{display_name} is a multi-element alloy; either supply "
                    f"the full composition or change composition_status to "
                    f"'approximate'/'needs_library'/'needs_confirmation' "
                    f"with an explicit warning."
                ),
                path=f"materials[{mat.material_id}].composition",
                expected=f"multi-element {display_name} composition",
                actual=f"pure {primary_elem}",
            )
        else:
            has_warning = any(
                "approxim" in w.lower() or "pure" in w.lower() or "alloy" in w.lower()
                for w in mat.warnings
            )
            if not has_warning:
                return PatchValidationIssue(
                    code="patch.materials.alloy_reduced_to_pure_element",
                    severity="warning",
                    message=(
                        f"material {mat.material_id!r} ({display_name}) is "
                        f"approximated as pure {primary_elem} but has no "
                        f"explicit approximation warning in warnings[]."
                    ),
                    path=f"materials[{mat.material_id}].warnings",
                )
            return None
    return None


# ---------------------------------------------------------------------------
# Per-type validators
# ---------------------------------------------------------------------------


def _validate_facts(
    patch: FactsPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    if patch.lattice_size is not None:
        nx, ny = patch.lattice_size
        if nx <= 0 or ny <= 0:
            issues.append(PatchValidationIssue(
                code="patch.schema_invalid",
                severity="error",
                message=f"lattice_size ({nx}, {ny}) must have positive dimensions",
                path="lattice_size",
                actual=patch.lattice_size,
            ))

    for label, z_range in (
        ("active_fuel_region_cm", patch.active_fuel_region_cm),
        ("axial_domain_cm", patch.axial_domain_cm),
    ):
        if z_range is not None and z_range[0] >= z_range[1]:
            issues.append(PatchValidationIssue(
                code="patch.schema_invalid",
                severity="error",
                message=f"{label} z_min={z_range[0]} must be < z_max={z_range[1]}",
                path=label,
                actual=z_range,
            ))

    for label, count in (
        ("expected_spacer_grid_count", patch.expected_spacer_grid_count),
        ("expected_pin_count", patch.expected_pin_count),
        ("expected_guide_tube_count", patch.expected_guide_tube_count),
        ("expected_instrument_tube_count", patch.expected_instrument_tube_count),
        ("expected_pyrex_count", patch.expected_pyrex_count),
        ("expected_thimble_plug_count", patch.expected_thimble_plug_count),
    ):
        if count is not None and count < 0:
            issues.append(PatchValidationIssue(
                code="patch.schema_invalid",
                severity="error",
                message=f"{label}={count} must be non-negative",
                path=label,
                actual=count,
            ))

    if (
        patch.selected_variant
        and "3b" in patch.selected_variant.lower()
        and (patch.expected_pyrex_count or 0) > 0
        and not patch.has_special_pin_map
    ):
        issues.append(PatchValidationIssue(
            code="patch.schema_invalid",
            severity="warning",
            message=(
                "variant '3B' with Pyrex facts should set "
                "has_special_pin_map=True"
            ),
            path="has_special_pin_map",
        ))

    if patch.missing_facts:
        for fact in patch.missing_facts[:5]:
            issues.append(PatchValidationIssue(
                code="patch.missing_required_field",
                severity="info",
                message=f"missing fact: {fact}",
                path="missing_facts",
            ))

    return issues


def _validate_materials(
    patch: MaterialsPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []
    seen_ids: set[str] = set()

    for mat in patch.materials:
        if mat.material_id in seen_ids:
            issues.append(PatchValidationIssue(
                code="patch.duplicate_id",
                severity="error",
                message=f"duplicate material_id {mat.material_id!r}",
                path=f"materials[{mat.material_id}]",
            ))
        seen_ids.add(mat.material_id)

        if mat.density_g_cm3 is not None and mat.density_g_cm3 <= 0:
            issues.append(PatchValidationIssue(
                code="patch.materials.invalid_density",
                severity="error",
                message=(
                    f"material {mat.material_id!r} density_g_cm3="
                    f"{mat.density_g_cm3} must be positive"
                ),
                path=f"materials[{mat.material_id}].density_g_cm3",
                actual=mat.density_g_cm3,
            ))

        if mat.composition and mat.composition_basis in {"atom_frac", "weight_frac"}:
            total = sum(float(value) for value in mat.composition.values())
            # ``composition_basis`` is the scientific meaning of the values,
            # not OpenMC's permissive add_* amount field.  Fraction bases must
            # be normalized fractions.  Percent-style vectors (sum≈100) must be
            # divided by 100 before they are accepted under atom_frac/weight_frac;
            # chemical ratios must declare stoichiometric_ratio instead.
            if total <= 0 or total > 1.0 + 1.0e-3:
                issues.append(PatchValidationIssue(
                    code="materials.composition_fraction_sum_invalid",
                    severity="error",
                    message=(
                        f"material {mat.material_id!r} declares composition_basis="
                        f"{mat.composition_basis!r}, but composition values sum to "
                        f"{total:g}; fraction bases must use normalized values "
                        "(sum <= 1.0). Divide source percentages by 100 before "
                        "using atom_frac/weight_frac, or use stoichiometric_ratio "
                        "for unnormalized chemical ratios."
                    ),
                    path=f"materials[{mat.material_id}].composition",
                    expected="0 < sum <= 1.0",
                    actual=total,
                ))

        # Static source/transport species contract.  This deliberately does
        # not import OpenMC or inspect nuclear data; library availability is a
        # runtime preflight concern.
        for name in mat.composition:
            kind = classify_species_name(name)
            if kind == "compound":
                issues.append(PatchValidationIssue(
                    code="materials.compound_in_transport_composition",
                    severity="error",
                    message=(f"material {mat.material_id!r} places chemical formula "
                             f"{name!r} in composition; use compound_components"),
                    path=f"materials[{mat.material_id}].composition.{name}",
                    actual=name,
                ))
            elif kind == "invalid":
                issues.append(PatchValidationIssue(
                    code="materials.species_name_invalid",
                    severity="error",
                    message=f"material {mat.material_id!r} has invalid transport species {name!r}",
                    path=f"materials[{mat.material_id}].composition.{name}",
                    actual=name,
                ))
        for index, component in enumerate(mat.compound_components):
            path = f"materials[{mat.material_id}].compound_components[{index}]"
            if component.fraction_basis is None:
                issues.append(PatchValidationIssue(
                    code="materials.compound_fraction_basis_missing", severity="error",
                    message=f"compound {component.formula!r} requires fraction_basis", path=path,
                ))
            if component.isotope_policy is None:
                issues.append(PatchValidationIssue(
                    code="materials.compound_isotope_policy_missing", severity="error",
                    message=f"compound {component.formula!r} requires isotope_policy", path=path,
                ))
            try:
                parsed = parse_empirical_formula(component.formula)
            except ValueError:
                issues.append(PatchValidationIssue(
                    code="materials.unresolved_species", severity="error",
                    message=f"compound formula {component.formula!r} is unsupported", path=path,
                ))
                continue
            if ({symbol for symbol, _ in parsed} & FISSILE_COMPOUND_ELEMENTS
                    and component.isotope_policy != "explicit_isotopes"):
                issues.append(PatchValidationIssue(
                    code="materials.fissile_compound_isotope_policy_missing", severity="error",
                    message=(f"fissile compound {component.formula!r} requires "
                             "explicit_isotopes; natural expansion would erase enrichment"),
                    path=path,
                ))

        alloy_issue = _detect_alloy_reduction(mat)
        if alloy_issue is not None:
            issues.append(alloy_issue)

        if mat.composition_status == "placeholder":
            issues.append(PatchValidationIssue(
                code="patch.materials.placeholder_composition",
                severity="warning",
                message=(
                    f"material {mat.material_id!r} has placeholder composition; "
                    "structure can still be built but material must be "
                    "resolved before export"
                ),
                path=f"materials[{mat.material_id}].composition_status",
            ))

        if mat.role and context.expected_material_roles:
            if mat.role not in context.expected_material_roles:
                issues.append(PatchValidationIssue(
                    code="patch.materials.missing_role",
                    severity="warning",
                    message=(
                        f"material {mat.material_id!r} role {mat.role!r} is not "
                        f"in expected roles {context.expected_material_roles}"
                    ),
                    path=f"materials[{mat.material_id}].role",
                ))

    # P2-FULLCORE-2D-B: fuel variant source contract checks
    if context.fuel_variant_requirements:
        required_variants = {
            v["variant_id"]: v for v in context.fuel_variant_requirements
            if v.get("variant_id")
        }
        fuel_mats_by_variant: dict[str, list[str]] = {vid: [] for vid in required_variants}
        for mat in patch.materials:
            if mat.role == "fuel" and mat.source_variant_id:
                if mat.source_variant_id in fuel_mats_by_variant:
                    fuel_mats_by_variant[mat.source_variant_id].append(mat.material_id)
                else:
                    issues.append(PatchValidationIssue(
                        code="materials.fuel_variant_source_id_missing",
                        severity="error",
                        message=(
                            f"fuel material {mat.material_id!r} has source_variant_id "
                            f"{mat.source_variant_id!r} not declared in facts"
                        ),
                        path=f"materials[{mat.material_id}].source_variant_id",
                        actual=mat.source_variant_id,
                    ))
        for vid, req in required_variants.items():
            mats = fuel_mats_by_variant.get(vid, [])
            if len(mats) == 0:
                issues.append(PatchValidationIssue(
                    code="materials.required_fuel_variant_missing",
                    severity="error",
                    message=(
                        f"required fuel variant {vid!r} (enrichment="
                        f"{req.get('enrichment_wt_percent')}) has no matching "
                        f"fuel material with source_variant_id={vid!r}"
                    ),
                    path="materials",
                    expected=vid,
                ))
            elif len(mats) > 1:
                issues.append(PatchValidationIssue(
                    code="materials.fuel_variant_duplicate_material",
                    severity="error",
                    message=(
                        f"fuel variant {vid!r} mapped to multiple materials: {mats}"
                    ),
                    path="materials",
                    actual=mats,
                ))

    return issues


def _validate_universes(
    patch: UniversesPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []
    seen_ids: set[str] = set()

    for univ in patch.universes:
        if univ.universe_id in seen_ids:
            issues.append(PatchValidationIssue(
                code="patch.universes.duplicate_universe_id",
                severity="error",
                message=f"duplicate universe_id {univ.universe_id!r}",
                path=f"universes[{univ.universe_id}]",
            ))
        seen_ids.add(univ.universe_id)

        if not univ.cells:
            issues.append(PatchValidationIssue(
                code="patch.universes.empty_universe",
                severity="error",
                message=f"universe {univ.universe_id!r} has no cells",
                path=f"universes[{univ.universe_id}].cells",
            ))
            continue

        for cell in univ.cells:
            if (
                cell.r_min_cm is not None
                and cell.r_max_cm is not None
                and cell.r_min_cm >= cell.r_max_cm
            ):
                issues.append(PatchValidationIssue(
                    code="patch.universes.invalid_radius_order",
                    severity="error",
                    message=(
                        f"cell {cell.id!r} in universe {univ.universe_id!r}: "
                        f"r_min_cm={cell.r_min_cm} >= r_max_cm={cell.r_max_cm}"
                    ),
                    path=f"universes[{univ.universe_id}].cells[{cell.id}]",
                    actual=(cell.r_min_cm, cell.r_max_cm),
                ))

        if univ.kind == "fuel_pin":
            has_fuel = any(
                "fuel" in c.role.lower() or c.material_id and "fuel" in c.material_id.lower()
                for c in univ.cells
            )
            if not has_fuel:
                issues.append(PatchValidationIssue(
                    code="patch.universes.fuel_cell_missing",
                    severity="warning",
                    message=(
                        f"fuel_pin universe {univ.universe_id!r} has no cell "
                        "with a fuel material/role"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))

        if univ.kind == "guide_tube":
            roles_lower = [c.role.lower() for c in univ.cells]
            has_water = any(
                "water" in r or "coolant" in r or "background" in r
                for r in roles_lower
            )
            has_wall = any(
                "wall" in r or "tube" in r or "clad" in r
                for r in roles_lower
            )
            if not has_water:
                issues.append(PatchValidationIssue(
                    code="patch.universes.guide_tube_wall_missing",
                    severity="warning",
                    message=(
                        f"guide_tube universe {univ.universe_id!r} has no "
                        "internal water/coolant region"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))
            if not has_wall:
                issues.append(PatchValidationIssue(
                    code="patch.universes.guide_tube_wall_missing",
                    severity="warning",
                    message=(
                        f"guide_tube universe {univ.universe_id!r} has no "
                        "tube wall material"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))

        if univ.kind == "pyrex_rod":
            has_pyrex = any(
                ("pyrex" in c.role.lower() or "pyrex" in (c.material_id or "").lower())
                for c in univ.cells
            )
            if not has_pyrex:
                issues.append(PatchValidationIssue(
                    code="patch.universes.pyrex_material_missing",
                    severity="warning",
                    message=(
                        f"pyrex_rod universe {univ.universe_id!r} has no cell "
                        "with pyrex material/role"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))
            # A pyrex rod has concentric annuli with gas/water gaps between
            # solid layers. If there are only 3-4 cells (pyrex + clad + water)
            # the LLM likely merged gap layers into adjacent solids.
            non_background = [c for c in univ.cells if c.region_kind != "background"]
            if len(non_background) <= 3:
                issues.append(PatchValidationIssue(
                    code="patch.universes.pyrex_gaps_missing",
                    severity="warning",
                    message=(
                        f"pyrex_rod universe {univ.universe_id!r} has only "
                        f"{len(non_background)} non-background cells — a pyrex "
                        f"rod should have inner_tube, gap, pyrex, gap, outer_clad "
                        f"(5+ cells); thin gas/water gaps between solid layers "
                        f"are likely missing"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))

        if univ.kind == "thimble_plug":
            roles_lower = [c.role.lower() for c in univ.cells]
            has_wall = any(
                "wall" in r or "tube" in r or "clad" in r
                for r in roles_lower
            )
            has_water = any(
                "water" in r or "coolant" in r
                for r in roles_lower
            )
            if not has_wall:
                issues.append(PatchValidationIssue(
                    code="patch.universes.thimble_plug_wall_missing",
                    severity="warning",
                    message=(
                        f"thimble_plug universe {univ.universe_id!r} has no tube "
                        "wall cell — a thimble plug sits inside a guide tube; "
                        "keep the Zircaloy-4 wall annulus"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))
            if not has_water:
                issues.append(PatchValidationIssue(
                    code="patch.universes.thimble_plug_water_missing",
                    severity="warning",
                    message=(
                        f"thimble_plug universe {univ.universe_id!r} has no "
                        "internal water gap — there should be water between "
                        "the plug and the tube wall"
                    ),
                    path=f"universes[{univ.universe_id}].cells",
                ))

        # Radial continuity check (reactor-neutral).
        radial_issues = validate_concentric_radial_profile(univ.universe_id, univ.cells)
        for ri in radial_issues:
            issues.append(PatchValidationIssue(
                code=ri.code,
                severity=ri.severity,
                message=ri.message,
                path=ri.path,
                actual=ri.details,
            ))

    # P2-FULLCORE-2D-B: fuel variant universe coverage checks
    if context.fuel_variant_requirements and context.material_summaries:
        required_variants = {
            v["variant_id"] for v in context.fuel_variant_requirements
            if v.get("variant_id")
        }
        # Build map: material_id -> source_variant_id from upstream materials
        mat_to_variant: dict[str, str] = {}
        for ms in context.material_summaries:
            mid = ms.get("material_id")
            vid = ms.get("source_variant_id")
            if mid and vid:
                mat_to_variant[mid] = vid

        # Check each active-fuel universe's fuel cell references exactly one variant
        for univ in patch.universes:
            fuel_variants_in_universe: set[str] = set()
            for cell in univ.cells:
                if cell.role == "fuel" and cell.material_id:
                    vid = mat_to_variant.get(cell.material_id)
                    if vid:
                        fuel_variants_in_universe.add(vid)
            if len(fuel_variants_in_universe) > 1:
                issues.append(PatchValidationIssue(
                    code="universes.multiple_fuel_variants_in_universe",
                    severity="error",
                    message=(
                        f"universe {univ.universe_id!r} contains fuel materials "
                        f"from multiple variants: {sorted(fuel_variants_in_universe)}"
                    ),
                    path=f"universes[{univ.universe_id}]",
                    actual=sorted(fuel_variants_in_universe),
                ))

        # Check each required variant has at least one reachable active-fuel universe
        reachable_variants: set[str] = set()
        for univ in patch.universes:
            for cell in univ.cells:
                if cell.role == "fuel" and cell.material_id:
                    vid = mat_to_variant.get(cell.material_id)
                    if vid:
                        reachable_variants.add(vid)

        for vid in required_variants - reachable_variants:
            issues.append(PatchValidationIssue(
                code="universes.fuel_material_unreachable",
                severity="error",
                message=(
                    f"required fuel variant {vid!r} is not reachable via any "
                    f"active-fuel universe (fuel cell with role='fuel')"
                ),
                path="universes",
                expected=vid,
            ))

        # Detect collapse: all required variants present in materials but
        # only one active-fuel universe with role='fuel' cell
        fuel_universe_count = sum(
            1 for u in patch.universes
            if any(c.role == "fuel" for c in u.cells)
        )
        if len(required_variants) > 1 and fuel_universe_count == 1:
            issues.append(PatchValidationIssue(
                code="universes.fuel_variant_collapsed",
                severity="error",
                message=(
                    f"{len(required_variants)} distinct fuel variants required but "
                    f"only 1 active-fuel universe generated — variants likely collapsed"
                ),
                path="universes",
                expected=len(required_variants),
                actual=fuel_universe_count,
            ))

    return issues


def _validate_pin_map(
    patch: PinMapPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    nx, ny = patch.lattice_size
    if nx <= 0 or ny <= 0:
        issues.append(PatchValidationIssue(
            code="patch.schema_invalid",
            severity="error",
            message=f"lattice_size ({nx}, {ny}) must have positive dimensions",
            path="lattice_size",
        ))

    conv = patch.coordinate_convention
    if context.strict_benchmark and conv.ordering == "unknown":
        issues.append(PatchValidationIssue(
            code="patch.pin_map.coordinate_convention_unknown",
            severity="error",
            message="coordinate_convention.ordering is 'unknown' for benchmark validation",
            path="coordinate_convention.ordering",
        ))

    coord_groups: dict[str, list[tuple[int, int]]] = {
        "guide_tube": patch.guide_tube_coords,
        "instrument_tube": patch.instrument_tube_coords,
        "pyrex_rod": patch.pyrex_rod_coords,
        "thimble_plug": patch.thimble_plug_coords,
        "water_cell": patch.water_cell_coords,
    }

    for group_name, coords in coord_groups.items():
        normalized = normalized_coords(coords, conv, patch.lattice_size)
        for row, col in normalized:
            if row < 0 or row >= nx or col < 0 or col >= ny:
                issues.append(PatchValidationIssue(
                    code="patch.pin_map.coord_out_of_bounds",
                    severity="error",
                    message=(
                        f"{group_name} coord ({row + conv.index_base}, "
                        f"{col + conv.index_base}) -> normalized ({row}, {col}) "
                        f"is out of bounds for lattice {nx}x{ny}"
                    ),
                    path=f"{group_name}_coords",
                    actual=(row, col),
                ))

    # Overlap detection with deterministic repair.
    # When a coordinate appears in multiple groups, keep the FIRST group
    # (by priority order) and remove it from subsequent groups.  This is
    # a safe, deterministic operation that resolves the common LLM mistake
    # of putting the same coordinate in both guide_tube_coords AND
    # pyrex_rod_coords / thimble_plug_coords.
    coord_map: dict[tuple[int, int], list[str]] = {}
    for group_name, coords in coord_groups.items():
        for nc in normalized_coords(coords, conv, patch.lattice_size):
            coord_map.setdefault(nc, []).append(group_name)

    overlaps = {
        coord: list(dict.fromkeys(groups))
        for coord, groups in coord_map.items()
        if len(dict.fromkeys(groups)) > 1
    }

    removed: list[str] = []

    if overlaps:
        # Report overlaps but do NOT modify the patch in-place.
        # The assembler's derive_localized_insert_loadings handles normalization.
        for coord, groups in overlaps.items():
            issues.append(PatchValidationIssue(
                code="patch.pin_map.coord_overlap_detected",
                severity="warning",
                message=(
                    f"coordinate {coord} appears in multiple groups: {groups}. "
                    f"The assembler will resolve this during localized insert derivation."
                ),
                path="*_coords",
            ))

    # Count checks against context. expected_counts may be partial when it came
    # from FactsPatch/LLM extraction; only reference/explicit-complete counts
    # are allowed to enforce full lattice sums.
    # Note: counts are computed AFTER overlap repair, so they reflect the
    # Base path counts (persistent, full-height positions in the lattice).
    # When localized_insert_intents are used, legacy coords are NOT base paths.
    # When only legacy coords exist (backward compat), they ARE treated as
    # base path positions (old behavior).
    has_modern_intents = bool(patch.localized_insert_intents)

    base_path_counts = {
        "guide_tube": len(patch.guide_tube_coords),
        "instrument_tube": len(patch.instrument_tube_coords),
        "water_cell": len(patch.water_cell_coords),
    }

    if has_modern_intents:
        # Modern mode: localized_insert_intents are insert counts.
        # Legacy coords (if any) are ignored — they should be empty.
        insert_counts: dict[str, int] = {}
        for intent in patch.localized_insert_intents:
            kind = intent.insert_kind
            insert_counts[kind] = insert_counts.get(kind, 0) + len(intent.coordinates)
        combined_insert_counts = insert_counts
        # Fuel pin = total - base path positions (inserts are within base paths).
        base_total = sum(base_path_counts.values())
    else:
        # Legacy mode: pyrex_rod_coords/thimble_plug_coords are base positions.
        combined_insert_counts = {
            "pyrex_rod": len(patch.pyrex_rod_coords),
            "thimble_plug": len(patch.thimble_plug_coords),
        }
        # Fuel pin = total - base - legacy insert positions (old behavior).
        base_total = sum(base_path_counts.values()) + sum(combined_insert_counts.values())

    special_counts = {
        **base_path_counts,
        **combined_insert_counts,
    }
    actual_counts = {
        "fuel_pin": nx * ny - base_total,
        **special_counts,
    }

    raw_expected = dict(context.reference_expected_counts or context.expected_counts)
    expected_counts: dict[str, int] = {}
    for key, val in raw_expected.items():
        if not isinstance(val, int):
            continue
        if key.startswith("expected_") and key.endswith("_count"):
            role = key.removeprefix("expected_").removesuffix("_count")
            role = {
                "pin": "fuel_pin",
                "pyrex": "pyrex_rod",
            }.get(role, role)
            expected_counts[role] = val
        else:
            expected_counts[key] = val
    expected_counts = {
        item: expected_val
        for item, expected_val in expected_counts.items()
        if item in actual_counts
    }

    for item, expected_val in expected_counts.items():
        count = actual_counts.get(item, 0)
        if expected_val is not None and count != expected_val:
            # No repaired_groups tracking — validator no longer modifies patches.
            # Count mismatch is evaluated on the raw input.
            severity: Literal["error", "warning", "info"] = (
                "error" if context.strict_benchmark else "warning"
            )
            issues.append(PatchValidationIssue(
                code="patch.pin_map.count_mismatch",
                severity=severity,
                message=(
                    f"{item} count={count} does not match "
                    f"expected {expected_val}"
                ),
                path=f"{item}_coords" if item != "fuel_pin" else "default_universe_id",
                expected=expected_val,
                actual=count,
            ))

    total_cells = nx * ny
    expected_complete = (
        context.expected_counts_complete
        or bool(context.reference_expected_counts)
    )
    if context.strict_benchmark and bool(context.reference_expected_counts):
        expected_complete = True
    if expected_counts and expected_complete:
        expected_sum = sum(expected_counts.values())
        if expected_sum != total_cells:
            issues.append(PatchValidationIssue(
                code="patch.pin_map.expected_counts_sum_mismatch",
                severity="error",
                message=(
                    f"complete expected_counts sum {expected_sum} does not "
                    f"match lattice size {total_cells}"
                ),
                path="expected_counts",
                expected=total_cells,
                actual=expected_sum,
            ))
    elif expected_counts:
        missing_roles = [
            role for role, count in actual_counts.items()
            if count > 0 and role not in expected_counts
        ]
        if missing_roles and sum(actual_counts.values()) == total_cells:
            issues.append(PatchValidationIssue(
                code="patch.pin_map.expected_counts_partial",
                severity="warning",
                message=(
                    "expected_counts is partial; checking only provided keys "
                    f"and accepting self-consistent lattice ({total_cells} pins). "
                    f"Missing roles: {missing_roles}"
                ),
                path="expected_counts",
                expected=expected_counts,
                actual=actual_counts,
            ))

    # Default universe reference
    if context.known_universe_ids:
        if patch.default_universe_id not in context.known_universe_ids:
            issues.append(PatchValidationIssue(
                code="patch.pin_map.default_universe_missing",
                severity="warning",
                message=(
                    f"default_universe_id {patch.default_universe_id!r} not "
                    f"in known universe ids"
                ),
                path="default_universe_id",
            ))

    return issues


def _validate_axial_layers(
    patch: AxialLayersPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    if not patch.layers:
        issues.append(PatchValidationIssue(
            code="patch.axial_layers.empty",
            severity="error",
            message="axial_layers patch has no layers",
            path="layers",
        ))
        return issues

    seen_ids: set[str] = set()
    layers_with_z: list[tuple[str, float, float]] = []

    for layer in patch.layers:
        if layer.layer_id in seen_ids:
            issues.append(PatchValidationIssue(
                code="patch.duplicate_id",
                severity="error",
                message=f"duplicate layer_id {layer.layer_id!r}",
                path=f"layers[{layer.layer_id}]",
            ))
        seen_ids.add(layer.layer_id)

        if layer.z_min_cm is not None and layer.z_max_cm is not None:
            if layer.z_min_cm >= layer.z_max_cm:
                issues.append(PatchValidationIssue(
                    code="patch.axial_layers.invalid_range",
                    severity="error",
                    message=(
                        f"layer {layer.layer_id!r}: z_min={layer.z_min_cm} "
                        f">= z_max={layer.z_max_cm}"
                    ),
                    path=f"layers[{layer.layer_id}]",
                    actual=(layer.z_min_cm, layer.z_max_cm),
                ))
            layers_with_z.append((layer.layer_id, layer.z_min_cm, layer.z_max_cm))
        else:
            # A layer cannot be materialized with a half-specified (or wholly
            # unspecified) z interval.  Treating this as a warning lets a
            # syntactically valid but unusable patch reach the assembler,
            # where it can obscure the owner error as a generic assembly
            # failure.  Human confirmation is not a substitute for a usable
            # interval: a controlled run must wait before accepting it.
            issues.append(PatchValidationIssue(
                code="patch.axial_layers.invalid_range",
                severity="error",
                message=(
                    f"layer {layer.layer_id!r} must provide both z_min_cm and "
                    "z_max_cm before it can be used in an axial plan"
                ),
                path=f"layers[{layer.layer_id}]",
                actual=(layer.z_min_cm, layer.z_max_cm),
            ))

        if layer.fill_type == "unknown" and (
            layer.z_min_cm is None or layer.z_max_cm is None
        ):
            issues.append(PatchValidationIssue(
                code="patch.axial_layers.fill_unknown",
                severity="error",
                message=(
                    f"layer {layer.layer_id!r} has an incomplete interval and "
                    "fill_type='unknown'; use a finite interval and an explicit "
                    "lattice, material, universe, or void fill"
                ),
                path=f"layers[{layer.layer_id}].fill_type",
            ))

        if layer.fill_type in ("lattice", "material", "universe") and not layer.fill_id:
            issues.append(PatchValidationIssue(
                code="patch.axial_layers.fill_missing",
                severity="error",
                message=(
                    f"layer {layer.layer_id!r} fill_type={layer.fill_type!r} "
                    "but fill_id is missing"
                ),
                path=f"layers[{layer.layer_id}].fill_id",
            ))

        # Validate fill_id references against known materials/universes so
        # the LLM cannot invent IDs that don't exist in upstream patches.
        if layer.fill_id and layer.fill_type == "material" and context.known_material_ids:
            resolved = resolve_material_id(
                layer.fill_id,
                set(context.known_material_ids),
                context.material_aliases,
            )
            if not resolved.ok:
                issues.append(PatchValidationIssue(
                    code="patch.axial_layers.fill_ref_missing",
                    severity="error",
                    message=(
                        f"layer {layer.layer_id!r} references missing "
                        f"material {layer.fill_id!r}. Known materials: "
                        f"{context.known_material_ids}"
                    ),
                    path=f"layers[{layer.layer_id}].fill_id",
                    actual=layer.fill_id,
                ))
            elif layer.role in _PATCH_STRUCTURAL_SLAB_ROLES:
                material_role = context.material_roles_by_id.get(
                    resolved.resolved_id or layer.fill_id, ""
                ).strip().lower()
                if material_role in _MODERATOR_MATERIAL_ROLES:
                    issues.append(PatchValidationIssue(
                        code="assembly3d.structural_slab_as_moderator",
                        severity="error",
                        message=(
                            f"layer {layer.layer_id!r} (role={layer.role!r}) is a "
                            f"structural slab but fill_id {layer.fill_id!r} has "
                            f"material role {material_role!r}. Use an input-defined "
                            "structural or homogenized-mixture material; coolant/moderator "
                            "belongs only in reflector or explicitly moderator layers."
                        ),
                        path=f"layers[{layer.layer_id}].fill_id",
                        actual=layer.fill_id,
                    ))

        if layer.fill_id and layer.fill_type == "universe" and context.known_universe_ids:
            if layer.fill_id not in set(context.known_universe_ids):
                issues.append(PatchValidationIssue(
                    code="patch.axial_layers.fill_ref_missing",
                    severity="error",
                    message=(
                        f"layer {layer.layer_id!r} references missing "
                        f"universe {layer.fill_id!r}. Known universes: "
                        f"{context.known_universe_ids}"
                    ),
                    path=f"layers[{layer.layer_id}].fill_id",
                    actual=layer.fill_id,
                ))

        # Early component-profile slab detection: a layer whose role is a
        # fuel-pin internal component profile (end plug, plenum, gas gap,
        # shoulder gap) must NOT use fill_type=material.  A material slab
        # replaces the entire cross section and truncates every pin/tube.
        # Catching this at patch-validation time avoids deferring the error
        # to the renderer capability stage.
        if layer.fill_type == "material" and layer.role in _PATCH_COMPONENT_PROFILE_ROLES:
            issues.append(PatchValidationIssue(
                code="assembly3d.component_profile_as_material_slab",
                severity="error",
                message=(
                    f"layer {layer.layer_id!r} (role={layer.role!r}) is a "
                    "component-profile segment but its fill is a single material "
                    f"({layer.fill_id!r}). This replaces the whole cross section "
                    "and truncates fuel pins, cladding, guide tubes and "
                    "instrument tubes. Use a lattice fill with a "
                    "replace_universe_family transformation so guide/instrument "
                    "tubes continue through while only the pin-internal content "
                    "changes."
                ),
                path=f"layers[{layer.layer_id}].fill_type",
            ))

    # Overlap detection
    for i in range(len(layers_with_z)):
        for j in range(i + 1, len(layers_with_z)):
            id_a, z_min_a, z_max_a = layers_with_z[i]
            id_b, z_min_b, z_max_b = layers_with_z[j]
            if z_min_a < z_max_b and z_max_a > z_min_b:
                issues.append(PatchValidationIssue(
                    code="patch.axial_layers.overlap",
                    severity="error",
                    message=(
                        f"layers {id_a!r} ({z_min_a}–{z_max_a}) and "
                        f"{id_b!r} ({z_min_b}–{z_max_b}) overlap"
                    ),
                    path="layers",
                ))

    # Domain containment
    domain = patch.axial_domain_cm or context.axial_domain_cm
    if domain is not None:
        for layer_id, z_min, z_max in layers_with_z:
            if z_min < domain[0] - 1e-9 or z_max > domain[1] + 1e-9:
                issues.append(PatchValidationIssue(
                    code="patch.axial_layers.invalid_range",
                    severity="warning",
                    message=(
                        f"layer {layer_id!r} ({z_min}–{z_max}) extends outside "
                        f"axial domain {domain}"
                    ),
                    path=f"layers[{layer_id}]",
                ))

    # Active fuel required for 3D assembly
    has_active_fuel = any(l.role == "active_fuel" for l in patch.layers)
    if not has_active_fuel:
        sev: Literal["error", "warning", "info"] = (
            "error" if context.strict_benchmark else "warning"
        )
        issues.append(PatchValidationIssue(
            code="patch.axial_layers.active_fuel_missing",
            severity=sev,
            message="no layer with role='active_fuel' found",
            path="layers",
        ))

    # Every declared lattice loading must be attached to at least one layer.
    # Otherwise localized-insert intents can look valid in the pin map but
    # have no physical axial interval in which to be applied.
    declared_loading_ids = {loading.loading_id for loading in patch.lattice_loadings}
    attached_loading_ids = {
        loading_id
        for layer in patch.layers
        for loading_id in ([layer.loading_id] if layer.loading_id else []) + list(layer.loading_ids)
    }
    for loading_id in sorted(declared_loading_ids - attached_loading_ids):
        issues.append(PatchValidationIssue(
            code="patch.axial_layers.loading_unattached",
            severity="error",
            message=(
                f"lattice loading {loading_id!r} is not referenced by any "
                "axial layer"
            ),
            path=f"lattice_loadings[{loading_id}]",
        ))

    # Default unit slab check
    for layer in patch.layers:
        if (
            layer.z_min_cm == -1.0
            and layer.z_max_cm == 1.0
            and (context.benchmark_id or context.selected_variant)
        ):
            issues.append(PatchValidationIssue(
                code="patch.axial_layers.default_unit_slab",
                severity="error",
                message=(
                    f"layer {layer.layer_id!r} uses default z=-1..1 unit slab "
                    "for an explicit 3D benchmark; provide real z ranges"
                ),
                path=f"layers[{layer.layer_id}]",
            ))

    # Lattice-loading transformation cross-reference validation.
    # Check replacement_universe_id and source_universe_id against known
    # universe IDs, cell IDs, and overlay summaries so defects are caught
    # at patch-validation time.
    if context.known_universe_ids and patch.lattice_loadings:
        universe_set = set(context.known_universe_ids)
        cell_set = set(context.known_cell_ids)
        cell_owners = context.cell_owner_universe_ids
        overlay_summaries = context.known_overlay_summaries
        has_grids = context.has_spacer_grids

        for loading in patch.lattice_loadings:
            lpath = f"lattice_loadings[{loading.loading_id}]"
            for t in loading.transformations:
                tpath = f"{lpath}.transformations[{t.operation_id}]"

                # replacement_universe_id must be a known universe
                rep = t.replacement_universe_id
                if rep and rep not in universe_set:
                    # Check if it's actually a cell ID
                    if rep in cell_set:
                        owners = cell_owners.get(rep, [])
                        issues.append(PatchValidationIssue(
                            code="lattice_transform.cell_id_used_as_universe",
                            severity="error",
                            message=(
                                f"operation {t.operation_id!r}: replacement_universe_id "
                                f"{rep!r} is a Cell ID, not a Universe ID. "
                                f"Owning universe(s): {owners}"
                            ),
                            path=f"{tpath}.replacement_universe_id",
                            actual=rep,
                        ))
                    else:
                        issues.append(PatchValidationIssue(
                            code="lattice_transform.replacement_universe_missing",
                            severity="error",
                            message=(
                                f"operation {t.operation_id!r}: replacement universe "
                                f"{rep!r} not defined"
                            ),
                            path=f"{tpath}.replacement_universe_id",
                            actual=rep,
                        ))

                        # Spacer-grid misuse detection
                        if _is_likely_grid_transformation(t, has_grids, overlay_summaries):
                            issues.append(PatchValidationIssue(
                                code="assembly3d.spacer_grid_transformation_misuse",
                                severity="error",
                                message=(
                                    f"operation {t.operation_id!r} appears to express a "
                                    "spacer grid as a lattice transformation "
                                    f"(replacement={rep!r}). Spacer grids must be "
                                    "expressed as axial_overlays with overlay_kind="
                                    "spacer_grid, not as replace_universe_family "
                                    "or coordinate_override transformations that "
                                    "replace the full pitch with a grid "
                                    "material/universe."
                                ),
                                path=f"{tpath}.replacement_universe_id",
                            ))

                # source_universe_id must be a known universe
                if t.operation_kind == "replace_universe_family":
                    sources = []
                    if t.source_universe_id:
                        sources.append(t.source_universe_id)
                    sources.extend(t.source_universe_ids)
                    for src in sources:
                        if src and src not in universe_set:
                            issues.append(PatchValidationIssue(
                                code="lattice_transform.source_universe_missing",
                                severity="error",
                                message=(
                                    f"operation {t.operation_id!r}: source universe "
                                    f"{src!r} not defined"
                                ),
                                path=f"{tpath}.source_universe_id",
                                actual=src,
                            ))

    return issues


def _validate_axial_overlays(
    patch: AxialOverlaysPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []
    seen_ids: set[str] = set()

    for ov in patch.overlays:
        if ov.overlay_id in seen_ids:
            issues.append(PatchValidationIssue(
                code="patch.axial_overlays.duplicate_overlay_id",
                severity="error",
                message=f"duplicate overlay_id {ov.overlay_id!r}",
                path=f"overlays[{ov.overlay_id}]",
            ))
        seen_ids.add(ov.overlay_id)

        # Z range requirements
        if ov.geometry_mode != "skeleton":
            if ov.z_min_cm is None or ov.z_max_cm is None:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.invalid_range",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        f"{ov.geometry_mode!r} requires z_min_cm and z_max_cm"
                    ),
                    path=f"overlays[{ov.overlay_id}]",
                ))
            elif ov.z_min_cm >= ov.z_max_cm:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.invalid_range",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r}: z_min={ov.z_min_cm} "
                        f">= z_max={ov.z_max_cm}"
                    ),
                    path=f"overlays[{ov.overlay_id}]",
                ))
        else:
            # Skeleton may omit z only if requires_human_confirmation
            if (
                ov.z_min_cm is None
                and ov.z_max_cm is None
                and not ov.requires_human_confirmation
                and not ov.material_id
            ):
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.material_missing",
                    severity="warning",
                    message=(
                        f"skeleton overlay {ov.overlay_id!r} has no z range, "
                        "no material, and no requires_human_confirmation"
                    ),
                    path=f"overlays[{ov.overlay_id}]",
                ))

        # Homogenized open region requirements
        if ov.geometry_mode == "homogenized_open_region":
            if not ov.target_lattice_id:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.target_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'homogenized_open_region' requires target_lattice_id"
                    ),
                    path=f"overlays[{ov.overlay_id}].target_lattice_id",
                ))
            if not ov.material_id:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.material_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'homogenized_open_region' requires material_id"
                    ),
                    path=f"overlays[{ov.overlay_id}].material_id",
                ))
            elif context.known_material_ids:
                resolved = resolve_material_id(
                    ov.material_id,
                    set(context.known_material_ids),
                    context.material_aliases,
                )
                if not resolved.ok:
                    issues.append(PatchValidationIssue(
                        code="patch.axial_overlays.material_missing",
                        severity="error",
                        message=resolved.reason or (
                            f"overlay {ov.overlay_id!r} references missing "
                            f"material_id {ov.material_id!r}"
                        ),
                        path=f"overlays[{ov.overlay_id}].material_id",
                        actual=ov.material_id,
                    ))
                elif resolved.resolved_id != ov.material_id:
                    issues.append(PatchValidationIssue(
                        code="patch.axial_overlays.material_alias_resolved",
                        severity="info",
                        message=(
                            f"overlay {ov.overlay_id!r} material_id "
                            f"{ov.material_id!r} resolves to "
                            f"{resolved.resolved_id!r}"
                        ),
                        path=f"overlays[{ov.overlay_id}].material_id",
                        expected=resolved.resolved_id,
                        actual=ov.material_id,
                    ))
            if ov.through_path_preserved is False:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.mode_semantic_contradiction",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'homogenized_open_region' inherently preserves "
                        "through-paths (fuel, gap, cladding, guide-tube wall, "
                        "instrument-tube wall, Pyrex, thimble plug, RCCA). "
                        "Setting through_path_preserved=false is a semantic "
                        "contradiction. Remove the field or set it to true."
                    ),
                    path=f"overlays[{ov.overlay_id}].through_path_preserved",
                    actual=False,
                    expected=True,
                ))

        # Mass-conserving outer-frame requirements
        if ov.geometry_mode == "mass_conserving_outer_frame":
            if not ov.target_lattice_id:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.target_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'mass_conserving_outer_frame' requires target_lattice_id"
                    ),
                    path=f"overlays[{ov.overlay_id}].target_lattice_id",
                ))
            if not ov.material_id:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.material_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'mass_conserving_outer_frame' requires material_id"
                    ),
                    path=f"overlays[{ov.overlay_id}].material_id",
                ))
            elif context.known_material_ids:
                resolved = resolve_material_id(
                    ov.material_id,
                    set(context.known_material_ids),
                    context.material_aliases,
                )
                if not resolved.ok:
                    issues.append(PatchValidationIssue(
                        code="patch.axial_overlays.material_missing",
                        severity="error",
                        message=resolved.reason or (
                            f"overlay {ov.overlay_id!r} references missing "
                            f"material_id {ov.material_id!r}"
                        ),
                        path=f"overlays[{ov.overlay_id}].material_id",
                        actual=ov.material_id,
                    ))
            if ov.total_mass_g is None or ov.total_mass_g <= 0:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.total_mass_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'mass_conserving_outer_frame' requires total_mass_g "
                        "(set from source grid mass per assembly)"
                    ),
                    path=f"overlays[{ov.overlay_id}].total_mass_g",
                ))
            if ov.through_path_preserved is False:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.mode_semantic_contradiction",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'mass_conserving_outer_frame' inherently preserves "
                        "through-paths (fuel, gap, cladding, guide-tube wall, "
                        "instrument-tube wall, Pyrex, thimble plug, RCCA). "
                        "Setting through_path_preserved=false is a semantic "
                        "contradiction. Remove the field or set it to true."
                    ),
                    path=f"overlays[{ov.overlay_id}].through_path_preserved",
                    actual=False,
                    expected=True,
                ))

        # Volume fraction calibrated requirements
        if ov.geometry_mode == "volume_fraction_calibrated":
            if ov.volume_fraction is None and ov.effective_density_g_cm3 is None:
                issues.append(PatchValidationIssue(
                    code="patch.axial_overlays.volume_fraction_missing",
                    severity="error",
                    message=(
                        f"overlay {ov.overlay_id!r} geometry_mode="
                        "'volume_fraction_calibrated' requires "
                        "volume_fraction or effective_density_g_cm3"
                    ),
                    path=f"overlays[{ov.overlay_id}]",
                ))

    return issues


def _validate_settings(
    patch: SettingsPatch,
    context: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    if patch.cross_sections_runtime_required:
        issues.append(PatchValidationIssue(
            code="patch.settings.cross_sections_runtime_only",
            severity="info",
            message=(
                "cross_sections path is a runtime requirement; it does not "
                "block plan generation"
            ),
            path="cross_sections_runtime_required",
        ))

    if not patch.tallies_required_for_smoke_test:
        issues.append(PatchValidationIssue(
            code="patch.settings.tallies_not_required_for_smoke",
            severity="info",
            message="tallies are not required for smoke test",
            path="tallies_required_for_smoke_test",
        ))

    if patch.plot_strategy != "full_assembly" and context.benchmark_id:
        issues.append(PatchValidationIssue(
            code="patch.settings.plot_not_full_assembly",
            severity="warning",
            message=(
                f"plot_strategy={patch.plot_strategy!r}; benchmark problems "
                "typically use 'full_assembly'"
            ),
            path="plot_strategy",
        ))

    return issues


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Assembly catalog validator (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


def _validate_assembly_catalog(
    patch: AssemblyCatalogPatch,
    ctx: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    if not patch.assembly_types:
        issues.append(PatchValidationIssue(
            code="assembly_catalog.empty",
            severity="error",
            message="assembly_catalog has no assembly_types",
        ))
        return issues

    seen_ids: set[str] = set()
    known_uvs = set(ctx.known_universe_ids)

    for atype in patch.assembly_types:
        if atype.assembly_type_id in seen_ids:
            issues.append(PatchValidationIssue(
                code="assembly_catalog.duplicate_type_id",
                severity="error",
                message=f"duplicate assembly_type_id {atype.assembly_type_id!r}",
                path=f"assembly_types[{atype.assembly_type_id}]",
            ))
            continue
        seen_ids.add(atype.assembly_type_id)

        pm = atype.pin_map
        nx, ny = pm.lattice_size
        if nx <= 0 or ny <= 0:
            issues.append(PatchValidationIssue(
                code="assembly_catalog.pin_map_invalid",
                severity="error",
                message=f"assembly type {atype.assembly_type_id!r} has invalid lattice_size ({nx},{ny})",
                path=f"assembly_types[{atype.assembly_type_id}].pin_map.lattice_size",
            ))

        if known_uvs and pm.default_universe_id not in known_uvs:
            issues.append(PatchValidationIssue(
                code="assembly_catalog.universe_missing",
                severity="error",
                message=(
                    f"assembly type {atype.assembly_type_id!r} pin_map default_universe_id "
                    f"{pm.default_universe_id!r} not found in known universes"
                ),
                path=f"assembly_types[{atype.assembly_type_id}].pin_map.default_universe_id",
                expected=list(known_uvs),
                actual=pm.default_universe_id,
            ))

        for intent in pm.localized_insert_intents:
            if known_uvs and intent.insert_universe_id not in known_uvs:
                issues.append(PatchValidationIssue(
                    code="assembly_catalog.universe_missing",
                    severity="error",
                    message=(
                        f"assembly type {atype.assembly_type_id!r} insert {intent.insert_id!r} "
                        f"universe {intent.insert_universe_id!r} not found"
                    ),
                    path=f"assembly_types[{atype.assembly_type_id}].pin_map.localized_insert_intents",
                ))

        total_cells = nx * ny
        special = (
            len(pm.guide_tube_coords)
            + len(pm.instrument_tube_coords)
            + len(pm.water_cell_coords)
        )
        if special > total_cells:
            issues.append(PatchValidationIssue(
                code="assembly_catalog.local_count_mismatch",
                severity="error",
                message=(
                    f"assembly type {atype.assembly_type_id!r} has {special} special coords "
                    f"but only {total_cells} total cells"
                ),
                path=f"assembly_types[{atype.assembly_type_id}].pin_map",
                expected=total_cells,
                actual=special,
            ))

        known_profile_ids = set(ctx.known_insert_profile_ids)
        for intent in pm.localized_insert_intents:
            if intent.axial_profile_id is not None:
                if known_profile_ids and intent.axial_profile_id not in known_profile_ids:
                    issues.append(PatchValidationIssue(
                        code="localized_insert.profile_ref_missing",
                        severity="error",
                        message=(
                            f"insert {intent.insert_id!r} in {atype.assembly_type_id!r} "
                            f"references profile {intent.axial_profile_id!r} "
                            "not in registry"
                        ),
                        path=f"assembly_types[{atype.assembly_type_id}].pin_map.localized_insert_intents",
                    ))
                if intent.z_min_cm is not None and intent.z_max_cm is not None:
                    issues.append(PatchValidationIssue(
                        code="localized_insert.profile_anchor_conflict",
                        severity="error",
                        message=(
                            f"insert {intent.insert_id!r} uses both axial_profile_id "
                            f"and z_min/z_max"
                        ),
                        path=f"assembly_types[{atype.assembly_type_id}].pin_map.localized_insert_intents",
                    ))
            elif intent.z_min_cm is None or intent.z_max_cm is None:
                issues.append(PatchValidationIssue(
                    code="localized_insert.profile_ref_missing",
                    severity="error",
                    message=(
                        f"insert {intent.insert_id!r} in {atype.assembly_type_id!r} "
                        "has neither axial_profile_id nor both z_min/z_max"
                    ),
                    path=f"assembly_types[{atype.assembly_type_id}].pin_map.localized_insert_intents",
                ))

    # P2-FULLCORE-2D-B: fuel variant binding checks
    if ctx.fuel_variant_requirements and ctx.universe_summaries:
        variant_by_assembly: dict[str, str] = {}
        for req in ctx.fuel_variant_requirements:
            vid = req.get("variant_id")
            if vid:
                for atid in req.get("assembly_type_ids", []):
                    variant_by_assembly[atid] = vid

        uv_fuel_variants: dict[str, list[str]] = {}
        for us in ctx.universe_summaries:
            uid = us.get("universe_id")
            if uid:
                uv_fuel_variants[uid] = us.get("fuel_variant_ids", [])

        for atype in patch.assembly_types:
            tid = atype.assembly_type_id
            expected_vid = variant_by_assembly.get(tid)
            if expected_vid is None:
                continue

            if atype.fuel_variant_id is None:
                issues.append(PatchValidationIssue(
                    code="assembly_catalog.fuel_variant_missing",
                    severity="error",
                    message=(
                        f"assembly type {tid!r} should have fuel_variant_id="
                        f"{expected_vid!r} (from facts) but field is missing"
                    ),
                    path=f"assembly_types[{tid}].fuel_variant_id",
                    expected=expected_vid,
                ))
                continue

            if atype.fuel_variant_id != expected_vid:
                issues.append(PatchValidationIssue(
                    code="assembly_catalog.fuel_variant_source_mismatch",
                    severity="error",
                    message=(
                        f"assembly type {tid!r} fuel_variant_id="
                        f"{atype.fuel_variant_id!r} does not match facts "
                        f"variant {expected_vid!r}"
                    ),
                    path=f"assembly_types[{tid}].fuel_variant_id",
                    expected=expected_vid,
                    actual=atype.fuel_variant_id,
                ))

            default_uv = atype.pin_map.default_universe_id
            uv_vids = uv_fuel_variants.get(default_uv, [])
            if uv_vids and expected_vid not in uv_vids:
                issues.append(PatchValidationIssue(
                    code="assembly_catalog.fuel_material_mismatch",
                    severity="error",
                    message=(
                        f"assembly type {tid!r} default_universe_id={default_uv!r} "
                        f"has fuel variant(s) {uv_vids} but expected {expected_vid!r}"
                    ),
                    path=f"assembly_types[{tid}].pin_map.default_universe_id",
                    expected=expected_vid,
                    actual=uv_vids,
                ))

        # Check for variant collapse: multiple distinct expected variants
        # sharing the same default universe
        if len(set(variant_by_assembly.values())) > 1:
            uv_to_types: dict[str, list[str]] = {}
            for atype in patch.assembly_types:
                tid = atype.assembly_type_id
                expected_vid = variant_by_assembly.get(tid)
                if expected_vid is None:
                    continue
                uv = atype.pin_map.default_universe_id
                uv_to_types.setdefault(uv, []).append(
                    f"{tid}({expected_vid})"
                )
            for uv, type_list in uv_to_types.items():
                vids_in_group = {
                    variant_by_assembly.get(t.split("(")[0])
                    for t in type_list
                }
                if len(vids_in_group) > 1:
                    issues.append(PatchValidationIssue(
                        code="assembly_catalog.distinct_fuel_variants_collapsed",
                        severity="error",
                        message=(
                            f"default_universe_id {uv!r} shared by assembly types "
                            f"with different fuel variants: {type_list}"
                        ),
                        path=f"assembly_types",
                        actual=type_list,
                    ))

    return issues


# ---------------------------------------------------------------------------
# Core layout validator (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


def _validate_core_layout(
    patch: CoreLayoutPatch,
    ctx: PatchValidationContext,
) -> list[PatchValidationIssue]:
    issues: list[PatchValidationIssue] = []

    n_rows, n_cols = patch.shape

    if n_rows <= 0 or n_cols <= 0:
        issues.append(PatchValidationIssue(
            code="core_layout.shape_mismatch",
            severity="error",
            message=f"core_layout shape ({n_rows},{n_cols}) is invalid",
            path="shape",
        ))
        return issues

    if len(patch.assembly_pattern) != n_rows:
        issues.append(PatchValidationIssue(
            code="core_layout.shape_mismatch",
            severity="error",
            message=(
                f"assembly_pattern has {len(patch.assembly_pattern)} rows "
                f"but shape says {n_rows}"
            ),
            path="assembly_pattern",
            expected=n_rows,
            actual=len(patch.assembly_pattern),
        ))

    for i, row in enumerate(patch.assembly_pattern):
        if len(row) != n_cols:
            issues.append(PatchValidationIssue(
                code="core_layout.row_length_mismatch",
                severity="error",
                message=f"row {i} has {len(row)} cols but shape says {n_cols}",
                path=f"assembly_pattern[{i}]",
                expected=n_cols,
                actual=len(row),
            ))

    known_types = set(ctx.known_assembly_type_ids)
    if not known_types:
        known_types = set(patch.expected_assembly_type_counts.keys())

    pattern_type_counts: dict[str, int] = {}
    for i, row in enumerate(patch.assembly_pattern):
        for j, type_id in enumerate(row):
            if known_types and type_id not in known_types:
                issues.append(PatchValidationIssue(
                    code="core_layout.assembly_type_missing",
                    severity="error",
                    message=f"assembly type {type_id!r} at ({i},{j}) not defined in catalog",
                    path=f"assembly_pattern[{i}][{j}]",
                    actual=type_id,
                ))
            pattern_type_counts[type_id] = pattern_type_counts.get(type_id, 0) + 1

    for type_id, expected_count in patch.expected_assembly_type_counts.items():
        actual_count = pattern_type_counts.get(type_id, 0)
        if actual_count != expected_count:
            issues.append(PatchValidationIssue(
                code="core_layout.multiplicity_mismatch",
                severity="error",
                message=(
                    f"assembly type {type_id!r}: expected {expected_count} instances "
                    f"but pattern has {actual_count}"
                ),
                path="expected_assembly_type_counts",
                expected=expected_count,
                actual=actual_count,
            ))

    if patch.assembly_pitch_cm is not None and patch.assembly_pitch_cm <= 0:
        issues.append(PatchValidationIssue(
            code="core_layout.pitch_invalid",
            severity="error",
            message=f"assembly_pitch_cm must be > 0, got {patch.assembly_pitch_cm}",
            path="assembly_pitch_cm",
            actual=patch.assembly_pitch_cm,
        ))

    if not patch.boundary:
        issues.append(PatchValidationIssue(
            code="core_layout.boundary_missing",
            severity="warning",
            message="boundary is empty; defaulting to 'vacuum'",
            path="boundary",
        ))

    return issues


# ---------------------------------------------------------------------------
# Catalog-layout cross-validation (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


def validate_catalog_layout_cross_references(
    catalog: AssemblyCatalogPatch,
    layout: CoreLayoutPatch,
) -> PatchValidationResult:
    """Cross-validate that core_layout references match the assembly catalog."""
    issues: list[PatchValidationIssue] = []
    catalog_type_ids = {at.assembly_type_id for at in catalog.assembly_types}

    for i, row in enumerate(layout.assembly_pattern):
        for j, type_id in enumerate(row):
            if type_id not in catalog_type_ids:
                issues.append(PatchValidationIssue(
                    code="core_layout.assembly_type_missing",
                    severity="error",
                    message=f"layout ({i},{j}) references {type_id!r} not in catalog",
                    path=f"assembly_pattern[{i}][{j}]",
                    actual=type_id,
                ))

    if layout.outer_assembly_type_id and layout.outer_assembly_type_id not in catalog_type_ids:
        issues.append(PatchValidationIssue(
            code="core_layout.assembly_type_missing",
            severity="error",
            message=(
                f"outer_assembly_type_id {layout.outer_assembly_type_id!r} "
                f"not in catalog"
            ),
            path="outer_assembly_type_id",
        ))

    errors = [i for i in issues if i.severity == "error"]
    return PatchValidationResult(
        patch_type="core_layout",
        ok=len(errors) == 0,
        issues=issues,
    )


def _validate_localized_insert_profiles(
    patch: LocalizedInsertProfilesPatch,
    ctx: PatchValidationContext,
) -> list[PatchValidationIssue]:
    """Validate a LocalizedInsertProfilesPatch (P2-FULLCORE-2C-A)."""
    from .localized_insert_profiles import validate_profile_registry

    result = validate_profile_registry(
        patch,
        known_universe_ids=ctx.known_universe_ids or None,
    )
    issues: list[PatchValidationIssue] = []
    for raw in result.issues:
        issues.append(PatchValidationIssue(
            code=raw.get("code", "localized_insert.unknown"),
            severity=raw.get("severity", "error"),
            message=raw.get("message", ""),
            path=raw.get("path"),
        ))
    if not patch.profiles:
        issues.append(PatchValidationIssue(
            code="localized_insert.profile_registry_empty",
            severity="warning",
            message="localized_insert_profiles patch has no profiles defined",
        ))
    return issues


_VALIDATORS: dict[str, Any] = {
    "facts": _validate_facts,
    "materials": _validate_materials,
    "universes": _validate_universes,
    "pin_map": _validate_pin_map,
    "axial_layers": _validate_axial_layers,
    "axial_overlays": _validate_axial_overlays,
    "settings": _validate_settings,
    "assembly_catalog": _validate_assembly_catalog,
    "core_layout": _validate_core_layout,
    "localized_insert_profiles": _validate_localized_insert_profiles,
}


def validate_patch(
    patch: BaseModel,
    context: PatchValidationContext | None = None,
) -> PatchValidationResult:
    """Validate a parsed patch model.

    Parameters
    ----------
    patch
        A parsed patch model (one of the ``*Patch`` classes from
        :mod:`openmc_agent.plan_builder.patches`).
    context
        Optional cross-reference context for reference checking.

    Returns
    -------
    PatchValidationResult
        The validation result with ``ok=True`` if no errors were found.
    """
    ctx = context or PatchValidationContext()
    patch_type = getattr(patch, "patch_type", None)
    if patch_type is None:
        return PatchValidationResult(
            patch_type="unknown",
            ok=False,
            issues=[PatchValidationIssue(
                code="patch.schema_invalid",
                severity="error",
                message=f"patch model {type(patch).__name__} has no patch_type field",
            )],
        )

    validator = _VALIDATORS.get(patch_type)
    if validator is None:
        return PatchValidationResult(
            patch_type=patch_type,
            ok=False,
            issues=[PatchValidationIssue(
                code="patch.schema_invalid",
                severity="error",
                message=f"no validator registered for patch_type {patch_type!r}",
            )],
        )

    issues = validator(patch, ctx)
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    infos = [i for i in issues if i.severity == "info"]

    return PatchValidationResult(
        patch_type=patch_type,
        ok=len(errors) == 0,
        issues=issues,
        summary={
            "error_count": len(errors),
            "warning_count": len(warnings),
            "info_count": len(infos),
        },
    )


__all__ = [
    "PatchValidationIssue",
    "PatchValidationResult",
    "PatchValidationContext",
    "validate_patch",
    "validate_catalog_layout_cross_references",
]
