"""Deterministic assembler for incremental plan building (Phase 3).

Reads validated patches (facts, materials, universes, pin_map, axial_layers,
axial_overlays, settings) and assembles them into a complete
:class:`~openmc_agent.schemas.SimulationPlan` — **without** calling an LLM,
OpenMC, or any renderer.

The assembler is the bridge between the future LLM patch generator (Phase 4+)
and the existing SimulationPlan validators / renderers.  It takes small,
independently-validatable patches and deterministically produces the same IR
that the monolithic path produces — but without the 25 KB JSON bottleneck.

Key responsibilities
--------------------
* **Pin map expansion**: expand special-pin coordinates into a full
  ``nx × ny`` ``universe_pattern``, so the LLM never emits 289 entries.
* **Material adaptation**: convert ``MaterialSpecPatch`` →
  ``ComplexMaterialSpec`` with ``NuclideSpec`` composition.
* **Universe / cell adaptation**: convert patch universes → plan universes.
* **Axial layer / overlay assembly**: map patch items → ``AxialLayerSpec`` /
  ``AxialOverlaySpec``.
* **Reference resolution**: check that all material / universe / lattice ids
  referenced by the patches actually exist.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from openmc_agent.plan_builder.grid_geometry_validation import (
    validate_grid_geometry_materialization,
)
from openmc_agent.schemas import (
    AgentBaseModel,
    AssemblySpec,
    AxialLayerSpec,
    AxialOverlaySpec,
    CellSpec,
    ComplexMaterialSpec,
    ComplexModelSpec,
    CoreSpec,
    ExecutionCheckSpec,
    FillRefSpec,
    LatticeLoadingSpec,
    LatticeSpec,
    LatticeTransformationOperation,
    NuclideSpec,
    PlotSpec,
    RegionSpec,
    RenderCapabilityReport,
    RunSettingsSpec,
    SimulationPlan,
    SurfaceSpec,
    UniverseSpec,
)

from .material_resolution import resolve_material_id
from ..material_policy import (
    DEFAULT_MATERIAL_POLICY,
    MaterialCompositionPolicy,
    MaterialCompositionReport,
    apply_policy_to_material_patch,
    build_composition_report,
    policy_from_value,
)
from openmc_agent.material_species import (
    MaterialSpeciesResolution,
    build_material_species_report,
    resolve_material_species,
)
from .patches import (
    AssemblyCatalogPatch,
    AxialLayerPatchItem,
    AxialLayersPatch,
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
    CellLayerPatch,
    CoordinateConvention,
    CoreLayoutPatch,
    FactsPatch,
    LatticeLoadingPatchItem,
    MaterialSpecPatch,
    MaterialsPatch,
    PinMapPatch,
    SettingsPatch,
    UniverseSpecPatch,
    UniversesPatch,
    normalized_coords,
)
from .pin_counts import compute_pin_role_counts


# ---------------------------------------------------------------------------
# Assembly issue / result models
# ---------------------------------------------------------------------------


class PlanAssemblyIssue(AgentBaseModel):
    """A single issue encountered during plan assembly."""

    code: str
    severity: Literal["error", "warning", "info"] = "error"
    message: str
    path: str | None = None
    expected: Any | None = None
    actual: Any | None = None


class PlanAssemblyResult(AgentBaseModel):
    """Result of assembling patches into a SimulationPlan."""

    ok: bool = True
    plan: SimulationPlan | None = None
    plan_dict: dict[str, Any] | None = None
    issues: list[PlanAssemblyIssue] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    material_composition_report: MaterialCompositionReport | None = None
    material_species_resolution_report: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Multi-assembly plot derivation (reactor-neutral)
# ---------------------------------------------------------------------------


def _derive_multi_assembly_plots(
    axial_layers: list[Any],
    overlays: list[Any],
    core_width: float,
    *,
    assembly_pitch_cm: float | None = None,
    axial_segment_index: list[dict[str, Any]] | None = None,
) -> list[PlotSpec]:
    """Derive plot specifications for a multi-assembly core.

    Generates XY slices at key axial elevations (active fuel, spacer grid,
    lower/upper structural) plus an XZ axial cross-section — all sized to
    cover the full core footprint.  Reactor-neutral: heights are derived
    from the actual axial layers and overlays, not hardcoded.
    """
    if not axial_layers:
        return [
            PlotSpec(
                basis="xy", origin=(0.0, 0.0, 0.0),
                width_cm=(core_width, core_width),
                filename="full_core_xy.png",
            ),
        ]

    z_min_all = min(L.z_min_cm for L in axial_layers)
    z_max_all = max(L.z_max_cm for L in axial_layers)
    axial_height = z_max_all - z_min_all

    # Active-fuel region: all lattice-filled layers.
    lattice_layers = [L for L in axial_layers if L.fill.type == "lattice"]
    if lattice_layers:
        af_z_min = min(L.z_min_cm for L in lattice_layers)
        af_z_max = max(L.z_max_cm for L in lattice_layers)
        af_mid = (af_z_min + af_z_max) / 2.0
    else:
        af_mid = (z_min_all + z_max_all) / 2.0

    plots: list[PlotSpec] = []
    high_res_xy = (2400, 2400)
    high_res_xz = (1600, 3600)
    zoom_width = assembly_pitch_cm if assembly_pitch_cm and assembly_pitch_cm > 0 else core_width

    def _add_unique(plot: PlotSpec) -> None:
        if plot.filename not in {p.filename for p in plots}:
            plots.append(plot)

    # 1. XY at active-fuel midplane — shows the pin lattice.
    _add_unique(PlotSpec(
        basis="xy", origin=(0.0, 0.0, af_mid),
        width_cm=(core_width, core_width),
        pixels=high_res_xy,
        filename="full_core_xy_active_fuel.png",
        purpose=f"XY slice at active-fuel midplane (z={af_mid:.1f} cm).",
    ))

    # Localized inserts (control rods, burnable absorbers, plugs, etc.) may be
    # active only over a subrange of the active fuel.  A single active-fuel
    # midplane can therefore miss them.  Prefer an inserted segment from the
    # materializer's concrete segment index when available.
    insert_segments = [
        s for s in (axial_segment_index or [])
        if s.get("active_types") and s.get("z_min_cm") is not None and s.get("z_max_cm") is not None
    ]
    if insert_segments:
        seg = insert_segments[len(insert_segments) // 2]
        z_mid = (float(seg["z_min_cm"]) + float(seg["z_max_cm"])) / 2.0
        _add_unique(PlotSpec(
            basis="xy", origin=(0.0, 0.0, z_mid),
            width_cm=(core_width, core_width),
            pixels=high_res_xy,
            filename="full_core_xy_localized_insert.png",
            purpose=(
                "XY slice through a localized-insert active axial segment "
                f"(z={z_mid:.1f} cm, active_types={seg.get('active_types')})."
            ),
        ))
        if zoom_width < core_width:
            _add_unique(PlotSpec(
                basis="xy", origin=(0.0, 0.0, z_mid),
                width_cm=(zoom_width, zoom_width),
                pixels=(1800, 1800),
                filename="center_assembly_xy_localized_insert.png",
                purpose=(
                    "Center assembly zoom through a localized-insert segment "
                    f"(z={z_mid:.1f} cm)."
                ),
            ))

    # 2. XY at spacer-grid mid-elevation — shows grid-decorated universes.
    if overlays:
        mid_idx = len(overlays) // 2
        ov = overlays[mid_idx]
        grid_mid = (ov.z_min_cm + ov.z_max_cm) / 2.0
        _add_unique(PlotSpec(
            basis="xy", origin=(0.0, 0.0, grid_mid),
            width_cm=(core_width, core_width),
            pixels=high_res_xy,
            filename="full_core_xy_grid_mid.png",
            purpose=f"XY slice at spacer-grid mid-elevation (z={grid_mid:.1f} cm).",
        ))
        if zoom_width < core_width:
            _add_unique(PlotSpec(
                basis="xy", origin=(0.0, 0.0, grid_mid),
                width_cm=(zoom_width, zoom_width),
                pixels=(1800, 1800),
                filename="center_assembly_xy_grid_mid.png",
                purpose=f"Center assembly zoom at spacer-grid mid-elevation (z={grid_mid:.1f} cm).",
            ))

    # 3. XY at lower structural region (bottom nozzle / core plate).
    material_layers = [L for L in axial_layers if L.fill.type == "material"]
    lower_struct = [L for L in material_layers if L.z_max_cm <= af_mid]
    if lower_struct:
        ls = lower_struct[-1]
        ls_mid = (ls.z_min_cm + ls.z_max_cm) / 2.0
        _add_unique(PlotSpec(
            basis="xy", origin=(0.0, 0.0, ls_mid),
            width_cm=(core_width, core_width),
            pixels=high_res_xy,
            filename="full_core_xy_lower_struct.png",
            purpose=f"XY slice at lower structural region (z={ls_mid:.1f} cm).",
        ))

    # 4. XY at upper structural region (upper nozzle / top plate).
    upper_struct = [L for L in material_layers if L.z_min_cm >= af_mid]
    if upper_struct:
        us = upper_struct[0]
        us_mid = (us.z_min_cm + us.z_max_cm) / 2.0
        _add_unique(PlotSpec(
            basis="xy", origin=(0.0, 0.0, us_mid),
            width_cm=(core_width, core_width),
            pixels=high_res_xy,
            filename="full_core_xy_upper_struct.png",
            purpose=f"XY slice at upper structural region (z={us_mid:.1f} cm).",
        ))

    # 5. XZ full-height axial cross-section.
    _add_unique(PlotSpec(
        basis="xz", origin=(0.0, 0.0, (z_min_all + z_max_all) / 2.0),
        width_cm=(core_width, axial_height),
        pixels=high_res_xz,
        filename="full_core_xz.png",
        purpose=f"XZ slice covering full axial domain ({axial_height:.1f} cm).",
    ))

    if zoom_width < core_width:
        _add_unique(PlotSpec(
            basis="xz", origin=(0.0, 0.0, (z_min_all + z_max_all) / 2.0),
            width_cm=(zoom_width, axial_height),
            pixels=(1400, 4200),
            filename="center_assembly_xz_full_height.png",
            purpose="Center assembly XZ zoom covering the full axial domain.",
        ))

    def _add_interface_zoom(filename: str, z_center: float, label: str) -> None:
        z_window = min(axial_height, max(40.0, axial_height * 0.18))
        z0 = max(z_min_all, z_center - z_window / 2.0)
        z1 = min(z_max_all, z_center + z_window / 2.0)
        _add_unique(PlotSpec(
            basis="xz", origin=(0.0, 0.0, (z0 + z1) / 2.0),
            width_cm=(zoom_width, z1 - z0),
            pixels=(1400, 2200),
            filename=filename,
            purpose=f"XZ zoom across the {label} structural interface.",
        ))

    if lattice_layers and lower_struct and zoom_width < core_width:
        _add_interface_zoom("center_assembly_xz_lower_interface.png", min(L.z_min_cm for L in lattice_layers), "lower")
    if lattice_layers and upper_struct and zoom_width < core_width:
        _add_interface_zoom("center_assembly_xz_upper_interface.png", max(L.z_max_cm for L in lattice_layers), "upper")

    return plots


# ---------------------------------------------------------------------------
# Pin map expansion
# ---------------------------------------------------------------------------

# PinMapPatch coordinate groups → universe kind.
_COORD_GROUP_TO_KIND: dict[str, str] = {
    "guide_tube_coords": "guide_tube",
    "instrument_tube_coords": "instrument_tube",
    "pyrex_rod_coords": "pyrex_rod",
    "thimble_plug_coords": "thimble_plug",
    "water_cell_coords": "water_cell",
}


def _build_kind_to_universe_map(
    universes_patch: UniversesPatch | None,
    pin_map: PinMapPatch,
) -> dict[str, str]:
    """Map universe kind → universe_id from the UniversesPatch."""
    kind_map: dict[str, str] = {}
    if universes_patch is not None:
        for univ in universes_patch.universes:
            kind_map[univ.kind] = univ.universe_id
    kind_map.setdefault("fuel_pin", pin_map.default_universe_id)
    return kind_map


def expand_pin_map(
    pin_map: PinMapPatch,
    *,
    universe_ids: dict[str, str] | None = None,
) -> list[list[str]]:
    """Expand a :class:`PinMapPatch` into a full ``nx × ny`` universe pattern.

    Parameters
    ----------
    pin_map
        The pin map patch with special coordinates.
    universe_ids
        Optional override mapping of kind → universe_id.  If ``None``, the
        function builds the map from ``pin_map.default_universe_id`` (fuel)
        and leaves special kinds unresolved (caller should supply this from
        the UniversesPatch).

    Returns
    -------
    list[list[str]]
        Nested ``[row][col]`` universe id grid of shape ``nx × ny``.

    Raises
    ------
    ValueError
        If a coordinate is out of bounds or two groups overlap.
    """
    nx, ny = pin_map.lattice_size
    if nx <= 0 or ny <= 0:
        raise ValueError(f"invalid lattice_size ({nx}, {ny})")

    kind_map = universe_ids or {}
    # PinMapPatch owns the base lattice universe explicitly.  A universe
    # catalog can legitimately contain several ``kind=fuel_pin`` entries for
    # axial profiles; catalog order must never turn one of those profiles into
    # the base lattice fill.
    default_uid = pin_map.default_universe_id

    # Initialize full grid with default universe.
    grid: list[list[str]] = [[default_uid for _ in range(ny)] for _ in range(nx)]

    conv = pin_map.coordinate_convention

    coord_groups: dict[str, list[tuple[int, int]]] = {
        "guide_tube": pin_map.guide_tube_coords,
        "instrument_tube": pin_map.instrument_tube_coords,
        "water_cell": pin_map.water_cell_coords,
    }

    assigned: dict[tuple[int, int], str] = {}

    for group_name, raw_coords in coord_groups.items():
        uid = kind_map.get(_COORD_GROUP_TO_KIND.get(group_name, group_name), "")
        if not uid:
            # Skip groups without a universe mapping — the caller should
            # supply the mapping from UniversesPatch.
            continue
        normalized = normalized_coords(raw_coords, conv, (nx, ny))
        for row, col in normalized:
            if row < 0 or row >= nx or col < 0 or col >= ny:
                raise ValueError(
                    f"{group_name} coord ({row}, {col}) out of bounds "
                    f"for lattice {nx}x{ny}"
                )
            if (row, col) in assigned and assigned[(row, col)] != uid:
                raise ValueError(
                    f"coord ({row}, {col}) assigned to both "
                    f"{assigned[(row, col)]!r} and {uid!r}"
                )
            grid[row][col] = uid
            assigned[(row, col)] = uid

    return grid


# ---------------------------------------------------------------------------
# Patch extraction helper
# ---------------------------------------------------------------------------


def _extract_patches(
    patches: list[Any],
) -> dict[str, Any]:
    """Index patches by patch_type.  Returns dict[str, patch_model]."""
    result: dict[str, Any] = {}
    for patch in patches:
        ptype = getattr(patch, "patch_type", None)
        if ptype and ptype not in result:
            result[ptype] = patch
    return result


# ---------------------------------------------------------------------------
# Per-component assemblers
# ---------------------------------------------------------------------------


def _assemble_materials(
    patch: MaterialsPatch,
    *,
    policy: MaterialCompositionPolicy = DEFAULT_MATERIAL_POLICY,
) -> tuple[list[ComplexMaterialSpec], list[PlanAssemblyIssue], MaterialCompositionReport, dict[str, Any]]:
    issues: list[PlanAssemblyIssue] = []
    materials: list[ComplexMaterialSpec] = []
    decisions: dict[str, Any] = {}
    rewritten_patches: list[MaterialSpecPatch] = []
    species_resolutions: list[MaterialSpeciesResolution] = []

    for mat in patch.materials:
        rewritten, decision = apply_policy_to_material_patch(mat, policy)
        rewritten_patches.append(rewritten)
        decisions[rewritten.material_id] = decision
        if decision.issue_code == "materials.alloy_library_applied":
            issues.append(PlanAssemblyIssue(
                code="materials.alloy_library_applied",
                severity="info",
                message=decision.reason,
                path=f"materials[{rewritten.material_id}]",
            ))
        elif decision.issue_code == "materials.alloy_library_missing":
            issues.append(PlanAssemblyIssue(
                code="materials.alloy_library_missing",
                severity="warning",
                message=decision.reason,
                path=f"materials[{rewritten.material_id}]",
            ))

    for mat in rewritten_patches:
        composition: list[NuclideSpec] = []
        if mat.composition_basis == "weight_frac":
            percent_type = "wo"
        elif mat.composition_basis == "atom_density_barn_cm":
            percent_type = "ao"  # absolute densities, used with set_density('sum')
        else:
            percent_type = "ao"
        species_resolution = resolve_material_species(
            material_id=mat.material_id,
            role=mat.role,
            composition=mat.composition,
            composition_basis=mat.composition_basis,
            compound_components=mat.compound_components,
            # Assembly accepts explicitly-audited historical plans. New LLM
            # patch generation is rejected earlier by the static validator.
            strict_composition=False,
            legacy_compatibility=True,
        )
        species_resolutions.append(species_resolution)
        for code in species_resolution.errors:
            issues.append(PlanAssemblyIssue(
                code=code,
                severity="error",
                message=(f"material {mat.material_id!r} species resolution failed: {code}"),
                path=f"materials[{mat.material_id}]",
            ))
        for warning in species_resolution.warnings:
            issues.append(PlanAssemblyIssue(
                code="material_resolution.warning", severity="warning", message=warning,
                path=f"materials[{mat.material_id}]",
            ))
        for component in species_resolution.species:
            composition.append(NuclideSpec(
                name=component.name,
                percent=component.fraction,
                percent_type=percent_type,
                kind=component.kind,
            ))

        assumptions = list(mat.warnings)
        if mat.composition_status in ("approximate", "needs_library", "placeholder"):
            assumptions.append(
                f"material {mat.material_id}: composition_status="
                f"{mat.composition_status}"
            )
        if mat.source_note:
            assumptions.append(f"material {mat.material_id}: {mat.source_note}")

        requires_conf: list[str] = []
        if mat.composition_status in ("needs_confirmation", "placeholder"):
            requires_conf.append(
                f"material {mat.material_id}: composition needs confirmation"
            )

        try:
            # Mixture materials: no direct composition, use component references.
            if mat.mixture_components:
                comp_ids = [c.material_id for c in mat.mixture_components]
                comp_fracs = [c.volume_fraction for c in mat.mixture_components]
                # Validate fractions sum to ~1
                frac_sum = sum(comp_fracs)
                if abs(frac_sum - 1.0) > 1e-6:
                    issues.append(PlanAssemblyIssue(
                        code="assembly.mixture_fraction_sum",
                        severity="error",
                        message=(
                            f"mixture material {mat.material_id!r} volume fractions "
                            f"sum to {frac_sum:.6f}, expected 1.0"
                        ),
                        path=f"materials[{mat.material_id}].mixture_components",
                    ))
                # Check for duplicate component IDs
                if len(set(comp_ids)) != len(comp_ids):
                    issues.append(PlanAssemblyIssue(
                        code="assembly.mixture_duplicate_component",
                        severity="error",
                        message=(
                            f"mixture material {mat.material_id!r} has duplicate "
                            f"component material IDs"
                        ),
                        path=f"materials[{mat.material_id}].mixture_components",
                    ))
                mat_spec = ComplexMaterialSpec(
                    id=mat.material_id,
                    name=mat.name,
                    density_value=mat.density_g_cm3,
                    density_unit="g/cm3" if mat.density_g_cm3 is not None else None,
                    temperature_k=mat.temperature_K,
                    source=mat.source_note,
                    assumptions=assumptions + [f"mixture of {comp_ids} with fractions {comp_fracs}"],
                    requires_human_confirmation=requires_conf,
                    mixture_component_ids=comp_ids,
                    mixture_volume_fractions=comp_fracs,
                    variant_scope=mat.variant_scope,
                    derivation_method=mat.derivation_method or "volume_fraction_mixture",
                )
            else:
                # For atom_density_barn_cm, use 'sum' density unit (OpenMC
                # interprets percent values as absolute atom/barn-cm).
                if mat.composition_basis == "atom_density_barn_cm":
                    density_unit = "sum"
                    density_value = None
                else:
                    density_unit = "g/cm3" if mat.density_g_cm3 is not None else None
                    density_value = mat.density_g_cm3

                # Map patch composition_basis to schema CompositionValueBasis.
                from openmc_agent.schemas import CompositionValueBasis as _Basis
                _PATCH_BASIS_MAP = {
                    "atom_frac": _Basis.ATOM_FRACTION,
                    "weight_frac": _Basis.WEIGHT_FRACTION,
                    "atom_density_barn_cm": _Basis.ATOM_DENSITY_BARN_CM,
                    "stoichiometric_ratio": _Basis.STOICHIOMETRIC_RATIO,
                    "ppm_by_weight": _Basis.PPM_BY_WEIGHT,
                    "ppm_by_atom": _Basis.PPM_BY_ATOM,
                    "unknown": _Basis.UNKNOWN,
                }
                mapped_basis = _PATCH_BASIS_MAP.get(mat.composition_basis, _Basis.UNKNOWN)

                mat_spec = ComplexMaterialSpec(
                    id=mat.material_id,
                    name=mat.name,
                    density_value=density_value,
                    density_unit=density_unit,
                    composition=composition,
                    temperature_k=mat.temperature_K,
                    source=mat.source_note,
                    assumptions=assumptions,
                    requires_human_confirmation=requires_conf,
                    composition_basis=mapped_basis,
                )
            # Apply deterministic normalization for materials with declared basis.
            from openmc_agent.material_normalization import normalize_material_semantics
            from openmc_agent.schemas import NormalizationStatus

            if mat_spec.composition and not mat_spec.is_mixture and mat_spec.macroscopic is None:
                try:
                    normalized_spec, norm_result = normalize_material_semantics(mat_spec)
                    if norm_result.normalization_status == NormalizationStatus.AMBIGUOUS:
                        issues.append(PlanAssemblyIssue(
                            code="material.normalization_ambiguous",
                            severity="error",
                            message=(
                                f"material {mat_spec.name!r} composition basis is ambiguous. "
                                f"Declare composition_basis explicitly (stoichiometric_ratio, "
                                f"ppm_by_weight, atom_frac, etc.)."
                            ),
                            path=f"materials[{mat.material_id}]",
                        ))
                    elif norm_result.normalization_status == NormalizationStatus.DETERMINISTICALLY_NORMALIZED:
                        mat_spec = normalized_spec
                        issues.append(PlanAssemblyIssue(
                            code="material.deterministically_normalized",
                            severity="info",
                            message=(
                                f"material {mat_spec.name!r} normalized: "
                                f"{norm_result.original_basis.value} → "
                                f"{norm_result.normalized_basis.value}"
                            ),
                            path=f"materials[{mat.material_id}]",
                        ))
                except Exception:
                    pass  # Don't block assembly on normalization errors.

            materials.append(mat_spec)
        except Exception as exc:
            issues.append(PlanAssemblyIssue(
                code="assembly.simulation_plan_schema_invalid",
                severity="error",
                message=f"material {mat.material_id!r} failed schema: {exc}",
                path=f"materials[{mat.material_id}]",
            ))

    report = build_composition_report(rewritten_patches, policy=policy, decisions=decisions)
    report.notes.append("material species resolution is recorded separately in material_species_resolution_report")
    # The caller carries the report in summary/result; attaching it to the
    # composition report avoids a second incompatible legacy report type.
    report.notes.extend([
        f"species:{r.material_id}: errors={len(r.errors)} warnings={len(r.warnings)}"
        for r in species_resolutions
    ])
    return materials, issues, report, build_material_species_report(species_resolutions)


def _assemble_universes(
    patch: UniversesPatch,
    material_ids: set[str],
    outer_moderator_material_id: str | None,
) -> tuple[list[UniverseSpec], list[CellSpec], list[SurfaceSpec], list[RegionSpec], list[PlanAssemblyIssue]]:
    issues: list[PlanAssemblyIssue] = []
    universes: list[UniverseSpec] = []
    all_cells: list[CellSpec] = []
    all_surfaces: list[SurfaceSpec] = []
    all_regions: list[RegionSpec] = []

    for univ in patch.universes:
        cell_ids: list[str] = []
        prev_surface_id: str | None = None
        has_background = False
        frame_surf_ids: list[str] = []  # P2-FULLCORE-2D-A-GRID-CLOSURE

        for cell_patch in univ.cells:
            cell_id = f"{univ.universe_id}_{cell_patch.id}"
            fill_type = _patch_fill_type_to_schema(cell_patch, material_ids)
            fill_id = _patch_fill_id_to_schema(cell_patch, material_ids)

            # Build surface/region from cell geometry (region_kind / r_min / r_max).
            region_id: str | None = None
            if cell_patch.region_kind in {"cylinder", "annulus"} and cell_patch.r_max_cm is not None:
                # Innermost solid cylinder (e.g. fuel pellet).
                surf_id = f"surf_{cell_id}"
                all_surfaces.append(SurfaceSpec(
                    id=surf_id, kind="zcylinder",
                    parameters={"r": cell_patch.r_max_cm},
                    purpose=f"Auto-generated pin cylinder for {cell_id}",
                ))
                is_annulus = (
                    cell_patch.region_kind == "annulus"
                    or (cell_patch.r_min_cm is not None and cell_patch.r_min_cm > 0.0)
                )
                if is_annulus and prev_surface_id is not None:
                    region_id = f"reg_{cell_id}_annulus"
                    all_regions.append(RegionSpec(
                        id=region_id,
                        expression=f"+{prev_surface_id} -{surf_id}",
                        surface_ids=[prev_surface_id, surf_id],
                        purpose=f"Annulus for {cell_id}",
                    ))
                else:
                    region_id = f"reg_{cell_id}_in"
                    all_regions.append(RegionSpec(
                        id=region_id, expression=f"-{surf_id}",
                        surface_ids=[surf_id],
                        purpose=f"Inside cylinder for {cell_id}",
                    ))
                prev_surface_id = surf_id

            elif cell_patch.region_kind == "background":
                # Outermost region (e.g. coolant outside cladding).
                has_background = True
                if prev_surface_id is not None:
                    region_id = f"reg_{cell_id}_out"
                    if frame_surf_ids:
                        # P2-FULLCORE-2D-A-GRID-CLOSURE: Exclude frame area from background.
                        # Background = outside cylinder AND NOT in frame
                        frame_expr = " ".join([
                            f"+{frame_surf_ids[0]}", f"-{frame_surf_ids[1]}",
                            f"+{frame_surf_ids[2]}", f"-{frame_surf_ids[3]}",
                            "~", "(",
                            f"+{frame_surf_ids[4]}", f"-{frame_surf_ids[5]}",
                            f"+{frame_surf_ids[6]}", f"-{frame_surf_ids[7]}",
                            ")",
                        ])
                        all_regions.append(RegionSpec(
                            id=region_id,
                            expression=f"+{prev_surface_id} ~ ( {frame_expr} )",
                            surface_ids=[prev_surface_id] + frame_surf_ids,
                            purpose=f"Inner moderator (outside cyl, excluding frame) for {cell_id}",
                        ))
                    else:
                        all_regions.append(RegionSpec(
                            id=region_id,
                            expression=f"+{prev_surface_id}",
                            surface_ids=[prev_surface_id],
                            purpose=f"Outside region for {cell_id}",
                        ))

            elif cell_patch.region_kind == "square_frame" and cell_patch.outer_side_cm is not None:
                # P2-FULLCORE-2D-A-HARDENING: Spacer grid square frame.
                # Creates 8 plane surfaces (4 outer + 4 inner) and a frame region.
                half_outer = cell_patch.outer_side_cm / 2.0
                inner_side = cell_patch.inner_side_cm or 0.0
                half_inner = inner_side / 2.0

                # Outer planes
                surf_ids_outer = []
                for axis, side, val in [
                    ("xplane", "lo", -half_outer), ("xplane", "hi", half_outer),
                    ("yplane", "lo", -half_outer), ("yplane", "hi", half_outer),
                ]:
                    s_id = f"surf_{cell_id}_o_{axis}_{side}"
                    param_key = "x0" if axis == "xplane" else "y0"
                    all_surfaces.append(SurfaceSpec(
                        id=s_id, kind=axis, parameters={param_key: val},
                        purpose=f"Grid frame outer {axis} {side} for {cell_id}",
                    ))
                    surf_ids_outer.append(s_id)

                # Inner planes
                surf_ids_inner = []
                for axis, side, val in [
                    ("xplane", "lo", -half_inner), ("xplane", "hi", half_inner),
                    ("yplane", "lo", -half_inner), ("yplane", "hi", half_inner),
                ]:
                    s_id = f"surf_{cell_id}_i_{axis}_{side}"
                    param_key = "x0" if axis == "xplane" else "y0"
                    all_surfaces.append(SurfaceSpec(
                        id=s_id, kind=axis, parameters={param_key: val},
                        purpose=f"Grid frame inner {axis} {side} for {cell_id}",
                    ))
                    surf_ids_inner.append(s_id)

                # Frame = inside outer box AND NOT inside inner box
                # Expression: +surf_xmin_o -surf_xmax_o +surf_ymin_o -surf_ymax_o
                #             ~ ( +surf_xmin_i -surf_xmax_i +surf_ymin_i -surf_ymax_i )
                expr_parts = [
                    f"+{surf_ids_outer[0]}", f"-{surf_ids_outer[1]}",
                    f"+{surf_ids_outer[2]}", f"-{surf_ids_outer[3]}",
                    "~", "(",
                    f"+{surf_ids_inner[0]}", f"-{surf_ids_inner[1]}",
                    f"+{surf_ids_inner[2]}", f"-{surf_ids_inner[3]}",
                    ")",
                ]
                region_id = f"reg_{cell_id}_frame"
                all_surfaces_ids = surf_ids_outer + surf_ids_inner
                all_regions.append(RegionSpec(
                    id=region_id,
                    expression=" ".join(expr_parts),
                    surface_ids=all_surfaces_ids,
                    purpose=f"Square frame for {cell_id}",
                ))
                # P2-FULLCORE-2D-A-GRID-CLOSURE: Save frame surfaces for background partition.
                # Do NOT reset prev_surface_id — the background cell needs it.
                frame_surf_ids = all_surfaces_ids

            elif cell_patch.region_kind == "box" and cell_patch.outer_side_cm is not None:
                # P2-FULLCORE-2D-A-HARDENING: Inner moderator box (inside grid frame).
                half = cell_patch.outer_side_cm / 2.0
                surf_ids_box = []
                for axis, side, val in [
                    ("xplane", "lo", -half), ("xplane", "hi", half),
                    ("yplane", "lo", -half), ("yplane", "hi", half),
                ]:
                    s_id = f"surf_{cell_id}_{axis}_{side}"
                    param_key = "x0" if axis == "xplane" else "y0"
                    all_surfaces.append(SurfaceSpec(
                        id=s_id, kind=axis, parameters={param_key: val},
                        purpose=f"Inner box {axis} {side} for {cell_id}",
                    ))
                    surf_ids_box.append(s_id)

                region_id = f"reg_{cell_id}_box"
                expr = " ".join([
                    f"+{surf_ids_box[0]}", f"-{surf_ids_box[1]}",
                    f"+{surf_ids_box[2]}", f"-{surf_ids_box[3]}",
                ])
                all_regions.append(RegionSpec(
                    id=region_id,
                    expression=expr,
                    surface_ids=surf_ids_box,
                    purpose=f"Inner moderator box for {cell_id}",
                ))
                prev_surface_id = None

            try:
                cell = CellSpec(
                    id=cell_id,
                    name=cell_patch.id,
                    fill_type=fill_type,
                    fill_id=fill_id,
                    region_id=region_id,
                    component_role=cell_patch.role,
                    protected_through_path=cell_patch.protected_through_path,
                    purpose=cell_patch.role,
                )
                all_cells.append(cell)
                cell_ids.append(cell_id)
            except Exception as exc:
                issues.append(PlanAssemblyIssue(
                    code="assembly.simulation_plan_schema_invalid",
                    severity="error",
                    message=f"cell {cell_id!r} failed schema: {exc}",
                    path=f"universes[{univ.universe_id}].cells[{cell_patch.id}]",
                ))
        if (
            prev_surface_id is not None
            and not has_background
            and outer_moderator_material_id is not None
        ):
            background_id = f"{univ.universe_id}_outer_moderator"
            background_region_id = f"reg_{background_id}_out"
            all_regions.append(RegionSpec(
                id=background_region_id,
                expression=f"+{prev_surface_id}",
                surface_ids=[prev_surface_id],
                purpose=f"Outer moderator for {univ.universe_id}",
            ))
            all_cells.append(CellSpec(
                id=background_id,
                name="outer_moderator",
                fill_type="material",
                fill_id=outer_moderator_material_id,
                region_id=background_region_id,
                component_role="coolant",
                purpose="Auto-generated outer moderator",
            ))
            cell_ids.append(background_id)
        try:
            universes.append(UniverseSpec(
                id=univ.universe_id,
                name=univ.kind,
                cell_ids=cell_ids,
                purpose=univ.kind,
            ))
        except Exception as exc:
            issues.append(PlanAssemblyIssue(
                code="assembly.simulation_plan_schema_invalid",
                severity="error",
                message=f"universe {univ.universe_id!r} failed schema: {exc}",
                path=f"universes[{univ.universe_id}]",
            ))

    return universes, all_cells, all_surfaces, all_regions, issues


def _outer_moderator_material_id(materials_patch: MaterialsPatch) -> str | None:
    """Return the input-declared moderator material for pin-universe closure."""
    for material in materials_patch.materials:
        if material.role.strip().lower() in {"coolant", "moderator", "water"}:
            return material.material_id
    return None


def _patch_fill_type_to_schema(
    cell: CellLayerPatch,
    material_ids: set[str],
) -> Literal["material", "universe", "lattice", "void"]:
    if cell.material_id is None and cell.fill_universe_id is None:
        return "void"
    if cell.fill_universe_id is not None:
        return "universe"
    return "material"


def _patch_fill_id_to_schema(
    cell: CellLayerPatch,
    material_ids: set[str],
) -> str | None:
    if cell.fill_universe_id is not None:
        return cell.fill_universe_id
    return cell.material_id


def _assemble_lattice(
    pin_map: PinMapPatch,
    facts: FactsPatch | None,
    universes_patch: UniversesPatch | None,
    universe_ids_on_plan: set[str],
) -> tuple[LatticeSpec | None, list[PlanAssemblyIssue], dict[str, int]]:
    issues: list[PlanAssemblyIssue] = []

    if pin_map.default_universe_id not in universe_ids_on_plan:
        issues.append(PlanAssemblyIssue(
            code="assembly.pin_map.default_universe_missing",
            severity="error",
            message=(
                f"PinMapPatch.default_universe_id {pin_map.default_universe_id!r} "
                "does not exist in UniversesPatch"
            ),
            path="pin_map.default_universe_id",
        ))
        return None, issues, {}

    kind_map = _build_kind_to_universe_map(universes_patch, pin_map)

    try:
        universe_pattern = expand_pin_map(pin_map, universe_ids=kind_map)
    except ValueError as exc:
        issues.append(PlanAssemblyIssue(
            code="assembly.pin_map_expansion_failed",
            severity="error",
            message=str(exc),
            path="pin_map",
        ))
        return None, issues, {}

    nx, ny = pin_map.lattice_size
    pitch = facts.pin_pitch_cm if facts and facts.pin_pitch_cm else 1.26

    outer_universe = pin_map.default_universe_id

    universe_kind_by_id = {
        univ.universe_id: univ.kind
        for univ in (universes_patch.universes if universes_patch is not None else [])
    }
    actual_pin_counts = compute_pin_role_counts(universe_pattern, universe_kind_by_id)

    expected_counts = _expected_counts_from_facts(facts)
    if expected_counts:
        expected_counts = _reconcile_expected_counts_with_actual(
            expected_counts,
            actual_pin_counts,
            total_cells=nx * ny,
            issues=issues,
        )
        # expected_counts is keyed by universe kind/role, but the lattice
        # universe_pattern is keyed by concrete universe id; re-key so the
        # downstream pin-count cross-check compares like with like.
        expected_counts = _rekey_expected_counts_to_universe_ids(
            expected_counts,
            universe_kind_by_id,
            used_universe_ids={
                uid for row in universe_pattern for uid in row
            },
        )

    try:
        lattice = LatticeSpec(
            id="assembly_lattice",
            name="assembly lattice",
            kind="rect",
            pitch_cm=(pitch, pitch),
            outer_universe_id=outer_universe,
            universe_pattern=universe_pattern,
            shape=(nx, ny),
            expected_counts=expected_counts,
            purpose="Expanded from PinMapPatch by deterministic assembler",
        )
    except Exception as exc:
        issues.append(PlanAssemblyIssue(
            code="assembly.simulation_plan_schema_invalid",
            severity="error",
            message=f"lattice assembly failed: {exc}",
            path="lattice",
        ))
        return None, issues, actual_pin_counts

    return lattice, issues, actual_pin_counts


def _normalize_axial_insert_pin_map(
    pin_map: PinMapPatch | None,
    axial_layers: AxialLayersPatch | None,
    universes_patch: UniversesPatch | None,
) -> tuple[PinMapPatch | None, AxialLayersPatch | None, list[PlanAssemblyIssue]]:
    """Treat finite insert rods as axial loadings over a guide-tube base.

    Some LLM patch generations put pyrex rods or thimble plugs directly in the
    base pin map.  That makes those locations occupy the whole active lattice,
    which blocks guide-tube water in axial regions where the insert is absent.
    The generic IR model is: guide tube in the base lattice, optional
    lattice_loading on axial layers that actually contain an insert.
    """
    issues: list[PlanAssemblyIssue] = []
    if pin_map is None or universes_patch is None:
        return pin_map, axial_layers, issues

    kind_map = _build_kind_to_universe_map(universes_patch, pin_map)
    guide_uid = kind_map.get("guide_tube")
    if not guide_uid:
        return pin_map, axial_layers, issues

    insert_coord_groups = {
        "pyrex_rod": list(pin_map.pyrex_rod_coords),
        "thimble_plug": list(pin_map.thimble_plug_coords),
    }
    if not any(insert_coord_groups.values()):
        return pin_map, axial_layers, issues

    guide_coords = _dedupe_coords(
        list(pin_map.guide_tube_coords)
        + insert_coord_groups["pyrex_rod"]
        + insert_coord_groups["thimble_plug"]
    )
    normalized_pin_map = pin_map.model_copy(update={
        "guide_tube_coords": guide_coords,
        "pyrex_rod_coords": [],
        "thimble_plug_coords": [],
    })
    issues.append(PlanAssemblyIssue(
        code="assembly.axial_insert_pin_map_normalized",
        severity="info",
        message=(
            "finite insert coordinates were moved out of the base pin map; "
            "base lattice keeps guide-tube geometry and axial layers carry "
            "insert-specific lattice loadings where available"
        ),
        path="pin_map",
        actual={
            "pyrex_rod_coords": len(insert_coord_groups["pyrex_rod"]),
            "thimble_plug_coords": len(insert_coord_groups["thimble_plug"]),
        },
    ))

    if axial_layers is None:
        return normalized_pin_map, axial_layers, issues

    axial_layers, loading_issues = _normalize_existing_axial_insert_loadings(
        axial_layers,
        pin_map,
        insert_coord_groups,
        kind_map,
        universes_patch,
    )
    issues.extend(loading_issues)

    pyrex_coords = insert_coord_groups["pyrex_rod"]
    pyrex_uid = kind_map.get("pyrex_rod")
    if not pyrex_coords or not pyrex_uid:
        return normalized_pin_map, axial_layers, issues

    if _axial_layers_already_load_universe(axial_layers, pyrex_uid):
        return normalized_pin_map, axial_layers, issues

    loading_id = _unique_loading_id(axial_layers, "pyrex_rod_loading")
    try:
        override_coords = normalized_coords(
            pyrex_coords,
            pin_map.coordinate_convention,
            pin_map.lattice_size,
        )
    except ValueError as exc:
        issues.append(PlanAssemblyIssue(
            code="assembly.axial_insert_loading_failed",
            severity="warning",
            message=f"could not normalize pyrex rod loading coordinates: {exc}",
            path="pin_map.pyrex_rod_coords",
        ))
        return normalized_pin_map, axial_layers, issues

    loading = LatticeLoadingPatchItem(
        loading_id=loading_id,
        base_lattice_id="assembly_lattice",
        derived_lattice_id=f"assembly_lattice_{loading_id}",
        overrides={pyrex_uid: override_coords},
        purpose="axial insert loading derived from finite insert coordinates",
    )
    updated_layers = _attach_loading_to_lattice_layers(
        axial_layers.layers,
        loading_id,
        axial_layers.lattice_loadings,
    )
    normalized_axial_layers = axial_layers.model_copy(update={
        "layers": updated_layers,
        "lattice_loadings": list(axial_layers.lattice_loadings) + [loading],
    })
    return normalized_pin_map, normalized_axial_layers, issues


def _normalize_existing_axial_insert_loadings(
    axial_layers: AxialLayersPatch,
    pin_map: PinMapPatch,
    insert_coord_groups: dict[str, list[tuple[int, int]]],
    kind_map: dict[str, str],
    universes_patch: UniversesPatch,
) -> tuple[AxialLayersPatch, list[PlanAssemblyIssue]]:
    """Normalize LLM-provided insert loadings against the pin-map convention.

    ``LatticeLoadingSpec`` is consumed by the renderer as 0-based row/col.
    LLMs often reuse the document's pin-map convention in the axial layer
    patch.  When the loading coordinates exactly match a finite-insert pin-map
    group, convert them deterministically.  Also keep plug-like finite inserts
    out of active-fuel loadings unless the universe has absorber/poison/control
    semantics; those positions should remain guide-tube water in the active
    lattice unless an explicit axial loading proves otherwise.
    """
    issues: list[PlanAssemblyIssue] = []
    if not axial_layers.lattice_loadings:
        return axial_layers, issues

    uid_to_kind = {uid: kind for kind, uid in kind_map.items()}
    active_loading_ids = {
        layer.loading_id
        for layer in axial_layers.layers
        if layer.fill_type == "lattice"
        and layer.role == "active_fuel"
        and layer.loading_id is not None
    }
    if not active_loading_ids:
        return axial_layers, issues

    changed = False
    normalized_loadings: list[LatticeLoadingPatchItem] = []
    for loading in axial_layers.lattice_loadings:
        overrides_changed = False
        normalized_overrides: dict[str, list[tuple[int, int]]] = {}
        for universe_id, positions in loading.overrides.items():
            kind = uid_to_kind.get(universe_id)
            if (
                loading.loading_id in active_loading_ids
                and kind in insert_coord_groups
                and not _universe_can_be_active_insert(universe_id, universes_patch)
            ):
                overrides_changed = True
                issues.append(PlanAssemblyIssue(
                    code="assembly.active_insert_loading_pruned",
                    severity="warning",
                    message=(
                        f"removed plug-like finite insert universe {universe_id!r} "
                        "from active-fuel lattice loading; base guide-tube water "
                        "is preserved for that axial region"
                    ),
                    path=f"axial_layers.lattice_loadings[{loading.loading_id}].overrides",
                    actual={universe_id: positions},
                ))
                continue

            new_positions = positions
            if kind in insert_coord_groups:
                new_positions, did_normalize = _normalize_loading_positions_if_raw_pin_coords(
                    positions,
                    insert_coord_groups[kind],
                    pin_map,
                )
                if did_normalize:
                    overrides_changed = True
                    issues.append(PlanAssemblyIssue(
                        code="assembly.axial_loading_coords_normalized",
                        severity="info",
                        message=(
                            f"converted lattice loading coordinates for {universe_id!r} "
                            "from pin-map convention to 0-based renderer convention"
                        ),
                        path=f"axial_layers.lattice_loadings[{loading.loading_id}].overrides.{universe_id}",
                    ))
            normalized_overrides[universe_id] = new_positions

        if overrides_changed:
            changed = True
            normalized_loadings.append(loading.model_copy(update={
                "overrides": normalized_overrides,
            }))
        else:
            normalized_loadings.append(loading)

    if not changed:
        return axial_layers, issues
    return axial_layers.model_copy(update={"lattice_loadings": normalized_loadings}), issues


def _normalize_loading_positions_if_raw_pin_coords(
    positions: list[tuple[int, int]],
    raw_pin_coords: list[tuple[int, int]],
    pin_map: PinMapPatch,
) -> tuple[list[tuple[int, int]], bool]:
    if not raw_pin_coords:
        return positions, False
    normalized = normalized_coords(
        raw_pin_coords,
        pin_map.coordinate_convention,
        pin_map.lattice_size,
    )
    if _coord_set(positions) == _coord_set(normalized):
        return positions, False
    if _coord_set(positions) == _coord_set(raw_pin_coords):
        return normalized, True
    return positions, False


def _coord_set(coords: list[tuple[int, int]]) -> set[tuple[int, int]]:
    return set(tuple(coord) for coord in coords)


def _universe_can_be_active_insert(
    universe_id: str,
    universes_patch: UniversesPatch,
) -> bool:
    universe = next(
        (item for item in universes_patch.universes if item.universe_id == universe_id),
        None,
    )
    if universe is None:
        return False
    terms = [universe.universe_id, universe.kind]
    terms.extend(cell.role for cell in universe.cells)
    text = " ".join(str(term).lower() for term in terms if term)
    active_tokens = (
        "absorber",
        "poison",
        "burnable",
        "control",
        "pyrex",
        "b4c",
        "boron",
        "gadol",
    )
    return any(token in text for token in active_tokens)


def _dedupe_coords(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    return list(dict.fromkeys(coords))


def _axial_layers_already_load_universe(
    axial_layers: AxialLayersPatch,
    universe_id: str,
) -> bool:
    return any(
        universe_id in loading.overrides
        for loading in axial_layers.lattice_loadings
    )


def _unique_loading_id(axial_layers: AxialLayersPatch, base: str) -> str:
    existing = {loading.loading_id for loading in axial_layers.lattice_loadings}
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _attach_loading_to_lattice_layers(
    layers: list[AxialLayerPatchItem],
    loading_id: str,
    existing_loadings: list[LatticeLoadingPatchItem],
) -> list[AxialLayerPatchItem]:
    lattice_layers = [layer for layer in layers if layer.fill_type == "lattice"]
    active_lattice_layers = [layer for layer in lattice_layers if layer.role == "active_fuel"]
    loading_by_id = {loading.loading_id: loading for loading in existing_loadings}
    target_ids = {
        layer.layer_id
        for layer in (active_lattice_layers or lattice_layers)
        if _layer_can_use_insert_loading(layer, loading_by_id)
    }
    if not target_ids:
        return layers
    return [
        layer.model_copy(update={"loading_id": loading_id})
        if layer.layer_id in target_ids else layer
        for layer in layers
    ]


def _layer_can_use_insert_loading(
    layer: AxialLayerPatchItem,
    loading_by_id: dict[str, LatticeLoadingPatchItem],
) -> bool:
    if layer.loading_id is None:
        return True
    existing = loading_by_id.get(layer.loading_id)
    return existing is not None and not existing.overrides


def _expected_counts_from_facts(facts: FactsPatch | None) -> dict[str, int] | None:
    if facts is None or facts.expected_pin_count is None:
        return None
    expected_counts = {"fuel_pin": facts.expected_pin_count}
    if facts.expected_guide_tube_count is not None:
        expected_counts["guide_tube"] = facts.expected_guide_tube_count
    if facts.expected_instrument_tube_count is not None:
        expected_counts["instrument_tube"] = facts.expected_instrument_tube_count
    if facts.expected_pyrex_count is not None:
        expected_counts["pyrex_rod"] = facts.expected_pyrex_count
    if facts.expected_thimble_plug_count is not None:
        expected_counts["thimble_plug"] = facts.expected_thimble_plug_count
    return expected_counts


def _reconcile_expected_counts_with_actual(
    expected_counts: dict[str, int],
    actual_pin_counts: dict[str, int],
    *,
    total_cells: int,
    issues: list[PlanAssemblyIssue],
) -> dict[str, int]:
    actual_total = sum(actual_pin_counts.values())
    mismatches = {
        role: {"expected": expected, "actual": actual_pin_counts.get(role, 0)}
        for role, expected in expected_counts.items()
        if actual_pin_counts.get(role, 0) != expected
    }
    expected_sum = sum(expected_counts.values())
    if mismatches and actual_total == total_cells:
        canonical = {
            role: actual_pin_counts.get(role, 0)
            for role in (
                "fuel_pin",
                "guide_tube",
                "instrument_tube",
                "pyrex_rod",
                "thimble_plug",
            )
            if role in actual_pin_counts
        }
        issues.append(PlanAssemblyIssue(
            code="assembly.expected_counts_reconciled",
            severity="warning",
            message=(
                "facts expected_counts disagreed with deterministic pin_map; "
                "using actual expanded pin counts for final lattice"
            ),
            path="lattice.expected_counts",
            expected=expected_counts,
            actual=actual_pin_counts,
        ))
        return canonical
    if expected_sum != total_cells and actual_total == total_cells:
        issues.append(PlanAssemblyIssue(
            code="assembly.expected_counts_reconciled",
            severity="warning",
            message=(
                f"facts expected_counts sum {expected_sum} != lattice size "
                f"{total_cells}; using actual expanded pin counts"
            ),
            path="lattice.expected_counts",
            expected=expected_counts,
            actual=actual_pin_counts,
        ))
        return dict(actual_pin_counts)
    return expected_counts


def _rekey_expected_counts_to_universe_ids(
    expected_counts: dict[str, int],
    universe_kind_by_id: dict[str, str],
    used_universe_ids: set[str] | None = None,
) -> dict[str, int]:
    """Re-key role/kind-keyed ``expected_counts`` to concrete universe-id keys.

    Facts/reconciliation ``expected_counts`` are typically keyed by universe
    kind/role (e.g. ``"fuel_pin"``), but the assembled lattice
    ``universe_pattern`` is keyed by concrete universe id (e.g.
    ``"fuel_variant_3B"``).  The downstream pin-count cross-check compares
    ``expected_counts`` against the pattern directly, so a role-keyed
    expected_counts would spuriously mismatch a correct pattern.

    Re-key to universe-id space when each kind maps to exactly one universe.
    When a kind maps to several universes (e.g. the LLM emitted fuel universes
    for multiple variants like ``fuel_pin_3A`` + ``fuel_pin_3B``), disambiguate
    by the universe id actually used in the pattern.  Counts whose key is
    already a universe id, or whose kind is ambiguous/unknown and cannot be
    disambiguated by usage, are preserved unchanged, so genuine count errors
    are still caught.
    """
    if not expected_counts or not universe_kind_by_id:
        return expected_counts
    kind_to_uids: dict[str, list[str]] = {}
    for uid, kind in universe_kind_by_id.items():
        kind_to_uids.setdefault(kind, []).append(uid)
    rekeyed: dict[str, int] = {}
    for key, count in expected_counts.items():
        uids = kind_to_uids.get(key)
        if uids and len(uids) == 1:
            rekeyed[uids[0]] = count
        elif uids and len(uids) > 1 and used_universe_ids:
            in_use = [u for u in uids if u in used_universe_ids]
            if len(in_use) == 1:
                rekeyed[in_use[0]] = count
            else:
                rekeyed[key] = count
        else:
            rekeyed[key] = count
    return rekeyed


def _assemble_axial_layers(
    patch: AxialLayersPatch,
    lattice_id: str,
    material_ids: set[str],
) -> tuple[list[AxialLayerSpec], list[LatticeLoadingSpec], list[PlanAssemblyIssue], dict[str, str]]:
    issues: list[PlanAssemblyIssue] = []
    layers: list[AxialLayerSpec] = []
    loadings: list[LatticeLoadingSpec] = []
    aliases_applied: dict[str, str] = {}

    for item in patch.lattice_loadings:
        transformations = [
            LatticeTransformationOperation(
                operation_id=t.operation_id,
                operation_kind=t.operation_kind,
                replacement_universe_id=t.replacement_universe_id,
                source_universe_id=t.source_universe_id,
                source_universe_ids=list(t.source_universe_ids),
                target_coordinates=[tuple(c) for c in t.target_coordinates],
                component_role=t.component_role,
                component_path_id=t.component_path_id,
                preserve_component_roles=list(t.preserve_component_roles),
                preserve_path_ids=list(t.preserve_path_ids),
                priority=t.priority,
                purpose=t.purpose,
            )
            for t in item.transformations
        ]
        try:
            loadings.append(LatticeLoadingSpec(
                id=item.loading_id,
                base_lattice_id=item.base_lattice_id or lattice_id,
                derived_lattice_id=item.derived_lattice_id,
                transformations=transformations,
                overrides=item.overrides,
                purpose=item.purpose,
            ))
        except Exception as exc:
            issues.append(PlanAssemblyIssue(
                code="assembly.simulation_plan_schema_invalid",
                severity="error",
                message=f"lattice loading {item.loading_id!r} failed schema: {exc}",
                path=f"axial_layers.lattice_loadings[{item.loading_id}]",
            ))

    for item in patch.layers:
        fill, alias = _layer_fill_ref(item, lattice_id, material_ids)
        if alias is not None:
            original_id, resolved_id = alias
            aliases_applied[original_id] = resolved_id
            issues.append(PlanAssemblyIssue(
                code="assembly.material_alias_resolved",
                severity="info",
                message=(
                    f"axial layer {item.layer_id!r} fill material "
                    f"{original_id!r} resolved to {resolved_id!r}"
                ),
                path=f"axial_layers[{item.layer_id}].fill_id",
                expected=resolved_id,
                actual=original_id,
            ))
        try:
            layer = AxialLayerSpec(
                id=item.layer_id,
                name=item.role,
                z_min_cm=item.z_min_cm if item.z_min_cm is not None else 0.0,
                z_max_cm=item.z_max_cm if item.z_max_cm is not None else 0.0,
                fill=fill,
                loading_id=item.loading_id,
                loading_ids=list(item.loading_ids) if item.loading_ids else [],
                purpose=item.role,
            )
            layers.append(layer)
        except Exception as exc:
            issues.append(PlanAssemblyIssue(
                code="assembly.simulation_plan_schema_invalid",
                severity="error",
                message=f"axial layer {item.layer_id!r} failed schema: {exc}",
                path=f"axial_layers[{item.layer_id}]",
            ))

    return layers, loadings, issues, aliases_applied


_COMPONENT_PROFILE_LOADING_TOKENS = (
    "shoulder_gap",
    "end_plug",
    "plenum",
    "gas_gap",
)


def _auto_attach_component_profile_loadings(
    layers: list[AxialLayerSpec],
    loadings: list[LatticeLoadingSpec],
) -> tuple[list[AxialLayerSpec], list[PlanAssemblyIssue]]:
    """Attach provably matching component-profile loadings to lattice layers.

    Localized inserts are attached deterministically before assembly, but LLM
    axial patches can declare an end-plug/plenum/shoulder loading and omit its
    layer reference.  Leaving such a layer bare makes it render the base fuel
    lattice.  Infer only generic component-role matches from the loading's
    structured id/purpose/replacement ids; ambiguous loadings remain
    unattached for the axial preflight to fail closed.
    """
    attached = {
        loading_id
        for layer in layers
        for loading_id in ([layer.loading_id] if layer.loading_id else []) + list(layer.loading_ids)
    }
    issues: list[PlanAssemblyIssue] = []
    result = list(layers)
    for loading in loadings:
        if loading.id in attached:
            continue
        if not any(
            transform.operation_kind == "replace_universe_family"
            for transform in loading.transformations
        ):
            continue
        semantic_text = " ".join(
            [loading.id, loading.purpose]
            + [
                transform.replacement_universe_id
                for transform in loading.transformations
                if transform.replacement_universe_id
            ]
        ).lower()
        hints = [
            token for token in _COMPONENT_PROFILE_LOADING_TOKENS
            if token in semantic_text
        ]
        if len(hints) != 1:
            continue
        hint = hints[0]
        direction = next(
            (
                token
                for token in ("lower", "upper")
                if f"{token}_" in semantic_text or f"_{token}" in semantic_text
            ),
            None,
        )
        candidates = [
            index
            for index, layer in enumerate(result)
            if (
                layer.fill.type == "lattice"
                and hint in layer.name.lower()
                and (direction is None or direction in layer.name.lower())
            )
        ]
        if not candidates:
            continue
        for index in candidates:
            layer = result[index]
            loading_ids = list(layer.loading_ids)
            if layer.loading_id and layer.loading_id not in loading_ids:
                loading_ids.insert(0, layer.loading_id)
            loading_ids.append(loading.id)
            result[index] = layer.model_copy(update={
                "loading_ids": list(dict.fromkeys(loading_ids)),
            })
        attached.add(loading.id)
        issues.append(PlanAssemblyIssue(
            code="assembly.component_profile_loading_auto_attached",
            severity="info",
            message=(
                f"attached component-profile loading {loading.id!r} to "
                f"{len(candidates)} {hint!r} lattice layer(s)"
            ),
            path=f"axial_layers.lattice_loadings[{loading.id}]",
        ))
    return result, issues


def _layer_fill_ref(
    item: AxialLayerPatchItem,
    lattice_id: str,
    material_ids: set[str],
) -> tuple[FillRefSpec, tuple[str, str] | None]:
    if item.fill_type == "lattice":
        return FillRefSpec(type="lattice", id=lattice_id), None
    if item.fill_type == "void":
        return FillRefSpec(type="void", id=None), None
    if item.fill_type == "material" and item.fill_id:
        resolution = resolve_material_id(item.fill_id, material_ids)
        if resolution.ok and resolution.resolved_id:
            alias = (
                (item.fill_id, resolution.resolved_id)
                if resolution.resolved_id != item.fill_id else None
            )
            return FillRefSpec(type="material", id=resolution.resolved_id), alias
        return FillRefSpec(type="material", id=item.fill_id), None
    if item.fill_type == "universe" and item.fill_id:
        return FillRefSpec(type="universe", id=item.fill_id), None
    return FillRefSpec(type="void", id=None), None


def _assemble_axial_overlays(
    patch: AxialOverlaysPatch,
    lattice_id: str,
    material_ids: set[str],
    material_aliases: dict[str, str] | None = None,
) -> tuple[list[AxialOverlaySpec], list[PlanAssemblyIssue], dict[str, str]]:
    issues: list[PlanAssemblyIssue] = []
    overlays: list[AxialOverlaySpec] = []
    aliases_applied: dict[str, str] = {}

    for item in patch.overlays:
        material_id = item.material_id
        if material_id is not None:
            resolution = resolve_material_id(material_id, material_ids, material_aliases)
            if resolution.ok and resolution.resolved_id is not None:
                if resolution.resolved_id != material_id:
                    aliases_applied[material_id] = resolution.resolved_id
                    issues.append(PlanAssemblyIssue(
                        code="assembly.material_alias_resolved",
                        severity="info",
                        message=(
                            f"overlay {item.overlay_id!r} material_id "
                            f"{material_id!r} resolved to {resolution.resolved_id!r}"
                        ),
                        path=f"axial_overlays[{item.overlay_id}].material_id",
                        expected=resolution.resolved_id,
                        actual=material_id,
                    ))
                material_id = resolution.resolved_id
            elif item.geometry_mode != "skeleton":
                issues.append(PlanAssemblyIssue(
                    code="assembly.unresolved_material_reference",
                    severity="error",
                    message=resolution.reason or (
                        f"overlay {item.overlay_id!r} references missing "
                        f"material_id {material_id!r}"
                    ),
                    path=f"axial_overlays[{item.overlay_id}].material_id",
                    actual=material_id,
                ))
        try:
            overlay = AxialOverlaySpec(
                id=item.overlay_id,
                overlay_kind=item.overlay_kind,
                z_min_cm=item.z_min_cm,
                z_max_cm=item.z_max_cm,
                target_lattice_id=item.target_lattice_id or lattice_id,
                material_id=material_id,
                geometry_mode=item.geometry_mode,
                through_path_preserved=item.through_path_preserved,
                volume_fraction=item.volume_fraction,
                effective_density_g_cm3=item.effective_density_g_cm3,
                total_mass_g=item.total_mass_g,
                cell_count=item.cell_count,
                pitch_cm=item.pitch_cm,
                material_density_source=item.material_density_source,
                frame_area_cm2=item.frame_area_cm2,
                frame_thickness_cm=item.frame_thickness_cm,
                mass_tolerance_rel=item.mass_tolerance_rel,
                requires_human_confirmation=item.requires_human_confirmation,
                assumptions=list(item.assumptions),
                source_note=item.source_note,
            )
            overlays.append(overlay)
        except Exception as exc:
            issues.append(PlanAssemblyIssue(
                code="assembly.simulation_plan_schema_invalid",
                severity="error",
                message=f"overlay {item.overlay_id!r} failed schema: {exc}",
                path=f"axial_overlays[{item.overlay_id}]",
            ))

    return overlays, issues, aliases_applied


# ---------------------------------------------------------------------------
# Required patch checking
# ---------------------------------------------------------------------------

_REQUIRED_3D_SINGLE: tuple[str, ...] = (
    "facts", "materials", "universes", "pin_map", "axial_layers", "settings",
)

_REQUIRED_3D_MULTI: tuple[str, ...] = (
    "facts", "materials", "universes", "assembly_catalog", "axial_layers",
    "core_layout", "settings",
)


def _check_required_patches(
    indexed: dict[str, Any],
    facts: FactsPatch | None,
    resolved_planning_scope: Any | None = None,
) -> list[PlanAssemblyIssue]:
    issues: list[PlanAssemblyIssue] = []

    is_multi = False
    if facts is not None:
        scope = getattr(resolved_planning_scope, "value", None) or getattr(facts, "model_scope", "single_assembly")
        has_catalog = "assembly_catalog" in indexed
        has_layout = "core_layout" in indexed
        has_pin_map = "pin_map" in indexed
        if scope in ("single_pin", "single_assembly") and (has_catalog or has_layout):
            return [PlanAssemblyIssue(code="assembly.model_scope_patch_family_conflict", severity="error", message="single-assembly Facts conflicts with assembly_catalog/core_layout patch family; regenerate Facts", path="model_scope")]
        if scope in ("multi_assembly_core", "full_core") and has_pin_map and not (has_catalog and has_layout):
            return [PlanAssemblyIssue(code="assembly.model_scope_patch_family_conflict", severity="error", message="multi-assembly Facts conflicts with top-level pin_map patch family", path="model_scope")]
        if scope in ("multi_assembly_core", "full_core"):
            is_multi = True
        elif facts.assembly_count is not None and facts.assembly_count > 1:
            is_multi = True

    required = _REQUIRED_3D_MULTI if is_multi else _REQUIRED_3D_SINGLE
    for ptype in required:
        if ptype not in indexed:
            hint = ""
            if ptype == "pin_map":
                hint = (
                    "; possible_cause: feature detection did not see structural "
                    "signals (guide tubes / large lattice / benchmark variant) — "
                    "check that the requirement includes the input file content"
                )
            issues.append(PlanAssemblyIssue(
                code="assembly.missing_patch",
                severity="error",
                message=f"required {ptype} patch is missing{hint}",
                path=ptype,
            ))
    if facts and facts.has_spacer_grids and "axial_overlays" not in indexed:
        issues.append(PlanAssemblyIssue(
            code="assembly.missing_patch",
            severity="error",
            message="required axial_overlays patch is missing (spacer grids expected)",
            path="axial_overlays",
        ))
    return issues


# ---------------------------------------------------------------------------
# Main assembler API
# ---------------------------------------------------------------------------


def assemble_simulation_plan_from_patches(
    patches: list[Any],
    *,
    strict: bool = True,
    material_policy: str | MaterialCompositionPolicy | None = None,
    resolved_planning_scope: Any | None = None,
) -> PlanAssemblyResult:
    """Assemble validated patches into a complete ``SimulationPlan``.

    Parameters
    ----------
    patches
        List of parsed patch models (from ``parse_patch_content``).
    strict
        When True, missing required patches or unresolved references produce
        errors that block assembly.
    material_policy
        Optional material composition policy. Accepts the enum, a string value,
        or None (uses :data:`DEFAULT_MATERIAL_POLICY`).

    Returns
    -------
    PlanAssemblyResult
        ``ok=True`` with a valid ``SimulationPlan`` on success;
        ``ok=False`` with structured issues on failure. The
        ``material_composition_report`` field is populated on success.
    """
    policy = policy_from_value(material_policy)
    issues: list[PlanAssemblyIssue] = []
    indexed = _extract_patches(patches)
    actual_pin_counts: dict[str, int] = {}
    material_aliases_applied: dict[str, str] = {}
    material_composition_report: MaterialCompositionReport | None = None
    material_species_resolution_report: dict[str, Any] | None = None

    facts: FactsPatch | None = indexed.get("facts")
    materials_patch: MaterialsPatch | None = indexed.get("materials")
    universes_patch: UniversesPatch | None = indexed.get("universes")
    pin_map_patch: PinMapPatch | None = indexed.get("pin_map")
    axial_layers_patch: AxialLayersPatch | None = indexed.get("axial_layers")
    axial_overlays_patch: AxialOverlaysPatch | None = indexed.get("axial_overlays")
    settings_patch: SettingsPatch | None = indexed.get("settings")
    assembly_catalog_patch: AssemblyCatalogPatch | None = indexed.get("assembly_catalog")
    core_layout_patch: CoreLayoutPatch | None = indexed.get("core_layout")
    profiles_patch = indexed.get("localized_insert_profiles")
    base_path_profiles_patch = indexed.get("base_path_axial_profiles")

    # Detect multi-assembly path.
    is_multi_assembly = False
    if assembly_catalog_patch is not None and core_layout_patch is not None:
        is_multi_assembly = True
    elif facts is not None:
        scope = getattr(facts, "model_scope", "single_assembly")
        if scope in ("multi_assembly_core", "full_core"):
            is_multi_assembly = True

    # 1. Check required patches.
    missing_issues = _check_required_patches(indexed, facts, resolved_planning_scope)
    issues.extend(missing_issues)
    if missing_issues and strict:
        return PlanAssemblyResult(
            ok=False,
            issues=issues,
            summary={"error_count": len(missing_issues)},
        )

    # 2. Assemble materials.
    if materials_patch is None:
        materials_patch = MaterialsPatch(materials=[])
    plan_materials, mat_issues, material_composition_report, material_species_resolution_report = _assemble_materials(
        materials_patch, policy=policy,
    )
    issues.extend(mat_issues)
    material_ids = {m.id for m in plan_materials}

    # 3. Assemble universes + cells.
    if universes_patch is None:
        universes_patch = UniversesPatch(universes=[])
    plan_universes, plan_cells, pin_surfaces, pin_regions, univ_issues = _assemble_universes(
        universes_patch,
        material_ids,
        _outer_moderator_material_id(materials_patch),
    )
    issues.extend(univ_issues)
    universe_ids = {u.id for u in plan_universes}

    # =====================================================================
    # MULTI-ASSEMBLY CORE PATH (P2-FULLCORE-2A)
    # =====================================================================
    if is_multi_assembly and assembly_catalog_patch is not None and core_layout_patch is not None:
        from openmc_agent.plan_builder.hierarchical_assembler import (
            build_hierarchical_core_plan,
            compile_global_axial_segments,
        )
        from openmc_agent.plan_builder.localized_insert_profiles import (
            resolve_all_profiles_for_catalog,
        )
        from openmc_agent.plan_builder.axial_state_materializer import (
            materialize_concrete_axial_states,
        )

        pitch = (facts.pin_pitch_cm if facts and facts.pin_pitch_cm else 1.26)
        coolant_material = _outer_moderator_material_id(materials_patch) or "water"

        kind_to_universe_map: dict[str, str] = {}
        if universes_patch is not None:
            for univ in universes_patch.universes:
                kind_to_universe_map[univ.kind] = univ.universe_id

        hier = build_hierarchical_core_plan(
            assembly_catalog_patch,
            core_layout_patch,
            facts,
            pitch_cm=pitch,
            kind_to_universe=kind_to_universe_map or None,
            moderator_universe_id="moderator_outer",
            coolant_material_id=coolant_material,
        )

        for rpt in hier.localized_insert_reports:
            if isinstance(rpt, dict) and rpt.get("assembly_type_id"):
                pass

        # Resolve profiles if available
        resolved_profiles = None
        if profiles_patch is not None:
            resolved_profiles = resolve_all_profiles_for_catalog(
                assembly_catalog_patch, profiles_patch,
            )

        # Compile global axial segments
        has_simple_inserts = any(
            intent.axial_profile_id is None
            for atype in assembly_catalog_patch.assembly_types
            for intent in atype.pin_map.localized_insert_intents
        )
        has_profile_inserts = resolved_profiles is not None and len(resolved_profiles) > 0

        # P2-FULLCORE-2D-A: Pass actual base axial layers for fill-mode classification
        base_axial_layers_list = None
        if axial_layers_patch is not None:
            base_axial_layers_list = list(axial_layers_patch.layers)

        # Always compile segments if we have base axial layers (even without inserts)
        # P2-FULLCORE-2D-A-GRID-CLOSURE: Add spacer grid z-boundaries as breakpoints
        _spacer_grid_z = None
        if axial_overlays_patch is not None:
            _spacer_grid_z = [
                (ov.z_min_cm, ov.z_max_cm)
                for ov in axial_overlays_patch.overlays
                if ov.overlay_kind == "spacer_grid"
                and ov.z_min_cm is not None and ov.z_max_cm is not None
            ]

        # so whole-plane segments get properly classified.
        if has_simple_inserts or has_profile_inserts or base_axial_layers_list or _spacer_grid_z:
            global_segments = compile_global_axial_segments(
                facts,
                assembly_catalog_patch,
                base_axial_layers=base_axial_layers_list,
                resolved_profiles=resolved_profiles,
                spacer_grid_z=_spacer_grid_z,
            )
        else:
            global_segments = []

        # Materialize concrete per-segment geometry
        concrete_result = None
        if global_segments:
            base_pin_lattice_map = {l.id.replace("assembly_lattice__", ""): l for l in hier.pin_lattices}
            # Build proper type_id → lattice mapping
            base_pin_lattice_by_type: dict[str, LatticeSpec] = {}
            for atype in assembly_catalog_patch.assembly_types:
                lat_id = f"assembly_lattice__{atype.assembly_type_id}"
                lat = next((l for l in hier.pin_lattices if l.id == lat_id), None)
                if lat is not None:
                    base_pin_lattice_by_type[atype.assembly_type_id] = lat

            _known_mat_ids = set(material_ids) if material_ids else None
            _known_uv_ids = {u.id for u in (list(plan_universes) + list(hier.assembly_universes))} or None

            # Build grid overlay list and density lookup
            _grid_overlays = None
            _grid_density_lookup = None
            if axial_overlays_patch is not None:
                _grid_overlays = [
                    ov for ov in axial_overlays_patch.overlays
                    if ov.overlay_kind == "spacer_grid"
                    and ov.geometry_mode != "skeleton"
                    and ov.material_id
                ]
                # Build density lookup from material catalog
                _grid_density_lookup = {}
                for mat in plan_materials:
                    if mat.density_value:
                        _grid_density_lookup[mat.id] = mat.density_value

            # Build base path profiles lookup
            _base_path_profiles_map = None
            if base_path_profiles_patch is not None:
                _base_path_profiles_map = {
                    p.profile_id: p for p in base_path_profiles_patch.profiles
                }

            # Build universe patches lookup for grid decoration
            _universe_patches_by_id = None
            if universes_patch is not None:
                _universe_patches_by_id = {
                    u.universe_id: u for u in universes_patch.universes
                }

            concrete_result = materialize_concrete_axial_states(
                assembly_catalog_patch,
                core_layout_patch,
                global_segments,
                base_pin_lattice_by_type,
                hier.assembly_universe_ids,
                kind_to_universe=kind_to_universe_map or None,
                resolved_profiles=resolved_profiles,
                pitch_cm=pitch,
                moderator_universe_id="moderator_outer",
                core_lattice_id_base="core_lattice",
                known_material_ids=_known_mat_ids,
                known_universe_ids=_known_uv_ids,
                grid_overlays=_grid_overlays,
                grid_density_lookup=_grid_density_lookup,
                base_path_profiles=_base_path_profiles_map,
                universe_patches_by_id=_universe_patches_by_id,
            )

            # Propagate materialization issues (fail-closed)
            for mi in concrete_result.issues:
                issues.append(PlanAssemblyIssue(
                    code=mi.code,
                    severity=mi.severity,
                    message=mi.message,
                ))

        all_lattices = list(hier.pin_lattices) + list(hier.core_lattices)
        all_universes = list(plan_universes) + list(hier.assembly_universes)
        all_cells = list(plan_cells) + list(hier.assembly_wrapper_cells)
        all_assemblies = list(hier.assembly_specs)

        # Merge concrete segment-specific geometry
        if concrete_result is not None:
            all_lattices = (
                all_lattices
                + concrete_result.derived_pin_lattices
                + concrete_result.segment_core_lattices
            )
            all_universes = list(all_universes) + list(concrete_result.derived_wrapper_universes)
            all_cells = list(all_cells) + list(concrete_result.derived_wrapper_cells)

            # P2-FULLCORE-2D-A-GRID-CLOSURE: Process grid-decorated universe patches
            if concrete_result.grid_decorated_universe_patches:
                grid_uvs_patch = UniversesPatch(
                    universes=concrete_result.grid_decorated_universe_patches,
                )
                grid_uvs, grid_cells_ir, grid_surfs, grid_regs, grid_issues = _assemble_universes(
                    grid_uvs_patch, material_ids, coolant_material,
                )
                issues.extend(grid_issues)
                all_universes.extend(grid_uvs)
                all_cells.extend(grid_cells_ir)
                pin_surfaces.extend(grid_surfs)
                pin_regions.extend(grid_regs)

        for uv_id in hier.assembly_universe_ids.values():
            if uv_id not in {u.id for u in all_universes}:
                issues.append(PlanAssemblyIssue(
                    code="fullcore.assembly_universe_missing",
                    severity="error",
                    message=f"Assembly wrapper universe {uv_id!r} not in universe catalog",
                ))

        # Determine axial layers: use concrete layers if available, else patch-based
        lattice_loadings = []
        if concrete_result is not None and concrete_result.axial_layers:
            axial_layers = concrete_result.axial_layers
        elif axial_layers_patch is not None:
            axial_layers, lattice_loadings, al_issues, axial_layer_aliases = _assemble_axial_layers(
                axial_layers_patch, "core_lattice", material_ids,
            )
            issues.extend(al_issues)
        else:
            axial_layers = []

        if axial_overlays_patch is not None:
            axial_overlays, ao_issues, overlay_aliases = _assemble_axial_overlays(
                axial_overlays_patch, "core_lattice", material_ids,
            )
            issues.extend(ao_issues)
        else:
            axial_overlays = []

        core_spec = hier.core_spec
        if core_spec is not None:
            core_spec = core_spec.model_copy(update={
                "axial_layers": axial_layers,
                "axial_overlays": axial_overlays,
            })

        all_assumptions: list[str] = []
        all_confirms: list[str] = []
        for mat in plan_materials:
            all_assumptions.extend(mat.assumptions)
            all_confirms.extend(mat.requires_human_confirmation)
        if facts and facts.assumptions:
            all_assumptions.extend(facts.assumptions)
        if facts and facts.missing_facts:
            all_confirms.extend(f"missing fact: {f}" for f in facts.missing_facts)

        assembly_pitch = (
            facts.assembly_pitch_cm if facts and facts.assembly_pitch_cm
            else (core_layout_patch.assembly_pitch_cm or 21.50)
        )

        complex_model = ComplexModelSpec(
            name="hierarchical core model",
            kind="core",
            materials=plan_materials,
            cells=all_cells,
            surfaces=pin_surfaces,
            regions=pin_regions,
            universes=all_universes,
            lattices=all_lattices,
            lattice_loadings=lattice_loadings,
            assemblies=all_assemblies,
            core=core_spec,
            settings=RunSettingsSpec(
                source_strategy=(
                    settings_patch.source_strategy if settings_patch else "active_fuel_box"
                ),
                source_requires_fissionable_constraint=(
                    settings_patch.source_requires_fissionable_constraint
                    if settings_patch else True
                ),
                manual_source_bounds_cm=(
                    settings_patch.manual_source_bounds_cm
                    if settings_patch and hasattr(settings_patch, "manual_source_bounds_cm")
                    else None
                ),
            ),
            assumptions=list(dict.fromkeys(all_assumptions)),
            requires_human_confirmation=list(dict.fromkeys(all_confirms)),
        )

        core_cols = core_layout_patch.shape[1] if core_layout_patch else 3
        core_width = assembly_pitch * core_cols
        plot_specs = _derive_multi_assembly_plots(
            axial_layers,
            axial_overlays,
            core_width,
            assembly_pitch_cm=assembly_pitch,
            axial_segment_index=concrete_result.segment_index if concrete_result is not None else None,
        )

        run_settings = RunSettingsSpec(
            batches=5, inactive=1, particles=100,
            source_strategy=(
                settings_patch.source_strategy if settings_patch else "active_fuel_box"
            ),
        )
        execution_check = ExecutionCheckSpec(
            enabled=True,
            settings=run_settings,
        )

        capability = RenderCapabilityReport(
            renderability="none",
            is_executable=False,
            supported_renderer="none",
            reasons=["assembled from multi-assembly patches; awaiting capability assessment"],
        )

        plan = SimulationPlan(
            schema_version="simulation_plan.v2",
            complex_model=complex_model,
            capability_report=capability,
            plot_specs=plot_specs,
            execution_check=execution_check,
            expert_assumptions=list(dict.fromkeys(all_assumptions)),
        )

        # P2-FULLCORE-2D-A-GRID-ACCEPTANCE-CLOSURE: Strict grid geometry gate.
        # When physical spacer_grid overlays exist, verify the full chain
        # (overlay → decorated universe → lattice → frame cell → material).
        grid_validation = validate_grid_geometry_materialization(plan)
        if not grid_validation.ok:
            for gi in grid_validation.issues:
                issues.append(PlanAssemblyIssue(
                    code=gi.code,
                    severity=gi.severity,
                    message=gi.message,
                ))

        # P2-FULLCORE-2C-C: Source-driven localized insert placement gate.
        # Verify that all source-declared placement requirements are fulfilled
        # by matching intents, profiles, and universes.
        from openmc_agent.plan_builder.required_placement_validator import (
            validate_required_localized_insert_placements,
        )
        placement_validation = validate_required_localized_insert_placements(
            facts,
            universes_patch,
            profiles_patch,
            assembly_catalog_patch,
            core_layout_patch,
            axial_domain_cm=(
                facts.axial_domain_cm
                if facts and facts.axial_domain_cm
                else None
            ),
        )
        if not placement_validation.ok:
            for pi in placement_validation.issues:
                issues.append(PlanAssemblyIssue(
                    code=pi.code,
                    severity=pi.severity,
                    message=pi.message,
                ))

        # P2-FULLCORE-2C-C: Final placement reachability report.
        from openmc_agent.plan_builder.placement_reachability import (
            build_localized_insert_placement_report,
        )
        placement_report = build_localized_insert_placement_report(
            facts,
            universes_patch,
            profiles_patch,
            assembly_catalog_patch,
            core_layout_patch,
            complex_model=complex_model,
            axial_domain_cm=(
                facts.axial_domain_cm
                if facts and facts.axial_domain_cm
                else None
            ),
        )
        if placement_report.overall_result == "fail":
            for pri in placement_report.issues:
                issues.append(PlanAssemblyIssue(
                    code=pri.get("code", "localized_insert.required_segment_unreachable"),
                    severity="error",
                    message=pri.get("message", ""),
                ))

        errors = [i for i in issues if i.severity == "error"]
        warnings = [i for i in issues if i.severity == "warning"]
        infos = [i for i in issues if i.severity == "info"]

        return PlanAssemblyResult(
            ok=len(errors) == 0,
            plan=plan,
            plan_dict=plan.model_dump(),
            issues=issues,
            material_composition_report=material_composition_report,
            material_species_resolution_report=material_species_resolution_report,
            summary={
                "path": "multi_assembly_core",
                "kind": "core",
                "error_count": len(errors),
                "warning_count": len(warnings),
                "info_count": len(infos),
                "material_count": len(plan_materials),
                "universe_count": len(all_universes),
                "cell_count": len(all_cells),
                "lattice_count": len(all_lattices),
                "assembly_type_count": len(all_assemblies),
                "core_total_fuel": hier.core_count_aggregation.core_total_for_role("fuel_pin") if hier.core_count_aggregation else 0,
                "core_total_guide_tube": hier.core_count_aggregation.core_total_for_role("guide_tube") if hier.core_count_aggregation else 0,
                "localized_inserts_in_base_lattice": False,
                "internal_assembly_boundary": "transmission",
                "localized_insert_placement_report": placement_report.to_dict(),
            },
        )

    # =====================================================================
    # SINGLE-ASSEMBLY PATH (existing VERA3 path)
    # =====================================================================

    # 3a. Derive localized insert loadings (replaces old _normalize_axial_insert_pin_map).
    if pin_map_patch is not None:
        from openmc_agent.plan_builder.localized_insert_derivation import (
            derive_localized_insert_loadings,
        )
        pin_map_patch, axial_layers_patch, insert_issues_dict, _insert_report = (
            derive_localized_insert_loadings(
                pin_map_patch, axial_layers_patch, universes_patch,
            )
        )
        for ii in insert_issues_dict:
            issues.append(PlanAssemblyIssue(
                code=ii["code"],
                severity=ii.get("severity", "warning"),
                message=ii.get("message", ""),
            ))

    # 4. Assemble lattice from pin map.
    lattice_id = "assembly_lattice"
    if pin_map_patch is not None:
        lattice, lat_issues, actual_pin_counts = _assemble_lattice(
            pin_map_patch, facts, universes_patch, universe_ids,
        )
        issues.extend(lat_issues)
    else:
        lattice = None

    # 5. Assemble axial layers.
    if axial_layers_patch is not None:
        axial_layers, lattice_loadings, al_issues, axial_layer_aliases = _assemble_axial_layers(
            axial_layers_patch, lattice_id, material_ids,
        )
        issues.extend(al_issues)
        material_aliases_applied.update(axial_layer_aliases)
        axial_layers, loading_attachment_issues = _auto_attach_component_profile_loadings(
            axial_layers, lattice_loadings,
        )
        issues.extend(loading_attachment_issues)
    else:
        axial_layers = []
        lattice_loadings = []

    # 6. Assemble axial overlays.
    if axial_overlays_patch is not None:
        axial_overlays, ao_issues, overlay_aliases = _assemble_axial_overlays(
            axial_overlays_patch, lattice_id, material_ids,
        )
        issues.extend(ao_issues)
        material_aliases_applied.update(overlay_aliases)
    else:
        axial_overlays = []

    # 7. Build assembly spec.
    assembly_pitch = (
        facts.assembly_pitch_cm if facts and facts.assembly_pitch_cm
        else None
    )
    assembly = AssemblySpec(
        id="assembly_1",
        name="assembled assembly",
        lattice_id=lattice.id if lattice else None,
        pitch_cm=assembly_pitch,
        boundary="reflective",
        purpose="Assembled from incremental patches",
    )

    # 8. Build core spec.
    core = CoreSpec(
        id="core_1",
        name="assembled core",
        lattice_id=lattice.id if lattice else None,
        assembly_ids=[assembly.id],
        axial_layers=axial_layers,
        axial_overlays=axial_overlays,
        boundary="vacuum",
        purpose="Assembled from incremental patches",
    )

    # 9. Collect all assumptions and confirmations.
    all_assumptions: list[str] = []
    all_confirms: list[str] = []
    for mat in plan_materials:
        all_assumptions.extend(mat.assumptions)
        all_confirms.extend(mat.requires_human_confirmation)
    if facts and facts.assumptions:
        all_assumptions.extend(facts.assumptions)
    if facts and facts.missing_facts:
        all_confirms.extend(f"missing fact: {f}" for f in facts.missing_facts)

    # 10. Build ComplexModelSpec.
    lattices = [lattice] if lattice else []
    complex_model = ComplexModelSpec(
        name=assembly.name,
        kind="assembly",
        materials=plan_materials,
        cells=plan_cells,
        surfaces=pin_surfaces,
        regions=pin_regions,
        universes=plan_universes,
        lattices=lattices,
        lattice_loadings=lattice_loadings,
        assemblies=[assembly],
        core=core,
        settings=RunSettingsSpec(
            source_strategy=(
                settings_patch.source_strategy if settings_patch else "active_fuel_box"
            ),
            source_requires_fissionable_constraint=(
                settings_patch.source_requires_fissionable_constraint
                if settings_patch
                else True
            ),
            manual_source_bounds_cm=(
                settings_patch.manual_source_bounds_cm
                if settings_patch and hasattr(settings_patch, "manual_source_bounds_cm")
                else None
            ),
        ),
        assumptions=list(dict.fromkeys(all_assumptions)),
        requires_human_confirmation=list(dict.fromkeys(all_confirms)),
    )

    # 11. Build plot specs — derive ranges from actual geometry, not hardcoded.
    plot_strategy = (
        settings_patch.plot_strategy if settings_patch else "full_assembly"
    )
    # Compute actual axial domain from assembled layers.
    if axial_layers:
        axial_z_min = min(l.z_min_cm for l in axial_layers)
        axial_z_max = max(l.z_max_cm for l in axial_layers)
    else:
        axial_z_min, axial_z_max = -1.0, 1.0
    axial_height = axial_z_max - axial_z_min
    # XY origin centered on assembly.
    pitch = assembly_pitch or 21.50
    xy_origin = (pitch / 2.0, pitch / 2.0, (axial_z_min + axial_z_max) / 2.0)
    xz_origin = (pitch / 2.0, 0.0, (axial_z_min + axial_z_max) / 2.0)
    if plot_strategy == "full_assembly":
        plot_specs = [
            PlotSpec(
                basis="xy",
                origin=xy_origin,
                width_cm=(pitch, pitch),
                filename="assembly_xy.png",
            ),
            PlotSpec(
                basis="xz",
                origin=xz_origin,
                width_cm=(pitch, axial_height),
                filename="assembly_xz.png",
            ),
        ]
    else:
        plot_specs = [
            PlotSpec(
                basis="xy",
                origin=xy_origin,
                width_cm=(pitch, pitch),
                filename="assembly_xy.png",
            ),
        ]

    # 12. Build execution check.
    source_strategy = (
        settings_patch.source_strategy if settings_patch else "active_fuel_box"
    )
    source_fissile = (
        settings_patch.source_requires_fissionable_constraint
        if settings_patch
        else True
    )
    manual_bounds = (
        settings_patch.manual_source_bounds_cm if settings_patch else None
    ) if hasattr(settings_patch, "manual_source_bounds_cm") else None
    run_settings = RunSettingsSpec(
        batches=5, inactive=1, particles=100,
        source_strategy=source_strategy,
        source_requires_fissionable_constraint=source_fissile,
        manual_source_bounds_cm=manual_bounds,
    )
    execution_check = ExecutionCheckSpec(
        enabled=True,
        settings=run_settings,
    )

    # 13. Build capability report — non-executable skeleton until renderer
    #     assesses the assembled plan.
    capability = RenderCapabilityReport(
        renderability="none",
        is_executable=False,
        supported_renderer="none",
        reasons=["assembled from incremental patches; awaiting capability assessment"],
    )

    # 14. Assemble SimulationPlan.
    try:
        plan = SimulationPlan(
            schema_version="simulation_plan.v2",
            complex_model=complex_model,
            capability_report=capability,
            plot_specs=plot_specs,
            execution_check=execution_check,
            expert_assumptions=list(dict.fromkeys(all_assumptions)),
        )
    except Exception as exc:
        issues.append(PlanAssemblyIssue(
            code="assembly.simulation_plan_schema_invalid",
            severity="error",
            message=f"final SimulationPlan validation failed: {exc}",
        ))
        return PlanAssemblyResult(
            ok=False,
            issues=issues,
            summary={"error_count": sum(1 for i in issues if i.severity == "error")},
        )

    errors = [i for i in issues if i.severity == "error"]

    # P2-FULLCORE-2D-B: fuel variant reachability check on assembled plan.
    if facts is not None and facts.fuel_variant_requirements:
        from openmc_agent.plan_builder.fuel_variant_reachability import (
            build_fuel_variant_reachability_report,
            reachability_report_to_issues,
        )
        mat_sv = {
            m.material_id: m.source_variant_id
            for m in materials_patch.materials
            if m.source_variant_id
        } if materials_patch else {}
        bindings = []
        if assembly_catalog_patch:
            for atype in assembly_catalog_patch.assembly_types:
                bindings.append({
                    "assembly_type_id": atype.assembly_type_id,
                    "default_universe_id": atype.pin_map.default_universe_id,
                    "fuel_variant_id": atype.fuel_variant_id,
                    "resolved_fuel_variant_ids": [atype.fuel_variant_id] if atype.fuel_variant_id else [],
                })
        layout_pattern = []
        if core_layout_patch:
            layout_pattern = core_layout_patch.assembly_pattern
        reach_report = build_fuel_variant_reachability_report(
            assembled_plan=plan.model_dump(mode="json") if plan else None,
            fuel_variant_requirements=[
                v.model_dump(mode="json") for v in facts.fuel_variant_requirements
            ],
            material_source_variants=mat_sv,
            assembly_fuel_bindings=bindings,
            core_layout_pattern=layout_pattern,
        )
        reach_issues = reachability_report_to_issues(reach_report)
        issues.extend(reach_issues)
        errors = [i for i in issues if i.severity == "error"]

    return PlanAssemblyResult(
        ok=len(errors) == 0,
        plan=plan,
        plan_dict=plan.model_dump(mode="json"),
        issues=issues,
        summary={
            "error_count": len(errors),
            "warning_count": sum(1 for i in issues if i.severity == "warning"),
            "info_count": sum(1 for i in issues if i.severity == "info"),
            "material_count": len(plan_materials),
            "universe_count": len(plan_universes),
            "lattice_present": lattice is not None,
            "lattice_loading_count": len(lattice_loadings),
            "lattice_transformation_count": sum(len(l.transformations) for l in lattice_loadings),
            "family_replacement_count": sum(
                1 for l in lattice_loadings
                for t in l.transformations
                if t.operation_kind == "replace_universe_family"
            ),
            "coordinate_override_count": sum(
                1 for l in lattice_loadings
                for t in l.transformations
                if t.operation_kind == "coordinate_override"
            ),
            "nested_component_override_count": sum(
                1 for l in lattice_loadings
                for t in l.transformations
                if t.operation_kind == "nested_component_override"
            ),
            "multi_loading_layer_count": sum(
                1 for layer in axial_layers
                if len(layer.loading_ids) > 1 or (layer.loading_ids and layer.loading_id)
            ),
            "derived_lattice_count": sum(
                1 for l in lattice_loadings if l.derived_lattice_id
            ),
            "axial_layer_count": len(axial_layers),
            "axial_overlay_count": len(axial_overlays),
            "actual_pin_counts": actual_pin_counts,
            "material_aliases_applied": material_aliases_applied,
            "material_composition_policy": policy.value,
            "material_composition_report_present": material_composition_report is not None,
            "material_species_resolution_report_present": material_species_resolution_report is not None,
        },
        material_composition_report=material_composition_report,
        material_species_resolution_report=material_species_resolution_report,
    )


__all__ = [
    "PlanAssemblyIssue",
    "PlanAssemblyResult",
    "expand_pin_map",
    "compute_pin_role_counts",
    "assemble_simulation_plan_from_patches",
]
