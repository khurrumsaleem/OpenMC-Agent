"""Concrete localized-insert state materializer (P2-FULLCORE-2D-A).

Transforms abstract localized-insert intents + resolved profiles + spacer
grid overlays into concrete per-segment pin lattices, assembly wrapper
universes, core lattices, and CoreSpec.axial_layers.

Pipeline per segment:
    fill_mode == whole_plane_material → material slab (no lattices)
    fill_mode == whole_plane_universe → universe slab (no lattices)
    fill_mode == void                 → void slab
    fill_mode == detailed_core:
        base pin lattice
        → localized insert overrides
        → spacer-grid coolant modification
        → assembly wrapper universe
        → segment-specific core lattice (deduplicated by content hash)
        → AxialLayerSpec

P2-FULLCORE-2D-A additions:
    * MaterializationIssue typed issues (replaces dict)
    * Canonical content-hash IDs (replaces seg_counter)
    * Core-state reuse (identical core patterns share one lattice)
    * Spacer-grid assembly-instance materialization
    * Fail-closed validation
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any

from openmc_agent.plan_builder.hierarchical_assembler import AxialSegment
from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    AxialOverlayPatchItem,
    BasePathAxialProfilePatchItem,
    BasePathAxialProfilesPatch,
    BasePathStateBindingPatchItem,
    CoordinateConvention,
    CoreLayoutPatch,
    LocalizedInsertIntentPatchItem,
    UniverseSpecPatch,
    CellLayerPatch,
)
from openmc_agent.plan_builder.localized_insert_profiles import (
    ResolvedLocalizedInsertProfile,
)
from openmc_agent.schemas import (
    AxialLayerSpec,
    CellSpec,
    CoreSpec,
    FillRefSpec,
    LatticeSpec,
    UniverseSpec,
)


_Z_TOL: float = 1e-6
_MATERIALIZATION_CONTRACT_VERSION: str = "2.1.0"


# ---------------------------------------------------------------------------
# Typed issue (Commit 9)
# ---------------------------------------------------------------------------


@dataclass
class MaterializationIssue:
    """Structured materialization issue (fail-closed)."""

    code: str
    severity: str  # "error" | "warning"
    message: str
    segment_id: str | None = None
    assembly_type_id: str | None = None


# ---------------------------------------------------------------------------
# Grid state types (Commit 6)
# ---------------------------------------------------------------------------


@dataclass
class GridFrameDerivationReport:
    """Mass-derived spacer grid frame parameters."""

    overlay_id: str
    material_id: str
    grid_height_cm: float
    total_mass_g: float
    density_g_cm3: float
    cell_count: int
    pitch_cm: float
    area_per_cell_cm2: float
    frame_thickness_cm: float
    volume_fraction: float


@dataclass
class AssemblyGridState:
    """Spacer grid overlay state for a segment."""

    active_overlay_ids: list[str] = field(default_factory=list)
    active_material_ids: list[str] = field(default_factory=list)
    derivation_reports: list[GridFrameDerivationReport] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ConcreteAxialStateResult:
    """Result of concrete axial-state materialization."""

    derived_pin_lattices: list[LatticeSpec] = field(default_factory=list)
    derived_wrapper_universes: list[UniverseSpec] = field(default_factory=list)
    derived_wrapper_cells: list[CellSpec] = field(default_factory=list)
    segment_core_lattices: list[LatticeSpec] = field(default_factory=list)
    axial_layers: list[AxialLayerSpec] = field(default_factory=list)
    segment_index: list[dict[str, Any]] = field(default_factory=list)
    issues: list[MaterializationIssue] = field(default_factory=list)
    grid_states: dict[str, AssemblyGridState] = field(default_factory=dict)
    state_reuse_report: dict[str, Any] = field(default_factory=dict)
    grid_decorated_universe_patches: list[Any] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == "error" for i in self.issues)


# ---------------------------------------------------------------------------
# Coordinate normalization
# ---------------------------------------------------------------------------


def _normalize_insert_coords(
    coords: list[tuple[int, int]],
    pm: AssemblyPinMapPatchItem,
) -> list[tuple[int, int]]:
    """Normalize insert coordinates to 0-based IR using the pin map convention."""
    base = pm.coordinate_convention.index_base
    if base == 0:
        return list(coords)
    return [(r - base, c - base) for r, c in coords]


def _apply_insert_overrides(
    base_pattern: list[list[str]],
    intent: LocalizedInsertIntentPatchItem,
    *,
    pm: AssemblyPinMapPatchItem | None = None,
) -> list[list[str]]:
    """Return a copy of base_pattern with insert overrides applied."""
    result = [row[:] for row in base_pattern]
    coords = (
        _normalize_insert_coords(intent.coordinates, pm)
        if pm is not None
        else intent.coordinates
    )
    for r, c in coords:
        if 0 <= r < len(result) and 0 <= c < len(result[r]):
            result[r][c] = intent.insert_universe_id
    return result


# ---------------------------------------------------------------------------
# Canonical content hashing (Commit 8)
# ---------------------------------------------------------------------------


def _compute_pin_state_hash(
    type_id: str,
    derived_pattern: list[list[str]],
    active_insert_ids: list[str],
    coord_index_base: int,
    grid_overlay_ids: list[str] | None = None,
    base_path_replacement_ids: list[str] | None = None,
    base_role: str = "",
) -> str:
    """Compute a stable SHA-256 content hash for a pin lattice state."""
    h = hashlib.sha256()
    h.update(_MATERIALIZATION_CONTRACT_VERSION.encode())
    h.update(f"|type={type_id}".encode())
    for row in derived_pattern:
        h.update(b"|")
        h.update("|".join(row).encode())
    for iid in sorted(active_insert_ids):
        h.update(f"|ins={iid}".encode())
    h.update(f"|ibase={coord_index_base}".encode())
    if grid_overlay_ids:
        for gid in sorted(grid_overlay_ids):
            h.update(f"|grid={gid}".encode())
    if base_path_replacement_ids:
        for rid in sorted(base_path_replacement_ids):
            h.update(f"|bpath={rid}".encode())
    if base_role:
        h.update(f"|role={base_role}".encode())
    return h.hexdigest()[:16]


def _compute_core_state_hash(
    core_pattern: list[list[str]],
    layout_id: str,
) -> str:
    """Compute a stable content hash for a core lattice pattern."""
    h = hashlib.sha256()
    h.update(_MATERIALIZATION_CONTRACT_VERSION.encode())
    h.update(f"|layout={layout_id}".encode())
    for row in core_pattern:
        h.update(b"|")
        h.update("|".join(row).encode())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Spacer grid frame derivation (Commit 6)
# ---------------------------------------------------------------------------


def _compute_grid_frame_derivation(
    overlay: AxialOverlayPatchItem,
    density_g_cm3: float,
    pitch_cm: float,
) -> GridFrameDerivationReport:
    """Derive grid frame parameters from mass conservation.

    A_cell = grid_mass / (density × cell_count × grid_height)
    frame_thickness ≈ A_cell / (4 × pitch)  (frame on 4 sides per cell)
    volume_fraction = A_cell / pitch²
    """
    grid_mass = overlay.total_mass_g or 0.0
    cell_count = overlay.cell_count or 289
    grid_height = (overlay.z_max_cm or 0.0) - (overlay.z_min_cm or 0.0)

    if density_g_cm3 <= 0 or grid_height <= 0:
        area_per_cell = 0.0
        frame_thickness = 0.0
        vf = 0.0
    else:
        area_per_cell = grid_mass / (density_g_cm3 * cell_count * grid_height)
        frame_thickness = area_per_cell / (4.0 * pitch_cm) if pitch_cm > 0 else 0.0
        vf = area_per_cell / (pitch_cm * pitch_cm) if pitch_cm > 0 else 0.0

    return GridFrameDerivationReport(
        overlay_id=overlay.overlay_id,
        material_id=overlay.material_id or "",
        grid_height_cm=grid_height,
        total_mass_g=grid_mass,
        density_g_cm3=density_g_cm3,
        cell_count=cell_count,
        pitch_cm=pitch_cm,
        area_per_cell_cm2=area_per_cell,
        frame_thickness_cm=frame_thickness,
        volume_fraction=vf,
    )


def _get_active_grids_for_segment(
    segment: AxialSegment,
    grid_overlays: list[AxialOverlayPatchItem] | None,
) -> list[AxialOverlayPatchItem]:
    """Return grid overlays active in the given segment's z range."""
    if not grid_overlays:
        return []
    active: list[AxialOverlayPatchItem] = []
    for ov in grid_overlays:
        if ov.geometry_mode == "skeleton" or not ov.material_id:
            continue
        ov_zmin = ov.z_min_cm if ov.z_min_cm is not None else float("-inf")
        ov_zmax = ov.z_max_cm if ov.z_max_cm is not None else float("inf")
        if (
            segment.z_min_cm >= ov_zmin - _Z_TOL
            and segment.z_max_cm <= ov_zmax + _Z_TOL
        ):
            active.append(ov)
    return active


# ---------------------------------------------------------------------------
# Active inserts for segment
# ---------------------------------------------------------------------------


def _get_active_inserts_for_segment(
    segment: AxialSegment,
    catalog: AssemblyCatalogPatch,
    resolved_profiles: list[ResolvedLocalizedInsertProfile] | None = None,
) -> dict[str, list[tuple[str, str, list[tuple[int, int]], AssemblyPinMapPatchItem]]]:
    """For a given segment, return active inserts per assembly type."""
    result: dict[str, list[tuple[str, str, list[tuple[int, int]], AssemblyPinMapPatchItem]]] = {}

    for atype in catalog.assembly_types:
        type_id = atype.assembly_type_id
        pm = atype.pin_map
        active: list[tuple[str, str, list[tuple[int, int]], AssemblyPinMapPatchItem]] = []

        for intent in pm.localized_insert_intents:
            norm_coords = _normalize_insert_coords(intent.coordinates, pm)

            if intent.axial_profile_id is not None:
                if resolved_profiles:
                    for rp in resolved_profiles:
                        if rp.insert_id == intent.insert_id and rp.assembly_type_id == type_id:
                            for rseg in rp.resolved_segments:
                                if (
                                    segment.z_min_cm >= rseg.absolute_z_min_cm - _Z_TOL
                                    and segment.z_max_cm <= rseg.absolute_z_max_cm + _Z_TOL
                                ):
                                    active.append((
                                        f"{intent.insert_id}::{rseg.segment_id}",
                                        rseg.universe_id,
                                        norm_coords,
                                        pm,
                                    ))
                                    break
                continue

            z_min = (
                intent.z_min_cm
                if intent.z_min_cm is not None
                else float("-inf")
            )
            z_max = (
                intent.z_max_cm
                if intent.z_max_cm is not None
                else float("inf")
            )
            if (
                segment.z_min_cm >= z_min - _Z_TOL
                and segment.z_max_cm <= z_max + _Z_TOL
            ):
                active.append((
                    intent.insert_id,
                    intent.insert_universe_id,
                    norm_coords,
                    pm,
                ))

        if active:
            result[type_id] = active

    return result


# ---------------------------------------------------------------------------
# Coordinate validation (fail-closed)
# ---------------------------------------------------------------------------


def _validate_insert_coordinates(
    coords: list[tuple[int, int]],
    lattice_size: tuple[int, int],
    type_id: str,
    segment_id: str,
) -> list[MaterializationIssue]:
    """Validate that insert coordinates are within lattice bounds."""
    issues: list[MaterializationIssue] = []
    nx, ny = lattice_size
    for r, c in coords:
        if r < 0 or r >= nx or c < 0 or c >= ny:
            issues.append(MaterializationIssue(
                code="fullcore.coordinate_out_of_bounds",
                severity="error",
                message=f"coordinate ({r},{c}) outside lattice {lattice_size} for type {type_id}",
                segment_id=segment_id,
                assembly_type_id=type_id,
            ))
    return issues


# ---------------------------------------------------------------------------
# Base path axial-state resolution (Commit 1: fuel-path switching)
# ---------------------------------------------------------------------------


def _resolve_base_path_bindings(
    base_role: str,
    type_id: str,
    base_path_profiles: dict[str, BasePathAxialProfilePatchItem] | None,
    profile_id: str | None,
) -> list[BasePathStateBindingPatchItem]:
    """Find state bindings matching the segment's base_role for this assembly type."""
    if not base_path_profiles or not profile_id:
        return []
    profile = base_path_profiles.get(profile_id)
    if profile is None:
        return []
    matches = [
        b for b in profile.state_bindings
        if b.axial_role == base_role
        and (
            not b.assembly_type_ids
            or type_id in b.assembly_type_ids
        )
    ]
    matches.sort(key=lambda b: b.priority, reverse=True)
    return matches


def _apply_base_path_state(
    pattern: list[list[str]],
    bindings: list[BasePathStateBindingPatchItem],
) -> tuple[list[list[str]], list[str]]:
    """Apply base path state bindings to a pin lattice pattern.

    Returns (modified_pattern, list_of_replacement_ids_applied).
    """
    if not bindings:
        return pattern, []
    modified = [row[:] for row in pattern]
    applied: set[str] = set()
    for binding in bindings:
        for r in range(len(modified)):
            for c in range(len(modified[r])):
                uv = modified[r][c]
                if uv in binding.source_universe_ids:
                    modified[r][c] = binding.replacement_universe_id
                    applied.add(binding.replacement_universe_id)
    return modified, sorted(applied)


# ---------------------------------------------------------------------------
# Grid-decorated universe generation (Commit 2-3: physical grid geometry)
# ---------------------------------------------------------------------------


def _compute_grid_frame_exact(
    grid_mass_g: float,
    density_g_cm3: float,
    cell_count: int,
    grid_height_cm: float,
    pitch_cm: float,
) -> tuple[float, float, float]:
    """Exact square-frame derivation.

    Returns (area_per_cell, inner_side, frame_thickness).

    A_cell = grid_mass / (density × cell_count × grid_height)
    inner_side = sqrt(pitch² - A_cell)
    frame_thickness = (pitch - inner_side) / 2
    """
    if density_g_cm3 <= 0 or grid_height_cm <= 0 or cell_count <= 0:
        return 0.0, pitch_cm, 0.0
    a_cell = grid_mass_g / (density_g_cm3 * cell_count * grid_height_cm)
    val = pitch_cm * pitch_cm - a_cell
    if val <= 0:
        return a_cell, 0.0, pitch_cm / 2.0
    inner_side = math.sqrt(val)
    frame_thickness = (pitch_cm - inner_side) / 2.0
    return a_cell, inner_side, frame_thickness


def _back_calculate_mass(
    inner_side: float,
    pitch_cm: float,
    density_g_cm3: float,
    cell_count: int,
    grid_height_cm: float,
) -> float:
    """Back-calculate mass from geometry to verify conservation."""
    a_cell = pitch_cm * pitch_cm - inner_side * inner_side
    vol = a_cell * cell_count * grid_height_cm
    return vol * density_g_cm3


def _make_grid_decorated_universe(
    base_universe_id: str,
    grid_state_hash: str,
    grid_material_id: str,
    inner_side: float,
    pitch_cm: float,
    base_universe_patch: UniverseSpecPatch | None,
) -> UniverseSpecPatch | None:
    """Create a grid-decorated variant of a base universe.

    Inserts a square_frame cell BEFORE the background cell.  The assembler
    partitions the background into inner moderator (outside cylinder,
    excluding frame area) to prevent overlap.

    Cell order: [original cylinders...] → [square_frame] → [background]

    Returns None if the base universe patch is not available.
    """
    if base_universe_patch is None:
        return None

    decorated_cells: list[CellLayerPatch] = []
    for cell in base_universe_patch.cells:
        if cell.region_kind == "background":
            # Insert grid frame BEFORE the background
            decorated_cells.append(CellLayerPatch(
                id="grid_frame",
                role="grid_frame",
                material_id=grid_material_id,
                region_kind="square_frame",
                outer_side_cm=pitch_cm,
                inner_side_cm=inner_side,
            ))
            # Background becomes bounded inner moderator
            # (assembler excludes frame area from this region)
            decorated_cells.append(CellLayerPatch(
                id=str(cell.id),
                role=str(cell.role),
                material_id=cell.material_id,
                region_kind="background",
            ))
        else:
            decorated_cells.append(cell)

    # If no background was present, add frame + background
    has_bg = any(c.region_kind == "background" for c in base_universe_patch.cells)
    if not has_bg:
        decorated_cells.append(CellLayerPatch(
            id="grid_frame", role="grid_frame",
            material_id=grid_material_id,
            region_kind="square_frame",
            outer_side_cm=pitch_cm, inner_side_cm=inner_side,
        ))
        decorated_cells.append(CellLayerPatch(
            id="grid_bg", role="coolant",
            region_kind="background",
        ))

    decorated_id = f"{base_universe_id}__grid__{grid_state_hash[:12]}"
    return UniverseSpecPatch(
        universe_id=decorated_id,
        kind=base_universe_patch.kind,
        cells=decorated_cells,
        source_note=f"Grid-decorated variant of {base_universe_id}",
    )


# ---------------------------------------------------------------------------
# Main materialization
# ---------------------------------------------------------------------------


def materialize_concrete_axial_states(
    catalog: AssemblyCatalogPatch,
    layout: CoreLayoutPatch,
    segments: list[AxialSegment],
    base_pin_lattices: dict[str, LatticeSpec],
    base_assembly_universe_ids: dict[str, str],
    *,
    kind_to_universe: dict[str, str] | None = None,
    resolved_profiles: list[ResolvedLocalizedInsertProfile] | None = None,
    pitch_cm: float = 1.26,
    moderator_universe_id: str = "moderator_outer",
    outer_universe_id: str | None = None,
    core_lattice_id_base: str = "core_lattice",
    known_material_ids: set[str] | None = None,
    known_universe_ids: set[str] | None = None,
    grid_overlays: list[AxialOverlayPatchItem] | None = None,
    grid_density_lookup: dict[str, float] | None = None,
    base_path_profiles: dict[str, BasePathAxialProfilePatchItem] | None = None,
    universe_patches_by_id: dict[str, UniverseSpecPatch] | None = None,
    fail_closed: bool = True,
) -> ConcreteAxialStateResult:
    """Materialize concrete per-segment geometry.

    For each global axial segment, the fill_mode determines the pipeline:

    * ``whole_plane_material`` → AxialLayerSpec with material fill.
    * ``whole_plane_universe`` → AxialLayerSpec with universe fill.
    * ``void`` → AxialLayerSpec with void fill.
    * ``detailed_core`` → base path state → localized insert overrides →
      grid decoration → assembly wrapper → segment-specific core lattice.

    P2-FULLCORE-2D-A-HARDENING:
    * Base path fuel-state switching per segment role.
    * Grid-decorated universes with physical square frame geometry.
    * Fail-closed: no silent fallbacks when ``fail_closed=True``.
    """
    result = ConcreteAxialStateResult()

    pitch = layout.assembly_pitch_cm or 21.50
    n_rows, n_cols = layout.shape
    core_width_x = n_cols * pitch
    core_width_y = n_rows * pitch
    lower_left = (-core_width_x / 2.0, -core_width_y / 2.0)

    # Fail-closed: base lattice must exist for each type
    for atype in catalog.assembly_types:
        tid = atype.assembly_type_id
        if tid not in base_pin_lattices:
            result.issues.append(MaterializationIssue(
                code="fullcore.base_lattice_missing",
                severity="error",
                message=f"base pin lattice for type {tid!r} not found",
                assembly_type_id=tid,
            ))

    # Caches for deduplication
    # pin_cache: content_hash → (lattice_id, wrapper_universe_id)
    pin_cache: dict[str, tuple[str, str]] = {}
    # core_cache: content_hash → core_lattice_id
    core_cache: dict[str, str] = {}

    # State reuse tracking
    unique_pin_states: set[str] = set()
    unique_core_states: set[str] = set()
    reused_pin_count = 0
    reused_core_count = 0
    total_pin_lookups = 0
    total_core_lookups = 0

    for seg_idx, segment in enumerate(segments):
        fm = segment.fill_mode

        # ---- Whole-plane material fill ----
        if fm == "whole_plane_material":
            fill_id = segment.base_fill_id
            if fill_id is None:
                result.issues.append(MaterializationIssue(
                    code="fullcore.whole_plane_fill_missing",
                    severity="error",
                    message=f"segment {segment.segment_id} fill_mode=whole_plane_material but base_fill_id is None",
                    segment_id=segment.segment_id,
                ))
                fill_id = "water"
            elif known_material_ids is not None and fill_id not in known_material_ids:
                result.issues.append(MaterializationIssue(
                    code="fullcore.whole_plane_ref_missing",
                    severity="error",
                    message=f"segment {segment.segment_id} material {fill_id!r} not in material catalog",
                    segment_id=segment.segment_id,
                ))
            layer = AxialLayerSpec(
                id=f"layer_seg{seg_idx}",
                name=f"axial layer segment {seg_idx} ({segment.base_role})",
                z_min_cm=segment.z_min_cm,
                z_max_cm=segment.z_max_cm,
                fill=FillRefSpec(type="material", id=fill_id),
                purpose=f"Whole-plane material slab z=[{segment.z_min_cm:.3f}, {segment.z_max_cm:.3f}]",
            )
            result.axial_layers.append(layer)
            result.segment_index.append({
                "segment_id": segment.segment_id,
                "z_min_cm": segment.z_min_cm,
                "z_max_cm": segment.z_max_cm,
                "fill_mode": fm,
                "base_role": segment.base_role,
                "fill_id": fill_id,
            })
            continue

        # ---- Whole-plane universe fill ----
        if fm == "whole_plane_universe":
            fill_id = segment.base_fill_id
            if fill_id is None:
                result.issues.append(MaterializationIssue(
                    code="fullcore.whole_plane_fill_missing",
                    severity="error",
                    message=f"segment {segment.segment_id} fill_mode=whole_plane_universe but base_fill_id is None",
                    segment_id=segment.segment_id,
                ))
                fill_id = moderator_universe_id
            elif known_universe_ids is not None and fill_id not in known_universe_ids:
                result.issues.append(MaterializationIssue(
                    code="fullcore.whole_plane_ref_missing",
                    severity="error",
                    message=f"segment {segment.segment_id} universe {fill_id!r} not in universe catalog",
                    segment_id=segment.segment_id,
                ))
            layer = AxialLayerSpec(
                id=f"layer_seg{seg_idx}",
                name=f"axial layer segment {seg_idx} ({segment.base_role})",
                z_min_cm=segment.z_min_cm,
                z_max_cm=segment.z_max_cm,
                fill=FillRefSpec(type="universe", id=fill_id),
                purpose=f"Whole-plane universe slab z=[{segment.z_min_cm:.3f}, {segment.z_max_cm:.3f}]",
            )
            result.axial_layers.append(layer)
            result.segment_index.append({
                "segment_id": segment.segment_id,
                "z_min_cm": segment.z_min_cm,
                "z_max_cm": segment.z_max_cm,
                "fill_mode": fm,
                "base_role": segment.base_role,
                "fill_id": fill_id,
            })
            continue

        # ---- Void fill ----
        if fm == "void":
            layer = AxialLayerSpec(
                id=f"layer_seg{seg_idx}",
                name=f"axial layer segment {seg_idx} (void)",
                z_min_cm=segment.z_min_cm,
                z_max_cm=segment.z_max_cm,
                fill=FillRefSpec(type="void"),
                purpose=f"Void slab z=[{segment.z_min_cm:.3f}, {segment.z_max_cm:.3f}]",
            )
            result.axial_layers.append(layer)
            result.segment_index.append({
                "segment_id": segment.segment_id,
                "z_min_cm": segment.z_min_cm,
                "z_max_cm": segment.z_max_cm,
                "fill_mode": fm,
                "base_role": segment.base_role,
            })
            continue

        # ---- Detailed core fill ----
        active_map = _get_active_inserts_for_segment(
            segment, catalog, resolved_profiles,
        )

        # Spacer grid state for this segment
        active_grids = _get_active_grids_for_segment(segment, grid_overlays)
        grid_state = AssemblyGridState()
        grid_lookup = grid_density_lookup or {}
        for gov in active_grids:
            grid_state.active_overlay_ids.append(gov.overlay_id)
            mat_id = gov.material_id or ""
            grid_state.active_material_ids.append(mat_id)
            density = grid_lookup.get(mat_id)
            if density is None:
                if fail_closed:
                    result.issues.append(MaterializationIssue(
                        code="fullcore.grid_density_missing",
                        severity="error",
                        message=f"grid overlay {gov.overlay_id!r} material {mat_id!r} has no density",
                        segment_id=segment.segment_id,
                    ))
                density = 0.0
            grid_mass = gov.total_mass_g
            if grid_mass is None:
                if fail_closed:
                    result.issues.append(MaterializationIssue(
                        code="fullcore.grid_mass_missing",
                        severity="error",
                        message=f"grid overlay {gov.overlay_id!r} has no total_mass_g",
                        segment_id=segment.segment_id,
                    ))
                grid_mass = 0.0
            grid_height = (gov.z_max_cm or 0.0) - (ov_zmin if (ov_zmin := gov.z_min_cm) is not None else 0.0)
            cell_ct = gov.cell_count or 289
            a_cell, inner_side, ft = _compute_grid_frame_exact(
                grid_mass, density, cell_ct, grid_height, pitch_cm,
            )
            # Back-calculate mass for conservation check
            back_mass = _back_calculate_mass(inner_side, pitch_cm, density, cell_ct, grid_height)
            mass_err = abs(back_mass - grid_mass) / grid_mass if grid_mass > 0 else 0.0
            grid_state.derivation_reports.append(GridFrameDerivationReport(
                overlay_id=gov.overlay_id,
                material_id=mat_id,
                grid_height_cm=grid_height,
                total_mass_g=grid_mass,
                density_g_cm3=density,
                cell_count=cell_ct,
                pitch_cm=pitch_cm,
                area_per_cell_cm2=a_cell,
                frame_thickness_cm=ft,
                volume_fraction=a_cell / (pitch_cm * pitch_cm) if pitch_cm > 0 else 0.0,
            ))
        if active_grids:
            result.grid_states[segment.segment_id] = grid_state

        # Build the state signature for tracking
        sig_parts: list[str] = []
        for atype in catalog.assembly_types:
            tid = atype.assembly_type_id
            acts = active_map.get(tid, [])
            if acts:
                sig_parts.append(f"{tid}:{sorted(a[0] for a in acts)}")
        if grid_state.active_overlay_ids:
            sig_parts.append(f"grid:{sorted(grid_state.active_overlay_ids)}")
        if segment.base_role:
            sig_parts.append(f"role:{segment.base_role}")
        state_sig = "|".join(sig_parts)

        # Build core lattice universe_pattern for this segment
        core_pattern: list[list[str]] = []
        for row in layout.assembly_pattern:
            pattern_row: list[str] = []
            for type_id in row:
                base_uv = base_assembly_universe_ids.get(type_id, outer_universe_id or "moderator_outer")
                acts = active_map.get(type_id)

                # Resolve base path state bindings for this segment role
                atype_obj = next(
                    (at for at in catalog.assembly_types if at.assembly_type_id == type_id),
                    None,
                )
                profile_id = getattr(atype_obj, "base_path_profile_id", None) if atype_obj else None
                bpath_bindings = _resolve_base_path_bindings(
                    segment.base_role, type_id, base_path_profiles, profile_id,
                )

                needs_derived = bool(acts) or bool(bpath_bindings) or (
                    bool(active_grids) and bool(universe_patches_by_id)
                )

                if needs_derived:
                    base_pattern = base_pin_lattices[type_id].universe_pattern
                    derived_pattern = [r[:] for r in base_pattern]

                    # Step 1: Apply base path state (fuel-path switching)
                    derived_pattern, bpath_ids = _apply_base_path_state(
                        derived_pattern, bpath_bindings,
                    )

                    # Step 2: Apply localized inserts
                    all_insert_ids: list[str] = []
                    for insert_id, insert_uv, coords, _pm in (acts or []):
                        coord_issues = _validate_insert_coordinates(
                            coords,
                            base_pin_lattices[type_id].shape or (len(base_pattern), len(base_pattern[0])),
                            type_id,
                            segment.segment_id,
                        )
                        result.issues.extend(coord_issues)

                        for r, c in coords:
                            if 0 <= r < len(derived_pattern) and 0 <= c < len(derived_pattern[r]):
                                derived_pattern[r][c] = insert_uv
                        all_insert_ids.append(insert_id)

                    # Step 3: Apply grid decoration (replace universe IDs with grid-decorated variants)
                    grid_deco_map: dict[str, str] = {}
                    if active_grids and universe_patches_by_id and grid_state.derivation_reports:
                        # Compute grid geometry hash from first active grid
                        dr = grid_state.derivation_reports[0]
                        grid_geo_hash = hashlib.sha256()
                        grid_geo_hash.update(f"|gmat={dr.material_id}".encode())
                        grid_geo_hash.update(f"|dens={dr.density_g_cm3}".encode())
                        grid_geo_hash.update(f"|mass={dr.total_mass_g}".encode())
                        grid_geo_hash.update(f"|inner={dr.area_per_cell_cm2}".encode())
                        grid_geo_hash.update(f"|pitch={dr.pitch_cm}".encode())
                        grid_geo_hash.update(f"|h={dr.grid_height_cm}".encode())
                        grid_geo_hash.update(_MATERIALIZATION_CONTRACT_VERSION.encode())
                        ggh = grid_geo_hash.hexdigest()[:12]

                        inner_side_val = 0.0
                        if dr.pitch_cm > 0 and dr.area_per_cell_cm2 >= 0:
                            val = dr.pitch_cm ** 2 - dr.area_per_cell_cm2
                            inner_side_val = math.sqrt(val) if val > 0 else 0.0

                        # Decorate every unique universe in the pattern
                        unique_uvs = set()
                        for row in derived_pattern:
                            unique_uvs.update(row)

                        for uv_id in sorted(unique_uvs):
                            base_patch = universe_patches_by_id.get(uv_id)
                            if base_patch is None:
                                continue
                            decorated = _make_grid_decorated_universe(
                                uv_id, ggh, dr.material_id,
                                inner_side_val, dr.pitch_cm, base_patch,
                            )
                            if decorated is not None:
                                grid_deco_map[uv_id] = decorated.universe_id
                                if decorated not in result.grid_decorated_universe_patches:
                                    result.grid_decorated_universe_patches.append(decorated)

                        # Replace universe IDs in pattern
                        if grid_deco_map:
                            for r in range(len(derived_pattern)):
                                for c in range(len(derived_pattern[r])):
                                    uv = derived_pattern[r][c]
                                    if uv in grid_deco_map:
                                        derived_pattern[r][c] = grid_deco_map[uv]

                    # Get the pin map for convention info
                    pm_obj = atype_obj.pin_map if atype_obj else None
                    ibase = pm_obj.coordinate_convention.index_base if pm_obj else 0

                    grid_ids = grid_state.active_overlay_ids if grid_state.active_overlay_ids else None
                    pin_hash = _compute_pin_state_hash(
                        type_id, derived_pattern, all_insert_ids, ibase, grid_ids,
                        bpath_ids if bpath_ids else None,
                        segment.base_role,
                    )
                    total_pin_lookups += 1

                    if pin_hash in pin_cache:
                        seg_uv = pin_cache[pin_hash][1]
                        reused_pin_count += 1
                    else:
                        unique_pin_states.add(pin_hash)
                        derived_lat_id = f"assembly_lattice__{type_id}__{pin_hash}"
                        derived_lat = LatticeSpec(
                            id=derived_lat_id,
                            name=f"derived pin lattice {type_id} {pin_hash[:8]}",
                            kind="rect",
                            pitch_cm=(pitch_cm, pitch_cm),
                            outer_universe_id=moderator_universe_id,
                            universe_pattern=derived_pattern,
                            shape=base_pin_lattices[type_id].shape,
                            purpose=f"Derived lattice {type_id} role={segment.base_role} inserts={all_insert_ids} grid={grid_ids or 'none'}",
                        )
                        result.derived_pin_lattices.append(derived_lat)

                        seg_cell_id = f"assembly_wrapper_cell__{type_id}__{pin_hash}"
                        seg_universe_id = f"assembly_universe__{type_id}__{pin_hash}"
                        seg_cell = CellSpec(
                            id=seg_cell_id,
                            name=f"wrapper cell {type_id} {pin_hash[:8]}",
                            fill_type="lattice",
                            fill_id=derived_lat_id,
                            purpose=f"Derived wrapper for {type_id}",
                        )
                        seg_universe = UniverseSpec(
                            id=seg_universe_id,
                            name=f"assembly universe {type_id} {pin_hash[:8]}",
                            cell_ids=[seg_cell_id],
                            purpose=f"Derived wrapper universe for {type_id}",
                        )
                        result.derived_wrapper_cells.append(seg_cell)
                        result.derived_wrapper_universes.append(seg_universe)

                        pin_cache[pin_hash] = (derived_lat_id, seg_universe_id)
                        seg_uv = seg_universe_id
                    pattern_row.append(seg_uv)
                else:
                    pattern_row.append(base_uv)
            core_pattern.append(pattern_row)

        # Core lattice dedup by content hash
        if not active_map and not grid_state.active_overlay_ids:
            seg_core_id_final = core_lattice_id_base
        else:
            core_hash = _compute_core_state_hash(core_pattern, core_lattice_id_base)
            total_core_lookups += 1
            if core_hash in core_cache:
                seg_core_id_final = core_cache[core_hash]
                reused_core_count += 1
            else:
                unique_core_states.add(core_hash)
                seg_core_id_final = f"{core_lattice_id_base}__{core_hash}"
                seg_core_lat = LatticeSpec(
                    id=seg_core_id_final,
                    name=f"core lattice {core_hash[:8]}",
                    kind="rect",
                    pitch_cm=(pitch, pitch),
                    lower_left_cm=lower_left,
                    center_cm=(0.0, 0.0),
                    outer_universe_id=moderator_universe_id,
                    universe_pattern=core_pattern,
                    shape=(n_rows, n_cols),
                    purpose=f"Core lattice for z=[{segment.z_min_cm:.2f}, {segment.z_max_cm:.2f}] sig={state_sig[:40]}",
                )
                result.segment_core_lattices.append(seg_core_lat)
                core_cache[core_hash] = seg_core_id_final

        # Create axial layer
        layer = AxialLayerSpec(
            id=f"layer_seg{seg_idx}",
            name=f"axial layer segment {seg_idx}",
            z_min_cm=segment.z_min_cm,
            z_max_cm=segment.z_max_cm,
            fill=FillRefSpec(type="lattice", id=seg_core_id_final),
            purpose=f"Segment {seg_idx}: z=[{segment.z_min_cm:.2f}, {segment.z_max_cm:.2f}] cm",
        )
        result.axial_layers.append(layer)

        result.segment_index.append({
            "segment_id": segment.segment_id,
            "z_min_cm": segment.z_min_cm,
            "z_max_cm": segment.z_max_cm,
            "fill_mode": fm,
            "base_role": segment.base_role,
            "core_lattice_id": seg_core_id_final,
            "active_types": list(active_map.keys()),
            "state_signature": state_sig,
            "has_derived_lattices": bool(active_map),
            "grid_overlay_ids": list(grid_state.active_overlay_ids),
        })

    # Build state reuse report
    result.state_reuse_report = {
        "contract_version": _MATERIALIZATION_CONTRACT_VERSION,
        "segment_count": len(segments),
        "pin_state_lookups": total_pin_lookups,
        "unique_pin_states": len(unique_pin_states),
        "reused_pin_states": reused_pin_count,
        "pin_reuse_ratio": reused_pin_count / total_pin_lookups if total_pin_lookups > 0 else 0.0,
        "core_state_lookups": total_core_lookups,
        "unique_core_states": len(unique_core_states),
        "reused_core_states": reused_core_count,
        "core_reuse_ratio": reused_core_count / total_core_lookups if total_core_lookups > 0 else 0.0,
        "pin_state_hashes": sorted(unique_pin_states),
        "core_state_hashes": sorted(unique_core_states),
    }

    return result


__all__ = [
    "MaterializationIssue",
    "GridFrameDerivationReport",
    "AssemblyGridState",
    "ConcreteAxialStateResult",
    "materialize_concrete_axial_states",
    "_compute_grid_frame_derivation",
    "_compute_pin_state_hash",
    "_compute_core_state_hash",
]
