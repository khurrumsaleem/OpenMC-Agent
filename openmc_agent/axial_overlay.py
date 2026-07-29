"""Level 1 ``homogenized_open_region`` axial-overlay logic.

Pure logic shared by the guard, the renderer capability check, and the
executor script generation. No OpenMC imports, no script text -- just
classification, segmentation, and derivation planning so the three call sites
stay consistent.

Level 1 approximation semantics
-------------------------------
A spacer/support grid is a thin structural band overlaid on the pin lattice.
Fuel rods, cladding, guide tubes and instrument tubes pass *through* it; only
the coolant / open region receives homogenized grid material. This module
identifies, per pin/tube universe, the single "open region" cell whose fill may
be swapped for the grid material, while every protected solid (fuel, clad, gap,
absorber, tube wall) is left untouched.

Conservative rules
------------------
* A universe with exactly one open cell -> derive an overlay universe (open cell
  fill becomes grid material, other cells reused).
* A universe with two or more open cells (e.g. a guide tube with an inner
  water channel plus outer moderator) -> do NOT alter it; reuse the base
  universe. The schema cannot safely tell inner through-path from outer
  moderator, so we preserve through-paths and forgo grid material there.
* A universe with zero open cells -> ``open_region_unresolved``: the renderer
  cannot safely place the grid material, so the model downgrades.

No benchmark facts live here: classification is by generic material/cell names.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from openmc_agent.schemas import (
    AxialLayerSpec,
    AxialOverlaySpec,
    CellSpec,
    ComplexMaterialSpec,
    ComplexModelSpec,
    LatticeSpec,
    UniverseSpec,
)

# Material id/name tokens that mark a cell as the coolant / open region whose
# fill may be replaced by homogenized grid material. Generic only.
_OPEN_MATERIAL_TOKENS: tuple[str, ...] = (
    "water",
    "coolant",
    "moderator",
    "h2o",
    "borated",
    "boron_water",  # alias
)

# Tokens that mark a solid that must NEVER be replaced by grid material.
_PROTECTED_MATERIAL_TOKENS: tuple[str, ...] = (
    "fuel",
    "uo2",
    "mox",
    "clad",
    "cladding",
    "zircaloy",
    "zr4",
    "gap",
    "helium",
    "absorber",
    "absorb",
    "burnable",
    "pyrex",
    "tube",
    "structural",
    "grid",
    "inconel",
    "ss304",
    "steel",
)

MaterialRole = Literal["open", "protected", "unknown"]

# Geometry modes the current renderer can actually turn into geometry. Level 1
# adds ``homogenized_open_region``; Level 2 adds ``mass_conserving_outer_frame``;
# explicit bars / annular shells / volume-fraction modes still downgrade.
SUPPORTED_GEOMETRY_MODES: frozenset[str] = frozenset({
    "homogenized_open_region",
    "mass_conserving_outer_frame",
})

_Z_TOLERANCE_CM: float = 1e-6


@dataclass(frozen=True)
class AxialSegment:
    """A rendered axial z-slice after overlay boundaries are merged in.

    ``overlay`` is the covering overlay when the segment lies inside a supported
    overlay's z-range and the overlay targets the segment's base-layer lattice;
    otherwise ``None`` and the segment keeps the original layer fill.
    """

    z_min: float
    z_max: float
    layer: AxialLayerSpec
    overlay: AxialOverlaySpec | None


@dataclass(frozen=True)
class DerivedUniversePlan:
    """How a single base universe is handled inside an overlay lattice.

    ``derived_universe_id`` is the new universe id when the open region was
    derived (single open cell swapped to grid material). ``reuse_base`` is True
    when the base universe is kept as-is (conservative multi-open-cell case).
    ``unresolved`` is True when no open region could be identified.
    """

    base_universe_id: str
    derived_universe_id: str | None
    open_cell_id: str | None
    reuse_base: bool
    unresolved: bool


def _material_text(material: ComplexMaterialSpec) -> str:
    parts = [material.id, material.name]
    if material.chemical_formula:
        parts.append(material.chemical_formula)
    if material.macroscopic:
        parts.append(material.macroscopic)
    return " ".join(parts).lower()


def classify_material_role(material: ComplexMaterialSpec) -> MaterialRole:
    """Classify a material as the coolant/open region, a protected solid, or unknown.

    Protected solids (fuel, clad, gap, absorber, tube wall, grid alloy itself)
    are never replaced. Open materials (water/coolant/moderator) may be swapped
    for grid material by the Level 1 overlay.
    """
    text = _material_text(material)
    # Grid alloy is itself protected: never replace a grid cell with grid.
    if any(token in text for token in _PROTECTED_MATERIAL_TOKENS):
        return "protected"
    if any(token in text for token in _OPEN_MATERIAL_TOKENS):
        return "open"
    return "unknown"


def _cells_by_id(model: ComplexModelSpec) -> dict[str, CellSpec]:
    return {cell.id: cell for cell in model.cells}


def _materials_by_id(model: ComplexModelSpec) -> dict[str, ComplexMaterialSpec]:
    return {material.id: material for material in model.materials}


def _universe_by_id(model: ComplexModelSpec) -> dict[str, UniverseSpec]:
    return {universe.id: universe for universe in model.universes}


def universe_open_cell_ids(
    universe: UniverseSpec,
    model: ComplexModelSpec,
) -> list[str]:
    """Open-region cell ids of ``universe`` (cells whose fill material is the
    coolant/moderator). Cells whose material role is protected or unknown, and
    non-material cells, are excluded.
    """
    cells = _cells_by_id(model)
    materials = _materials_by_id(model)
    open_ids: list[str] = []
    for cell_id in universe.cell_ids:
        cell = cells.get(cell_id)
        if cell is None or cell.fill_type != "material" or cell.fill_id is None:
            continue
        material = materials.get(cell.fill_id)
        if material is None:
            continue
        if classify_material_role(material) == "open":
            open_ids.append(cell_id)
    return open_ids


def lattice_by_id(model: ComplexModelSpec, lattice_id: str | None) -> LatticeSpec | None:
    if lattice_id is None:
        return None
    for lattice in model.lattices:
        if lattice.id == lattice_id:
            return lattice
    return None


def overlay_target_lattice(overlay: AxialOverlaySpec, model: ComplexModelSpec) -> LatticeSpec | None:
    return lattice_by_id(model, overlay.target_lattice_id)


def overlay_material(overlay: AxialOverlaySpec, model: ComplexModelSpec) -> ComplexMaterialSpec | None:
    if overlay.material_id is None:
        return None
    for material in model.materials:
        if material.id == overlay.material_id:
            return material
    return None


def overlay_is_promotable_to_level1(overlay: AxialOverlaySpec, model: ComplexModelSpec) -> bool:
    """A skeleton overlay that already carries enough data for Level 1 rendering.

    Planners sometimes write ``geometry_mode='skeleton'`` (a conservative
    "please confirm") even when they have supplied the target lattice, grid
    material and z-range. Such an overlay can be auto-promoted to
    ``homogenized_open_region`` instead of forcing a review-only skeleton.
    Returns True when the overlay has a grid-like kind, a valid z-range, a
    resolvable rectangular target lattice, and a resolvable material.
    """
    if overlay.overlay_kind not in {"spacer_grid", "support_plate"}:
        return False
    if overlay.z_min_cm is None or overlay.z_max_cm is None:
        return False
    if overlay.z_min_cm >= overlay.z_max_cm:
        return False
    target = overlay_target_lattice(overlay, model)
    if target is None or target.kind != "rect":
        return False
    return overlay_material(overlay, model) is not None


def overlay_is_structurally_renderable(overlay: AxialOverlaySpec, model: ComplexModelSpec) -> bool:
    """True when an overlay meets all structural preconditions for rendering.

    * ``homogenized_open_region`` — promotable + through-path preserved.
    * ``mass_conserving_outer_frame`` — promotable + through-path preserved +
      total_mass_g provided.
    * ``skeleton`` — auto-promotes when promotable.

    Does NOT check the deep open-region derivability or clearance — those are
    handled by the guard and the planner respectively.
    """
    if overlay.geometry_mode == "homogenized_open_region":
        return overlay_is_promotable_to_level1(overlay, model) and overlay.through_path_preserved is True
    if overlay.geometry_mode == "mass_conserving_outer_frame":
        return (
            overlay_is_promotable_to_level1(overlay, model)
            and overlay.through_path_preserved is True
            and overlay.total_mass_g is not None
            and overlay.total_mass_g > 0
        )
    if overlay.geometry_mode == "skeleton":
        return overlay_is_promotable_to_level1(overlay, model)
    return False


def _layer_effective_lattice_id(layer: AxialLayerSpec) -> str | None:
    """Lattice id a layer renders with (its fill lattice, before any loading
    override). Overlay targeting matches against this."""
    if layer.fill.type == "lattice":
        return layer.fill.id
    return None


def _layer_matches_overlay_target(
    layer: AxialLayerSpec,
    overlay: AxialOverlaySpec,
    model: ComplexModelSpec,
) -> bool:
    """Whether an overlay targets a layer's effective or loading-base lattice.

    Axial lattice transformations materialize a layer to a derived lattice while
    retaining its loading metadata for provenance. An overlay authored against
    the loading's base lattice must continue to cover that transformed layer.
    """
    if _layer_effective_lattice_id(layer) == overlay.target_lattice_id:
        return True
    if overlay.target_lattice_id is None:
        return False
    loading_ids = list(layer.loading_ids) or (
        [layer.loading_id] if layer.loading_id is not None else []
    )
    loading_by_id = {loading.id: loading for loading in model.lattice_loadings}
    return any(
        (loading := loading_by_id.get(loading_id)) is not None
        and loading.base_lattice_id == overlay.target_lattice_id
        for loading_id in loading_ids
    )


def overlay_targets_any_layer(overlay: AxialOverlaySpec, model: ComplexModelSpec) -> bool:
    """True when the overlay's target lattice is the fill lattice of at least
    one axial layer (i.e. the overlay has somewhere to apply)."""
    if model.core is None:
        return False
    target = overlay.target_lattice_id
    if target is None:
        return False
    return any(
        _layer_matches_overlay_target(layer, overlay, model)
        for layer in model.core.axial_layers
    )


def detect_overlay_z_overlaps(
    overlays: list[AxialOverlaySpec],
) -> list[tuple[AxialOverlaySpec, AxialOverlaySpec]]:
    """Pairs of overlays whose z-ranges overlap (beyond tolerance). Used to flag
    unsupported concurrent overlays with different material/mode."""
    pairs: list[tuple[AxialOverlaySpec, AxialOverlaySpec]] = []
    for i, a in enumerate(overlays):
        if a.z_min_cm is None or a.z_max_cm is None:
            continue
        for b in overlays[i + 1 :]:
            if b.z_min_cm is None or b.z_max_cm is None:
                continue
            if a.z_min_cm < b.z_max_cm - _Z_TOLERANCE_CM and b.z_min_cm < a.z_max_cm - _Z_TOLERANCE_CM:
                pairs.append((a, b))
    return pairs


def derive_overlay_universe_plan(
    overlay: AxialOverlaySpec,
    model: ComplexModelSpec,
) -> tuple[list[DerivedUniversePlan], list[str]]:
    """Plan the derived overlay universe for each unique base universe in the
    overlay's target lattice.

    Returns ``(plans, unresolved_universe_ids)``. When ``unresolved`` is non-empty
    the renderer cannot safely place the grid material for those universes.
    Universes with 0 open cells are conservatively reused (no grid material,
    through-paths preserved) rather than blocking -- the same safe degradation
    as the 2+-open-cells case.
    """
    target = overlay_target_lattice(overlay, model)
    plans: list[DerivedUniversePlan] = []
    unresolved: list[str] = []
    if target is None:
        return plans, unresolved

    seen: set[str] = set()
    universes = _universe_by_id(model)
    for row in target.universe_pattern:
        for universe_id in row:
            if universe_id in seen:
                continue
            seen.add(universe_id)
            universe = universes.get(universe_id)
            if universe is None:
                # Unknown universe -- nothing we can derive; reuse (let other
                # guards flag the dangling ref).
                plans.append(
                    DerivedUniversePlan(
                        base_universe_id=universe_id,
                        derived_universe_id=None,
                        open_cell_id=None,
                        reuse_base=True,
                        unresolved=False,
                    )
                )
                continue
            open_cells = universe_open_cell_ids(universe, model)
            if len(open_cells) == 1:
                derived_id = f"{universe_id}__overlay_{overlay.id}"
                plans.append(
                    DerivedUniversePlan(
                        base_universe_id=universe_id,
                        derived_universe_id=derived_id,
                        open_cell_id=open_cells[0],
                        reuse_base=False,
                        unresolved=False,
                    )
                )
            elif len(open_cells) >= 2:
                # Several open/coolant cells (e.g. a fuel pin with both an
                # inner coolant annulus and an outer background moderator).
                # The grid frame belongs in the OUTER moderator, so when
                # exactly one open cell is the background
                # (component_role='background'), derive the grid universe
                # from it instead of degrading to no grid material.
                cells_map = _cells_by_id(model)
                bg_cells = [
                    cid for cid in open_cells
                    if getattr(cells_map.get(cid), "component_role", "") == "background"
                ]
                if len(bg_cells) == 1:
                    derived_id = f"{universe_id}__overlay_{overlay.id}"
                    plans.append(
                        DerivedUniversePlan(
                            base_universe_id=universe_id,
                            derived_universe_id=derived_id,
                            open_cell_id=bg_cells[0],
                            reuse_base=False,
                            unresolved=False,
                        )
                    )
                else:
                    # Genuinely ambiguous: preserve the base universe so
                    # through-paths are guaranteed.
                    plans.append(
                        DerivedUniversePlan(
                            base_universe_id=universe_id,
                            derived_universe_id=None,
                            open_cell_id=None,
                            reuse_base=True,
                            unresolved=False,
                        )
                    )
            else:
                # No recognizable open/coolant cell. Conservatively reuse the
                # base universe (no grid material added at these positions)
                # instead of blocking the entire model. This is the same safe
                # degradation as the 2+-open-cells case: through-paths are
                # preserved, grid material simply doesn't appear here. The
                # verification digest can note which universes were skipped.
                plans.append(
                    DerivedUniversePlan(
                        base_universe_id=universe_id,
                        derived_universe_id=None,
                        open_cell_id=None,
                        reuse_base=True,
                        unresolved=False,
                    )
                )
    return plans, unresolved


def compute_axial_segments(
    model: ComplexModelSpec,
) -> list[AxialSegment]:
    """Merge ``core.axial_layers`` boundaries with renderable overlay boundaries
    and return the ordered z-segments, each annotated with its base layer and
    covering overlay (or None).

    Overlays only split the layers whose fill lattice matches the overlay's
    ``target_lattice_id``; nozzle / plenum / end-plug layers are untouched.
    """
    if model.core is None:
        return []

    layers = model.core.axial_layers
    if not layers:
        return []

    boundaries: set[float] = set()
    for layer in layers:
        boundaries.add(layer.z_min_cm)
        boundaries.add(layer.z_max_cm)

    renderable_overlays = [
        o
        for o in model.core.axial_overlays
        if overlay_is_structurally_renderable(o, model)
        and overlay_targets_any_layer(o, model)
        and o.z_min_cm is not None
        and o.z_max_cm is not None
    ]
    for overlay in renderable_overlays:
        boundaries.add(overlay.z_min_cm)
        boundaries.add(overlay.z_max_cm)

    sorted_z = sorted(boundaries)
    segments: list[AxialSegment] = []
    for z0, z1 in zip(sorted_z[:-1], sorted_z[1:]):
        if z1 - z0 <= _Z_TOLERANCE_CM:
            continue
        # Find the layer that fully contains this segment.
        layer = next(
            (
                L
                for L in layers
                if L.z_min_cm - _Z_TOLERANCE_CM <= z0
                and z1 <= L.z_max_cm + _Z_TOLERANCE_CM
            ),
            None,
        )
        if layer is None:
            continue  # gap between layers (non-tiling axial stack)
        # Find a renderable overlay covering this segment whose target matches
        # the layer's effective lattice or its loading's base lattice.
        covering = next(
            (
                o
                for o in renderable_overlays
                if _layer_matches_overlay_target(layer, o, model)
                and o.z_min_cm - _Z_TOLERANCE_CM <= z0
                and z1 <= o.z_max_cm + _Z_TOLERANCE_CM
            ),
            None,
        )
        segments.append(AxialSegment(z_min=z0, z_max=z1, layer=layer, overlay=covering))
    return segments


__all__ = [
    "AxialSegment",
    "DerivedUniversePlan",
    "SUPPORTED_GEOMETRY_MODES",
    "classify_material_role",
    "compute_axial_segments",
    "detect_overlay_z_overlaps",
    "derive_overlay_universe_plan",
    "lattice_by_id",
    "overlay_is_promotable_to_level1",
    "overlay_is_structurally_renderable",
    "overlay_material",
    "overlay_target_lattice",
    "overlay_targets_any_layer",
    "universe_open_cell_ids",
]
