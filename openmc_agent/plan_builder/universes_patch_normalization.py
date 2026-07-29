"""Deterministic post-merge normalizations for UniversesPatch content.

Mirrors ``materials_patch_normalization.py`` but targets universe-level
corrections that require full cross-fragment visibility — specifically the
merge of LLM-generated ``implicit_*`` satellite universes into their
intended host (e.g. a ``fuel_pin`` emitted with only a fuel pellet cell
while the gas-gap and cladding layers were split into a separate
``implicit_gas_gap`` universe).

All corrections are input-driven and reactor-neutral: cell-role →
material-role mismatches are inferred from the valid MaterialsPatch in
state; no geometry dimensions are hardcoded; merges happen only when the
host is unambiguous.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class UniversesPatchNormalizationResult:
    """Result of patch-level deterministic normalization."""

    content: dict[str, Any]
    operations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.operations)


# Canonical cell-role label for each known material role.  When a cell's
# declared role is incompatible with its material's role, the role is
# corrected to this canonical value.
_CANONICAL_CELL_ROLE: dict[str, str] = {
    "fuel": "fuel",
    "cladding": "cladding",
    "coolant": "coolant",
    "moderator": "moderator",
    "structural": "structural",
    "absorber": "absorber",
    "poison": "poison",
    "gas": "gas_gap",
}

# Role compatibility — mirrors the table in material_universe_binding.py so
# the normalizer can decide whether a cell role is an acceptable alias for
# a material role (e.g. ``clad`` is fine for ``cladding``) or truly
# incompatible (e.g. ``gas_gap`` for ``cladding``) and needs correction.
_ROLE_COMPATIBILITY: dict[str, set[str]] = {
    "fuel": {"fuel"},
    "cladding": {"clad", "cladding", "wall", "tube", "endplug", "internal", "structural", "plug", "frame"},
    "coolant": {"coolant", "moderator", "background", "water", "gas", "inner_flow"},
    "moderator": {"coolant", "moderator", "background", "water"},
    "structural": {"clad", "cladding", "wall", "tube", "structural", "frame", "can", "endplug", "internal", "plug"},
    "absorber": {"absorber", "poison", "control"},
    "poison": {"absorber", "poison", "control"},
    "gas": {"gap", "plenum", "coolant", "gas", "gas_gap"},
}

_ALL_CELL_ROLES: set[str] = {tok for toks in _ROLE_COMPATIBILITY.values() for tok in toks}

# Roles that indicate a radial layer of a fuel_pin (complement the fuel
# pellet).  Only implicit universes providing at least one of these roles
# are candidates for merge into a fuel_pin host.  Axial structures (end
# plugs, nozzles) and background fills (moderator, coolant) are NOT radial
# layers and must remain as separate universes.
_RADIAL_LAYER_ROLES: set[str] = {"gap", "gas_gap", "cladding", "clad"}

# Radial ordering priority for merged cells (innermost → outermost).
# When cells from an implicit satellite are merged into a host fuel_pin,
# their r_min may start at 0.0 (the satellite's own coordinate origin).
# The normalizer rebuilds radial contiguity using this ordering so that
# merged cells do not overlap existing host cells.
_RADIAL_ORDER: list[tuple[str, int]] = [
    ("fuel", 0),
    ("gap", 1), ("gas_gap", 1),
    ("clad", 2), ("cladding", 2), ("structural", 2), ("wall", 2), ("tube", 2),
    ("coolant", 3), ("moderator", 3), ("background", 3), ("water", 3),
]


def _radial_priority(role: str) -> int:
    role_l = role.lower()
    for token, prio in _RADIAL_ORDER:
        if token in role_l:
            return prio
    return 99


def _role_compatible(cell_role: str, material_role: str) -> bool:
    """Return True if *cell_role* is an acceptable alias for *material_role*."""
    cell_l = cell_role.lower()
    mat_l = material_role.lower()
    allowed = _ROLE_COMPATIBILITY.get(mat_l)
    if allowed is None:
        return True
    for token in allowed:
        if token in cell_l or cell_l in token:
            return True
    return False


def _is_standard_cell_role(cell_role: str) -> bool:
    cell_l = cell_role.lower()
    return any(token in cell_l for token in _ALL_CELL_ROLES)


def _build_material_role_map(state: Any) -> dict[str, str]:
    """Extract material_id → role from state's valid MaterialsPatch."""
    roles: dict[str, str] = {}
    if state is None:
        return roles
    for env in getattr(state, "patches", {}).values():
        if (
            getattr(env, "patch_type", None) == "materials"
            and getattr(env, "status", None) == "valid"
        ):
            for mat in env.content.get("materials", []):
                mid = mat.get("material_id")
                role = mat.get("role")
                if mid and role:
                    roles[mid] = role
            break
    return roles


def _correct_cell_role(
    cell: dict[str, Any],
    material_roles: dict[str, str],
) -> dict[str, Any] | None:
    """Return a corrected copy of *cell* if its role mismatches its material.

    Returns ``None`` when no correction is needed.
    """
    mat_id = cell.get("material_id")
    cell_role = cell.get("role", "")
    if not mat_id or not cell_role:
        return None
    mat_role = material_roles.get(mat_id)
    if not mat_role or mat_role not in _CANONICAL_CELL_ROLE:
        return None
    if _role_compatible(cell_role, mat_role):
        return None
    if not _is_standard_cell_role(cell_role):
        return None
    corrected = dict(cell)
    corrected["role"] = _CANONICAL_CELL_ROLE[mat_role]
    return corrected


def _adjust_merged_radii(
    cells: list[dict[str, Any]],
    host_outer_r_max: float,
) -> list[dict[str, Any]]:
    """Rebuild radial contiguity for cells merged from an implicit satellite.

    Satellite cells were authored in their own coordinate origin (r_min=0).
    After merge they must be contiguous with the host's existing cells so
    that the MU preflight does not report ``radial_overlap``.  Cells are
    sorted by radial role priority (fuel → gap → cladding) and each cell's
    r_min is clamped to the running cursor.  If a cell's r_max ends up
    below r_min it is clamped to r_min (zero-thickness — flagged by
    downstream checks but never overlapping).
    """
    sorted_cells = sorted(cells, key=lambda c: _radial_priority(c.get("role", "")))
    cursor = host_outer_r_max
    for cell in sorted_cells:
        r_min = cell.get("r_min_cm")
        r_max = cell.get("r_max_cm")
        if r_min is None or r_max is None:
            continue
        if r_min < cursor:
            cell["r_min_cm"] = cursor
            if r_max < cursor:
                cell["r_max_cm"] = cursor
            cursor = cell["r_max_cm"]
        else:
            cursor = max(cursor, r_max)
    return sorted_cells


def normalize_universes_patch_content(
    content: dict[str, Any],
    *,
    state: Any | None = None,
    requirement_text: str | None = None,
) -> UniversesPatchNormalizationResult:
    """Apply deterministic post-merge normalizations to a UniversesPatch dict.

    Currently handles:

    - **implicit_universe_merged**: When the LLM splits a fuel_pin into a
      main universe (fuel pellet only) and a satellite ``implicit_*``
      universe (gap + cladding), the satellite's cells are merged back
      into the host.  Cell roles that are incompatible with their
      material's role are corrected, resolving
      ``material_role_mismatch`` errors at the source.

    The normalizer is idempotent.
    """
    operations: list[dict[str, Any]] = []
    universes = content.get("universes")
    if not isinstance(universes, list) or not universes:
        return UniversesPatchNormalizationResult(content=content, operations=operations)

    has_implicit = any(
        isinstance(u, dict) and str(u.get("universe_id", "")).startswith("implicit_")
        for u in universes
    )
    if not has_implicit:
        return UniversesPatchNormalizationResult(content=content, operations=operations)

    material_roles = _build_material_role_map(state)
    new_content = copy.deepcopy(content)
    new_universes = new_content["universes"]

    # Build implicit list from the deep copy so mutations land on new_content.
    implicit_universes = [
        u for u in new_universes
        if isinstance(u, dict) and str(u.get("universe_id", "")).startswith("implicit_")
    ]

    for implicit in list(implicit_universes):
        if implicit not in new_universes:
            continue
        implicit_id = implicit.get("universe_id", "")
        implicit_cells = implicit.get("cells", [])

        # ---- Step 1: Correct cell roles (always, even if merge is skipped).
        corrected_cells: list[dict[str, Any]] = []
        role_corrections: list[dict[str, Any]] = []
        for cell in implicit_cells:
            corrected = _correct_cell_role(cell, material_roles)
            if corrected is not None:
                role_corrections.append({
                    "cell_id": corrected.get("id"),
                    "material_id": corrected.get("material_id"),
                    "old_role": cell.get("role", ""),
                    "new_role": corrected.get("role"),
                })
                corrected_cells.append(corrected)
            else:
                corrected_cells.append(dict(cell))

        if role_corrections:
            implicit["cells"] = corrected_cells
            operations.append({
                "operation": "implicit_universe_role_corrected",
                "implicit_universe_id": implicit_id,
                "role_corrections": role_corrections,
            })

        provided_roles = {c.get("role", "") for c in corrected_cells}

        # ---- Step 2: Only merge radial-layer satellites (gap / cladding).
        if not (provided_roles & _RADIAL_LAYER_ROLES):
            continue

        # ---- Step 3: Find exactly one fuel_pin host missing provided roles.
        hosts = []
        for u in new_universes:
            if u is implicit or not isinstance(u, dict):
                continue
            if u.get("kind") != "fuel_pin":
                continue
            existing_roles = {
                c.get("role", "") for c in u.get("cells", [])
                if isinstance(c, dict)
            }
            if provided_roles - existing_roles:
                hosts.append(u)

        if len(hosts) != 1:
            operations.append({
                "operation": "implicit_universe_merge_skipped",
                "implicit_universe_id": implicit_id,
                "host_count": len(hosts),
                "reason": "ambiguous_host" if len(hosts) > 1 else "no_host",
            })
            continue

        host = hosts[0]
        host_id = host.get("universe_id", "")

        # ---- Step 4: Check radial compatibility.
        # Satellite cells may have been authored with r_min=0 in their own
        # coordinate origin.  If a cell's r_max is ≤ the host's outermost
        # r_max, shifting it would produce zero/negative thickness.
        host_r_max = 0.0
        for c in host.get("cells", []):
            if isinstance(c, dict):
                r_max = c.get("r_max_cm")
                if r_max is not None and r_max > host_r_max:
                    host_r_max = r_max
        radius_incompatible = False
        for c in corrected_cells:
            r_min = c.get("r_min_cm")
            r_max = c.get("r_max_cm")
            if r_min is not None and r_max is not None:
                if r_min < host_r_max and r_max <= host_r_max:
                    radius_incompatible = True
                    break

        if radius_incompatible:
            operations.append({
                "operation": "implicit_universe_merge_skipped",
                "implicit_universe_id": implicit_id,
                "host_universe_id": host_id,
                "reason": "radius_incompatible",
            })
            continue

        # ---- Step 5: Merge with radial contiguity adjustment.
        existing_cell_ids = {
            c.get("id") for c in host.get("cells", [])
            if isinstance(c, dict)
        }
        cells_to_merge = [
            c for c in corrected_cells
            if not (c.get("id") and c.get("id") in existing_cell_ids)
        ]
        cells_to_merge = _adjust_merged_radii(cells_to_merge, host_r_max)

        merged_count = 0
        for cell in cells_to_merge:
            host.setdefault("cells", []).append(cell)
            merged_count += 1

        new_universes.remove(implicit)

        operations.append({
            "operation": "implicit_universe_merged",
            "implicit_universe_id": implicit_id,
            "host_universe_id": host_id,
            "cells_merged": merged_count,
            "role_corrections": role_corrections,
        })

    if operations:
        new_content["universes"] = new_universes
        return UniversesPatchNormalizationResult(content=new_content, operations=operations)
    return UniversesPatchNormalizationResult(content=content, operations=operations)
