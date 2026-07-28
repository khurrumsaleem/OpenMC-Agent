"""Deterministic annular-insert universe construction oracle.

When an annular-insert universe fragment (a ``pyrex_rod`` and similar
localized inserts that have a concentric radial cross-section) repeatedly
fails LLM qualification, this oracle constructs the canonical concentric
cell sequence deterministically by parsing the radial cross-section table
declared in the source document.

The oracle is reactor-neutral and input-driven:

* It does NOT hardcode any benchmark radii, material ids, or layer counts.
* It reads the radial structure (a ``<rmin>-<rmax> cm`` cross-section table
  or a ``<role>=<rmin>-<rmax> cm`` note) from the requirement text.
* It binds every layer's material to the available material catalog by role
  and name matching, and fail-closes when a layer cannot be bound.
* The constructed universe is returned to the caller, which must still pass it
  through the standard :func:`qualify_universe_fragment` pipeline, so the
  oracle's output is held to the same structural standard as the LLM's.

The oracle is a fallback only: it is invoked after the LLM fragment attempts
have failed, and it never overrides an accepted LLM-produced fragment.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from ..patches import CellLayerPatch, UniverseSpecPatch

# Kinds that describe a localized insert with a concentric radial profile.
#
# Production invocation is currently gated to ``pyrex_rod``: that kind has a
# strict four-check deterministic qualification gate (annular poison, center
# helium, >=3 helium cells, >3 non-background cells) plus the concentric
# radial-profile validator, so a wrongly constructed universe cannot pass.
# Other annular inserts (e.g. control rods, thimble plugs) can be enabled once
# their kind-specific qualification gates are verified to reject a mis-parsed
# structure; the construction logic below is already kind-agnostic.
_ANNULAR_INSERT_KINDS: frozenset[str] = frozenset({
    "pyrex_rod",
})

# Tolerance (cm) for radial contiguity checks between adjacent layers.
_CONTIGUITY_TOL_CM = 1e-3

# Matches a bounded radial range ``<rmin>-<rmax> cm`` allowing en/em dashes.
_RANGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*[–—\-]\s*(\d+(?:\.\d+)?)\s*cm",
)
# Matches an unbounded ``r >= <r>`` background region.
_BACKGROUND_RE = re.compile(r"r\s*[≥>]\s*(\d+(?:\.\d+)?)")

# Material-label keyword groups.  Each group maps a set of label keywords to a
# (material_role_hint, name_hints, cell_role) triple used both to resolve the
# material_id from the catalog and to assign the cell's structural role.
#
# Ordering is significant: water/coolant is checked first because water-gap
# layers are frequently described by what they sit between (e.g. "borated water
# between the Pyrex rod and the guide tube"), so their labels can mention other
# materials.  Helium/gas is next (gas gaps are described cleanly).  The solid
# absorbers and structural alloys follow; poison (Pyrex) is matched only when no
# earlier group matches, so a poison label that happens to mention another
# material is still classified by its primary content.
_MATERIAL_KEYWORD_GROUPS: tuple[tuple[tuple[str, ...], str, tuple[str, ...], str], ...] = (
    (("water", "水", "coolant", "moderator", "含硼", "慢化", "borated"), "coolant", ("water", "coolant", "borated", "moderator"), "inner_flow"),
    (("helium", "氦", "gas", "气体", "气腔", "间隙", "gap", "plenum", "空隙"), "gap", ("helium", "he"), "gas_gap"),
    (("aic", "ag-in-cd", "silver", "silver-indium-cadmium", "吸收体"), "poison", ("aic", "ag", "silver"), "poison"),
    (("b4c", "boron carbide", "碳化硼", "硼碳"), "poison", ("b4c", "boron_carbide", "boron"), "poison"),
    (("zircaloy", "zr-4", "zr4", "zr ", "锆", "guide tube", "导向管"), "cladding", ("zircaloy", "zr"), "cladding"),
    (("ss304", "ss-304", "ss 304", "ss316", "stainless", "不锈钢", "ss ", "ss_"), "structural", ("ss304", "ss-304", "ss 304", "stainless"), "cladding"),
    (("pyrex", "毒物", "poison", "borosilicate", "硼硅", "burnable", "可燃毒物"), "poison", ("pyrex",), "poison"),
)


@dataclass
class _RadialLayerRow:
    r_min: float
    r_max: float | None  # None for an unbounded background region.
    label: str
    is_background: bool


@dataclass
class AnnularInsertOracleProposal:
    """Result of a deterministic annular-insert universe proposal."""

    ok: bool
    universe_data: dict[str, Any] | None = None
    reason: str = ""
    layer_count: int = 0
    source_excerpt: str = ""

    @property
    def universe(self) -> UniverseSpecPatch | None:
        if not self.ok or not self.universe_data:
            return None
        return UniverseSpecPatch.model_validate(self.universe_data)


def _strip_markdown(text: str) -> str:
    return text.replace("`", "").replace("**", "").strip()


def _extract_material_label(line: str, range_cell: str) -> str:
    """Best-effort extraction of the material label associated with a range.

    For a markdown table row the label is the next non-empty cell.  For a
    ``role=rmin-rmax`` note the label is the prefix before ``=``.  Otherwise
    the label is whatever non-range text remains on the line.
    """

    cells = [cell.strip() for cell in line.split("|")]
    cells = [_strip_markdown(c) for c in cells if c.strip()]
    # Drop pure numeric/separator cells and the matched range cell.
    cleaned: list[str] = []
    for cell in cells:
        norm = cell.replace("–", "-").replace("—", "-")
        if _RANGE_RE.match(norm) or _BACKGROUND_RE.match(norm):
            continue
        if cell.strip() in {":", "-", "—", "–", "|"}:
            continue
        cleaned.append(cell)
    if len(cleaned) >= 1:
        return cleaned[0]
    # Semicolon ``label=...`` fallback.
    eq_match = re.match(r"\s*([A-Za-z][\w \-]*?)\s*=", line)
    if eq_match:
        return eq_match.group(1).strip()
    return _strip_markdown(range_cell)


def extract_radial_layers(
    requirement_text: str,
    *,
    min_layers: int = 3,
) -> tuple[list[_RadialLayerRow], str] | None:
    """Parse a concentric radial cross-section from the requirement text.

    Returns the ordered layer rows and the matched source excerpt, or ``None``
    when no coherent radial table/note of at least ``min_layers`` contiguous
    monotonic layers is found.  Accepts both markdown-table form
    (``| 0–0.214 cm | helium |``) and semicolon-note form
    (``helium=0-0.214 cm; ...``).
    """

    if not requirement_text:
        return None
    candidates: list[tuple[list[_RadialLayerRow], str]] = []

    # --- Strategy A: markdown table rows -------------------------------------
    table_rows: list[_RadialLayerRow] = []
    table_excerpt: list[str] = []
    for raw_line in requirement_text.splitlines():
        if "|" not in raw_line:
            if table_rows and len(table_rows) >= min_layers:
                candidates.append((list(table_rows), "\n".join(table_excerpt)))
            table_rows = []
            table_excerpt = []
            continue
        cells = [c.strip() for c in raw_line.split("|")]
        cells = [c for c in cells if c]
        matched = False
        for idx, cell in enumerate(cells):
            norm = cell.replace("–", "-").replace("—", "-")
            rmatch = _RANGE_RE.match(_strip_markdown(norm))
            bmatch = _BACKGROUND_RE.match(_strip_markdown(norm))
            if rmatch and idx + 1 < len(cells):
                rmin, rmax = float(rmatch.group(1)), float(rmatch.group(2))
                if rmax > rmin:
                    label = _strip_markdown(cells[idx + 1])
                    table_rows.append(_RadialLayerRow(rmin, rmax, label, False))
                    table_excerpt.append(raw_line)
                    matched = True
                    break
            if bmatch and idx + 1 < len(cells):
                rmin = float(bmatch.group(1))
                label = _strip_markdown(cells[idx + 1])
                table_rows.append(_RadialLayerRow(rmin, None, label, True))
                table_excerpt.append(raw_line)
                matched = True
                break
        if not matched and table_rows and len(table_rows) >= min_layers:
            candidates.append((list(table_rows), "\n".join(table_excerpt)))
            table_rows = []
            table_excerpt = []
    if table_rows and len(table_rows) >= min_layers:
        candidates.append((list(table_rows), "\n".join(table_excerpt)))

    # --- Strategy B: semicolon ``label=rmin-rmax cm`` notes ------------------
    note_rows: list[_RadialLayerRow] = []
    for note_match in re.finditer(
        r"([A-Za-z][\w \-]*?)\s*=\s*(\d+(?:\.\d+)?)\s*[–—\-]\s*(\d+(?:\.\d+)?)\s*cm",
        requirement_text,
    ):
        label = note_match.group(1).strip()
        rmin, rmax = float(note_match.group(2)), float(note_match.group(3))
        if rmax > rmin:
            note_rows.append(_RadialLayerRow(rmin, rmax, label, False))
    if len(note_rows) >= min_layers:
        candidates.append((note_rows, requirement_text))

    # Pick the candidate whose layers form the most coherent monotonic
    # contiguous radial profile.
    best: tuple[list[_RadialLayerRow], str] | None = None
    best_score = -1.0
    for rows, excerpt in candidates:
        ordered = _select_coherent_block(rows, min_layers=min_layers)
        if not ordered:
            continue
        score = _coherence_score(ordered)
        if score > best_score:
            best_score = score
            best = (ordered, excerpt)
    return best


def _select_coherent_block(
    rows: list[_RadialLayerRow],
    *,
    min_layers: int,
) -> list[_RadialLayerRow] | None:
    """Pick the longest contiguous monotonic radial sub-sequence."""

    if not rows:
        return None
    # Sort bounded layers by r_min, keep background last.
    bounded = sorted((r for r in rows if not r.is_background), key=lambda r: r.r_min)
    backgrounds = [r for r in rows if r.is_background]
    if not bounded:
        return None
    # Greedily extend a contiguous chain.
    chain: list[_RadialLayerRow] = [bounded[0]]
    for row in bounded[1:]:
        prev = chain[-1]
        if row.r_min >= prev.r_min and (prev.r_max is None or abs(row.r_min - prev.r_max) <= _CONTIGUITY_TOL_CM):
            chain.append(row)
        elif row.r_min >= prev.r_max if prev.r_max else False:
            chain.append(row)
    if len(chain) < min_layers:
        return None
    # Attach at most one background region whose r_min >= last bounded r_max.
    if backgrounds:
        tail = chain[-1].r_max if chain[-1].r_max is not None else chain[-1].r_min
        bg = min(backgrounds, key=lambda r: abs(r.r_min - tail))
        if bg.r_min >= tail - _CONTIGUITY_TOL_CM:
            chain.append(bg)
    return chain


def _coherence_score(rows: list[_RadialLayerRow]) -> float:
    bounded = [r for r in rows if not r.is_background]
    if len(bounded) < 2:
        return float(len(rows))
    gaps = 0.0
    for prev, cur in zip(bounded, bounded[1:]):
        if prev.r_max is not None:
            gaps += abs(cur.r_min - prev.r_max)
    return float(len(rows)) - gaps


def _classify_label(label: str) -> tuple[str | None, tuple[str, ...], str]:
    """Return (material_role_hint, name_hints, cell_role) for a material label."""

    lower = label.lower()
    for keywords, role_hint, name_hints, cell_role in _MATERIAL_KEYWORD_GROUPS:
        if any(kw in lower for kw in keywords):
            return role_hint, name_hints, cell_role
    return None, (), "filler"


def _resolve_material_id(label: str, materials: list[Any]) -> str | None:
    """Bind a material label to a material_id in the catalog."""

    role_hint, name_hints, _ = _classify_label(label)
    # Prefer an explicit name/id substring match first (disambiguates materials
    # that share a role, e.g. SS304 vs Inconel both structural).
    for m in materials:
        mid = (getattr(m, "material_id", "") or "").lower()
        mname = (getattr(m, "name", "") or "").lower()
        if name_hints and any(hint in mid or hint in mname for hint in name_hints):
            return getattr(m, "material_id", None)
    if role_hint:
        for m in materials:
            if (getattr(m, "role", "") or "").lower() == role_hint:
                return getattr(m, "material_id", None)
    return None


def _slugify(label: str, fallback: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    return slug or fallback


def build_annular_insert_universe(
    *,
    universe_id: str,
    kind: str,
    rows: list[_RadialLayerRow],
    materials: list[Any],
) -> tuple[UniverseSpecPatch, list[str]] | None:
    """Construct a concentric UniverseSpecPatch from parsed radial layers.

    Returns ``(universe, warnings)`` or ``None`` when a layer cannot be bound
    to a material (fail-closed).  The center cavity (r_min == 0) is rendered
    as a ``cylinder``; every other bounded layer as an ``annulus``; the
    trailing unbounded region as ``background`` coolant.
    """

    cells: list[CellLayerPatch] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    for index, row in enumerate(rows):
        role_hint, name_hints, cell_role = _classify_label(row.label)
        material_id = _resolve_material_id(row.label, materials)
        if material_id is None:
            return None
        if row.is_background:
            region_kind = "background"
            r_min = None
            r_max = None
        elif row.r_min == 0.0:
            region_kind = "cylinder"
            r_min = row.r_min
            r_max = row.r_max
        else:
            region_kind = "annulus"
            r_min = row.r_min
            r_max = row.r_max
        cell_id = _slugify(row.label, fallback=f"layer_{index}")
        if cell_id in seen_ids:
            cell_id = f"{cell_id}_{index}"
        seen_ids.add(cell_id)
        cells.append(CellLayerPatch(
            id=cell_id,
            role=cell_role,
            material_id=material_id,
            region_kind=region_kind,  # type: ignore[arg-type]
            r_min_cm=r_min,
            r_max_cm=r_max,
        ))
    source_note = (
        "Deterministic annular-insert universe constructed from the source "
        "radial cross-section by the annular_insert_universe_oracle."
    )
    universe = UniverseSpecPatch(
        universe_id=universe_id,
        kind=kind,  # type: ignore[arg-type]
        cells=cells,
        source_note=source_note,
        assumptions=[],
        metadata={"constructed_by": "annular_insert_universe_oracle"},
    )
    return universe, warnings


def propose_annular_insert_universe(
    *,
    manifest_item: Any,
    requirement: str,
    materials: list[Any],
) -> AnnularInsertOracleProposal:
    """Propose a deterministic annular-insert universe for a failed fragment.

    ``manifest_item`` must expose ``universe_id``, ``kind`` and
    ``required_layer_roles``.  ``materials`` is the parsed material catalog
    (objects with ``material_id``, ``role``, ``name``).  The proposal is
    ``ok=False`` when the requirement carries no parseable radial structure or
    a layer cannot be bound to the catalog; the caller then falls back to the
    normal fragment-failure path.
    """

    kind = getattr(manifest_item, "kind", "") or ""
    if kind not in _ANNULAR_INSERT_KINDS:
        return AnnularInsertOracleProposal(ok=False, reason=f"kind_not_annular_insert:{kind}")
    parsed = extract_radial_layers(requirement)
    if not parsed:
        return AnnularInsertOracleProposal(ok=False, reason="no_radial_cross_section_found")
    rows, excerpt = parsed
    if len(rows) < 3:
        return AnnularInsertOracleProposal(
            ok=False, reason=f"radial_cross_section_too_short:{len(rows)}",
        )
    built = build_annular_insert_universe(
        universe_id=getattr(manifest_item, "universe_id", "annular_insert"),
        kind=kind,
        rows=rows,
        materials=materials,
    )
    if built is None:
        return AnnularInsertOracleProposal(ok=False, reason="material_binding_failed")
    universe, _warnings = built
    return AnnularInsertOracleProposal(
        ok=True,
        universe_data=universe.model_dump(mode="json"),
        reason="constructed_from_source_radial_cross_section",
        layer_count=len(rows),
        source_excerpt=excerpt,
    )
