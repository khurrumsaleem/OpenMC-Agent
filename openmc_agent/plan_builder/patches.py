"""Patch schemas for incremental plan building (Phase 2).

Each patch type is a self-contained Pydantic model that captures one
*component* of a future SimulationPlan (materials, universes, pin map, axial
layers, ...).  Patches are designed to be:

* **Small** — each patch is a few KB, not a 25 KB monolithic JSON.
* **Independently validatable** — a validator can check a patch in isolation
  (plus lightweight cross-references via :class:`PatchValidationContext`).
* **Assemblable** — a future deterministic assembler (Phase 3) will merge
  validated patches into a complete ``SimulationPlan``.

No OpenMC, no renderer, no LLM dependencies.
"""

from __future__ import annotations

import json
from typing import Any, Literal, get_origin

from pydantic import BaseModel, ConfigDict, Field, model_validator

from openmc_agent.schemas import AgentBaseModel, CoreBoundarySpec
from openmc_agent.plan_builder.control_state_normalization import (
    canonicalize_control_state_id,
)


PatchType = Literal[
    "facts",
    "materials",
    "universes",
    "localized_insert_profiles",
    "base_path_axial_profiles",
    "pin_map",
    "axial_layers",
    "axial_overlays",
    "settings",
    "assembly_catalog",
    "core_layout",
]

ModelScope = Literal[
    "single_pin",
    "single_assembly",
    "multi_assembly_core",
    "full_core",
    "unknown",
]

CountScope = Literal[
    "pin_cell",
    "pin_map",
    "assembly_type",
    "assembly_instance",
    "core_total",
    "unknown",
]


class _PatchBase(AgentBaseModel):
    """Common base for all patch models (extra='forbid', strip whitespace)."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _coerce_null_collections(cls, data: Any) -> Any:
        """Tolerate LLMs that emit ``null`` for optional list/dict fields.

        A pydantic ``default_factory=list`` only applies when a field is
        *absent*; an explicit ``null`` is rejected (``list`` is not
        ``Optional[list]``). LLMs routinely write ``null`` for list/dict
        fields they leave empty (e.g. ``"loading_ids": null``), so coerce
        those to empty collections before validation. Reactor-neutral.
        """
        if not isinstance(data, dict):
            return data
        for fname, finfo in cls.model_fields.items():
            if data.get(fname) is not None:
                continue
            if fname not in data:
                continue
            origin = get_origin(finfo.annotation) or finfo.annotation
            if origin is list:
                data[fname] = []
            elif origin is dict:
                data[fname] = {}
        return data

    @model_validator(mode="before")
    @classmethod
    def _coerce_assumption_items(cls, data: Any) -> Any:
        """Normalize LLM-emitted structured assumptions to strings.

        Patch schemas deliberately keep ``assumptions`` lightweight:
        they are audit notes, not typed contracts.  Real reviewers and
        patch generators sometimes emit objects such as
        ``{"assumption_id": "...", "statement": "..."}``; rejecting the
        whole patch for that recoverable shape wastes a retry and can push
        large downstream patches into truncation.  Preserve the information
        as a compact string while keeping the public schema unchanged.
        """
        if not isinstance(data, dict) or "assumptions" not in data:
            return data
        raw = data.get("assumptions")
        if raw is None:
            data["assumptions"] = []
            return data
        if not isinstance(raw, list):
            data["assumptions"] = [_stringify_assumption(raw)]
            return data
        data["assumptions"] = [_stringify_assumption(item) for item in raw]
        return data


def _stringify_assumption(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        ident = item.get("assumption_id") or item.get("id") or item.get("key")
        text = (
            item.get("statement")
            or item.get("assumption")
            or item.get("note")
            or item.get("message")
            or item.get("description")
            or item.get("rationale")
        )
        if ident and text:
            return f"{ident}: {text}"
        if text:
            return str(text)
        if ident:
            return str(ident)
        return json.dumps(item, sort_keys=True, ensure_ascii=False)
    return str(item)


# ---------------------------------------------------------------------------
# Scope Contracts (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


class ScopedExpectedCount(_PatchBase):
    """An expected count with explicit scope binding.

    Replaces the old un-scoped ``expected_pin_count`` etc. for
    multi-assembly / full-core models.  Each count is explicitly tied to a
    scope (pin_map, assembly_type, core_total, ...) so that the validator
    only compares counts at the same scope level.
    """

    role: str
    value: int
    scope: CountScope = "unknown"
    assembly_type_id: str | None = None
    assembly_instance_id: str | None = None
    source_note: str | None = None
    provenance_refs: list[str] = Field(default_factory=list)
    derived: bool = False
    derivation: str | None = None
    requires_human_confirmation: bool = False


class FuelVariantRequirementPatchItem(_PatchBase):
    """A distinct fuel variant required by the source document.

    Reactor-neutral: each entry records one fuel enrichment / composition
    variant and which assembly types use it.  This drives the downstream
    material → universe → assembly-type binding contract.
    """

    variant_id: str
    source_label: str | None = None
    enrichment_wt_percent: float | None = None
    density_g_cm3: float | None = None
    assembly_type_ids: list[str] = Field(default_factory=list)
    expected_assembly_count: int | None = None
    source_note: str | None = None


class LocalizedInsertPlacementRequirementPatchItem(_PatchBase):
    """Source-driven placement requirement for a localized insert.

    Declares that the source document requires a specific kind of
    localized insert (e.g., control rod) to be physically placed in
    certain assembly types.  This is distinct from *defining* universes
    or profiles — it is a source-level contract that downstream patches
    (assembly_catalog intents, profiles, universes) must fulfill.

    Reactor-neutral: no reactor-specific names, values, or identifiers
    are hardcoded.  All specifics come from the source document.
    """

    requirement_id: str
    insert_kind: Literal[
        "pyrex_rod",
        "thimble_plug",
        "control_rod",
        "absorber_insert",
        "instrumentation_insert",
        "custom",
    ]
    assembly_type_ids: list[str] = Field(default_factory=list)
    expected_coordinate_count_per_assembly: int | None = None
    expected_assembly_instance_count: int | None = None
    host_kind: Literal["guide_tube", "instrument_tube", "custom"] = "guide_tube"
    required_profile_id: str | None = None
    required_segment_roles: list[str] = Field(default_factory=list)
    expected_insert_universe_ids: list[str] = Field(default_factory=list)
    anchor_z_cm: float | None = None
    control_state_id: str | None = None
    required_in_detailed_domain: bool = True
    source_note: str | None = None
    requires_human_confirmation: bool = False

    @model_validator(mode="before")
    @classmethod
    def _normalize_control_state_id(cls, data: Any) -> Any:
        if isinstance(data, dict) and "control_state_id" in data:
            data["control_state_id"] = canonicalize_control_state_id(
                data.get("control_state_id")
            )
        return data


# ---------------------------------------------------------------------------
# FactsPatch
# ---------------------------------------------------------------------------


class FactsPatch(_PatchBase):
    """Extracted benchmark / geometry / variant facts.

    Does NOT generate any OpenMC plan content; only records what the
    requirement / benchmark document tells us about the problem shape.
    """

    patch_type: Literal["facts"] = "facts"
    benchmark_id: str | None = None
    selected_variant: str | None = None

    geometry_type: str | None = None
    lattice_size: tuple[int, int] | None = None
    pin_pitch_cm: float | None = None
    assembly_pitch_cm: float | None = None

    # Tri-state feature flags.  ``None`` is the LLM-omitted state and must
    # never be coerced to ``False`` by the schema — that would silently
    # downgrade multi-assembly contracts.  Source evidence or the planning
    # feature contract must promote the value to ``True``/``False``.
    has_axial_geometry: bool | None = None
    has_spacer_grids: bool | None = None
    has_special_pin_map: bool | None = None

    active_fuel_region_cm: tuple[float, float] | None = None
    axial_domain_cm: tuple[float, float] | None = None

    expected_spacer_grid_count: int | None = None
    expected_pin_count: int | None = None
    expected_guide_tube_count: int | None = None
    expected_instrument_tube_count: int | None = None
    expected_pyrex_count: int | None = None
    expected_thimble_plug_count: int | None = None

    # --- P2-FULLCORE-1: Scope-aware fields ---
    # ``unknown`` is the safe default.  The previous default
    # ``"single_assembly"`` silently forced the entire single-assembly patch
    # family whenever the LLM omitted the field, which contaminated every
    # multi-assembly benchmark.  Only source evidence or the planning feature
    # contract may promote the scope to a concrete value.
    model_scope: ModelScope = "unknown"
    assembly_count: int | None = None
    core_lattice_size: tuple[int, int] | None = None
    assembly_type_counts: dict[str, int] = Field(default_factory=dict)
    scoped_expected_counts: list[ScopedExpectedCount] = Field(default_factory=list)
    boundary_scope: str | None = None
    symmetry_description: str | None = None

    material_roles: list[str] = Field(default_factory=list)
    fuel_variant_requirements: list[FuelVariantRequirementPatchItem] = Field(
        default_factory=list,
        description="Distinct fuel variants required by the source document.",
    )
    localized_insert_requirements: list[LocalizedInsertPlacementRequirementPatchItem] = Field(
        default_factory=list,
        description="Source-driven placement requirements for localized inserts (e.g., control rods).",
    )
    missing_facts: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_notes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# MaterialsPatch
# ---------------------------------------------------------------------------


class MixtureComponentPatch(_PatchBase):
    """A single component in a homogenized material mixture."""

    material_id: str
    volume_fraction: float = Field(gt=0, le=1.0)


class CompoundComponentPatchItem(_PatchBase):
    """A source-level chemical compound; never a transport nuclide entry."""

    formula: str
    fraction: float
    fraction_basis: Literal["weight_frac", "atom_frac"] | None = None
    isotope_policy: Literal[
        "natural_elements", "explicit_isotopes", "requires_confirmation",
    ] | None = None
    isotope_overrides: dict[str, dict[str, float]] = Field(default_factory=dict)
    source_note: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class MaterialSpecPatch(_PatchBase):
    """A single material entry in a MaterialsPatch."""

    material_id: str
    name: str
    role: str
    density_g_cm3: float | None = None
    # Density provenance is intentionally independent from transport
    # composition confidence.  A structural density can be source-provided or
    # library-provided while alloy composition still needs confirmation.
    density_status: Literal[
        "confirmed", "source_provided", "library", "approximate", "needs_confirmation",
    ] = "needs_confirmation"
    density_source: str | None = None
    density_requires_human_confirmation: bool = False
    temperature_K: float | None = None
    # Transport-ready element or nuclide names only. Source formulae belong in
    # compound_components and are deterministically resolved before rendering.
    composition: dict[str, float] = Field(default_factory=dict)
    compound_components: list[CompoundComponentPatchItem] = Field(default_factory=list)
    composition_basis: Literal[
        "atom_frac",
        "weight_frac",
        "atom_density_barn_cm",
        "stoichiometric_ratio",
        "ppm_by_weight",
        "ppm_by_atom",
        "unknown",
    ] = "unknown"
    composition_status: Literal[
        "confirmed",
        "approximate",
        "needs_library",
        "needs_confirmation",
        "placeholder",
        "derived_from_mixture",
    ] = "needs_confirmation"
    source_note: str | None = None
    warnings: list[str] = Field(default_factory=list)
    mixture_components: list[MixtureComponentPatch] = Field(default_factory=list)
    source_variant_id: str | None = None
    variant_scope: str | None = None
    derivation_method: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce_mixture_nulls(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        has_mixture = bool(data.get("mixture_components"))
        if has_mixture:
            if data.get("composition") is None:
                data["composition"] = {}
            if data.get("composition_basis") is None:
                data["composition_basis"] = "unknown"
        return data


class MaterialsPatch(_PatchBase):
    """Material catalog patch."""

    patch_type: Literal["materials"] = "materials"
    materials: list[MaterialSpecPatch]
    assumptions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# UniversesPatch
# ---------------------------------------------------------------------------


class CellLayerPatch(_PatchBase):
    """A concentric cell layer inside a universe."""

    id: str
    role: str
    material_id: str | None = None
    region_kind: Literal[
        "cylinder",
        "annulus",
        "box",
        "square_frame",
        "background",
        "unknown",
    ] = "unknown"
    r_min_cm: float | None = None
    r_max_cm: float | None = None
    outer_side_cm: float | None = None
    inner_side_cm: float | None = None
    fill_universe_id: str | None = None
    protected_through_path: bool = False


class UniverseSpecPatch(_PatchBase):
    """A single universe definition (fuel pin, guide tube, pyrex rod, ...)."""

    universe_id: str
    kind: Literal[
        "fuel_pin",
        "guide_tube",
        "instrument_tube",
        "pyrex_rod",
        "thimble_plug",
        "control_rod",
        "water_cell",
        "custom",
    ]
    cells: list[CellLayerPatch]
    source_note: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UniversesPatch(_PatchBase):
    """Universe catalog patch."""

    patch_type: Literal["universes"] = "universes"
    universes: list[UniverseSpecPatch]


# ---------------------------------------------------------------------------
# PinMapPatch
# ---------------------------------------------------------------------------


class CoordinateConvention(_PatchBase):
    """Coordinate system convention for pin map positions."""

    index_base: Literal[0, 1] = 0
    row_origin: Literal["top", "bottom", "unknown"] = "top"
    col_origin: Literal["left", "right", "unknown"] = "left"
    ordering: Literal["row_col", "x_y", "unknown"] = "row_col"


class LocalizedInsertIntentPatchItem(_PatchBase):
    """Declares a finite axial insert that replaces part of a host path
    within a specific z interval.

    Unlike base pin-map groups (guide_tube_coords, etc.) which persist for
    the full assembly height, a localized insert only affects its declared
    z range.  Outside that range the host path keeps its base universe.

    Examples: Pyrex poison rods, thimble plugs, control-rod inserts,
    discrete absorbers.
    """

    insert_id: str
    insert_kind: Literal[
        "pyrex_rod",
        "thimble_plug",
        "absorber_insert",
        "control_rod",
        "instrumentation_insert",
        "custom",
    ]
    host_kind: Literal["guide_tube", "instrument_tube", "custom"] = "guide_tube"
    host_universe_id: str | None = None
    insert_universe_id: str

    coordinates: list[tuple[int, int]] = Field(default_factory=list)

    z_min_cm: float | None = None
    z_max_cm: float | None = None

    application_mode: Literal[
        "nested_component_override",
        "coordinate_override",
    ] = "nested_component_override"

    component_role: str | None = None
    component_path_id: str | None = None
    preserve_component_roles: list[str] = Field(default_factory=list)
    preserve_path_ids: list[str] = Field(default_factory=list)

    priority: int = 0

    source_note: str | None = None
    requires_human_confirmation: bool = False
    assumptions: list[str] = Field(default_factory=list)

    # P2-FULLCORE-2B: multi-segment axial profile support (e.g., RCCA)
    axial_profile_id: str | None = None
    anchor_z_cm: float | None = None
    control_state_id: str | None = None


class LocalizedInsertAxialSegmentPatchItem(_PatchBase):
    """A single axial segment within a multi-segment insert profile (P2-FULLCORE-2B)."""

    segment_id: str
    relative_z_min_cm: float
    relative_z_max_cm: float
    universe_id: str
    role: str = ""
    material_role: str | None = None
    source_note: str | None = None


class LocalizedInsertAxialProfilePatchItem(_PatchBase):
    """A reusable axial profile for multi-segment localized inserts (P2-FULLCORE-2B)."""

    profile_id: str
    anchor_kind: Literal["absolute", "bottom", "top", "center"] = "absolute"
    anchor_z_cm: float | None = None
    segments: list[LocalizedInsertAxialSegmentPatchItem] = Field(default_factory=list)
    source_note: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    requires_human_confirmation: bool = False


class PinMapPatch(_PatchBase):
    """Pin map replacement rules (not a full expanded lattice).

    Instead of a 17×17 = 289-entry ``universe_pattern``, this patch only lists
    the *special* positions (guide tubes, instrument tubes, ...).  The default
    universe fills everything else.  The assembler (Phase 3) will expand this
    into the full lattice.

    **Base path groups** (``guide_tube_coords``, ``instrument_tube_coords``,
    ``water_cell_coords``) define persistent paths that exist for the full
    assembly height.

    **Localized insert intents** (``localized_insert_intents``) declare
    finite-height inserts (Pyrex, thimble plugs, absorbers) that only affect
    a specific z range within a host path.  Their coordinates must be a subset
    of a base path group (typically guide_tube_coords).

    Legacy fields ``pyrex_rod_coords`` and ``thimble_plug_coords`` are kept
    for backward compatibility but should not be used for new Lane B output.
    Use ``localized_insert_intents`` instead.
    """

    patch_type: Literal["pin_map"] = "pin_map"
    variant: str | None = None
    lattice_size: tuple[int, int]
    default_universe_id: str
    coordinate_convention: CoordinateConvention = Field(
        default_factory=CoordinateConvention
    )

    # --- Base path groups (persistent, full-height) ---
    guide_tube_coords: list[tuple[int, int]] = Field(default_factory=list)
    instrument_tube_coords: list[tuple[int, int]] = Field(default_factory=list)
    water_cell_coords: list[tuple[int, int]] = Field(default_factory=list)

    # --- Localized insert intents (finite-height, within a host path) ---
    localized_insert_intents: list[LocalizedInsertIntentPatchItem] = Field(
        default_factory=list,
        description="Finite axial inserts (Pyrex, thimble plugs, etc.). "
        "Coordinates must be a subset of a base path group.",
    )

    # --- Legacy fields (backward compat, not for new Lane B output) ---
    pyrex_rod_coords: list[tuple[int, int]] = Field(
        default_factory=list,
        description="[LEGACY] Use localized_insert_intents with insert_kind='pyrex_rod' instead.",
    )
    thimble_plug_coords: list[tuple[int, int]] = Field(
        default_factory=list,
        description="[LEGACY] Use localized_insert_intents with insert_kind='thimble_plug' instead.",
    )

    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


# ---------------------------------------------------------------------------
# AxialLayersPatch
# ---------------------------------------------------------------------------


class AxialLayerPatchItem(_PatchBase):
    """A single axial layer in an AxialLayersPatch."""

    layer_id: str
    role: Literal[
        "lower_moderator_buffer",
        "lower_core_plate",
        "lower_nozzle",
        "lower_shoulder_gap",
        "lower_fuel_endplug",
        "lower_end_plug",
        "lower_plenum",
        "active_fuel",
        "gas_gap",
        "upper_fuel_endplug",
        "upper_end_plug",
        "fuel_upper_plenum",
        "upper_plenum",
        "upper_shoulder_gap",
        "upper_nozzle",
        "upper_core_plate",
        "upper_moderator_buffer",
        "core_plate",
        "reflector",
        "shoulder_gap",
        "lower_shoulder_gap",
        "upper_shoulder_gap",
        "custom",
    ] = "custom"
    z_min_cm: float | None = None
    z_max_cm: float | None = None
    fill_type: Literal["lattice", "material", "universe", "void", "unknown"] = "unknown"
    fill_id: str | None = None
    loading_id: str | None = None
    loading_ids: list[str] = Field(default_factory=list)
    requires_human_confirmation: bool = False
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


class LatticeTransformationPatchItem(_PatchBase):
    """A single composable lattice transformation in a patch."""

    operation_id: str
    operation_kind: Literal[
        "replace_universe_family",
        "coordinate_override",
        "nested_component_override",
    ]
    replacement_universe_id: str

    source_universe_id: str | None = None
    source_universe_ids: list[str] = Field(default_factory=list)
    target_coordinates: list[tuple[int, int]] = Field(default_factory=list)

    component_role: str | None = None
    component_path_id: str | None = None
    preserve_component_roles: list[str] = Field(default_factory=list)
    preserve_path_ids: list[str] = Field(default_factory=list)

    priority: int = 0
    purpose: str = ""


class LatticeLoadingPatchItem(_PatchBase):
    """A per-axial-layer loading applied to an existing lattice."""

    loading_id: str
    base_lattice_id: str
    derived_lattice_id: str | None = None
    transformations: list[LatticeTransformationPatchItem] = Field(default_factory=list)
    overrides: dict[str, list[tuple[int, int]]] = Field(default_factory=dict)
    purpose: str = ""


class AxialLayersPatch(_PatchBase):
    """Axial layer segmentation patch."""

    patch_type: Literal["axial_layers"] = "axial_layers"
    layers: list[AxialLayerPatchItem]
    axial_domain_cm: tuple[float, float] | None = None
    lattice_loadings: list[LatticeLoadingPatchItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AxialOverlaysPatch
# ---------------------------------------------------------------------------


class AxialOverlayPatchItem(_PatchBase):
    """A single axial overlay (e.g. spacer grid) in an AxialOverlaysPatch."""

    overlay_id: str
    overlay_kind: Literal[
        "spacer_grid",
        "support_plate",
        "absorber_insert",
        "custom",
    ]
    z_min_cm: float | None = None
    z_max_cm: float | None = None
    target_lattice_id: str | None = None
    material_id: str | None = None
    geometry_mode: Literal[
        "skeleton",
        "homogenized_open_region",
        "mass_conserving_outer_frame",
        "annular_shell",
        "explicit_bars",
        "volume_fraction_calibrated",
    ] = "skeleton"
    through_path_preserved: bool | None = None
    volume_fraction: float | None = None
    effective_density_g_cm3: float | None = None
    total_mass_g: float | None = None
    cell_count: int | None = None
    pitch_cm: float | None = None
    material_density_source: str | None = None
    frame_area_cm2: float | None = None
    frame_thickness_cm: float | None = None
    mass_tolerance_rel: float = 1e-6
    requires_human_confirmation: bool = False
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None

    @model_validator(mode="after")
    def _derive_through_path_preserved(self) -> "AxialOverlayPatchItem":
        """Deterministically derive through_path_preserved=True when the field
        is absent (None) for modes that inherently preserve through-paths.

        An explicit ``False`` is left untouched — it is a semantic
        contradiction caught by the validator as
        ``patch.axial_overlays.mode_semantic_contradiction``.

        Protected paths (fuel pellet, gap, cladding, guide-tube wall,
        instrument-tube wall, Pyrex, thimble plug, RCCA, and necessary
        coolant paths) remain because these geometry modes only replace the
        outer moderator frame or open region."""
        if (
            self.geometry_mode
            in ("mass_conserving_outer_frame", "homogenized_open_region")
            and self.through_path_preserved is None
        ):
            self.through_path_preserved = True
        return self


class AxialOverlaysPatch(_PatchBase):
    """Axial overlays patch (spacer grids, support plates, ...)."""

    patch_type: Literal["axial_overlays"] = "axial_overlays"
    overlays: list[AxialOverlayPatchItem]


# ---------------------------------------------------------------------------
# SettingsPatch
# ---------------------------------------------------------------------------


class SettingsPatch(_PatchBase):
    """Execution settings strategy patch."""

    patch_type: Literal["settings"] = "settings"
    source_strategy: Literal[
        "active_fuel_box",
        "assembly_box",
        "manual",
        "unknown",
    ] = "active_fuel_box"
    source_requires_fissionable_constraint: bool = True
    manual_source_bounds_cm: list[float] | None = None
    plot_strategy: Literal[
        "full_assembly",
        "quarter_assembly",
        "manual",
        "none",
    ] = "full_assembly"
    cross_sections_runtime_required: bool = True
    tallies_required_for_smoke_test: bool = False
    assumptions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AssemblyCatalogPatch (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


class AssemblyPinMapPatchItem(_PatchBase):
    """Sparse pin map for a single assembly type.

    Reuses the same structure as PinMapPatch but is scoped to an assembly
    type template rather than the top-level plan.
    """

    lattice_size: tuple[int, int]
    default_universe_id: str
    coordinate_convention: CoordinateConvention = Field(
        default_factory=CoordinateConvention
    )
    guide_tube_coords: list[tuple[int, int]] = Field(default_factory=list)
    instrument_tube_coords: list[tuple[int, int]] = Field(default_factory=list)
    water_cell_coords: list[tuple[int, int]] = Field(default_factory=list)
    localized_insert_intents: list[LocalizedInsertIntentPatchItem] = Field(
        default_factory=list,
    )
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


class AssemblyTypePatchItem(_PatchBase):
    """A single assembly type template in an AssemblyCatalogPatch."""

    assembly_type_id: str
    name: str = ""
    role: str = ""
    fuel_variant_id: str | None = None
    multiplicity_hint: int | None = None
    pin_map: AssemblyPinMapPatchItem
    axial_profile_id: str | None = None
    base_path_profile_id: str | None = None
    overlay_set_id: str | None = None
    source_note: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    requires_human_confirmation: bool = False


class AssemblyCatalogPatch(_PatchBase):
    """Catalog of assembly type templates for multi-assembly core models.

    Each entry defines one assembly type's sparse pin map and localized
    inserts.  The core_layout patch references these type IDs for placement.
    """

    patch_type: Literal["assembly_catalog"] = "assembly_catalog"
    assembly_types: list[AssemblyTypePatchItem]
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


# ---------------------------------------------------------------------------
# LocalizedInsertProfilesPatch (P2-FULLCORE-2C-A)
# ---------------------------------------------------------------------------


class LocalizedInsertProfilesPatch(_PatchBase):
    """Registry of reusable axial profiles for multi-segment inserts.

    Each profile defines a relative axial segmentation (e.g., absorber +
    plenum + end-plug for a control rod).  Profiles are referenced by
    ``axial_profile_id`` from :class:`LocalizedInsertIntentPatchItem`.

    The actual insert *position* (anchor) is provided by the intent, not
    the profile.  This separation allows the same profile to be reused
    across different rod positions or control states.
    """

    patch_type: Literal["localized_insert_profiles"] = "localized_insert_profiles"
    profiles: list[LocalizedInsertAxialProfilePatchItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


# ---------------------------------------------------------------------------
# BasePathAxialProfilesPatch (P2-FULLCORE-2D-A-HARDENING)
# ---------------------------------------------------------------------------


class BasePathStateBindingPatchItem(_PatchBase):
    """Maps one axial role to a universe replacement for fuel-path switching.

    When a segment's ``base_role`` matches ``axial_role``, the materializer
    replaces ``source_universe_ids`` with ``replacement_universe_id`` in the
    derived pin lattice, preserving paths listed in ``preserve_path_roles``.
    """

    axial_role: str
    source_universe_family: str | None = None
    source_universe_ids: list[str] = Field(default_factory=list)
    replacement_universe_id: str
    assembly_type_ids: list[str] = Field(default_factory=list)
    preserve_path_roles: list[str] = Field(default_factory=list)
    priority: int = 0


class BasePathAxialProfilePatchItem(_PatchBase):
    """A profile mapping axial roles to state bindings for one path family."""

    profile_id: str
    path_family: str = "fuel_rod"
    state_bindings: list[BasePathStateBindingPatchItem] = Field(default_factory=list)
    source_note: str | None = None
    assumptions: list[str] = Field(default_factory=list)


class BasePathAxialProfilesPatch(_PatchBase):
    """Registry of base path axial profiles for per-segment fuel-state switching."""

    patch_type: Literal["base_path_axial_profiles"] = "base_path_axial_profiles"
    profiles: list[BasePathAxialProfilePatchItem] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None


# ---------------------------------------------------------------------------
# CoreLayoutPatch (P2-FULLCORE-1)
# ---------------------------------------------------------------------------


class CoreLayoutPatch(_PatchBase):
    """Core-level assembly placement patch.

    Defines a 2-D pattern of assembly type IDs that references the
    AssemblyCatalogPatch.  The assembler uses this to build a second-level
    core lattice.
    """

    patch_type: Literal["core_layout"] = "core_layout"
    core_lattice_id: str = "core_lattice"
    shape: tuple[int, int]
    assembly_pitch_cm: float | None = None
    coordinate_convention: CoordinateConvention = Field(
        default_factory=CoordinateConvention
    )
    assembly_pattern: list[list[str]]
    outer_assembly_type_id: str | None = None
    boundary: str = "vacuum"
    boundary_conditions: CoreBoundarySpec | None = Field(
        default=None,
        description=(
            "Optional explicit six-face core boundary conditions. Use this when "
            "radial and axial boundaries differ, e.g. radial reflective with "
            "axial vacuum."
        ),
    )
    expected_assembly_type_counts: dict[str, int] = Field(default_factory=dict)
    symmetry_description: str | None = None
    assumptions: list[str] = Field(default_factory=list)
    source_note: str | None = None
    requires_human_confirmation: bool = False


# ---------------------------------------------------------------------------
# Patch parsing helpers
# ---------------------------------------------------------------------------

_PATCH_MODELS: dict[str, type[BaseModel]] = {
    "facts": FactsPatch,
    "materials": MaterialsPatch,
    "universes": UniversesPatch,
    "localized_insert_profiles": LocalizedInsertProfilesPatch,
    "base_path_axial_profiles": BasePathAxialProfilesPatch,
    "pin_map": PinMapPatch,
    "axial_layers": AxialLayersPatch,
    "axial_overlays": AxialOverlaysPatch,
    "settings": SettingsPatch,
    "assembly_catalog": AssemblyCatalogPatch,
    "core_layout": CoreLayoutPatch,
}


class PatchParseError(Exception):
    """Raised when a patch content dict cannot be parsed into its schema."""

    def __init__(self, patch_type: str, message: str, details: Any = None) -> None:
        self.patch_type = patch_type
        self.details = details
        super().__init__(f"[{patch_type}] {message}")


def parse_patch_content(patch_type: str, content: dict[str, Any]) -> BaseModel:
    """Parse a content dict into the appropriate patch model.

    Parameters
    ----------
    patch_type
        One of the :data:`PatchType` literal values.
    content
        Raw content dict (e.g. from ``PlanPatchEnvelope.content``).

    Returns
    -------
    BaseModel
        The parsed patch model.

    Raises
    ------
    PatchParseError
        If ``patch_type`` is unknown or the content fails schema validation.
    """
    model_cls = _PATCH_MODELS.get(patch_type)
    if model_cls is None:
        raise PatchParseError(patch_type, f"unknown patch_type {patch_type!r}")
    try:
        return model_cls.model_validate(content)
    except Exception as exc:
        raise PatchParseError(
            patch_type,
            f"content does not match {model_cls.__name__} schema: {exc}",
            details=str(exc),
        ) from exc


def parse_patch_envelope(envelope: Any) -> BaseModel:
    """Parse a :class:`~openmc_agent.plan_builder.state.PlanPatchEnvelope`-like object.

    Accepts a ``PlanPatchEnvelope`` instance or a dict with ``patch_type`` and
    ``content`` keys.
    """
    patch_type: str | None = None
    content: dict[str, Any] | None = None
    if hasattr(envelope, "patch_type") and hasattr(envelope, "content"):
        patch_type = envelope.patch_type
        content = envelope.content
    elif isinstance(envelope, dict):
        patch_type = envelope.get("patch_type")
        content = envelope.get("content")
    if patch_type is None or content is None:
        raise PatchParseError(
            str(patch_type or "unknown"),
            "envelope must have patch_type and content",
        )
    return parse_patch_content(patch_type, content)


def normalized_coords(
    coords: list[tuple[int, int]],
    convention: CoordinateConvention,
    lattice_size: tuple[int, int],
) -> list[tuple[int, int]]:
    """Normalize coordinates to 0-indexed (row, col) within lattice bounds.

    Does not validate bounds; callers should check the output against
    ``lattice_size`` separately.
    """
    nx, ny = lattice_size
    result: list[tuple[int, int]] = []
    for row, col in coords:
        r = row - convention.index_base
        c = col - convention.index_base
        result.append((r, c))
    return result


# ---------------------------------------------------------------------------
# Schema hint helpers (Phase 7C)
# ---------------------------------------------------------------------------

# SimulationPlan-only fields that should NEVER appear in a patch.
_PLAN_ONLY_FIELDS: frozenset[str] = frozenset({
    "complex_model", "capability_report", "execution_check",
    "plot_specs", "schema_version", "model_spec",
    "expert_assumptions", "expert_feedback",
    # complex_model sub-fields that indicate full-plan output.
    "core", "surfaces", "regions", "assemblies",
    "reflectors", "control_rods", "trisos", "pebbles",
    "packed_spheres",
})

# Pin-map-only forbidden fields (full lattice expansion markers).
_PIN_MAP_FORBIDDEN_FIELDS: frozenset[str] = frozenset({
    "universe_pattern", "full_map", "lattice_map", "rows",
})


def get_patch_allowed_top_level_keys(patch_type: str) -> set[str]:
    """Return the set of allowed top-level JSON keys for a patch type."""
    model_cls = _PATCH_MODELS.get(patch_type)
    if model_cls is None:
        return set()
    return set(model_cls.model_fields.keys())


def get_patch_forbidden_top_level_keys(patch_type: str) -> set[str]:
    """Return keys that are forbidden in this patch type's top-level JSON.

    Includes SimulationPlan-only fields and (for pin_map) full-lattice markers.
    """
    forbidden = set(_PLAN_ONLY_FIELDS)
    if patch_type == "pin_map":
        forbidden |= _PIN_MAP_FORBIDDEN_FIELDS
    return forbidden


def get_patch_json_schema(patch_type: str) -> dict[str, Any]:
    """Return the JSON schema for a patch type's Pydantic model."""
    model_cls = _PATCH_MODELS.get(patch_type)
    if model_cls is None:
        return {}
    return model_cls.model_json_schema()


__all__ = [
    "PatchType",
    "ModelScope",
    "CountScope",
    "ScopedExpectedCount",
    "FactsPatch",
    "MaterialSpecPatch",
    "MaterialsPatch",
    "CellLayerPatch",
    "UniverseSpecPatch",
    "UniversesPatch",
    "CoordinateConvention",
    "LocalizedInsertIntentPatchItem",
    "LocalizedInsertAxialSegmentPatchItem",
    "LocalizedInsertAxialProfilePatchItem",
    "LocalizedInsertProfilesPatch",
    "BasePathStateBindingPatchItem",
    "BasePathAxialProfilePatchItem",
    "BasePathAxialProfilesPatch",
    "PinMapPatch",
    "AssemblyPinMapPatchItem",
    "AssemblyTypePatchItem",
    "AssemblyCatalogPatch",
    "CoreLayoutPatch",
    "AxialLayerPatchItem",
    "LatticeTransformationPatchItem",
    "LatticeLoadingPatchItem",
    "AxialLayersPatch",
    "AxialOverlayPatchItem",
    "AxialOverlaysPatch",
    "SettingsPatch",
    "PatchParseError",
    "parse_patch_content",
    "parse_patch_envelope",
    "normalized_coords",
    "get_patch_allowed_top_level_keys",
    "get_patch_forbidden_top_level_keys",
    "get_patch_json_schema",
]
