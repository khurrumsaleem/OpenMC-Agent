"""Per-patch-type prompt builders for incremental plan building (Phase 4/7B).

Each builder produces a self-contained prompt that asks the LLM for **exactly
one** patch — never a full SimulationPlan, never a full lattice pattern.

Phase 7B hardening: added CRITICAL OUTPUT CONTRACT, minimal per-patch examples,
and stronger prohibition language to prevent real LLMs from returning full
plans instead of small patches.
"""

from __future__ import annotations

import json
from typing import Any

from openmc_agent.few_shot_cases import load_patch_few_shots

from .patches import (
    get_patch_allowed_top_level_keys,
    get_patch_forbidden_top_level_keys,
)


# ---------------------------------------------------------------------------
# Critical output contract (injected into every patch prompt)
# ---------------------------------------------------------------------------

_OUTPUT_CONTRACT = """\
CRITICAL OUTPUT CONTRACT:
- You are NOT generating a SimulationPlan.
- You are NOT generating OpenMC code or XML files.
- You are NOT generating a full reactor model.
- You are generating exactly ONE JSON object for patch_type="{patch_type}".
- The top-level JSON key "patch_type" MUST equal "{patch_type}".
- Any response containing "complex_model", "simulation_plan", "core", \
"axial_layers", "axial_overlays", "universes", "lattices", "capability_report", \
"execution_check", or "plot_specs" OUTSIDE the requested patch schema will be \
REJECTED immediately.
- Return JSON only. No markdown. No prose. No code fences. No explanation."""


# ---------------------------------------------------------------------------
# Per-patch-type rules + minimal examples
# ---------------------------------------------------------------------------

_PATCH_RULES: dict[str, str] = {
    "facts": """\
Requested patch type: facts
Schema fields: benchmark_id, selected_variant, geometry_type, lattice_size [int,int],
  pin_pitch_cm, assembly_pitch_cm, has_axial_geometry, has_spacer_grids,
  has_special_pin_map, active_fuel_region_cm [float,float], axial_domain_cm [float,float],
  expected_spacer_grid_count, expected_pin_count, expected_guide_tube_count,
  expected_instrument_tube_count, expected_pyrex_count, expected_thimble_plug_count,
  model_scope (single_pin|single_assembly|multi_assembly_core|full_core|unknown),
  assembly_count, core_lattice_size [int,int], assembly_type_counts {{type_id: count}},
  scoped_expected_counts [list of {{role, value, scope, assembly_type_id}}],
  boundary_scope, symmetry_description,
  material_roles [list[str]],
  fuel_variant_requirements [list of {{"variant_id", "source_label", "enrichment_wt_percent",
    "density_g_cm3", "assembly_type_ids", "expected_assembly_count", "source_note"}}],
  localized_insert_requirements [list of {{"requirement_id", "insert_kind",
    "assembly_type_ids", "expected_coordinate_count_per_assembly",
    "expected_assembly_instance_count", "host_kind", "required_profile_id",
    "required_segment_roles", "expected_insert_universe_ids",
    "anchor_z_cm", "control_state_id", "required_in_detailed_domain",
    "source_note", "requires_human_confirmation"}}],
  missing_facts [list[str]], assumptions [list[str]],
  source_notes [list[str]].

Rules:
- Extract benchmark facts from the requirement; do NOT invent numbers.
- If variant is 3A/3B, set selected_variant.
- If requirement mentions spacer grids, set has_spacer_grids=true.
- If requirement mentions Pyrex/thimble plug/guide tube/instrument tube, set has_special_pin_map=true.
- Determine model_scope from source evidence or planning feature contract ONLY.
  Safe default is "unknown" — do NOT default to "single_assembly".
  Use "single_assembly" only when the source actually describes one assembly.
  Use "multi_assembly_core" when the source describes an N×N core.
  Use "full_core" only when the source describes a full radial core.
- For multi-assembly cores: set assembly_count, core_lattice_size, assembly_type_counts.
- For multi-assembly cores: use scoped_expected_counts with explicit scope, NOT legacy fields.
- Do NOT divide core totals by assembly count to guess per-assembly counts.
- Legacy fields (expected_pin_count etc.) are for single_assembly only.
- Unknown facts go into missing_facts, NOT fabricated values.
- If the source specifies multiple fuel enrichments or compositions, declare each as a
  fuel_variant_requirements entry with variant_id, enrichment, density, and which
  assembly_type_ids use it. This is the source-of-truth for fuel identity.
- If the source document says certain assembly types contain control rods, RCCA,
  absorber inserts, burnable poisons, or other localized inserts that must be
  PHYSICALLY PLACED at specific coordinates, declare a localized_insert_requirements
  entry for each kind. Defining materials, universes, or profiles for an insert
  does NOT satisfy this requirement — the entry declares that placement is REQUIRED.
  Fields:
  * requirement_id: unique identifier for this requirement
  * insert_kind: pyrex_rod|thimble_plug|control_rod|absorber_insert|instrumentation_insert|custom
  * assembly_type_ids: which assembly type IDs must contain this insert
  * expected_coordinate_count_per_assembly: how many paths per assembly instance
  * expected_assembly_instance_count: how many assembly instances have this insert
  * host_kind: guide_tube|instrument_tube|custom
  * required_profile_id: the axial profile ID this insert must reference (or null)
  * required_segment_roles: roles of the required profile segments (e.g., ["absorber","plenum"])
  * expected_insert_universe_ids: universe IDs the insert should use
  * anchor_z_cm: the source-specified operating position (e.g., poison bottom)
  * control_state_id: the current operating state label
  * required_in_detailed_domain: whether the insert must appear in the detailed lattice
- A requirement entry must NOT be omitted just because the insert has materials or
  universes defined. The requirement is the source contract for placement.

Minimal example (single assembly, source-backed):
{{"patch_type": "facts", "benchmark_id": "EXAMPLE", "selected_variant": "3B",
  "model_scope": "single_assembly", "lattice_size": [17, 17], "has_axial_geometry": true,
  "has_spacer_grids": true, "has_special_pin_map": true, "missing_facts": []}}

Reactor-neutral multi-assembly example:
{{"patch_type": "facts", "model_scope": "multi_assembly_core",
  "core_lattice_size": [2, 2], "assembly_count": 4,
  "assembly_type_counts": {{"type_a": 2, "type_b": 2}},
  "scoped_expected_counts": [
    {{"role": "fuel_pin", "value": 1000, "scope": "core_total"}},
    {{"role": "fuel_pin", "value": 250, "scope": "assembly_type", "assembly_type_id": "type_a"}}
  ]}}

When the source does NOT clearly describe the scope, emit
{{"model_scope": "unknown"}} and list the ambiguity in missing_facts.  Do NOT
fall back to "single_assembly" as a default — that selects the entire
single-assembly patch family and corrupts downstream generation.""",

    "materials": """\
Requested patch type: materials
Schema: {{"patch_type": "materials", "materials": [{{"material_id", "name", "role",
  "density_g_cm3", "temperature_K", "composition": {{"element": fraction}},
  "compound_components": [{{"formula", "fraction", "fraction_basis", "isotope_policy",
    "isotope_overrides", "source_note", "assumptions"}}],
  "composition_basis", "composition_status", "source_variant_id", "source_note",
  "warnings": [], "mixture_components": [{{"material_id", "volume_fraction"}}]}}],
  "assumptions": []}}

Rules:
- Do NOT generate universes, cells, lattices, or axial structures.
- Zircaloy-4 is NOT pure Zr; SS-304 is NOT pure Fe; Inconel-718 is NOT pure Ni.
- If composition is incomplete, set composition_status to "needs_library" or "needs_confirmation".
- approximate alloy is allowed ONLY with composition_status="approximate" AND an explicit warning.
- Never set composition_status="confirmed" for a known multi-element alloy with a single-element composition.
- When the input specifies a homogenized structural slab (for example a nozzle,
  support plate, baffle, or other steel/coolant structure), define a distinct
  material with role="structural" and mixture_components referencing the existing
  structural and coolant material IDs with the input-provided volume fractions.
  Do not represent that mixture only in an assumption and do not substitute pure
  coolant for the structural slab.
- Cross section paths do NOT belong here.
- If fuel_variant_requirements are present in Context, define one fuel material per
  variant. Set source_variant_id on each role="fuel" material to match the variant_id.
- Do NOT merge different enrichments into one fuel material.
- Do NOT define a required fuel material and leave it unused.
- Localized insert material roles are input-driven through Context's
  localized_insert_requirements: pyrex_rod requires at least one material with
  role="poison"; thimble_plug requires role="structural"; control_rod and
  absorber_insert require role="absorber".  Do not substitute a nearby role
  (for example, role="absorber" does not satisfy a required poison material).
- OpenMC composition entries MUST be transport-ready element or nuclide names only.
  Chemical formulas such as B2O3, SiO2, UO2, H2O, and Al2O3 are NOT nuclide names:
  never place them directly in composition.
- Put source chemical compounds in compound_components. Every component MUST include
  fraction, fraction_basis (weight_frac or atom_frac), and isotope_policy.
- For ordinary compounds with no isotope vector, use isotope_policy="natural_elements".
  For enriched fuel compounds, provide explicit isotope composition; never reduce
  enriched UO2 to generic U and O, and do not overwrite a fuel variant requirement.
- Every material must be transport-ready after deterministic species resolution.

composition_basis semantics (MUST declare for every material with a composition):
- "atom_frac": each value is a normalized atom fraction. If the source gives
  atom percent, divide by 100 first (e.g., 3.1 at% -> 0.031).
- "weight_frac": each value is a normalized weight fraction. If the source
  gives wt%, divide by 100 first (e.g., 98.2 wt% -> 0.982).
- "atom_density_barn_cm": each value is an absolute atom density in atoms/barn-cm
  (use this when the source gives values like 2.233e-2).
- "stoichiometric_ratio": the fuel nuclides sum to ~100 (enrichment vector,
  e.g., U235=3.1 means 3.1 at% of the uranium) and the oxygen entry gives the
  O/U ratio (e.g., O16=2.0 means O/U=2 for UO2).
  Use this when the source specifies enrichment per-100-uranium and oxygen as
  a molecular ratio. If U isotopes sum to ~100 and O16 is a small number
  like 2.0, this is ALWAYS stoichiometric_ratio, NEVER weight_frac.
- "ppm_by_weight": a coolant entry like B10=1066 means 1066 ppm total boron
  by weight (NOT an atom fraction).  Use this when the source gives boron
  concentration in ppm.
- "unknown": only if truly unclear; this will BLOCK rendering and require retry.

Minimal example:
{{"patch_type": "materials", "materials": [
 {{"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.257,
   "composition": {{"U235": 3.1, "U238": 96.9, "O16": 2.0}},
   "composition_basis": "stoichiometric_ratio",
   "composition_status": "approximate",
   "warnings": ["enrichment approximate"]}},
 {{"material_id": "coolant", "name": "Borated water", "role": "coolant",
   "density_g_cm3": 0.743, "composition": {{"B10": 1066, "H1": 2.0, "O16": 1.0}},
   "composition_basis": "ppm_by_weight",
   "composition_status": "confirmed"}},
 {{"material_id": "glass", "name": "Borosilicate glass", "role": "absorber",
   "density_g_cm3": 2.25, "composition": {{}}, "composition_basis": "weight_frac",
   "compound_components": [
     {{"formula": "B2O3", "fraction": 0.125, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"}},
     {{"formula": "SiO2", "fraction": 0.875, "fraction_basis": "weight_frac", "isotope_policy": "natural_elements"}}
   ], "composition_status": "approximate"}}
]}}""",

    "universes": """\
Requested patch type: universes
Schema: {{"patch_type": "universes", "universes": [{{"universe_id", "kind",
  "cells": [{{"id", "role", "material_id", "region_kind", "r_min_cm", "r_max_cm",
  "protected_through_path"}}], "source_note", "assumptions": []}}]}}

Rules:
- Do NOT generate the lattice, full pin map, or axial layers.
- Each universe must have at least one cell.
- guide_tube must include internal coolant AND tube wall material.
- thimble_plug is a guide tube with a plug inside: it MUST keep the same tube wall
  (Zircaloy-4 annulus) as a guide_tube, plus a water gap between the plug and the
  wall. Do NOT make it a solid rod of cladding material with water outside.
  Correct cell order (inside→out):
     plug (SS304 cylinder) → water_gap (water annulus) → wall (Zircaloy-4 annulus) → outer_coolant (water background)
- A thimble plug's central plug cell must use the Context-required structural
  material (normally the source-specified plug material), never the guide-tube
  wall material.  The wall remains a separate Zircaloy-4 annulus.
- pyrex_rod must include ALL gap layers from the input spec, not just pyrex+clad.
  A pyrex rod has concentric annuli with thin gas/water gaps between solid layers.
  Do NOT merge a gap into an adjacent solid layer. Correct cell order (inside→out):
    inner_water_or_helium (cylinder, inside inner tube)
    → inner_tube (SS304 annulus)
    → gap_1 (water or helium annulus, between inner tube and pyrex)
    → pyrex (pyrex glass annulus)
    → gap_2 (water or helium annulus, between pyrex and outer clad)
    → outer_clad (SS304 annulus)
    → outer_coolant (water background)
   Use the exact radii from the input problem description for each layer boundary.
   If the input does not specify a gap, insert a thin water annulus (0.001–0.01 cm).
 - The pyrex cell must reference a material whose role is "poison" when the
   corresponding Facts requirement is pyrex_rod; it must not reuse an absorber
   material merely because both are neutron poisons.
- In addition to fuel and localized-insert universes, define a universe for EVERY
  distinct cross-sectional profile that appears in ANY axial region of the source
  document.  At minimum this typically includes:
  * fuel_pin — active fuel pellet + gap + clad + outer coolant
  * guide_tube — inner water + Zircaloy-4 wall + outer coolant
  * instrument_tube — inner water + Zircaloy-4 wall + outer coolant (if separate)
  * end_plug — solid Zircaloy-4 cylinder + outer coolant
  * gas_gap / plenum — helium inner + Zircaloy-4 clad + outer coolant
  * water_pin / moderator_only — pure coolant pin cell (no fuel, no clad)
  These auxiliary universes are required because downstream axial_layers use them
  as replacement targets for non-active-fuel axial regions (end plugs, plenum,
  shoulder gaps, nozzle transitions).  If any is absent, axial_layers generation
  will fail with lattice_transform.replacement_universe_missing.
- Every universe_id used in downstream patches MUST exactly match a universe_id
  declared here.  Do NOT expect downstream patches to create their own universes.
- fuel_pin should have a fuel material cell.
- Mark through-path cells with protected_through_path=true.
- If Context shows N distinct fuel_variant_requirements, generate at least N distinct
  active-fuel universes (one per variant). Each active-fuel universe must use exactly
  one fuel material whose source_variant_id matches.
- Do NOT collapse different fuel enrichments into one generic fuel_pin universe.
- Every required fuel material must be reachable via at least one active-fuel universe.

Minimal example (two fuel variants):
{{"patch_type": "universes", "universes": [
  {{"universe_id": "fuel_pin_low", "kind": "fuel_pin", "cells": [
    {{"id": "fuel", "role": "fuel", "material_id": "fuel_low", "region_kind": "cylinder"}}
  ]}},
  {{"universe_id": "fuel_pin_high", "kind": "fuel_pin", "cells": [
    {{"id": "fuel", "role": "fuel", "material_id": "fuel_high", "region_kind": "cylinder"}}
  ]}}
]}}""",

    "pin_map": """\
Requested patch type: pin_map
Schema: {{"patch_type": "pin_map", "variant", "lattice_size": [int,int],
  "default_universe_id", "coordinate_convention": {{"index_base", "row_origin",
  "col_origin", "ordering"}},
  "guide_tube_coords": [[int,int]], "instrument_tube_coords": [[int,int]],
  "water_cell_coords": [[int,int]],
  "localized_insert_intents": [{{"insert_id", "insert_kind",
  "host_kind", "insert_universe_id", "coordinates": [[int,int]],
  "z_min_cm", "z_max_cm", "application_mode", "component_role",
  "preserve_component_roles": [], "priority", "requires_human_confirmation"}}],
  "assumptions": [], "source_note"}}

CRITICAL RULES:
- Do NOT output the full 17x17 lattice. Do NOT enumerate all 289 cells.
- Do NOT output rows of fuel_pin_universe.
- ONLY output special coordinate lists and default_universe_id.
- lattice_size is [rows, cols], e.g. [17, 17].
- coordinate_convention.index_base must be 0 or 1.
- Each coordinate is [row, col] using the stated convention.
- guide_tube_coords lists ALL persistent guide tube paths (full assembly height).
  This includes positions that will host localized inserts (Pyrex, thimble plugs, etc.).
  A guide tube coordinate that hosts an insert is still a guide_tube_coord —
  the insert only affects a specific z range within that guide tube.
- localized_insert_intents declares finite-height inserts (Pyrex rods, thimble plugs,
  absorbers, control rods). Each intent specifies:
  * insert_kind: pyrex_rod | thimble_plug | absorber_insert | control_rod | custom
  * host_kind: guide_tube (the host path whose inner component is replaced)
  * insert_universe_id: the universe to use inside the host for this z range
  * coordinates: positions affected (must be a subset of guide_tube_coords)
  * z_min_cm, z_max_cm: the axial interval where the insert is active
  * application_mode: "nested_component_override" (default, preserves host wall)
  * component_role: role of the replaced inner component (e.g. "internal_coolant")
  * preserve_component_roles: roles to keep (e.g. ["tube_wall", "outer_coolant"])
- Coordinates of localized inserts MUST also appear in guide_tube_coords.
- Do NOT put insert universes as base lattice positions.
- Do NOT stretch a finite insert over the full assembly height.

Minimal example:
{{"patch_type": "pin_map", "lattice_size": [17, 17], "default_universe_id": "fuel_pin",
  "coordinate_convention": {{"index_base": 1, "row_origin": "top", "col_origin": "left", "ordering": "row_col"}},
  "guide_tube_coords": [[3,6], [3,9], [6,6], [9,3]],
  "instrument_tube_coords": [[9,9]],
  "localized_insert_intents": [
    {{"insert_id": "absorber_group_1", "insert_kind": "absorber_insert",
      "host_kind": "guide_tube", "insert_universe_id": "absorber_inner_profile",
      "coordinates": [[3,6], [6,6]], "z_min_cm": 20.0, "z_max_cm": 200.0,
      "application_mode": "nested_component_override",
      "component_role": "internal_coolant",
      "preserve_component_roles": ["tube_wall", "outer_coolant"], "priority": 0}}
  ]}}""",

    "axial_layers": """\
Requested patch type: axial_layers
Schema: {{"patch_type": "axial_layers", "axial_domain_cm": [float,float],
  "layers": [{{"layer_id", "role", "z_min_cm", "z_max_cm", "fill_type", "fill_id",
  "loading_id", "loading_ids", "requires_human_confirmation", "assumptions": [], "source_note"}}],
  "lattice_loadings": [{{"loading_id", "base_lattice_id", "derived_lattice_id",
  "transformations": [{{"operation_id", "operation_kind", "replacement_universe_id",
  "source_universe_id", "source_universe_ids", "target_coordinates",
  "component_role", "component_path_id", "preserve_component_roles",
  "preserve_path_ids", "priority", "purpose"}}],
  "overrides": {{"universe_id": [[int,int]]}}, "purpose"}}]}}

Rules:
- layer.role MUST be exactly one of: "lower_moderator_buffer",
  "lower_core_plate", "lower_nozzle", "lower_shoulder_gap",
  "lower_fuel_endplug", "lower_end_plug", "lower_plenum",
  "active_fuel", "gas_gap", "upper_fuel_endplug", "upper_end_plug",
  "fuel_upper_plenum", "upper_plenum", "upper_shoulder_gap",
  "upper_nozzle", "upper_core_plate", "upper_moderator_buffer",
  "core_plate", "reflector", "shoulder_gap", or "custom".  In
  particular, use "lower_end_plug" / "upper_end_plug", never
  "lower_endplug" / "upper_endplug".
- transformation.priority is optional.  Omit it when no ordering is needed;
  if present it MUST be an integer such as 0.  Never emit null for priority.
- Do NOT generate materials/universes/lattice/full plan.
- For fill_type="material", fill_id MUST be one of Context.known_material_ids. Do not
  invent homogenized mixture IDs; use an existing ID or mark the layer for human
  confirmation when no supported material is available.
- lower_nozzle, upper_nozzle, and core_plate are whole-cross-section structural
  slabs. They MUST use an existing material whose Context material role is
  structural/steel/mixture, never a coolant/moderator material merely because the
  slab is homogenized with coolant. Do not encode a steel/coolant mixture only as
  an assumption while filling the layer with pure coolant.
- For fill_type="universe" and every transformation replacement_universe_id,
  use only Context.known_universe_ids. Do not invent end-plug or plenum universes.
- active_fuel layer must exist if the problem is a 3D fuel assembly.
- active_fuel fill_type should be "lattice" and fill_id should be "assembly_lattice".
- Do NOT use default z=-1..1 for an explicit 3D benchmark.
- If z values are unknown, set requires_human_confirmation=true; do NOT fabricate.
- Do NOT represent spacer grids as axial layer material slabs.
- Do NOT enumerate every default fuel-pin coordinate. Use replace_universe_family
  for component profiles shared by all pins of a universe family.
- Use sparse coordinate_override only for localized positions.
- Use nested_component_override when an insert occupies the inside of an existing
  tube and the tube wall must remain.
- IMPORTANT: If localized_insert_intents are declared in the pin_map, do NOT
  create separate lattice_loadings for those inserts. The assembler automatically
  derives loadings from localized_insert_intents. Only create lattice_loadings
  for fuel-component-profile transformations (replace_universe_family) and
  for inserts NOT declared as localized_insert_intents.
- Use loading_ids when an axial layer requires more than one localized loading.
- Spacer grids remain axial_overlays, not lattice transformations.
- Component-profile layers (end_plug, plenum, gas_gap, shoulder_gap) must use lattice fill
  with a replace_universe_family transformation, NOT a whole-layer material slab.
- shoulder_gap is the moderator region between the fuel stack and the nozzle/end structure.
  Guide tubes and instrument tubes continue through it. Use role "shoulder_gap"
  (or "lower_shoulder_gap" / "upper_shoulder_gap" for positional clarity).
  fill_type must be "lattice" with fill_id "assembly_lattice" and a loading_id that
  replaces only the fuel-pin family with a moderator-only universe.
  Do NOT use fill_type=material for shoulder_gap.
  Do NOT label shoulder_gap as lower_plenum/upper_plenum.

Transformation operation_kind values:
- "replace_universe_family": source_universe_id -> replacement_universe_id for all positions.
- "coordinate_override": specific target_coordinates -> replacement_universe_id.
- "nested_component_override": specific target_coordinates, component_role identifies the
  cell to replace; preserve_component_roles lists cells that must survive.

Minimal example (family replacement for a plenum layer):
{{"patch_type": "axial_layers", "axial_domain_cm": [0.0, 400.0],
  "lattice_loadings": [
    {{"loading_id": "plenum_loading", "base_lattice_id": "assembly_lattice",
      "derived_lattice_id": "assembly_lattice_plenum",
      "transformations": [
        {{"operation_id": "family_plenum", "operation_kind": "replace_universe_family",
          "replacement_universe_id": "fuel_pin_plenum", "source_universe_id": "fuel_pin"}}
      ]}}
  ],
  "layers": [
    {{"layer_id": "active_fuel", "role": "active_fuel", "z_min_cm": 10.0, "z_max_cm": 375.0,
      "fill_type": "lattice", "fill_id": "assembly_lattice"}},
    {{"layer_id": "upper_plenum", "role": "upper_plenum", "z_min_cm": 379.0, "z_max_cm": 395.0,
      "fill_type": "lattice", "fill_id": "assembly_lattice", "loading_id": "plenum_loading"}}
  ]}}""",

    "axial_overlays": """\
Requested patch type: axial_overlays
Schema: {{"patch_type": "axial_overlays", "overlays": [{{"overlay_id", "overlay_kind",
  "z_min_cm", "z_max_cm", "target_lattice_id", "material_id", "geometry_mode",
  "total_mass_g", "through_path_preserved", "requires_human_confirmation", "assumptions": []}}]}}

Rules:
- Spacer grids MUST be overlays, NOT material slabs.
- When the source provides grid mass or density, use geometry_mode="mass_conserving_outer_frame"
  with target_lattice_id="assembly_lattice", material_id, and total_mass_g set from the source.
  This preserves through-paths (fuel, clad, guide tubes) while adding the grid frame.
- For geometry_mode="mass_conserving_outer_frame" or "homogenized_open_region", you MUST set
  through_path_preserved=true. This is a required field for these modes.
- total_mass_g is REQUIRED for geometry_mode="mass_conserving_outer_frame". Read the grid
  mass per assembly from the source document (e.g., end grid 1017 g, middle grid 875 g).
- Only use geometry_mode="homogenized_open_region" when mass data is truly unavailable.
- target_lattice_id should be "assembly_lattice".
- If grid z positions are unknown, use geometry_mode="skeleton" with requires_human_confirmation=true.

Minimal example (1 overlay):
{{"patch_type": "axial_overlays", "overlays": [
  {{"overlay_id": "grid_1", "overlay_kind": "spacer_grid", "z_min_cm": 50.0, "z_max_cm": 52.0,
    "target_lattice_id": "assembly_lattice", "material_id": "grid_mat",
    "geometry_mode": "mass_conserving_outer_frame", "total_mass_g": 1017.0,
    "through_path_preserved": true}}
]}}""",

    "settings": """\
Requested patch type: settings
Schema: {{"patch_type": "settings", "source_strategy",
  "source_requires_fissionable_constraint", "plot_strategy",
  "cross_sections_runtime_required", "tallies_required_for_smoke_test", "assumptions": []}}

Rules:
- source_strategy defaults to "active_fuel_box".
- plot_strategy defaults to "full_assembly".
- cross_sections_runtime_required=true (runtime concern, not plan-generation blocker).
- tallies_required_for_smoke_test=false.

Minimal example:
{{"patch_type": "settings", "source_strategy": "active_fuel_box",
  "plot_strategy": "full_assembly", "cross_sections_runtime_required": true,
  "tallies_required_for_smoke_test": false}}""",

    "assembly_catalog": """\
Requested patch type: assembly_catalog
Schema: {{"patch_type": "assembly_catalog",
  "assembly_types": [
    {{"assembly_type_id": "type_id", "name": "descriptive name", "role": "fuel/guide/etc",
      "fuel_variant_id": "variant_id_or_omit_if_no_fuel_variant_requirements",
      "multiplicity_hint": int_or_null,
      "pin_map": {{
        "lattice_size": [int, int],
        "default_universe_id": "universe_id",
        "coordinate_convention": {{"index_base": 0, "row_origin": "top", "col_origin": "left", "ordering": "row_col"}},
        "guide_tube_coords": [[row, col], ...],
        "instrument_tube_coords": [[row, col], ...],
        "water_cell_coords": [[row, col], ...],
        "localized_insert_intents": [
          {{"insert_id": "id", "insert_kind": "pyrex_rod|thimble_plug|absorber_insert|...",
            "host_kind": "guide_tube", "insert_universe_id": "universe_id",
            "coordinates": [[row, col], ...], "z_min_cm": float, "z_max_cm": float,
            "application_mode": "coordinate_override"}}
        ]
      }},
      "axial_profile_id": "id_or_null",
      "overlay_set_id": "id_or_null",
      "requires_human_confirmation": false
    }}
  ],
  "assumptions": [], "source_note": "optional"}}

Rules:
- Only output assembly type TEMPLATES — not core placement.
- Do NOT output a full expanded pin lattice (e.g. 17x17 matrix).
- Each assembly type outputs ONLY: default universe + sparse special coordinates.
- Different assembly types CAN share universe IDs ONLY when the source facts
  explicitly show they use the same fuel variant and same base pin geometry.
- If the source specifies different enrichment, fuel composition, or fuel region
  for different assembly types, they MUST NOT share the active-fuel default universe.
- Localized inserts being the same or different must NOT mask a fuel material
  difference. Check fuel variant first.
- If fuel_variant_requirements are in Context, set fuel_variant_id on each
  fuel assembly type to match the variant for its assembly_type_id.
- default_universe_id must use a universe whose fuel material has the same
  source_variant_id as the assembly type's fuel_variant_id.
- The multiplicity_hint is advisory; the core_layout patch determines actual placement.
- Every localized_insert_requirement in Context MUST be fulfilled by a matching
  localized_insert_intent in the corresponding assembly type's pin_map.
- For each localized_insert_requirement, copy required_profile_id exactly into
  the matching intent.axial_profile_id when it is provided. Do NOT substitute
  insert_kind or an invented profile synonym.
- For each localized_insert_requirement, copy control_state_id exactly into the
  matching intent.control_state_id when it is provided for a movable insert.
- An assembly type whose name or role contains "control" or "RCCA" MUST have a
  localized_insert_intent with insert_kind="control_rod" — the name alone does
  NOT satisfy the placement requirement.
- Defining control-rod universes or profiles does NOT satisfy placement. You MUST
  create a localized_insert_intent with actual coordinates, axial_profile_id,
  anchor_z_cm, and control_state_id matching the requirement.
- Multi-segment control rods MUST reference an axial_profile_id from Context.
  The insert_universe_id is the initial/compatible segment universe; the actual
  multi-segment structure is resolved from the profile.
- Intent coordinates MUST be actual positions — do NOT leave empty.
- Intent anchor_z_cm must match the source operating state.
- Use EITHER axial_profile_id (for multi-segment profiled inserts) OR z_min_cm/z_max_cm
  (for simple fixed-range inserts). NEVER set both axial_profile_id and z_min/z_max on
  the same intent — the validator rejects this as a profile_anchor_conflict.
- When axial_profile_id is set, anchor_z_cm provides the profile's base position; do NOT
  also set z_min_cm or z_max_cm.
- control_state_id must match the current selected variant/state.
- Localized inserts MUST NOT be written to axial_overlays.
- Do NOT permanently bake control rods into the base pin lattice.
- Do NOT put instrument-tube coordinates in a control_rod intent.

Reactor-neutral multi-segment control rod example:
{{"patch_type": "assembly_catalog",
  "assembly_types": [
    {{"assembly_type_id": "controlled_type", "fuel_variant_id": "fuel_a",
      "pin_map": {{
        "lattice_size": [5, 5], "default_universe_id": "fuel_pin_a",
        "coordinate_convention": {{"index_base": 0, "row_origin": "top", "col_origin": "left", "ordering": "row_col"}},
        "guide_tube_coords": [[1,1], [1,3], [3,1], [3,3]],
        "localized_insert_intents": [
          {{"insert_id": "bank_a", "insert_kind": "control_rod",
            "host_kind": "guide_tube", "host_universe_id": "guide_tube",
            "insert_universe_id": "rod_absorber_lower",
            "coordinates": [[1,1], [1,3], [3,1], [3,3]],
            "axial_profile_id": "rod_profile_a",
            "anchor_z_cm": 120.0,
            "control_state_id": "state_1",
            "application_mode": "coordinate_override"}}
        ]
      }}}}
  ]}}""",

    "localized_insert_profiles": """\
Requested patch type: localized_insert_profiles
Schema: {{"patch_type": "localized_insert_profiles",
  "profiles": [
    {{"profile_id": "unique_id",
      "anchor_kind": "bottom|top|center|absolute",
      "anchor_z_cm": float_or_null,
      "segments": [
        {{"segment_id": "unique_within_profile",
          "relative_z_min_cm": float,
          "relative_z_max_cm": float,
          "universe_id": "known_universe",
          "role": "absorber|plenum|end_structure|...",
          "source_note": "optional"}}
      ],
      "source_note": "optional",
      "assumptions": []}}
  ],
  "assumptions": [],
  "source_note": "optional"}}

Rules:
- Define ONLY reusable axial profiles — no pin coordinates, no core layout.
- Each segment uses RELATIVE coordinates (relative to anchor point).
- Each segment MUST reference a universe already defined in the universes patch.
- Segments MUST be ordered by relative_z (ascending), no overlaps.
- Mechanical gaps between segments MUST be explicitly modeled or noted in assumptions.
- Do NOT adjust profiles based on keff or criticality targets.
- For movable inserts (control rods), the ACTUAL position is provided by intent.anchor_z_cm,
  NOT the profile. The profile only defines the relative structure.
- When the intent provides the anchor (all non-absolute profiles), set anchor_z_cm=null
  (NOT 0.0). A non-null profile anchor_z_cm conflicts with the intent anchor and blocks
  downstream assembly_catalog generation.
- anchor_kind determines coordinate translation: bottom=additive, top=subtractive, center=bilateral.
- Use anchor_kind="absolute" only when segments are already in global coordinates.
- If Context/Facts.localized_insert_requirements declares required_profile_id,
  define that exact profile_id. Do NOT replace it with insert_kind or an invented synonym.
- If Context/Facts.localized_insert_requirements declares required_segment_roles and
  expected_insert_universe_ids with the same length, the profile segments MUST use those
  role and universe_id values one-to-one in source order.
- If the required universe IDs are not present in known_universe_ids, do not invent
  substitute profile_* IDs; leave a clear assumption/warning only if the upstream
  universes patch is incomplete.

Reactor-neutral example (multi-segment control rod profile):
{{"patch_type": "localized_insert_profiles",
  "profiles": [
    {{"profile_id": "rod_type_a",
      "anchor_kind": "bottom",
      "segments": [
        {{"segment_id": "lower_end", "relative_z_min_cm": 0.0, "relative_z_max_cm": 10.0,
          "universe_id": "rod_end_structure", "role": "end_structure"}},
        {{"segment_id": "absorber", "relative_z_min_cm": 10.0, "relative_z_max_cm": 110.0,
          "universe_id": "rod_absorber", "role": "absorber"}},
        {{"segment_id": "plenum", "relative_z_min_cm": 110.0, "relative_z_max_cm": 140.0,
          "universe_id": "rod_plenum", "role": "plenum"}}
      ]}}
  ]}}""",

    "core_layout": """\
Requested patch type: core_layout
Schema: {{"patch_type": "core_layout",
  "core_lattice_id": "core_lattice",
  "shape": [int, int],
  "assembly_pitch_cm": float,
  "coordinate_convention": {{"index_base": 0, "row_origin": "top", "col_origin": "left", "ordering": "row_col"}},
  "assembly_pattern": [["type_id", ...], ...],
  "outer_assembly_type_id": "type_id_or_null",
  "boundary": "reflective|vacuum|periodic",
  "expected_assembly_type_counts": {{"type_id": count}},
  "symmetry_description": "optional",
  "requires_human_confirmation": false}}

Rules:
- Only output assembly type PLACEMENT — not pin coordinates.
- Do NOT re-define materials or universes.
- Each entry in assembly_pattern MUST be a known assembly_type_id from the assembly_catalog.
- The pattern shape MUST match the shape field [rows, cols].
- All rows MUST have the same length.
- Expected type multiplicities MUST match the pattern counts.
- Use the boundary from the requirement document.
- Do NOT fill missing positions with guesses.

Reactor-neutral example (2x2 heterogeneous):
{{"patch_type": "core_layout",
  "core_lattice_id": "core_lattice",
  "shape": [2, 2],
  "assembly_pitch_cm": 21.5,
  "assembly_pattern": [["type_a", "type_b"], ["type_b", "type_a"]],
  "expected_assembly_type_counts": {{"type_a": 2, "type_b": 2}},
  "boundary": "reflective"}}""",
}


# Total/per-patch character budgets for the few-shot reference block in a patch
# prompt. Each incremental layer is generated independently, so this budget
# applies per layer (not accumulated across the 7-patch pipeline).
FEW_SHOT_PATCH_BUDGET: int = 2400
_FEWSHOT_PER_PATCH_MAX: int = 1200


def _few_shot_block(patch_type: str, context: Any | None) -> str:
    """Render a reference-patch block from gold few-shot cases, if any.

    Returns ``""`` when there is no context, no case ids, or no patch of the
    requested type is available — leaving the prompt unchanged for backward
    compatibility.
    """
    case_ids = list(getattr(context, "few_shot_case_ids", []) or []) if context else []
    if not case_ids:
        return ""
    patches = load_patch_few_shots(patch_type, case_ids, limit=2)
    if not patches:
        return ""
    header = (
        f"Reference {patch_type} patch(es) from similar successful cases "
        f"— illustrative, not authoritative (adapt structure & values, do NOT "
        f"copy verbatim):"
    )
    lines = [header]
    remaining = FEW_SHOT_PATCH_BUDGET - len(header)
    for patch in patches:
        blob = json.dumps(patch, ensure_ascii=False)
        if len(blob) > _FEWSHOT_PER_PATCH_MAX:
            blob = blob[: _FEWSHOT_PER_PATCH_MAX] + "...(truncated)"
        if len(blob) + 1 > remaining:
            break
        lines.append(blob)
        remaining -= len(blob) + 1
    if len(lines) == 1:
        return ""
    return "\n".join(lines) + "\n\n"


def build_patch_prompt(
    patch_type: str,
    requirement: str,
    context: Any | None = None,
) -> str:
    """Build a prompt for generating exactly one patch of the given type."""
    patch_rules = _PATCH_RULES.get(patch_type, "")
    if not patch_rules:
        patch_rules = f"Requested patch type: {patch_type}\n(No specific rules defined.)"

    contract = _OUTPUT_CONTRACT.format(patch_type=patch_type)
    context_block = _context_block(context)
    if patch_type == "facts" and getattr(context, "confirmed_facts", None):
        context_block = (
            "Confirmed facts from human review (authoritative only at their stated JSON paths; "
            "preserve source=human_confirmation provenance and do not re-ask these facts):\n"
            f"{getattr(context, 'confirmed_facts')}\n\n" + context_block
        )
    placement_confirmations = (
        getattr(context, "confirmed_facts", {}).get("plan_closed_loop", {}).get("placement", {})
        if getattr(context, "confirmed_facts", None) else {}
    )
    if patch_type in {"localized_insert_profiles", "pin_map", "assembly_catalog", "core_layout"} and placement_confirmations:
        context_block = (
            "Confirmed placement facts from human review (authoritative only at their stated JSON paths; "
            "apply only to the target placement patch and preserve human-confirmation provenance):\n"
            f"{placement_confirmations}\n\n" + context_block
        )
    few_shot_block = _few_shot_block(patch_type, context)

    # Phase 7C: add allowed/forbidden keys.
    allowed = sorted(get_patch_allowed_top_level_keys(patch_type))
    forbidden = sorted(get_patch_forbidden_top_level_keys(patch_type))
    keys_block = (
        f"Allowed top-level keys for patch_type=\"{patch_type}\":\n"
        f"  {', '.join(allowed)}\n\n"
        f"FORBIDDEN top-level keys (will cause immediate rejection):\n"
        f"  {', '.join(forbidden)}\n\n"
    )

    # Phase 8A Step 3: structured investigation evidence.  Empty list (the
    # default) produces an empty section, so legacy prompts are unchanged.
    evidence_block = _investigation_evidence_block(context)
    # Phase 8A Step 6: derived planning constraints (inventory-derived).
    # Rendered in a separate section from source-backed evidence so the
    # LLM and reviewers can distinguish "what the source said" from
    # "what the inventory inferred".
    constraints_block = _planning_constraints_block(context)
    # Phase-8B Step 3: authoritative Facts requirement skeleton.  Only
    # rendered for the facts patch type when the skeleton is available.
    skeleton_block = _facts_skeleton_block(context)

    return (
        f"{contract}\n\n"
        f"{keys_block}"
        f"{patch_rules}\n\n"
        f"{few_shot_block}"
        f"{context_block}"
        f"{skeleton_block}"
        f"{evidence_block}"
        f"{constraints_block}"
        f"Requirement:\n{requirement}\n\n"
        f'Return ONLY the JSON object with patch_type="{patch_type}". No other text.'
    )


def _planning_constraints_block(context: Any | None) -> str:
    """Render the Phase 8A Step 6 derived-constraints section.

    Returns "" when ``context.planning_constraints`` is empty or absent.
    """

    if context is None:
        return ""
    constraints = getattr(context, "planning_constraints", None)
    if not constraints:
        return ""
    lines = [
        "Derived Planning Constraints (from inventory; do NOT contradict)",
        "",
    ]
    for c in constraints:
        cid = c.get("constraint_id", "")
        kind = c.get("constraint_kind", "")
        subject = c.get("subject", "")
        predicate = c.get("predicate", "")
        value_repr = json.dumps(c.get("value"), ensure_ascii=False, sort_keys=True)
        if len(value_repr) > 120:
            value_repr = value_repr[:117] + "..."
        status = c.get("derivation_status", "deterministically_derived")
        crit = c.get("criticality", "supporting")
        req_ids = ", ".join(c.get("inventory_requirement_ids", []) or [])
        lines.append(
            f"- [{cid}] ({kind}) {subject}.{predicate} = {value_repr}"
            f"  ({status}/{crit})"
        )
        if req_ids:
            lines.append(f"    requirement_ids: {req_ids}")
        unresolved = c.get("unresolved_fields", []) or []
        if unresolved:
            lines.append(f"    unresolved_fields: {', '.join(unresolved)}")
    lines.append("")
    lines.append(
        "These constraints are deterministically derived from the geometry "
        "component inventory.  Treat them as hard requirements; do NOT invent "
        "values that contradict them, and do NOT silently drop unresolved ones."
    )
    return "\n".join(lines) + "\n\n"


def _facts_skeleton_block(context: Any | None) -> str:
    """Render the Phase-8B Step 3 authoritative Facts requirement skeleton.

    Returns "" when ``context.facts_evidence_contract`` is absent (off mode)
    so legacy prompts are byte-identical.
    """
    if context is None:
        return ""
    contract = getattr(context, "facts_evidence_contract", None)
    if not contract or not isinstance(contract, dict):
        return ""

    lines: list[str] = [
        "=== AUTHORITATIVE FACTS REQUIREMENT SKELETON ===",
        "",
        "The fields below are LOCKED from evidence investigation. "
        "Do NOT change their values.",
        "",
    ]

    # Resolved scope
    scope = contract.get("resolved_scope", "unknown")
    if scope not in ("unknown",):
        lines.append(f"LOCKED — model_scope: {scope}")
        lines.append("")

    # Feature flags
    flags = contract.get("required_feature_flags", [])
    if flags:
        lines.append("LOCKED feature flags:")
        for flag in flags:
            lines.append(f"  - {flag}: true")
        lines.append("")

    # Assembly layout
    if contract.get("required_assembly_layout"):
        lines.append("LOCKED — multi-assembly layout is required.")
        lines.append("")

    # Fuel variants
    variants = contract.get("required_fuel_variants", [])
    if variants:
        lines.append("REQUIRED fuel variants (each MUST have a matching fuel_variant_requirements entry):")
        for v in variants:
            vid = v.get("variant_id", "?")
            enrich = v.get("enrichment_wt_percent")
            density = v.get("density_g_cm3")
            atypes = ", ".join(v.get("assembly_type_ids", []))
            enrich_str = f", enrichment={enrich} wt%" if enrich is not None else ""
            density_str = f", density={density} g/cm3" if density is not None else ""
            atypes_str = f", assembly_types=[{atypes}]" if atypes else ""
            lines.append(f"  - variant_id={vid}{enrich_str}{density_str}{atypes_str}")
        lines.append("")

    # Localized inserts
    inserts = contract.get("required_localized_inserts", [])
    if inserts:
        lines.append("REQUIRED localized insert placements (each MUST have a matching localized_insert_requirements entry):")
        for ins in inserts:
            rid = ins.get("requirement_id", "?")
            kind = ins.get("insert_kind", "?")
            atypes = ", ".join(ins.get("assembly_type_ids", []))
            coord_count = ins.get("expected_coordinate_count_per_assembly")
            coord_str = f", coords_per_assembly={coord_count}" if coord_count is not None else ""
            lines.append(f"  - requirement_id={rid}, insert_kind={kind}, assembly_types=[{atypes}]{coord_str}")
        lines.append("")

    # Unresolved fields
    unresolved = contract.get("unresolved_source_critical_fields", [])
    if unresolved:
        lines.append("UNRESOLVED slots (must be filled by analysis — do NOT leave empty):")
        for u in unresolved:
            lines.append(f"  - {u}")
        lines.append("")

    # Conflicts
    conflicts = contract.get("conflicts", [])
    if conflicts:
        lines.append("WARNING — conflicting slots detected:")
        for c in conflicts:
            lines.append(f"  - {c}")
        lines.append("")

    lines.append(
        "These values are sourced from evidence investigation. "
        "They are authoritative and must NOT be changed."
    )

    return "\n".join(lines) + "\n\n"


def _investigation_evidence_block(context: Any | None) -> str:
    """Render the Phase 8A Step 3 investigation-evidence section.

    Returns "" when ``context.investigation_evidence`` is empty or
    absent.  This keeps the legacy prompt byte-identical when the
    investigation stage is disabled (which is the default).
    """

    if context is None:
        return ""
    evidence = getattr(context, "investigation_evidence", None)
    if not evidence:
        return ""
    # Imported lazily so plan_builder does not pay for the
    # plan_investigation package import when the flag is off.
    from openmc_agent.plan_investigation.prompt import (
        render_investigation_evidence_for_prompt,
    )

    rendered = render_investigation_evidence_for_prompt(evidence)
    if not rendered:
        return ""
    return rendered + "\n\n"


# ---------------------------------------------------------------------------
# Compact overlay repair prompt (issue-scoped retry)
# ---------------------------------------------------------------------------


def _build_compact_overlay_retry(
    issues: list[dict[str, Any]],
    attempt_index: int,
    previous_patch: dict[str, Any],
) -> str:
    """Build a compact repair prompt for axial_overlay semantic issues.

    Includes the previous parsed patch JSON, exact failing overlays, and
    allowed vs locked fields — preventing the LLM from re-deriving the
    entire patch from scratch.
    """
    # Extract failing overlay IDs from issue paths.
    failing_ids: list[str] = []
    for issue in issues:
        code = issue.get("code", "")
        if "mode_semantic_contradiction" in code or "through_path_not_preserved" in code:
            path = issue.get("path", "")
            # path format: overlays[grid_id].through_path_preserved
            oid = path.replace("overlays[", "").replace("].through_path_preserved", "")
            if oid:
                failing_ids.append(oid)

    id_list = ", ".join(failing_ids) if failing_ids else "all affected overlays"

    previous_json = json.dumps(previous_patch, indent=2, ensure_ascii=False)

    return (
        "Your previous axial_overlays patch is structurally correct except\n"
        "for a semantic contradiction in the following overlays:\n"
        f"  {id_list}\n\n"
        "PROBLEM:\n"
        'For geometry_mode="mass_conserving_outer_frame" or\n'
        '"homogenized_open_region", through_path_preserved must be true.\n'
        "These modes only replace the outer moderator frame or open region.\n"
        "Protected paths that must remain:\n"
        "  - fuel pellet\n"
        "  - fuel-clad gap\n"
        "  - cladding\n"
        "  - guide-tube wall\n"
        "  - instrument-tube wall\n"
        "  - Pyrex internal structure\n"
        "  - thimble plug\n"
        "  - RCCA\n"
        "  - necessary coolant paths outside these structures\n\n"
        "ALLOWED CHANGE:\n"
        "  - overlays[...].through_path_preserved  (set to true or remove)\n"
        "  - overlays[...].assumptions             (text only)\n"
        "  - overlays[...].source_note             (text only)\n\n"
        "LOCKED (must NOT change):\n"
        "  - overlay IDs and count\n"
        "  - overlay_kind, z_min_cm, z_max_cm\n"
        "  - material_id, geometry_mode, total_mass_g\n"
        "  - target_lattice_id, cell_count, pitch_cm\n"
        "  - frame_area_cm2, frame_thickness_cm\n"
        "  - ordering and all other fields\n\n"
        "YOUR PREVIOUS PATCH:\n"
        f"{previous_json}\n\n"
        "Return the COMPLETE axial_overlays JSON with the fix applied.\n"
        "Output ONLY the JSON — no reasoning, no prose, no markdown fences."
    )


def build_retry_prompt(
    patch_type: str,
    requirement: str,
    context: Any | None,
    issues: list[dict[str, Any]],
    attempt_index: int,
    *,
    previous_patch: dict[str, Any] | None = None,
) -> str:
    """Build a retry prompt for a patch that failed validation.

    When ``previous_patch`` is available and the failure is a scoped issue
    (e.g. ``mode_semantic_contradiction``), a **compact repair prompt** is
    generated instead of the full base prompt + requirement.  The compact
    prompt includes the previous parsed JSON, the exact failing overlays,
    and the allowed vs locked fields — minimising token cost and preventing
    the LLM from re-deriving the entire patch from scratch.
    """
    issue_codes = [i.get("code", "") for i in issues]

    # Determine whether compact retry is applicable.
    is_semantic_contradiction = any(
        "mode_semantic_contradiction" in c or "through_path_not_preserved" in c
        for c in issue_codes
    )

    if previous_patch is not None and is_semantic_contradiction and patch_type == "axial_overlays":
        return _build_compact_overlay_retry(
            issues, attempt_index, previous_patch,
        )

    # Fall back to full-prompt retry for other issue types.
    base_prompt = build_patch_prompt(patch_type, requirement, context)

    # Check failure type for targeted retry message.
    is_full_plan = any(
        "full_plan" in code or "full_lattice" in code
        for code in issue_codes
    )
    is_patch_type_issue = any(
        "patch_type" in code for code in issue_codes
    )
    is_parse_error = any(
        "json_parse" in code for code in issue_codes
    )
    is_fuel_variant_missing = any(
        "fuel_variant_missing" in code for code in issue_codes
    )
    is_through_path = any(
        "through_path_not_preserved" in code or "mode_semantic_contradiction" in code
        for code in issue_codes
    )
    material_species_codes = {
        "materials.compound_in_transport_composition",
        "materials.unsupported_compound_formula",
        "materials.compound_fraction_basis_missing",
        "materials.compound_isotope_policy_missing",
        "materials.fissile_compound_isotope_policy_missing",
        "materials.fissile_compound_would_erase_enrichment",
    }
    is_material_species = patch_type == "materials" and any(
        code in material_species_codes for code in issue_codes
    )

    issue_lines: list[str] = []
    for issue in issues:
        sev = issue.get("severity", "error")
        code = issue.get("code", "unknown")
        msg = issue.get("message", "")
        issue_lines.append(f"  - [{sev}] {code}: {msg}")

    if is_material_species:
        affected = [
            {"path": i.get("path"), "species": i.get("actual"), "message": i.get("message")}
            for i in issues if i.get("code") in material_species_codes
        ]
        previous_json = json.dumps(previous_patch or {}, ensure_ascii=False, indent=2)
        forbidden_block = (
            "\n\nTARGETED MATERIALS RETRY — regenerate only the materials patch.\n"
            f"Affected entries: {json.dumps(affected, ensure_ascii=False)}\n"
            "B2O3 and SiO2 are chemical compounds, not nuclides. Move any chemical "
            "formula from composition to compound_components. Keep density, fractions, "
            "source notes, and fuel-variant requirements unchanged. Do not invent explicit "
            "isotope fractions. Allowed changes: materials[*].composition, "
            "materials[*].compound_components, and necessary composition_basis/status/warnings.\n"
            "Do not regenerate facts, universes, or fuel variant requirements.\n"
            f"Previous materials patch:\n{previous_json}"
        )
    elif is_full_plan:
        forbidden_block = (
            f"\n\nYour previous response was REJECTED because it looked like a "
            f"full SimulationPlan, not a {patch_type} patch.\n"
            f"Do not include complex_model/core/materials/universes/lattices "
            f"unless they are part of the {patch_type} schema.\n"
            f'The first character must be "{{". The last character must be "}}".\n'
            f'Regenerate ONLY the {patch_type} JSON with patch_type="{patch_type}".'
        )
    elif is_patch_type_issue:
        forbidden_block = (
            f'\n\nYour response must include patch_type="{patch_type}" as a '
            f'top-level key. Add it exactly. Do not omit or rename it.'
        )
    elif is_parse_error:
        forbidden_block = (
            f"\n\nYour previous response was not valid JSON.\n"
            f"Return JSON only. No prose. No markdown."
        )
    elif is_fuel_variant_missing:
        # Build explicit per-assembly-type directives from issue data.
        directives: list[str] = []
        for issue in issues:
            if "fuel_variant_missing" not in issue.get("code", ""):
                continue
            expected = issue.get("expected", "")
            path = issue.get("path", "")
            tid = path.replace("assembly_types[", "").replace("].fuel_variant_id", "")
            if expected and tid:
                directives.append(f'  "{tid}": add "fuel_variant_id": "{expected}"')
        directive_block = "\n".join(directives) if directives else ""
        forbidden_block = (
            f"\n\nYou MUST add the field \"fuel_variant_id\" to each assembly type.\n"
            f"This field is in the schema template. Do NOT omit it.\n"
            f"Exact values to use:\n{directive_block}\n"
            f"Output ONLY JSON — no reasoning, no prose."
        )
    elif is_through_path:
        overlay_ids = [
            issue.get("path", "").replace("overlays[", "").replace("].through_path_preserved", "")
            for issue in issues
            if "through_path_not_preserved" in issue.get("code", "")
        ]
        id_list = ", ".join(overlay_ids) if overlay_ids else "all overlays"
        forbidden_block = (
            f"\n\nYou MUST set \"through_path_preserved\": true on every overlay\n"
            f"with geometry_mode \"mass_conserving_outer_frame\" or \"homogenized_open_region\".\n"
            f"Affected overlays: {id_list}\n"
            f"Output ONLY JSON — no reasoning, no prose."
        )
    elif any("localized_insert.required_placement_missing" in c for c in issue_codes):
        placement_lines: list[str] = []
        for issue in issues:
            if "localized_insert" not in issue.get("code", ""):
                continue
            placement_lines.append(
                f"  - {issue.get('code')}: {issue.get('message')}"
            )
        forbidden_block = (
            "\n\nTARGETED ASSEMBLY_CATALOG RETRY — add missing localized_insert_intent.\n"
            "Each localized_insert_requirement in Context MUST be fulfilled by a\n"
            "matching intent in the assembly type's pin_map.localized_insert_intents.\n"
            "Creating universes or profiles is NOT sufficient — you MUST create an intent\n"
            "with actual coordinates, axial_profile_id, anchor_z_cm, and control_state_id.\n"
            "Only modify: assembly_types[type_id].pin_map.localized_insert_intents\n"
            "Do NOT change: fuel_variant_id, default_universe_id, other assembly types,\n"
            "core layout, materials, universes, or grid overlays.\n\n"
            "Placement failures:\n"
            + "\n".join(placement_lines)
        )
    elif any("localized_insert.coordinate_count_mismatch" in c or
             "localized_insert.anchor_mismatch" in c or
             "localized_insert.control_state_mismatch" in c for c in issue_codes):
        forbidden_block = (
            "\n\nFIX the localized_insert_intent fields:\n"
            "- coordinates must match the exact count from Context\n"
            "- anchor_z_cm must match the source operating position\n"
            "- control_state_id must match the current state\n"
            "Only modify the affected intent fields.\n"
        )
    else:
        forbidden_block = ""

    return (
        f"{base_prompt}"
        f"\n\nThe previous {patch_type} (attempt {attempt_index}) failed:\n"
        + "\n".join(issue_lines)
        + forbidden_block
        + f"\n\nFix these issues and regenerate ONLY the {patch_type} JSON."
    )


# ---------------------------------------------------------------------------
# Context serialization
# ---------------------------------------------------------------------------


def _context_block(context: Any | None) -> str:
    if context is None:
        return ""

    parts: list[str] = ["Context:"]
    for attr in (
        "benchmark_id", "selected_variant",
        "confirmed_facts", "extracted_facts",
        "validated_patch_summaries",
        "expected_counts",
        "known_material_ids", "known_universe_ids", "known_lattice_ids",
        "active_fuel_region_cm", "axial_domain_cm",
        "strict_benchmark",
        "model_scope", "assembly_count", "core_lattice_size",
        "assembly_type_counts", "known_assembly_type_ids",
        "assembly_pitch_cm", "scoped_expected_counts",
        "known_insert_profile_ids", "insert_profile_summaries",
        "movable_insert_facts",
        "fuel_variant_requirements", "material_summaries",
        "universe_summaries", "assembly_fuel_binding_summaries",
    ):
        val = getattr(context, attr, None)
        if val is None:
            continue
        if isinstance(val, str) and val == "unknown":
            continue
        if isinstance(val, dict) and not val:
            continue
        if isinstance(val, list) and not val:
            continue
        if isinstance(val, tuple):
            val = list(val)
        parts.append(f"  {attr}: {val}")

    # Render context_facts with provenance labels.
    context_facts = getattr(context, "context_facts", None)
    if context_facts:
        cf_parts = ["  context_facts (with provenance):"]
        for cf_key, cf_val in context_facts.items():
            if isinstance(cf_val, dict):
                prov = cf_val.get("provenance_kind", "unresolved")
                authoritative = cf_val.get("authoritative", False)
                tag = "AUTHORITATIVE" if authoritative else prov
                cf_parts.append(f"    {cf_key}: {cf_val.get('value')} [{tag}]")
            else:
                cf_parts.append(f"    {cf_key}: {cf_val}")
        parts.append("\n".join(cf_parts))

    if len(parts) == 1:
        return ""
    return "\n".join(parts) + "\n\n"


__all__ = [
    "build_patch_prompt",
    "build_retry_prompt",
]
