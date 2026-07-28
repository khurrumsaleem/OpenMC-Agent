"""VERA4 deterministic base-case fixture builder (P2-FULLCORE-2D-A).

Builds complete deterministic VERA4 patches with:
- Full axial domain -55.000 → 463.937 cm (12 base layers)
- Whole-plane materials (nozzle, core plate, moderator buffer)
- Detailed-core segments (shoulder, endplug, active fuel, plenum)
- Two fuel variants (2.11%, 2.619%)
- Exact Pyrex coordinates (20 per edge assembly)
- Thimble plugs (24 per corner, 4 per edge)
- RCCA multi-segment profile (AIC/B4C/plenum/endplug)
- Assembly-scoped spacer grids (8 bands, 72 instances)
- Guide-tube and instrument-tube wall continuity

All coordinates use 1-based source convention with index_base=1.
Conversion to 0-based IR happens once in the assembler/materializer.
"""

from __future__ import annotations

from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    AxialLayerPatchItem,
    AxialLayersPatch,
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
    BasePathAxialProfilePatchItem,
    BasePathAxialProfilesPatch,
    BasePathStateBindingPatchItem,
    CoreLayoutPatch,
    FactsPatch,
    FuelVariantRequirementPatchItem,
    LocalizedInsertAxialProfilePatchItem,
    LocalizedInsertAxialSegmentPatchItem,
    LocalizedInsertIntentPatchItem,
    LocalizedInsertPlacementRequirementPatchItem,
    LocalizedInsertProfilesPatch,
    MaterialsPatch,
    MaterialSpecPatch,
    MixtureComponentPatch,
    SettingsPatch,
    UniversesPatch,
    UniverseSpecPatch,
    CellLayerPatch,
    ScopedExpectedCount,
    CoordinateConvention,
)
from openmc_agent.schemas import CoreBoundarySpec


# ---------------------------------------------------------------------------
# Axial domain boundaries (VERA4 base case)
# ---------------------------------------------------------------------------

Z_DOMAIN_MIN = -55.000
Z_DOMAIN_MAX = 463.937

Z_LOWER_MOD_BUFFER = -5.000
Z_LOWER_CORE_PLATE = 0.000
Z_LOWER_NOZZLE_TOP = 6.053
Z_LOWER_SHOULDER_TOP = 10.281
Z_LOWER_ENDPLUG_TOP = 11.951
Z_ACTIVE_FUEL_TOP = 377.711
Z_UPPER_ENDPLUG_TOP = 379.381
Z_UPPER_PLENUM_TOP = 395.381
Z_UPPER_SHOULDER_TOP = 397.510
Z_UPPER_NOZZLE_TOP = 406.337
Z_UPPER_CORE_PLATE_TOP = 413.937

Z_DETAILED_CORE_MIN = Z_LOWER_SHOULDER_TOP  # 10.281
Z_DETAILED_CORE_MAX = Z_UPPER_SHOULDER_TOP  # 397.510


# ---------------------------------------------------------------------------
# Pin radii (VERA4 W-17x17)
# ---------------------------------------------------------------------------

R_FUEL_PELLET = 0.4096
R_GAP = 0.4180
R_CLAD_OUTER = 0.4750
R_GT_INNER = 0.5615
R_GT_WALL_OUTER = 0.6121
R_IT_INNER = 0.5588
R_IT_WALL_OUTER = 0.6048


# ---------------------------------------------------------------------------
# Guide tube coordinates (1-based source convention)
# ---------------------------------------------------------------------------

GT_COORDS_1B = [
    (3, 6), (3, 9), (3, 12),
    (4, 4), (4, 14),
    (6, 3), (6, 6), (6, 9), (6, 12), (6, 15),
    (9, 3), (9, 6), (9, 12), (9, 15),
    (12, 3), (12, 6), (12, 9), (12, 12), (12, 15),
    (14, 4), (14, 14),
    (15, 6), (15, 9), (15, 12),
]

INST_COORD_1B = (9, 9)


# ---------------------------------------------------------------------------
# Pyrex coordinates for edge assembly (1-based source, exactly 20 positions)
# ---------------------------------------------------------------------------

PYREX_COORDS_1B = [
    (3, 6), (3, 12),
    (4, 4), (4, 14),
    (6, 3), (6, 6), (6, 9), (6, 12), (6, 15),
    (9, 6), (9, 12),
    (12, 3), (12, 6), (12, 9), (12, 12), (12, 15),
    (14, 4), (14, 14),
    (15, 6), (15, 12),
]


# ---------------------------------------------------------------------------
# Thimble plug coordinates (1-based source)
# ---------------------------------------------------------------------------

# Corner: all 24 guide tube positions
THIMBLE_CORNER_1B = list(GT_COORDS_1B)

# Edge: only 4 specific positions
THIMBLE_EDGE_1B = [
    (3, 9),
    (9, 3),
    (9, 15),
    (15, 9),
]


# ---------------------------------------------------------------------------
# RCCA profile dynamic boundaries
# ---------------------------------------------------------------------------

RCCA_ANCHOR_Z = 257.900
RCCA_AIC_HEIGHT = 101.60
RCCA_B4C_TOTAL = 360.68
RCCA_PLENUM_TOTAL = 371.38
RCCA_ENDPLUG_TOTAL = 373.28


# ---------------------------------------------------------------------------
# Spacer grid z-ranges
# ---------------------------------------------------------------------------

GRID_BANDS = [
    (11.951, 15.817),   # end grid 1 (Inconel)
    (73.295, 77.105),   # mid grid 1 (Zircaloy)
    (125.495, 129.305),
    (177.695, 181.505),
    (229.895, 233.705),
    (282.095, 285.905),
    (334.295, 338.105),
    (386.267, 390.133), # end grid 2 (Inconel)
]


def build_vera4_materials() -> MaterialsPatch:
    """Build VERA4 materials with full composition."""
    return MaterialsPatch(materials=[
        # --- Fuel variants (source: Table P4-2) ---
        MaterialSpecPatch(
            material_id="fuel_r1", name="UO2 2.11%", role="fuel",
            density_g_cm3=10.257,
            composition={"U234": 0.0174, "U235": 2.11, "U236": 0.0097,
                         "U238": 97.8629, "O16": 2.0},
            composition_basis="stoichiometric_ratio",
            composition_status="confirmed",
            source_variant_id="fuel_region1",
            source_note="VERA4 Table P4-2 Region 1; UO2 stoichiometry O/U=2",
        ),
        MaterialSpecPatch(
            material_id="fuel_r2", name="UO2 2.619%", role="fuel",
            density_g_cm3=10.257,
            composition={"U234": 0.0219, "U235": 2.619, "U236": 0.012,
                         "U238": 97.3471, "O16": 2.0},
            composition_basis="stoichiometric_ratio",
            composition_status="confirmed",
            source_variant_id="fuel_region2",
            source_note="VERA4 Table P4-2 Region 2; UO2 stoichiometry O/U=2",
        ),
        # --- Coolant ---
        MaterialSpecPatch(
            material_id="water", name="borated water", role="coolant",
            density_g_cm3=0.743,
            composition={
                # 1360 ppm soluble boron by mass with natural boron
                # (B-10 ≈ 19.9 at%), converted to atom fractions on an
                # H2O + natural-B atom basis.
                "H1": 0.666162697,
                "O16": 0.333081349,
                "B10": 0.000150435,
                "B11": 0.000605519,
            },
            composition_basis="atom_frac",
            composition_status="confirmed",
            source_note=(
                "VERA4 input: HZP coolant density 0.743 g/cc; soluble boron "
                "1360 ppm by mass, natural boron B-10 ≈ 19.9 at%"
            ),
        ),
        # --- Structural ---
        MaterialSpecPatch(
            material_id="zircaloy4", name="Zircaloy-4", role="cladding",
            density_g_cm3=6.56,
            composition={"Zr": 0.9823, "Sn": 0.0145, "Fe": 0.0021, "Cr": 0.0010},
            composition_basis="weight_frac",
            composition_status="approximate",
        ),
        MaterialSpecPatch(
            material_id="ss304", name="SS-304", role="structural",
            density_g_cm3=7.94,
            composition={"Fe": 0.695, "Cr": 0.190, "Ni": 0.095, "Mn": 0.020},
            composition_basis="weight_frac",
            composition_status="approximate",
        ),
        MaterialSpecPatch(
            material_id="inconel718", name="Inconel-718", role="structural",
            density_g_cm3=8.19,
            composition={"Ni": 0.528, "Cr": 0.186, "Fe": 0.185,
                         "Nb": 0.051, "Mo": 0.030, "Ti": 0.010},
            composition_basis="weight_frac",
            composition_status="approximate",
        ),
        # --- Gas ---
        MaterialSpecPatch(
            material_id="helium", name="helium gas", role="gap_gas",
            density_g_cm3=0.001,
            composition={"He4": 1.0},
            composition_basis="atom_frac",
            composition_status="approximate",
        ),
        # --- Absorbers ---
        MaterialSpecPatch(
            material_id="pyrex_glass", name="pyrex glass", role="absorber",
            density_g_cm3=2.25,
            composition={},
            composition_basis="weight_frac",
            compound_components=[
                {"formula": "B2O3", "fraction": 12.5,
                 "fraction_basis": "weight_frac", "isotope_policy": "natural_elements",
                 "source_note": "VERA4 §13.1 B2O3 mass fraction"},
                {"formula": "SiO2", "fraction": 87.5,
                 "fraction_basis": "weight_frac", "isotope_policy": "natural_elements",
                 "source_note": "silica balance after specified B2O3 fraction"},
            ],
            composition_status="approximate",
        ),
        MaterialSpecPatch(
            material_id="rcca_aic_mat", name="Ag-In-Cd absorber", role="absorber",
            density_g_cm3=10.2,
            composition={"Ag107": 0.416467, "In115": 0.148802, "Cd114": 0.052794,
                         "Ag109": 0.381936},
            composition_basis="atom_frac",
            composition_status="approximate",
        ),
        MaterialSpecPatch(
            material_id="rcca_b4c_mat", name="B4C absorber", role="absorber",
            density_g_cm3=1.76,
            composition={"B10": 0.1592, "B11": 0.6408, "C0": 0.2},
            composition_basis="atom_frac",
            composition_status="confirmed",
            source_note="VERA4 input §16.1 B4C density 1.76 g/cc; natural boron B4C stoichiometry",
        ),
        MaterialSpecPatch(
            material_id="thimble_plug_mat", name="SS-304 thimble plug", role="structural",
            density_g_cm3=7.94,
            composition={"Fe": 0.695, "Cr": 0.190, "Ni": 0.095, "Mn": 0.020},
            composition_basis="weight_frac",
            composition_status="approximate",
        ),
        # --- Mixture materials (nozzle, core plate) ---
        MaterialSpecPatch(
            material_id="lower_nozzle_mix", name="lower nozzle mixture", role="structural",
            density_g_cm3=3.50,
            mixture_components=[
                MixtureComponentPatch(material_id="ss304", volume_fraction=0.279217),
                MixtureComponentPatch(material_id="water", volume_fraction=0.720783),
            ],
            derivation_method="vol% mixture: 27.9217% SS304 + 72.0783% coolant",
            source_note="VERA4 lower nozzle homogenized mixture",
        ),
        MaterialSpecPatch(
            material_id="upper_nozzle_mix", name="upper nozzle mixture", role="structural",
            density_g_cm3=3.00,
            mixture_components=[
                MixtureComponentPatch(material_id="ss304", volume_fraction=0.191470),
                MixtureComponentPatch(material_id="water", volume_fraction=0.808530),
            ],
            derivation_method="vol% mixture: 19.1470% SS304 + 80.8530% coolant",
            source_note="VERA4 upper nozzle homogenized mixture",
        ),
        MaterialSpecPatch(
            material_id="lower_core_plate_mix", name="lower core plate mixture", role="structural",
            density_g_cm3=4.50,
            mixture_components=[
                MixtureComponentPatch(material_id="ss304", volume_fraction=0.50),
                MixtureComponentPatch(material_id="water", volume_fraction=0.50),
            ],
            derivation_method="vol% mixture: 50% SS304 + 50% coolant",
            source_note="VERA4 lower core plate homogenized mixture",
        ),
        MaterialSpecPatch(
            material_id="upper_core_plate_mix", name="upper core plate mixture", role="structural",
            density_g_cm3=4.50,
            mixture_components=[
                MixtureComponentPatch(material_id="ss304", volume_fraction=0.50),
                MixtureComponentPatch(material_id="water", volume_fraction=0.50),
            ],
            derivation_method="vol% mixture: 50% SS304 + 50% coolant",
            source_note="VERA4 upper core plate homogenized mixture",
        ),
    ])


def build_vera4_universes() -> UniversesPatch:
    """Build VERA4 universes with complete pin-cell structures."""
    return UniversesPatch(universes=[
        # --- Active fuel ---
        UniverseSpecPatch(
            universe_id="fuel_active_r1", kind="fuel_pin",
            cells=[
                CellLayerPatch(id="pellet", role="fuel_internal", material_id="fuel_r1",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_FUEL_PELLET),
                CellLayerPatch(id="gap", role="gas_gap", material_id="helium",
                               region_kind="cylinder", r_min_cm=R_FUEL_PELLET, r_max_cm=R_GAP),
                CellLayerPatch(id="clad", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GAP, r_max_cm=R_CLAD_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        UniverseSpecPatch(
            universe_id="fuel_active_r2", kind="fuel_pin",
            cells=[
                CellLayerPatch(id="pellet", role="fuel_internal", material_id="fuel_r2",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_FUEL_PELLET),
                CellLayerPatch(id="gap", role="gas_gap", material_id="helium",
                               region_kind="cylinder", r_min_cm=R_FUEL_PELLET, r_max_cm=R_GAP),
                CellLayerPatch(id="clad", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GAP, r_max_cm=R_CLAD_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Fuel endplug (solid Zircaloy rod) ---
        UniverseSpecPatch(
            universe_id="fuel_endplug", kind="fuel_pin",
            cells=[
                CellLayerPatch(id="endplug", role="fuel_internal", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_CLAD_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Fuel plenum (helium gas inside cladding) ---
        UniverseSpecPatch(
            universe_id="fuel_plenum", kind="fuel_pin",
            cells=[
                CellLayerPatch(id="gas", role="gas_gap", material_id="helium",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_GAP),
                CellLayerPatch(id="clad", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GAP, r_max_cm=R_CLAD_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Water pin (no fuel rod) ---
        UniverseSpecPatch(
            universe_id="water_pin", kind="water_cell",
            cells=[
                CellLayerPatch(id="water", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Guide tube ---
        UniverseSpecPatch(
            universe_id="guide_tube", kind="guide_tube",
            cells=[
                CellLayerPatch(id="inner", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Instrument tube ---
        UniverseSpecPatch(
            universe_id="inst_tube", kind="instrument_tube",
            cells=[
                CellLayerPatch(id="inner", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_IT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_IT_INNER, r_max_cm=R_IT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Pyrex poison rod ---
        UniverseSpecPatch(
            universe_id="pyrex_poison", kind="pyrex_rod",
            cells=[
                CellLayerPatch(id="pyrex", role="absorber", material_id="pyrex_glass",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.3835),
                CellLayerPatch(id="inner_tube", role="cladding", material_id="ss304",
                               region_kind="cylinder", r_min_cm=0.3835, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="gas_gap", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Pyrex plenum (above poison, helium + SS tube + GT wall) ---
        UniverseSpecPatch(
            universe_id="pyrex_plenum", kind="custom",
            cells=[
                CellLayerPatch(id="gas", role="gas_gap", material_id="helium",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- Thimble plug ---
        UniverseSpecPatch(
            universe_id="thimble_plug", kind="thimble_plug",
            cells=[
                CellLayerPatch(id="plug", role="structural", material_id="thimble_plug_mat",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- RCCA AIC absorber ---
        UniverseSpecPatch(
            universe_id="rcca_aic", kind="control_rod",
            cells=[
                CellLayerPatch(id="absorber", role="absorber", material_id="rcca_aic_mat",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.3860),
                CellLayerPatch(id="clad", role="cladding", material_id="ss304",
                               region_kind="cylinder", r_min_cm=0.3860, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- RCCA B4C absorber ---
        UniverseSpecPatch(
            universe_id="rcca_b4c", kind="control_rod",
            cells=[
                CellLayerPatch(id="absorber", role="absorber", material_id="rcca_b4c_mat",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.3860),
                CellLayerPatch(id="clad", role="cladding", material_id="ss304",
                               region_kind="cylinder", r_min_cm=0.3860, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- RCCA plenum (helium inside SS cladding) ---
        UniverseSpecPatch(
            universe_id="rcca_plenum", kind="custom",
            cells=[
                CellLayerPatch(id="gas", role="gas_gap", material_id="helium",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
        # --- RCCA endplug (solid SS) ---
        UniverseSpecPatch(
            universe_id="rcca_endplug", kind="custom",
            cells=[
                CellLayerPatch(id="plug", role="structural", material_id="ss304",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.4180),
                CellLayerPatch(id="annulus", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.4180, r_max_cm=R_GT_INNER),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=R_GT_INNER, r_max_cm=R_GT_WALL_OUTER),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        ),
    ])


def build_vera4_axial_layers() -> AxialLayersPatch:
    """Build the complete 12-layer VERA4 axial domain."""
    return AxialLayersPatch(
        axial_domain_cm=(Z_DOMAIN_MIN, Z_DOMAIN_MAX),
        layers=[
            # --- Below-core whole-plane slabs ---
            AxialLayerPatchItem(
                layer_id="lower_mod_buffer", role="lower_moderator_buffer",
                z_min_cm=Z_DOMAIN_MIN, z_max_cm=Z_LOWER_MOD_BUFFER,
                fill_type="material", fill_id="water",
            ),
            AxialLayerPatchItem(
                layer_id="lower_core_plate", role="lower_core_plate",
                z_min_cm=Z_LOWER_MOD_BUFFER, z_max_cm=Z_LOWER_CORE_PLATE,
                fill_type="material", fill_id="lower_core_plate_mix",
            ),
            AxialLayerPatchItem(
                layer_id="lower_nozzle", role="lower_nozzle",
                z_min_cm=Z_LOWER_CORE_PLATE, z_max_cm=Z_LOWER_NOZZLE_TOP,
                fill_type="material", fill_id="lower_nozzle_mix",
            ),
            # --- Detailed-core region ---
            AxialLayerPatchItem(
                layer_id="lower_shoulder", role="lower_shoulder_gap",
                z_min_cm=Z_LOWER_NOZZLE_TOP, z_max_cm=Z_LOWER_SHOULDER_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            AxialLayerPatchItem(
                layer_id="lower_endplug", role="lower_fuel_endplug",
                z_min_cm=Z_LOWER_SHOULDER_TOP, z_max_cm=Z_LOWER_ENDPLUG_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            AxialLayerPatchItem(
                layer_id="active_fuel", role="active_fuel",
                z_min_cm=Z_LOWER_ENDPLUG_TOP, z_max_cm=Z_ACTIVE_FUEL_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            AxialLayerPatchItem(
                layer_id="upper_endplug", role="upper_fuel_endplug",
                z_min_cm=Z_ACTIVE_FUEL_TOP, z_max_cm=Z_UPPER_ENDPLUG_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            AxialLayerPatchItem(
                layer_id="upper_plenum", role="fuel_upper_plenum",
                z_min_cm=Z_UPPER_ENDPLUG_TOP, z_max_cm=Z_UPPER_PLENUM_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            AxialLayerPatchItem(
                layer_id="upper_shoulder", role="upper_shoulder_gap",
                z_min_cm=Z_UPPER_PLENUM_TOP, z_max_cm=Z_UPPER_SHOULDER_TOP,
                fill_type="lattice", fill_id="core_lattice",
            ),
            # --- Above-core whole-plane slabs ---
            AxialLayerPatchItem(
                layer_id="upper_nozzle", role="upper_nozzle",
                z_min_cm=Z_UPPER_SHOULDER_TOP, z_max_cm=Z_UPPER_NOZZLE_TOP,
                fill_type="material", fill_id="upper_nozzle_mix",
            ),
            AxialLayerPatchItem(
                layer_id="upper_core_plate", role="upper_core_plate",
                z_min_cm=Z_UPPER_NOZZLE_TOP, z_max_cm=Z_UPPER_CORE_PLATE_TOP,
                fill_type="material", fill_id="upper_core_plate_mix",
            ),
            AxialLayerPatchItem(
                layer_id="upper_mod_buffer", role="upper_moderator_buffer",
                z_min_cm=Z_UPPER_CORE_PLATE_TOP, z_max_cm=Z_DOMAIN_MAX,
                fill_type="material", fill_id="water",
            ),
        ],
    )


def build_vera4_rcca_profile() -> LocalizedInsertProfilesPatch:
    """Build the RCCA multi-segment axial profile."""
    return LocalizedInsertProfilesPatch(
        profiles=[
            LocalizedInsertAxialProfilePatchItem(
                profile_id="rcca_base",
                anchor_kind="bottom",
                anchor_z_cm=RCCA_ANCHOR_Z,
                segments=[
                    LocalizedInsertAxialSegmentPatchItem(
                        segment_id="aic",
                        relative_z_min_cm=0.0,
                        relative_z_max_cm=RCCA_AIC_HEIGHT,
                        universe_id="rcca_aic",
                        role="absorber_aic",
                    ),
                    LocalizedInsertAxialSegmentPatchItem(
                        segment_id="b4c",
                        relative_z_min_cm=RCCA_AIC_HEIGHT,
                        relative_z_max_cm=RCCA_B4C_TOTAL,
                        universe_id="rcca_b4c",
                        role="absorber_b4c",
                    ),
                    LocalizedInsertAxialSegmentPatchItem(
                        segment_id="plenum",
                        relative_z_min_cm=RCCA_B4C_TOTAL,
                        relative_z_max_cm=RCCA_PLENUM_TOTAL,
                        universe_id="rcca_plenum",
                        role="plenum",
                    ),
                    LocalizedInsertAxialSegmentPatchItem(
                        segment_id="endplug",
                        relative_z_min_cm=RCCA_PLENUM_TOTAL,
                        relative_z_max_cm=RCCA_ENDPLUG_TOTAL,
                        universe_id="rcca_endplug",
                        role="upper_endplug",
                    ),
                ],
                source_note="VERA4 RCCA base position, anchor at rod bottom",
            ),
        ],
    )


def build_vera4_assembly_catalog() -> AssemblyCatalogPatch:
    """Build VERA4 assembly catalog with exact coordinates (1-based source)."""
    conv = CoordinateConvention(index_base=1, row_origin="top", col_origin="left", ordering="row_col")

    # Corner: fuel_r1, no inserts, thimble plugs at all 24 GT positions
    corner_inserts = [
        LocalizedInsertIntentPatchItem(
            insert_id="corner_thimble",
            insert_kind="thimble_plug",
            host_kind="guide_tube",
            host_universe_id="guide_tube",
            insert_universe_id="thimble_plug",
            coordinates=THIMBLE_CORNER_1B,
            z_min_cm=383.310,
            z_max_cm=394.310,
            source_note="VERA4 thimble plugs in all corner guide tubes",
        ),
    ]

    # Edge: fuel_r2, Pyrex at 20 positions, thimble at 4 positions
    edge_inserts = [
        LocalizedInsertIntentPatchItem(
            insert_id="edge_pyrex",
            insert_kind="pyrex_rod",
            host_kind="guide_tube",
            host_universe_id="guide_tube",
            insert_universe_id="pyrex_poison",
            coordinates=PYREX_COORDS_1B,
            z_min_cm=15.761,
            z_max_cm=376.441,
            source_note="VERA4 Pyrex poison rods at 20 edge guide tube positions",
        ),
        LocalizedInsertIntentPatchItem(
            insert_id="edge_pyrex_plenum",
            insert_kind="pyrex_rod",
            host_kind="guide_tube",
            host_universe_id="guide_tube",
            insert_universe_id="pyrex_plenum",
            coordinates=PYREX_COORDS_1B,
            z_min_cm=376.441,
            z_max_cm=397.510,
            source_note="VERA4 Pyrex plenum above poison segment",
        ),
        LocalizedInsertIntentPatchItem(
            insert_id="edge_thimble",
            insert_kind="thimble_plug",
            host_kind="guide_tube",
            host_universe_id="guide_tube",
            insert_universe_id="thimble_plug",
            coordinates=THIMBLE_EDGE_1B,
            z_min_cm=383.310,
            z_max_cm=394.310,
            source_note="VERA4 thimble plugs at 4 edge guide tube positions",
        ),
    ]

    # Center: fuel_r1, RCCA at 24 GT positions via profile
    center_inserts = [
        LocalizedInsertIntentPatchItem(
            insert_id="center_rcca",
            insert_kind="control_rod",
            host_kind="guide_tube",
            host_universe_id="guide_tube",
            insert_universe_id="rcca_aic",
            coordinates=GT_COORDS_1B,
            axial_profile_id="rcca_base",
            anchor_z_cm=RCCA_ANCHOR_Z,
            control_state_id="base",
            source_note="VERA4 RCCA at center assembly, 24 guide tube positions",
        ),
    ]

    return AssemblyCatalogPatch(
        assembly_types=[
            AssemblyTypePatchItem(
                assembly_type_id="corner",
                name="corner assembly (2.11%)",
                role="fuel",
                fuel_variant_id="fuel_region1",
                multiplicity_hint=4,
                base_path_profile_id="vera4_fuel_path",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(17, 17),
                    default_universe_id="fuel_active_r1",
                    coordinate_convention=conv,
                    guide_tube_coords=GT_COORDS_1B,
                    instrument_tube_coords=[INST_COORD_1B],
                    localized_insert_intents=corner_inserts,
                    source_note="VERA4 corner assembly",
                ),
            ),
            AssemblyTypePatchItem(
                assembly_type_id="edge",
                name="edge assembly (2.619%)",
                role="fuel",
                fuel_variant_id="fuel_region2",
                multiplicity_hint=4,
                base_path_profile_id="vera4_fuel_path",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(17, 17),
                    default_universe_id="fuel_active_r2",
                    coordinate_convention=conv,
                    guide_tube_coords=GT_COORDS_1B,
                    instrument_tube_coords=[INST_COORD_1B],
                    localized_insert_intents=edge_inserts,
                    source_note="VERA4 edge assembly with Pyrex and thimble plugs",
                ),
            ),
            AssemblyTypePatchItem(
                assembly_type_id="center_rcca",
                name="center RCCA assembly (2.11%)",
                role="fuel",
                fuel_variant_id="fuel_region1",
                multiplicity_hint=1,
                base_path_profile_id="vera4_fuel_path",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(17, 17),
                    default_universe_id="fuel_active_r1",
                    coordinate_convention=conv,
                    guide_tube_coords=GT_COORDS_1B,
                    instrument_tube_coords=[INST_COORD_1B],
                    localized_insert_intents=center_inserts,
                    source_note="VERA4 center assembly with RCCA",
                ),
            ),
        ],
        source_note="VERA4 3-assembly catalog",
    )


def build_vera4_core_layout() -> CoreLayoutPatch:
    """Build VERA4 3×3 core layout."""
    return CoreLayoutPatch(
        core_lattice_id="core_lattice",
        shape=(3, 3),
        assembly_pitch_cm=21.50,
        coordinate_convention=CoordinateConvention(index_base=0),
        assembly_pattern=[
            ["corner", "edge", "corner"],
            ["edge", "center_rcca", "edge"],
            ["corner", "edge", "corner"],
        ],
        boundary="reflective",
        boundary_conditions=CoreBoundarySpec(
            xmin="reflective",
            xmax="reflective",
            ymin="reflective",
            ymax="reflective",
            zmin="vacuum",
            zmax="vacuum",
        ),
        expected_assembly_type_counts={"corner": 4, "edge": 4, "center_rcca": 1},
        symmetry_description="1/8 symmetric 3x3 core",
    )


def build_vera4_facts() -> FactsPatch:
    """Build VERA4 facts patch."""
    return FactsPatch(
        benchmark_id="VERA4",
        model_scope="multi_assembly_core",
        lattice_size=(17, 17),
        pin_pitch_cm=1.26,
        assembly_pitch_cm=21.50,
        core_lattice_size=(3, 3),
        assembly_count=9,
        assembly_type_counts={"corner": 4, "edge": 4, "center_rcca": 1},
        has_axial_geometry=True,
        has_spacer_grids=True,
        has_special_pin_map=True,
        scoped_expected_counts=[
            ScopedExpectedCount(role="fuel_pin", value=2376, scope="core_total"),
            ScopedExpectedCount(role="guide_tube", value=216, scope="core_total"),
            ScopedExpectedCount(role="instrument_tube", value=9, scope="core_total"),
            ScopedExpectedCount(role="pyrex_rod", value=80, scope="core_total"),
            ScopedExpectedCount(role="thimble_plug", value=112, scope="core_total"),
        ],
        axial_domain_cm=(Z_DOMAIN_MIN, Z_DOMAIN_MAX),
        active_fuel_region_cm=(Z_LOWER_ENDPLUG_TOP, Z_ACTIVE_FUEL_TOP),
        fuel_variant_requirements=[
            FuelVariantRequirementPatchItem(
                variant_id="fuel_region1",
                source_label="Region 1",
                enrichment_wt_percent=2.11,
                density_g_cm3=10.257,
                assembly_type_ids=["corner", "center_rcca"],
                expected_assembly_count=5,
                source_note="Table P4-2 Region 1; C and R assemblies",
            ),
            FuelVariantRequirementPatchItem(
                variant_id="fuel_region2",
                source_label="Region 2",
                enrichment_wt_percent=2.619,
                density_g_cm3=10.257,
                assembly_type_ids=["edge"],
                expected_assembly_count=4,
                source_note="Table P4-2 Region 2; E assemblies",
            ),
        ],
        localized_insert_requirements=[
            LocalizedInsertPlacementRequirementPatchItem(
                requirement_id="rcca_center",
                insert_kind="control_rod",
                assembly_type_ids=["center_rcca"],
                expected_coordinate_count_per_assembly=24,
                expected_assembly_instance_count=1,
                host_kind="guide_tube",
                required_profile_id="rcca_base",
                required_segment_roles=["absorber_aic", "absorber_b4c", "plenum", "end_structure"],
                expected_insert_universe_ids=["rcca_aic", "rcca_b4c", "rcca_plenum", "rcca_endplug"],
                anchor_z_cm=RCCA_ANCHOR_Z,
                control_state_id="base",
                required_in_detailed_domain=True,
                source_note="VERA4 Problem Sec. 15: center R assembly 24 RCCA paths at base poison bottom 257.900 cm",
            ),
        ],
    )


def build_vera4_spacer_grids() -> AxialOverlaysPatch:
    """Build VERA4 spacer grid overlays (8 bands)."""
    overlays = []
    for i, (z_min, z_max) in enumerate(GRID_BANDS):
        is_end = (i == 0 or i == len(GRID_BANDS) - 1)
        mat_id = "inconel718" if is_end else "zircaloy4"
        grid_kind = "end" if is_end else "middle"
        overlays.append(AxialOverlayPatchItem(
            overlay_id=f"grid_{grid_kind}_{i}",
            overlay_kind="spacer_grid",
            z_min_cm=z_min,
            z_max_cm=z_max,
            target_lattice_id=None,
            material_id=mat_id,
            geometry_mode="mass_conserving_outer_frame",
            through_path_preserved=True,
            total_mass_g=1017.0 if is_end else 875.0,
            cell_count=289,
            pitch_cm=1.26,
            source_note=f"VERA4 {grid_kind} spacer grid band {i}",
        ))
    return AxialOverlaysPatch(overlays=overlays)


def build_vera4_base_path_profiles() -> BasePathAxialProfilesPatch:
    """Build VERA4 base fuel-path axial-state profiles.

    Maps axial roles to fuel-path universe replacements:
    - lower_shoulder_gap → water_pin (no fuel rod above nozzle)
    - lower_fuel_endplug → fuel_endplug (solid Zircaloy rod)
    - upper_fuel_endplug → fuel_endplug
    - fuel_upper_plenum → fuel_plenum (helium + cladding)
    - upper_shoulder_gap → water_pin
    """
    fuel_sources = ["fuel_active_r1", "fuel_active_r2"]
    return BasePathAxialProfilesPatch(
        profiles=[
            BasePathAxialProfilePatchItem(
                profile_id="vera4_fuel_path",
                path_family="fuel_rod",
                state_bindings=[
                    BasePathStateBindingPatchItem(
                        axial_role="lower_shoulder_gap",
                        source_universe_ids=fuel_sources,
                        replacement_universe_id="water_pin",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    ),
                    BasePathStateBindingPatchItem(
                        axial_role="lower_fuel_endplug",
                        source_universe_ids=fuel_sources,
                        replacement_universe_id="fuel_endplug",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    ),
                    BasePathStateBindingPatchItem(
                        axial_role="upper_fuel_endplug",
                        source_universe_ids=fuel_sources,
                        replacement_universe_id="fuel_endplug",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    ),
                    BasePathStateBindingPatchItem(
                        axial_role="fuel_upper_plenum",
                        source_universe_ids=fuel_sources,
                        replacement_universe_id="fuel_plenum",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    ),
                    BasePathStateBindingPatchItem(
                        axial_role="upper_shoulder_gap",
                        source_universe_ids=fuel_sources,
                        replacement_universe_id="water_pin",
                        preserve_path_roles=["guide_tube", "instrument_tube"],
                    ),
                ],
                source_note="VERA4 fuel-path axial-state switching",
            ),
        ],
    )


def build_vera4_settings() -> SettingsPatch:
    return SettingsPatch(
        source_strategy="active_fuel_box",
        source_requires_fissionable_constraint=True,
    )


def build_all_vera4_patches() -> list:
    """Build all deterministic VERA4 patches in execution order."""
    return [
        build_vera4_facts(),
        build_vera4_materials(),
        build_vera4_universes(),
        build_vera4_rcca_profile(),
        build_vera4_base_path_profiles(),
        build_vera4_axial_layers(),
        build_vera4_assembly_catalog(),
        build_vera4_core_layout(),
        build_vera4_spacer_grids(),
        build_vera4_settings(),
    ]
