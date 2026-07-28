"""Tests for host-path equivalence validation (P2-FULLCORE-2D-A-HARDENING)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from openmc_agent.plan_builder.host_path_validation import (
    validate_replacement_host_equivalence,
    validate_all_replacements,
    HostEquivalenceReport,
)
from openmc_agent.plan_builder.validators import PatchValidationContext, validate_patch
from openmc_agent.plan_builder.patches import (
    CellLayerPatch,
    UniverseSpecPatch,
    UniversesPatch,
    AssemblyCatalogPatch,
    AssemblyTypePatchItem,
    AssemblyPinMapPatchItem,
    LocalizedInsertIntentPatchItem,
)


def _make_guide_tube() -> UniverseSpecPatch:
    return UniverseSpecPatch(
        universe_id="guide_tube", kind="guide_tube",
        cells=[
            CellLayerPatch(id="inner", role="inner_flow", material_id="water",
                           region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.5615),
            CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                           region_kind="cylinder", r_min_cm=0.5615, r_max_cm=0.6121),
            CellLayerPatch(id="coolant", role="coolant", material_id="water",
                           region_kind="background"),
        ],
    )


class TestHostEquivalence:
    def test_valid_replacement_passes(self):
        """A replacement that preserves the guide tube wall should pass."""
        host = _make_guide_tube()
        replacement = UniverseSpecPatch(
            universe_id="pyrex_poison", kind="pyrex_rod",
            cells=[
                CellLayerPatch(id="absorber", role="absorber", material_id="pyrex_glass",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.4),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=0.5615, r_max_cm=0.6121),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        )
        issues = validate_replacement_host_equivalence(replacement, host)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_missing_wall_fails(self):
        """A replacement that deletes the wall should fail."""
        host = _make_guide_tube()
        replacement = UniverseSpecPatch(
            universe_id="bad_absorber", kind="custom",
            cells=[
                CellLayerPatch(id="absorber", role="absorber", material_id="pyrex_glass",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.6),
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        )
        issues = validate_replacement_host_equivalence(replacement, host)
        codes = [i.code for i in issues]
        assert "fullcore.localized_insert_host_wall_unproven" in codes

    def test_missing_background_fails(self):
        """A replacement without background should report missing background."""
        host = _make_guide_tube()
        replacement = UniverseSpecPatch(
            universe_id="no_bg", kind="custom",
            cells=[
                CellLayerPatch(id="absorber", role="absorber", material_id="abs",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.6121),
            ],
        )
        issues = validate_replacement_host_equivalence(replacement, host)
        codes = [i.code for i in issues]
        assert "fullcore.localized_insert_background_missing" in codes

    def test_radius_mismatch_fails(self):
        """Wall radius mismatch should be reported."""
        host = _make_guide_tube()
        replacement = UniverseSpecPatch(
            universe_id="wrong_radius", kind="custom",
            cells=[
                CellLayerPatch(id="inner", role="inner_flow", material_id="water",
                               region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.5),
                CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                               region_kind="cylinder", r_min_cm=0.5, r_max_cm=0.7),  # wrong radius
                CellLayerPatch(id="coolant", role="coolant", material_id="water",
                               region_kind="background"),
            ],
        )
        issues = validate_replacement_host_equivalence(replacement, host)
        codes = [i.code for i in issues]
        assert "fullcore.localized_insert_outer_boundary_mismatch" in codes


class TestVERA4HostEquivalence:
    def test_vera4_all_replacements_valid(self):
        """All VERA4 replacement universes should pass host equivalence."""
        from vera4_base_fixture import build_all_vera4_patches
        patches = build_all_vera4_patches()
        universes_patch = next(p for p in patches if p.patch_type == "universes")
        catalog = next(p for p in patches if p.patch_type == "assembly_catalog")
        report = validate_all_replacements(universes_patch, catalog)
        errors = [i for i in report.issues if i.severity == "error"]
        assert len(errors) == 0, f"Host equivalence errors: {[i.code for i in errors]}"
        assert len(report.validated_pairs) > 0

    def test_vera4_pyrex_preserves_wall(self):
        """Pyrex universe should preserve the guide tube wall."""
        from vera4_base_fixture import build_vera4_universes
        uvs = build_vera4_universes()
        host = next(u for u in uvs.universes if u.universe_id == "guide_tube")
        pyrex = next(u for u in uvs.universes if u.universe_id == "pyrex_poison")
        issues = validate_replacement_host_equivalence(pyrex, host)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_vera4_pyrex_poison_is_annular_with_helium_center_and_gaps(self):
        """VERA4 Pyrex poison follows the input radial contract, not a solid rod."""
        from vera4_base_fixture import build_vera4_universes
        uvs = build_vera4_universes()
        pyrex = next(u for u in uvs.universes if u.universe_id == "pyrex_poison")
        result = validate_patch(uvs, PatchValidationContext())
        assert result.ok, [i.code for i in result.issues]

        cells = {c.id: c for c in pyrex.cells}
        assert cells["inner_helium"].material_id == "helium"
        assert cells["inner_helium"].r_min_cm == 0.0
        assert cells["inner_helium"].r_max_cm == 0.214
        assert cells["inner_tube"].material_id == "ss304"
        assert cells["inner_tube"].r_min_cm == 0.214
        assert cells["inner_gap"].material_id == "helium"
        assert cells["pyrex"].material_id == "pyrex_glass"
        assert cells["pyrex"].region_kind == "annulus"
        assert cells["pyrex"].r_min_cm == 0.241
        assert cells["pyrex"].r_max_cm == 0.427
        assert cells["outer_gap"].material_id == "helium"
        assert cells["outer_clad"].material_id == "ss304"
        assert cells["water_gap"].material_id == "water"

    def test_vera4_pyrex_upper_plenum_is_helium_not_pyrex_or_water_center(self):
        """The Pyrex upper gas cavity preserves SS304 tubes and does not contain poison glass."""
        from vera4_base_fixture import build_vera4_universes
        uvs = build_vera4_universes()
        plenum = next(u for u in uvs.universes if u.universe_id == "pyrex_plenum")
        result = validate_patch(uvs, PatchValidationContext())
        assert result.ok, [i.code for i in result.issues]

        cells = {c.id: c for c in plenum.cells}
        assert cells["inner_helium"].material_id == "helium"
        assert cells["upper_plenum_helium"].material_id == "helium"
        assert cells["upper_plenum_helium"].r_min_cm == 0.231
        assert cells["upper_plenum_helium"].r_max_cm == 0.437
        assert cells["inner_tube"].material_id == "ss304"
        assert cells["outer_clad"].material_id == "ss304"
        assert cells["water_gap"].material_id == "water"
        assert not any(c.material_id == "pyrex_glass" for c in plenum.cells)

    def test_solid_pyrex_rod_fails_deterministic_validation(self):
        """Regression: future LLM output must not pass with solid Pyrex or water-filled center."""
        bad = UniversesPatch(universes=[
            UniverseSpecPatch(
                universe_id="bad_pyrex", kind="pyrex_rod",
                cells=[
                    CellLayerPatch(id="pyrex", role="poison", material_id="pyrex_glass",
                                   region_kind="cylinder", r_min_cm=0.0, r_max_cm=0.427),
                    CellLayerPatch(id="water_gap", role="inner_flow", material_id="water",
                                   region_kind="annulus", r_min_cm=0.427, r_max_cm=0.561),
                    CellLayerPatch(id="wall", role="cladding", material_id="zircaloy4",
                                   region_kind="annulus", r_min_cm=0.561, r_max_cm=0.602),
                    CellLayerPatch(id="coolant", role="coolant", material_id="water",
                                   region_kind="background"),
                ],
            ),
        ])
        result = validate_patch(bad, PatchValidationContext())
        codes = {i.code for i in result.issues if i.severity == "error"}
        assert "patch.universes.pyrex_annular_poison_missing" in codes
        assert "patch.universes.pyrex_center_helium_missing" in codes

    def test_vera4_thimble_preserves_wall(self):
        from vera4_base_fixture import build_vera4_universes
        uvs = build_vera4_universes()
        host = next(u for u in uvs.universes if u.universe_id == "guide_tube")
        thimble = next(u for u in uvs.universes if u.universe_id == "thimble_plug")
        issues = validate_replacement_host_equivalence(thimble, host)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0

    def test_vera4_rcca_aic_preserves_wall(self):
        from vera4_base_fixture import build_vera4_universes
        uvs = build_vera4_universes()
        host = next(u for u in uvs.universes if u.universe_id == "guide_tube")
        rcca = next(u for u in uvs.universes if u.universe_id == "rcca_aic")
        issues = validate_replacement_host_equivalence(rcca, host)
        errors = [i for i in issues if i.severity == "error"]
        assert len(errors) == 0
