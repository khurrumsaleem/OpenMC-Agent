"""Tests for the deterministic annular-insert universe construction oracle.

Covers:

* Parsing a markdown radial cross-section table into ordered layers.
* Parsing a semicolon ``role=rmin-rmax cm`` note.
* Material binding against the available material catalog.
* The constructed universe passes the full fragment qualification pipeline
  (all four pyrex annular validators + the concentric radial profile).
* Fail-closed behaviour: no radial table, unbindable material, non-annular kind.
* End-to-end oracle rescue when the LLM fragment fails.
"""

from __future__ import annotations

from typing import Any

from openmc_agent.plan_builder.closed_loop.annular_insert_universe_oracle import (
    AnnularInsertOracleProposal,
    build_annular_insert_universe,
    extract_radial_layers,
    propose_annular_insert_universe,
)
from openmc_agent.plan_builder.patches import MaterialSpecPatch
from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseDefinitionFragment,
    UniverseManifestItem,
)
from openmc_agent.plan_builder.universe_fragment_qualification import (
    qualify_universe_fragment,
)


VERA3_PYREX_TABLE_REQUIREMENT = """
### 12.2 毒物有效段的完整嵌套径向结构

| 半径范围 | 材料 |
|---|---|
| `0–0.214 cm` | 氦气中心腔 |
| `0.214–0.231 cm` | SS304 内管 |
| `0.231–0.241 cm` | 氦气内间隙 |
| `0.241–0.427 cm` | Pyrex 环形毒物 |
| `0.427–0.437 cm` | 氦气外间隙 |
| `0.437–0.484 cm` | SS304 外包壳 |
| `0.484–0.561 cm` | 含硼水：Pyrex 棒与导向管之间的水隙 |
| `0.561–0.602 cm` | Zircaloy-4 原导向管壁 |
| `r≥0.602 cm` 至栅元边界 | 含硼水 |
"""

SEMICOLON_NOTE_REQUIREMENT = (
    "pyrex radial profile: "
    "inner_helium=0-0.214 cm; "
    "ss304_inner_tube=0.214-0.231 cm; "
    "helium_gap=0.231-0.241 cm; "
    "annular_pyrex=0.241-0.427 cm; "
    "helium_gap_outer=0.427-0.437 cm; "
    "ss304_clad=0.437-0.484 cm; "
    "water_gap=0.484-0.561 cm; "
    "zircaloy4_wall=0.561-0.602 cm"
)


def _materials() -> list[MaterialSpecPatch]:
    specs = [
        ("helium", "Helium", "gap"),
        ("pyrex_glass", "Pyrex Borosilicate Glass", "poison"),
        ("ss304", "SS-304", "structural"),
        ("zircaloy4", "Zircaloy-4", "cladding"),
        ("coolant", "Borated Water", "coolant"),
    ]
    return [
        MaterialSpecPatch(material_id=mid, name=name, role=role)
        for mid, name, role in specs
    ]


def _manifest_item(*, universe_id: str = "u_pyrex", kind: str = "pyrex_rod") -> UniverseManifestItem:
    item = UniverseManifestItem(
        universe_id=universe_id,
        kind=kind,
        required_cell_roles=[],
        required_material_roles=["poison"],
        required_material_ids=[],
        protected_through_path_roles=[],
        source_requirement_ids=[f"req:{universe_id}"],
    )
    item.recompute_contract_hash()
    return item


class TestRadialLayerExtraction:
    def test_markdown_table_parsed(self):
        parsed = extract_radial_layers(VERA3_PYREX_TABLE_REQUIREMENT)
        assert parsed is not None
        rows, _ = parsed
        assert len(rows) >= 8
        # Center cavity starts at r=0; background region is last.
        assert rows[0].r_min == 0.0
        assert rows[0].r_max == 0.214
        assert rows[-1].is_background

    def test_semicolon_note_parsed(self):
        parsed = extract_radial_layers(SEMICOLON_NOTE_REQUIREMENT)
        assert parsed is not None
        rows, _ = parsed
        assert len(rows) >= 7
        assert rows[0].r_min == 0.0

    def test_no_radial_structure_returns_none(self):
        assert extract_radial_layers("A plain description with no radii.") is None


class TestAnnularInsertUniverseConstruction:
    def test_constructed_universe_has_canonical_annular_structure(self):
        parsed = extract_radial_layers(VERA3_PYREX_TABLE_REQUIREMENT)
        assert parsed is not None
        rows, _ = parsed
        built = build_annular_insert_universe(
            universe_id="u_pyrex", kind="pyrex_rod", rows=rows, materials=_materials(),
        )
        assert built is not None
        universe, _warnings = built
        cells = universe.cells
        # Poison must be an annulus with r_min > 0.
        poison = [c for c in cells if c.role == "poison"]
        assert len(poison) == 1
        assert poison[0].region_kind == "annulus"
        assert poison[0].r_min_cm is not None and poison[0].r_min_cm > 0.0
        # Center cavity must be a cylinder starting at r=0 with helium material.
        center = [c for c in cells if c.region_kind == "cylinder"]
        assert len(center) == 1
        assert center[0].r_min_cm == 0.0
        assert center[0].material_id == "helium"
        # At least 3 helium (gas_gap) cells and >3 non-background cells.
        assert sum(1 for c in cells if c.role == "gas_gap") >= 3
        assert sum(1 for c in cells if c.region_kind != "background") > 3
        # Trailing background coolant.
        assert cells[-1].region_kind == "background"

    def test_materials_bound_to_catalog(self):
        parsed = extract_radial_layers(VERA3_PYREX_TABLE_REQUIREMENT)
        rows = parsed[0]
        built = build_annular_insert_universe(
            universe_id="u_pyrex", kind="pyrex_rod", rows=rows, materials=_materials(),
        )
        ids = {c.material_id for c in built[0].cells}
        assert ids <= {"helium", "pyrex_glass", "ss304", "zircaloy4", "coolant"}

    def test_ss304_outer_clad_not_misclassified_as_zircaloy(self):
        """The 'SS304 外包壳' layer must bind to ss304, not zircaloy4.

        Regression: a generic 'clad' keyword in the zircaloy group previously
        captured SS304 outer-clad labels because zircaloy is matched before
        ss304.
        """
        parsed = extract_radial_layers(VERA3_PYREX_TABLE_REQUIREMENT)
        rows = parsed[0]
        built = build_annular_insert_universe(
            universe_id="u_pyrex", kind="pyrex_rod", rows=rows, materials=_materials(),
        )
        outer_clad = [c for c in built[0].cells if c.r_min_cm == 0.437]
        assert len(outer_clad) == 1
        assert outer_clad[0].material_id == "ss304"

    def test_unbindable_material_returns_none(self):
        parsed = extract_radial_layers(VERA3_PYREX_TABLE_REQUIREMENT)
        rows = parsed[0]
        # Catalog missing the poison material -> fail-closed.
        partial = [m for m in _materials() if m.role != "poison"]
        assert build_annular_insert_universe(
            universe_id="u_pyrex", kind="pyrex_rod", rows=rows, materials=partial,
        ) is None


class TestProposalAndQualification:
    def test_proposal_passes_full_qualification(self):
        proposal = propose_annular_insert_universe(
            manifest_item=_manifest_item(),
            requirement=VERA3_PYREX_TABLE_REQUIREMENT,
            materials=_materials(),
        )
        assert proposal.ok
        assert proposal.universe_data is not None

        fragment = UniverseDefinitionFragment(
            universe_id="u_pyrex",
            universe=proposal.universe_data,
            manifest_contract_hash=_manifest_item().contract_hash,
        )
        result = qualify_universe_fragment(
            manifest_item=_manifest_item(),
            fragment=fragment,
            known_material_ids={m.material_id for m in _materials()},
            material_roles_by_id={m.material_id: m.role for m in _materials()},
            material_source_variants_by_id={},
        )
        assert result.ok, [
            (i.code, i.message) for i in result.issues if i.severity == "error"
        ]

    def test_proposal_fail_closed_without_radial_structure(self):
        proposal = propose_annular_insert_universe(
            manifest_item=_manifest_item(),
            requirement="No radial geometry here.",
            materials=_materials(),
        )
        assert not proposal.ok
        assert proposal.reason == "no_radial_cross_section_found"

    def test_proposal_declines_non_annular_kind(self):
        proposal = propose_annular_insert_universe(
            manifest_item=_manifest_item(kind="fuel_pin"),
            requirement=VERA3_PYREX_TABLE_REQUIREMENT,
            materials=_materials(),
        )
        assert not proposal.ok
        assert proposal.reason.startswith("kind_not_annular_insert")


class TestOracleRescuesFailedFragment:
    """End-to-end: when the LLM fragment fails, the oracle rescues it."""

    def test_oracle_record_constructed_after_llm_failure(self):
        from openmc_agent.plan_builder.universe_patch_pipeline import (
            _try_annular_insert_oracle,
        )

        record, telemetry = _try_annular_insert_oracle(
            item=_manifest_item(universe_id="u_pyrex"),
            requirement=VERA3_PYREX_TABLE_REQUIREMENT,
            materials_obj=_materials_catalog_obj(),
            known_material_ids={m.material_id for m in _materials()},
            material_roles_by_id={m.material_id: m.role for m in _materials()},
            material_source_variants_by_id={},
            prior_failures=["patch.universes.pyrex_annular_poison_missing: solid pyrex"],
        )
        assert record is not None
        assert record.metadata["constructed_by_oracle"] is True
        assert telemetry["ok"] is True
        assert telemetry["qualification_ok"] is True
        # The record's universe must satisfy the annular poison contract.
        poison = [c for c in record.universe["cells"] if c["role"] == "poison"]
        assert poison[0]["region_kind"] == "annulus"
        assert poison[0]["r_min_cm"] > 0.0

    def test_oracle_declines_when_kind_not_annular(self):
        from openmc_agent.plan_builder.universe_patch_pipeline import (
            _try_annular_insert_oracle,
        )

        record, telemetry = _try_annular_insert_oracle(
            item=_manifest_item(universe_id="u_fuel", kind="fuel_pin"),
            requirement=VERA3_PYREX_TABLE_REQUIREMENT,
            materials_obj=_materials_catalog_obj(),
            known_material_ids={m.material_id for m in _materials()},
            material_roles_by_id={m.material_id: m.role for m in _materials()},
            material_source_variants_by_id={},
            prior_failures=[],
        )
        assert record is None
        assert telemetry["ok"] is False


class _Catalog:
    """Minimal stand-in exposing ``.materials`` like MaterialsPatch."""

    def __init__(self, materials: list[MaterialSpecPatch]) -> None:
        self.materials = materials


def _materials_catalog_obj() -> _Catalog:
    return _Catalog(_materials())
