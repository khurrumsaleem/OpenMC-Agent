"""Tests for hierarchical core assembler (P2-FULLCORE-2A)."""

from openmc_agent.plan_builder.patches import (
    AssemblyCatalogPatch,
    AssemblyPinMapPatchItem,
    AssemblyTypePatchItem,
    CoreLayoutPatch,
)
from openmc_agent.plan_builder.hierarchical_assembler import (
    HierarchicalCoreAssemblyResult,
    build_hierarchical_core_plan,
    lift_single_pin_map_to_catalog,
    assemble_assembly_templates,
    assemble_core_lattice,
)
from openmc_agent.schemas import CoreBoundarySpec


def _make_catalog():
    return AssemblyCatalogPatch(
        assembly_types=[
            AssemblyTypePatchItem(
                assembly_type_id="type_a",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(3, 3),
                    default_universe_id="fuel",
                    guide_tube_coords=[(1, 1)],
                ),
            ),
            AssemblyTypePatchItem(
                assembly_type_id="type_b",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(3, 3),
                    default_universe_id="fuel",
                    guide_tube_coords=[(0, 0), (2, 2)],
                ),
            ),
        ],
    )


def _make_layout():
    return CoreLayoutPatch(
        shape=(2, 2),
        assembly_pitch_cm=21.5,
        assembly_pattern=[["type_a", "type_b"], ["type_b", "type_a"]],
        boundary="reflective",
    )


def test_build_hierarchical_core_plan():
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    assert isinstance(result, HierarchicalCoreAssemblyResult)
    assert len(result.pin_lattices) == 2
    assert len(result.assembly_specs) == 2
    assert len(result.core_lattices) == 1
    assert result.core_spec is not None


def test_pin_lattice_ids_per_type():
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    lattice_ids = [lat.id for lat in result.pin_lattices]
    assert "assembly_lattice__type_a" in lattice_ids
    assert "assembly_lattice__type_b" in lattice_ids


def test_core_lattice_shape():
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    core_lat = result.core_lattices[0]
    assert core_lat.shape == (2, 2)
    assert len(core_lat.universe_pattern) == 2
    assert len(core_lat.universe_pattern[0]) == 2


def test_core_count_aggregation():
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    agg = result.core_count_aggregation
    assert agg.total_assembly_instances == 4
    # type_a: 8 fuel × 2 = 16, type_b: 7 fuel × 2 = 14 → total = 30
    assert agg.core_total_for_role("fuel_pin") == 30


def test_assembly_wrapper_universes_created():
    """Each assembly type must have real UniverseSpec and CellSpec."""
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    # 2 assembly types + 1 moderator outer universe
    wrapper_uvs = [u for u in result.assembly_universes if u.id.startswith("assembly_universe__")]
    assert len(wrapper_uvs) == 2
    assert len(result.assembly_wrapper_cells) >= 2
    uv_ids = {u.id for u in wrapper_uvs}
    assert "assembly_universe__type_a" in uv_ids
    assert "assembly_universe__type_b" in uv_ids
    cell_ids = {c.id for c in result.assembly_wrapper_cells}
    assert "assembly_wrapper_cell__type_a" in cell_ids
    assert "assembly_wrapper_cell__type_b" in cell_ids


def test_assembly_boundary_transmission():
    """Internal assembly boundaries must be transmission, not reflective."""
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    for asm in result.assembly_specs:
        assert asm.boundary == "transmission", f"Assembly {asm.id} has boundary={asm.boundary}"


def test_core_lattice_centered():
    """Core lattice should be centered at origin."""
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    core_lat = result.core_lattices[0]
    assert core_lat.center_cm == (0.0, 0.0)
    # lower_left should be negative (centered)
    assert core_lat.lower_left_cm is not None
    assert core_lat.lower_left_cm[0] < 0
    assert core_lat.lower_left_cm[1] < 0


def test_lift_single_pin_map_to_catalog():
    from openmc_agent.plan_builder.patches import PinMapPatch
    pin_map = PinMapPatch(
        lattice_size=(3, 3),
        default_universe_id="fuel",
        guide_tube_coords=[(1, 1)],
    )
    catalog = lift_single_pin_map_to_catalog(pin_map)
    assert len(catalog.assembly_types) == 1
    assert catalog.assembly_types[0].assembly_type_id == "assembly_type_1"
    assert catalog.assembly_types[0].pin_map.lattice_size == (3, 3)
    assert catalog.source_note is not None


def test_assemble_assembly_templates_returns_summaries():
    catalog = _make_catalog()
    lattices, assemblies, w_universes, w_cells, uv_ids, summaries, reports, mod_uv, mod_cell = (
        assemble_assembly_templates(catalog)
    )
    assert len(lattices) == 2
    assert len(assemblies) == 2
    assert len(w_universes) == 2
    assert len(w_cells) == 2
    assert mod_uv is not None
    assert mod_cell is not None
    assert "type_a" in summaries
    assert "type_b" in summaries
    assert summaries["type_a"].fuel_pin_count == 8
    assert summaries["type_b"].fuel_pin_count == 7


def test_core_lattice_assembly_universe_ids():
    catalog = _make_catalog()
    layout = _make_layout()
    assembly_uvs = {"type_a": "universe_a", "type_b": "universe_b"}
    core_lat = assemble_core_lattice(layout, assembly_uvs)
    assert core_lat.universe_pattern[0][0] == "universe_a"
    assert core_lat.universe_pattern[0][1] == "universe_b"


def test_core_layout_boundary_conditions_are_carried_to_core_spec():
    catalog = _make_catalog()
    layout = CoreLayoutPatch(
        shape=(2, 2),
        assembly_pitch_cm=21.5,
        assembly_pattern=[["type_a", "type_b"], ["type_b", "type_a"]],
        boundary="reflective",
        boundary_conditions=CoreBoundarySpec(
            xmin="reflective",
            xmax="reflective",
            ymin="reflective",
            ymax="reflective",
            zmin="vacuum",
            zmax="vacuum",
        ),
    )

    result = build_hierarchical_core_plan(catalog, layout, facts=None)

    assert result.core_spec is not None
    assert result.core_spec.boundary == "mixed"
    assert result.core_spec.boundary_conditions is not None
    assert result.core_spec.boundary_conditions.zmax == "vacuum"
    assert result.summary["core_boundary_conditions"] == {
        "xmin": "reflective",
        "xmax": "reflective",
        "ymin": "reflective",
        "ymax": "reflective",
        "zmin": "vacuum",
        "zmax": "vacuum",
    }


def test_base_lattice_no_localized_inserts():
    """Base pin lattice must NOT contain localized insert universes."""
    from openmc_agent.plan_builder.patches import LocalizedInsertIntentPatchItem
    catalog = AssemblyCatalogPatch(
        assembly_types=[
            AssemblyTypePatchItem(
                assembly_type_id="type_a",
                pin_map=AssemblyPinMapPatchItem(
                    lattice_size=(3, 3),
                    default_universe_id="fuel",
                    guide_tube_coords=[(1, 1)],
                    localized_insert_intents=[
                        LocalizedInsertIntentPatchItem(
                            insert_id="pyrex1",
                            insert_kind="pyrex_rod",
                            insert_universe_id="pyrex_universe",
                            coordinates=[(1, 1)],
                        ),
                    ],
                ),
            ),
        ],
    )
    layout = CoreLayoutPatch(
        shape=(1, 1),
        assembly_pattern=[["type_a"]],
        boundary="reflective",
    )
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    pin_lattice = result.pin_lattices[0]
    # Check no position contains "pyrex_universe"
    for row in pin_lattice.universe_pattern:
        for cell in row:
            assert cell != "pyrex_universe", "Base lattice contains localized insert!"


def test_summary_contains_key_info():
    catalog = _make_catalog()
    layout = _make_layout()
    result = build_hierarchical_core_plan(catalog, layout, facts=None)
    s = result.summary
    assert s["assembly_type_count"] == 2
    assert s["total_instances"] == 4
    assert s["internal_assembly_boundary"] == "transmission"
    assert s["localized_inserts_in_base_lattice"] is False
