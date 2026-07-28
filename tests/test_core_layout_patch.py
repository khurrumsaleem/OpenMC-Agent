"""Tests for CoreLayoutPatch schema and parsing (P2-FULLCORE-1)."""

import pytest

from openmc_agent.plan_builder.patches import (
    CoreLayoutPatch,
    parse_patch_content,
)
from openmc_agent.schemas import CoreBoundarySpec


def test_core_layout_basic():
    layout = CoreLayoutPatch(
        shape=(2, 2),
        assembly_pitch_cm=21.5,
        assembly_pattern=[
            ["type_a", "type_b"],
            ["type_b", "type_a"],
        ],
        expected_assembly_type_counts={"type_a": 2, "type_b": 2},
        boundary="reflective",
    )
    assert layout.shape == (2, 2)
    assert len(layout.assembly_pattern) == 2
    assert layout.assembly_pitch_cm == 21.5
    assert layout.boundary == "reflective"


def test_core_layout_extra_forbid():
    with pytest.raises(Exception):
        CoreLayoutPatch(
            shape=(1, 1),
            assembly_pattern=[["type_a"]],
            bogus_key=True,
        )


def test_parse_patch_content_core_layout():
    content = {
        "patch_type": "core_layout",
        "shape": [2, 2],
        "assembly_pitch_cm": 21.5,
        "assembly_pattern": [["a", "b"], ["b", "a"]],
        "expected_assembly_type_counts": {"a": 2, "b": 2},
        "boundary": "reflective",
    }
    patch = parse_patch_content("core_layout", content)
    assert isinstance(patch, CoreLayoutPatch)
    assert patch.shape == (2, 2)
    assert patch.assembly_pattern[0] == ["a", "b"]


def test_core_layout_accepts_explicit_mixed_boundary_conditions():
    layout = CoreLayoutPatch(
        shape=(1, 1),
        assembly_pattern=[["type_a"]],
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

    assert layout.boundary_conditions is not None
    assert layout.boundary_conditions.zmin == "vacuum"


def test_core_layout_outer_assembly_type():
    layout = CoreLayoutPatch(
        shape=(3, 3),
        assembly_pitch_cm=21.5,
        assembly_pattern=[
            ["outer", "edge", "outer"],
            ["edge", "center", "edge"],
            ["outer", "edge", "outer"],
        ],
        outer_assembly_type_id="outer",
        boundary="vacuum",
    )
    assert layout.outer_assembly_type_id == "outer"


def test_core_layout_default_boundary():
    layout = CoreLayoutPatch(
        shape=(1, 1),
        assembly_pattern=[["type_a"]],
    )
    assert layout.boundary == "vacuum"
    assert layout.core_lattice_id == "core_lattice"
