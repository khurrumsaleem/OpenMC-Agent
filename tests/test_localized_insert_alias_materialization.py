"""Tests for localized-insert universe alias materialization.

Covers the gas-plenum transformation: when an annular absorber insert (e.g. a
Pyrex rod) declares both an absorber segment and an upper gas-plenum segment as
expected universe ids, fragment generation produces the absorber segment and
the alias step must materialize the plenum segment by replacing the absorber
material with gas (preserving structural tubes/cladding and the guide-tube
wall) rather than naively copying the absorber universe.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.plan_builder.universe_patch_pipeline import (
    materialize_localized_insert_universe_aliases,
)
from openmc_agent.plan_builder.validators import (
    PatchValidationContext,
    validate_patch,
)


def _poison_universe(uid: str = "localized_insert_pyrex_3B") -> dict[str, Any]:
    """A canonical Pyrex poison-segment universe (oracle-shaped)."""

    return {
        "universe_id": uid,
        "kind": "pyrex_rod",
        "cells": [
            {"id": "c0", "role": "gas_gap", "material_id": "helium", "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.214},
            {"id": "c1", "role": "cladding", "material_id": "ss304", "region_kind": "annulus", "r_min_cm": 0.214, "r_max_cm": 0.231},
            {"id": "c2", "role": "gas_gap", "material_id": "helium", "region_kind": "annulus", "r_min_cm": 0.231, "r_max_cm": 0.241},
            {"id": "c3", "role": "poison", "material_id": "pyrex_glass", "region_kind": "annulus", "r_min_cm": 0.241, "r_max_cm": 0.427},
            {"id": "c4", "role": "gas_gap", "material_id": "helium", "region_kind": "annulus", "r_min_cm": 0.427, "r_max_cm": 0.437},
            {"id": "c5", "role": "cladding", "material_id": "ss304", "region_kind": "annulus", "r_min_cm": 0.437, "r_max_cm": 0.484},
            {"id": "c6", "role": "inner_flow", "material_id": "coolant", "region_kind": "annulus", "r_min_cm": 0.484, "r_max_cm": 0.561},
            {"id": "c7", "role": "cladding", "material_id": "zircaloy4", "region_kind": "annulus", "r_min_cm": 0.561, "r_max_cm": 0.602},
            {"id": "c8", "role": "coolant", "material_id": "coolant", "region_kind": "background"},
        ],
        "metadata": {
            "localized_insert_requirement_ids": ["pyrex_3B"],
            "source_requirement_ids": ["localized_insert:pyrex_3B"],
        },
    }


def _facts_obj(expected_ids: list[str]) -> Any:
    return SimpleNamespace(
        localized_insert_requirements=[
            SimpleNamespace(
                requirement_id="pyrex_3B",
                expected_insert_universe_ids=expected_ids,
            ),
        ]
    )


class TestGasPlenumAliasMaterialization:
    def test_plenum_alias_replaces_absorber_with_gas(self):
        patch = {"patch_type": "universes", "universes": [_poison_universe()]}
        facts_obj = _facts_obj(["pyrex_poison_segment", "pyrex_plenum_segment"])

        issues = materialize_localized_insert_universe_aliases(patch, facts_obj=facts_obj)

        assert issues == []
        ids = [u["universe_id"] for u in patch["universes"]]
        assert "pyrex_poison_segment" in ids
        assert "pyrex_plenum_segment" in ids

        by_id = {u["universe_id"]: u for u in patch["universes"]}
        poison = by_id["pyrex_poison_segment"]
        plenum = by_id["pyrex_plenum_segment"]

        # The poison segment keeps the absorber material.
        assert any(c["material_id"] == "pyrex_glass" for c in poison["cells"])
        # The plenum segment must NOT contain the absorber material.
        assert all(c["material_id"] != "pyrex_glass" for c in plenum["cells"])
        # The former absorber cell is now gas.
        assert any(
            c["role"] == "gas_gap" and c["material_id"] == "helium" and c["r_min_cm"] == 0.241
            for c in plenum["cells"]
        )
        # Structural layers preserved.
        plenum_mats = {c["material_id"] for c in plenum["cells"]}
        assert {"ss304", "zircaloy4", "coolant"} <= plenum_mats
        assert plenum["metadata"]["gas_plenum_transform"] is True

    def test_plenum_alias_passes_universes_validation(self):
        patch = {"patch_type": "universes", "universes": [_poison_universe()]}
        facts_obj = _facts_obj(["pyrex_poison_segment", "pyrex_plenum_segment"])
        materialize_localized_insert_universe_aliases(patch, facts_obj=facts_obj)

        ids = [u["universe_id"] for u in patch["universes"]]
        parsed = parse_patch_content("universes", patch)
        result = validate_patch(
            parsed,
            context=PatchValidationContext(
                known_material_ids=["helium", "ss304", "pyrex_glass", "zircaloy4", "coolant"],
                known_universe_ids=ids,
            ),
        )
        errors = [i for i in result.issues if i.severity == "error"]
        assert errors == [(i.code, i.message) for i in []] or errors == []
        # Explicit: no plenum-contains-poison error.
        assert not any(i.code == "patch.universes.pyrex_plenum_contains_poison" for i in result.issues)

    def test_non_plenum_expected_id_is_plain_copy(self):
        patch = {"patch_type": "universes", "universes": [_poison_universe()]}
        # Only a poison-segment id: must remain a plain copy (with absorber).
        facts_obj = _facts_obj(["pyrex_poison_segment"])
        materialize_localized_insert_universe_aliases(patch, facts_obj=facts_obj)

        by_id = {u["universe_id"]: u for u in patch["universes"]}
        poison = by_id["pyrex_poison_segment"]
        assert any(c["material_id"] == "pyrex_glass" for c in poison["cells"])
        assert not poison["metadata"].get("gas_plenum_transform")

    def test_gas_material_uninferrable_leaves_plain_alias(self):
        """If the source has no gas cell, the transform declines (fail-closed)."""
        src = _poison_universe()
        # Remove all helium cells so no gas material can be inferred.
        src["cells"] = [c for c in src["cells"] if c["material_id"] != "helium"]
        patch = {"patch_type": "universes", "universes": [src]}
        facts_obj = _facts_obj(["pyrex_plenum_segment"])
        materialize_localized_insert_universe_aliases(patch, facts_obj=facts_obj)

        plenum = next(u for u in patch["universes"] if u["universe_id"] == "pyrex_plenum_segment")
        # Transform declined -> absorber remains (honest, validation will reject).
        assert any(c["material_id"] == "pyrex_glass" for c in plenum["cells"])
        assert not plenum["metadata"].get("gas_plenum_transform")
