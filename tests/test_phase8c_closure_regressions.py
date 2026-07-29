"""Regression coverage for v20 fuel binding, poison isolation, and axial loading."""

from __future__ import annotations

from openmc_agent.plan_builder.assembler import _auto_attach_component_profile_loadings
from openmc_agent.plan_builder.materials_patch_normalization import (
    normalize_materials_patch_content,
)
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
from openmc_agent.plan_builder.universes_patch_normalization import (
    normalize_universes_patch_content,
)
from openmc_agent.schemas import (
    AxialLayerSpec,
    FillRefSpec,
    LatticeLoadingSpec,
    LatticeTransformationOperation,
)


def _facts_with_variants(*variant_ids: str) -> dict:
    return {
        "patch_type": "facts",
        "model_scope": "single_assembly",
        "fuel_variant_requirements": [
            {"variant_id": variant_id, "enrichment_wt_percent": 3.0}
            for variant_id in variant_ids
        ],
    }


def test_non_fuel_source_variant_id_is_stripped_and_fuel_is_canonicalized() -> None:
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content=_facts_with_variants("3B_fuel"), status="valid",
    ))
    result = normalize_materials_patch_content({
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel_3b", "name": "fuel", "role": "fuel", "density_g_cm3": 10.0, "source_variant_id": "3B"},
            {"material_id": "coolant_3b", "name": "water", "role": "coolant", "density_g_cm3": 0.7, "source_variant_id": "3B"},
        ],
    }, state=state)
    materials = {m["material_id"]: m for m in result.content["materials"]}
    assert materials["fuel_3b"]["source_variant_id"] == "3B_fuel"
    assert materials["coolant_3b"]["source_variant_id"] is None
    assert {op["operation"] for op in result.operations} >= {
        "fuel_source_variant_id_canonicalized",
        "non_fuel_source_variant_id_stripped",
    }


def test_facts_undeclared_poison_universes_are_stripped_via_state_hook() -> None:
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts",
        content=_facts_with_variants(), status="valid",
    ))
    patch = PlanPatchEnvelope(
        patch_id="universes", patch_type="universes", status="valid",
        content={
            "patch_type": "universes",
            "universes": [
                {"universe_id": "fuel", "kind": "fuel_pin", "cells": []},
                {"universe_id": "hallucinated_pyrex", "kind": "pyrex_rod", "cells": []},
                {"universe_id": "hallucinated_aic", "kind": "aic_rod", "cells": []},
                {"universe_id": "hallucinated_b4c", "kind": "b4c_rod", "cells": []},
                {"universe_id": "hallucinated_waba", "kind": "waba_rod", "cells": []},
                {"universe_id": "hallucinated_ifba", "kind": "ifba_pin", "cells": []},
                {"universe_id": "hallucinated_gad", "kind": "gadolinia_pin", "cells": []},
                {"universe_id": "hallucinated_thimble", "kind": "instrument_thimble", "cells": []},
            ],
        },
    )
    state.add_patch(patch)
    assert [u["universe_id"] for u in patch.content["universes"]] == ["fuel"]
    assert any(
        op["operation"] == "spurious_poison_universe_stripped"
        for op in patch.metadata["deterministic_normalizations"]
    )


def test_declared_pyrex_universe_is_preserved() -> None:
    state = PlanBuildState(state_id="s", requirement_text="r")
    facts = _facts_with_variants()
    facts["localized_insert_requirements"] = [{"insert_kind": "pyrex_rod"}]
    state.add_patch(PlanPatchEnvelope(
        patch_id="facts", patch_type="facts", content=facts, status="valid",
    ))
    content = {
        "patch_type": "universes",
        "universes": [{"universe_id": "pyrex", "kind": "pyrex_rod", "cells": []}],
    }
    result = normalize_universes_patch_content(content, state=state)
    assert result.content == content
    assert not result.changed


def test_component_profile_loading_is_auto_attached_to_matching_layer() -> None:
    layers = [
        AxialLayerSpec(
            id="upper_plenum", name="upper_plenum", z_min_cm=10.0,
            z_max_cm=20.0, fill=FillRefSpec(type="lattice", id="assembly_lattice"),
        ),
        AxialLayerSpec(
            id="active_fuel", name="active_fuel", z_min_cm=0.0,
            z_max_cm=10.0, fill=FillRefSpec(type="lattice", id="assembly_lattice"),
        ),
    ]
    loading = LatticeLoadingSpec(
        id="plenum_loading", base_lattice_id="assembly_lattice",
        purpose="Replace fuel pin family with plenum universe.",
        transformations=[LatticeTransformationOperation(
            operation_id="replace_plenum", operation_kind="replace_universe_family",
            replacement_universe_id="fuel_pin_plenum", source_universe_id="fuel_pin",
        )],
    )
    normalized, issues = _auto_attach_component_profile_loadings(layers, [loading])
    plenum = next(layer for layer in normalized if layer.id == "upper_plenum")
    active = next(layer for layer in normalized if layer.id == "active_fuel")
    assert plenum.loading_ids == ["plenum_loading"]
    assert active.loading_ids == []
    assert [issue.code for issue in issues] == [
        "assembly.component_profile_loading_auto_attached"
    ]


# ---------------------------------------------------------------------------
# v21 regressions: Facts fallback requirements + dynamic MU owner
# ---------------------------------------------------------------------------


def test_facts_fallback_derives_roles_from_inserts_and_geometry() -> None:
    """When Facts has material_roles=[] but declares fuel variants + localized
    inserts (pyrex_rod, thimble_plug), the Facts fallback must still produce
    requirements for poison, structural, cladding, coolant, and gas — not
    just fuel."""
    from openmc_agent.plan_builder.material_requirements import (
        extract_material_requirements_from_facts,
    )
    from openmc_agent.plan_builder.patches import parse_patch_content

    facts = parse_patch_content("facts", {
        "patch_type": "facts",
        "model_scope": "single_assembly",
        "fuel_variant_requirements": [{"variant_id": "UO2_2.619", "enrichment_wt_percent": 2.619}],
        "material_roles": [],
        "localized_insert_requirements": [
            {"requirement_id": "pyrex_3B", "insert_kind": "pyrex_rod", "host_kind": "guide_tube"},
            {"requirement_id": "thimble_plug_3B", "insert_kind": "thimble_plug", "host_kind": "guide_tube"},
        ],
        "expected_pin_count": 264,
    })
    reqs = extract_material_requirements_from_facts(facts)
    roles = {r.role for r in reqs.requirements}
    assert "fuel_UO2" in roles
    assert "poison" in roles
    assert "structural" in roles
    assert "cladding" in roles
    assert "coolant" in roles
    assert "gas" in roles


def test_material_reference_missing_routes_to_materials_owner() -> None:
    """When a universe cell references a material_id that doesn't exist in the
    Materials patch, the MU preflight issue should carry
    owner_patch_type='materials' so the retry targets Materials, not
    Universes."""
    from openmc_agent.plan_builder.closed_loop.material_universe_preflight import (
        run_material_universe_preflight,
    )
    from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy

    materials = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel", "name": "f", "role": "fuel", "density_g_cm3": 10.0},
        ],
    }
    universes = {
        "patch_type": "universes",
        "universes": [
            {"universe_id": "u1", "kind": "fuel_pin", "cells": [
                {"id": "c1", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder"},
                {"id": "c2", "role": "clad", "material_id": "zircaloy4", "region_kind": "cylinder"},
            ]},
        ],
    }
    facts = {"patch_type": "facts", "model_scope": "single_assembly"}
    state = PlanBuildState(state_id="s", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(patch_id="facts", patch_type="facts", content=facts, status="valid"))
    state.add_patch(PlanPatchEnvelope(patch_id="materials", patch_type="materials", content=materials, status="valid"))
    state.add_patch(PlanPatchEnvelope(patch_id="universes", patch_type="universes", content=universes, status="valid"))

    result = run_material_universe_preflight(state=state, policy=PlanClosedLoopPolicy())
    ref_missing = [i for i in result.issues if i["code"] == "material_universe.material_reference_missing"]
    assert ref_missing
    assert all(i.get("owner_patch_type") == "materials" for i in ref_missing)
