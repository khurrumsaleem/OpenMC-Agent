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
