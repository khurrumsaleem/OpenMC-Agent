"""Phase 4: Material-Universe deterministic preflight."""

from __future__ import annotations

from openmc_agent.plan_builder.closed_loop.material_universe_preflight import run_material_universe_preflight
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope


def _state(*, materials=None, universes=None, facts=None) -> PlanBuildState:
    state = PlanBuildState(state_id="mu-pre", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(patch_id="facts", patch_type="facts", content=facts or {"patch_type": "facts", "model_scope": "single_assembly"}, status="valid"))
    state.add_patch(PlanPatchEnvelope(patch_id="materials", patch_type="materials", content=materials or {"patch_type": "materials", "materials": [
        {"material_id": "fuel", "name": "f", "role": "fuel", "density_g_cm3": 10.0},
        {"material_id": "clad", "name": "c", "role": "cladding", "density_g_cm3": 6.5},
        {"material_id": "coolant", "name": "w", "role": "coolant", "density_g_cm3": 0.99},
    ]}, status="valid"))
    state.add_patch(PlanPatchEnvelope(patch_id="universes", patch_type="universes", content=universes or {"patch_type": "universes", "universes": [
        {"universe_id": "fuel", "kind": "fuel_pin", "cells": [
            {"id": "pellet", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder", "r_min_cm": 0, "r_max_cm": 0.4},
            {"id": "clad", "role": "clad", "material_id": "clad", "region_kind": "annulus", "r_min_cm": 0.4, "r_max_cm": 0.45},
            {"id": "bg", "role": "background", "material_id": "coolant", "region_kind": "background"},
        ]},
    ]}, status="valid"))
    return state


def _policy() -> PlanClosedLoopPolicy:
    return PlanClosedLoopPolicy(mode="controlled", gate_enabled={"facts": True, "material_universe": True})


def test_clean_state_passes_preflight() -> None:
    state = _state()
    result = run_material_universe_preflight(state=state, policy=_policy())
    assert result.ok
    assert not [i for i in result.issues if i.get("severity") == "error"]


def test_duplicate_material_detected() -> None:
    state = _state(materials={"patch_type": "materials", "materials": [
        {"material_id": "dup", "name": "a", "role": "fuel", "density_g_cm3": 10.0},
        {"material_id": "dup", "name": "b", "role": "fuel", "density_g_cm3": 10.0},
    ]}, universes={"patch_type": "universes", "universes": [{"universe_id": "u", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "dup", "region_kind": "cylinder"}]}]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_duplicate" in codes


def test_invalid_density_detected() -> None:
    state = _state(materials={"patch_type": "materials", "materials": [{"material_id": "fuel", "name": "f", "role": "fuel", "density_g_cm3": -1.0}]}, universes={"patch_type": "universes", "universes": [{"universe_id": "u", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder"}]}]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_density_invalid" in codes


def test_atom_frac_stoichiometric_ratio_detected_before_reviewer() -> None:
    state = _state(materials={"patch_type": "materials", "materials": [
        {
            "material_id": "coolant",
            "name": "water",
            "role": "coolant",
            "density_g_cm3": 0.743,
            "composition": {"H1": 2.0, "O16": 1.0},
            "composition_basis": "atom_frac",
            "composition_status": "confirmed",
        },
        {"material_id": "fuel", "name": "f", "role": "fuel", "density_g_cm3": 10.0},
        {"material_id": "clad", "name": "c", "role": "cladding", "density_g_cm3": 6.5},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "materials.composition_fraction_sum_invalid" in codes
    assert any(i.get("source_validator") for i in result.issues if i["code"] == "materials.composition_fraction_sum_invalid")


def test_weight_frac_percent_style_vector_detected_before_reviewer() -> None:
    state = _state(materials={"patch_type": "materials", "materials": [
        {
            "material_id": "steel",
            "name": "SS304",
            "role": "structural",
            "density_g_cm3": 7.9,
            "composition": {"Fe": 69.5, "Cr": 19.0, "Ni": 9.5, "Mn": 2.0},
            "composition_basis": "weight_frac",
            "composition_status": "approximate",
        },
        {"material_id": "fuel", "name": "f", "role": "fuel", "density_g_cm3": 10.0},
        {"material_id": "coolant", "name": "w", "role": "coolant", "density_g_cm3": 0.99},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    assert not result.ok
    assert any(
        i["code"] == "materials.composition_fraction_sum_invalid"
        and i.get("source_validator")
        for i in result.issues
    )


def test_shared_localized_insert_profile_metadata_covers_multiple_requirements() -> None:
    facts = {
        "patch_type": "facts",
        "model_scope": "multi_assembly_core",
        "localized_insert_requirements": [
            {
                "requirement_id": "plug_C",
                "insert_kind": "thimble_plug",
                "required_segment_roles": ["thimble_plug"],
                "expected_insert_universe_ids": ["u_thimble_plug"],
            },
            {
                "requirement_id": "plug_E",
                "insert_kind": "thimble_plug",
                "required_segment_roles": ["thimble_plug"],
                "expected_insert_universe_ids": ["u_thimble_plug"],
            },
        ],
    }
    universes = {
        "patch_type": "universes",
        "universes": [
            {
                "universe_id": "profile_shared_plug",
                "kind": "thimble_plug",
                "cells": [
                    {"id": "plug", "role": "structural", "material_id": "clad", "region_kind": "cylinder"},
                    {"id": "water", "role": "coolant", "material_id": "coolant", "region_kind": "background"},
                ],
                "metadata": {
                    "geometry_profile_id": "profile_shared_plug",
                    "localized_insert_requirement_id": "plug_C",
                    "localized_insert_requirement_ids": ["plug_C", "plug_E"],
                },
            }
        ],
    }
    state = _state(facts=facts, universes=universes)
    result = run_material_universe_preflight(state=state, policy=_policy())
    missing = [
        i for i in result.issues
        if i["code"] == "material_universe.localized_insert_universe_missing"
    ]
    assert missing == []


def test_shared_localized_insert_profile_inventory_fallback_covers_old_checkpoint_metadata() -> None:
    facts = {
        "patch_type": "facts",
        "model_scope": "multi_assembly_core",
        "localized_insert_requirements": [
            {
                "requirement_id": "plug_C",
                "insert_kind": "thimble_plug",
                "required_segment_roles": ["thimble_plug"],
                "expected_insert_universe_ids": ["u_thimble_plug"],
            },
            {
                "requirement_id": "plug_E",
                "insert_kind": "thimble_plug",
                "required_segment_roles": ["thimble_plug"],
                "expected_insert_universe_ids": ["u_thimble_plug"],
            },
        ],
    }
    universes = {
        "patch_type": "universes",
        "universes": [
            {
                "universe_id": "profile_shared_plug",
                "kind": "thimble_plug",
                "cells": [
                    {"id": "plug", "role": "structural", "material_id": "clad", "region_kind": "cylinder"},
                    {"id": "water", "role": "coolant", "material_id": "coolant", "region_kind": "background"},
                ],
                "metadata": {
                    "geometry_profile_id": "profile_shared_plug",
                    "localized_insert_requirement_id": "plug_E",
                },
            }
        ],
    }
    state = _state(facts=facts, universes=universes)
    state.metadata["planning_geometry_inventory"] = {
        "localized_insert_profiles": [
            {"insert_requirement_id": "plug_C", "profile_id": "profile_shared_plug"},
            {"insert_requirement_id": "plug_E", "profile_id": "profile_shared_plug"},
        ],
        "radial_profiles": [],
    }
    result = run_material_universe_preflight(state=state, policy=_policy())
    missing = [
        i for i in result.issues
        if i["code"] == "material_universe.localized_insert_universe_missing"
    ]
    assert missing == []


def test_unknown_material_reference_detected() -> None:
    state = _state(universes={"patch_type": "universes", "universes": [
        {"universe_id": "u", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "does_not_exist", "region_kind": "cylinder"}]},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_reference_missing" in codes


def test_duplicate_universe_detected() -> None:
    state = _state(universes={"patch_type": "universes", "universes": [
        {"universe_id": "dup", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder"}]},
        {"universe_id": "dup", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder"}]},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.universe_duplicate" in codes


def test_radial_overlap_detected() -> None:
    state = _state(universes={"patch_type": "universes", "universes": [
        {"universe_id": "u", "kind": "fuel_pin", "cells": [
            {"id": "a", "role": "fuel", "material_id": "fuel", "region_kind": "cylinder", "r_min_cm": 0, "r_max_cm": 0.5},
            {"id": "b", "role": "clad", "material_id": "clad", "region_kind": "annulus", "r_min_cm": 0.3, "r_max_cm": 0.6},
        ]},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.radial_overlap" in codes


def test_required_fuel_variant_material_missing_detected() -> None:
    state = _state(facts={"patch_type": "facts", "model_scope": "single_assembly", "fuel_variant_requirements": [{"variant_id": "v1", "enrichment_wt_percent": 3.0}]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.required_fuel_variant_material_missing" in codes


def test_fuel_variant_collapse_detected() -> None:
    """Two variants sharing the same source_variant_id → duplicate material error."""
    state = PlanBuildState(state_id="mu-collapse", requirement_text="r")
    state.add_patch(PlanPatchEnvelope(patch_id="facts", patch_type="facts", content={"patch_type": "facts", "model_scope": "single_assembly", "fuel_variant_requirements": [{"variant_id": "v1", "enrichment_wt_percent": 2.0}, {"variant_id": "v2", "enrichment_wt_percent": 3.0}]}, status="valid"))
    # Both variants map to materials with the SAME source_variant_id → duplicate.
    state.add_patch(PlanPatchEnvelope(patch_id="materials", patch_type="materials", content={"patch_type": "materials", "materials": [
        {"material_id": "v1_m", "name": "f1", "role": "fuel", "density_g_cm3": 10.0, "source_variant_id": "v1"},
        {"material_id": "v2_m", "name": "f2", "role": "fuel", "density_g_cm3": 10.0, "source_variant_id": "v1"},  # wrong variant!
    ]}, status="valid"))
    state.add_patch(PlanPatchEnvelope(patch_id="universes", patch_type="universes", content={"patch_type": "universes", "universes": [
        {"universe_id": "fuel", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "v1_m", "region_kind": "cylinder", "r_min_cm": 0, "r_max_cm": 0.4}]},
    ]}, status="valid"))
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    # v2 has no material; v1 has two materials.
    assert "material_universe.required_fuel_variant_material_missing" in codes or "material_universe.fuel_variant_material_duplicate" in codes


def test_fuel_universe_material_variant_mismatch_detected_from_material_reference() -> None:
    """Universe metadata variant must match the source_variant_id of fuel materials."""

    state = _state(
        facts={
            "patch_type": "facts",
            "model_scope": "multi_assembly_core",
            "fuel_variant_requirements": [
                {"variant_id": "region1", "enrichment_wt_percent": 2.11, "assembly_type_ids": ["C"]},
                {"variant_id": "region2", "enrichment_wt_percent": 2.619, "assembly_type_ids": ["E"]},
            ],
        },
        materials={
            "patch_type": "materials",
            "materials": [
                {"material_id": "fuel_r1", "name": "fuel r1", "role": "fuel", "density_g_cm3": 10.0, "source_variant_id": "region1"},
                {"material_id": "fuel_r2", "name": "fuel r2", "role": "fuel", "density_g_cm3": 10.0, "source_variant_id": "region2"},
                {"material_id": "clad", "name": "clad", "role": "cladding", "density_g_cm3": 6.5},
                {"material_id": "coolant", "name": "water", "role": "coolant", "density_g_cm3": 0.99},
            ],
        },
        universes={
            "patch_type": "universes",
            "universes": [
                {
                    "universe_id": "u_fuel_region1",
                    "kind": "fuel_pin",
                    "metadata": {"fuel_variant_id": "region1"},
                    "cells": [
                        {"id": "fuel", "role": "fuel", "material_id": "fuel_r2", "region_kind": "cylinder", "r_min_cm": 0, "r_max_cm": 0.4},
                        {"id": "clad", "role": "clad", "material_id": "clad", "region_kind": "annulus", "r_min_cm": 0.4, "r_max_cm": 0.45},
                        {"id": "bg", "role": "background", "material_id": "coolant", "region_kind": "background"},
                    ],
                },
                {
                    "universe_id": "u_fuel_region2",
                    "kind": "fuel_pin",
                    "metadata": {"fuel_variant_id": "region2"},
                    "cells": [
                        {"id": "fuel", "role": "fuel", "material_id": "fuel_r2", "region_kind": "cylinder", "r_min_cm": 0, "r_max_cm": 0.4},
                        {"id": "clad", "role": "clad", "material_id": "clad", "region_kind": "annulus", "r_min_cm": 0.4, "r_max_cm": 0.45},
                        {"id": "bg", "role": "background", "material_id": "coolant", "region_kind": "background"},
                    ],
                },
            ],
        },
    )

    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}

    assert "material_universe.fuel_variant_material_mismatch" in codes
    assert "material_universe.multiple_variants_in_one_universe" not in codes
    assert "material_universe.fuel_variant_collapsed" not in codes
    assert any(
        i["code"] == "material_universe.fuel_variant_material_mismatch"
        and i["universe_id"] == "u_fuel_region1"
        and i["expected_variant_id"] == "region1"
        and i["actual_variant_id"] == "region2"
        for i in result.issues
    )
    assert result.binding_view is not None
    assert {
        item.universe_id: item.fuel_variant_id
        for item in result.binding_view.universe_records
        if item.kind == "fuel_pin"
    } == {"u_fuel_region1": "region1", "u_fuel_region2": "region2"}


def test_role_mismatch_detected() -> None:
    """Fuel cell referencing a cladding material → role mismatch."""
    state = _state(universes={"patch_type": "universes", "universes": [
        {"universe_id": "u", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel", "material_id": "clad", "region_kind": "cylinder"}]},
    ]})
    result = run_material_universe_preflight(state=state, policy=_policy())
    codes = {i["code"] for i in result.issues}
    assert "material_universe.material_role_mismatch" in codes
