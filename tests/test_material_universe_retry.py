from __future__ import annotations

from openmc_agent.plan_builder.closed_loop.retry_owner_policy import retry_owner_policy
from openmc_agent.plan_builder.closed_loop.retry_controller import normalize_retry_request
from openmc_agent.plan_builder.closed_loop.retry_models import RetryTriggerOrigin
from openmc_agent.plan_builder.closed_loop.models import PlanGateId
from openmc_agent.plan_builder.state import PlanBuildState


def test_material_issues_route_to_material_owner():
    for code in (
        "material_universe.material_density_missing",
        "material_universe.required_density_missing",
        "material_universe.compound_isotope_unresolved",
        "materials.composition_fraction_sum_invalid",
        "qualification.schema_materials.composition_fraction_sum_invalid",
    ):
        policy = retry_owner_policy(code)
        assert policy is not None
        assert policy.owner_patch_types == ["materials"]


def test_universe_issues_route_to_universe_owner():
    for code in (
        "material_universe.required_universe_missing",
        "material_universe.localized_insert_universe_missing",
        "material_universe.protected_path_missing",
    ):
        policy = retry_owner_policy(code)
        assert policy is not None
        assert policy.owner_patch_types == ["universes"]


def test_mu_retry_does_not_make_facts_placement_or_axial_owner():
    policy = retry_owner_policy("material_universe.material_density_missing")
    assert policy is not None
    assert policy.owner_patch_types == ["materials"]
    assert "facts" not in policy.owner_patch_types
    assert "placement" not in policy.owner_patch_types
    assert "axial" not in policy.owner_patch_types


def test_missing_universe_retry_preserves_exact_target_id():
    state = PlanBuildState(state_id="mu-retry", requirement_text="r")
    request = normalize_retry_request(
        {
            "code": "material_universe.localized_insert_universe_missing",
            "issue_codes": ["material_universe.localized_insert_universe_missing"],
            "required_ids": ["u_pyrex_poison"],
            "affected_json_paths": ["/universes/u_pyrex_poison"],
        }, state=state, origin=RetryTriggerOrigin.MATERIAL_UNIVERSE_GATE,
    )
    assert request is not None
    assert request.gate_id is PlanGateId.MATERIAL_UNIVERSE
    assert request.owner_patch_types == ["universes"]
    assert request.targets[0].required_ids == ["u_pyrex_poison"]
    assert request.targets[0].affected_json_paths == ["/universes/u_pyrex_poison"]


def test_fuel_variant_material_mismatch_routes_to_universes_owner():
    state = PlanBuildState(state_id="mu-retry", requirement_text="r")
    request = normalize_retry_request(
        {
            "code": "material_universe.fuel_variant_material_mismatch",
            "issue_codes": ["material_universe.fuel_variant_material_mismatch"],
            "universe_id": "u_fuel_region2",
            "material_id": "mat_region1",
            "affected_json_paths": ["/universes/u_fuel_region2/cells/0/material_id"],
        },
        state=state,
        origin=RetryTriggerOrigin.MATERIAL_UNIVERSE_GATE,
    )
    assert request is not None
    assert request.owner_patch_types == ["universes"]
    assert request.consumer_ids == ["u_fuel_region2"]
    assert request.targets[0].required_ids == ["u_fuel_region2"]
    assert request.targets[0].affected_json_paths == [
        "/universes/u_fuel_region2/cells/0/material_id"
    ]
