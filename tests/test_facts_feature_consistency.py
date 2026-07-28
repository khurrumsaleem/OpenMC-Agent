from openmc_agent.plan_builder.planning_scope import planning_feature_contract
from openmc_agent.plan_builder.closed_loop.facts_consistency import run_facts_consistency_preflight


def test_source_critical_features_cannot_be_erased_from_facts():
    contract = planning_feature_contract({"feature_summary": {"multi_assembly_core": True, "has_spacer_grid": True, "has_localized_insert": True, "has_multi_segment_localized_insert": True, "has_control_state": True, "has_multiple_fuel_variants": True}})
    result = run_facts_consistency_preflight(feature_contract=contract, facts_patch={"patch_type":"facts", "model_scope":"single_assembly", "has_spacer_grids":False, "localized_insert_requirements":[], "fuel_variant_requirements":[]})
    codes = {item["code"] for item in result.issues}
    assert {"facts.model_scope_conflicts_with_planning_features", "facts.localized_insert_contract_missing", "facts.localized_insert_profile_contract_missing", "facts.spacer_grid_contract_missing", "facts.fuel_variant_contract_missing"} <= codes


def test_selected_benchmark_state_with_one_fuel_variant_is_closed():
    """A benchmark state label like 3B is not itself a multi-fuel contract."""
    contract = planning_feature_contract({"feature_summary": {"has_benchmark_variant": True}})
    assert contract.has_multiple_fuel_variants is False
    result = run_facts_consistency_preflight(
        feature_contract=contract,
        facts_patch={
            "patch_type": "facts",
            "model_scope": "single_assembly",
            "fuel_variant_requirements": [{"variant_id": "3B_fuel"}],
        },
    )
    assert "facts.fuel_variant_contract_missing" not in {
        item["code"] for item in result.issues
    }


def test_unknown_counts_do_not_downgrade_multi_scope():
    contract = planning_feature_contract({"feature_summary": {"multi_assembly_core": True}})
    result = run_facts_consistency_preflight(feature_contract=contract, facts_patch={"patch_type":"facts", "model_scope":"multi_assembly_core"})
    assert result.scope.value == "multi_assembly_core"
    assert "facts.multi_assembly_contract_incomplete" in {item["code"] for item in result.issues}


def test_blank_control_state_id_satisfies_source_control_state_contract():
    contract = planning_feature_contract({"feature_summary": {"has_localized_insert": True, "has_control_state": True}})
    result = run_facts_consistency_preflight(
        feature_contract=contract,
        facts_patch={
            "patch_type": "facts",
            "localized_insert_requirements": [
                {"requirement_id": "rcca", "insert_kind": "control_rod", "control_state_id": ""}
            ],
        },
    )
    assert "facts.control_state_contract_missing" not in {
        item["code"] for item in result.issues
    }
