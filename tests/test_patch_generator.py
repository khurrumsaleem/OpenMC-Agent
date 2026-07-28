"""Tests for LLM patch generator with fake LLM (Phase 4)."""

from __future__ import annotations

import json

import pytest

from openmc_agent.plan_builder.assembler import assemble_simulation_plan_from_patches
from openmc_agent.plan_builder.patches import (
    FactsPatch,
    MaterialsPatch,
    PinMapPatch,
    parse_patch_content,
)
from openmc_agent.plan_builder.patch_generator import (
    FakePatchLLM,
    PatchGenerationContext,
    RetryPatchGenerationContext,
    generate_patch,
    parse_llm_patch_json,
)
from openmc_agent.plan_builder.patch_prompts import build_patch_prompt
from openmc_agent.plan_builder.closed_loop.fingerprints import compute_candidate_hash
from openmc_agent.plan_builder.patches import PatchParseError
from openmc_agent.plan_builder.state import (
    PlanBuildState,
    PlanPatchEnvelope,
    add_validated_patch_to_state,
    generate_and_add_patch_to_state,
)
from openmc_agent.plan_builder.validators import validate_patch


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _patch_json(patch_type: str, **kwargs: object) -> str:
    """Build a minimal valid patch JSON string for testing."""
    base = {"patch_type": patch_type}
    base.update(kwargs)
    return json.dumps(base)


# ---------------------------------------------------------------------------
# 2. FactsPatch generation success
# ---------------------------------------------------------------------------


def test_facts_generation_success() -> None:
    raw = json.dumps({
        "patch_type": "facts",
        "benchmark_id": "VERA3",
        "selected_variant": "3A",
        "lattice_size": [17, 17],
        "pin_pitch_cm": 1.26,
        "has_axial_geometry": True,
        "has_spacer_grids": True,
    })
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="facts",
        requirement="VERA3 3A benchmark assembly",
        llm_client=fake,
        max_attempts=1,
    )
    assert result.ok is True
    assert result.envelope is not None
    assert result.envelope.status == "valid"
    assert result.envelope.patch_type == "facts"
    assert len(result.attempts) == 1
    assert result.attempts[0].parsed is True
    assert result.attempts[0].validated is True


def test_materials_generation_normalizes_schema_surface_before_parse() -> None:
    raw = json.dumps({
        "patch_type": "materials",
        "materials": [
            {
                "material_id": "ss304",
                "name": "SS304",
                "role": "structural",
                "density_g_cm3": 8.0,
                "composition_basis": "weight_frac",
                "composition_status": "approximate",
                "compound_components": [
                    {"formula": "Fe", "fraction": 0.70, "fraction_basis": "weight_frac"},
                    {"formula": "Cr", "fraction": 0.19, "fraction_basis": "weight_frac"},
                    {"formula": "Ni", "fraction": 0.10, "fraction_basis": "weight_frac"},
                ],
            },
        ],
    })
    result = generate_patch(
        patch_type="materials",
        requirement="stainless steel alloy with element fractions",
        llm_client=FakePatchLLM([raw]),
        max_attempts=1,
    )

    assert result.ok
    assert result.envelope is not None
    material = result.envelope.content["materials"][0]
    assert material["compound_components"] == []
    assert material["composition"] == {"Cr": 0.19, "Fe": 0.70, "Ni": 0.10}
    assert {
        item["operation"] for item in result.attempts[0].semantic_normalizations
    } == {"compound_element_component_to_composition_repair"}


# ---------------------------------------------------------------------------
# 3. PinMapPatch generation does not require full 17x17 map
# ---------------------------------------------------------------------------


def test_pin_map_generation_no_full_lattice() -> None:
    raw = json.dumps({
        "patch_type": "pin_map",
        "lattice_size": [17, 17],
        "default_universe_id": "fuel_pin",
        "coordinate_convention": {"index_base": 1, "row_origin": "top", "col_origin": "left", "ordering": "row_col"},
        "guide_tube_coords": [[3, 6], [3, 9], [3, 12]],
    })
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="pin_map",
        requirement="17x17 assembly",
        llm_client=fake,
        max_attempts=1,
    )
    assert result.ok is True
    assert result.parsed_patch is not None
    # Should only have 3 coords, not 289
    assert len(result.parsed_patch.get("guide_tube_coords", [])) == 3
    # Full lattice pattern must NOT be present
    assert "universe_pattern" not in result.parsed_patch


# ---------------------------------------------------------------------------
# 4. PinMapPatch overlap triggers retry only current patch
# ---------------------------------------------------------------------------


def test_pin_map_overlap_repaired_no_retry_needed() -> None:
    """Overlap is detected and reported → patch validates on first attempt."""
    overlap_raw = json.dumps({
        "patch_type": "pin_map",
        "lattice_size": [17, 17],
        "default_universe_id": "fuel_pin",
        "coordinate_convention": {"index_base": 0, "row_origin": "top", "col_origin": "left", "ordering": "row_col"},
        "guide_tube_coords": [[5, 5]],
        "pyrex_rod_coords": [[5, 5]],
    })
    fake = FakePatchLLM([overlap_raw])
    result = generate_patch(
        patch_type="pin_map",
        requirement="17x17 assembly",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is True
    assert len(result.attempts) == 1  # overlap detected, not blocking
    assert result.attempts[0].validated is True
    # First attempt should have coord_overlap_detected warning
    first_codes = [i["code"] for i in result.attempts[0].issues]
    assert "patch.pin_map.coord_overlap_detected" in first_codes


# ---------------------------------------------------------------------------
# 5. Invalid JSON retry
# ---------------------------------------------------------------------------


def test_invalid_json_retry() -> None:
    # Unbalanced braces (object never closes) -> flagged as truncated, which
    # is a distinct, more actionable code than a generic parse error.
    bad_raw = '{"patch_type": "materials", "materials": [{"material_id": "fuel", broken'
    good_raw = json.dumps({
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0}
        ],
    })
    fake = FakePatchLLM([bad_raw, good_raw])
    result = generate_patch(
        patch_type="materials",
        requirement="UO2 fuel",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is True
    assert len(result.attempts) == 2
    assert result.attempts[0].parsed is False
    first_codes = [i["code"] for i in result.attempts[0].issues]
    assert "patch_generation.json_truncated" in first_codes


def test_non_truncated_json_parse_error() -> None:
    # Output with no JSON object at all (e.g. pure prose from a reasoning
    # model that never answered) is NOT truncation -> generic parse error.
    bad_raw = "I cannot produce JSON for this requirement."
    good_raw = json.dumps({
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0}
        ],
    })
    fake = FakePatchLLM([bad_raw, good_raw])
    result = generate_patch(
        patch_type="materials",
        requirement="UO2 fuel",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is True
    first_codes = [i["code"] for i in result.attempts[0].issues]
    assert "patch_generation.json_parse_error" in first_codes
    assert "patch_generation.json_truncated" not in first_codes


def test_json_recovered_from_chain_of_thought() -> None:
    # Reasoning models (e.g. ds:deepseek-v4-flash) emit CoT before the JSON
    # answer; the CoT may contain stray '{' that defeat a greedy regex. The
    # parser must recover the real patch (declared patch_type) from after the
    # prose.
    cot = (
        'We need a JSON object for patch_type="facts". The schema looks like '
        '{"role": "fuel_pin", "value": 264}. Let me list counts: fuel 2376.\n'
        'Here is the final answer:\n'
    )
    real = json.dumps({
        "patch_type": "facts", "benchmark_id": "VERA4",
        "lattice_size": [17, 17], "model_scope": "multi_assembly_core",
    })
    raw = cot + real
    obj = parse_llm_patch_json(raw, "facts")
    assert obj["patch_type"] == "facts"
    assert obj["benchmark_id"] == "VERA4"


def test_no_false_recovery_from_cot_examples() -> None:
    # CoT that only contains small JSON examples (no object declaring the
    # expected patch_type) must NOT be mis-recovered as the patch.
    raw = (
        'Reasoning... the schema is {"role": "x", "value": 1} '
        'and maybe {"a": 1}. No actual patch emitted.'
    )
    try:
        parse_llm_patch_json(raw, "facts")
        assert False, "expected PatchParseError"
    except PatchParseError:
        pass


def test_json_repaired_when_object_opener_dropped_in_array() -> None:
    # Some LLMs drop `{"material_id":` for one or more entries in a homogeneous
    # array, emitting `,"<value>","name":` instead of `},{"material_id":"<value>","name":`.
    # The parser must learn the shape from a valid sibling and re-insert the opener.
    raw = (
        '{"patch_type":"materials","materials":['
        '{"material_id":"fuel","name":"Fuel","density_g_cm3":10.257},'
        '"coolant","name":"Water","density_g_cm3":0.743},'
        '{"material_id":"helium","name":"Helium","density_g_cm3":0.0001785}'
        ']}'
    )
    obj = parse_llm_patch_json(raw, "materials")
    ids = [m["material_id"] for m in obj["materials"]]
    assert ids == ["fuel", "coolant", "helium"]
    assert obj["materials"][1]["name"] == "Water"


def test_dropped_opener_repair_does_not_touch_valid_json() -> None:
    # A valid materials patch must round-trip unchanged.
    raw = json.dumps({
        "patch_type": "materials",
        "materials": [
            {"material_id": "fuel", "name": "Fuel", "density_g_cm3": 10.257},
            {"material_id": "coolant", "name": "Water", "density_g_cm3": 0.743},
        ],
    })
    obj = parse_llm_patch_json(raw, "materials")
    assert [m["material_id"] for m in obj["materials"]] == ["fuel", "coolant"]


# ---------------------------------------------------------------------------
# 6. Validation failure max attempts exceeded
# ---------------------------------------------------------------------------


def test_max_attempts_exceeded() -> None:
    # AxialLayersPatch with inverted z range (always fails validation with error)
    first_bad_raw = json.dumps({
        "patch_type": "axial_layers",
        "layers": [
            {"layer_id": "active_fuel", "role": "active_fuel",
             "z_min_cm": 100.0, "z_max_cm": 50.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"}
        ],
    })
    second_bad_raw = json.dumps({
        "patch_type": "axial_layers",
        "layers": [
            {"layer_id": "active_fuel", "role": "active_fuel",
             "z_min_cm": 100.0, "z_max_cm": 49.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"}
        ],
    })
    fake = FakePatchLLM([first_bad_raw, second_bad_raw])
    result = generate_patch(
        patch_type="axial_layers",
        requirement="3D assembly",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is False
    codes = [i["code"] for i in result.issues]
    assert "patch_generation.max_attempts_exceeded" in codes
    assert "patch_generation.no_progress_duplicate_candidate" not in codes
    assert len(result.attempts) == 2


def test_duplicate_schema_invalid_candidate_stops_before_a_third_call() -> None:
    """Repeated JSON is no-progress even when it never parses as a patch model."""
    bad_raw = json.dumps({
        "patch_type": "axial_layers",
        "layers": [
            {"layer_id": "active_fuel", "role": "active_fuel",
             "z_min_cm": 100.0, "z_max_cm": 50.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"}
        ],
    })
    fake = FakePatchLLM([bad_raw, bad_raw, bad_raw])

    result = generate_patch(
        patch_type="axial_layers",
        requirement="3D assembly",
        llm_client=fake,
        max_attempts=3,
    )

    assert result.ok is False
    assert len(result.attempts) == 2
    assert len(fake.prompts) == 2
    assert len(fake.responses) == 1
    assert result.attempts[0].candidate_hash
    assert result.attempts[1].candidate_hash == result.attempts[0].candidate_hash
    assert "patch_generation.no_progress_duplicate_candidate" in {
        issue["code"] for issue in result.issues
    }
    assert "patch_generation.max_attempts_exceeded" not in {
        issue["code"] for issue in result.issues
    }
    duplicate = next(
        issue for issue in result.issues
        if issue["code"] == "patch_generation.no_progress_duplicate_candidate"
    )
    assert duplicate["duplicate_scope"] == "current_generation"


def test_persisted_duplicate_candidate_stops_before_budget_exhaustion() -> None:
    """A retry-context candidate stops after one call without a budget error."""
    bad_content = {
        "patch_type": "axial_layers",
        "layers": [{
            "layer_id": "active_fuel", "role": "active_fuel",
            "z_min_cm": 100.0, "z_max_cm": 50.0,
            "fill_type": "lattice", "fill_id": "assembly_lattice",
        }],
    }
    bad_raw = json.dumps(bad_content)
    candidate_hash = compute_candidate_hash(
        target_patch_type="axial_layers", candidate_patch=bad_content,
    )
    fake = FakePatchLLM([bad_raw, bad_raw])

    result = generate_patch(
        patch_type="axial_layers",
        requirement="3D assembly",
        llm_client=fake,
        context=RetryPatchGenerationContext(
            base_context=PatchGenerationContext(),
            prior_candidate_hashes=[candidate_hash],
        ),
        max_attempts=3,
    )

    assert result.ok is False
    assert len(result.attempts) == 1
    assert len(fake.prompts) == 1
    assert len(fake.responses) == 1
    duplicate = next(
        issue for issue in result.issues
        if issue["code"] == "patch_generation.no_progress_duplicate_candidate"
    )
    assert duplicate["duplicate_scope"] == "persisted_context"
    assert "patch_generation.max_attempts_exceeded" not in {
        issue["code"] for issue in result.issues
    }


def test_retry_context_is_rendered_on_first_generation_prompt() -> None:
    raw = json.dumps({
        "patch_type": "universes",
        "universes": [
            {"universe_id": "rcca_aic", "kind": "control_rod", "cells": [{"id": "aic", "role": "absorber", "material_id": "aic"}]},
        ],
    })
    fake = FakePatchLLM([raw])

    result = generate_patch(
        patch_type="universes",
        requirement="reactor-neutral retry",
        llm_client=fake,
        context=RetryPatchGenerationContext(
            base_context=PatchGenerationContext(
                known_material_ids=["aic"],
            ),
            retry_request_id="retry_1",
            reason_code="localized_insert.required_universe_missing",
            source_issue_codes=["localized_insert.required_universe_missing"],
            required_ids=["rcca_aic", "rcca_b4c"],
        ),
        max_attempts=1,
    )

    assert result.ok
    assert fake.prompts
    assert "=== Executable retry context ===" in fake.prompts[0]
    assert "Required IDs: rcca_aic, rcca_b4c" in fake.prompts[0]


def test_localized_insert_requirements_are_rendered_in_patch_context() -> None:
    prompt = build_patch_prompt(
        "universes",
        "reactor-neutral prompt",
        PatchGenerationContext(
            localized_insert_requirements=[
                {
                    "requirement_id": "control_insert",
                    "insert_kind": "control_rod",
                    "required_profile_id": "control_profile",
                    "expected_insert_universe_ids": ["rcca_aic", "rcca_b4c"],
                }
            ]
        ),
    )

    assert "localized_insert_requirements" in prompt
    assert "rcca_aic" in prompt
    assert "rcca_b4c" in prompt


def test_axial_schema_normalizes_only_known_role_aliases_and_default_priority() -> None:
    raw = json.dumps({
        "patch_type": "axial_layers",
        "axial_domain_cm": [0.0, 3.0],
        "layers": [
            {"layer_id": "lower", "role": "lower_endplug", "z_min_cm": 0.0,
             "z_max_cm": 1.0, "fill_type": "lattice", "fill_id": "assembly_lattice",
             "loading_id": "lower_loading"},
            {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 1.0,
             "z_max_cm": 2.0, "fill_type": "lattice", "fill_id": "assembly_lattice"},
            {"layer_id": "upper", "role": "upper_endplug", "z_min_cm": 2.0,
             "z_max_cm": 3.0, "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ],
        "lattice_loadings": [{
            "loading_id": "lower_loading",
            "base_lattice_id": "assembly_lattice",
            "transformations": [{
                "operation_id": "lower_end_plug",
                "operation_kind": "replace_universe_family",
                "replacement_universe_id": "fuel_end_plug",
                "source_universe_id": "fuel_pin",
                "priority": None,
            }],
        }],
    })
    result = generate_patch(
        patch_type="axial_layers",
        requirement="reactor-neutral axial assembly",
        llm_client=FakePatchLLM([raw]),
        context=PatchGenerationContext(
            known_universe_ids=["fuel_pin", "fuel_end_plug"],
            known_lattice_ids=["assembly_lattice"],
        ),
        max_attempts=1,
    )

    assert result.ok is True
    assert result.parsed_patch is not None
    assert [layer["role"] for layer in result.parsed_patch["layers"]] == [
        "lower_end_plug", "active_fuel", "upper_end_plug",
    ]
    assert result.parsed_patch["lattice_loadings"][0]["transformations"][0]["priority"] == 0
    kinds = {item["kind"] for item in result.attempts[0].semantic_normalizations}
    assert {"axial_role_lexical_alias", "axial_priority_default_omitted"} <= kinds


def test_axial_layers_normalize_unique_universe_id_variant() -> None:
    """Separator-only ID variants resolve to an unambiguous known universe."""
    raw = _patch_json(
        "axial_layers",
        layers=[
            {
                "layer_id": "active_fuel", "role": "active_fuel",
                "z_min_cm": 0.0, "z_max_cm": 10.0,
                "fill_type": "lattice", "fill_id": "assembly_lattice",
            },
            {
                "layer_id": "upper_end_plug", "role": "upper_end_plug",
                "z_min_cm": 10.0, "z_max_cm": 11.0,
                "fill_type": "lattice", "fill_id": "assembly_lattice",
                "loading_id": "plug_loading",
            },
        ],
        lattice_loadings=[
            {
                "loading_id": "plug_loading",
                "base_lattice_id": "assembly_lattice",
                "derived_lattice_id": "assembly_lattice_plug",
                "transformations": [
                    {
                        "operation_id": "replace_plug",
                        "operation_kind": "replace_universe_family",
                        "source_universe_id": "fuel_pin",
                        "replacement_universe_id": "fuel_pin_endplug",
                    }
                ],
            }
        ],
    )
    result = generate_patch(
        patch_type="axial_layers",
        requirement="3D assembly",
        llm_client=FakePatchLLM([raw]),
        context=PatchGenerationContext(
            known_universe_ids=["fuel_pin", "fuel_pin_end_plug"],
        ),
        max_attempts=1,
    )
    assert result.ok is True
    assert result.parsed_patch is not None
    transformation = result.parsed_patch["lattice_loadings"][0]["transformations"][0]
    assert transformation["replacement_universe_id"] == "fuel_pin_end_plug"


# ---------------------------------------------------------------------------
# 10. SettingsPatch treats cross sections as runtime
# ---------------------------------------------------------------------------


def test_settings_generation_ok() -> None:
    raw = json.dumps({"patch_type": "settings"})
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="settings",
        requirement="assembly",
        llm_client=fake,
        max_attempts=1,
    )
    assert result.ok is True
    assert result.parsed_patch is not None
    assert result.parsed_patch["cross_sections_runtime_required"] is True
    assert result.parsed_patch["tallies_required_for_smoke_test"] is False


# ---------------------------------------------------------------------------
# 11. generate_and_add_patch_to_state success
# ---------------------------------------------------------------------------


def test_generate_and_add_to_state_success() -> None:
    state = PlanBuildState(state_id="test_gen", requirement_text="VERA3 3A")
    raw = json.dumps({"patch_type": "facts", "benchmark_id": "VERA3", "selected_variant": "3A"})
    fake = FakePatchLLM([raw])

    state = generate_and_add_patch_to_state(
        state, "facts", "VERA3 3A benchmark",
        llm_client=fake, max_attempts=1,
    )
    assert len(state.patches) == 1
    env = list(state.patches.values())[0]
    assert env.status == "valid"
    event_types = [e.event_type for e in state.build_log]
    assert "planning.patch_generation_started" in event_types
    assert "planning.patch_generated" in event_types


# ---------------------------------------------------------------------------
# 12. generate_and_add_patch_to_state failure preserves valid patches
# ---------------------------------------------------------------------------


def test_generation_failure_preserves_valid_patches() -> None:
    state = PlanBuildState(state_id="test_preserve", requirement_text="3D assembly")

    # First add a valid facts patch manually.
    facts = FactsPatch(benchmark_id="VERA3", selected_variant="3A")
    env = PlanPatchEnvelope(
        patch_id="facts_manual", patch_type="facts",
        content=facts.model_dump(mode="json"), status="valid",
    )
    state.add_patch(env)
    assert len(state.get_valid_patches()) == 1

    # Now try to generate axial_layers with a bad LLM.
    bad_raw = json.dumps({
        "patch_type": "axial_layers",
        "layers": [{"layer_id": "active_fuel", "role": "active_fuel",
                     "z_min_cm": 100.0, "z_max_cm": 50.0,
                     "fill_type": "lattice", "fill_id": "assembly_lattice"}],
    })
    fake = FakePatchLLM([bad_raw, bad_raw])
    state = generate_and_add_patch_to_state(
        state, "axial_layers", "3D assembly",
        llm_client=fake, max_attempts=2,
    )

    # The facts patch should still be valid.
    assert state.patches["facts_manual"].status == "valid"
    assert len(state.get_valid_patches()) == 1

    # Failure event should be recorded.
    event_types = [e.event_type for e in state.build_log]
    assert "planning.patch_generation_failed" in event_types


# ---------------------------------------------------------------------------
# Phase 7B: forbidden full-plan output detection
# ---------------------------------------------------------------------------


def test_facts_patch_rejects_full_plan_output() -> None:
    """LLM returns a full SimulationPlan instead of a FactsPatch → rejected."""
    full_plan_raw = json.dumps({
        "schema_version": "simulation_plan.v2",
        "complex_model": {"name": "VERA3", "kind": "assembly"},
        "capability_report": {"renderability": "none"},
    })
    fake = FakePatchLLM([full_plan_raw, full_plan_raw])
    result = generate_patch(
        patch_type="facts",
        requirement="VERA3 3B",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.full_plan_output_forbidden" in all_codes


def test_retry_after_full_plan_succeeds() -> None:
    """Full plan first, then valid patch → success on second attempt."""
    full_plan_raw = json.dumps({
        "schema_version": "simulation_plan.v2",
        "complex_model": {"name": "VERA3", "kind": "assembly"},
    })
    valid_facts = json.dumps({
        "patch_type": "facts",
        "benchmark_id": "VERA3",
        "selected_variant": "3B",
        "lattice_size": [17, 17],
        "has_axial_geometry": True,
    })
    fake = FakePatchLLM([full_plan_raw, valid_facts])
    result = generate_patch(
        patch_type="facts",
        requirement="VERA3 3B",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is True
    assert len(result.attempts) == 2
    assert result.attempts[0].contains_full_plan_markers is True
    first_codes = [i["code"] for i in result.attempts[0].issues]
    assert "patch_generation.full_plan_output_forbidden" in first_codes


def test_pin_map_full_lattice_forbidden_error() -> None:
    """A near-full enumerated pin_map (one coord list ≈ rows×cols) is forbidden.

    "Full lattice" means the LLM enumerated (almost) every cell into a single
    coordinate list instead of using special coordinates.  Here 250 of 289
    cells (≈86%) sit in one list, so the structural full-lattice gate fires.
    """
    near_full = [[r, c] for r in range(17) for c in range(17)][:250]
    raw = json.dumps({
        "patch_type": "pin_map",
        "lattice_size": [17, 17],
        "default_universe_id": "fp",
        "coordinate_convention": {"index_base": 0},
        "guide_tube_coords": near_full,
    })
    fake = FakePatchLLM([raw, raw])
    result = generate_patch(
        patch_type="pin_map",
        requirement="17x17",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.pin_map_full_lattice_forbidden" in all_codes


def test_pin_map_special_coords_multi_insert_not_flagged() -> None:
    """Regression for VERA Problem 4: a legitimate special-coordinate pin_map
    with many localized inserts (guide tubes + pyrex + thimble plugs + RCCA)
    must NOT be rejected as a full lattice.

    The largest single coordinate list here is 24 (guide tubes) out of 289
    cells (≈8%), yet the raw JSON has >80 brackets in aggregate (24 guide +
    1 instrument + 20 + 24 + 4 + 24 insert coords).  The old raw-text
    heuristic summed every bracket and falsely flagged this; the structural
    check must accept it.
    """
    guide_tubes = [
        [3, 6], [3, 9], [3, 12], [4, 4], [4, 14],
        [6, 3], [6, 6], [6, 9], [6, 12], [6, 15],
        [9, 3], [9, 6], [9, 12], [9, 15],
        [12, 3], [12, 6], [12, 9], [12, 12], [12, 15],
        [14, 4], [14, 14], [15, 6], [15, 9], [15, 12],
    ]
    pyrex_e = [
        [3, 6], [3, 12], [4, 4], [4, 14],
        [6, 3], [6, 9], [6, 15], [9, 6], [9, 12],
        [12, 3], [12, 9], [12, 15], [14, 4], [14, 14],
        [15, 6], [15, 12], [3, 9], [9, 3], [9, 15], [15, 9],
    ]
    plug_e = [[6, 6], [6, 12], [12, 6], [12, 12]]
    raw = json.dumps({
        "patch_type": "pin_map",
        "lattice_size": [17, 17],
        "default_universe_id": "fuel_pin",
        "coordinate_convention": {
            "index_base": 1, "row_origin": "top",
            "col_origin": "left", "ordering": "row_col",
        },
        "guide_tube_coords": guide_tubes,
        "instrument_tube_coords": [[9, 9]],
        "localized_insert_intents": [
            {
                "insert_id": "pyrex_E", "insert_kind": "pyrex_rod",
                "host_kind": "guide_tube", "insert_universe_id": "pyrex_rod",
                "coordinates": pyrex_e,
                "z_min_cm": 15.761, "z_max_cm": 376.441,
                "application_mode": "nested_component_override",
                "component_role": "internal_coolant",
                "preserve_component_roles": ["tube_wall", "outer_coolant"],
            },
            {
                "insert_id": "plug_C", "insert_kind": "thimble_plug",
                "host_kind": "guide_tube", "insert_universe_id": "thimble_plug",
                "coordinates": list(guide_tubes),
                "z_min_cm": 383.31, "z_max_cm": 394.31,
                "application_mode": "nested_component_override",
                "component_role": "internal_coolant",
                "preserve_component_roles": ["tube_wall", "outer_coolant"],
            },
            {
                "insert_id": "plug_E", "insert_kind": "thimble_plug",
                "host_kind": "guide_tube", "insert_universe_id": "thimble_plug",
                "coordinates": plug_e,
                "z_min_cm": 383.31, "z_max_cm": 394.31,
                "application_mode": "nested_component_override",
            },
            {
                "insert_id": "rcca_R", "insert_kind": "control_rod",
                "host_kind": "guide_tube", "insert_universe_id": "control_rod_AIC",
                "coordinates": list(guide_tubes),
                "z_min_cm": 257.9, "z_max_cm": 359.5,
                "application_mode": "nested_component_override",
                "component_role": "internal_coolant",
                "preserve_component_roles": ["tube_wall", "outer_coolant"],
            },
        ],
        "assumptions": [
            "Inserts are finite-height; guide-tube wall and outer coolant are "
            "preserved outside each insert's z range.",
            "Pyrex/thimble-plug/RCCA coordinates are subsets of guide_tube_coords.",
        ],
        "source_note": "VERA Problem 4 base state; multi-insert special-coordinate map.",
    })
    # This input trips the OLD raw-text heuristic (sums brackets across sections).
    assert raw.count("[") + raw.count("(") > 80
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="pin_map",
        requirement="VERA4 3x3 multi-assembly",
        llm_client=fake,
        max_attempts=1,
    )
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.pin_map_full_lattice_forbidden" not in all_codes
    assert result.ok is True


def test_non_json_natural_language_response() -> None:
    """Natural language response → json_parse_error."""
    fake = FakePatchLLM([
        "I cannot generate this patch because I need more information.",
        "I cannot generate this patch.",
    ])
    result = generate_patch(
        patch_type="facts",
        requirement="test",
        llm_client=fake,
        max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert any("json_parse_error" in c for c in all_codes)


def test_prompt_has_output_contract() -> None:
    """Verify the hardened prompt contains the output contract."""
    from openmc_agent.plan_builder.patch_prompts import build_patch_prompt
    prompt = build_patch_prompt("facts", "test requirement", None)
    assert "CRITICAL OUTPUT CONTRACT" in prompt
    assert 'patch_type="facts"' in prompt
    assert "NOT generating a SimulationPlan" in prompt
    assert "complex_model" in prompt  # forbidden markers listed
    assert "Allowed top-level keys" in prompt  # Phase 7C: allowed keys
    assert "FORBIDDEN top-level keys" in prompt  # Phase 7C: forbidden keys


# ---------------------------------------------------------------------------
# Phase 7C: patch contract validation
# ---------------------------------------------------------------------------


def test_facts_patch_missing_patch_type_rejected() -> None:
    """JSON without patch_type field → patch_type_missing error."""
    raw = json.dumps({"benchmark_id": "VERA3", "lattice_size": [17, 17]})
    fake = FakePatchLLM([raw, raw])
    result = generate_patch(
        patch_type="facts", requirement="test", llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.patch_type_missing" in all_codes


def test_facts_patch_wrong_patch_type_rejected() -> None:
    """JSON with wrong patch_type → patch_type_mismatch error."""
    raw = json.dumps({"patch_type": "materials", "materials": []})
    fake = FakePatchLLM([raw, raw])
    result = generate_patch(
        patch_type="facts", requirement="test", llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.patch_type_mismatch" in all_codes


def test_materials_patch_with_core_rejected() -> None:
    """Materials patch with forbidden core field → full_plan_output_forbidden."""
    raw = json.dumps({
        "patch_type": "materials",
        "materials": [{"material_id": "m", "name": "M", "role": "fuel"}],
        "core": {"axial_layers": []},
    })
    fake = FakePatchLLM([raw, raw])
    result = generate_patch(
        patch_type="materials", requirement="test", llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    all_codes = [i["code"] for i in result.issues]
    assert "patch_generation.full_plan_output_forbidden" in all_codes


def test_structured_client_method_used() -> None:
    """When client has generate_patch_json, it should be preferred."""

    class StructuredFakeClient:
        def __init__(self) -> None:
            self.called_structured = False
            self.response = '{"patch_type": "facts", "benchmark_id": "T"}'

        def __call__(self, prompt: str) -> str:
            return self.response

        def generate_patch_json(self, *, prompt, patch_type, json_schema=None, **kw):
            self.called_structured = True
            assert patch_type == "facts"
            assert json_schema is not None
            return self.response

    client = StructuredFakeClient()
    result = generate_patch(
        patch_type="facts", requirement="test",
        llm_client=client, max_attempts=1,
    )
    assert result.ok is True
    assert client.called_structured is True


def test_max_tokens_threaded_to_structured_client() -> None:
    """generate_patch(max_tokens=...) must reach generate_patch_json."""

    class RecordingClient:
        def __init__(self) -> None:
            self.received_max_tokens = "not called"

        def __call__(self, prompt: str) -> str:
            return '{"patch_type": "facts", "benchmark_id": "T"}'

        def generate_patch_json(self, *, prompt, patch_type, json_schema=None, **kw):
            self.received_max_tokens = kw.get("max_tokens")
            return '{"patch_type": "facts", "benchmark_id": "T"}'

    client = RecordingClient()
    generate_patch(
        patch_type="facts", requirement="test",
        llm_client=client, max_attempts=1, max_tokens=1234,
    )
    assert client.received_max_tokens == 1234


def test_attempt_records_prompt_and_output_mode() -> None:
    """PatchGenerationAttempt should record prompt_text and output_mode_used."""
    raw = json.dumps({"patch_type": "facts", "benchmark_id": "T"})
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="facts", requirement="test requirement",
        llm_client=fake, max_attempts=1,
    )
    assert result.ok is True
    att = result.attempts[0]
    assert att.prompt_text is not None
    assert "CRITICAL OUTPUT CONTRACT" in att.prompt_text
    assert att.output_mode_used == "plain_prompt"


def test_retry_prompt_for_full_plan_is_explicit() -> None:
    """Retry prompt for full_plan output should explicitly mention the issue."""
    from openmc_agent.plan_builder.patch_prompts import build_retry_prompt
    issues = [
        {"code": "patch_generation.full_plan_output_forbidden", "severity": "error", "message": "test"},
    ]
    prompt = build_retry_prompt("facts", "req", None, issues, 1)
    assert "REJECTED" in prompt or "rejected" in prompt
    assert "full SimulationPlan" in prompt or "full plan" in prompt.lower()


# ---------------------------------------------------------------------------
# 13. VERA3 3B pin_map patch generation
# ---------------------------------------------------------------------------


def test_vera3_3b_pin_map_generation() -> None:
    raw = json.dumps({
        "patch_type": "pin_map",
        "variant": "3B",
        "lattice_size": [17, 17],
        "default_universe_id": "fuel_pin",
        "coordinate_convention": {"index_base": 1, "row_origin": "top", "col_origin": "left", "ordering": "row_col"},
        "instrument_tube_coords": [[9, 9]],
        "pyrex_rod_coords": [
            [3, 6], [4, 4], [6, 3], [3, 12], [4, 14], [6, 15], [12, 3], [14, 4],
            [15, 6], [12, 15], [14, 14], [15, 12], [9, 6], [9, 12], [6, 9], [12, 9],
        ],
        "thimble_plug_coords": [
            [3, 9], [6, 6], [6, 12], [9, 3], [9, 15], [12, 6], [12, 12], [15, 9],
        ],
    })
    fake = FakePatchLLM([raw])
    ctx = PatchGenerationContext(
        benchmark_id="VERA3", selected_variant="3B",
        expected_counts={"expected_pyrex_rod_count": 16, "expected_thimble_plug_count": 8},
        known_universe_ids=["fuel_pin", "pyrex_rod", "thimble_plug", "instrument_tube"],
    )
    result = generate_patch(
        patch_type="pin_map",
        requirement="VERA3 3B benchmark",
        context=ctx,
        llm_client=fake,
        max_attempts=1,
    )
    assert result.ok is True
    assert len(result.parsed_patch["pyrex_rod_coords"]) == 16
    assert len(result.parsed_patch["thimble_plug_coords"]) == 8
    # Total patch JSON should be small
    patch_bytes = len(json.dumps(result.parsed_patch))
    assert patch_bytes < 2000, f"pin_map patch is {patch_bytes} bytes"


# ---------------------------------------------------------------------------
# 14. VERA3 3B axial_overlays generation
# ---------------------------------------------------------------------------


def test_vera3_3b_axial_overlays_generation() -> None:
    grids = []
    for i in range(8):
        grids.append({
            "overlay_id": f"grid_{i}",
            "overlay_kind": "spacer_grid",
            "z_min_cm": 10.0 + i * 50,
            "z_max_cm": 12.0 + i * 50,
            "target_lattice_id": "assembly_lattice",
            "material_id": "grid_mat",
            "geometry_mode": "homogenized_open_region",
            "through_path_preserved": True,
        })
    raw = json.dumps({"patch_type": "axial_overlays", "overlays": grids})
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="spacer grids",
        llm_client=fake,
        max_attempts=1,
    )
    assert result.ok is True
    assert len(result.parsed_patch["overlays"]) == 8
    for ov in result.parsed_patch["overlays"]:
        assert ov["geometry_mode"] == "homogenized_open_region"
        assert ov["through_path_preserved"] is True


def test_axial_overlay_unique_material_alias_normalized_before_validation() -> None:
    raw = json.dumps({
        "patch_type": "axial_overlays",
        "overlays": [{
            "overlay_id": "grid_1",
            "overlay_kind": "spacer_grid",
            "z_min_cm": 10.0,
            "z_max_cm": 12.0,
            "target_lattice_id": "assembly_lattice",
            "material_id": "zircaloy4",
            "geometry_mode": "homogenized_open_region",
            "through_path_preserved": True,
        }],
    })
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="spacer grids",
        llm_client=FakePatchLLM([raw]),
        context=PatchGenerationContext(
            known_material_ids=["mat_zirc"],
            material_summaries=[
                {"material_id": "mat_zirc", "name": "Zircaloy-4 Cladding"},
            ],
        ),
        max_attempts=1,
    )
    assert result.ok is True
    assert result.parsed_patch["overlays"][0]["material_id"] == "mat_zirc"
    assert result.attempts[0].semantic_normalizations[0]["kind"] == (
        "axial_overlay_material_alias"
    )


def test_axial_overlay_ambiguous_or_missing_material_degrades_to_skeleton() -> None:
    raw = json.dumps({
        "patch_type": "axial_overlays",
        "overlays": [
            {
                "overlay_id": "grid_1",
                "overlay_kind": "spacer_grid",
                "z_min_cm": 10.0,
                "z_max_cm": 12.0,
                "target_lattice_id": "assembly_lattice",
                "material_id": "zircaloy4",
                "geometry_mode": "homogenized_open_region",
                "through_path_preserved": True,
            },
            {
                "overlay_id": "grid_2",
                "overlay_kind": "spacer_grid",
                "z_min_cm": 20.0,
                "z_max_cm": 22.0,
                "target_lattice_id": "assembly_lattice",
                "material_id": "inconel718",
                "geometry_mode": "homogenized_open_region",
                "through_path_preserved": True,
            },
        ],
    })
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="spacer grids",
        llm_client=FakePatchLLM([raw]),
        context=PatchGenerationContext(
            known_material_ids=["mat_zirc_a", "mat_zirc_b"],
            material_summaries=[
                {"material_id": "mat_zirc_a", "name": "Zircaloy-4 Cladding"},
                {"material_id": "mat_zirc_b", "name": "Zircaloy-4 Guide Tube"},
            ],
        ),
        max_attempts=1,
    )
    assert result.ok is True
    overlays = result.parsed_patch["overlays"]
    assert [ov["geometry_mode"] for ov in overlays] == ["skeleton", "skeleton"]
    assert [ov["material_id"] for ov in overlays] == [None, None]
    assert all(ov["requires_human_confirmation"] is True for ov in overlays)
    kinds = [n["kind"] for n in result.attempts[0].semantic_normalizations]
    assert kinds == [
        "axial_overlay_unresolved_material_skeleton",
        "axial_overlay_unresolved_material_skeleton",
    ]


# ---------------------------------------------------------------------------
# 15. VERA3 3B patch generation + assembler smoke at unit level
# ---------------------------------------------------------------------------


def test_vera3_3b_generation_plus_assembler_smoke() -> None:
    """Generate facts/pin_map/axial_layers/axial_overlays via fake LLM,
    use fixture materials/universes/settings, assemble, check guard."""
    from tests.test_vera3_patch_fixtures import _load_fixture

    # Load fixture patches for materials/universes/settings.
    fixture_patches = _load_fixture("3b")
    materials_patch = next(p for p in fixture_patches if p.patch_type == "materials")
    universes_patch = next(p for p in fixture_patches if p.patch_type == "universes")
    settings_patch = next(p for p in fixture_patches if p.patch_type == "settings")

    # Generate facts/pin_map/axial_layers/axial_overlays via fake LLM.
    facts_raw = json.dumps({
        "patch_type": "facts",
        "benchmark_id": "VERA3", "selected_variant": "3B",
        "lattice_size": [17, 17], "pin_pitch_cm": 1.26,
        "assembly_pitch_cm": 21.50,
        "has_axial_geometry": True, "has_spacer_grids": True,
        "has_special_pin_map": True,
        "active_fuel_region_cm": [11.951, 377.711],
        "axial_domain_cm": [-55.0, 463.937],
        "expected_pyrex_count": 16, "expected_thimble_plug_count": 8,
    })
    pin_map_raw = json.dumps({
        "patch_type": "pin_map", "variant": "3B",
        "lattice_size": [17, 17], "default_universe_id": "fuel_pin",
        "coordinate_convention": {"index_base": 1, "row_origin": "top", "col_origin": "left", "ordering": "row_col"},
        "instrument_tube_coords": [[9, 9]],
        "pyrex_rod_coords": [
            [3,6],[4,4],[6,3],[3,12],[4,14],[6,15],[12,3],[14,4],
            [15,6],[12,15],[14,14],[15,12],[9,6],[9,12],[6,9],[12,9],
        ],
        "thimble_plug_coords": [
            [3,9],[6,6],[6,12],[9,3],[9,15],[12,6],[12,12],[15,9],
        ],
    })
    axial_layers_raw = json.dumps({
        "patch_type": "axial_layers",
        "axial_domain_cm": [-55.0, 463.937],
        "layers": [
            {"layer_id": "active_fuel", "role": "active_fuel",
             "z_min_cm": 11.951, "z_max_cm": 377.711,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ],
    })
    overlays_raw = json.dumps({
        "patch_type": "axial_overlays",
        "overlays": [
            {"overlay_id": "grid_0", "overlay_kind": "spacer_grid",
             "z_min_cm": 11.951, "z_max_cm": 15.817,
             "target_lattice_id": "assembly_lattice", "material_id": "inconel718",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ],
    })

    fake = FakePatchLLM([facts_raw, pin_map_raw, axial_layers_raw, overlays_raw])
    ctx = PatchGenerationContext(benchmark_id="VERA3", selected_variant="3B")

    generated: list = []
    for ptype in ("facts", "pin_map", "axial_layers", "axial_overlays"):
        result = generate_patch(
            patch_type=ptype, requirement="VERA3 3B",
            context=ctx, llm_client=fake, max_attempts=1,
        )
        assert result.ok, f"{ptype} generation failed: {[i['code'] for i in result.issues if i.get('severity') == 'error']}"
        parsed = parse_patch_content(ptype, result.parsed_patch)
        generated.append(parsed)

    # Combine generated + fixture patches.
    all_patches = generated + [materials_patch, universes_patch, settings_patch]

    # Assemble.
    asm_result = assemble_simulation_plan_from_patches(all_patches)
    assert asm_result.ok, [
        (i.code, i.message[:80]) for i in asm_result.issues if i.severity == "error"
    ]

    # Check assembly3d guard.
    from openmc_agent.assembly3d_guard import validate_assembly3d_plan
    issues = validate_assembly3d_plan(
        asm_result.plan,
        requirement="VERA3 3B benchmark: 3D assembly with axial layers, spacer grids",
    )
    error_codes = [i.code for i in issues if i.severity == "error"]
    assert "assembly3d.axial_layers_required" not in error_codes


# ---------------------------------------------------------------------------
# parse_llm_patch_json helper tests
# ---------------------------------------------------------------------------


def test_parse_json_pure() -> None:
    result = parse_llm_patch_json('{"patch_type": "facts"}', "facts")
    assert result["patch_type"] == "facts"


def test_parse_json_markdown_fenced() -> None:
    result = parse_llm_patch_json('```json\n{"patch_type": "facts"}\n```', "facts")
    assert result["patch_type"] == "facts"


def test_parse_json_with_preamble() -> None:
    result = parse_llm_patch_json(
        'Here is the patch:\n{"patch_type": "facts"}\nDone.', "facts"
    )
    assert result["patch_type"] == "facts"


def test_parse_json_empty_raises() -> None:
    from openmc_agent.plan_builder.patches import PatchParseError
    with pytest.raises(PatchParseError):
        parse_llm_patch_json("", "facts")
