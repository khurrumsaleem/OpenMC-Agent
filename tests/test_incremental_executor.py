"""Tests for the incremental executor (Phase 5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from openmc_agent.plan_builder.assembler import assemble_simulation_plan_from_patches
from openmc_agent.plan_builder.executor import (
    IncrementalExecutionResult,
    _order_for_controlled_gate_barriers,
    build_deterministic_settings_patch,
    build_generation_context_from_state,
    default_patch_task_order,
    required_patch_types_for_state,
    run_incremental_planning,
)
from openmc_agent.plan_builder.mode import should_use_incremental_planning
from openmc_agent.plan_builder.patches import (
    FactsPatch,
    parse_patch_content,
)
from openmc_agent.plan_builder.patch_generator import FakePatchLLM
from openmc_agent.plan_builder.state import (
    PlanBuildState,
    PlanPatchEnvelope,
    initialize_plan_build_state,
)
from openmc_agent.graph import (
    _incremental_plan_repair_patch_targets,
    _incremental_targeted_repair_requirement,
    _plan_build_state_with_repair_request,
)
from openmc_agent.plan_builder.validators import validate_patch
from openmc_agent.assembly3d_guard import validate_assembly3d_plan
from openmc_agent.schemas import SimulationPlan
from openmc_agent.validator import validate_simulation_plan


_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "vera3_patches"


def _load_fixture_raw(variant: str) -> list[dict]:
    """Load raw patch dicts (not parsed) from fixture file."""
    raw = json.loads((_FIXTURE_DIR / f"vera3_{variant}_patches.json").read_text("utf-8"))
    return raw["patches"]


# ---------------------------------------------------------------------------
# Helper: VERA3 3B requirement text
# ---------------------------------------------------------------------------

_VERA3_3B_REQ = (
    "VERA3 3B benchmark: 3D assembly with axial layers, spacer grids, "
    "三维, 定位格架, Pyrex rods, thimble plugs, 17x17 lattice"
)


def _init_3b_state() -> PlanBuildState:
    decision = should_use_incremental_planning(_VERA3_3B_REQ)
    state = initialize_plan_build_state(
        requirement=_VERA3_3B_REQ,
        decision=decision,
        benchmark_id="VERA3",
        selected_variant="3B",
    )
    return state


def test_executor_surfaces_detailed_assembly_issues(monkeypatch) -> None:
    """Assembly diagnostics must not be reduced to a generic wrapper only."""
    import openmc_agent.plan_builder.executor as executor_module

    state = PlanBuildState(state_id="assembly-detail", requirement_text="r")
    state.validation_issues.append({
        "code": "preexisting.warning",
        "severity": "warning",
        "message": "must not be duplicated",
    })
    monkeypatch.setattr(executor_module, "default_patch_task_order", lambda _state: [])
    monkeypatch.setattr(executor_module, "required_patch_types_for_state", lambda _state: [])

    def fail_assembly(current_state, **_kwargs):
        current_state.validation_issues.extend([
            {
                "code": "localized_insert.loading_missing",
                "severity": "error",
                "message": "AIC has no overlapping lattice layer",
                "path": "axial_layers.layers",
            },
            {
                "code": "localized_insert.loading_missing",
                "severity": "error",
                "message": "B4C has no overlapping lattice layer",
                "path": "axial_layers.layers",
            },
        ])
        return current_state

    monkeypatch.setattr(executor_module, "assemble_state_if_ready", fail_assembly)
    result = run_incremental_planning(
        requirement="r",
        state=state,
        llm_client=lambda _prompt: (_ for _ in ()).throw(AssertionError("no LLM")),
        task_order=[],
    )

    assert result.ok is False
    codes = [issue.code for issue in result.issues]
    assert "localized_insert.loading_missing" in codes
    assert "incremental.assembly_failed" in codes
    assert "preexisting.warning" not in codes
    assert result.summary["failed_stage"] == "assembly"
    assert "localized_insert.loading_missing" in result.summary["issue_codes"]


# ---------------------------------------------------------------------------
# 1. Executor generates patches in dependency order
# ---------------------------------------------------------------------------


def test_executor_generates_in_dependency_order() -> None:
    """Minimal fake LLM that returns valid patches."""
    responses = [
        json.dumps({"patch_type": "facts", "benchmark_id": "TEST",
                     "lattice_size": [17, 17], "pin_pitch_cm": 1.26,
                     "has_axial_geometry": True, "has_spacer_grids": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0},
            {"material_id": "water", "name": "Water", "role": "coolant", "density_g_cm3": 0.74},
            {"material_id": "clad", "name": "Zircaloy-4", "role": "cladding",
             "density_g_cm3": 6.56, "composition": {"Zr": 1.0},
             "composition_status": "approximate",
             "warnings": ["approximated as pure Zr"]},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fuel_pin", "kind": "fuel_pin", "cells": [
                {"id": "fuel", "role": "fuel", "material_id": "fuel"},
                {"id": "clad", "role": "cladding", "material_id": "clad"},
                {"id": "water", "role": "coolant", "material_id": "water"},
            ]},
            {"universe_id": "gt", "kind": "guide_tube", "cells": [
                {"id": "iw", "role": "coolant", "material_id": "water"},
                {"id": "wall", "role": "cladding", "material_id": "clad"},
                {"id": "ow", "role": "background", "material_id": "water"},
            ]},
        ]}),
        json.dumps({"patch_type": "pin_map", "lattice_size": [17, 17],
                     "default_universe_id": "fuel_pin",
                     "coordinate_convention": {"index_base": 0},
                     "guide_tube_coords": [[2, 2]]}),
        json.dumps({"patch_type": "axial_layers", "layers": [
            {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ]}),
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "clad",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]
    fake = FakePatchLLM(responses)
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=1,
    )
    assert result.ok is True
    assert result.assembled_plan is not None

    # Check generation order: facts was the first LLM call.
    first_prompt = fake.prompts[0]
    assert "patch_type=\"facts\"" in first_prompt or "patch_type: facts" in first_prompt

    # Check valid patch types.
    valid_types = {e.patch_type for e in state.patches.values() if e.status == "valid"}
    assert "facts" in valid_types
    assert "materials" in valid_types
    assert "universes" in valid_types
    assert "pin_map" in valid_types
    assert "axial_layers" in valid_types
    assert "axial_overlays" in valid_types
    assert "settings" in valid_types


# ---------------------------------------------------------------------------
# 2. Executor skips already valid patch
# ---------------------------------------------------------------------------


def test_executor_skips_valid_patch() -> None:
    state = _init_3b_state()
    # Pre-add a valid facts patch.
    facts = FactsPatch(benchmark_id="VERA3", selected_variant="3B")
    env = PlanPatchEnvelope(
        patch_id="pre_facts", patch_type="facts",
        content=facts.model_dump(mode="json"), status="valid",
    )
    state.add_patch(env)

    responses = [
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0},
            {"material_id": "water", "name": "Water", "role": "coolant", "density_g_cm3": 0.74},
            {"material_id": "clad", "name": "Zircaloy-4", "role": "cladding",
             "density_g_cm3": 6.56, "composition": {"Zr": 1.0},
             "composition_status": "approximate",
             "warnings": ["approximated as pure Zr"]},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fuel_pin", "kind": "fuel_pin", "cells": [
                {"id": "fuel", "role": "fuel", "material_id": "fuel"},
                {"id": "clad", "role": "cladding", "material_id": "clad"},
                {"id": "water", "role": "coolant", "material_id": "water"},
            ]},
            {"universe_id": "gt", "kind": "guide_tube", "cells": [
                {"id": "iw", "role": "coolant", "material_id": "water"},
                {"id": "wall", "role": "cladding", "material_id": "clad"},
                {"id": "ow", "role": "background", "material_id": "water"},
            ]},
        ]}),
        json.dumps({"patch_type": "pin_map", "lattice_size": [17, 17],
                     "default_universe_id": "fuel_pin",
                     "coordinate_convention": {"index_base": 0},
                     "guide_tube_coords": [[2, 2]]}),
        json.dumps({"patch_type": "axial_layers", "layers": [
            {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ]}),
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "clad",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]
    fake = FakePatchLLM(responses)
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=1,
    )
    assert result.ok is True

    # The first LLM call should be for materials, NOT facts.
    first_prompt = fake.prompts[0]
    assert "patch_type=\"materials\"" in first_prompt or "patch_type: materials" in first_prompt

    # Event should record the skip.
    event_types = [e.event_type for e in state.build_log]
    assert "planning.patch_skipped_already_valid" in event_types


# ---------------------------------------------------------------------------
# 3. Context propagation from facts to pin_map
# ---------------------------------------------------------------------------


def test_context_propagation_facts_to_pin_map() -> None:
    state = _init_3b_state()
    facts = FactsPatch(
        benchmark_id="VERA3", selected_variant="3B",
        lattice_size=(17, 17),
        expected_pyrex_count=16, expected_thimble_plug_count=8,
        has_spacer_grids=True, has_special_pin_map=True,
    )
    env = PlanPatchEnvelope(
        patch_id="pre_facts", patch_type="facts",
        content=facts.model_dump(mode="json"), status="valid",
    )
    state.add_patch(env)

    ctx = build_generation_context_from_state(state, "pin_map")
    assert ctx.benchmark_id == "VERA3"
    assert ctx.selected_variant == "3B"
    assert ctx.expected_counts.get("expected_pyrex_count") == 16
    assert ctx.expected_counts.get("expected_thimble_plug_count") == 8


def test_plan_validation_repair_regenerates_targeted_patches_only() -> None:
    state = _init_3b_state()
    common_prefix = [
        json.dumps({"patch_type": "facts", "benchmark_id": "TEST",
                     "lattice_size": [17, 17], "pin_pitch_cm": 1.26,
                     "has_axial_geometry": True, "has_spacer_grids": True,
                     "has_special_pin_map": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0},
            {"material_id": "water", "name": "Water", "role": "coolant", "density_g_cm3": 0.74},
            {"material_id": "clad", "name": "Zircaloy-4", "role": "cladding",
             "density_g_cm3": 6.56, "composition": {"Zr": 1.0},
             "composition_status": "approximate",
             "warnings": ["approximated as pure Zr"]},
        ]}),
    ]
    bad_universes = json.dumps({"patch_type": "universes", "universes": [
        {"universe_id": "fuel_pin", "kind": "fuel_pin", "cells": [
            {"id": "fuel", "role": "fuel", "material_id": "fuel"},
        ]},
        {"universe_id": "gt", "kind": "guide_tube", "cells": [
            {"id": "iw", "role": "coolant", "material_id": "water"},
            {"id": "wall", "role": "cladding", "material_id": "clad"},
            {"id": "ow", "role": "background", "material_id": "water"},
        ]},
    ]})
    fixed_universes = json.dumps({"patch_type": "universes", "universes": [
        {"universe_id": "fuel_pin", "kind": "fuel_pin", "cells": [
            {"id": "fuel", "role": "fuel", "material_id": "fuel"},
            {"id": "clad", "role": "cladding", "material_id": "clad"},
            {"id": "water", "role": "coolant", "material_id": "water"},
        ]},
        {"universe_id": "gt", "kind": "guide_tube", "cells": [
            {"id": "iw", "role": "coolant", "material_id": "water"},
            {"id": "wall", "role": "cladding", "material_id": "clad"},
            {"id": "ow", "role": "background", "material_id": "water"},
        ]},
    ]})
    downstream = [
        json.dumps({"patch_type": "pin_map", "lattice_size": [17, 17],
                     "default_universe_id": "fuel_pin",
                     "coordinate_convention": {"index_base": 0},
                     "guide_tube_coords": [[2, 2]]}),
        json.dumps({"patch_type": "axial_layers", "layers": [
            {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ]}),
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "clad",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]

    first = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=FakePatchLLM(common_prefix + [bad_universes] + downstream),
        max_patch_attempts=1,
    )
    assert first.ok is True
    assert first.assembled_plan is not None

    report = validate_simulation_plan(
        SimulationPlan.model_validate(first.assembled_plan),
        requirement=_VERA3_3B_REQ,
    )
    assert any(i.code == "lattice.universe_missing_coolant" for i in report.issues)
    target_patch_types = _incremental_plan_repair_patch_targets(report)
    assert target_patch_types == ["universes"]

    repaired_payload = _plan_build_state_with_repair_request(
        state.model_dump(mode="json"),
        report=report,
        target_patch_types=target_patch_types,
    )
    assert repaired_payload is not None
    repaired_state = PlanBuildState.model_validate(repaired_payload)
    repair_llm = FakePatchLLM([fixed_universes] + downstream)
    second = run_incremental_planning(
        requirement=_incremental_targeted_repair_requirement(
            _VERA3_3B_REQ, report, target_patch_types
        ),
        state=repaired_state,
        llm_client=repair_llm,
        max_patch_attempts=1,
    )
    assert second.ok is True
    assert second.assembled_plan is not None
    repaired_report = validate_simulation_plan(
        SimulationPlan.model_validate(second.assembled_plan),
        requirement=_VERA3_3B_REQ,
    )
    assert not any(i.code == "lattice.universe_missing_coolant" for i in repaired_report.issues)
    assert repair_llm.prompts
    assert "patch_type=\"universes\"" in repair_llm.prompts[0]
    assert all("patch_type=\"facts\"" not in prompt for prompt in repair_llm.prompts)
    assert all("patch_type=\"materials\"" not in prompt for prompt in repair_llm.prompts)
    assert any(
        event.event_type == "planning.patch_invalidated_for_plan_repair"
        for event in repaired_state.build_log
    )


def test_context_carries_few_shot_case_ids() -> None:
    state = _init_3b_state()
    ctx = build_generation_context_from_state(
        state, "materials", few_shot_case_ids=["assembly_3d_with_spacer_grids"]
    )
    assert ctx.few_shot_case_ids == ["assembly_3d_with_spacer_grids"]


def test_context_defaults_few_shot_case_ids_empty() -> None:
    state = _init_3b_state()
    ctx = build_generation_context_from_state(state, "materials")
    assert ctx.few_shot_case_ids == []


# ---------------------------------------------------------------------------
# 4. Materials context propagates to overlays
# ---------------------------------------------------------------------------


def test_context_materials_to_overlays() -> None:
    state = _init_3b_state()
    mat_content = {
        "patch_type": "materials",
        "materials": [
            {"material_id": "inconel718", "name": "Inconel-718", "role": "grid_inconel",
             "density_g_cm3": 8.19, "composition": {"Ni": 1.0},
             "composition_status": "approximate",
             "warnings": ["approximated as pure Ni"]},
        ],
    }
    env = PlanPatchEnvelope(
        patch_id="pre_mat", patch_type="materials",
        content=mat_content, status="valid",
    )
    state.add_patch(env)

    ctx = build_generation_context_from_state(state, "axial_overlays")
    assert "inconel718" in ctx.known_material_ids


# ---------------------------------------------------------------------------
# 5. Deterministic settings patch created
# ---------------------------------------------------------------------------


def test_deterministic_settings_patch() -> None:
    state = _init_3b_state()
    settings = build_deterministic_settings_patch(state)
    assert settings.source_strategy == "active_fuel_box"
    assert settings.plot_strategy == "full_assembly"
    assert settings.cross_sections_runtime_required is True
    assert settings.tallies_required_for_smoke_test is False


def test_executor_uses_deterministic_settings() -> None:
    """Settings should not need an LLM call."""
    responses = [
        json.dumps({"patch_type": "facts", "benchmark_id": "TEST",
                     "has_axial_geometry": True, "has_spacer_grids": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "fuel", "name": "UO2", "role": "fuel", "density_g_cm3": 10.0},
            {"material_id": "water", "name": "Water", "role": "coolant", "density_g_cm3": 0.74},
            {"material_id": "clad", "name": "Zircaloy-4", "role": "cladding",
             "density_g_cm3": 6.56, "composition": {"Zr": 1.0},
             "composition_status": "approximate",
             "warnings": ["approx"]},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fuel_pin", "kind": "fuel_pin", "cells": [
                {"id": "fuel", "role": "fuel", "material_id": "fuel"},
                {"id": "clad", "role": "cladding", "material_id": "clad"},
                {"id": "water", "role": "coolant", "material_id": "water"},
            ]},
        ]}),
        json.dumps({"patch_type": "pin_map", "lattice_size": [17, 17],
                     "default_universe_id": "fuel_pin",
                     "coordinate_convention": {"index_base": 0}}),
        json.dumps({"patch_type": "axial_layers", "layers": [
            {"layer_id": "fuel", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ]}),
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "clad",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]
    fake = FakePatchLLM(responses)
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=1,
    )
    assert result.ok is True
    # Settings envelope should be source="deterministic"
    settings_env = next(
        e for e in state.patches.values() if e.patch_type == "settings"
    )
    assert settings_env.source == "deterministic"
    event_types = [e.event_type for e in state.build_log]
    assert "planning.deterministic_settings_patch_created" in event_types


# ---------------------------------------------------------------------------
# 6. Pin_map invalid first then retry
# ---------------------------------------------------------------------------


def test_pin_map_overlap_repaired_on_first_attempt() -> None:
    """PinMapPatch with coord overlap is deterministically repaired → 1 attempt."""
    base_responses = [
        json.dumps({"patch_type": "facts", "benchmark_id": "T",
                     "has_axial_geometry": True, "has_spacer_grids": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "m", "name": "M", "role": "fuel", "density_g_cm3": 10.0},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fp", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel"}]},
            {"universe_id": "gt", "kind": "guide_tube", "cells": [
                {"id": "iw", "role": "coolant"}, {"id": "w", "role": "cladding"},
                {"id": "ow", "role": "background"},
            ]},
        ]}),
    ]
    # Overlap: [5,5] in both guide_tube AND pyrex_rod → repaired (pyrex wins)
    overlap_pin = json.dumps({
        "patch_type": "pin_map", "lattice_size": [17, 17],
        "default_universe_id": "fp",
        "coordinate_convention": {"index_base": 0},
        "guide_tube_coords": [[5, 5]],
        "pyrex_rod_coords": [[5, 5]],
    })
    tail_responses = [
        json.dumps({"patch_type": "axial_layers", "layers": [
            {"layer_id": "f", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ]}),
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "m",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]
    fake = FakePatchLLM(base_responses + [overlap_pin] + tail_responses)
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=2,
        reference_patch_policy="off",
    )
    assert result.ok is True
    # pin_map should only need 1 attempt (overlap repaired deterministically)
    # Count prompts that specifically request pin_map generation
    pin_map_gen_prompts = [p for p in fake.prompts if 'patch_type="pin_map"' in p]
    assert len(pin_map_gen_prompts) == 1


# ---------------------------------------------------------------------------
# 7. Invalid JSON in axial_layers retries only axial_layers
# ---------------------------------------------------------------------------


def test_axial_layers_invalid_json_retries_only_axial() -> None:
    base_responses = [
        json.dumps({"patch_type": "facts", "benchmark_id": "T",
                     "has_axial_geometry": True, "has_spacer_grids": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "m", "name": "M", "role": "fuel", "density_g_cm3": 10.0},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fp", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel"}]},
        ]}),
        json.dumps({"patch_type": "pin_map", "lattice_size": [17, 17],
                     "default_universe_id": "fp",
                     "coordinate_convention": {"index_base": 0}}),
    ]
    bad_json = '{"patch_type": "axial_layers", broken'
    good_json = json.dumps({"patch_type": "axial_layers", "layers": [
        {"layer_id": "f", "role": "active_fuel", "z_min_cm": 0.0, "z_max_cm": 100.0,
         "fill_type": "lattice", "fill_id": "assembly_lattice"},
    ]})
    tail = [
        json.dumps({"patch_type": "axial_overlays", "overlays": [
            {"overlay_id": "g1", "overlay_kind": "spacer_grid",
             "z_min_cm": 10.0, "z_max_cm": 12.0,
             "target_lattice_id": "assembly_lattice", "material_id": "m",
             "geometry_mode": "homogenized_open_region", "through_path_preserved": True},
        ]}),
    ]
    fake = FakePatchLLM(base_responses + [bad_json, good_json] + tail)
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=2,
    )
    assert result.ok is True
    # Earlier patches should be valid and unchanged.
    assert any(e.patch_type == "facts" and e.status == "valid" for e in state.patches.values())
    assert any(e.patch_type == "materials" and e.status == "valid" for e in state.patches.values())


# ---------------------------------------------------------------------------
# 8. Repeated pin_map failure stops executor
# ---------------------------------------------------------------------------


def test_repeated_axial_layers_failure_stops() -> None:
    """Axial layers with invalid z range always fails → executor stops."""
    base_responses = [
        json.dumps({"patch_type": "facts", "benchmark_id": "T",
                     "has_axial_geometry": True, "has_spacer_grids": True}),
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "m", "name": "M", "role": "fuel", "density_g_cm3": 10.0},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fp", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel"}]},
        ]}),
    ]
    # pin_map now passes (overlap repair fixes it)
    good_pin = json.dumps({
        "patch_type": "pin_map", "lattice_size": [17, 17],
        "default_universe_id": "fp",
        "coordinate_convention": {"index_base": 0},
        "guide_tube_coords": [[2, 2]],
    })
    # axial_layers with inverted z range always fails
    bad_axial = json.dumps({
        "patch_type": "axial_layers", "layers": [
            {"layer_id": "f", "role": "active_fuel", "z_min_cm": 100.0, "z_max_cm": 50.0,
             "fill_type": "lattice", "fill_id": "assembly_lattice"},
        ],
    })
    fake = FakePatchLLM(base_responses + [good_pin, bad_axial, bad_axial])
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=2,
        reference_patch_policy="off",
    )
    assert result.ok is False
    # Earlier patches should still be valid.
    assert any(e.patch_type == "facts" and e.status == "valid" for e in state.patches.values())
    assert any(e.patch_type == "materials" and e.status == "valid" for e in state.patches.values())
    # No assembled plan.
    assert result.assembled_plan is None
    # Should fail at axial_layers now, not pin_map.
    assert result.summary.get("failed_patch_type") == "axial_layers"


# ---------------------------------------------------------------------------
# 9. Valid patch remains unchanged after later failure
# ---------------------------------------------------------------------------


def test_valid_patch_unchanged_after_later_failure() -> None:
    state = _init_3b_state()
    facts_env = PlanPatchEnvelope(
        patch_id="pre_facts", patch_type="facts",
        content=FactsPatch(benchmark_id="VERA3").model_dump(mode="json"),
        status="valid",
    )
    state.add_patch(facts_env)

    bad_pin = json.dumps({
        "patch_type": "pin_map", "lattice_size": [17, 17],
        "default_universe_id": "fp",
        "coordinate_convention": {"index_base": 0},
        "guide_tube_coords": [[5, 5]],
        "pyrex_rod_coords": [[5, 5]],
    })
    # Provide enough responses to get through materials/universes then fail pin_map.
    responses = [
        json.dumps({"patch_type": "materials", "materials": [
            {"material_id": "m", "name": "M", "role": "fuel", "density_g_cm3": 10.0},
        ]}),
        json.dumps({"patch_type": "universes", "universes": [
            {"universe_id": "fp", "kind": "fuel_pin", "cells": [{"id": "c", "role": "fuel"}]},
        ]}),
        bad_pin, bad_pin,
    ]
    fake = FakePatchLLM(responses)
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ, state=state,
        llm_client=fake, max_patch_attempts=2,
        reference_patch_policy="off",
    )
    assert result.ok is False
    # The pre-existing facts patch should be unchanged.
    assert state.patches["pre_facts"].status == "valid"
    assert state.patches["pre_facts"].content.get("benchmark_id") == "VERA3"


# ---------------------------------------------------------------------------
# 13. VERA3 3B fake incremental execution (full)
# ---------------------------------------------------------------------------


def test_vera3_3b_full_incremental_execution() -> None:
    """Full VERA3 3B incremental execution with fake LLM."""
    raw_patches = _load_fixture_raw("3b")
    # Use first 6 patches as LLM responses (skip settings — deterministic).
    llm_responses = [json.dumps(p) for p in raw_patches if p["patch_type"] != "settings"]
    fake = FakePatchLLM(llm_responses)

    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=1,
    )
    assert result.ok is True
    assert result.assembled_plan is not None

    plan_data = result.assembled_plan
    cm = plan_data["complex_model"]

    # Lattice checks.
    lattice = cm["lattices"][0]
    pattern = lattice["universe_pattern"]
    assert len(pattern) == 17
    assert all(len(row) == 17 for row in pattern)
    flat = [uid for row in pattern for uid in row]
    assert sum(1 for u in flat if u == "guide_tube") == 24
    assert sum(1 for u in flat if u == "pyrex_rod") == 0
    assert sum(1 for u in flat if u == "thimble_plug") == 0
    assert sum(1 for u in flat if u == "instrument_tube") == 1
    assert sum(1 for u in flat if u == "fuel_pin") == 264
    assert pattern[2][8] == "guide_tube"  # 1-based (3,9)
    assert pattern[5][5] == "guide_tube"  # 1-based (6,6)
    assert pattern[8][2] == "guide_tube"  # 1-based (9,3)
    loading = next(l for l in cm["lattice_loadings"] if l["id"] == "pyrex_active_loading")
    # Pyrex is now a nested_component_override (not legacy overrides)
    transforms = loading.get("transformations", [])
    nested_ops = [t for t in transforms if t["operation_kind"] == "nested_component_override"]
    assert len(nested_ops) == 1
    assert nested_ops[0]["replacement_universe_id"] == "pyrex_inner_profile"
    assert len(nested_ops[0]["target_coordinates"]) == 16

    # Axial layers + overlays.
    assert len(cm["core"]["axial_layers"]) == 16  # upper plenum split into 3
    assert len(cm["core"]["axial_overlays"]) == 8

    # assembly3d guard.
    from openmc_agent.schemas import SimulationPlan
    plan = SimulationPlan.model_validate(plan_data)
    issues = validate_assembly3d_plan(plan, requirement=_VERA3_3B_REQ)
    error_codes = [i.code for i in issues if i.severity == "error"]
    assert "assembly3d.axial_layers_required" not in error_codes
    assert "assembly3d.default_z_extent_for_axial_problem" not in error_codes
    assert "assembly3d.spacer_grid_material_slab" not in error_codes


def test_vera3_3b_reference_structural_run_succeeds() -> None:
    raw_patches = _load_fixture_raw("3b")
    llm_responses = [
        json.dumps(p)
        for p in raw_patches
        if p["patch_type"] in {"facts", "materials", "universes"}
    ]
    fake = FakePatchLLM(llm_responses)
    state = _init_3b_state()
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=1,
        reference_patch_policy="reference_only_for_structural",
    )
    assert result.ok is True
    assert result.summary["reference_patches_used"] == [
        "pin_map", "axial_layers", "axial_overlays", "settings"
    ]
    assert result.summary["actual_pin_counts"] == {
        "fuel_pin": 264,
        "guide_tube": 24,
        "instrument_tube": 1,
        "pyrex_rod": 0,
        "thimble_plug": 0,
    }
    assert result.summary["lattice_loading_count"] == 6  # end_plug + plenum + pyrex + pyrex_upper_gas + thimble + shoulder_water
    assert result.summary["material_aliases_applied"] == {
        "grid_zircaloy4": "zircaloy4",
        "borated_water": "borated_water_3b",
    }

    plan = SimulationPlan.model_validate(result.assembled_plan)
    issues = validate_assembly3d_plan(plan, requirement=_VERA3_3B_REQ)
    assert [i for i in issues if i.severity == "error"] == []


def test_reference_only_matches_equivalent_benchmark_id_from_facts() -> None:
    raw_patches = _load_fixture_raw("3b")
    llm_responses = []
    for p in raw_patches:
        if p["patch_type"] in {"facts", "materials", "universes"}:
            patched = dict(p)
            if patched["patch_type"] == "facts":
                patched["benchmark_id"] = "VERA_PROBLEM_3"
            llm_responses.append(json.dumps(patched))
    fake = FakePatchLLM(llm_responses)
    decision = should_use_incremental_planning(_VERA3_3B_REQ)
    state = initialize_plan_build_state(requirement=_VERA3_3B_REQ, decision=decision)

    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=1,
        reference_patch_policy="reference_only_for_structural",
    )

    assert result.ok is True
    assert state.benchmark_id == "VERA_PROBLEM_3"
    assert result.summary["reference_match_status"] == "normalized"
    assert result.summary["reference_patches_used"] == [
        "pin_map", "axial_layers", "axial_overlays", "settings"
    ]
    assert len(fake.prompts) == 3


def test_reference_only_missing_reference_fails_without_structural_llm() -> None:
    raw_patches = _load_fixture_raw("3b")
    fixture_by_type = {p["patch_type"]: p for p in raw_patches}
    state = _init_3b_state()
    state.benchmark_id = "UNKNOWN_BENCHMARK"
    state.selected_variant = "3B"
    for ptype in ("facts", "materials", "universes"):
        content = dict(fixture_by_type[ptype])
        if ptype == "facts":
            content["benchmark_id"] = "UNKNOWN_BENCHMARK"
        state.add_patch(PlanPatchEnvelope(
            patch_id=f"pre_{ptype}",
            patch_type=ptype,
            content=content,
            status="valid",
        ))
    fake = FakePatchLLM([])

    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=1,
        task_order=["pin_map"],
        reference_patch_policy="reference_only_for_structural",
    )

    assert result.ok is False
    assert result.summary["failed_patch_type"] == "pin_map"
    assert "reference_patch.required_unavailable" in result.summary["issue_codes"]
    assert len(fake.prompts) == 0


def test_missing_axial_replacement_universe_schedules_targeted_repair() -> None:
    """Missing axial profile universes regenerate universes, not all patches."""
    raw_patches = _load_fixture_raw("3b")
    fixture_by_type = {patch["patch_type"]: patch for patch in raw_patches}
    state = _init_3b_state()
    for patch_type in ("facts", "materials", "universes", "pin_map"):
        content = dict(fixture_by_type[patch_type])
        if patch_type == "universes":
            content = {
                **content,
                "universes": [
                    universe for universe in content["universes"]
                    if universe["universe_id"] not in {"fuel_pin_end_plug", "fuel_pin_plenum"}
                ],
            }
        state.add_patch(PlanPatchEnvelope(
            patch_id=f"pre_{patch_type}",
            patch_type=patch_type,
            content=content,
            status="valid",
        ))

    fake = FakePatchLLM([json.dumps(fixture_by_type["axial_layers"])] * 2)
    result = run_incremental_planning(
        requirement=_VERA3_3B_REQ,
        state=state,
        llm_client=fake,
        max_patch_attempts=2,
        task_order=["axial_layers"],
    )

    assert result.ok is False
    repair = state.metadata["incremental_dependency_repair"]
    assert repair["missing_universe_ids"] == ["fuel_pin_end_plug", "fuel_pin_plenum"]
    request = state.metadata["plan_validation_repair"]
    assert request["target_patch_types"] == ["universes", "axial_layers"]
    assert state.patches["pre_facts"].status == "valid"
    assert state.patches["pre_materials"].status == "valid"


# ---------------------------------------------------------------------------
# 14. VERA3 3B pin_map response size remains small
# ---------------------------------------------------------------------------


def test_vera3_3b_pin_map_response_small() -> None:
    raw_patches = _load_fixture_raw("3b")
    pin_map_raw = next(p for p in raw_patches if p["patch_type"] == "pin_map")
    pin_map_str = json.dumps(pin_map_raw)
    # Should be well under 25K.
    assert len(pin_map_str) < 2000, f"pin_map patch is {len(pin_map_str)} bytes"
    # Should NOT contain universe_pattern.
    assert "universe_pattern" not in pin_map_raw


# ---------------------------------------------------------------------------
# Default task order / required patches
# ---------------------------------------------------------------------------


def test_default_task_order_3d_assembly() -> None:
    state = _init_3b_state()
    order = default_patch_task_order(state)
    assert "facts" in order
    assert "axial_overlays" in order  # spacer grids expected
    assert order.index("facts") < order.index("materials")
    assert order.index("materials") < order.index("universes")


def test_controlled_placement_barrier_order_precedes_axial_patches() -> None:
    order = [
        "facts",
        "materials",
        "universes",
        "assembly_catalog",
        "axial_layers",
        "axial_overlays",
        "core_layout",
        "settings",
    ]

    reordered = _order_for_controlled_gate_barriers(
        order,
        placement_controlled=True,
        material_universe_controlled=True,
        axial_geometry_controlled=False,
    )

    assert reordered.index("assembly_catalog") < reordered.index("axial_layers")
    assert reordered.index("core_layout") < reordered.index("axial_layers")
    assert reordered.index("core_layout") < reordered.index("axial_overlays")
    assert reordered.index("settings") > reordered.index("axial_overlays")


def test_required_patches_includes_overlays_for_spacer() -> None:
    state = _init_3b_state()
    required = required_patch_types_for_state(state)
    assert "axial_overlays" in required
    assert "pin_map" in required  # special pin map
    assert "axial_layers" in required
