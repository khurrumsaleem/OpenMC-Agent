"""Tests for incremental planning mode decision (Phase 0)."""

from __future__ import annotations

import pytest

from openmc_agent.plan_builder.mode import (
    PlanningModeDecision,
    should_use_incremental_planning,
    TRIGGER_FEATURE_3D_AXIAL,
    TRIGGER_FEATURE_SPACER_GRID,
    TRIGGER_FEATURE_SPECIAL_PIN_MAP,
    TRIGGER_FEATURE_MULTIPLE_VARIANTS,
    TRIGGER_FEATURE_LARGE_LATTICE,
    TRIGGER_FEATURE_MULTI_ASSEMBLY,
    TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR,
    TRIGGER_HISTORY_REPAIR_LOST_AXIAL,
    TRIGGER_OVERRIDE_FORCE_INCREMENTAL,
)


# ---------------------------------------------------------------------------
# 1. Simple 2D assembly stays monolithic
# ---------------------------------------------------------------------------


def test_simple_2d_assembly_stays_monolithic() -> None:
    requirement = (
        "Build a 17x17 PWR UO2 fuel assembly, pitch 1.26 cm, "
        "reflective radial boundary, water moderator"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "monolithic"
    assert decision.triggers == []
    assert decision.confidence == 1.0


# ---------------------------------------------------------------------------
# 2. 3D axial requirement selects incremental
# ---------------------------------------------------------------------------


def test_3d_axial_requirement_selects_incremental() -> None:
    requirement = (
        "Build a 3D assembly with axial layers from z_min=0 to z_max=365 cm, "
        "including active fuel and axial reflector regions"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "incremental"
    assert TRIGGER_FEATURE_3D_AXIAL in decision.triggers
    assert decision.feature_summary["has_axial_geometry"] is True


# ---------------------------------------------------------------------------
# 3. Spacer grid requirement selects incremental
# ---------------------------------------------------------------------------


def test_spacer_grid_requirement_selects_incremental() -> None:
    requirement = (
        "Model a fuel assembly with spacer grids, mixing vanes, "
        "and grid straps at specific axial positions"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "incremental"
    assert TRIGGER_FEATURE_SPACER_GRID in decision.triggers


# ---------------------------------------------------------------------------
# 4. VERA3 / 3B special pin map selects incremental
# ---------------------------------------------------------------------------


def test_vera3_special_pin_map_selects_incremental() -> None:
    requirement = (
        "VERA3 3B benchmark: 17x17 assembly with Pyrex rods at specific "
        "positions, thimble plugs, guide tubes, and instrument tube"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "incremental"
    assert TRIGGER_FEATURE_SPECIAL_PIN_MAP in decision.triggers
    assert TRIGGER_FEATURE_MULTIPLE_VARIANTS in decision.triggers


# ---------------------------------------------------------------------------
# 5. Large JSON parse failure history selects incremental
# ---------------------------------------------------------------------------


def test_large_json_parse_error_history_selects_incremental() -> None:
    requirement = "Build a fuel assembly"
    retry_history = [
        {
            "requirement": requirement,
            "retry_count": 0,
            "plan": None,
            "validation_errors": [
                "Could not parse model response: Expecting ',' delimiter: line 42 column 3"
            ],
            "fix_suggestion": "",
        },
    ]
    plan_context = {"raw_output_length": 25000}
    decision = should_use_incremental_planning(
        requirement,
        retry_history=retry_history,
        plan_context=plan_context,
    )
    assert decision.mode == "incremental"
    assert TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR in decision.triggers


# ---------------------------------------------------------------------------
# 6. Repair lost axial layers selects incremental
# ---------------------------------------------------------------------------


def test_repair_lost_axial_layers_selects_incremental() -> None:
    requirement = "Build a 3D assembly with axial layers and spacer grids"
    retry_history = [
        {
            "requirement": requirement,
            "retry_count": 0,
            "plan": {
                "complex_model": {
                    "core": {"axial_layers": [], "axial_overlays": []},
                },
            },
            "validation_errors": [
                "assembly3d.axial_layers_required: requirement describes 3D "
                "axial geometry but the plan has no core.axial_layers"
            ],
            "fix_suggestion": "repair format",
        },
    ]
    decision = should_use_incremental_planning(
        requirement,
        retry_history=retry_history,
    )
    assert decision.mode == "incremental"
    # The requirement itself has axial triggers; the history adds repair-lost trigger
    assert TRIGGER_HISTORY_REPAIR_LOST_AXIAL in decision.triggers


# ---------------------------------------------------------------------------
# 7. force_incremental override
# ---------------------------------------------------------------------------


def test_force_incremental_override() -> None:
    requirement = "simple 2D pin cell"
    decision = should_use_incremental_planning(
        requirement,
        plan_context={"force_incremental_planning": True},
    )
    assert decision.mode == "incremental"
    assert TRIGGER_OVERRIDE_FORCE_INCREMENTAL in decision.triggers
    assert decision.confidence == 1.0


# ---------------------------------------------------------------------------
# 8. force_monolithic override keeps monolithic even with triggers
# ---------------------------------------------------------------------------


def test_force_monolithic_override() -> None:
    requirement = "3D assembly with axial layers and spacer grids"
    decision = should_use_incremental_planning(
        requirement,
        plan_context={"force_monolithic_planning": True},
    )
    assert decision.mode == "monolithic"
    assert decision.confidence == 1.0
    # Triggers are still recorded for observability.
    assert len(decision.triggers) > 0


# ---------------------------------------------------------------------------
# 9. Large lattice alone (without axial features) selects incremental
# ---------------------------------------------------------------------------


def test_large_lattice_alone_selects_incremental() -> None:
    requirement = "Build a 21x21 fast reactor assembly with guide tube replacements"
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "incremental"
    assert TRIGGER_FEATURE_LARGE_LATTICE in decision.triggers


def test_hyphenated_document_identifier_is_not_a_lattice_dimension() -> None:
    """Publication IDs such as CASL-U-2012-0131-004 are not lattice sizes."""
    decision = should_use_incremental_planning("Reference CASL-U-2012-0131-004")
    assert decision.feature_summary["large_lattice_dimension"] is None


def test_decimal_dimension_not_misread_as_large_lattice() -> None:
    """A decimal assembly cross-section like '21.50x21.50 cm^2' must not be
    misread as a 50-dim lattice (the '50x21' substring inside the decimal).
    Regression: VERA3 was wrongly tagged large_lattice_dimension=50 from its
    21.50x21.50 assembly footprint."""
    from openmc_agent.plan_builder.mode import _detect_large_lattice
    assert _detect_large_lattice("nozzle uses the full 21.50x21.50 cm^2 footprint") is None
    assert _detect_large_lattice("17x17 fuel assembly with 21.50x21.50 cm pitch") is None
    # A genuine large lattice is still detected.
    assert _detect_large_lattice("a 50x50 full-core lattice") == 50


# ---------------------------------------------------------------------------
# 10. Feature summary is populated correctly
# ---------------------------------------------------------------------------


def test_feature_summary_populated() -> None:
    requirement = (
        "3D assembly with spacer grids, Pyrex rods, 21x21 lattice, "
        "axial layers from 0 to 365 cm"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.mode == "incremental"
    fs = decision.feature_summary
    assert fs["has_axial_geometry"] is True
    assert fs["has_spacer_grid"] is True
    assert fs["has_special_pin_map"] is True
    assert fs["large_lattice_dimension"] == 21


# ---------------------------------------------------------------------------
# 11. Empty requirement defaults to monolithic
# ---------------------------------------------------------------------------


def test_empty_requirement_defaults_to_monolithic() -> None:
    decision = should_use_incremental_planning("")
    assert decision.mode == "monolithic"
    assert decision.triggers == []


# ---------------------------------------------------------------------------
# 12. PlanningModeDecision is JSON serializable
# ---------------------------------------------------------------------------


def test_planning_mode_decision_json_serializable() -> None:
    import json

    decision = should_use_incremental_planning("3D axial assembly")
    payload = decision.model_dump(mode="json")
    json_str = json.dumps(payload, ensure_ascii=False)
    restored = json.loads(json_str)
    assert restored["mode"] == decision.mode
    assert restored["triggers"] == decision.triggers


# ---------------------------------------------------------------------------
# 13. Repeated axial contract violation in history
# ---------------------------------------------------------------------------


def test_repeated_axial_contract_violation() -> None:
    requirement = "Build a 3D assembly with axial layers"
    retry_history = [
        {
            "requirement": requirement,
            "retry_count": 0,
            "plan": {
                "complex_model": {
                    "core": {"axial_layers": []},
                },
            },
            "validation_errors": ["assembly3d.axial_layers_required: ..."],
            "fix_suggestion": "repair",
        },
        {
            "requirement": requirement,
            "retry_count": 1,
            "plan": {
                "complex_model": {
                    "core": {"axial_layers": []},
                },
            },
            "validation_errors": ["assembly3d.axial_layers_required: ..."],
            "fix_suggestion": "repair",
        },
    ]
    decision = should_use_incremental_planning(
        requirement,
        retry_history=retry_history,
    )
    assert decision.mode == "incremental"
    assert TRIGGER_HISTORY_REPAIR_LOST_AXIAL in decision.triggers


# ---------------------------------------------------------------------------
# Multi-assembly core detection: schedules assembly_catalog + core_layout.
# Without this, a multi-assembly requirement is mis-routed onto the
# single-assembly task order and the assembler fails with assembly.missing_patch.
# ---------------------------------------------------------------------------


def test_multi_assembly_core_detected_en() -> None:
    requirement = (
        "Build a 3x3 assembly core (9 fuel assemblies) of types C, E, R, "
        "with core-level assembly placement and reflective outer boundary."
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.feature_summary["multi_assembly_core"] is True
    assert TRIGGER_FEATURE_MULTI_ASSEMBLY in decision.triggers


def test_multi_assembly_core_detected_cn() -> None:
    requirement = (
        "VERA4 基准问题：九个燃料组件组成的 3×3 多组件区域，"
        "反射边界只设在 3×3 区域最外侧。"
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.feature_summary["multi_assembly_core"] is True
    assert TRIGGER_FEATURE_MULTI_ASSEMBLY in decision.triggers


def test_single_assembly_not_flagged_multi() -> None:
    requirement = (
        "Build a single 17x17 PWR fuel assembly with 24 guide tubes. "
        "Model one assembly only."
    )
    decision = should_use_incremental_planning(requirement)
    assert decision.feature_summary.get("multi_assembly_core") is False
    assert TRIGGER_FEATURE_MULTI_ASSEMBLY not in decision.triggers
