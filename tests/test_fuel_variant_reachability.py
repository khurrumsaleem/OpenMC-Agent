"""Tests for fuel variant reachability report (P2-FULLCORE-2D-B phase 8)."""

from __future__ import annotations

from openmc_agent.plan_builder.fuel_variant_reachability import (
    build_fuel_variant_reachability_report,
    reachability_report_to_issues,
)


def test_reachability_pass_with_correct_binding():
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[
            {"variant_id": "v1", "assembly_type_ids": ["A", "C"]},
            {"variant_id": "v2", "assembly_type_ids": ["B"]},
        ],
        material_source_variants={"fuel_a": "v1", "fuel_b": "v2"},
        assembly_fuel_bindings=[
            {"assembly_type_id": "A", "default_universe_id": "fp_a",
             "resolved_fuel_variant_ids": ["v1"]},
            {"assembly_type_id": "B", "default_universe_id": "fp_b",
             "resolved_fuel_variant_ids": ["v2"]},
            {"assembly_type_id": "C", "default_universe_id": "fp_a",
             "resolved_fuel_variant_ids": ["v1"]},
        ],
        core_layout_pattern=[["A", "B", "A"], ["B", "C", "B"], ["A", "B", "A"]],
        fuel_paths_per_assembly=264,
    )
    assert report.result == "pass"
    v1 = next(e for e in report.required_variants if e.variant_id == "v1")
    v2 = next(e for e in report.required_variants if e.variant_id == "v2")
    assert v1.physical_assembly_count == 5  # A×4 + C×1
    assert v2.physical_assembly_count == 4  # B×4
    assert v1.active_fuel_path_count == 5 * 264
    assert v2.active_fuel_path_count == 4 * 264
    assert v1.reachable
    assert v2.reachable


def test_reachability_fail_when_variant_unreachable():
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[
            {"variant_id": "v1", "assembly_type_ids": ["A"]},
            {"variant_id": "v2", "assembly_type_ids": ["B"]},
        ],
        material_source_variants={"fuel_a": "v1", "fuel_b": "v2"},
        assembly_fuel_bindings=[
            {"assembly_type_id": "A", "default_universe_id": "fp_a",
             "resolved_fuel_variant_ids": ["v1"]},
            # B has no binding — v2 unreachable
        ],
        core_layout_pattern=[["A", "B"], ["B", "A"]],
    )
    assert report.result == "fail"
    v2 = next(e for e in report.required_variants if e.variant_id == "v2")
    assert not v2.reachable
    issues = reachability_report_to_issues(report)
    assert any("v2" in i.message and "unreachable" in i.message for i in issues)


def test_reachability_fail_when_assembly_failed():
    report = build_fuel_variant_reachability_report(
        assembled_plan=None,
        fuel_variant_requirements=[
            {"variant_id": "v1", "assembly_type_ids": ["A"]},
        ],
        material_source_variants={"fuel_a": "v1"},
        assembly_fuel_bindings=[
            {"assembly_type_id": "A", "default_universe_id": "fp_a",
             "resolved_fuel_variant_ids": ["v1"]},
        ],
        core_layout_pattern=[["A"]],
    )
    assert report.result == "assembly_failed"


def test_reachability_detects_collapsed_variants():
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[
            {"variant_id": "v1", "assembly_type_ids": ["A"]},
            {"variant_id": "v2", "assembly_type_ids": ["B"]},
        ],
        material_source_variants={"fuel_a": "v1", "fuel_b": "v2"},
        assembly_fuel_bindings=[
            {"assembly_type_id": "A", "default_universe_id": "fp_shared",
             "resolved_fuel_variant_ids": ["v1"]},
            {"assembly_type_id": "B", "default_universe_id": "fp_shared",
             "resolved_fuel_variant_ids": ["v2"]},
        ],
        core_layout_pattern=[["A", "B"]],
    )
    assert len(report.collapsed_variants) > 0
    issues = reachability_report_to_issues(report)
    assert any("collapsed" in i.code for i in issues)


def test_reachability_detects_unused_material():
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[
            {"variant_id": "v1", "assembly_type_ids": ["A"]},
        ],
        material_source_variants={"fuel_a": "v1", "fuel_orphan": "v_orphan"},
        assembly_fuel_bindings=[
            {"assembly_type_id": "A", "default_universe_id": "fp_a",
             "resolved_fuel_variant_ids": ["v1"]},
        ],
        core_layout_pattern=[["A"]],
    )
    assert "fuel_orphan" in report.unreachable_material_ids
    issues = reachability_report_to_issues(report)
    assert any("fuel_orphan" in i.message for i in issues)


def test_reachability_no_variants():
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[],
    )
    assert report.result == "no_fuel_variants_required"


def test_reachability_skipped_for_single_assembly_non_core():
    """A single-assembly (non-core) model has neither a core layout nor
    assembly-catalog bindings, so the full-core variant trace is not
    applicable and must not report the variant as unreachable.

    Regression: VERA3 3B (single_assembly) aborted assembly with
    ``fullcore.fuel_variant_unreachable`` because the trace structurally found
    materials/universes/assemblies all empty.  Full-core models that DO supply
    a layout/bindings are still checked (see test_reachability_fail_* above)."""
    report = build_fuel_variant_reachability_report(
        assembled_plan={"placeholder": True},
        fuel_variant_requirements=[
            {"variant_id": "3B", "assembly_type_ids": ["A"]},
        ],
        material_source_variants={"fuel_3B": "3B"},
        assembly_fuel_bindings=[],
        core_layout_pattern=[],
    )
    assert report.result == "not_applicable_non_core"
    assert reachability_report_to_issues(report) == []
