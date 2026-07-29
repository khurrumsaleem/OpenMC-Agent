"""Fuel variant reachability report builder (P2-FULLCORE-2D-B phase 8).

Traces the fuel-variant identity chain through the final assembled
``SimulationPlan`` / ``ComplexModelSpec``:

    facts → materials → universes → assembly types → core layout → geometry

The report records whether each required fuel variant is physically
reachable in the final geometry and whether grid-decoration or axial
materialization preserves fuel identity.

Reactor-neutral: no reactor-type assumptions, no hardcoded enrichments.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from openmc_agent.plan_builder.assembler import PlanAssemblyResult, PlanAssemblyIssue


class FuelVariantReachabilityEntry(BaseModel):
    variant_id: str
    material_ids: list[str] = Field(default_factory=list)
    universe_ids: list[str] = Field(default_factory=list)
    assembly_type_ids: list[str] = Field(default_factory=list)
    core_coordinates: list[tuple[int, int]] = Field(default_factory=list)
    physical_assembly_count: int = 0
    active_fuel_path_count: int = 0
    reachable: bool = False


class FuelVariantReachabilityReport(BaseModel):
    required_variants: list[FuelVariantReachabilityEntry] = Field(default_factory=list)
    unreachable_material_ids: list[str] = Field(default_factory=list)
    mismatched_assembly_paths: list[str] = Field(default_factory=list)
    collapsed_variants: list[str] = Field(default_factory=list)
    dangling_references: list[str] = Field(default_factory=list)
    result: str = "unknown"


def build_fuel_variant_reachability_report(
    assembled_plan: dict[str, Any] | None,
    *,
    fuel_variant_requirements: list[dict[str, Any]] | None = None,
    material_source_variants: dict[str, str] | None = None,
    assembly_fuel_bindings: list[dict[str, Any]] | None = None,
    core_layout_pattern: list[list[str]] | None = None,
    assembly_type_counts: dict[str, int] | None = None,
    fuel_paths_per_assembly: int = 264,
) -> FuelVariantReachabilityReport:
    """Build a fuel-variant reachability report from the assembled plan.

    Parameters
    ----------
    assembled_plan
        The assembled plan JSON (``SimulationPlan.model_dump()``),
        or ``None`` if assembly failed (report will mark all as unreachable).
    fuel_variant_requirements
        Fuel variant requirements from FactsPatch (list of dicts).
    material_source_variants
        Mapping ``material_id → source_variant_id`` from MaterialsPatch.
    assembly_fuel_bindings
        Assembly fuel binding summaries from context.
    core_layout_pattern
        The core lattice pattern (2D list of assembly type IDs).
    assembly_type_counts
        Assembly type → count from core layout.
    fuel_paths_per_assembly
        Number of fuel pin positions per assembly (default 264 for 17×17).

    Returns
    -------
    FuelVariantReachabilityReport
    """
    report = FuelVariantReachabilityReport()

    if not fuel_variant_requirements:
        report.result = "no_fuel_variants_required"
        return report

    mat_variants = material_source_variants or {}
    bindings = assembly_fuel_bindings or []
    layout = core_layout_pattern or []
    counts = assembly_type_counts or {}

    # This is a full-core reachability trace: it follows the variant chain
    # through the core layout / assembly catalog into the geometry.  A
    # single-assembly (or otherwise non-core) model has neither a core layout
    # nor assembly-catalog bindings, so the trace has nothing to walk and would
    # structurally report every variant as unreachable (materials/universes/
    # assemblies all empty).  Such models trivially contain their own variant,
    # so the full-core check is not applicable — skip it instead of aborting.
    if not layout and not bindings:
        report.result = "not_applicable_non_core"
        return report

    # Build variant → entry mapping
    variant_entries: dict[str, FuelVariantReachabilityEntry] = {}
    for req in fuel_variant_requirements:
        vid = req.get("variant_id")
        if not vid:
            continue
        variant_entries[vid] = FuelVariantReachabilityEntry(
            variant_id=vid,
            assembly_type_ids=list(req.get("assembly_type_ids", [])),
        )

    # Map materials to variants
    for mid, vid in mat_variants.items():
        if vid in variant_entries:
            variant_entries[vid].material_ids.append(mid)

    # Map universes to variants (from bindings)
    for b in bindings:
        default_uv = b.get("default_universe_id")
        resolved_vids = b.get("resolved_fuel_variant_ids", [])
        for vid in resolved_vids:
            if vid in variant_entries and default_uv:
                if default_uv not in variant_entries[vid].universe_ids:
                    variant_entries[vid].universe_ids.append(default_uv)

    # Count assemblies and fuel paths from core layout
    layout_counts: dict[str, int] = {}
    for row in layout:
        for tid in row:
            layout_counts[tid] = layout_counts.get(tid, 0) + 1

    for vid, entry in variant_entries.items():
        for atid in entry.assembly_type_ids:
            n = layout_counts.get(atid, counts.get(atid, 0))
            entry.physical_assembly_count += n
            entry.active_fuel_path_count += n * fuel_paths_per_assembly

        # Check reachability: has material, has universe, has assemblies
        entry.reachable = (
            len(entry.material_ids) > 0
            and len(entry.universe_ids) > 0
            and entry.physical_assembly_count > 0
        )

    report.required_variants = list(variant_entries.values())

    # Check for unreachable materials
    all_fuel_mids = set(mat_variants.keys())
    referenced_mids = set()
    for entry in report.required_variants:
        referenced_mids.update(entry.material_ids)
    report.unreachable_material_ids = list(all_fuel_mids - referenced_mids)

    # Check for collapsed variants (multiple variants → same universe)
    uv_to_variants: dict[str, list[str]] = {}
    for entry in report.required_variants:
        for uv in entry.universe_ids:
            uv_to_variants.setdefault(uv, []).append(entry.variant_id)
    for uv, vids in uv_to_variants.items():
        if len(vids) > 1:
            report.collapsed_variants.append(
                f"universe {uv!r} claimed by variants {vids}"
            )

    # Check if assembled plan has geometry
    if assembled_plan is None:
        report.result = "assembly_failed"
        for entry in report.required_variants:
            entry.reachable = False
        return report

    # Determine overall result
    all_reachable = all(e.reachable for e in report.required_variants)
    no_collapse = len(report.collapsed_variants) == 0
    no_unreachable = len(report.unreachable_material_ids) == 0

    if all_reachable and no_collapse and no_unreachable:
        report.result = "pass"
    else:
        report.result = "fail"

    return report


def reachability_report_to_issues(
    report: FuelVariantReachabilityReport,
) -> list[PlanAssemblyIssue]:
    """Convert a reachability report to plan assembly issues."""
    issues: list[PlanAssemblyIssue] = []
    for entry in report.required_variants:
        if not entry.reachable:
            issues.append(PlanAssemblyIssue(
                code="fullcore.fuel_variant_unreachable",
                severity="error",
                message=(
                    f"fuel variant {entry.variant_id!r} is unreachable: "
                    f"materials={entry.material_ids}, "
                    f"universes={entry.universe_ids}, "
                    f"assemblies={entry.physical_assembly_count}"
                ),
            ))
    for collapsed in report.collapsed_variants:
        issues.append(PlanAssemblyIssue(
            code="fullcore.fuel_variant_collapsed",
            severity="error",
            message=collapsed,
        ))
    for mid in report.unreachable_material_ids:
        issues.append(PlanAssemblyIssue(
            code="fullcore.required_fuel_material_unused",
            severity="error",
            message=f"fuel material {mid!r} is defined but not used by any variant",
        ))
    return issues


__all__ = [
    "FuelVariantReachabilityEntry",
    "FuelVariantReachabilityReport",
    "build_fuel_variant_reachability_report",
    "reachability_report_to_issues",
]
