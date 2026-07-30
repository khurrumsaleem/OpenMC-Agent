"""Rectangular-assembly renderer.

Handles ``complex_model.kind == "assembly"`` with at least one ``rect`` lattice.
It reports four capability levels:

- ``runnable``: structure and materials complete and a low-cost smoke test is configured.
- ``exportable``: structure and materials complete enough to export XML.
- ``skeleton``: recognizable assembly IR but missing materials or structural issues;
  emits a review-only skeleton instead.
- ``none``: this renderer does not apply (no complex_model or kind != "assembly").
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from openmc_agent.assembly3d_guard import (
    assembly3d_structural_issues,
)
from openmc_agent.executor import render_openmc_assembly_script
from openmc_agent.lattice_validation import lattice_pin_count_issues
from openmc_agent.renderers.core import _core_renderability_errors
from openmc_agent.renderers.base import BaseRenderer, RenderResult, low_cost_runnable
from openmc_agent.reachability import (
    ActiveDependencies,
    collect_active_dependencies,
)
from openmc_agent.renderers.skeleton import (
    _write_capability_report,
    _write_todo,
    emit_skeleton,
)
from openmc_agent.error_catalog import issue_from_catalog
from openmc_agent.schemas import (
    ComplexMaterialSpec,
    ComplexModelSpec,
    LatticeSpec,
    RenderCapabilityReport,
    SimulationPlan,
    ValidationIssue,
)
from openmc_agent.validator import validate_openmc_script


_COMPOSITE_REGION_SURFACE_KINDS = {"rectangular_prism", "hexagonal_prism"}


def _iss(code: str, message: str, schema_path: str = "") -> ValidationIssue:
    """Build a structured diagnostic issue with a stable code.

    Registered codes pull severity/knowledge/repair_hints from
    :data:`error_catalog.ERROR_CATALOG`; unknown codes degrade gracefully via
    :func:`issue_from_catalog` so every diagnostic still carries a stable code.
    """
    return issue_from_catalog(code, message=message, schema_path=schema_path or None)


class RectAssemblyRenderer(BaseRenderer):
    name = "assembly"
    supported_kinds = ("assembly",)

    def can_render(self, plan: SimulationPlan) -> RenderCapabilityReport:
        model = plan.complex_model
        if model is None or model.kind != "assembly":
            return RenderCapabilityReport(
                renderability="none",
                is_executable=False,
                supported_renderer="none",
                reasons=["assembly renderer requires complex_model.kind='assembly'"],
            )

        deps = collect_active_dependencies(plan)
        errors, warnings = _assembly_diagnostics(model, deps)
        if errors:
            return RenderCapabilityReport(
                renderability="skeleton",
                is_executable=False,
                supported_renderer="assembly",
                executable_subsystems=[],
                unsupported_subsystems=_assembly_subsystems(model),
                reasons=[issue.message for issue in errors],
                issues=errors,
                warnings=warnings,
                required_human_confirmations=_material_confirmations(model, deps),
            )
        renderability = "runnable" if low_cost_runnable(plan) else "exportable"
        executable_subsystems = [
            "materials",
            "cells",
            "universes",
            "rect_lattice",
            "assembly",
        ]
        if _has_axial_assembly_layers(model):
            executable_subsystems.extend(["axial_layers", "lattice_loadings"])
        if _has_renderable_overlay(model):
            executable_subsystems.append("axial_overlays")
        return RenderCapabilityReport(
            renderability=renderability,
            supported_renderer="assembly",
            executable_subsystems=executable_subsystems,
            unsupported_subsystems=[],
            reasons=["Current executor supports rectangular assembly lattice rendering."],
            warnings=warnings,
            required_human_confirmations=_material_confirmations(model, deps),
        )

    def render(self, plan: SimulationPlan, outdir: Path) -> RenderResult:
        outdir.mkdir(parents=True, exist_ok=True)
        capability = self.can_render(plan)
        model = plan.complex_model
        assert model is not None

        if capability.renderability in {"exportable", "runnable"}:
            try:
                script = render_openmc_assembly_script(model, plot_specs=plan.plot_specs)
            except ValueError as exc:
                # Defensive: if the executor rejects what can_render accepted,
                # fall back to a skeleton so the user still gets reviewable output.
                return emit_skeleton(self.name, plan, outdir, _skeleton_capability(plan, str(exc)))
            script_report = validate_openmc_script(script)
            if not script_report.is_valid:
                fallback = _skeleton_capability(plan, "; ".join(script_report.errors))
                return emit_skeleton(self.name, plan, outdir, fallback)
            model_path = outdir / "model.py"
            model_path.write_text(script, encoding="utf-8")
            # A stale TODO.md from a previous skeleton run must not persist when
            # the current render is executable.
            stale_todo = outdir / "TODO.md"
            if stale_todo.exists():
                stale_todo.unlink()
            files = [str(model_path), _write_capability_report(outdir, capability)]
            todo = _write_todo(outdir, self.name, capability)
            if todo is not None:
                files.append(todo)
            # Verification digest: a structured pass/fail report + roster an
            # expert can scan in seconds (complements the visual plots).
            try:
                from openmc_agent.verification import write_verification_digest
                digest_files = write_verification_digest(model, outdir)
                files.append(digest_files["markdown"])
            except Exception:
                pass
            # Voxel converter script: expert runs this after `openmc -p` to get
            # a ParaView-ready .vti from the raw voxel .h5 dump.
            try:
                convert_script = outdir / "convert_voxel.py"
                convert_script.write_text(
                    '"""Convert verification_voxel.h5 -> .vti for ParaView/VisIt.\n'
                    'Run after: openmc -p\n"""\n'
                    "from openmc_agent.verification import convert_voxel_to_vti\n"
                    "from pathlib import Path\n"
                    "d = Path(__file__).parent\n"
                    "vti = convert_voxel_to_vti(d / 'plots/verification_voxel.h5', d / 'plots.xml')\n"
                    "print(f'wrote {vti} -- open in ParaView')\n",
                    encoding="utf-8",
                )
                files.append(str(convert_script))
            except Exception:
                pass
            return RenderResult(
                renderer_name=self.name,
                renderability=capability.renderability,
                is_executable=capability.is_executable,
                script=script,
                output_files=files,
                warnings=capability.warnings,
                errors=[],
                capability=capability,
            )

        # skeleton mode
        return emit_skeleton(self.name, plan, outdir, capability)


# -- diagnostics ----------------------------------------------------------


def _assembly_diagnostics(
    model: ComplexModelSpec,
    deps: ActiveDependencies,
) -> tuple[list[ValidationIssue], list[str]]:
    """Return (blocking_issues, warnings). Empty issue list means exportable.

    ``deps`` partitions declared objects by reachability from the default
    lattice. Only *active* materials can block the default model; gaps in
    candidate / inactive materials become warnings instead.
    """
    errors: list[ValidationIssue] = []
    warnings: list[str] = []

    material_ids = {material.id for material in model.materials}
    region_ids = {region.id for region in model.regions}
    composite_region_ids = {
        surface.id
        for surface in model.surfaces
        if surface.kind in _COMPOSITE_REGION_SURFACE_KINDS
    }
    region_like_ids = region_ids | composite_region_ids
    surface_ids = {surface.id for surface in model.surfaces}
    universe_ids = {universe.id for universe in model.universes}
    cell_ids = {cell.id for cell in model.cells}

    # Check 2: at least one rect lattice.
    rect_lattices = [lat for lat in model.lattices if lat.kind == "rect"]
    hex_lattices = [lat for lat in model.lattices if lat.kind == "hex"]
    if not model.lattices:
        errors.append(_iss("assembly.requires_lattice", "assembly renderer requires a RectLattice", "complex_model.lattices"))
    elif not rect_lattices:
        errors.append(_iss("assembly.requires_rect_lattice", "assembly renderer currently supports RectLattice only", "complex_model.lattices"))
    for lattice in hex_lattices:
        schema_base = f"complex_model.lattices.{lattice.id}"
        errors.append(
            _iss(
                "lattice.hex.renderer_unsupported",
                f"hex lattice {lattice.id!r} requires a HexAssemblyRenderer; current renderer output remains skeleton.",
                schema_base,
            )
        )
        if not lattice.rings:
            errors.append(
                _iss(
                    "lattice.hex.rings_missing",
                    f"hex lattice {lattice.id!r} rings are missing",
                    f"{schema_base}.rings",
                )
            )
        else:
            invalid = _invalid_hex_ring_lengths(lattice.rings)
            if invalid:
                errors.append(
                    _iss(
                        "lattice.hex.ring_shape_invalid",
                        f"hex lattice {lattice.id!r} has invalid ring lengths: {invalid}",
                        f"{schema_base}.rings",
                    )
                )
            missing = sorted(
                {
                    uid
                    for ring in lattice.rings
                    for uid in ring
                    if uid not in universe_ids
                }
            )
            if missing:
                errors.append(
                    _iss(
                        "lattice.universe_ref_missing",
                        f"hex lattice {lattice.id!r} rings reference missing universes: {missing}",
                        f"{schema_base}.rings",
                    )
                )
        if lattice.outer_universe_id is None:
            errors.append(
                _iss(
                    "lattice.hex.outer_universe_missing",
                    f"hex lattice {lattice.id!r} outer_universe_id is missing",
                    f"{schema_base}.outer_universe_id",
                )
            )
        errors.append(
            _iss(
                "lattice.hex.orientation_unverified",
                f"hex lattice {lattice.id!r} orientation, pitch convention, and ring ordering require documentation verification before renderer work.",
                schema_base,
            )
        )

    if not model.assemblies or model.assemblies[0].lattice_id is None:
        errors.append(_iss("assembly.requires_assembly_spec", "assembly renderer requires an AssemblySpec with lattice_id", "complex_model.assemblies"))

    if not model.cells:
        errors.append(_iss("assembly.requires_cells", "assembly renderer requires cells", "complex_model.cells"))
    if not model.universes:
        errors.append(_iss("assembly.requires_universes", "assembly renderer requires universes", "complex_model.universes"))
    if not model.materials:
        errors.append(_iss("assembly.requires_materials", "assembly renderer requires materials", "complex_model.materials"))

    # Check 8: material completeness (density + composition/chemical_formula).
    # Only materials reachable from the default lattice (active) can block;
    # candidate / inactive material gaps only warn, so an un-inserted
    # burnable-poison universe with an incomplete borosilicate glass does not
    # downgrade the default F/G assembly.
    warnings.extend(_material_completeness_messages(model, deps, errors))

    # Candidate universes defined but not inserted into the default lattice.
    for universe_id in sorted(deps.inactive_universe_ids):
        warnings.append(
            f"universe {universe_id!r} is not inserted in the default lattice "
            "(candidate only)"
        )

    # Check 4 + 3: universe_pattern references and shape consistency.
    for lattice in rect_lattices:
        errors.extend(_lattice_pattern_errors(lattice, universe_ids))

    # Check 5: cell.region_id references an existing region (when set).
    for cell in model.cells:
        if cell.region_id is not None and cell.region_id not in region_like_ids:
            errors.append(_iss(
                "cell.region_ref_missing",
                f"cell {cell.id!r} references missing region {cell.region_id!r}",
                f"complex_model.cells.{cell.id}.region_id",
            ))

    # Check 6: region.surface_ids reference existing surfaces.
    for region in model.regions:
        missing = [sid for sid in region.surface_ids if sid not in surface_ids]
        if missing:
            errors.append(_iss(
                "region.surface_ref_missing",
                f"region {region.id!r} references missing surfaces: {missing}",
                f"complex_model.regions.{region.id}.surface_ids",
            ))

    # Check 7: material cells reference existing materials.
    for cell in model.cells:
        if cell.fill_type == "material" and cell.fill_id and cell.fill_id not in material_ids:
            errors.append(_iss(
                "cell.material_ref_missing",
                f"cell {cell.id!r} references missing material {cell.fill_id!r}",
                f"complex_model.cells.{cell.id}.fill_id",
            ))

    # Universe -> cell references.
    for universe in model.universes:
        missing = [cid for cid in universe.cell_ids if cid not in cell_ids]
        if not missing:
            continue
        if universe.id in deps.universe_ids:
            errors.append(_iss(
                "universe.cell_ref_missing",
                f"universe {universe.id!r} references missing cells: {missing}",
                f"complex_model.universes.{universe.id}.cell_ids",
            ))
        else:
            warnings.append(
                f"inactive candidate universe {universe.id!r} references missing "
                f"cells: {missing}; this only blocks models that use that universe"
            )

    # Existing reflector / control-rod reference checks.
    for reflector in model.reflectors:
        if reflector.material_id not in material_ids:
            errors.append(_iss(
                "reflector.material_ref_missing",
                f"reflector {reflector.id!r} references missing material",
                f"complex_model.reflectors.{reflector.id}.material_id",
            ))
        if reflector.region_id is None or reflector.region_id not in region_like_ids:
            errors.append(_iss(
                "reflector.region_ref_missing",
                f"reflector {reflector.id!r} requires a valid region_id",
                f"complex_model.reflectors.{reflector.id}.region_id",
            ))

    lattice_universe_ids = {
        uid
        for lattice in rect_lattices
        for row in lattice.universe_pattern
        for uid in row
    }
    for control_rod in model.control_rods:
        if control_rod.absorber_material_id not in material_ids:
            errors.append(_iss(
                "control_rod.material_ref_missing",
                f"control rod {control_rod.id!r} references missing absorber material",
                f"complex_model.control_rods.{control_rod.id}.absorber_material_id",
            ))
        if (
            control_rod.guide_tube_region_id is not None
            and control_rod.guide_tube_region_id not in region_like_ids
        ):
            errors.append(_iss(
                "control_rod.region_ref_missing",
                f"control rod {control_rod.id!r} references missing guide_tube_region_id",
                f"complex_model.control_rods.{control_rod.id}.guide_tube_region_id",
            ))
        if control_rod.guide_tube_region_id is None and not any(
            position_id in lattice_universe_ids for position_id in control_rod.position_ids
        ):
            errors.append(_iss(
                "control_rod.position_ref_missing",
                f"control rod {control_rod.id!r} must reference a lattice universe "
                "position or a guide_tube_region_id",
                f"complex_model.control_rods.{control_rod.id}.position_ids",
            ))

    # Check 9 + 10: cylinder radii positive and within pitch/2.
    errors.extend(_cylinder_geometry_errors(model, warnings))

    # Check 11: assembly boundary explicitly specified.
    if model.assemblies:
        boundary = model.assemblies[0].boundary
        if boundary is None:
            warnings.append(
                "assembly boundary is not specified; export will default to vacuum"
            )

    if _has_axial_assembly_layers(model):
        errors.extend(_axial_assembly_modeling_errors(model))
        errors.extend(_core_renderability_errors(model))

    # Overlay fidelity note: record the rendering level so the capability
    # report states the fidelity honestly.
    if not errors and _has_renderable_overlay(model):
        has_level2 = any(
            o.geometry_mode == "mass_conserving_outer_frame"
            for o in (model.core.axial_overlays if model.core else [])
        )
        if has_level2:
            warnings.append(
                "Spacer/support grids rendered as Level 2 mass-conserving "
                "outer-frame overlays: pin/tube through-paths preserved; "
                "grid mass conserved via thin square frame per pitch cell."
            )
        else:
            warnings.append(
                "Spacer/support grids rendered as Level 1 homogenized open-region "
                "overlays: pin/tube through-paths preserved; grid straps and mixing "
                "vanes not explicitly modeled; volume fraction not calibrated."
            )

    return errors, warnings


def _has_renderable_overlay(model: ComplexModelSpec) -> bool:
    """True when at least one overlay will be turned into Level 1 geometry."""
    from openmc_agent.axial_overlay import overlay_is_structurally_renderable

    if model.core is None:
        return False
    return any(
        overlay_is_structurally_renderable(overlay, model)
        for overlay in model.core.axial_overlays
    )


def _has_axial_assembly_layers(model: ComplexModelSpec) -> bool:
    return model.core is not None and bool(model.core.axial_layers)


def _axial_assembly_modeling_errors(model: ComplexModelSpec) -> list[ValidationIssue]:
    """Generic 3D assembly modeling guards.

    Thin spacer/support-grid layers are not an entire slab of grid material:
    fuel rods and guide/instrument tubes pass through them. This delegates to
    :func:`openmc_agent.assembly3d_guard.assembly3d_structural_issues` so the
    plan validator and the renderer share one source of truth for the
    ``assembly3d.*`` codes.

    Lattice-loading structural issues (replacement universe missing, source
    universe missing, loading ref missing, etc.) are also surfaced here via
    :func:`lattice_loading_structural_issues` so they are visible at
    ``can_render`` time, not deferred to ``render()``.
    """
    from openmc_agent.lattice_loading_validation import (
        lattice_loading_structural_issues,
    )

    assert model.core is not None
    issues = assembly3d_structural_issues(model)
    issues.extend(lattice_loading_structural_issues(model))
    return issues


def _lattice_pattern_errors(
    lattice: LatticeSpec,
    universe_ids: set[str],
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    schema_base = f"complex_model.lattices.{lattice.id}.universe_pattern"
    pattern = lattice.universe_pattern
    if not pattern:
        if lattice.kind == "rect":
            errors.append(
                _iss(
                    "lattice.pattern_missing",
                    f"lattice {lattice.id!r} requires universe_pattern before export",
                    schema_base,
                )
            )
        return errors
    if lattice.shape is not None:
        expected = tuple(lattice.shape)
        # OpenMC RectLattice shape is (num_x, num_y); pattern is rows of columns.
        shape_rows_cols = _shape_to_rows_cols(expected)
        if shape_rows_cols is not None:
            expected_rows, expected_cols = shape_rows_cols
            actual_rows = len(pattern)
            actual_cols = len(pattern[0]) if pattern else 0
            if expected_rows != actual_rows or expected_cols != actual_cols:
                errors.append(
                    _iss(
                        "lattice.shape_pattern_mismatch",
                        f"lattice {lattice.id!r} shape {list(expected)} does not match "
                        f"universe_pattern dimensions {actual_rows}x{actual_cols}",
                        schema_base,
                    )
                )
    else:
        # Without an explicit shape, just confirm the pattern is rectangular.
        if pattern:
            col_counts = {len(row) for row in pattern}
            if len(col_counts) > 1:
                errors.append(
                    _iss(
                        "lattice.pattern_ragged_rows",
                        f"lattice {lattice.id!r} universe_pattern rows have unequal lengths",
                        schema_base,
                    )
                )

    missing = [
        uid
        for row in pattern
        for uid in row
        if uid not in universe_ids
    ]
    if missing:
        errors.append(
            _iss(
                "lattice.universe_ref_missing",
                f"lattice {lattice.id!r} universe_pattern references missing universes: "
                f"{_dedupe(missing)}",
                schema_base,
            )
        )

    # Hard gate at export: a pin-count mismatch must block XML export so a wrong
    # map can never become a runnable model.
    errors.extend(lattice_pin_count_issues([lattice], message_style="renderer"))
    return errors


def _shape_to_rows_cols(shape: tuple[int, ...]) -> tuple[int, int] | None:
    if len(shape) == 2:
        nx, ny = int(shape[0]), int(shape[1])
        return ny, nx  # pattern is [ny rows][nx cols]
    if len(shape) == 1:
        n = int(shape[0])
        return n, n
    return None


def _invalid_hex_ring_lengths(rings: list[list[str]]) -> list[str]:
    invalid: list[str] = []
    for index, ring in enumerate(rings):
        expected = 1 if index == 0 else 6 * index
        if len(ring) != expected:
            invalid.append(f"ring {index}: expected {expected}, got {len(ring)}")
    return invalid


def _cylinder_geometry_errors(
    model: ComplexModelSpec,
    warnings: list[str],
) -> list[ValidationIssue]:
    errors: list[ValidationIssue] = []
    radii: list[float] = []
    for surface in model.surfaces:
        if surface.kind in {"zcylinder", "ycylinder", "xcylinder"}:
            schema_path = f"complex_model.surfaces.{surface.id}.parameters.r"
            r = surface.parameters.get("r")
            if r is None:
                warnings.append(
                    f"{surface.kind} surface {surface.id!r} has no 'r' parameter"
                )
                continue
            try:
                radius = float(r)
            except (TypeError, ValueError):
                errors.append(_iss(
                    "surface.cylinder_radius_invalid",
                    f"{surface.kind} surface {surface.id!r} has non-numeric radius {r!r}",
                    schema_path,
                ))
                continue
            if radius <= 0:
                errors.append(_iss(
                    "surface.cylinder_radius_invalid",
                    f"{surface.kind} surface {surface.id!r} radius must be positive, got {radius}",
                    schema_path,
                ))
            else:
                radii.append(radius)

    if radii:
        pitch = _assembly_pitch_cm(model)
        if pitch is None:
            warnings.append("cannot verify cylinder radius < pitch/2: assembly pitch unknown")
        else:
            max_radius = max(radii)
            # A cylinder may touch the half-pitch boundary of a square lattice
            # cell; the corner regions remain covered by the background cell.
            if max_radius > pitch / 2.0 + 1.0e-12:
                # Downgrade to warning: an oversized cylinder (e.g. an
                # outer-moderator boundary or a component whose radius exceeds
                # the fuel pin pitch) is a geometry imperfection, not a
                # structural break. The background cell still fills the
                # uncovered lattice corners, so rendering proceeds.
                warnings.append(
                    f"maximum cylinder outer radius {max_radius} exceeds pitch/2 "
                    f"({pitch / 2.0}) — geometry may overlap but model remains renderable"
                )
    return errors


def _assembly_pitch_cm(model: ComplexModelSpec) -> float | None:
    for lattice in model.lattices:
        if lattice.kind == "rect" and lattice.pitch_cm:
            return float(lattice.pitch_cm[0])
    if model.assemblies and model.assemblies[0].pitch_cm is not None:
        return float(model.assemblies[0].pitch_cm)
    return None


def _assembly_subsystems(model: ComplexModelSpec) -> list[str]:
    present = []
    for name, values in (
        ("materials", model.materials),
        ("surfaces", model.surfaces),
        ("regions", model.regions),
        ("cells", model.cells),
        ("universes", model.universes),
        ("lattices", model.lattices),
        ("assemblies", model.assemblies),
        ("reflectors", model.reflectors),
        ("control_rods", model.control_rods),
    ):
        if values:
            present.append(name)
    if model.core is not None and model.core.axial_layers:
        present.append("axial_layers")
    if model.lattice_loadings:
        present.append("lattice_loadings")
    return present


def _material_completeness_messages(
    model: ComplexModelSpec,
    deps: ActiveDependencies,
    errors: list[ValidationIssue],
) -> list[str]:
    """Split material gaps into blocking errors (active) and warnings (inactive).

    Mutates ``errors`` in place with active-material gaps and returns the
    inactive-material warnings. Active = reachable from the default lattice.
    """
    warnings: list[str] = []
    for material in model.materials:
        is_macroscopic = material.macroscopic is not None
        is_mixture = getattr(material, "is_mixture", False) or (
            len(getattr(material, "mixture_component_ids", [])) > 0
        )
        is_sum_density = getattr(material, "density_unit", None) == "sum"
        missing_density = (
            not is_macroscopic
            and not is_mixture
            and not is_sum_density
            and (material.density_unit is None or material.density_value is None)
        )
        missing_composition = (
            not is_macroscopic
            and not is_mixture
            and not material.composition
            and not material.chemical_formula
        )
        mixed_percent_types = _material_has_mixed_percent_types(material)
        schema_path = f"complex_model.materials.{material.id}"
        if material.id in deps.material_ids:
            if missing_density:
                errors.append(_iss(
                    "material.missing_density",
                    f"material {material.id!r} is missing density",
                    schema_path,
                ))
            if missing_composition:
                errors.append(_iss(
                    "material.missing_composition",
                    f"material {material.id!r} is missing composition or chemical_formula",
                    schema_path,
                ))
            if mixed_percent_types and material.chemical_formula is None:
                errors.append(_iss(
                    "material.mixed_percent_type",
                    f"material {material.id!r} mixes atom and weight percents "
                    "without chemical_formula fallback",
                    schema_path,
                ))
            elif mixed_percent_types:
                warnings.append(
                    f"material {material.id!r} mixes atom and weight percents; "
                    "using chemical_formula fallback"
                )
        elif missing_density or missing_composition or mixed_percent_types:
            # Candidate / orphan material: gaps only warn, never block.
            suffix = "; this only blocks models that use its universe"
            if missing_density:
                warnings.append(
                    f"inactive candidate material {material.id!r} is missing density{suffix}"
                )
            if missing_composition:
                warnings.append(
                    f"inactive candidate material {material.id!r} is missing "
                    f"composition or chemical_formula{suffix}"
                )
            if mixed_percent_types:
                warnings.append(
                    f"inactive candidate material {material.id!r} mixes atom and "
                    f"weight percents{suffix}"
                )
    return warnings


def _material_has_mixed_percent_types(material: ComplexMaterialSpec) -> bool:
    percent_types = {component.percent_type for component in material.composition}
    return len(percent_types) > 1


def _material_confirmations(
    model: ComplexModelSpec,
    deps: ActiveDependencies | None = None,
) -> list[str]:
    """Human-confirmation entries for every material gap.

    ``deps`` tags candidate / inactive materials so reviewers can see that a gap
    only matters if the owning universe is later inserted. Accepts ``None`` so
    callers without a reachability analysis (e.g. defensive skeleton fallbacks)
    still work, treating every material as potentially active.
    """
    active_material_ids = deps.material_ids if deps is not None else None
    confirmations: list[str] = []
    for material in model.materials:
        is_macroscopic = material.macroscopic is not None
        is_active = (
            material.id in active_material_ids if active_material_ids is not None else True
        )
        prefix = "material" if is_active else "inactive candidate material"
        if not is_macroscopic and (
            material.density_unit is None or material.density_value is None
        ):
            confirmations.append(f"{prefix} {material.id}: is missing density")
        if not is_macroscopic and not material.composition and not material.chemical_formula:
            confirmations.append(
                f"{prefix} {material.id}: is missing composition or chemical_formula"
            )
        confirmations.extend(
            f"{prefix} {material.id}: {item}"
            for item in material.requires_human_confirmation
        )
    for lattice in model.lattices:
        confirmations.extend(
            f"lattice {lattice.id}: {item}"
            for item in lattice.requires_human_confirmation
        )
    return list(dict.fromkeys(confirmations))


def _dedupe(items: list[str]) -> list[str]:
    return list(dict.fromkeys(items))


# -- skeleton emission ----------------------------------------------------


def _skeleton_capability(plan: SimulationPlan, extra_reason: str) -> RenderCapabilityReport:
    model = plan.complex_model
    assert model is not None
    # Preserve active/inactive labelling even in the defensive skeleton fallback.
    deps = collect_active_dependencies(plan)
    base_reasons = [
        "Assembly IR could not be exported safely; generating a review-only skeleton.",
        extra_reason,
    ]
    return RenderCapabilityReport(
        renderability="skeleton",
        is_executable=False,
        supported_renderer="assembly",
        executable_subsystems=[],
        unsupported_subsystems=_assembly_subsystems(model),
        reasons=base_reasons,
        required_human_confirmations=_material_confirmations(model, deps),
    )
