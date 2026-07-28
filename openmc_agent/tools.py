import subprocess
import sys
import re
import os
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from difflib import get_close_matches
from pathlib import Path
from typing import Any

from openmc_agent.error_catalog import issue_from_catalog
from openmc_agent.schemas import RepairHint, SimulationPlan, ValidationIssue, ValidationReport


@dataclass(frozen=True)
class ToolResult:
    name: str
    ok: bool
    command: list[str] = field(default_factory=list)
    returncode: int | None = None
    stdout: str = ""
    stderr: str = ""
    artifacts: list[str] = field(default_factory=list)
    error: str = ""
    issues: list[ValidationIssue] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {
            "name": self.name,
            "ok": self.ok,
            "command": self.command,
            "returncode": self.returncode,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "artifacts": self.artifacts,
            "error": self.error,
            "issues": [issue.model_dump(mode="json") for issue in self.issues],
        }


def export_xml(model_path: str | Path, *, timeout: float = 60.0) -> ToolResult:
    path = Path(model_path)
    command = [sys.executable, path.name]
    result = subprocess.run(
        command,
        cwd=path.parent,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    artifacts = _existing_xml_artifacts(path.parent)
    missing_required = _missing_required_xml_artifacts(path.parent)
    closure_issues: list[ValidationIssue] = []
    if result.returncode == 0 and not missing_required:
        closure_issues = _geometry_lattice_reference_issues(path.parent / "geometry.xml")
    closure_error = _format_geometry_reference_issues(closure_issues)
    ok = result.returncode == 0 and not missing_required and not closure_issues
    error = ""
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
    elif missing_required:
        error = "export_xml produced no required XML artifacts: " + ", ".join(missing_required)
    elif closure_error:
        error = closure_error
    return ToolResult(
        name="export_xml",
        ok=ok,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        artifacts=artifacts,
        error=error,
        issues=closure_issues,
    )


def run_geometry_plots(run_dir: str | Path, *, timeout: float = 60.0) -> ToolResult:
    path = Path(run_dir)
    command = ["openmc", "-p"]
    result = subprocess.run(
        command,
        cwd=path,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    return ToolResult(
        name="run_geometry_plots",
        ok=result.returncode == 0,
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        artifacts=_plot_artifacts(path),
        error="" if result.returncode == 0 else (result.stderr or result.stdout).strip(),
    )


def run_smoke_test(
    run_dir: str | Path,
    plan: SimulationPlan,
    *,
    max_particles: int = 1000,
    max_batches: int = 20,
    timeout: float = 120.0,
) -> ToolResult:
    from openmc_agent.executor import render_openmc_smoke_test_script

    cross_sections_path = os.environ.get("OPENMC_CROSS_SECTIONS")
    if cross_sections_path:
        from openmc_agent.material_species import preflight_plan_material_species
        try:
            preflight_errors = preflight_plan_material_species(plan, cross_sections_path)
        except Exception as exc:
            preflight_errors = [{
                "code": "runtime.material_species_unresolved",
                "cross_sections_path": cross_sections_path,
                "detail": str(exc),
                "suggested_patch_type": "materials",
            }]
        if preflight_errors:
            return ToolResult(
                name="run_smoke_test", ok=False, returncode=None,
                error=json.dumps(preflight_errors, ensure_ascii=False),
                issues=[issue_from_catalog(
                    error["code"],
                    message=(f"material species preflight failed for {error.get('material_id')}: "
                             f"{error.get('species_name')}; repair materials patch before OpenMC"),
                ) for error in preflight_errors],
            )

    settings = plan.execution_check.settings
    if settings.particles > max_particles or settings.batches > max_batches:
        return ToolResult(
            name="run_smoke_test",
            ok=False,
            error=(
                "Smoke test settings exceed safety limits: "
                f"particles={settings.particles} max_particles={max_particles}, "
                f"batches={settings.batches} max_batches={max_batches}"
            ),
        )

    plan = _refresh_executable_capability(plan)

    path = Path(run_dir)
    path.mkdir(parents=True, exist_ok=True)
    smoke_model_path = path / "smoke_model.py"
    try:
        smoke_script = render_openmc_smoke_test_script(plan)
    except ValueError as exc:
        return ToolResult(
            name="run_smoke_test",
            ok=False,
            returncode=None,
            error=str(exc),
        )
    smoke_model_path.write_text(smoke_script, encoding="utf-8")

    export_command = [sys.executable, smoke_model_path.name]
    export_result = subprocess.run(
        export_command,
        cwd=path,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if export_result.returncode != 0:
        return ToolResult(
            name="run_smoke_test",
            ok=False,
            command=export_command,
            returncode=export_result.returncode,
            stdout=export_result.stdout,
            stderr=export_result.stderr,
            artifacts=[str(smoke_model_path), *_existing_xml_artifacts(path)],
            error=(export_result.stderr or export_result.stdout).strip(),
        )

    run_command = ["openmc"]
    try:
        run_result = subprocess.run(
            run_command,
            cwd=path,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            name="run_smoke_test",
            ok=False,
            command=run_command,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nTimed out after {timeout}s",
            artifacts=[str(smoke_model_path), *_existing_xml_artifacts(path)],
            error=f"OpenMC transport timed out after {timeout}s",
            issues=[
                issue_from_catalog(
                    "runtime.openmc_timeout",
                    message=f"OpenMC transport timed out after {timeout}s",
                    grep_patterns=["timeout", "timed out"],
                )
            ],
        )
    return ToolResult(
        name="run_smoke_test",
        ok=run_result.returncode == 0,
        command=run_command,
        returncode=run_result.returncode,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        artifacts=[str(smoke_model_path), *_existing_xml_artifacts(path), *_statepoint_artifacts(path)],
        error="" if run_result.returncode == 0 else (run_result.stderr or run_result.stdout).strip(),
    )


def _refresh_executable_capability(plan: SimulationPlan) -> SimulationPlan:
    """Return ``plan`` with a registry-derived executable capability when needed.

    Legacy pin-cell plans do not use ``complex_model`` and remain unchanged.
    Newer assembly/core/TRISO plans may carry a stale or placeholder capability
    report from deterministic assembly.  The renderer registry is the canonical
    source for whether those plans can produce XML-ready OpenMC scripts.
    """
    if plan.model_spec is not None or plan.complex_model is None:
        return plan
    if (
        plan.capability_report.is_executable
        and plan.capability_report.supported_renderer != "none"
    ):
        return plan
    from openmc_agent.renderers import choose_renderer

    _renderer, capability = choose_renderer(plan)
    if capability.is_executable and capability.supported_renderer != "none":
        return plan.model_copy(update={"capability_report": capability})
    return plan


def run_geometry_debug(
    run_dir: str | Path,
    plan: SimulationPlan,
    *,
    max_particles: int = 2000,
    timeout: float = 120.0,
) -> ToolResult:
    """Run OpenMC geometry-debug mode in an isolated subdirectory.

    Executes ``openmc -g`` (geometry debugging) inside ``<run_dir>/geometry_debug/``
    so it never overwrites the smoke-test statepoint or plots. Uses a low-cost
    settings override (few particles, no inactive batches). Timeout is **not**
    treated as a geometry overlap — it produces ``runtime.openmc_timeout``.
    Source rejection remains the primary root cause when present.
    """
    path = Path(run_dir)

    # If geometry.xml is absent there is nothing to debug (export was faked,
    # skipped, or the model is not XML-based). Return ok so the pipeline can
    # proceed — geometry debug is not applicable in this state.
    if not (path / "geometry.xml").exists():
        return ToolResult(
            name="run_geometry_debug",
            ok=True,
            command=[],
            error="",
            issues=[],
        )

    gd_dir = path / "geometry_debug"
    gd_dir.mkdir(parents=True, exist_ok=True)
    # Rendered plots.xml may reference the relative ``plots/`` output path.
    # Geometry debug runs in an isolated cwd, so create it independently of
    # the optional plotting stage.
    (gd_dir / "plots").mkdir(exist_ok=True)

    # Copy the required XML artifacts into the geometry-debug subdirectory.
    for xml_name in ("materials.xml", "geometry.xml", "settings.xml", "tallies.xml", "plots.xml"):
        src = path / xml_name
        if src.exists():
            (gd_dir / xml_name).write_bytes(src.read_bytes())

    # Write a low-cost settings.xml that overrides the original to keep the
    # geometry-debug run cheap. OpenMC geometry-debug only samples particles
    # for overlap detection; it does not need a real transport run.
    settings_xml = gd_dir / "settings.xml"
    settings_xml.write_text(
        f"""<?xml version="1.0"?>
<settings>
  <run_mode>plot</run_mode>
  <particles>{max_particles}</particles>
  <batches>1</batches>
  <inactive>0</inactive>
  <source strength="1.0">
    <space type="box">
      <parameters>-1e99 -1e99 -1e99 1e99 1e99 1e99</parameters>
    </space>
  </source>
  <geometry_debug>true</geometry_debug>
</settings>
""",
        encoding="utf-8",
    )

    command = ["openmc", "-g"]
    try:
        result = subprocess.run(
            command,
            cwd=gd_dir,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ToolResult(
            name="run_geometry_debug",
            ok=False,
            command=command,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nTimed out after {timeout}s",
            artifacts=[str(gd_dir / name) for name in ("geometry_debug.log",) if (gd_dir / name).exists()],
            error=f"OpenMC geometry debug timed out after {timeout}s (not treated as overlap)",
            issues=[
                issue_from_catalog(
                    "runtime.openmc_timeout",
                    message=f"OpenMC geometry debug timed out after {timeout}s",
                    grep_patterns=["timeout", "timed out"],
                )
            ],
        )

    combined = f"{result.stdout}\n{result.stderr}"
    issues = parse_openmc_output(result.stdout, result.stderr).issues

    error = ""
    if result.returncode != 0:
        error = (result.stderr or result.stdout).strip()
    elif issues:
        error = "; ".join(i.message for i in issues if i.severity == "error")

    artifacts = [
        str(artifact)
        for artifact in sorted(gd_dir.glob("*"))
        if artifact.is_file()
    ]

    return ToolResult(
        name="run_geometry_debug",
        ok=result.returncode == 0 and not any(i.severity == "error" for i in issues),
        command=command,
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        artifacts=artifacts,
        error=error,
        issues=issues,
    )


def parse_openmc_output(stdout: str, stderr: str) -> ValidationReport:
    combined = f"{stdout}\n{stderr}"
    lowered = combined.lower()
    issues: list[ValidationIssue] = []
    warnings: list[str] = []

    if _has_cross_section_error(lowered):
        code = (
            "runtime.cross_sections_invalid"
            if re.search(r"(invalid|not present in cross_sections\.xml|not found in cross_sections)", lowered)
            else "runtime.cross_sections_missing"
        )
        issues.append(_runtime_issue(code, combined))
    if _has_source_rejection_failure(lowered):
        # Source initialization failure is the PRIMARY root cause; later
        # 'double free' / 'segmentation fault' / MPI abort lines are downstream
        # crash noise and must not override it.
        rejection_line = ""
        for needle in ("too few source sites", "minimum source rejection", "source rejection fraction"):
            rejection_line = _first_matching_line(combined, needle)
            if rejection_line:
                break
        issues.append(
            _runtime_issue(
                "runtime.openmc_source_rejection_failure",
                combined,
                message=(
                    "OpenMC rejected too many initial source sites. The source box "
                    "likely does not overlap the fissionable active-fuel region. "
                    f"Detail: {rejection_line.strip()}"
                ),
            )
        )
    if "could not be located in any cell" in lowered or "undefined" in lowered:
        issues.append(
            _runtime_issue(
                "runtime.lost_particle",
                combined,
                message="OpenMC reported an undefined region or geometry containment issue.",
            )
        )
    if "overlap" in lowered and "no overlap" not in lowered and "no overlaps" not in lowered:
        issues.append(_runtime_issue("runtime.geometry_overlap", combined))
    if "lost particle" in lowered or "lost particles" in lowered:
        issues.append(_runtime_issue("runtime.lost_particle", combined))
    if _has_material_missing_nuclide_data(lowered):
        issues.append(_runtime_issue("runtime.material_missing_nuclide_data", combined))
    if _has_geometry_load_error(lowered):
        issues.append(_runtime_issue("runtime.dagmc_or_geometry_load_failed", combined))
    if _has_process_crash(lowered):
        issues.append(_runtime_issue("runtime.openmc_process_crash", combined))
    if _has_timeout(lowered):
        issues.append(_runtime_issue("runtime.openmc_timeout", combined))
    if "traceback (most recent call last)" in lowered:
        issues.append(
            _runtime_issue(
                "runtime.openmc_unknown_error",
                combined,
                message="Python traceback occurred while preparing or running OpenMC.",
            )
        )
    if _has_unknown_runtime_error(stdout, stderr) and not issues:
        issues.append(_runtime_issue("runtime.openmc_unknown_error", combined))
    if "warning" in lowered:
        warnings.append(_first_matching_line(combined, "warning"))

    report = ValidationReport.from_issues(_dedupe_issues(issues))
    return report.model_copy(update={"warnings": [*report.warnings, *warnings]})


def _existing_xml_artifacts(path: Path) -> list[str]:
    names = ("materials.xml", "geometry.xml", "settings.xml", "tallies.xml", "plots.xml")
    return [str(path / name) for name in names if (path / name).exists()]


def _missing_required_xml_artifacts(path: Path) -> list[str]:
    names = ("materials.xml", "geometry.xml", "settings.xml")
    return [name for name in names if not (path / name).exists()]


def _geometry_lattice_reference_error(geometry_path: Path) -> str:
    return _format_geometry_reference_issues(_geometry_lattice_reference_issues(geometry_path))


def _geometry_lattice_reference_issues(geometry_path: Path) -> list[ValidationIssue]:
    try:
        root = ET.parse(geometry_path).getroot()
    except ET.ParseError as exc:
        return [
            issue_from_catalog(
                "export_xml.geometry_reference_unknown",
                message=f"geometry.xml is not valid XML: {exc}",
                grep_patterns=["geometry.xml", "ParseError", str(exc)],
                route_hint="manual_review",
            )
        ]

    exported_universe_numbers = {
        universe
        for cell in root.findall(".//cell")
        for universe in [cell.attrib.get("universe")]
        if universe is not None
    }
    exported_lattice_numbers = {
        lattice_id
        for lattice in root.findall(".//lattice")
        for lattice_id in [lattice.attrib.get("id")]
        if lattice_id is not None
    }
    exported_fill_numbers = exported_universe_numbers | exported_lattice_numbers
    missing_by_cell: dict[str, str] = {}
    for cell in root.findall(".//cell"):
        fill = cell.attrib.get("fill")
        cell_id = cell.attrib.get("id", "<unknown>")
        if fill is not None and fill not in exported_fill_numbers:
            missing_by_cell[cell_id] = fill

    missing_by_lattice: dict[str, list[str]] = {}
    for lattice in root.findall(".//lattice"):
        lattice_id = lattice.attrib.get("id", "<unknown>")
        referenced: set[str] = set()
        for universes in lattice.findall("universes"):
            if universes.text:
                referenced.update(re.findall(r"-?\d+", universes.text))
        missing = sorted(referenced - exported_universe_numbers, key=int)
        if missing:
            missing_by_lattice[lattice_id] = missing
    missing_outer_by_lattice: dict[str, str] = {}
    for lattice in root.findall(".//lattice"):
        lattice_id = lattice.attrib.get("id", "<unknown>")
        outer = lattice.attrib.get("outer")
        if outer is not None and outer not in exported_universe_numbers:
            missing_outer_by_lattice[lattice_id] = outer

    issues: list[ValidationIssue] = []
    fill_candidates = sorted(exported_fill_numbers, key=_numeric_string_sort_key)
    universe_candidates = sorted(exported_universe_numbers, key=_numeric_string_sort_key)
    for cell_id, fill in sorted(missing_by_cell.items(), key=_xml_id_sort_key):
        issues.append(
            _dangling_issue(
                "export_xml.dangling_cell_fill",
                message=f"cell {cell_id} fill {fill} is not an exported universe or lattice",
                schema_path=f"geometry.xml.cell[{cell_id}].fill",
                source_id=cell_id,
                missing_id=fill,
                candidates=fill_candidates,
            )
        )
    for lattice_id, missing in sorted(missing_by_lattice.items(), key=_xml_id_sort_key):
        for missing_id in missing:
            issues.append(
                _dangling_issue(
                    "export_xml.dangling_lattice_universe",
                    message=f"lattice {lattice_id} missing universe number {missing_id}",
                    schema_path=f"geometry.xml.lattice[{lattice_id}].universes",
                    source_id=lattice_id,
                    missing_id=missing_id,
                    candidates=universe_candidates,
                )
            )
    for lattice_id, missing_id in sorted(missing_outer_by_lattice.items(), key=_xml_id_sort_key):
        issues.append(
            _dangling_issue(
                "export_xml.dangling_lattice_outer_universe",
                message=f"lattice {lattice_id} outer universe {missing_id} is not exported",
                schema_path=f"geometry.xml.lattice[{lattice_id}].outer",
                source_id=lattice_id,
                missing_id=missing_id,
                candidates=universe_candidates,
            )
        )
    return issues


def _format_geometry_reference_issues(issues: list[ValidationIssue]) -> str:
    if not issues:
        return ""
    details = "; ".join(issue.message for issue in issues)
    return f"geometry.xml has dangling geometry references: {details}"


def _xml_id_sort_key(item: tuple[str, object]) -> tuple[int, int | str]:
    xml_id = item[0]
    return (0, int(xml_id)) if xml_id.isdigit() else (1, xml_id)


def _numeric_string_sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if value.isdigit() else (1, value)


def _plot_artifacts(path: Path) -> list[str]:
    artifacts = [*path.glob("*.png"), *path.glob("*.ppm")]
    return [str(artifact) for artifact in sorted(artifacts)]


def _statepoint_artifacts(path: Path) -> list[str]:
    return [str(artifact) for artifact in sorted(path.glob("statepoint.*.h5"))]


def _first_matching_line(text: str, pattern: str) -> str:
    lowered_pattern = pattern.lower()
    for line in text.splitlines():
        if lowered_pattern in line.lower():
            return line.strip()
    return ""


def _has_cross_section_error(lowered_text: str) -> bool:
    patterns = (
        r"no\s+cross_sections\.xml\s+was\s+specified",
        r"cross_sections\.xml\s+(?:was\s+)?not\s+(?:specified|found)",
        r"could\s+not\s+(?:find|open|read).{0,120}cross",
        r"unable\s+to\s+(?:find|open|read).{0,120}cross",
        r"failed\s+to\s+(?:find|open|read).{0,120}cross",
        r"no\s+cross\s+section\s+data",
        r"cross\s+section\s+data\s+.*(?:missing|not\s+found|unavailable)",
        r"not\s+present\s+in\s+cross_sections\.xml",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _has_material_missing_nuclide_data(lowered_text: str) -> bool:
    patterns = (
        r"nuclide .{0,80} not (?:found|present|available)",
        r"not present in cross_sections\.xml",
        r"could not find .{0,80}\.h5",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _has_source_rejection_failure(lowered_text: str) -> bool:
    """Detect OpenMC initial-source rejection ('too few source sites').

    This is the primary root cause; downstream 'double free' / 'segmentation
    fault' / 'MPI abort' lines after a source rejection are crash noise.
    """
    patterns = (
        r"too few source sites satisfied",
        r"minimum source rejection fraction",
        r"source[_ ]rejection[_ ]fraction",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)



def _has_geometry_load_error(lowered_text: str) -> bool:
    patterns = (
        r"dagmc",
        r"failed to load.{0,80}geometry",
        r"could not load.{0,80}geometry",
        r"error reading.{0,80}geometry\.xml",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _has_process_crash(lowered_text: str) -> bool:
    patterns = (
        r"segmentation fault",
        r"\bsegfault\b",
        r"mpi_abort",
        r"signal \d+",
        r"core dumped",
        r"double free",
        r"abort\b",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _has_timeout(lowered_text: str) -> bool:
    patterns = (
        r"\btimed? ?out\b",
        r"timeoutexpired",
        r"subprocess\.timeout",
        r"deadline exceeded",
    )
    return any(re.search(pattern, lowered_text) for pattern in patterns)


def _has_unknown_runtime_error(stdout: str, stderr: str) -> bool:
    text = f"{stdout}\n{stderr}".lower()
    if not (stdout.strip() or stderr.strip()):
        return False
    return bool(stderr.strip()) and any(token in text for token in ("error", "fatal", "exception", "traceback"))


def _runtime_issue(
    code: str,
    raw_output: str,
    *,
    message: str | None = None,
) -> ValidationIssue:
    summary = _first_error_summary(raw_output)
    grep_patterns = _runtime_grep_patterns(raw_output)
    final_message = message or None
    if summary:
        base = final_message or issue_from_catalog(code).message
        final_message = f"{base} Raw OpenMC summary: {summary}"
    overrides: dict[str, Any] = {"grep_patterns": grep_patterns}
    if final_message is not None:
        overrides["message"] = final_message
    return issue_from_catalog(code, **overrides)


def _first_error_summary(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and any(token in stripped.lower() for token in ("error", "fatal", "traceback", "lost particle", "overlap")):
            return stripped[:240]
    return ""


def _runtime_grep_patterns(text: str) -> list[str]:
    patterns: list[str] = []
    for regex in (
        r"\b[A-Z][a-z]?\d{1,3}\b",
        r"\bcross_sections\.xml\b",
        r"\bOPENMC_CROSS_SECTIONS\b",
        r"\bmaterial\s+['\"]?([\w.-]+)",
        r"\blattice\s+['\"]?([\w.-]+)",
        r"\buniverse\s+['\"]?([\w.-]+)",
        r"\bcell\s+['\"]?([\w.-]+)",
    ):
        for match in re.finditer(regex, text, flags=re.IGNORECASE):
            value = match.group(1) if match.groups() else match.group(0)
            if value:
                patterns.append(value)
    return list(dict.fromkeys(patterns))


def _dedupe_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _dangling_issue(
    code: str,
    *,
    message: str,
    schema_path: str,
    source_id: str,
    missing_id: str,
    candidates: list[str],
) -> ValidationIssue:
    close = _candidate_ids(missing_id, candidates)
    route_hint = "auto_repair" if len(close) == 1 else ("reflect_plan" if close else "manual_review")
    repair_hints: list[RepairHint] = []
    if len(close) == 1:
        replacement = close[0]
        repair_hints.append(
            RepairHint(
                action="edit_field",
                message=f"Replace missing id {missing_id!r} with the unique candidate {replacement!r}.",
                target_path=schema_path,
                example_patch={"replace": {missing_id: replacement}},
            )
        )
    elif close:
        repair_hints.append(
            RepairHint(
                action="edit_field",
                message=f"Choose one valid referenced id for {missing_id!r}: {close}.",
                target_path=schema_path,
            )
        )
    return issue_from_catalog(
        code,
        message=f"{message}; source_id={source_id}; missing_id={missing_id}; candidates={close or candidates}",
        schema_path=schema_path,
        grep_patterns=[source_id, missing_id, *close],
        repair_hints=repair_hints,
        route_hint=route_hint,
    )


def _candidate_ids(missing_id: str, candidates: list[str]) -> list[str]:
    if not candidates:
        return []
    if missing_id in candidates:
        return [missing_id]
    close = get_close_matches(missing_id, candidates, n=3, cutoff=0.75)
    if close:
        return close
    if missing_id.isdigit():
        value = int(missing_id)
        near = [candidate for candidate in candidates if candidate.isdigit() and abs(int(candidate) - value) <= 1]
        if len(near) == 1:
            return near
    return []
