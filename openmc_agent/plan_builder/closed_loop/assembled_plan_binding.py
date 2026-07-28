"""Assembled Plan binding view: object graph, root reachability, renderer matrix.

Builds a deterministic, reactor-neutral view of the assembled SimulationPlan:
the object graph (materials, cells, universes, lattices, assemblies, core,
axial layers, overlays, plots, execution check), root selection, reachability
from selected roots, renderer capability matrix, static source feasibility,
and plot/execution-check coherence.
"""

from __future__ import annotations

from typing import Any

from .fingerprints import compute_candidate_hash, compute_evidence_pack_hash
from .models import (
    AssembledObjectGraph,
    AssembledPlanBindingView,
    ExecutionCheckRecord,
    ObjectGraphEdge,
    ObjectGraphNode,
    PlotCoverageRecord,
    ReachabilityRecord,
    RendererCapabilityRow,
    StaticSourceFeasibility,
)


def _valid(state: Any, patch_type: str) -> Any | None:
    matches = [item for item in state.patches.values() if item.patch_type == patch_type and item.status == "valid"]
    return matches[0] if matches else None


def _accepted_hash(state: Any, stage_key: str) -> str:
    stage = state.plan_loop_stages.get(stage_key)
    if stage is None:
        return ""
    return str(stage.metadata.get("accepted_input_hash", ""))


def _build_object_graph(plan: Any) -> AssembledObjectGraph:
    """Extract a typed object graph from the assembled SimulationPlan."""
    model = getattr(plan, "complex_model", None)
    nodes: list[ObjectGraphNode] = []
    edges: list[ObjectGraphEdge] = []
    seen_ids: dict[str, str] = {}

    def _add(node_id: str, node_kind: str, path: str = "") -> None:
        if not node_id:
            return
        if node_id in seen_ids and seen_ids[node_id] != node_kind:
            return
        seen_ids[node_id] = node_kind
        nodes.append(ObjectGraphNode(node_id=str(node_id), node_kind=node_kind, source_path=path))

    def _edge(src: str, tgt: str, kind: str, path: str = "") -> None:
        if src and tgt:
            edges.append(ObjectGraphEdge(source_id=str(src), target_id=str(tgt), edge_kind=kind, source_path=path))

    if model is not None:
        for m in getattr(model, "materials", []) or []:
            _add(m.id, "material", f"/materials/{m.id}")
        for s in getattr(model, "surfaces", []) or []:
            _add(s.id, "surface", f"/surfaces/{s.id}")
        for r in getattr(model, "regions", []) or []:
            _add(r.id, "region", f"/regions/{r.id}")
            for sid in getattr(r, "surface_ids", []) or []:
                _edge(r.id, sid, "region_surface", f"/regions/{r.id}")
        for c in getattr(model, "cells", []) or []:
            _add(c.id, "cell", f"/cells/{c.id}")
            rid = getattr(c, "region_id", None)
            if rid:
                _edge(c.id, rid, "cell_region", f"/cells/{c.id}")
            fill_id = getattr(c, "fill_id", None)
            fill_type = getattr(c, "fill_type", "")
            if fill_id and fill_type:
                _edge(c.id, fill_id, f"cell_{fill_type}", f"/cells/{c.id}")
        for u in getattr(model, "universes", []) or []:
            _add(u.id, "universe", f"/universes/{u.id}")
            for cid in getattr(u, "cell_ids", []) or []:
                _edge(u.id, cid, "universe_cell", f"/universes/{u.id}")
        for lat in getattr(model, "lattices", []) or []:
            _add(lat.id, "lattice", f"/lattices/{lat.id}")
            pattern = getattr(lat, "universe_pattern", None) or []
            for row in pattern:
                if isinstance(row, list):
                    for uid in row:
                        if uid:
                            _edge(lat.id, uid, "lattice_universe", f"/lattices/{lat.id}")
                elif isinstance(row, str) and row:
                    _edge(lat.id, row, "lattice_universe", f"/lattices/{lat.id}")
        for asm in getattr(model, "assemblies", []) or []:
            _add(asm.id, "assembly", f"/assemblies/{asm.id}")
            if getattr(asm, "lattice_id", None):
                _edge(asm.id, asm.lattice_id, "assembly_lattice", f"/assemblies/{asm.id}")
        core = getattr(model, "core", None)
        if core is not None:
            cid = getattr(core, "id", "core")
            _add(cid, "core", "/core")
            if getattr(core, "lattice_id", None):
                _edge(cid, core.lattice_id, "core_lattice", "/core")
            for aid in getattr(core, "assembly_ids", []) or []:
                _edge(cid, aid, "core_assembly", "/core")
        for layer in getattr(core, "axial_layers", []) or [] if core else []:
            _add(getattr(layer, "layer_id", ""), "axial_layer", f"/axial_layers/{getattr(layer, 'layer_id', '')}")
        for overlay in getattr(core, "axial_overlays", []) or [] if core else []:
            _add(getattr(overlay, "overlay_id", ""), "axial_overlay", f"/axial_overlays/{getattr(overlay, 'overlay_id', '')}")
    # Detect duplicate IDs across kinds.
    id_kinds: dict[str, set[str]] = {}
    for node in nodes:
        id_kinds.setdefault(node.node_id, set()).add(node.node_kind)
    duplicates = sorted([nid for nid, kinds in id_kinds.items() if len(kinds) > 1])
    return AssembledObjectGraph(nodes=nodes, edges=edges, duplicate_ids=duplicates)


def _select_roots(model: Any) -> tuple[list[str], list[str]]:
    """Reactor-neutral root selection: prefer core, then assembly, then pin-cell."""
    candidates: list[str] = []
    selected: list[str] = []
    model_kind = getattr(model, "kind", "") if model else ""
    if model is not None:
        core = getattr(model, "core", None)
        if core is not None and getattr(core, "lattice_id", None):
            cid = getattr(core, "id", "core")
            selected.append(cid)
            candidates.append(cid)
        assemblies = getattr(model, "assemblies", []) or []
        if assemblies and not selected:
            if len(assemblies) == 1 and getattr(assemblies[0], "lattice_id", None):
                selected.append(assemblies[0].id)
            candidates.extend(a.id for a in assemblies if getattr(a, "lattice_id", None))
    return candidates, selected


def _compute_reachability(graph: AssembledObjectGraph, roots: list[str]) -> list[ReachabilityRecord]:
    """Deterministic BFS from selected roots through typed edges."""
    adj: dict[str, list[str]] = {}
    for edge in graph.edges:
        adj.setdefault(edge.source_id, []).append(edge.target_id)
    visited: set[str] = set()
    queue: list[str] = list(roots)
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        queue.extend(adj.get(current, []))
    records: list[ReachabilityRecord] = []
    for node in graph.nodes:
        reachable = node.node_id in visited
        records.append(ReachabilityRecord(
            object_id=node.node_id, object_kind=node.node_kind,
            reachable=reachable,
            required=False,
        ))
    return records


def _compute_cycles(graph: AssembledObjectGraph) -> list[list[str]]:
    """Detect cycles via DFS (returns at most 10 cycles for safety)."""
    adj: dict[str, list[str]] = {}
    for edge in graph.edges:
        adj.setdefault(edge.source_id, []).append(edge.target_id)
    cycles: list[list[str]] = []
    visited: set[str] = set()
    stack: list[str] = []

    def _dfs(node: str) -> None:
        if len(cycles) >= 10:
            return
        if node in stack:
            idx = stack.index(node)
            cycles.append(stack[idx:] + [node])
            return
        if node in visited:
            return
        stack.append(node)
        for nbr in adj.get(node, []):
            _dfs(nbr)
        stack.pop()
        visited.add(node)

    for node in sorted(adj):
        _dfs(node)
    return cycles


def _build_renderer_matrix(plan: Any) -> tuple[list[RendererCapabilityRow], str, str, dict[str, Any]]:
    """Call can_render on each registered renderer; return matrix + selected renderer."""
    try:
        from openmc_agent.renderers.registry import RENDERERS, choose_renderer
    except Exception:
        return [], "", "none", {}
    rows: list[RendererCapabilityRow] = []
    selected_name = ""
    selected_renderability = "none"
    computed_report: dict[str, Any] = {}
    for renderer in RENDERERS:
        try:
            report = renderer.can_render(plan)
        except Exception:
            report = None
        if report is None:
            continue
        rank = 0
        try:
            rank = renderer.renderability_rank(report.renderability)
        except Exception:
            pass
        rows.append(RendererCapabilityRow(
            renderer_name=renderer.name,
            supported_kinds=list(renderer.supported_kinds),
            renderability=report.renderability,
            is_executable=report.is_executable,
            executable_subsystems=list(report.executable_subsystems),
            unsupported_subsystems=list(report.unsupported_subsystems),
            reasons=list(report.reasons),
            warnings=list(report.warnings),
            required_human_confirmations=list(report.required_human_confirmations),
            structured_issue_codes=[str(i.code) for i in report.issues],
            rank=rank,
        ))
    try:
        chosen, chosen_report = choose_renderer(plan)
        if chosen is not None:
            selected_name = chosen.name
            selected_renderability = chosen_report.renderability
            computed_report = chosen_report.model_dump(mode="json")
    except Exception:
        pass
    return rows, selected_name, selected_renderability, computed_report


def _assess_source_feasibility(plan: Any, reachability: list[ReachabilityRecord]) -> StaticSourceFeasibility:
    """Static source feasibility check (no OpenMC runtime claims)."""
    model = getattr(plan, "complex_model", None)
    settings = getattr(model, "settings", None) if model else None
    strategy = getattr(settings, "source_strategy", "unknown") if settings else "unknown"
    issue_codes: list[str] = []
    feasible = True
    bounds_valid = False
    has_reachable_fissionable = False
    intersection_nonzero = False

    if strategy == "unknown" or not strategy:
        issue_codes.append("assembled.source_strategy_unknown")
        feasible = False
    else:
        bounds = getattr(settings, "manual_source_bounds_cm", None) if settings else None
        if strategy == "manual":
            if bounds and len(bounds) >= 6:
                if all(b is not None for b in bounds[:6]):
                    bounds_valid = bounds[0] < bounds[1] and bounds[2] < bounds[3] and bounds[4] < bounds[5]
                if not bounds_valid:
                    issue_codes.append("assembled.source_bounds_invalid")
                    feasible = False
            else:
                issue_codes.append("assembled.source_bounds_missing")
                feasible = False
        elif strategy in ("active_fuel_box", "assembly_box"):
            bounds_valid = True

    # Check for reachable fissionable material.
    for rec in reachability:
        if rec.object_kind == "material" and rec.reachable:
            has_reachable_fissionable = True
            break
    if strategy in ("active_fuel_box", "manual") and not has_reachable_fissionable:
        issue_codes.append("assembled.source_no_reachable_fissionable_material")
        if strategy == "active_fuel_box":
            feasible = False
    intersection_nonzero = bounds_valid

    return StaticSourceFeasibility(
        source_strategy=str(strategy),
        feasible=feasible,
        bounds_valid=bounds_valid,
        has_reachable_fissionable=has_reachable_fissionable,
        intersection_nonzero=intersection_nonzero,
        issue_codes=issue_codes,
    )


def _assess_plots(plan: Any) -> list[PlotCoverageRecord]:
    records: list[PlotCoverageRecord] = []
    for plot in getattr(plan, "plot_specs", []) or []:
        width = getattr(plot, "width_cm", None)
        positive = True
        within = True
        codes: list[str] = []
        if width is not None:
            if isinstance(width, (list, tuple)) and any(w is not None and w <= 0 for w in width):
                positive = False
                codes.append("assembled.plot_invalid_extent")
        records.append(PlotCoverageRecord(
            plot_id=str(getattr(plot, "filename", "")),
            plot_type=str(getattr(plot, "kind", "")),
            within_domain=within,
            positive_extent=positive,
            issue_codes=codes,
        ))
    return records


def _assess_execution_check(plan: Any, renderability: str) -> ExecutionCheckRecord:
    ec = getattr(plan, "execution_check", None)
    enabled = bool(getattr(ec, "enabled", False)) if ec else False
    inactive_lt_batches = True
    within_smoke = True
    consistent = True
    codes: list[str] = []
    settings = getattr(ec, "settings", {}) if ec else {}
    if isinstance(settings, dict):
        batches = settings.get("batches")
        inactive = settings.get("inactive")
        particles = settings.get("particles")
    else:
        batches = getattr(settings, "batches", None)
        inactive = getattr(settings, "inactive", None)
        particles = getattr(settings, "particles", None)
    if enabled:
        if inactive is not None and batches is not None and inactive >= batches:
            inactive_lt_batches = False
            codes.append("assembled.execution_check_invalid")
        if particles is not None and particles > 1000:
            within_smoke = False
            codes.append("assembled.smoke_settings_too_large")
        if batches is not None and batches > 20:
            within_smoke = False
            codes.append("assembled.smoke_settings_too_large")
    if renderability == "runnable" and not enabled:
        consistent = False
        codes.append("assembled.execution_check_disabled_for_runnable")
    return ExecutionCheckRecord(
        enabled=enabled, inactive_lt_batches=inactive_lt_batches,
        within_smoke_limits=within_smoke,
        consistent_with_renderability=consistent,
        issue_codes=codes,
    )


def build_assembled_plan_binding_view(*, state: Any, plan: Any) -> AssembledPlanBindingView:
    """Construct the AssembledPlanBindingView from the assembled SimulationPlan."""
    model = getattr(plan, "complex_model", None)
    model_kind = getattr(model, "kind", "") if model else (getattr(getattr(plan, "model_spec", None), "kind", "pin_cell") if getattr(plan, "model_spec", None) else "")

    object_graph = _build_object_graph(plan)
    root_candidates, selected_roots = _select_roots(model)
    reachability = _compute_reachability(object_graph, selected_roots)
    cycles = _compute_cycles(object_graph)
    renderer_rows, selected_renderer, selected_renderability, computed_report = _build_renderer_matrix(plan)
    source_feasibility = _assess_source_feasibility(plan, reachability)
    plot_records = _assess_plots(plan)
    exec_check = _assess_execution_check(plan, selected_renderability)

    # Categorize unreachable.
    node_ids = {n.node_id for n in object_graph.nodes}
    reachable_ids = {r.object_id for r in reachability if r.reachable}
    unreachable_ids = node_ids - reachable_ids

    patch_hashes: dict[str, str] = {}
    for ptype in ("facts", "materials", "universes", "axial_layers", "axial_overlays"):
        env = _valid(state, ptype)
        if env is not None:
            patch_hashes[ptype] = compute_candidate_hash(target_patch_type=ptype, candidate_patch=env.content)

    accepted_gate_hashes = {
        "facts": _accepted_hash(state, "plan_gate_facts"),
        "material_universe": _accepted_hash(state, "plan_gate_material_universe"),
        "placement": _accepted_hash(state, "plan_gate_placement"),
        "axial_geometry": _accepted_hash(state, "plan_gate_axial_geometry"),
    }

    plan_hash = compute_evidence_pack_hash({"plan": plan.model_dump(mode="json") if hasattr(plan, "model_dump") else {}})

    _scope_obj = getattr(state, "resolved_planning_scope", None)
    if _scope_obj is None:
        planning_scope = "single_assembly"
    elif isinstance(_scope_obj, str):
        planning_scope = _scope_obj
    else:
        planning_scope = str(
            getattr(_scope_obj, "value", None)
            or getattr(_scope_obj, "model_scope", None)
            or getattr(_scope_obj, "status", None)
            or "unknown"
        )

    return AssembledPlanBindingView(
        assembled_plan_hash=plan_hash,
        assembly_input_hash=plan_hash,
        canonical_task_plan_hash=getattr(state, "canonical_task_plan", None).plan_hash if getattr(state, "canonical_task_plan", None) else "",
        patch_hashes=patch_hashes,
        accepted_gate_hashes=accepted_gate_hashes,
        planning_scope=planning_scope,
        model_kind=model_kind,
        object_graph=object_graph,
        root_candidates=root_candidates,
        selected_roots=selected_roots,
        reference_edges=object_graph.edges,
        reachability_records=reachability,
        unreachable_required_ids=sorted(unreachable_ids),
        cycles=cycles,
        renderer_capability_matrix=renderer_rows,
        selected_renderer=selected_renderer,
        selected_renderability=selected_renderability,
        computed_capability_report=computed_report,
        static_source_feasibility=source_feasibility,
        plot_coverage_records=plot_records,
        execution_check_record=exec_check,
    )


__all__ = ["build_assembled_plan_binding_view"]
