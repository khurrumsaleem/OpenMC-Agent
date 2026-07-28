"""Tests for AssembledPlanBindingView construction."""

from tests._assembled_plan_fixtures import state_with_assembled_plan, make_assembled_plan
from openmc_agent.plan_builder.closed_loop.assembled_plan_binding import build_assembled_plan_binding_view
from openmc_agent.plan_builder.planning_scope import ResolvedPlanningScope


def test_binding_view_builds_object_graph():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.object_graph.nodes) > 0
    node_kinds = {n.node_kind for n in view.object_graph.nodes}
    assert "material" in node_kinds
    assert "cell" in node_kinds
    assert "universe" in node_kinds
    assert "lattice" in node_kinds


def test_binding_view_extracts_model_kind():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert view.model_kind == "assembly"


def test_binding_view_extracts_resolved_planning_scope_value():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    state.resolved_planning_scope = ResolvedPlanningScope(
        value="multi_assembly_core",
        status="resolved",
    )
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert view.planning_scope == "multi_assembly_core"


def test_binding_view_selects_root():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.selected_roots) >= 1


def test_binding_view_computes_reachability():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.reachability_records) > 0
    reachable = [r for r in view.reachability_records if r.reachable]
    assert len(reachable) > 0


def test_binding_view_does_not_mark_optional_orphans_as_required_unreachable():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert view.unreachable_required_ids == []
    assert all(not r.required for r in view.reachability_records)


def test_binding_view_builds_renderer_matrix():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.renderer_capability_matrix) > 0


def test_binding_view_assesses_source_feasibility():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert view.static_source_feasibility.source_strategy == "assembly_box"


def test_binding_view_assesses_plots():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.plot_coverage_records) >= 1


def test_binding_view_detects_edges():
    plan = make_assembled_plan()
    state = state_with_assembled_plan(plan=plan)
    view = build_assembled_plan_binding_view(state=state, plan=plan)
    assert len(view.object_graph.edges) > 0
