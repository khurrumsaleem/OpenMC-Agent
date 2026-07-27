"""Tests for Axial Geometry issue owner policy routing."""

from openmc_agent.plan_builder.closed_loop.axial_geometry_issue_policy import axial_geometry_issue_owner
from openmc_agent.plan_builder.closed_loop.models import PlanGateId


def test_domain_missing_routes_to_facts():
    policy = axial_geometry_issue_owner("axial.domain_missing")
    assert policy is not None
    assert "facts" in policy.owner_patch_types
    assert PlanGateId.AXIAL_GEOMETRY in policy.gates_to_invalidate


def test_overlay_density_routes_to_axial_overlays():
    policy = axial_geometry_issue_owner("axial.overlay_density_required")
    assert policy is not None
    assert "axial_overlays" in policy.owner_patch_types


def test_universe_reference_routes_to_universes():
    policy = axial_geometry_issue_owner("axial.universe_reference_missing")
    assert policy is not None
    assert "universes" in policy.owner_patch_types


def test_layer_issue_routes_to_axial_layers():
    policy = axial_geometry_issue_owner("axial.layer_overlap")
    assert policy is not None
    assert "axial_layers" in policy.owner_patch_types


def test_overlay_issue_routes_to_axial_overlays():
    policy = axial_geometry_issue_owner("axial.overlay_interval_invalid")
    assert policy is not None
    assert "axial_overlays" in policy.owner_patch_types


def test_base_path_routes_to_profiles():
    policy = axial_geometry_issue_owner("axial.base_path_segment_gap")
    assert policy is not None
    assert "base_path_axial_profiles" in policy.owner_patch_types


def test_localized_insert_profile_routes_to_placement():
    policy = axial_geometry_issue_owner("axial.localized_insert_profile_missing")
    assert policy is not None
    assert "localized_insert_profiles" in policy.owner_patch_types


def test_unknown_code_returns_none():
    policy = axial_geometry_issue_owner("unknown.code")
    assert policy is None
