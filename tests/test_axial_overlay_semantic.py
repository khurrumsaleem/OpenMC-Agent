"""Tests for axial overlay semantic closure (P2-FULLCORE-2D-B-AXIAL-OVERLAY).

Covers:
- through_path_preserved deterministic derivation from None (audit recorded)
- through_path_preserved explicit False → mode_semantic_contradiction
- skeleton mode does NOT force derivation
- compact retry prompt includes previous parsed patch
- retry changes only through_path_preserved → pass
- retry changes z range / material / total_mass_g → unexpected_semantic_drift
- retry preserves overlay count and order

At least half the tests use reactor-neutral synthetic examples.
"""

from __future__ import annotations

import json

import pytest

from openmc_agent.plan_builder.patches import (
    AxialOverlayPatchItem,
    AxialOverlaysPatch,
)
from openmc_agent.plan_builder.patch_generator import (
    FakePatchLLM,
    PatchGenerationContext,
    _detect_retry_drift,
    generate_patch,
)
from openmc_agent.plan_builder.patch_prompts import (
    build_retry_prompt,
    _build_compact_overlay_retry,
)
from openmc_agent.plan_builder.validators import validate_patch


# ---------------------------------------------------------------------------
# Helpers (reactor-neutral)
# ---------------------------------------------------------------------------

def _codes(result) -> list[str]:
    return [i.code for i in result.issues]


def _make_overlay_json(
    overlay_id: str = "grid_a",
    geometry_mode: str = "mass_conserving_outer_frame",
    through_path_preserved=...,
    **extra,
) -> dict:
    """Build a single overlay dict (reactor-neutral)."""
    ov: dict = {
        "overlay_id": overlay_id,
        "overlay_kind": "spacer_grid",
        "z_min_cm": 10.0,
        "z_max_cm": 12.0,
        "target_lattice_id": "assembly_lattice",
        "material_id": "grid_mat",
        "geometry_mode": geometry_mode,
        "total_mass_g": 500.0,
    }
    if through_path_preserved is not ...:
        ov["through_path_preserved"] = through_path_preserved
    ov.update(extra)
    return ov


def _make_overlays_patch_json(overlays: list[dict]) -> str:
    return json.dumps({"patch_type": "axial_overlays", "overlays": overlays})


# ---------------------------------------------------------------------------
# A. mass_conserving_outer_frame + true → pass
# ---------------------------------------------------------------------------

def test_A_mass_conserving_with_true_passes() -> None:
    """Reactor-neutral: grid with through_path_preserved=true passes."""
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="sg1", overlay_kind="spacer_grid",
            z_min_cm=10, z_max_cm=12,
            target_lattice_id="lat1", material_id="m1",
            geometry_mode="mass_conserving_outer_frame",
            total_mass_g=300, through_path_preserved=True,
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is True


# ---------------------------------------------------------------------------
# B. mass_conserving_outer_frame + missing field → derive true + audit
# ---------------------------------------------------------------------------

def test_B_mass_conserving_missing_derives_true() -> None:
    """Field absent → model_validator derives True."""
    ov = AxialOverlayPatchItem(
        overlay_id="sg1", overlay_kind="spacer_grid",
        geometry_mode="mass_conserving_outer_frame",
    )
    assert ov.through_path_preserved is True


def test_B2_semantic_normalization_detected_in_generate() -> None:
    """generate_patch records the derivation in semantic_normalizations."""
    raw = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=None),
    ])
    fake = FakePatchLLM([raw])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor with spacer grids",
        llm_client=fake, max_attempts=1,
    )
    assert result.ok is True
    norms = result.attempts[0].semantic_normalizations
    assert len(norms) == 1
    assert norms[0]["field"] == "through_path_preserved"
    assert norms[0]["original_value"] is None
    assert norms[0]["derived_value"] is True
    assert norms[0]["geometry_mode"] == "mass_conserving_outer_frame"


# ---------------------------------------------------------------------------
# C. mass_conserving_outer_frame + null → derive true + audit
# ---------------------------------------------------------------------------

def test_C_mass_conserving_null_derives_true() -> None:
    """Explicit null → derived to True (same as missing)."""
    ov = AxialOverlayPatchItem(
        overlay_id="sg1", overlay_kind="spacer_grid",
        geometry_mode="mass_conserving_outer_frame",
        through_path_preserved=None,
    )
    assert ov.through_path_preserved is True


# ---------------------------------------------------------------------------
# D. mass_conserving_outer_frame + false → contradiction, no silent rewrite
# ---------------------------------------------------------------------------

def test_D_mass_conserving_false_contradiction() -> None:
    """Explicit False is NOT silently rewritten — validator catches it."""
    ov = AxialOverlayPatchItem(
        overlay_id="sg1", overlay_kind="spacer_grid",
        geometry_mode="mass_conserving_outer_frame",
        through_path_preserved=False,
    )
    # Model validator should NOT touch explicit False.
    assert ov.through_path_preserved is False

    patch = AxialOverlaysPatch(overlays=[ov])
    result = validate_patch(patch)
    codes = _codes(result)
    assert "patch.axial_overlays.mode_semantic_contradiction" in codes
    assert result.ok is False


# ---------------------------------------------------------------------------
# E. homogenized_open_region + false → contradiction
# ---------------------------------------------------------------------------

def test_E_homogenized_false_contradiction() -> None:
    """Explicit False for homogenized_open_region is also a contradiction."""
    ov = AxialOverlayPatchItem(
        overlay_id="sg1", overlay_kind="spacer_grid",
        geometry_mode="homogenized_open_region",
        through_path_preserved=False,
    )
    assert ov.through_path_preserved is False
    patch = AxialOverlaysPatch(overlays=[ov])
    result = validate_patch(patch)
    codes = _codes(result)
    assert "patch.axial_overlays.mode_semantic_contradiction" in codes


# ---------------------------------------------------------------------------
# F. skeleton + missing → no forced true
# ---------------------------------------------------------------------------

def test_F_skeleton_missing_no_derive() -> None:
    """Skeleton mode does NOT derive through_path_preserved."""
    ov = AxialOverlayPatchItem(
        overlay_id="sg1", overlay_kind="spacer_grid",
        geometry_mode="skeleton",
    )
    assert ov.through_path_preserved is None


# ---------------------------------------------------------------------------
# G. retry prompt includes previous parsed patch
# ---------------------------------------------------------------------------

def test_G_retry_prompt_includes_previous_patch() -> None:
    """Compact retry prompt contains the previous JSON."""
    previous = {"patch_type": "axial_overlays", "overlays": [
        _make_overlay_json("sg1", through_path_preserved=False),
    ]}
    issues = [{
        "code": "patch.axial_overlays.mode_semantic_contradiction",
        "severity": "error",
        "message": "contradiction",
        "path": "overlays[sg1].through_path_preserved",
    }]
    prompt = build_retry_prompt(
        "axial_overlays", "req", None, issues, 1,
        previous_patch=previous,
    )
    assert '"overlay_id": "sg1"' in prompt
    assert "through_path_preserved" in prompt
    assert "ALLOWED CHANGE" in prompt
    assert "LOCKED" in prompt


# ---------------------------------------------------------------------------
# H. retry prompt lists exact overlay IDs
# ---------------------------------------------------------------------------

def test_H_retry_prompt_lists_overlay_ids() -> None:
    """Compact retry prompt explicitly names the failing overlays."""
    previous = {"patch_type": "axial_overlays", "overlays": [
        _make_overlay_json("top_grid", through_path_preserved=False),
        _make_overlay_json("bot_grid", through_path_preserved=False),
    ]}
    issues = [
        {"code": "patch.axial_overlays.mode_semantic_contradiction",
         "severity": "error", "message": "x",
         "path": "overlays[top_grid].through_path_preserved"},
        {"code": "patch.axial_overlays.mode_semantic_contradiction",
         "severity": "error", "message": "x",
         "path": "overlays[bot_grid].through_path_preserved"},
    ]
    prompt = _build_compact_overlay_retry(issues, 1, previous)
    assert "top_grid" in prompt
    assert "bot_grid" in prompt


# ---------------------------------------------------------------------------
# I. retry changes only through_path_preserved → pass
# ---------------------------------------------------------------------------

def test_I_retry_changes_only_through_path_passes() -> None:
    """Reactor-neutral: first attempt has False, retry fixes only that field."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False),
    ])
    good = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=True),
    ])
    fake = FakePatchLLM([bad, good])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is True
    assert len(result.attempts) == 2
    # No drift on the retry attempt
    assert result.attempts[1].retry_drift == []


def test_I2_retry_schema_default_false_is_not_drift() -> None:
    """Explicit default booleans emitted on retry are not semantic changes."""

    previous = {"patch_type": "axial_overlays", "overlays": [{
        "overlay_id": "sg1",
        "overlay_kind": "spacer_grid",
        "z_min_cm": 0.0,
        "z_max_cm": 1.0,
        "target_lattice_id": "lat",
        "material_id": "missing_material",
        "geometry_mode": "skeleton",
    }]}
    retry = {"patch_type": "axial_overlays", "overlays": [{
        **previous["overlays"][0],
        "material_id": "resolved_material",
        "requires_human_confirmation": False,
    }]}

    assert _detect_retry_drift(
        "axial_overlays",
        previous,
        retry,
        ["patch.axial_overlays.material_missing"],
    ) == []


# ---------------------------------------------------------------------------
# J. retry changes z range → unexpected_semantic_drift
# ---------------------------------------------------------------------------

def test_J_retry_changes_z_range_drift() -> None:
    """Retry that changes z_min_cm produces drift error."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False,
                           z_min_cm=10.0, z_max_cm=12.0),
    ])
    drifted = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=True,
                           z_min_cm=15.0, z_max_cm=12.0),
    ])
    fake = FakePatchLLM([bad, drifted])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    drift_codes = [
        i["code"] for i in result.attempts[1].issues
        if "drift" in i.get("code", "")
    ]
    assert any("unexpected_semantic_drift" in c for c in drift_codes)


# ---------------------------------------------------------------------------
# K. retry changes material → unexpected_semantic_drift
# ---------------------------------------------------------------------------

def test_K_retry_changes_material_drift() -> None:
    """Retry that changes material_id produces drift error."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False,
                           material_id="inconel_718"),
    ])
    drifted = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=True,
                           material_id="zircaloy_4"),
    ])
    fake = FakePatchLLM([bad, drifted])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    drift_issues = result.attempts[1].retry_drift
    assert any("material_id" in d["json_path"] for d in drift_issues)


# ---------------------------------------------------------------------------
# L. retry changes total_mass_g → unexpected_semantic_drift
# ---------------------------------------------------------------------------

def test_L_retry_changes_total_mass_drift() -> None:
    """Retry that changes total_mass_g produces drift error."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False,
                           total_mass_g=875.0),
    ])
    drifted = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=True,
                           total_mass_g=1017.0),
    ])
    fake = FakePatchLLM([bad, drifted])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    drift_issues = result.attempts[1].retry_drift
    assert any("total_mass_g" in d["json_path"] for d in drift_issues)


# ---------------------------------------------------------------------------
# M. retry preserves all eight overlays and their order
# ---------------------------------------------------------------------------

def test_M_retry_preserves_eight_overlays_order() -> None:
    """Eight-overlay patch (VERA4-like but reactor-neutral IDs)."""
    ids = [f"grid_{i}" for i in range(8)]
    bad_ovs = [_make_overlay_json(oid, through_path_preserved=False) for oid in ids]
    good_ovs = [_make_overlay_json(oid, through_path_preserved=True) for oid in ids]
    bad = _make_overlays_patch_json(bad_ovs)
    good = _make_overlays_patch_json(good_ovs)
    fake = FakePatchLLM([bad, good])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="reactor with 8 spacer grids",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is True
    parsed = result.parsed_patch
    assert parsed is not None
    result_ids = [ov["overlay_id"] for ov in parsed["overlays"]]
    assert result_ids == ids


# ---------------------------------------------------------------------------
# N. upstream valid patches are not regenerated
# ---------------------------------------------------------------------------

def test_N_upstream_not_regenerated() -> None:
    """When previous_patch is available, compact retry is used (not full prompt)."""
    previous = {"patch_type": "axial_overlays", "overlays": [
        _make_overlay_json("sg1", through_path_preserved=False),
    ]}
    issues = [{
        "code": "patch.axial_overlays.mode_semantic_contradiction",
        "severity": "error", "message": "x",
        "path": "overlays[sg1].through_path_preserved",
    }]
    prompt = build_retry_prompt(
        "axial_overlays", "req", None, issues, 1,
        previous_patch=previous,
    )
    # Compact retry does NOT contain the base prompt's output contract
    assert "NOT generating a SimulationPlan" not in prompt
    assert "YOUR PREVIOUS PATCH" in prompt


# ---------------------------------------------------------------------------
# O. LLM call count proves only axial_overlays was retried
# ---------------------------------------------------------------------------

def test_O_llm_call_count_proves_single_patch_retry() -> None:
    """generate_patch for axial_overlays only calls LLM for axial_overlays."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False),
    ])
    good = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=True),
    ])
    fake = FakePatchLLM([bad, good])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is True
    assert len(fake.prompts) == 2  # only axial_overlays calls


# ---------------------------------------------------------------------------
# P. max retry budget fail closed
# ---------------------------------------------------------------------------

def test_P_max_retry_fail_closed() -> None:
    """When all attempts have contradiction, generation fails."""
    bad = _make_overlays_patch_json([
        _make_overlay_json("sg1", through_path_preserved=False),
    ])
    fake = FakePatchLLM([bad, bad])
    result = generate_patch(
        patch_type="axial_overlays",
        requirement="generic reactor",
        llm_client=fake, max_attempts=2,
    )
    assert result.ok is False
    codes = [i["code"] for i in result.issues]
    assert "patch.axial_overlays.mode_semantic_contradiction" in codes
    assert "patch_generation.no_progress_duplicate_candidate" in codes
    assert "patch_generation.max_attempts_exceeded" in codes
    assert len(result.attempts) == 2


# ---------------------------------------------------------------------------
# Q. full planning status remains blocked while axial_overlays invalid
# ---------------------------------------------------------------------------

def test_Q_planning_blocked_with_invalid_overlay() -> None:
    """An invalid axial_overlays patch produces validation issues that would
    block plan assembly."""
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="sg1", overlay_kind="spacer_grid",
            geometry_mode="mass_conserving_outer_frame",
            through_path_preserved=False,
            target_lattice_id="lat1", material_id="m1", total_mass_g=500,
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is False
    assert any(
        i.code == "patch.axial_overlays.mode_semantic_contradiction"
        for i in result.issues
    )


# ---------------------------------------------------------------------------
# R. full planning status passes only after fix
# ---------------------------------------------------------------------------

def test_R_validation_passes_after_fix() -> None:
    """Same overlay with through_path_preserved=True passes validation."""
    patch = AxialOverlaysPatch(overlays=[
        AxialOverlayPatchItem(
            overlay_id="sg1", overlay_kind="spacer_grid",
            geometry_mode="mass_conserving_outer_frame",
            through_path_preserved=True,
            target_lattice_id="lat1", material_id="m1", total_mass_g=500,
            z_min_cm=10, z_max_cm=12,
        ),
    ])
    result = validate_patch(patch)
    assert result.ok is True
