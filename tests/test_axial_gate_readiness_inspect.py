"""Tests for the Step 3K Axial Gate readiness diagnostic script."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

from openmc_agent.plan_builder.closed_loop.campaign_checkpoint import (
    BOUNDARY_GATE_PLACEMENT,
    CampaignCheckpointStore,
    CampaignStateSnapshot,
    checkpoint_fingerprint,
)
from openmc_agent.plan_builder.state import PlanComponentTask
from tests._axial_geometry_fixtures import state_with_axial_patches

ROOT = Path(__file__).resolve().parent.parent


def _load_script():
    spec = importlib.util.spec_from_file_location(
        "inspect_axial_gate_readiness",
        str(ROOT / "scripts" / "inspect_axial_gate_readiness.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_state(tmp_path: Path, *, remove_axial_patches: bool = False) -> Path:
    state = state_with_axial_patches(include_profiles=True)
    state.add_task(
        PlanComponentTask(
            task_id="task_axial_layers",
            patch_type="axial_layers",
            description="Define axial layers",
        )
    )
    state.add_task(
        PlanComponentTask(
            task_id="task_axial_overlays",
            patch_type="axial_overlays",
            description="Define axial overlays",
            dependencies=["task_axial_layers"],
        )
    )
    if remove_axial_patches:
        for patch_id, envelope in list(state.patches.items()):
            if envelope.patch_type in {"base_path_axial_profiles", "axial_layers", "axial_overlays"}:
                del state.patches[patch_id]
    path = tmp_path / "plan_build_state.json"
    path.write_text(json.dumps(state.model_dump(mode="json"), indent=2), encoding="utf-8")
    return path


def test_ready_axial_state_reports_review_ready(tmp_path: Path) -> None:
    inspect = _load_script()
    payload = inspect.inspect_axial_gate_readiness(_write_state(tmp_path))
    assert payload["gate_id"] == "axial_geometry"
    assert payload["gate_required_by_tasks"] is True
    assert payload["gate_ready"] is True
    assert payload["missing_required_valid_patch_types"] == []
    assert payload["preflight"]["ok"] is True
    assert payload["next_recommended_action"] == "run_axial_geometry_target_review"


def test_missing_axial_patches_recommends_stop_after_generation(tmp_path: Path) -> None:
    inspect = _load_script()
    payload = inspect.inspect_axial_gate_readiness(
        _write_state(tmp_path, remove_axial_patches=True)
    )
    assert payload["gate_required_by_tasks"] is True
    assert payload["gate_ready"] is False
    assert payload["missing_required_valid_patch_types"] == [
        "base_path_axial_profiles",
        "axial_layers",
        "axial_overlays",
    ]
    assert payload["preflight"]["blocking_issue_count"] > 0
    assert (
        payload["next_recommended_action"]
        == "run_stop_after_gate_axial_geometry_to_generate_missing_axial_patches"
    )


def test_checkpoint_input_uses_latest_snapshot(tmp_path: Path) -> None:
    inspect = _load_script()
    state = state_with_axial_patches(include_profiles=True)
    store_path = tmp_path / "campaign_checkpoint.json"
    store = CampaignCheckpointStore(store_path)
    snap = CampaignStateSnapshot(
        campaign_id="synthetic",
        boundary=BOUNDARY_GATE_PLACEMENT,
        sequence=3,
        state_hash=checkpoint_fingerprint(state.model_dump(mode="json")),
        plan_build_state=state.model_dump(mode="json"),
        requirement_hash="r",
        input_hash="i",
        policy_hash="p",
        git_sha="g",
        structured_output_policy_hash="s",
        accepted_at="t",
    )
    store.accept_state_snapshot(snap)
    payload = inspect.inspect_axial_gate_readiness(store_path)
    assert payload["source"]["kind"] == "campaign_checkpoint"
    assert payload["source"]["boundary"] == BOUNDARY_GATE_PLACEMENT
    assert payload["preflight"]["ok"] is True
