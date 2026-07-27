"""Tests for the Step 3K Axial Gate readiness diagnostic script."""

from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
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
    assert payload["gate_applicable_by_evidence"] is True
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


def test_real_canary_cli_accepts_sanitized_placement_seed(tmp_path: Path) -> None:
    script = _load_real_canary_script()
    state = state_with_axial_patches(include_profiles=True)
    requirement_path = tmp_path / "input.md"
    requirement_path.write_text(state.requirement_text, encoding="utf-8")
    from openmc_agent.inspect import compose_operating_state_requirement
    state.requirement_text = compose_operating_state_requirement(
        requirement_path.read_text(encoding="utf-8"), ""
    )
    seed_path = tmp_path / "seed_state.json"
    seed_path.write_text(json.dumps(state.model_dump(mode="json")), encoding="utf-8")
    payload = script._load_accepted_plan_build_state_seed(
        seed_path,
        case=SimpleNamespace(input_path=str(requirement_path), operating_state=""),
        stop_after_gate="axial_geometry",
    )
    assert payload["accepted_plan_build_state_path"] == str(seed_path)
    assert payload["accepted_plan_build_state"]["state_id"] == state.state_id


def test_real_canary_cli_sanitizes_seed_audit_raw_outputs(tmp_path: Path) -> None:
    script = _load_real_canary_script()
    state = state_with_axial_patches(include_profiles=True)
    requirement_path = tmp_path / "input.md"
    requirement_path.write_text(state.requirement_text, encoding="utf-8")
    from openmc_agent.inspect import compose_operating_state_requirement
    state.requirement_text = compose_operating_state_requirement(
        requirement_path.read_text(encoding="utf-8"), ""
    )
    raw = state.model_dump(mode="json")
    raw["facts_review_history"] = [{"raw_outputs": ["raw provider output"], "prompt_text": "prompt"}]
    next(iter(raw["patches"].values()))["raw_text"] = "raw patch output"
    seed_path = tmp_path / "seed_state.json"
    seed_path.write_text(json.dumps(raw), encoding="utf-8")
    payload = script._load_accepted_plan_build_state_seed(
        seed_path,
        case=SimpleNamespace(input_path=str(requirement_path), operating_state=""),
        stop_after_gate="axial_geometry",
    )
    assert payload["accepted_plan_build_state"]["facts_review_history"] == []
    assert next(iter(payload["accepted_plan_build_state"]["patches"].values()))["raw_text"] is None


def test_real_canary_cli_rejects_seed_with_secret_like_field(tmp_path: Path) -> None:
    script = _load_real_canary_script()
    state = state_with_axial_patches(include_profiles=True)
    requirement_path = tmp_path / "input.md"
    requirement_path.write_text(state.requirement_text, encoding="utf-8")
    from openmc_agent.inspect import compose_operating_state_requirement
    state.requirement_text = compose_operating_state_requirement(
        requirement_path.read_text(encoding="utf-8"), ""
    )
    raw = state.model_dump(mode="json")
    raw["metadata"]["api_key"] = "sk-test"
    seed_path = tmp_path / "seed_state.json"
    seed_path.write_text(json.dumps(raw), encoding="utf-8")
    try:
        script._load_accepted_plan_build_state_seed(
            seed_path,
            case=SimpleNamespace(input_path=str(requirement_path), operating_state=""),
            stop_after_gate="axial_geometry",
        )
    except ValueError as exc:
        assert "secret-like" in str(exc)
    else:
        raise AssertionError("expected secret-like seed rejection")


def _load_real_canary_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_plan_closed_loop_real_canary",
        str(ROOT / "scripts" / "evaluate_plan_closed_loop_real_canary.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module
