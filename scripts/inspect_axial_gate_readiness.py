#!/usr/bin/env python
"""Inspect Axial Geometry Gate readiness from a saved campaign state.

This is a sanitized, offline diagnostic helper.  It reads either a
``plan_build_state.json`` or a ``campaign_checkpoint.json`` and reports only
gate/stage status, patch availability, accepted upstream hashes, owner routes,
and deterministic preflight issue codes.  It does not read or emit raw prompts,
provider responses, reasoning, or credentials.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openmc_agent.plan_builder.closed_loop.axial_geometry_evidence import (
    axial_geometry_gate_applicable,
    axial_geometry_gate_ready,
)
from openmc_agent.plan_builder.closed_loop.axial_geometry_issue_policy import (
    axial_geometry_issue_owner,
)
from openmc_agent.plan_builder.closed_loop.axial_geometry_preflight import (
    run_axial_geometry_preflight,
)
from openmc_agent.plan_builder.closed_loop.campaign_checkpoint import (
    CampaignCheckpointStore,
)
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.state import PlanBuildState


AXIAL_REQUIRED_PATCH_TYPES: tuple[str, ...] = (
    "base_path_axial_profiles",
    "axial_layers",
    "axial_overlays",
)
UPSTREAM_STAGE_KEYS: dict[str, str] = {
    "facts": "plan_gate_facts",
    "material_universe": "plan_gate_material_universe",
    "placement": "plan_gate_placement",
}


def _load_state(path: str | Path) -> tuple[PlanBuildState, dict[str, Any]]:
    """Load a PlanBuildState from a state file or latest checkpoint snapshot."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "state_snapshots" in raw:
        store = CampaignCheckpointStore(source)
        snapshot = store.latest_state_snapshot()
        if snapshot is None:
            raise ValueError(f"checkpoint has no state snapshots: {source}")
        return PlanBuildState.model_validate(snapshot.plan_build_state), {
            "kind": "campaign_checkpoint",
            "path": str(source),
            "boundary": snapshot.boundary,
            "campaign_id": snapshot.campaign_id,
            "sequence": snapshot.sequence,
        }
    return PlanBuildState.model_validate(raw), {"kind": "plan_build_state", "path": str(source)}


def _stage_summary(state: PlanBuildState, stage_key: str) -> dict[str, Any]:
    stage = state.plan_loop_stages.get(stage_key)
    if stage is None:
        return {"status": "missing", "accepted_input_hash": ""}
    return {
        "status": str(getattr(stage.status, "value", stage.status)),
        "accepted_input_hash": str(stage.metadata.get("accepted_input_hash", "")),
        "reviewed_input_hash": str(stage.metadata.get("reviewed_input_hash", "")),
    }


def _patch_status_by_type(state: PlanBuildState) -> dict[str, dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for patch_id, envelope in state.patches.items():
        item = grouped.setdefault(
            envelope.patch_type,
            {"valid_patch_ids": [], "invalid_patch_ids": [], "other_patch_ids": []},
        )
        status = str(getattr(envelope.status, "value", envelope.status))
        if status == "valid":
            item["valid_patch_ids"].append(patch_id)
        elif status == "invalid":
            item["invalid_patch_ids"].append(patch_id)
        else:
            item["other_patch_ids"].append(patch_id)
    for item in grouped.values():
        for key in tuple(item):
            item[key] = sorted(item[key])
        item["valid_count"] = len(item["valid_patch_ids"])
        item["invalid_count"] = len(item["invalid_patch_ids"])
        item["other_count"] = len(item["other_patch_ids"])
    return dict(sorted(grouped.items()))


def _component_task_status_by_patch_type(state: PlanBuildState) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = {}
    for task in state.component_tasks:
        grouped.setdefault(task.patch_type, []).append(
            {
                "task_id": task.task_id,
                "status": str(getattr(task.status, "value", task.status)),
            }
        )
    return {key: sorted(value, key=lambda item: item["task_id"]) for key, value in sorted(grouped.items())}


def _gate_required_by_tasks(state: PlanBuildState) -> bool:
    return any(task.patch_type in AXIAL_REQUIRED_PATCH_TYPES for task in state.component_tasks)


def _issue_summary(issues: list[dict[str, Any]]) -> dict[str, Any]:
    blocking = [issue for issue in issues if issue.get("blocking")]
    owner_routes: dict[str, Any] = {}
    for issue in issues:
        code = str(issue.get("code", ""))
        owner = axial_geometry_issue_owner(code, issue)
        owner_routes[code] = owner.model_dump(mode="json") if owner is not None else None
    return {
        "issue_count": len(issues),
        "blocking_issue_count": len(blocking),
        "codes": sorted({str(issue.get("code", "")) for issue in issues}),
        "blocking_codes": sorted({str(issue.get("code", "")) for issue in blocking}),
        "owner_routes": owner_routes,
    }


def _recommended_action(
    *,
    upstream: dict[str, dict[str, Any]],
    target_stage: dict[str, Any],
    missing_required_valid_patch_types: list[str],
    preflight_ok: bool,
    blocking_issue_count: int,
) -> str:
    if target_stage["status"] == "accepted":
        return "proceed_to_assembled_plan_gate"
    if any(item["status"] != "accepted" for item in upstream.values()):
        return "close_upstream_gate_before_axial_geometry"
    if missing_required_valid_patch_types:
        return "run_stop_after_gate_axial_geometry_to_generate_missing_axial_patches"
    if not preflight_ok or blocking_issue_count:
        return "close_deterministic_axial_preflight_blockers_offline"
    return "run_axial_geometry_target_review"


def inspect_axial_gate_readiness(path: str | Path) -> dict[str, Any]:
    """Return a sanitized Axial Gate readiness report."""
    state, source = _load_state(path)
    policy = PlanClosedLoopPolicy(mode="controlled", axial_geometry_review_mode="controlled")
    patch_status = _patch_status_by_type(state)
    missing_required = [
        patch_type
        for patch_type in AXIAL_REQUIRED_PATCH_TYPES
        if not patch_status.get(patch_type, {}).get("valid_patch_ids")
    ]
    upstream = {
        gate_id: _stage_summary(state, stage_key)
        for gate_id, stage_key in UPSTREAM_STAGE_KEYS.items()
    }
    target_stage = _stage_summary(state, "plan_gate_axial_geometry")

    preflight_error = ""
    preflight_issues: list[dict[str, Any]] = []
    preflight_ok = False
    try:
        preflight = run_axial_geometry_preflight(state=state, policy=policy)
        preflight_ok = bool(preflight.ok)
        preflight_issues = [dict(issue) for issue in preflight.issues]
    except Exception as exc:  # pragma: no cover - defensive fail-closed diagnostics.
        preflight_error = f"{type(exc).__name__}: {exc}"

    issues = _issue_summary(preflight_issues)
    gate_required_by_tasks = _gate_required_by_tasks(state)
    gate_applicable = bool(axial_geometry_gate_applicable(state))
    gate_ready = bool(axial_geometry_gate_ready(state))

    return {
        "ok": True,
        "gate_id": "axial_geometry",
        "source": source,
        "upstream_stages": upstream,
        "target_stage": target_stage,
        "gate_required_by_tasks": gate_required_by_tasks,
        "gate_applicable_by_evidence": gate_applicable,
        "gate_ready": gate_ready,
        "required_patch_types": list(AXIAL_REQUIRED_PATCH_TYPES),
        "missing_required_valid_patch_types": missing_required,
        "patch_status_by_type": patch_status,
        "component_task_status_by_patch_type": _component_task_status_by_patch_type(state),
        "preflight": {
            "ok": preflight_ok,
            "error": preflight_error,
            **issues,
        },
        "next_recommended_action": _recommended_action(
            upstream=upstream,
            target_stage=target_stage,
            missing_required_valid_patch_types=missing_required,
            preflight_ok=preflight_ok,
            blocking_issue_count=int(issues["blocking_issue_count"]),
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect Axial Geometry Gate readiness offline.")
    parser.add_argument("path", help="Path to plan_build_state.json or campaign_checkpoint.json")
    parser.add_argument("--out", default=None, help="Optional output JSON path")
    args = parser.parse_args(argv)

    payload = inspect_axial_gate_readiness(args.path)
    text = json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
