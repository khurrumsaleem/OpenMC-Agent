"""Regression tests for the VERA4 four-gate accepted Assembled seed."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from openmc_agent.graph import _run_incremental_plan_generation
from openmc_agent.plan_builder.closed_loop.models import (
    PlanClosedLoopPolicy,
    PlanGateId,
    PlanStageStatus,
)
from openmc_agent.plan_builder.executor import IncrementalExecutionResult
from openmc_agent.plan_builder.state import PlanBuildState


ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = ROOT / "tests" / "fixtures" / "accepted_plan_seeds" / "vera4_four_gate_accepted_plan_build_state.json"


def _load_real_canary_script():
    spec = importlib.util.spec_from_file_location(
        "evaluate_plan_closed_loop_real_canary",
        str(ROOT / "scripts" / "evaluate_plan_closed_loop_real_canary.py"),
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _walk(value, path=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield f"{path}/{key}", item
            yield from _walk(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield f"{path}/{index}", item
            yield from _walk(item, f"{path}/{index}")


def test_vera4_four_gate_seed_shape_and_privacy() -> None:
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    state = PlanBuildState.model_validate(payload)

    assert state.assembled_plan is None
    assert {patch.patch_type for patch in state.patches.values()} == {
        "assembly_catalog",
        "axial_layers",
        "core_layout",
        "facts",
        "localized_insert_profiles",
        "materials",
        "universes",
    }
    for gate_id in ("facts", "material_universe", "placement", "axial_geometry"):
        assert state.plan_loop_stages[f"plan_gate_{gate_id}"].status is PlanStageStatus.ACCEPTED
        assert state.plan_loop_stages[f"plan_gate_{gate_id}"].metadata.get("accepted_input_hash")
    assert state.plan_loop_stages["plan_gate_assembled_plan"].status is PlanStageStatus.PENDING

    forbidden_key_fragments = ("api_key", "secret", "password", "authorization", "prompt", "reasoning")
    for path, value in _walk(payload):
        key = path.rsplit("/", 1)[-1].lower()
        if key == "prompt_hash":
            continue
        assert not any(fragment in key for fragment in forbidden_key_fragments), path
        if any(fragment in key for fragment in ("raw_text", "raw_output", "raw_response")):
            assert value in (None, "", [], {}), path


def test_real_canary_loader_accepts_four_gate_seed_for_assembled_plan() -> None:
    script = _load_real_canary_script()
    payload = script._load_accepted_plan_build_state_seed(
        SEED_PATH,
        case=SimpleNamespace(input_path=str(ROOT / "Input" / "VERA4_problem.md"), operating_state=""),
        stop_after_gate="assembled_plan",
    )
    state = payload["accepted_plan_build_state"]
    assert payload["accepted_plan_build_state_path"] == str(SEED_PATH)
    assert state["plan_loop_stages"]["plan_gate_axial_geometry"]["status"] == "accepted"
    assert state["plan_loop_stages"]["plan_gate_assembled_plan"]["status"] == "pending"


def test_real_canary_loader_rejects_seed_without_axial_acceptance_for_assembled_plan(tmp_path: Path) -> None:
    script = _load_real_canary_script()
    payload = json.loads(SEED_PATH.read_text(encoding="utf-8"))
    payload["plan_loop_stages"]["plan_gate_axial_geometry"]["status"] = "pending"
    seed_path = tmp_path / "seed_missing_axial.json"
    seed_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        script._load_accepted_plan_build_state_seed(
            seed_path,
            case=SimpleNamespace(input_path=str(ROOT / "Input" / "VERA4_problem.md"), operating_state=""),
            stop_after_gate="assembled_plan",
        )
    except ValueError as exc:
        assert "accepted axial_geometry gate" in str(exc)
    else:
        raise AssertionError("expected missing axial acceptance to fail closed")


def test_assembled_target_seed_from_four_gate_fixture_enters_incremental_executor(monkeypatch) -> None:
    seed = PlanBuildState.model_validate_json(SEED_PATH.read_text(encoding="utf-8"))
    calls: list[dict[str, object]] = []

    def fake_run_incremental_planning(**kwargs):
        calls.append(kwargs)
        state = kwargs["state"]
        state.metadata["executor_called_for_assembled_seed"] = True
        return IncrementalExecutionResult(
            ok=True,
            state=state,
            assembled_plan=None,
            summary={"stopped_after_gate": "assembled_plan"},
            plan_loop_outcome={"status": "stopped_after_gate"},
        )

    def fail_fast_path(*_args, **_kwargs):
        raise AssertionError("assembled target seed must not bypass the incremental executor")

    monkeypatch.setattr(
        "openmc_agent.plan_builder.executor.run_incremental_planning",
        fake_run_incremental_planning,
    )
    monkeypatch.setattr("openmc_agent.graph._write_final_simulation_plan", fail_fast_path)

    result = _run_incremental_plan_generation(
        {
            "requirement": seed.requirement_text,
            "planning_mode_decision": {"mode": "incremental"},
            "plan_build_state": seed.model_dump(mode="json"),
            "accepted_plan_build_state": seed.model_dump(mode="json"),
        },
        patch_llm_client=lambda _prompt: "",
        plan_loop_policy=PlanClosedLoopPolicy(
            mode="controlled",
            stop_after_gate=PlanGateId.ASSEMBLED_PLAN,
            assembled_plan_review_mode="controlled",
        ),
    )

    assert calls
    assert result["plan_build_state"]["metadata"]["accepted_plan_build_state_seed"] is True
    assert result["plan_build_state"]["metadata"]["target_gate_seed_gate"] == "assembled_plan"
