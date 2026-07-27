"""Phase 7A real controlled five-gate canary CLI.

Reactor-neutral real-LLM campaign that drives the full production
planning stack with the five-gate controlled Plan Closed Loop, fragmented
universes, strict structured patch output, and (optionally) real OpenMC
export / geometry debug / smoke.

Stage modes:

    planning        Run until Assembled Plan Gate accepted (no render).
    render-compile  Add renderer invocation, model.py, validate_openmc_script.
    openmc-smoke    Add real XML export, geometry debug and low-cost OpenMC smoke.

Built-in case presets (``--case``):

    vera3-3a, vera3-3b, vera4

``--input`` always overrides the preset's input path.  Production code
never branches on a case name.

Examples
--------

VERA3 3B planning canary (offline-safe: environment-gated)::

    python scripts/evaluate_plan_closed_loop_real_canary.py \\
        --case vera3-3b --stage planning \\
        --model ds:deepseek-v4-flash \\
        --runs 1 --confirm-real-campaign \\
        --output-dir data/runs/phase7a_vera3_3b

VERA4 planning canary with fragmented universes::

    python scripts/evaluate_plan_closed_loop_real_canary.py \\
        --case vera4 --stage planning \\
        --model ds:deepseek-v4-flash \\
        --universes-generation-mode fragmented \\
        --runs 1 --confirm-real-campaign \\
        --output-dir data/runs/phase7a_vera4
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def _load_seed_state_json(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "state_snapshots" in raw:
        from openmc_agent.plan_builder.closed_loop.campaign_checkpoint import CampaignCheckpointStore

        snapshot = CampaignCheckpointStore(path).latest_state_snapshot()
        if snapshot is None:
            raise ValueError(f"accepted state checkpoint has no state snapshots: {path}")
        raw = snapshot.plan_build_state
    if not isinstance(raw, dict):
        raise ValueError(f"accepted state seed must be a JSON object: {path}")
    return raw


def _sanitize_accepted_seed_state(raw: dict) -> dict:
    """Drop audit-only raw LLM fields before using an accepted-state seed."""
    data = copy.deepcopy(raw)
    for key in (
        "facts_review_history",
        "facts_revision_history",
        "placement_review_history",
        "placement_revision_history",
    ):
        if key in data:
            data[key] = []
    metadata = data.get("metadata")
    if isinstance(metadata, dict):
        metadata.pop("patch_attempt_artifacts", None)
    patches = data.get("patches")
    if isinstance(patches, dict):
        for envelope in patches.values():
            if isinstance(envelope, dict):
                envelope["raw_text"] = None
    return data


def _has_forbidden_secret_seed_field(value: object) -> bool:
    fragments = (
        "api_key",
        "token",
        "secret",
        "password",
        "credential",
        "authorization",
    )
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if any(fragment in key_text for fragment in fragments) and item not in (None, "", [], {}):
                return True
            if _has_forbidden_secret_seed_field(item):
                return True
    elif isinstance(value, list):
        return any(_has_forbidden_secret_seed_field(item) for item in value)
    return False


def _has_raw_seed_field(value: object) -> bool:
    fragments = ("raw_text", "raw_output", "raw_response", "prompt", "reasoning")
    if isinstance(value, dict):
        for key, item in value.items():
            key_text = str(key).lower()
            if key_text == "prompt_hash":
                continue
            if any(fragment in key_text for fragment in fragments) and item not in (None, "", [], {}):
                return True
            if _has_raw_seed_field(item):
                return True
    elif isinstance(value, list):
        return any(_has_raw_seed_field(item) for item in value)
    return False


def _load_accepted_plan_build_state_seed(path: Path, *, case: object, stop_after_gate: str | None) -> dict[str, object]:
    """Validate an explicit accepted PlanBuildState seed for target-gate canaries."""
    if stop_after_gate not in {"axial_geometry", "assembled_plan"}:
        raise ValueError(
            "--accepted-plan-build-state is only supported for "
            "--stop-after-gate axial_geometry or assembled_plan"
        )
    raw = _load_seed_state_json(path)
    if _has_forbidden_secret_seed_field(raw):
        raise ValueError("accepted PlanBuildState seed contains secret-like fields")
    raw = _sanitize_accepted_seed_state(raw)
    if _has_raw_seed_field(raw):
        raise ValueError("accepted PlanBuildState seed contains raw provider fields")

    from openmc_agent.inspect import compose_operating_state_requirement
    from openmc_agent.plan_builder.closed_loop.models import PlanStageStatus
    from openmc_agent.plan_builder.state import PlanBuildState

    state = PlanBuildState.model_validate(raw)
    input_path = Path(getattr(case, "input_path"))
    raw_requirement = input_path.read_text(encoding="utf-8")
    expected_requirement = compose_operating_state_requirement(
        raw_requirement,
        str(getattr(case, "operating_state", "") or ""),
    )
    if state.requirement_text.rstrip() != expected_requirement.rstrip():
        raise ValueError("accepted PlanBuildState seed requirement_text does not match this case/input")

    required_upstream = ["facts", "material_universe", "placement"]
    if stop_after_gate == "assembled_plan":
        required_upstream.append("axial_geometry")
    for gate_id in required_upstream:
        stage = state.plan_loop_stages.get(f"plan_gate_{gate_id}")
        if stage is None or stage.status is not PlanStageStatus.ACCEPTED:
            status = stage.status.value if stage is not None else "missing"
            raise ValueError(
                f"accepted PlanBuildState seed requires accepted {gate_id} gate "
                f"(current status={status})"
            )

    return {
        "accepted_plan_build_state": state.model_dump(mode="json"),
        "accepted_plan_build_state_path": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 7A real controlled five-gate canary campaign."
    )
    parser.add_argument(
        "--case",
        choices=("vera3-3a", "vera3-3b", "vera4"),
        help="Labelled case preset.  --input always overrides the preset's input path.",
    )
    parser.add_argument(
        "--input", type=Path,
        help="Override the case input requirement document.",
    )
    parser.add_argument(
        "--operating-state", default="",
        help="Substate identifier inside the document (3A / 3B / empty).",
    )
    parser.add_argument(
        "--stage", choices=("planning", "render-compile", "openmc-smoke"),
        default="planning",
        help="How far to run.",
    )
    parser.add_argument("--model", default="ds:deepseek-v4-flash")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--confirm-real-campaign", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--accepted-plan-build-state",
        type=Path,
        help=(
            "Sanitized PlanBuildState JSON seed for target-gate runs. "
            "Requires matching requirement text and accepted upstream gates; "
            "does not relax --resume fingerprint checks."
        ),
    )
    parser.add_argument("--human-answer-file", type=Path)
    parser.add_argument(
        "--universes-generation-mode",
        choices=("auto", "monolithic", "fragmented"),
        default="auto",
    )
    parser.add_argument("--universe-fragment-max-tokens", type=int)
    parser.add_argument("--large-patch-safe-output-ratio", type=float, default=0.6)
    parser.add_argument(
        "--strict-structured-patch-output", action="store_true", default=True,
    )
    parser.add_argument(
        "--no-strict-structured-patch-output", dest="strict_structured_patch_output",
        action="store_false",
    )
    parser.add_argument(
        "--material-policy", choices=("strict", "permissive"), default="strict",
    )
    parser.add_argument(
        "--runtime-supervisor-mode",
        choices=("deterministic", "real"), default="deterministic",
    )
    parser.add_argument(
        "--runtime-repair-mode",
        choices=("diagnose_only", "apply_if_safe", "off"), default="diagnose_only",
    )
    parser.add_argument("--max-runtime-iterations", type=int, default=0)
    parser.add_argument("--enable-runtime-llm-repair", action="store_true")
    parser.add_argument("--enable-plots", action="store_true")
    parser.add_argument("--wall-time-limit-seconds", type=float, default=1800.0)
    parser.add_argument("--campaign-timeout-seconds", type=float, default=14400.0)
    parser.add_argument(
        "--max-llm-calls", type=int,
        help="Override the estimated budget; --max-llm-calls always wins.",
    )
    parser.add_argument(
        "--plan-loop-max-review-rounds", type=int, default=2,
        help="Maximum reviewer rounds per enabled gate.",
    )
    parser.add_argument(
        "--plan-loop-max-repair-rounds", type=int, default=2,
        help="Maximum repair rounds per enabled gate.",
    )
    parser.add_argument(
        "--plan-loop-max-additional-llm-calls", type=int, default=24,
        help="Shared deterministic closed-loop review/repair call budget.",
    )
    parser.add_argument("--expected-patch-count", type=int, default=8)
    parser.add_argument("--expected-universe-count", type=int, default=0)
    parser.add_argument(
        "--acceptance-profile", choices=("pilot", "qualification"), default="pilot",
    )
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--json", action="store_true")
    # Phase 8A Step 4: plan investigation flags.  Default mode=off keeps
    # legacy behaviour.  Controlled real canary in this step only allows
    # the facts patch type; materials/universes controlled qualification
    # is reserved for Step 5+.
    parser.add_argument(
        "--plan-investigation-mode",
        choices=("off", "advisory", "controlled"),
        default="off",
        help="Plan investigation mode (default: off).",
    )
    parser.add_argument(
        "--plan-investigation-patch-types",
        type=str,
        default="facts",
        help="Comma-separated patch types to investigate (Step 4: facts only).",
    )
    parser.add_argument("--plan-investigation-max-tool-calls", type=int, default=5)
    parser.add_argument("--plan-investigation-max-results-per-tool", type=int, default=50)
    parser.add_argument("--plan-investigation-max-evidence-claims", type=int, default=100)
    parser.add_argument("--plan-investigation-model", type=str, default=None)
    parser.add_argument("--plan-investigation-reasoning-effort", type=str, default=None)
    parser.add_argument("--plan-investigation-output-mode", type=str, default=None)
    parser.add_argument("--plan-investigation-max-tokens", type=int, default=None)
    parser.add_argument(
        "--no-plan-investigation-require-source-backed-evidence",
        dest="plan_investigation_require_source_backed_evidence",
        action="store_false",
        default=True,
    )
    parser.add_argument(
        "--stop-after-gate",
        choices=("facts", "material_universe", "placement", "axial_geometry", "assembled_plan"),
        default=None,
        help="Reactor-neutral gate name; the campaign enables only this gate.",
    )
    args = parser.parse_args()

    # Phase 8A Step 6: controlled/advisory investigation now supports
    # facts, materials, and universes patch types (Materials and
    # Universes investigations are wired in Step 6A).  Validate the
    # patch-type list against the allowed set; reject unknown names
    # so a typo doesn't silently fall back to the legacy path.
    requested_patch_types = tuple(
        p.strip() for p in args.plan_investigation_patch_types.split(",") if p.strip()
    )
    if args.plan_investigation_mode in {"advisory", "controlled"}:
        allowed = {"facts", "materials", "universes"}
        disallowed = [p for p in requested_patch_types if p not in allowed]
        if disallowed:
            print(
                f"ERROR: Phase 8A Step 6 controlled/advisory investigation "
                f"supports {sorted(allowed)}.  Requested: {requested_patch_types}.  "
                f"Disallowed: {disallowed}.",
                file=sys.stderr,
            )
            return 2

    if not args.confirm_real_campaign:
        print(
            "ERROR: pass --confirm-real-campaign to acknowledge this makes real "
            "LLM (and optionally OpenMC) calls.",
            file=sys.stderr,
        )
        return 2

    from openmc_agent.real_campaign_harness import (
        CanaryCampaignConfig,
        builtin_case_registry,
        detect_provider_environment,
        load_human_answer_file,
        resolve_case,
        run_real_canary_campaign,
    )

    env_status = detect_provider_environment(args.model)
    if not env_status.llm_environment_available:
        print(
            f"BLOCKED_BY_LLM_ENVIRONMENT: {env_status.api_key_env} not set "
            f"(provider={env_status.provider}).",
            file=sys.stderr,
        )
        if not args.json:
            return 2

    if args.stage in {"render-compile", "openmc-smoke"} and not env_status.openmc_environment_available:
        print(
            f"BLOCKED_BY_OPENMC_ENVIRONMENT: openmc library or "
            f"OPENMC_CROSS_SECTIONS not available for stage={args.stage}.",
            file=sys.stderr,
        )
        if not args.json:
            return 2

    # Load human answers (typed JSON) if supplied.
    try:
        human_answers, human_answer_hash = load_human_answer_file(
            str(args.human_answer_file) if args.human_answer_file else None
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    # Resolve case (preset + overrides).
    case = resolve_case(
        case=args.case,
        input_path=str(args.input) if args.input else None,
        operating_state=args.operating_state,
        model=args.model,
        output_dir=str(args.output_dir),
        planning_stage=args.stage,
        human_answer_file=str(args.human_answer_file) if args.human_answer_file else None,
        acceptance_profile=args.acceptance_profile,
    )

    metadata: dict[str, object] = {}
    if args.accepted_plan_build_state:
        try:
            metadata.update(
                _load_accepted_plan_build_state_seed(
                    args.accepted_plan_build_state,
                    case=case,
                    stop_after_gate=args.stop_after_gate,
                )
            )
        except (FileNotFoundError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2

    campaign = CanaryCampaignConfig(
        case=case,
        runs=args.runs,
        model=args.model,
        planning_stage=args.stage,
        universes_generation_mode=args.universes_generation_mode,
        universe_fragment_max_tokens=args.universe_fragment_max_tokens,
        large_patch_safe_output_ratio=args.large_patch_safe_output_ratio,
        strict_structured_patch_output=args.strict_structured_patch_output,
        material_policy=args.material_policy,
        runtime_supervisor_mode=args.runtime_supervisor_mode,
        runtime_repair_mode=args.runtime_repair_mode,
        max_runtime_iterations=args.max_runtime_iterations,
        enable_runtime_llm_repair=args.enable_runtime_llm_repair,
        enable_plots=args.enable_plots,
        wall_time_limit_s=args.wall_time_limit_seconds,
        campaign_timeout_s=args.campaign_timeout_seconds,
        max_llm_calls=args.max_llm_calls,
        expected_patch_count=args.expected_patch_count,
        expected_universe_count=args.expected_universe_count,
        human_answers=human_answers,
        human_answer_hash=human_answer_hash,
        fail_fast=args.fail_fast,
        resume=args.resume,
        plan_investigation_mode=args.plan_investigation_mode,
        plan_investigation_patch_types=requested_patch_types,
        plan_investigation_model=args.plan_investigation_model,
        plan_investigation_reasoning_effort=args.plan_investigation_reasoning_effort,
        plan_investigation_output_mode=args.plan_investigation_output_mode,
        plan_investigation_max_tool_calls=args.plan_investigation_max_tool_calls,
        plan_investigation_max_results_per_tool=args.plan_investigation_max_results_per_tool,
        plan_investigation_max_evidence_claims=args.plan_investigation_max_evidence_claims,
        plan_investigation_require_source_backed_evidence=args.plan_investigation_require_source_backed_evidence,
        plan_investigation_max_tokens=args.plan_investigation_max_tokens,
        stop_after_gate=args.stop_after_gate,
        plan_loop_max_review_rounds=args.plan_loop_max_review_rounds,
        plan_loop_max_repair_rounds=args.plan_loop_max_repair_rounds,
        plan_loop_max_additional_llm_calls=args.plan_loop_max_additional_llm_calls,
        metadata=metadata,
    )

    manifest = run_real_canary_campaign(args.output_dir, campaign)

    status = manifest.get("aggregate_status", "UNKNOWN")
    if args.json:
        print(json.dumps(manifest, indent=2, default=str))
    else:
        print(f"Status: {status}")
        print(f"Completed: {manifest.get('completed_runs', 0)}/{manifest.get('requested_runs', 0)}")
        print(f"Successful: {manifest.get('successful_runs', 0)}")
        if manifest.get("resume_mismatches"):
            print(f"Resume mismatches: {manifest['resume_mismatches']}")

    # Exit codes.
    if "BLOCKED_BY" in status or "CONFIG_MISMATCH" in status:
        return 2
    if "PASSED" in status or status == "CAMPAIGN_RUNNING" and manifest.get("successful_runs", 0) > 0:
        return 0
    if "FAIL" in status:
        return 1
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
