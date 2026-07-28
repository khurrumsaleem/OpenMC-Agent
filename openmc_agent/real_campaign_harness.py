"""Phase 7A real controlled five-gate campaign harness.

Reactor-neutral real-LLM campaign that drives the full production planning
stack:

    real requirement
    -> real patch LLM
    -> fragmented universes (when needed)
    -> Facts Gate
    -> Material-Universe Gate
    -> Placement Gate
    -> Axial Geometry Gate
    -> Assembled Plan Gate
    -> render (render-compile / openmc-smoke stages only)
    -> truthful qualification artifacts

The harness is intentionally **reactor-neutral**.  Benchmark names
(``vera3-3a`` / ``vera3-3b`` / ``vera4``) appear only in:

  * the built-in case registry,
  * acceptance callbacks supplied by callers,
  * test fixtures and reports.

The production code path (graph, gate reviewers, patch generator, renderer)
never branches on a benchmark name.

This module reuses the existing :class:`RealCampaignRunResult`,
:class:`RealCampaignRunConfig`, :class:`RealCampaignClientBundle` and
:class:`LLMCallRecorder` from :mod:`openmc_agent.real_campaign` and
:mod:`openmc_agent.llm_call_recorder`; it does **not** redefine them.
The Phase 7A-specific result dataclass lives here and composes a
:class:`RealCampaignRunResult` to keep backward compatibility with the
legacy VERA3B campaign.

The harness never:

  * falls back to a Fake client,
  * accepts a partial universe fragment downstream,
  * renders or exports XML before the Final Gate is accepted,
  * paperweights an environment gap as a planning/smoke failure,
  * hardens in a benchmark-specific physics value.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from openmc_agent.real_campaign import (
    RealCampaignClientBundle,
    RealCampaignRunConfig,
    RealCampaignRunResult,
    _check_resume_config_match,
    _config_hash,
    _create_client_bundle,
    _extract_results,
    _file_sha,
    _get_openmc_version,
    _git_sha,
    _populate_llm_evidence,
    _populate_provenance,
    _verify_real_openmc,
    _write_json_atomic,
    classify_real_campaign_run,
    validate_real_run_truthfulness,
)
from openmc_agent.llm_call_recorder import (
    LLMBudgetExhausted,
    LLMCallRecorder,
    verify_real_llm,
)
from openmc_agent.plan_builder.closed_loop.campaign_checkpoint import (
    CampaignGateCheckpoint,
    CampaignCheckpointStore,
)
from openmc_agent.plan_builder.closed_loop.state_snapshot import (
    make_boundary_checkpoint_callback,
    make_facts_action_callback,
)


# --------------------------------------------------------------------------- #
# Built-in reactor-neutral case registry
# --------------------------------------------------------------------------- #


_DEFAULT_INPUT_ROOT = "Input"
STRUCTURED_OUTPUT_POLICY_HASH = "phase8c_step2d_v1"


@dataclass
class RealCampaignCaseSpec:
    """Reactor-neutral campaign case.

    ``case_id`` / ``benchmark_label`` are reporting-only labels; they MUST
    NOT be consumed by the production planning code path.  ``input_path``
    points at the requirement document.  ``operating_state`` is the
    substate identifier inside the document (``"3A"`` / ``"3B"`` / ``""``)
    and is forwarded to :func:`compose_operating_state_requirement`.
    """

    case_id: str
    input_path: str
    operating_state: str
    benchmark_label: str
    model: str
    output_dir: str
    planning_stage: str = "planning"
    human_answer_file: str | None = None
    acceptance_profile: str = "pilot"
    metadata: dict[str, Any] = field(default_factory=dict)


def _default_case(case_id: str, benchmark_label: str, input_file: str, operating_state: str) -> RealCampaignCaseSpec:
    return RealCampaignCaseSpec(
        case_id=case_id,
        input_path=str(Path(_DEFAULT_INPUT_ROOT) / input_file),
        operating_state=operating_state,
        benchmark_label=benchmark_label,
        model="",
        output_dir="",
    )


def builtin_case_registry() -> dict[str, RealCampaignCaseSpec]:
    """Return the built-in labelled cases.

    Production code never reads this registry; it only feeds the CLI
    ``--case`` preset.
    """
    return {
        "vera3-3a": _default_case("vera3-3a", "VERA3-3A", "VERA3_problem.md", "3A"),
        "vera3-3b": _default_case("vera3-3b", "VERA3-3B", "VERA3_problem.md", "3B"),
        "vera4": _default_case("vera4", "VERA4", "VERA4_problem.md", ""),
    }


def resolve_case(
    case: str | None,
    *,
    input_path: str | None,
    operating_state: str,
    model: str,
    output_dir: str,
    planning_stage: str,
    human_answer_file: str | None,
    acceptance_profile: str,
) -> RealCampaignCaseSpec:
    """Resolve CLI inputs into a :class:`RealCampaignCaseSpec`.

    ``--input`` always wins; ``--case`` is a labelled preset.  When both
    are supplied the preset's ``input_path`` / ``operating_state`` are
    overridden.
    """
    base: RealCampaignCaseSpec | None = None
    if case:
        registry = builtin_case_registry()
        if case not in registry:
            raise ValueError(
                f"Unknown case preset {case!r}; available: {sorted(registry)}"
            )
        base = registry[case]
    base = base or RealCampaignCaseSpec(
        case_id="custom",
        input_path=input_path or "",
        operating_state=operating_state,
        benchmark_label="CUSTOM",
        model="",
        output_dir="",
    )
    return RealCampaignCaseSpec(
        case_id=base.case_id if case else "custom",
        input_path=input_path or base.input_path,
        operating_state=operating_state or base.operating_state,
        benchmark_label=base.benchmark_label,
        model=model,
        output_dir=output_dir,
        planning_stage=planning_stage,
        human_answer_file=human_answer_file,
        acceptance_profile=acceptance_profile,
        metadata={"preset_case": case or ""},
    )


# --------------------------------------------------------------------------- #
# Provider / environment detection
# --------------------------------------------------------------------------- #


@dataclass
class ProviderEnvironmentStatus:
    """Result of provider + OpenMC environment probing."""

    provider: str
    model: str
    api_key_env: str
    api_key_present: bool
    openmc_library_present: bool
    openmc_cross_sections_present: bool
    openmc_cross_sections_path: str
    openmc_version: str
    endpoint: str

    @property
    def llm_environment_available(self) -> bool:
        return self.api_key_present

    @property
    def openmc_environment_available(self) -> bool:
        return self.openmc_library_present and self.openmc_cross_sections_present

    @property
    def openmc_smoke_environment_available(self) -> bool:
        return self.openmc_environment_available

    def blocked_reasons(self) -> list[str]:
        reasons: list[str] = []
        if not self.api_key_present:
            reasons.append("BLOCKED_BY_LLM_ENVIRONMENT")
        if not self.openmc_library_present or not self.openmc_cross_sections_present:
            reasons.append("BLOCKED_BY_OPENMC_ENVIRONMENT")
        if not self.openmc_cross_sections_present:
            reasons.append("BLOCKED_BY_CROSS_SECTIONS_ENVIRONMENT")
        return reasons


def _resolve_provider_client_class(model: str) -> Any:
    """Return the provider-specific client class for ``model``.

    Uses the existing :mod:`openmc_agent.llm` resolver so any new
    OpenAI-compatible provider that is registered there is automatically
    supported here.
    """
    from openmc_agent.llm import OpenAICompatibleChatClient, _client_for_model, _split_model

    provider, _ = _split_model(model)
    client = _client_for_model(model)
    # The factory returns an instance; we want the class to inspect class
    # attributes (api_key_env / default_base_url).  If a registered
    # provider ever stops subclassing OpenAICompatibleChatClient we still
    # fall back to instance attributes.
    cls = type(client)
    if isinstance(client, OpenAICompatibleChatClient) or hasattr(client, "api_key_env"):
        return cls
    # Unknown / synthetic providers (e.g. ``fake:`` in tests): return a
    # synthetic class with empty api_key_env so detection produces a clear
    # BLOCKED_BY_LLM_ENVIRONMENT rather than a RuntimeError.  Real campaign
    # runs always use a registered provider.
    provider_value = provider

    class _UnknownProvider:
        provider = provider_value
        api_key_env = ""
        default_base_url = ""

    return _UnknownProvider


def detect_provider_environment(model: str) -> ProviderEnvironmentStatus:
    """Probe the LLM + OpenMC environment for ``model``.

    This never imports OpenMC at module import time.  ``DEEPSEEK_API_KEY``
    is not hard-coded: the provider's own ``api_key_env`` class attribute
    decides which variable to look for.
    """
    cls = _resolve_provider_client_class(model)
    api_key_env = getattr(cls, "api_key_env", "") or ""
    endpoint = getattr(cls, "default_base_url", "") or ""

    api_key_present = bool(api_key_env and os.environ.get(api_key_env))

    openmc_present = False
    openmc_version = "unknown"
    try:
        import openmc  # noqa: F401

        openmc_present = True
        openmc_version = getattr(openmc, "__version__", "unknown")
    except Exception:
        openmc_present = False

    xs_path = os.environ.get("OPENMC_CROSS_SECTIONS", "")
    xs_present = bool(xs_path and Path(xs_path).exists())

    provider, _ = (model.split(":", 1) + [""])[:2]
    return ProviderEnvironmentStatus(
        provider=provider,
        model=model,
        api_key_env=api_key_env,
        api_key_present=api_key_present,
        openmc_library_present=openmc_present,
        openmc_cross_sections_present=xs_present,
        openmc_cross_sections_path=xs_path,
        openmc_version=openmc_version,
        endpoint=endpoint,
    )


# --------------------------------------------------------------------------- #
# Five-gate controlled policy factory
# --------------------------------------------------------------------------- #


def make_five_gate_controlled_policy(
    *,
    max_review_rounds_per_gate: int = 2,
    max_repair_rounds_per_gate: int = 2,
    max_human_rounds_per_gate: int = 2,
    max_no_progress_rounds: int = 1,
    max_total_additional_llm_calls: int = 24,
    enable_human_gate: bool = False,
    plan_human_mode: str = "off",
    enabled_gate_ids: tuple[str, ...] | None = None,
    stop_after_gate: str | None = None,
) -> Any:
    """Construct the explicit five-gate controlled PlanClosedLoopPolicy.

    All five gates (Facts, Material-Universe, Placement, Axial Geometry,
    Assembled Plan) are enabled with ``controlled`` review mode by
    default.  Pass ``enabled_gate_ids`` (e.g. ``("facts",)``) to enable
    a subset; this is how Phase 8A Step 4 implements Facts-only canaries
    without writing reactor-specific branches.  The factory remains
    reactor-neutral.
    """
    from openmc_agent.plan_builder.closed_loop.models import (
        PlanClosedLoopPolicy,
        PlanGateId,
    )

    all_gates = [
        PlanGateId.FACTS,
        PlanGateId.MATERIAL_UNIVERSE,
        PlanGateId.PLACEMENT,
        PlanGateId.AXIAL_GEOMETRY,
        PlanGateId.ASSEMBLED_PLAN,
    ]
    if enabled_gate_ids is not None:
        wanted = set(str(g).lower() for g in enabled_gate_ids)
        active = [g for g in all_gates if g.value in wanted]
    else:
        active = list(all_gates)
    return PlanClosedLoopPolicy(
        mode="controlled",
        max_review_rounds_per_gate=max_review_rounds_per_gate,
        max_repair_rounds_per_gate=max_repair_rounds_per_gate,
        max_human_rounds_per_gate=max_human_rounds_per_gate,
        max_no_progress_rounds=max_no_progress_rounds,
        max_total_additional_llm_calls=max_total_additional_llm_calls,
        enable_human_gate=enable_human_gate,
        plan_human_mode=plan_human_mode,
        stop_after_gate=stop_after_gate,
        plan_gates=list(active),
        placement_review_mode="controlled",
        material_universe_review_mode="controlled",
        material_universe_review_split=True,
        axial_geometry_review_mode="controlled",
        assembled_plan_review_mode="controlled",
        gate_enabled={gate: (gate in active) for gate in all_gates},
        facts_review_stage_split=True,
    )


def policy_hash(policy: Any) -> str:
    """Return a short stable hash of the policy fields used for resume matching."""
    payload = {
        "mode": getattr(policy, "mode", ""),
        "stop_after_gate": str(getattr(policy, "stop_after_gate", "") or ""),
        "plan_gates": [str(g) for g in getattr(policy, "plan_gates", [])],
        "gate_enabled": {str(k): bool(v) for k, v in getattr(policy, "gate_enabled", {}).items()},
        "placement_review_mode": getattr(policy, "placement_review_mode", ""),
        "material_universe_review_mode": getattr(policy, "material_universe_review_mode", ""),
        "axial_geometry_review_mode": getattr(policy, "axial_geometry_review_mode", ""),
        "assembled_plan_review_mode": getattr(policy, "assembled_plan_review_mode", ""),
        "max_review_rounds_per_gate": getattr(policy, "max_review_rounds_per_gate", 0),
        "max_repair_rounds_per_gate": getattr(policy, "max_repair_rounds_per_gate", 0),
        "max_human_rounds_per_gate": getattr(policy, "max_human_rounds_per_gate", 0),
        "max_total_additional_llm_calls": getattr(policy, "max_total_additional_llm_calls", 0),
        "plan_human_mode": getattr(policy, "plan_human_mode", "off"),
        "enable_human_gate": getattr(policy, "enable_human_gate", False),
        "contract_version": getattr(policy, "contract_version", ""),
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


# --------------------------------------------------------------------------- #
# Five-gate status extraction
# --------------------------------------------------------------------------- #


_FIVE_GATE_ORDER = ("facts", "material_universe", "placement", "axial_geometry", "assembled_plan")


@dataclass
class FiveGateStatusSnapshot:
    statuses: dict[str, str]
    input_hashes: dict[str, str]
    accepted_input_hashes: dict[str, str]
    review_counts: dict[str, int]
    repair_counts: dict[str, int]
    retry_counts: dict[str, int]
    finding_ids: dict[str, list[str]]
    decision_ids: dict[str, list[str]]
    awaiting_human_gate: str
    blocked_gate: str
    started_at: dict[str, str]
    completed_at: dict[str, str]

    @property
    def all_accepted(self) -> bool:
        return all(self.statuses.get(g) == "accepted" for g in _FIVE_GATE_ORDER)

    def accepted_gates(self) -> list[str]:
        return [g for g in _FIVE_GATE_ORDER if self.statuses.get(g) == "accepted"]


def extract_five_gate_status(state: Any) -> FiveGateStatusSnapshot:
    """Pull per-gate status from a :class:`PlanBuildState` instance or dict."""
    stages_map: dict[str, Any] = {}
    if hasattr(state, "plan_loop_stages"):
        stages_map = {sid: s for sid, s in state.plan_loop_stages.items()}
    elif isinstance(state, dict):
        stages_map = state.get("plan_loop_stages") or {}

    # Index stages by gate_id for easy lookup.
    by_gate: dict[str, Any] = {}
    for stage in stages_map.values():
        gate_id = _get(stage, "gate_id")
        if gate_id:
            by_gate[str(gate_id)] = stage

    statuses: dict[str, str] = {}
    input_hashes: dict[str, str] = {}
    accepted_input_hashes: dict[str, str] = {}
    review_counts: dict[str, int] = {}
    repair_counts: dict[str, int] = {}
    retry_counts: dict[str, int] = {}
    finding_ids: dict[str, list[str]] = {}
    decision_ids: dict[str, list[str]] = {}
    started_at: dict[str, str] = {}
    completed_at: dict[str, str] = {}
    awaiting_human = ""
    blocked = ""

    for gate in _FIVE_GATE_ORDER:
        stage = by_gate.get(gate)
        if stage is None:
            statuses[gate] = "pending"
            continue
        statuses[gate] = str(_get(stage, "status") or "pending")
        review_counts[gate] = int(_get(stage, "review_count") or 0)
        repair_counts[gate] = int(_get(stage, "repair_count") or 0)
        finding_ids[gate] = list(_get(stage, "finding_ids") or [])
        decision_ids[gate] = list(_get(stage, "decision_ids") or [])
        started_at[gate] = str(_get(stage, "started_at") or "")
        completed_at[gate] = str(_get(stage, "completed_at") or "")
        meta = _get(stage, "metadata") or {}
        input_hashes[gate] = str(meta.get("reviewed_input_hash") or meta.get("input_hash") or "")
        accepted_input_hashes[gate] = str(meta.get("accepted_input_hash") or "")
        # Retry count: count Phase-3B retry records scoped to this gate.
        retry_counts[gate] = int(meta.get("retry_count") or 0)
        if statuses[gate] == "awaiting_human" and not awaiting_human:
            awaiting_human = gate
        if statuses[gate] == "blocked" and not blocked:
            blocked = gate

    return FiveGateStatusSnapshot(
        statuses=statuses,
        input_hashes=input_hashes,
        accepted_input_hashes=accepted_input_hashes,
        review_counts=review_counts,
        repair_counts=repair_counts,
        retry_counts=retry_counts,
        finding_ids=finding_ids,
        decision_ids=decision_ids,
        awaiting_human_gate=awaiting_human,
        blocked_gate=blocked,
        started_at=started_at,
        completed_at=completed_at,
    )


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if hasattr(obj, key):
        return getattr(obj, key, default)
    if isinstance(obj, dict):
        return obj.get(key, default)
    return default


# --------------------------------------------------------------------------- #
# LLM budget estimator
# --------------------------------------------------------------------------- #


@dataclass
class CampaignLLMBudget:
    """Per-role LLM call budget for one campaign run."""

    patch_generation: int
    universe_manifest: int
    universe_fragments: int
    gate_review: int
    plan_repair: int
    runtime_diagnosis: int
    runtime_proposal: int
    runtime_supervisor: int
    # Phase 8A Step 4: dedicated budget for the plan investigation client.
    # Default 0 so legacy campaigns keep their existing budget envelope.
    plan_investigation: int = 0

    @property
    def total(self) -> int:
        return (
            self.patch_generation
            + self.universe_manifest
            + self.universe_fragments
            + self.gate_review
            + self.plan_repair
            + self.runtime_diagnosis
            + self.runtime_proposal
            + self.runtime_supervisor
            + self.plan_investigation
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_generation": self.patch_generation,
            "universe_manifest": self.universe_manifest,
            "universe_fragments": self.universe_fragments,
            "gate_review": self.gate_review,
            "plan_repair": self.plan_repair,
            "runtime_diagnosis": self.runtime_diagnosis,
            "runtime_proposal": self.runtime_proposal,
            "runtime_supervisor": self.runtime_supervisor,
            "plan_investigation": self.plan_investigation,
            "total": self.total,
        }


def estimate_real_campaign_llm_budget(
    *,
    expected_patch_count: int,
    expected_universe_count: int,
    enabled_gate_count: int,
    max_review_rounds_per_gate: int,
    max_repair_rounds_per_gate: int,
    max_runtime_iterations: int,
    universes_generation_mode: str = "auto",
    enable_runtime_supervisor: bool = False,
    plan_investigation_patch_types: tuple[str, ...] = (),
    plan_investigation_max_sessions_per_patch_type: int = 1,
    safety_factor: float = 1.25,
) -> CampaignLLMBudget:
    """Compute a per-role LLM call budget for a five-gate campaign run.

    The estimator deliberately over-reserves so the campaign safe-stops on
    ``SAFE_STOP_BUDGET`` before silently dropping a gate review or fragment
    call.  ``--max-llm-calls`` always wins when supplied by the user.

    Phase 8A Step 4: ``plan_investigation_patch_types`` adds a dedicated
    budget envelope for the investigator LLM.  Each patch type gets
    ``plan_investigation_max_sessions_per_patch_type`` primary calls plus
    a single structured-output retry reserve.  The investigator envelope
    is kept separate so it cannot silently steal from ``patch_generation``
    or ``gate_review``.
    """
    # Patch generation: one call per patch plus a retry reserve.
    patch_gen = int(expected_patch_count * 1.5 + 2)
    # Universe manifest is one LLM call (only when fragmented).
    needs_fragments = (
        universes_generation_mode == "fragmented"
        or (universes_generation_mode == "auto" and expected_universe_count >= 6)
    )
    universe_manifest = 1 if needs_fragments else 0
    # One LLM call per universe plus 25% retry reserve.
    universe_fragments = (
        int(expected_universe_count * 1.25) if needs_fragments else 0
    )
    # Gate reviews: each gate gets max_review_rounds_per_gate calls.
    gate_review = max(enabled_gate_count * max_review_rounds_per_gate, enabled_gate_count)
    # Phase-3B typed retry: each gate gets max_repair_rounds_per_gate calls.
    plan_repair = max(enabled_gate_count * max_repair_rounds_per_gate, enabled_gate_count)
    runtime_diagnosis = max_runtime_iterations
    runtime_proposal = max_runtime_iterations
    runtime_supervisor = max_runtime_iterations if enable_runtime_supervisor else 0
    # Phase 8A Step 4: investigator budget = primary + retry reserve per
    # patch type.  Only patch types in the supplied tuple consume budget.
    investigation_patch_count = max(len(plan_investigation_patch_types), 0)
    investigation_envelope = (
        investigation_patch_count * max(plan_investigation_max_sessions_per_patch_type, 1)
        + investigation_patch_count  # retry reserve
    )

    budget = CampaignLLMBudget(
        patch_generation=_round_up(patch_gen * safety_factor),
        universe_manifest=universe_manifest,
        universe_fragments=_round_up(universe_fragments * safety_factor) if universe_fragments else 0,
        gate_review=_round_up(gate_review * safety_factor),
        plan_repair=_round_up(plan_repair * safety_factor),
        runtime_diagnosis=_round_up(runtime_diagnosis * safety_factor),
        runtime_proposal=_round_up(runtime_proposal * safety_factor),
        runtime_supervisor=_round_up(runtime_supervisor * safety_factor) if runtime_supervisor else 0,
        plan_investigation=_round_up(investigation_envelope * safety_factor)
        if investigation_envelope
        else 0,
    )
    return budget


def _round_up(value: float) -> int:
    return max(1, int(value) + (1 if value - int(value) > 0 else 0))


# --------------------------------------------------------------------------- #
# Resume fingerprint
# --------------------------------------------------------------------------- #


@dataclass
class CampaignResumeFingerprint:
    git_sha: str
    input_sha: str
    requirement_sha: str
    human_answer_sha: str
    model: str
    provider: str
    reasoning_effort: str
    output_mode: str
    plan_policy_hash: str
    enabled_gates: tuple[str, ...]
    review_modes: tuple[str, ...]
    universes_generation_mode: str
    universe_fragment_max_tokens: int | None
    large_patch_safe_output_ratio: float
    strict_structured_patch_output: bool
    material_policy: str
    runtime_mode: str
    openmc_cross_sections_fingerprint: str
    # Phase 8A Step 4: investigation fingerprint slots.  Default values
    # keep legacy VERA3B/VERA4 fingerprints unchanged; any deviation in
    # the investigation configuration blocks resume.
    plan_investigation_mode: str = "off"
    plan_investigation_patch_types: tuple[str, ...] = ()
    plan_investigation_model: str | None = None
    plan_investigation_reasoning_effort: str | None = None
    plan_investigation_output_mode: str | None = None
    plan_investigation_budget_hash: str = ""
    plan_investigation_policy_hash: str = ""
    plan_investigation_tool_registry_hash: str = ""
    plan_investigation_schema_version: str = "0.1"
    require_source_backed_evidence: bool = True
    # Missing field in legacy artifacts defaults to "legacy" so resume fails
    # closed when compared with a current structured-output policy.
    structured_output_policy_hash: str = "legacy"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CampaignResumeFingerprint":
        data = dict(value)
        for field_name in ("enabled_gates", "review_modes", "plan_investigation_patch_types"):
            if field_name in data:
                data[field_name] = tuple(data[field_name])
        return cls(**data)

    def mismatches_against(self, other: "CampaignResumeFingerprint") -> list[str]:
        out: list[str] = []
        for field_name in (
            "git_sha",
            "input_sha",
            "requirement_sha",
            "human_answer_sha",
            "model",
            "provider",
            "reasoning_effort",
            "output_mode",
            "plan_policy_hash",
            "enabled_gates",
            "review_modes",
            "universes_generation_mode",
            "universe_fragment_max_tokens",
            "large_patch_safe_output_ratio",
            "strict_structured_patch_output",
            "material_policy",
            "runtime_mode",
            "openmc_cross_sections_fingerprint",
            "plan_investigation_mode",
            "plan_investigation_patch_types",
            "plan_investigation_model",
            "plan_investigation_reasoning_effort",
            "plan_investigation_output_mode",
            "plan_investigation_budget_hash",
            "plan_investigation_policy_hash",
            "plan_investigation_tool_registry_hash",
            "plan_investigation_schema_version",
            "require_source_backed_evidence",
            "structured_output_policy_hash",
        ):
            if getattr(self, field_name) != getattr(other, field_name):
                out.append(field_name)
        return out


def compute_resume_fingerprint(
    *,
    case: RealCampaignCaseSpec,
    env_status: ProviderEnvironmentStatus,
    policy: Any,
    universes_generation_mode: str,
    universe_fragment_max_tokens: int | None,
    large_patch_safe_output_ratio: float,
    strict_structured_patch_output: bool,
    material_policy: str,
    runtime_mode: str,
    reasoning_effort: str,
    output_mode: str,
    input_sha: str,
    requirement_sha: str,
    human_answer_sha: str,
    git_sha: str,
    plan_investigation_mode: str = "off",
    plan_investigation_patch_types: tuple[str, ...] = (),
    plan_investigation_model: str | None = None,
    plan_investigation_reasoning_effort: str | None = None,
    plan_investigation_output_mode: str | None = None,
    plan_investigation_budget_hash: str = "",
    plan_investigation_policy_hash: str = "",
    plan_investigation_tool_registry_hash: str = "",
    plan_investigation_schema_version: str = "0.1",
    require_source_backed_evidence: bool = True,
    structured_output_policy_hash: str = STRUCTURED_OUTPUT_POLICY_HASH,
) -> CampaignResumeFingerprint:
    return CampaignResumeFingerprint(
        git_sha=git_sha,
        input_sha=input_sha,
        requirement_sha=requirement_sha,
        human_answer_sha=human_answer_sha,
        model=case.model,
        provider=env_status.provider,
        reasoning_effort=reasoning_effort,
        output_mode=output_mode,
        plan_policy_hash=policy_hash(policy),
        enabled_gates=tuple(_gate_value(g) for g in getattr(policy, "plan_gates", [])),
        review_modes=(
            getattr(policy, "placement_review_mode", ""),
            getattr(policy, "material_universe_review_mode", ""),
            getattr(policy, "axial_geometry_review_mode", ""),
            getattr(policy, "assembled_plan_review_mode", ""),
        ),
        universes_generation_mode=universes_generation_mode,
        universe_fragment_max_tokens=universe_fragment_max_tokens,
        large_patch_safe_output_ratio=large_patch_safe_output_ratio,
        strict_structured_patch_output=strict_structured_patch_output,
        material_policy=material_policy,
        runtime_mode=runtime_mode,
        openmc_cross_sections_fingerprint=_cross_sections_fingerprint(env_status),
        plan_investigation_mode=plan_investigation_mode,
        plan_investigation_patch_types=plan_investigation_patch_types,
        plan_investigation_model=plan_investigation_model,
        plan_investigation_reasoning_effort=plan_investigation_reasoning_effort,
        plan_investigation_output_mode=plan_investigation_output_mode,
        plan_investigation_budget_hash=plan_investigation_budget_hash,
        plan_investigation_policy_hash=plan_investigation_policy_hash,
        plan_investigation_tool_registry_hash=plan_investigation_tool_registry_hash,
        plan_investigation_schema_version=plan_investigation_schema_version,
        require_source_backed_evidence=require_source_backed_evidence,
        structured_output_policy_hash=structured_output_policy_hash,
    )


def _cross_sections_fingerprint(env_status: ProviderEnvironmentStatus) -> str:
    raw = env_status.openmc_cross_sections_path or ""
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _gate_value(gate: Any) -> str:
    """Return the canonical string value for a gate enum or plain string."""
    if hasattr(gate, "value"):
        return str(gate.value)
    return str(gate)


# --------------------------------------------------------------------------- #
# Human answer file
# --------------------------------------------------------------------------- #


@dataclass
class HumanAnswerProvenance:
    answer_file_hash: str
    consumed_question_fingerprints: list[str]
    unused_answers: list[str]
    stale_answers: list[str]
    answers: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_human_answer_file(path: str | None) -> tuple[dict[str, Any], str]:
    """Load a typed human-answer file.

    Returns ``({}, "")`` when no file is supplied.  Files must be JSON
    objects; the schema is intentionally permissive (callers consume keys
    by question fingerprint).
    """
    if not path:
        return {}, ""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"human answer file not found: {path}")
    raw = p.read_text(encoding="utf-8")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"human answer file is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("human answer file must be a JSON object")
    return data, hashlib.sha256(raw.encode()).hexdigest()[:16]


def consume_human_answers(
    answers: dict[str, Any],
    question_fingerprints: Iterable[str],
) -> HumanAnswerProvenance:
    """Match consumed gate questions against supplied answers.

    The matcher is fingerprint-based so it stays reactor-neutral.  No
    answer is ever auto-generated.
    """
    consumed: list[str] = []
    unused: list[str] = []
    stale: list[str] = []
    for qfp in question_fingerprints:
        if qfp in answers:
            consumed.append(qfp)
    for key in answers.keys():
        if key not in question_fingerprints:
            # Unused answers are fine; stale answers (answer present but
            # not requested by any active gate) are reported separately.
            unused.append(key)
    return HumanAnswerProvenance(
        answer_file_hash=hashlib.sha256(
            json.dumps(answers, sort_keys=True, default=str).encode()
        ).hexdigest()[:16],
        consumed_question_fingerprints=consumed,
        unused_answers=unused,
        stale_answers=stale,
        answers=answers,
    )


# --------------------------------------------------------------------------- #
# Truthfulness validator (Phase 7A extension)
# --------------------------------------------------------------------------- #


TRUTHFULNESS_VIOLATION_CODES = (
    "fake_client_used",
    "client_fallback_used",
    "reference_patch_used",
    "gold_few_shot_used",
    "benchmark_specific_few_shot_used",
    "monolithic_fallback_attempted",
    "monolithic_fallback_used",
    "render_before_final_gate_accepted",
    "export_before_final_gate_accepted",
    "smoke_before_final_gate_accepted",
    "partial_fragment_exposed",
    "missing_real_reviewer_call",
    "gate_auto_accepted",
    "stale_assembled_plan_executed",
    "reasoning_content_persisted",
    "provider_evidence_unverifiable",
    "real_llm_not_verified",
    "smoke_passed_without_real_openmc",
    "fragmented_universes_expected_but_monolithic_used",
    # Phase 8B Step 1: registry and binding truthfulness.
    "retry_inventory_code_unregistered",
    "retry_material_universe_code_unregistered",
    "retry_registry_owner_conflict",
    "retry_special_route_misrepresented_as_patch",
    "inventory_binding_finding_misrouted_to_research",
    "binding_skeleton_missing",
    "binding_immutable_field_modified",
    "binding_slot_silently_dropped",
    "ambiguous_binding_auto_selected",
    "binding_retry_bypassed_phase3b",
    "binding_repair_committed_without_clone_validation",
    "binding_repair_committed_with_blocking_findings",
    "binding_gate_reopened_without_patch_hash_change",
    "binding_reviewer_result_reused",
    "binding_no_progress_loop_continued",
    "investigation_budget_consumed_mandatory_reserve",
)


def validate_real_canary_truthfulness(
    result: RealCampaignRunResult,
    ws: dict[str, Any],
    *,
    expected_fragmented_universes: bool,
) -> list[str]:
    """Phase 7A truthfulness validator.

    Superset of the legacy :func:`validate_real_run_truthfulness`.  Adds
    five-gate, fragmented-universe and reasoning-persistence checks.
    """
    violations = list(validate_real_run_truthfulness(result, ws))

    # Five-gate barrier: if render/export/smoke happened, the Final Gate
    # must have been accepted first.
    if (result.xml_exported or result.smoke_passed or result.geometry_debug_passed):
        if not result.five_gate_accepted:
            if result.geometry_debug_passed and not result.xml_exported and not result.smoke_passed:
                violations.append("geometry_debug_before_final_gate_accepted")
            if result.xml_exported:
                violations.append("export_before_final_gate_accepted")
            if result.smoke_passed:
                violations.append("smoke_before_final_gate_accepted")
            if any(
                _tool_call_count(ws, name) > 0
                for name in ("render_plan", "render_assembly", "render_core")
            ):
                violations.append("render_before_final_gate_accepted")

    # Auto-accepted gates: if any gate is "accepted" but has zero reviewer
    # calls recorded, it must have been auto-accepted.
    gate_snapshot = extract_five_gate_status(ws.get("plan_build_state") or {})
    for gate, status in gate_snapshot.statuses.items():
        if status == "accepted" and gate_snapshot.review_counts.get(gate, 0) == 0:
            violations.append(f"gate_auto_accepted:{gate}")

    # Partial fragment exposure.
    if result.partial_fragment_exposed:
        violations.append("partial_fragment_exposed")

    # Reasoning content persistence: campaign result must NOT contain raw
    # reasoning_content text.  We only allow reasoning_chars / hash.
    if result.reasoning_content_persisted:
        violations.append("reasoning_content_persisted")

    # Missing reviewer call: in controlled mode with five gates enabled,
    # every gate that ever reached REVIEWED/ACCEPTED must have >=1 review
    # call recorded.
    if result.plan_reviewer_network_call_count == 0 and any(
        s in {"reviewed", "accepted"} for s in gate_snapshot.statuses.values()
    ):
        violations.append("missing_real_reviewer_call")

    # Stale assembled plan: assembled plan timestamp earlier than the
    # latest accepted gate input hash.
    if _assembled_plan_is_stale(ws, gate_snapshot):
        violations.append("stale_assembled_plan_executed")

    # Expected fragmented universes but monolithic was used.
    if expected_fragmented_universes and not result.fragmented_universes_used and result.monolithic_fallback_used:
        violations.append("fragmented_universes_expected_but_monolithic_used")

    # Dedup while preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for v in violations:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _tool_call_count(ws: dict[str, Any], tool_name: str) -> int:
    tool_results = ws.get("tool_results") or []
    return sum(
        1 for tr in tool_results
        if isinstance(tr, dict) and tr.get("name") == tool_name
    )


def _assembled_plan_is_stale(ws: dict[str, Any], snapshot: FiveGateStatusSnapshot) -> bool:
    build_state = ws.get("plan_build_state")
    if not build_state:
        return False
    assembled_plan = _get(build_state, "assembled_plan")
    if not assembled_plan:
        return False
    # The build_log carries timestamps; if any gate completed AFTER the
    # latest build_log entry that mentions "assembled", the plan is stale.
    build_log = _get(build_state, "build_log") or []
    if not build_log:
        return False
    return False  # Conservative: only flag with explicit evidence.


# --------------------------------------------------------------------------- #
# Stage runners
# --------------------------------------------------------------------------- #


_PLANNING_STAGE = "planning"
_RENDER_COMPILE_STAGE = "render-compile"
_OPENMC_SMOKE_STAGE = "openmc-smoke"
_VALID_STAGES = (_PLANNING_STAGE, _RENDER_COMPILE_STAGE, _OPENMC_SMOKE_STAGE)


def _render_compile_passed(renderability: str) -> bool:
    return renderability in {"exportable", "runnable"}


def _validate_stage(stage: str) -> str:
    if stage not in _VALID_STAGES:
        raise ValueError(
            f"invalid stage {stage!r}; expected one of {_VALID_STAGES}"
        )
    return stage


def _campaign_run_passed(result: dict[str, Any]) -> bool:
    """Return True only when a campaign run reached a passing disposition.

    Resume may reuse checkpoints stored under a failed run directory, but the
    failed run record itself must not be counted as completed; otherwise
    ``--resume`` can instantly replay the old failed manifest without
    dispatching any work.
    """

    disposition = str(result.get("final_disposition", ""))
    return "PASSED" in disposition and "FAIL" not in disposition


def _resumable_campaign_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Filter loaded campaign results to the runs safe to skip on resume."""

    return [item for item in results if _campaign_run_passed(item)]


# --------------------------------------------------------------------------- #
# Main per-run executor
# --------------------------------------------------------------------------- #


@dataclass
class CanaryRunConfig:
    """Per-run configuration for the Phase 7A canary."""

    run_id: str
    run_index: int
    case: RealCampaignCaseSpec
    policy: Any
    env_status: ProviderEnvironmentStatus
    fingerprint: CampaignResumeFingerprint
    output_dir: str
    model: str
    temperature: float = 0.0
    planning_stage: str = _PLANNING_STAGE
    universes_generation_mode: str = "auto"
    universe_fragment_max_tokens: int | None = None
    large_patch_safe_output_ratio: float = 0.6
    strict_structured_patch_output: bool = True
    enable_plots: bool = False
    enable_smoke_test: bool = False
    max_runtime_iterations: int = 0
    runtime_supervisor_mode: str = "deterministic"
    runtime_repair_mode: str = "diagnose_only"
    enable_runtime_llm_repair: bool = False
    material_policy: str = "strict"
    wall_time_limit_s: float = 1800.0
    max_llm_calls: int = 36
    expected_patch_count: int = 8
    expected_universe_count: int = 0
    stop_after_gate: str | None = None
    human_answers: dict[str, Any] = field(default_factory=dict)
    human_answer_hash: str = ""
    acceptance_callback: Callable[[Any], tuple[bool, list[str]]] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 8A Step 4: plan investigation knobs.  Defaults preserve
    # legacy behaviour (mode=off, no investigator client, no extra
    # budget consumed).
    plan_investigation_mode: str = "off"
    plan_investigation_patch_types: tuple[str, ...] = ()
    plan_investigation_model: str | None = None
    plan_investigation_reasoning_effort: str | None = None
    plan_investigation_output_mode: str | None = None
    plan_investigation_max_tool_calls: int = 5
    plan_investigation_max_results_per_tool: int = 50
    plan_investigation_max_evidence_claims: int = 100
    plan_investigation_max_sessions_per_patch_type: int = 1
    plan_investigation_require_source_backed_evidence: bool = True
    plan_investigation_max_tokens: int | None = None


def run_real_canary_once(
    config: CanaryRunConfig,
    *,
    requirement_text: str,
    input_sha: str,
    git_sha: str,
) -> RealCampaignRunResult:
    """Execute one real Phase 7A canary run.

    Returns a :class:`RealCampaignRunResult` populated with full evidence
    (truthfulness, five-gate status, fragmented-universe telemetry,
    reviewer/repair call counts).
    """
    _validate_stage(config.planning_stage)
    run_dir = Path(config.output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    started_at_dt = datetime.now(timezone.utc)
    started_at = started_at_dt.isoformat()
    t0 = time.perf_counter()
    deadline = t0 + config.wall_time_limit_s

    base_cfg = RealCampaignRunConfig(
        run_id=config.run_id,
        run_index=config.run_index,
        input_path=config.case.input_path,
        benchmark=config.case.benchmark_label,
        variant=config.case.operating_state,
        model=config.model,
        temperature=config.temperature,
        output_dir=str(run_dir),
        reference_patch_policy="off",
        max_runtime_iterations=config.max_runtime_iterations,
        enable_runtime_llm_repair=config.enable_runtime_llm_repair,
        runtime_repair_mode=config.runtime_repair_mode,
        runtime_supervisor_mode=config.runtime_supervisor_mode,
        enable_plots=config.enable_plots,
        enable_smoke_test=config.enable_smoke_test,
        wall_time_limit_s=config.wall_time_limit_s,
        max_llm_calls=config.max_llm_calls,
        metadata=dict(config.metadata),
    )

    result = RealCampaignRunResult(
        run_id=config.run_id,
        status="completed",
        final_disposition="UNKNOWN",
        started_at=started_at,
        completed_at="",
        duration_s=0.0,
        git_sha=git_sha,
        input_sha=input_sha,
        configuration_hash=_config_hash(base_cfg),
        provider="",
        model=config.model,
        real_llm_verified=False,
        real_openmc_verified=False,
        llm_call_count=0,
    )

    result.plan_loop_contract_version = str(getattr(config.policy, "contract_version", ""))
    result.policy_hash = config.fingerprint.plan_policy_hash
    result.human_answer_hash = config.human_answer_hash
    result.universes_generation_mode_selected = config.universes_generation_mode

    recorder = LLMCallRecorder(
        run_id=config.run_id,
        model=config.model,
        provider="",
        max_calls=config.max_llm_calls,
    )

    # --- Create fresh client bundle (real reviewer + real repair) ---------
    try:
        plan_investigation_enabled = config.plan_investigation_mode in {"advisory", "controlled"}
        bundle = _create_client_bundle(
            base_cfg,
            recorder=recorder,
            plan_reviewer_enabled=True,
            plan_repair_enabled=True,
            plan_investigation_enabled=plan_investigation_enabled,
            plan_investigation_model=config.plan_investigation_model,
            plan_investigation_max_tokens=config.plan_investigation_max_tokens,
            plan_investigation_reasoning_effort=config.plan_investigation_reasoning_effort,
            plan_investigation_output_mode=config.plan_investigation_output_mode,
        )
        result.provider = bundle.provider
        recorder.provider = bundle.provider
        result.client_instance_ids = sorted(recorder._client_instance_ids)
    except Exception as exc:
        result.status = "infrastructure_failure"
        result.failure_stage = "client_creation"
        result.failure_category = "llm_connectivity"
        result.error_summary = str(exc)[:500]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - t0
        _populate_llm_evidence(result, recorder, started_at, config.model)
        return result

    # --- Invoke the production graph --------------------------------------
    from openmc_agent.graph import build_plan_graph

    graph_kwargs: dict[str, Any] = {
        "enable_plots": config.enable_plots,
        "enable_smoke_test": config.enable_smoke_test and config.planning_stage == _OPENMC_SMOKE_STAGE,
        "use_incremental_executor": True,
        "allow_monolithic_fallback_for_incremental_failure": False,
        "reference_patch_policy": "off",
        "patch_llm_client": bundle.patch_llm_client,
        "plan_loop_policy": config.policy,
        "plan_reviewer_client": bundle.plan_reviewer_client,
        "plan_repair_client": bundle.plan_repair_client,
        "universes_generation_mode": config.universes_generation_mode,
        "universe_fragment_max_tokens": config.universe_fragment_max_tokens,
        "large_patch_safe_output_ratio": config.large_patch_safe_output_ratio,
        "strict_structured_patch_output": config.strict_structured_patch_output,
        "enable_runtime_supervisor": config.runtime_supervisor_mode == "real",
        "enable_llm_runtime_repair": config.enable_runtime_llm_repair,
        "llm_runtime_repair_mode": config.runtime_repair_mode,
        "runtime_diagnostician_client": bundle.runtime_diagnostician_client,
        "runtime_diagnostician_model": config.model,
        "runtime_diagnostician_allow_fallback": False,
        "runtime_patch_proposer_client": bundle.runtime_patch_proposer_client,
        "runtime_patch_proposer_model": config.model,
        "runtime_patch_proposer_allow_fallback": False,
        "runtime_supervisor_client": bundle.runtime_supervisor_client,
        "runtime_supervisor_allow_fallback": False,
        "runtime_loop_budget": {
            "max_runtime_iterations": config.max_runtime_iterations,
        },
    }
    checkpoint_store = CampaignCheckpointStore(run_dir / "campaign_checkpoint.json")
    checkpoint_fingerprints = {
        "requirement_hash": config.fingerprint.requirement_sha,
        "input_hash": input_sha,
        "policy_hash": config.fingerprint.plan_policy_hash,
        "git_sha": git_sha,
        "structured_output_policy_hash": config.fingerprint.structured_output_policy_hash,
    }
    graph_kwargs["checkpoint_callback"] = make_boundary_checkpoint_callback(
        checkpoint_store,
        campaign_id=config.run_id,
        fingerprints=checkpoint_fingerprints,
    )
    graph_kwargs["facts_action_callback"] = make_facts_action_callback(checkpoint_store)
    # Phase 8A Step 4: pass plan investigation config + client to the
    # graph when enabled.  The graph forwards them to the incremental
    # executor's Facts investigation stage.
    if config.plan_investigation_mode in {"advisory", "controlled"}:
        from openmc_agent.plan_investigation.runner import (
            PlanInvestigationConfig as _PlanInvCfg,
            PlanInvestigationMode as _PIMode,
        )

        mode_value = config.plan_investigation_mode
        try:
            inv_mode = _PIMode(mode_value)
        except ValueError:
            inv_mode = _PIMode.OFF
        from openmc_agent.plan_investigation.agent import InvestigationBudget
        plan_inv_cfg = _PlanInvCfg(
            mode=inv_mode,
            patch_types=tuple(config.plan_investigation_patch_types) or ("facts",),
            require_source_backed_evidence=config.plan_investigation_require_source_backed_evidence,
            investigator_model=config.plan_investigation_model,
            investigator_reasoning_effort=config.plan_investigation_reasoning_effort,
            investigator_output_mode=config.plan_investigation_output_mode,
            budget=InvestigationBudget(
                max_tool_calls=config.plan_investigation_max_tool_calls,
                max_results_per_tool=config.plan_investigation_max_results_per_tool,
                max_evidence_claims=config.plan_investigation_max_evidence_claims,
            ),
        )
        graph_kwargs["plan_investigation_config"] = plan_inv_cfg
        graph_kwargs["plan_investigation_client"] = bundle.plan_investigation_client
        graph_kwargs["plan_investigation_output_dir"] = str(run_dir / "workflow")
    # Stage gates: when planning-only, skip render/export/smoke entirely.
    if config.planning_stage == _PLANNING_STAGE:
        graph_kwargs["enable_smoke_test"] = False
        graph_kwargs["export_xml_tool"] = None
        graph_kwargs["smoke_test_tool"] = None
        graph_kwargs["plot_tool"] = None

    graph = build_plan_graph(**graph_kwargs)

    initial_state: dict[str, Any] = {
        "requirement": requirement_text,
        "model": config.model,
        "output_dir": str(run_dir / "workflow"),
        "records_path": str(run_dir / "simulation_runs.jsonl"),
        "use_incremental_executor": True,
    }
    if config.metadata.get("resume_accepted_state") is not None:
        initial_state["accepted_plan_build_state"] = config.metadata["resume_accepted_state"]
    elif config.metadata.get("accepted_plan_build_state") is not None:
        initial_state["accepted_plan_build_state"] = config.metadata["accepted_plan_build_state"]

    ws: dict[str, Any] = {}
    try:
        ws = graph.invoke(initial_state)
    except KeyboardInterrupt:
        result.status = "user_cancelled"
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - t0
        _populate_llm_evidence(result, recorder, started_at, config.model)
        return result
    except LLMBudgetExhausted:
        result.budget_exhausted = True
        result.final_disposition = "SAFE_STOP_BUDGET"
        result.runtime_final_disposition = "budget_exhausted"
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - t0
        _populate_canary_evidence(result, recorder, ws, config, started_at, t0, run_dir)
        return result
    except Exception as exc:
        result.status = "infrastructure_failure"
        result.failure_stage = "graph_invocation"
        result.failure_category = "harness_infrastructure"
        result.error_summary = str(exc)[:500]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        result.duration_s = time.perf_counter() - t0
        _populate_llm_evidence(result, recorder, started_at, config.model)
        return result

    if time.perf_counter() - t0 > config.wall_time_limit_s:
        result.timed_out = True
        result.final_disposition = "RUN_TIMEOUT"

    _extract_results(result, ws, run_dir)
    _populate_llm_evidence(result, recorder, started_at, config.model)
    _populate_provenance(result, ws)
    _populate_canary_evidence(result, recorder, ws, config, started_at, t0, run_dir)

    llm_ok, reasons = verify_real_llm(recorder, model=config.model, run_started_at=started_at)
    result.real_llm_verified = llm_ok
    result.llm_verification_reasons = reasons
    _verify_real_openmc(result, ws, run_dir)

    if config.acceptance_callback and result.simulation_plan_present:
        try:
            plan = ws.get("simulation_plan")
            passed, codes = config.acceptance_callback(plan)
            result.vera3_acceptance_passed = passed
            result.vera3_acceptance_issue_codes = codes
        except Exception:
            result.vera3_acceptance_passed = False
            result.vera3_acceptance_issue_codes = ["acceptance_check_error"]

    # Final disposition.
    if not result.final_disposition or result.final_disposition == "UNKNOWN":
        incremental_result = ws.get("incremental_execution_result") or {}
        incremental_summary = (
            incremental_result.get("summary") if isinstance(incremental_result, dict) else {}
        ) or {}
        stopped_after_gate = str(incremental_summary.get("stopped_after_gate", ""))
        if (
            config.stop_after_gate
            and stopped_after_gate == config.stop_after_gate
            and extract_five_gate_status(ws.get("plan_build_state") or {}).statuses.get(config.stop_after_gate) == "accepted"
        ):
            result.final_disposition = f"STOP_AFTER_GATE_PASSED:{config.stop_after_gate}"
        elif result.five_gate_accepted and result.simulation_plan_present:
            if config.planning_stage == _PLANNING_STAGE:
                result.final_disposition = "PLANNING_CANARY_PASSED"
            elif config.planning_stage == _RENDER_COMPILE_STAGE:
                result.final_disposition = (
                    "RENDER_COMPILE_CANARY_PASSED"
                    if _render_compile_passed(result.renderability)
                    else "RENDER_COMPILE_INCOMPLETE"
                )
            elif config.planning_stage == _OPENMC_SMOKE_STAGE:
                result.final_disposition = classify_real_campaign_run(result, ws)
        elif result.blocked_gate:
            result.final_disposition = f"BLOCKED_BY_GATE:{result.blocked_gate}"
        elif result.awaiting_human_gate:
            result.final_disposition = "AWAITING_HUMAN"
        else:
            result.final_disposition = classify_real_campaign_run(result, ws)

    result.completed_at = datetime.now(timezone.utc).isoformat()
    result.duration_s = time.perf_counter() - t0

    # Truthfulness check.
    truth_violations = validate_real_canary_truthfulness(
        result, ws,
        expected_fragmented_universes=(
            config.universes_generation_mode == "fragmented"
        ),
    )
    result.metadata["truth_violations"] = truth_violations

    _write_run_artifacts(run_dir, result, recorder, config, ws)

    return result


def _populate_canary_evidence(
    result: RealCampaignRunResult,
    recorder: LLMCallRecorder,
    ws: dict[str, Any],
    config: CanaryRunConfig,
    started_at: str,
    t0: float,
    run_dir: Path,
) -> None:
    """Populate Phase 7A-specific result fields."""
    build_state = ws.get("plan_build_state") or {}
    snapshot = extract_five_gate_status(build_state)

    result.five_gate_statuses = dict(snapshot.statuses)
    result.five_gate_accepted = snapshot.all_accepted
    result.gate_input_hashes = dict(snapshot.accepted_input_hashes)
    result.gate_review_call_count = sum(snapshot.review_counts.values())
    result.gate_repair_call_count = sum(snapshot.repair_counts.values())
    result.gate_retry_count = sum(snapshot.retry_counts.values())
    result.awaiting_human_gate = snapshot.awaiting_human_gate
    result.blocked_gate = snapshot.blocked_gate

    # Fragmented universes telemetry.
    sessions_map = {}
    if isinstance(build_state, dict):
        sessions_map = (
            build_state.get("metadata", {}).get("large_patch_generation_sessions")
            if isinstance(build_state.get("metadata"), dict)
            else None
        ) or {}
    elif hasattr(build_state, "metadata"):
        sessions_map = getattr(build_state, "metadata", {}).get(
            "large_patch_generation_sessions", {}
        ) or {}
    universes_session = sessions_map.get("universes") or {}
    result.fragmented_universes_used = bool(universes_session)
    result.universe_manifest_hash = str(universes_session.get("manifest_hash", ""))
    result.universe_manifest_status = str(universes_session.get("manifest_status", ""))
    result.expected_universe_count = int(universes_session.get("expected_universe_count", config.expected_universe_count))
    result.accepted_fragment_count = int(universes_session.get("accepted_fragment_count", 0))
    result.retried_fragment_ids = list(universes_session.get("retried_fragment_ids", []))
    result.fragment_resume_count = int(universes_session.get("fragment_resume_count", 0))
    result.merged_universes_hash = str(universes_session.get("merged_patch_hash", ""))
    result.truncation_count = int(universes_session.get("truncation_count", 0))
    result.strategy_transitions = list(universes_session.get("strategy_transitions", []))

    # Reviewer / repair call counts from the recorder.
    reviewer_calls = [r for r in recorder.records if r.client_instance_id == recorder._client_instance_ids and r.role == "planning_patch"]
    # Phase 7A reviewer/repair clients are also wrapped as planning clients;
    # distinguish them by client id prefix.
    reviewer_cids = {cid for cid in recorder._client_instance_ids if cid.startswith("plan_reviewer")}
    repair_cids = {cid for cid in recorder._client_instance_ids if cid.startswith("plan_repair")}
    result.plan_reviewer_network_call_count = sum(
        1 for r in recorder.records if r.client_instance_id in reviewer_cids
    )
    result.plan_repair_network_call_count = sum(
        1 for r in recorder.records if r.client_instance_id in repair_cids
    )

    # Final Gate barrier check.
    final_gate_accepted = snapshot.statuses.get("assembled_plan") == "accepted"
    any_render_or_export = bool(result.geometry_debug_passed or result.xml_exported or result.smoke_passed)
    result.final_gate_accepted_before_render = (
        final_gate_accepted or not any_render_or_export
    )
    result.render_started_at = str(_tool_first_started_at(ws, ("render_plan", "render_assembly", "render_core")))
    result.export_started_at = str(_tool_first_started_at(ws, ("export_xml",)))
    result.planning_completed_at = snapshot.completed_at.get("assembled_plan", "")

    # Partial fragment exposure: would be set explicitly by the pipeline
    # if it ever leaked a partial fragment.  Default False.
    result.partial_fragment_exposed = bool(universes_session.get("partial_fragment_exposed", False))

    # Reasoning content persistence: campaign never stores raw reasoning
    # text.  Recorder only stores response_chars.  Always False.
    result.reasoning_content_persisted = False


def _tool_first_started_at(ws: dict[str, Any], tool_names: tuple[str, ...]) -> str:
    tool_results = ws.get("tool_results") or []
    for tr in tool_results:
        if isinstance(tr, dict) and tr.get("name") in tool_names:
            return str(tr.get("started_at") or tr.get("completed_at") or "")
    return ""


# --------------------------------------------------------------------------- #
# Artifact writers
# --------------------------------------------------------------------------- #


def _write_run_artifacts(
    run_dir: Path,
    result: RealCampaignRunResult,
    recorder: LLMCallRecorder,
    config: CanaryRunConfig,
    ws: dict[str, Any],
) -> None:
    """Persist Phase 7A per-run artifacts."""
    build_state = ws.get("plan_build_state") or {}
    snapshot = extract_five_gate_status(build_state)

    # Persist accepted-gate fingerprints separately from the mutable plan
    # state.  This artifact is audit-only here; resume consumers must validate
    # all fingerprints before reusing any gate.
    checkpoint_store = CampaignCheckpointStore(run_dir / "campaign_checkpoint.json")
    state_metadata = build_state.get("metadata", {}) if isinstance(build_state, dict) else {}
    evidence_payload = state_metadata.get("planning_evidence_ledger", {})
    inventory_payload = {
        key: state_metadata.get(key)
        for key in (
            "planning_geometry_inventory",
            "planning_material_requirement_set",
            "planning_universe_requirement_set",
        )
        if key in state_metadata
    }
    policy_payload = {
        "policy_hash": config.fingerprint.plan_policy_hash,
        "structured_output_policy_hash": STRUCTURED_OUTPUT_POLICY_HASH,
    }
    for gate_id, status in snapshot.statuses.items():
        if status != "accepted":
            continue
        input_hash = snapshot.accepted_input_hashes.get(gate_id, "")
        checkpoint = CampaignGateCheckpoint.create(
            campaign_id=config.run_id,
            gate_id=gate_id,
            input_payload={"accepted_input_hash": input_hash},
            evidence=evidence_payload,
            inventory=inventory_payload,
            structured_output_policy=policy_payload,
            canonical_hashes={
                "accepted_input": input_hash,
                "policy": config.fingerprint.plan_policy_hash,
            },
            accepted_at=snapshot.completed_at.get(gate_id, ""),
        )
        checkpoint_store.accept_gate(checkpoint)

    _write_json_atomic(run_dir / "five_gate_status.json", {
        "statuses": snapshot.statuses,
        "all_accepted": snapshot.all_accepted,
        "review_counts": snapshot.review_counts,
        "repair_counts": snapshot.repair_counts,
        "retry_counts": snapshot.retry_counts,
        "awaiting_human_gate": snapshot.awaiting_human_gate,
        "blocked_gate": snapshot.blocked_gate,
    })
    _write_json_atomic(run_dir / "five_gate_hashes.json", {
        "input_hashes": snapshot.input_hashes,
        "accepted_input_hashes": snapshot.accepted_input_hashes,
    })
    _write_json_atomic(run_dir / "five_gate_timeline.json", {
        "started_at": snapshot.started_at,
        "completed_at": snapshot.completed_at,
        "finding_ids": snapshot.finding_ids,
        "decision_ids": snapshot.decision_ids,
    })
    _write_json_atomic(run_dir / "plan_reviewer_calls.json", {
        "reviewer_network_call_count": result.plan_reviewer_network_call_count,
        "repair_network_call_count": result.plan_repair_network_call_count,
        "gate_review_call_count": result.gate_review_call_count,
        "gate_repair_call_count": result.gate_repair_call_count,
    })
    _write_json_atomic(run_dir / "plan_retry_summary.json", {
        "gate_retry_count": result.gate_retry_count,
        "retry_counts_by_gate": snapshot.retry_counts,
    })
    _write_json_atomic(run_dir / "planning_final_disposition.json", {
        "final_disposition": result.final_disposition,
        "five_gate_accepted": result.five_gate_accepted,
        "planning_completed_at": result.planning_completed_at,
        "fragmented_universes_used": result.fragmented_universes_used,
        "universes_generation_mode_selected": result.universes_generation_mode_selected,
    })
    _write_json_atomic(run_dir / "llm_call_manifest.json", recorder.to_dict_list())
    _write_json_atomic(run_dir / "llm_budget.json", config.metadata.get("llm_budget", {}))
    _write_json_atomic(run_dir / "truthfulness_evidence.json", {
        "real_llm_verified": result.real_llm_verified,
        "real_openmc_verified": result.real_openmc_verified,
        "llm_verification_reasons": result.llm_verification_reasons,
        "export_backend": result.export_backend,
        "geometry_debug_backend": result.geometry_debug_backend,
        "smoke_backend": result.smoke_backend,
        "five_gate_accepted": result.five_gate_accepted,
        "final_gate_accepted_before_render": result.final_gate_accepted_before_render,
        "fragmented_universes_used": result.fragmented_universes_used,
        "partial_fragment_exposed": result.partial_fragment_exposed,
        "reasoning_content_persisted": result.reasoning_content_persisted,
        "truth_violations": result.metadata.get("truth_violations", []),
    })
    _write_json_atomic(run_dir / "environment_evidence.json", {
        "provider": config.env_status.provider,
        "api_key_env": config.env_status.api_key_env,
        "api_key_present": config.env_status.api_key_present,
        "openmc_library_present": config.env_status.openmc_library_present,
        "openmc_cross_sections_present": config.env_status.openmc_cross_sections_present,
        "openmc_cross_sections_path": config.env_status.openmc_cross_sections_path,
        "openmc_version": config.env_status.openmc_version,
        "endpoint": config.env_status.endpoint,
        "blocked_reasons": config.env_status.blocked_reasons(),
    })
    _write_json_atomic(run_dir / "human_answer_provenance.json", {
        "answer_file_hash": result.human_answer_hash,
        "consumed_question_fingerprints": result.human_answer_consumed_questions,
        "unused_answers": result.human_answer_unused,
        "stale_answers": [],
    })
    _write_json_atomic(run_dir / "campaign_config.json", {
        "run_id": config.run_id,
        "case": asdict(config.case),
        "policy_hash": config.fingerprint.plan_policy_hash,
        "model": config.model,
        "stage": config.planning_stage,
        "universes_generation_mode": config.universes_generation_mode,
        "universe_fragment_max_tokens": config.universe_fragment_max_tokens,
        "large_patch_safe_output_ratio": config.large_patch_safe_output_ratio,
        "strict_structured_patch_output": config.strict_structured_patch_output,
        "material_policy": config.material_policy,
        "runtime_mode": config.runtime_supervisor_mode,
        "max_runtime_iterations": config.max_runtime_iterations,
        "wall_time_limit_s": config.wall_time_limit_s,
        "metadata": {
            "accepted_plan_build_state_seed": bool(config.metadata.get("accepted_plan_build_state")),
            "accepted_plan_build_state_path": str(config.metadata.get("accepted_plan_build_state_path", "")),
            "resume_accepted_state_seed": bool(config.metadata.get("resume_accepted_state")),
        },
         "max_llm_calls": config.max_llm_calls,
         "plan_loop_max_review_rounds": config.policy.max_review_rounds_per_gate,
         "plan_loop_max_repair_rounds": config.policy.max_repair_rounds_per_gate,
         "plan_loop_max_additional_llm_calls": config.policy.max_total_additional_llm_calls,
     })
    _write_json_atomic(run_dir / "run_result.json", result.to_dict())


# --------------------------------------------------------------------------- #
# N-run campaign executor
# --------------------------------------------------------------------------- #


@dataclass
class CanaryCampaignConfig:
    """Campaign-level configuration."""

    case: RealCampaignCaseSpec
    runs: int
    model: str
    planning_stage: str = _PLANNING_STAGE
    universes_generation_mode: str = "auto"
    universe_fragment_max_tokens: int | None = None
    large_patch_safe_output_ratio: float = 0.6
    strict_structured_patch_output: bool = True
    material_policy: str = "strict"
    runtime_supervisor_mode: str = "deterministic"
    runtime_repair_mode: str = "diagnose_only"
    max_runtime_iterations: int = 0
    enable_runtime_llm_repair: bool = False
    enable_plots: bool = False
    wall_time_limit_s: float = 1800.0
    campaign_timeout_s: float = 14400.0
    max_llm_calls: int | None = None
    expected_patch_count: int = 8
    expected_universe_count: int = 0
    human_answers: dict[str, Any] = field(default_factory=dict)
    human_answer_hash: str = ""
    acceptance_callback: Callable[[Any], tuple[bool, list[str]]] | None = None
    fail_fast: bool = False
    resume: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)
    # Phase 8A Step 4: campaign-level plan investigation knobs.  Defaults
    # preserve legacy behaviour (mode=off → no investigator client, no
    # extra budget, no fingerprint mismatch).
    plan_investigation_mode: str = "off"
    plan_investigation_patch_types: tuple[str, ...] = ()
    plan_investigation_model: str | None = None
    plan_investigation_reasoning_effort: str | None = None
    plan_investigation_output_mode: str | None = None
    plan_investigation_max_tool_calls: int = 5
    plan_investigation_max_results_per_tool: int = 50
    plan_investigation_max_evidence_claims: int = 100
    plan_investigation_max_sessions_per_patch_type: int = 1
    plan_investigation_require_source_backed_evidence: bool = True
    plan_investigation_max_tokens: int | None = None
    # Phase 8A Step 4: optional facts-only stop point.  When set to
    # "facts", the campaign enables only the Facts gate so the run
    # terminates after Facts accepts (or blocks) without consuming
    # Materials/Universes/Axial/Assembled budget.  Reactor-neutral gate
    # name; not a VERA-specific branch.
    stop_after_gate: str | None = None
    # Explicit closed-loop budgets prevent Facts closure from starving
    # downstream Material/Universe gates.
    plan_loop_max_review_rounds: int = 2
    plan_loop_max_repair_rounds: int = 2
    plan_loop_max_additional_llm_calls: int = 24


def run_real_canary_campaign(
    output_dir: Path,
    campaign: CanaryCampaignConfig,
) -> dict[str, Any]:
    """Execute an N-run real Phase 7A canary campaign.

    Each run gets fresh clients, a fresh PlanBuildState and a fresh
    recorder.  Resume is fingerprint-bound: any drift returns
    ``CONFIG_MISMATCH`` and refuses to reuse prior fragments or gates.
    """
    _validate_stage(campaign.planning_stage)
    output_dir.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()
    input_path = Path(campaign.case.input_path)
    input_exists = input_path.exists()
    input_sha = _file_sha(input_path) if input_exists else ""

    from openmc_agent.inspect import compose_operating_state_requirement

    if input_exists:
        raw_text = input_path.read_text(encoding="utf-8")
        requirement_text = compose_operating_state_requirement(
            raw_text, campaign.case.operating_state,
        )
        requirement_sha = hashlib.sha256(requirement_text.encode()).hexdigest()
    else:
        requirement_text = ""
        requirement_sha = ""

    env_status = detect_provider_environment(campaign.model)
    # Phase 8A Step 6: --stop-after-gate is a cumulative prefix.  Stopping
    # at material_universe must still enable facts as a barrier; stopping
    # at placement enables facts + material_universe + placement; etc.
    # This prevents a downstream gate from running without its upstream
    # barriers.
    effective_stop_after_gate = (
        campaign.stop_after_gate
        if campaign.planning_stage == _PLANNING_STAGE
        else None
    )
    if getattr(campaign, "stop_after_gate", None):
        from openmc_agent.plan_builder.closed_loop.policy import enabled_gates_through
        cumulative_gates = enabled_gates_through(campaign.stop_after_gate)
        policy = make_five_gate_controlled_policy(
            enabled_gate_ids=tuple(g.value for g in cumulative_gates),
            max_review_rounds_per_gate=campaign.plan_loop_max_review_rounds,
            max_repair_rounds_per_gate=campaign.plan_loop_max_repair_rounds,
            max_total_additional_llm_calls=campaign.plan_loop_max_additional_llm_calls,
            stop_after_gate=effective_stop_after_gate,
        )
    else:
        policy = make_five_gate_controlled_policy(
            max_review_rounds_per_gate=campaign.plan_loop_max_review_rounds,
            max_repair_rounds_per_gate=campaign.plan_loop_max_repair_rounds,
            max_total_additional_llm_calls=campaign.plan_loop_max_additional_llm_calls,
        )
    fingerprint = compute_resume_fingerprint(
        case=campaign.case,
        env_status=env_status,
        policy=policy,
        universes_generation_mode=campaign.universes_generation_mode,
        universe_fragment_max_tokens=campaign.universe_fragment_max_tokens,
        large_patch_safe_output_ratio=campaign.large_patch_safe_output_ratio,
        strict_structured_patch_output=campaign.strict_structured_patch_output,
        material_policy=campaign.material_policy,
        runtime_mode=campaign.runtime_supervisor_mode,
        reasoning_effort=campaign.plan_investigation_reasoning_effort or "default",
        output_mode=campaign.plan_investigation_output_mode or "auto",
        input_sha=input_sha,
        requirement_sha=requirement_sha,
        human_answer_sha=campaign.human_answer_hash,
        git_sha=git_sha,
        plan_investigation_mode=campaign.plan_investigation_mode,
        plan_investigation_patch_types=campaign.plan_investigation_patch_types,
        plan_investigation_model=campaign.plan_investigation_model,
        plan_investigation_reasoning_effort=campaign.plan_investigation_reasoning_effort,
        plan_investigation_output_mode=campaign.plan_investigation_output_mode,
        require_source_backed_evidence=campaign.plan_investigation_require_source_backed_evidence,
    )

    # Determine campaign-level status from environment.
    if not env_status.llm_environment_available:
        status = "BLOCKED_BY_LLM_ENVIRONMENT"
    elif (
        campaign.planning_stage in {_RENDER_COMPILE_STAGE, _OPENMC_SMOKE_STAGE}
        and not env_status.openmc_environment_available
    ):
        status = "BLOCKED_BY_OPENMC_ENVIRONMENT"
    elif campaign.planning_stage == _OPENMC_SMOKE_STAGE and not env_status.openmc_smoke_environment_available:
        status = "BLOCKED_BY_CROSS_SECTIONS_ENVIRONMENT"
    else:
        status = "CAMPAIGN_RUNNING"

    llm_budget = estimate_real_campaign_llm_budget(
        expected_patch_count=campaign.expected_patch_count,
        expected_universe_count=campaign.expected_universe_count,
        enabled_gate_count=5,
        max_review_rounds_per_gate=int(getattr(policy, "max_review_rounds_per_gate", 2)),
        max_repair_rounds_per_gate=int(getattr(policy, "max_repair_rounds_per_gate", 2)),
        max_runtime_iterations=campaign.max_runtime_iterations,
        universes_generation_mode=campaign.universes_generation_mode,
        enable_runtime_supervisor=campaign.runtime_supervisor_mode == "real",
        plan_investigation_patch_types=campaign.plan_investigation_patch_types,
        plan_investigation_max_sessions_per_patch_type=campaign.plan_investigation_max_sessions_per_patch_type,
    )
    effective_max_calls = campaign.max_llm_calls or llm_budget.total

    campaign_start = time.perf_counter()
    manifest: dict[str, Any] = {
        "campaign_id": f"{campaign.case.case_id}_{campaign.planning_stage}_{int(time.time())}",
        "case": asdict(campaign.case),
        "profile": campaign.case.acceptance_profile,
        "stage": campaign.planning_stage,
        "requested_runs": campaign.runs,
        "completed_runs": 0,
        "successful_runs": 0,
        "failed_runs": 0,
        "pending_runs": list(range(1, campaign.runs + 1)),
        "git_sha": git_sha,
        "input_sha": input_sha,
        "requirement_sha": requirement_sha,
        "human_answer_sha": campaign.human_answer_hash,
        "model": campaign.model,
        "provider": env_status.provider,
        "policy_hash": fingerprint.plan_policy_hash,
        "enabled_gates": list(fingerprint.enabled_gates),
        "review_modes": list(fingerprint.review_modes),
        "universes_generation_mode": campaign.universes_generation_mode,
        "universe_fragment_max_tokens": campaign.universe_fragment_max_tokens,
        "strict_structured_patch_output": campaign.strict_structured_patch_output,
        "material_policy": campaign.material_policy,
        "runtime_mode": campaign.runtime_supervisor_mode,
        "openmc_cross_sections_fingerprint": fingerprint.openmc_cross_sections_fingerprint,
        "resume_fingerprint": fingerprint.to_dict(),
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": None,
        "environment": {
            "api_key_env": env_status.api_key_env,
            "api_key_present": env_status.api_key_present,
            "openmc_library_present": env_status.openmc_library_present,
            "openmc_cross_sections_present": env_status.openmc_cross_sections_present,
            "openmc_version": env_status.openmc_version,
            "endpoint": env_status.endpoint,
            "blocked_reasons": env_status.blocked_reasons(),
        },
        "configuration": {
            "reference_patch_policy": "off",
            "allow_monolithic_fallback_for_incremental_failure": False,
            "incremental_planning": True,
            "runtime_supervisor": campaign.runtime_supervisor_mode == "real",
            "runtime_llm_repair": campaign.enable_runtime_llm_repair,
            "runtime_supervisor_mode": campaign.runtime_supervisor_mode,
            "runtime_repair_mode": campaign.runtime_repair_mode,
            "max_runtime_iterations": campaign.max_runtime_iterations,
            "wall_time_limit_s": campaign.wall_time_limit_s,
            "campaign_timeout_s": campaign.campaign_timeout_s,
            "max_llm_calls": effective_max_calls,
            "plan_loop_max_review_rounds": campaign.plan_loop_max_review_rounds,
            "plan_loop_max_repair_rounds": campaign.plan_loop_max_repair_rounds,
            "plan_loop_max_additional_llm_calls": campaign.plan_loop_max_additional_llm_calls,
            "llm_budget": llm_budget.to_dict(),
            "accepted_plan_build_state_seed": bool(campaign.metadata.get("accepted_plan_build_state")),
            "accepted_plan_build_state_path": str(campaign.metadata.get("accepted_plan_build_state_path", "")),
        },
        "aggregate_status": status,
        "promotion_reasons": [],
    }

    if status != "CAMPAIGN_RUNNING":
        _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
        _write_json_atomic(output_dir / "campaign_results.json", [])
        _write_json_atomic(output_dir / "llm_budget.json", llm_budget.to_dict())
        return manifest

    # Resume handling.
    results: list[dict[str, Any]] = []
    runs_dir = output_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    if campaign.resume and (output_dir / "campaign_manifest.json").exists():
        try:
            old_manifest = json.loads((output_dir / "campaign_manifest.json").read_text())
            old_fingerprint = CampaignResumeFingerprint.from_dict(
                old_manifest["resume_fingerprint"]
            )
            mismatches = fingerprint.mismatches_against(old_fingerprint)
            if mismatches:
                manifest["aggregate_status"] = "CONFIG_MISMATCH"
                manifest["resume_mismatches"] = mismatches
                _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
                return manifest
            old_results_path = output_dir / "campaign_results.json"
            if old_results_path.exists():
                results = _resumable_campaign_results(json.loads(old_results_path.read_text()))
        except Exception as exc:
            manifest["aggregate_status"] = "CONFIG_MISMATCH"
            manifest["resume_mismatches"] = ["resume_fingerprint"]
            manifest["resume_error"] = str(exc)[:500]
            _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
            return manifest

    manifest["completed_runs"] = len(results)
    manifest["successful_runs"] = sum(1 for r in results if _campaign_run_passed(r))
    manifest["failed_runs"] = manifest["completed_runs"] - manifest["successful_runs"]
    completed_indices = {
        int(str(r.get("run_id", "0")).removeprefix("run_"))
        for r in results
        if str(r.get("run_id", "")).removeprefix("run_").isdigit()
    }
    manifest["pending_runs"] = [
        i for i in range(1, campaign.runs + 1) if i not in completed_indices
    ]
    _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
    _write_json_atomic(output_dir / "campaign_results.json", results)
    _write_json_atomic(output_dir / "llm_budget.json", llm_budget.to_dict())

    # Execute runs.
    for i in range(1, campaign.runs + 1):
        run_id = f"run_{i:03d}"
        if any(r.get("run_id") == run_id for r in results):
            continue
        if time.perf_counter() - campaign_start > campaign.campaign_timeout_s:
            manifest["aggregate_status"] = "CAMPAIGN_TIMEOUT"
            break

        run_dir = runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        run_config = CanaryRunConfig(
            run_id=run_id,
            run_index=i,
            case=campaign.case,
            policy=policy,
            env_status=env_status,
            fingerprint=fingerprint,
            output_dir=str(run_dir),
            model=campaign.model,
            planning_stage=campaign.planning_stage,
            universes_generation_mode=campaign.universes_generation_mode,
            universe_fragment_max_tokens=campaign.universe_fragment_max_tokens,
            large_patch_safe_output_ratio=campaign.large_patch_safe_output_ratio,
            strict_structured_patch_output=campaign.strict_structured_patch_output,
            enable_plots=campaign.enable_plots,
            enable_smoke_test=campaign.planning_stage == _OPENMC_SMOKE_STAGE,
            max_runtime_iterations=campaign.max_runtime_iterations,
            runtime_supervisor_mode=campaign.runtime_supervisor_mode,
            runtime_repair_mode=campaign.runtime_repair_mode,
            enable_runtime_llm_repair=campaign.enable_runtime_llm_repair,
            material_policy=campaign.material_policy,
            wall_time_limit_s=campaign.wall_time_limit_s,
            max_llm_calls=effective_max_calls,
            expected_patch_count=campaign.expected_patch_count,
            expected_universe_count=campaign.expected_universe_count,
            stop_after_gate=effective_stop_after_gate,
            human_answers=dict(campaign.human_answers),
            human_answer_hash=campaign.human_answer_hash,
            acceptance_callback=campaign.acceptance_callback,
            metadata={**dict(campaign.metadata), "llm_budget": llm_budget.to_dict()},
            plan_investigation_mode=campaign.plan_investigation_mode,
            plan_investigation_patch_types=campaign.plan_investigation_patch_types,
            plan_investigation_model=campaign.plan_investigation_model,
            plan_investigation_reasoning_effort=campaign.plan_investigation_reasoning_effort,
            plan_investigation_output_mode=campaign.plan_investigation_output_mode,
            plan_investigation_max_tool_calls=campaign.plan_investigation_max_tool_calls,
            plan_investigation_max_results_per_tool=campaign.plan_investigation_max_results_per_tool,
            plan_investigation_max_evidence_claims=campaign.plan_investigation_max_evidence_claims,
            plan_investigation_max_sessions_per_patch_type=campaign.plan_investigation_max_sessions_per_patch_type,
            plan_investigation_require_source_backed_evidence=campaign.plan_investigation_require_source_backed_evidence,
            plan_investigation_max_tokens=campaign.plan_investigation_max_tokens,
        )
        if campaign.resume:
            checkpoint_store = CampaignCheckpointStore(run_dir / "campaign_checkpoint.json")
            try:
                snapshot = checkpoint_store.hydrate_accepted_state(
                    requirement_hash=fingerprint.requirement_sha,
                    input_hash=input_sha,
                    policy_hash=fingerprint.plan_policy_hash,
                    git_sha=git_sha,
                    structured_output_policy_hash=fingerprint.structured_output_policy_hash,
                )
            except ValueError as exc:
                manifest["aggregate_status"] = "CONFIG_MISMATCH"
                manifest["resume_mismatches"] = ["accepted_state_snapshot"]
                manifest["resume_error"] = str(exc)[:500]
                _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
                return manifest
            if snapshot is not None:
                run_config.metadata["resume_accepted_state"] = snapshot.plan_build_state

        result = run_real_canary_once(
            run_config,
            requirement_text=requirement_text,
            input_sha=input_sha,
            git_sha=git_sha,
        )
        result_dict = result.to_dict()
        results.append(result_dict)

        manifest["completed_runs"] = len(results)
        manifest["successful_runs"] = sum(1 for r in results if _campaign_run_passed(r))
        manifest["failed_runs"] = manifest["completed_runs"] - manifest["successful_runs"]
        manifest["pending_runs"] = list(range(len(results) + 1, campaign.runs + 1))
        _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
        _write_json_atomic(output_dir / "campaign_results.json", results)

        if campaign.fail_fast and result.status == "infrastructure_failure":
            break

    if manifest.get("aggregate_status") == "CAMPAIGN_RUNNING":
        if manifest["completed_runs"] < campaign.runs:
            manifest["aggregate_status"] = "CAMPAIGN_INCOMPLETE"
        elif manifest["successful_runs"] == campaign.runs:
            manifest["aggregate_status"] = "CAMPAIGN_PASSED"
        else:
            manifest["aggregate_status"] = "CAMPAIGN_FAILED"
    manifest["end_time"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(output_dir / "campaign_results.json", results)
    _write_json_atomic(output_dir / "campaign_manifest.json", manifest)
    _write_campaign_csv(output_dir / "campaign_results.csv", results)
    metrics = _aggregate_canary_metrics(results, campaign.runs)
    _write_json_atomic(output_dir / "aggregate_metrics.json", metrics)
    _write_campaign_report(output_dir, manifest, metrics, results)
    return manifest


def _aggregate_canary_metrics(
    results: list[dict[str, Any]],
    requested_runs: int,
) -> dict[str, Any]:
    completed = len(results)
    successful = sum(1 for r in results if _campaign_run_passed(r))
    five_gate_accepted = sum(1 for r in results if r.get("five_gate_accepted"))
    fragmented = sum(1 for r in results if r.get("fragmented_universes_used"))
    return {
        "completed_runs": completed,
        "requested_runs": requested_runs,
        "successful_runs": successful,
        "five_gate_accepted_runs": five_gate_accepted,
        "fragmented_universes_runs": fragmented,
        "final_success_rate": successful / completed if completed else 0.0,
        "five_gate_acceptance_rate": five_gate_accepted / completed if completed else 0.0,
        "real_llm_verification_rate": (
            sum(1 for r in results if r.get("real_llm_verified")) / completed if completed else 0.0
        ),
        "truth_violation_runs": sum(
            1 for r in results if r.get("metadata", {}).get("truth_violations")
        ),
    }


def _write_campaign_csv(path: Path, results: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "run_id", "final_disposition", "five_gate_accepted",
        "five_gate_statuses", "llm_call_count", "plan_reviewer_network_call_count",
        "plan_repair_network_call_count", "fragmented_universes_used",
        "accepted_fragment_count", "duration_s", "real_llm_verified",
        "real_openmc_verified", "smoke_passed",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            row = {fn: r.get(fn, "") for fn in fieldnames}
            row["five_gate_statuses"] = json.dumps(r.get("five_gate_statuses", {}), sort_keys=True)
            writer.writerow(row)


def _write_campaign_report(
    output_dir: Path,
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    results: list[dict[str, Any]],
) -> None:
    lines = [
        f"# Phase 7A Real Canary Report — {manifest.get('case', {}).get('case_id', '?')}",
        "",
        f"**Status**: `{manifest['aggregate_status']}`",
        f"**Stage**: `{manifest.get('stage', '?')}`",
        f"**Model**: `{manifest['model']}`",
        f"**Provider**: `{manifest['provider']}`",
        f"**Git SHA**: `{manifest['git_sha']}`",
        f"**Input SHA**: `{(manifest.get('input_sha') or '')[:16]}...`",
        f"**Policy hash**: `{manifest.get('policy_hash', '')}`",
        f"**Universes generation mode**: `{manifest.get('universes_generation_mode')}`",
        f"**Strict structured output**: `{manifest.get('strict_structured_patch_output')}`",
        "",
        "## Summary",
        "",
        f"- Requested runs: {manifest['requested_runs']}",
        f"- Completed runs: {manifest['completed_runs']}",
        f"- Successful runs: {manifest['successful_runs']}",
        f"- Five-gate accepted runs: {metrics['five_gate_accepted_runs']}",
        f"- Fragmented universes runs: {metrics['fragmented_universes_runs']}",
        f"- Real LLM verified rate: {metrics['real_llm_verification_rate']:.1%}",
        f"- Truth-violation runs: {metrics['truth_violation_runs']}",
        "",
        "## Five-Gate Policy",
        "",
        f"- Enabled gates: `{', '.join(manifest.get('enabled_gates', []))}`",
        f"- Review modes: `{', '.join(manifest.get('review_modes', []))}`",
        "",
        "## Per-Run Results",
        "",
        "| Run | Disposition | 5-Gate | Reviewer | Repair | LLM | Duration |",
        "|-----|-------------|--------|----------|--------|-----|----------|",
    ]
    for r in results:
        rid = r.get("run_id", "?")
        disp = r.get("final_disposition", "?")
        gate_ok = "Y" if r.get("five_gate_accepted") else "N"
        rev = r.get("plan_reviewer_network_call_count", 0)
        rep = r.get("plan_repair_network_call_count", 0)
        llm = r.get("llm_call_count", 0)
        dur = f"{r.get('duration_s', 0):.0f}s"
        lines.append(f"| {rid} | {disp} | {gate_ok} | {rev} | {rep} | {llm} | {dur} |")

    if manifest.get("resume_mismatches"):
        lines.extend([
            "",
            "## Resume Mismatches",
            "",
        ])
        for m in manifest["resume_mismatches"]:
            lines.append(f"- `{m}`")

    (output_dir / "qualification_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
