import os
from pathlib import Path
import sys
from typing import Any, Callable, Literal, TypedDict

import functools
import json
import logging
import re
import shutil
import sqlite3
from datetime import datetime, timezone
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from openmc_agent.executor import render_openmc_script
from openmc_agent.auto_repair import auto_repair_lattice_structure
from openmc_agent.few_shots import select_few_shots
from openmc_agent.grep_search import RetrievedEvidence
from openmc_agent.knowledge_graph import GraphContext
from openmc_agent.lattice_validation import (
    extract_canonical_pin_map,
    is_structural_error_confirmation,
    lattice_cell_mismatches,
)
from openmc_agent.llm import (
    StructuredOutputResult,
    generate_structured_output,
    normalize_capability_report,
    repair_structured_output,
)
from openmc_agent.openmc_api import retrieve_openmc_context
from openmc_agent.plan_builder import (
    should_use_incremental_planning,
    initialize_plan_build_state,
)
from openmc_agent.plan_builder.assembler import assemble_simulation_plan_from_patches
from openmc_agent.plan_builder.state import PlanBuildState
from openmc_agent.plan_builder.validation_repair import (
    PatchRepairEvaluation,
    PatchRepairModelOutput,
    PatchRepairProposal,
    build_patch_repair_request,
    commit_accepted_patch_repair,
    evaluate_patch_repair_proposal,
    normalize_patch_repair_model_output,
)
from openmc_agent.plan_builder.validation_repair_policy import policy_for_issue_code
from openmc_agent.plan_builder.validation_repair_prompts import (
    build_patch_repair_prompt,
    build_patch_repair_schema_correction_prompt,
)
from openmc_agent.plan_builder.pin_map_repair import diagnose_pin_map_count_mismatch
from openmc_agent.plan_builder.component_profile_repair import (
    diagnose_component_profile_slab,
    propose_shoulder_gap_repair_bundle,
    evaluate_shoulder_gap_repair_bundle,
    commit_accepted_repair_bundle,
)
from openmc_agent.plan_builder.grid_loading_repair import (
    diagnose_grid_loading_failure,
    propose_grid_migration_repair_bundle,
    evaluate_grid_migration_repair_bundle,
)
from openmc_agent.plan_builder.patches import parse_patch_content
from openmc_agent.records import append_simulation_record
from openmc_agent.renderers import choose_renderer
from openmc_agent.repair_proposal import (
    RepairProposalMode,
    run_repair_proposal_flow,
)
from openmc_agent.retrieval import (
    RetrievalOutcome,
    ToolSpec,
    run_retrieval_loop,
)
from openmc_agent.retrieval_orchestrator import (
    RetrievalContext,
    RetrievalPolicy,
    format_retrieval_context,
    gather_retrieval_context_for_issues,
    retrieval_context_from_raw,
)
from openmc_agent.schemas import (
    ComplexModelSpec,
    ExecutionCheckSpec,
    ExpertFeedback,
    PlotSpec,
    RenderCapabilityReport,
    ResolvedExpertItem,
    SimulationPlan,
    SimulationSpec,
    ValidationIssue,
    ValidationReport,
)
from openmc_agent.tools import (
    ToolResult,
    export_xml,
    parse_openmc_output,
    run_geometry_debug,
    run_geometry_plots,
    run_smoke_test,
)
from openmc_agent.validator import (
    validate_openmc_script,
    validate_simulation_plan,
    validate_simulation_spec,
)
from openmc_agent.semantic_audit import (
    SemanticAuditMode,
    build_semantic_audit_input,
    run_semantic_plan_audit,
)
from openmc_agent.workflow_trace import (
    TraceConfig,
    TraceRecorder,
    preview_plan,
    summarize_capability_report,
    summarize_retrieval_context as summarize_retrieval_context_for_trace,
    summarize_validation_report,
)

_logger = logging.getLogger("openmc_agent")


GenerateSpecFn = Callable[..., StructuredOutputResult[SimulationSpec]]
RepairSpecFn = Callable[..., StructuredOutputResult[SimulationSpec]]
GeneratePlanFn = Callable[..., StructuredOutputResult[SimulationPlan]]
RepairPlanFn = Callable[..., StructuredOutputResult[SimulationPlan]]
ExportXmlToolFn = Callable[[str | Path], ToolResult]
PlotToolFn = Callable[[str | Path], ToolResult]
SmokeTestToolFn = Callable[[str | Path, SimulationPlan], ToolResult]
GeometryDebugToolFn = Callable[..., ToolResult]
RetrieveOpenMCDocsFn = Callable[[str], list[dict[str, str]]]
SelectFewShotsFn = Callable[[str], list[dict[str, str]]]


class GraphState(TypedDict, total=False):
    requirement: str
    model: str
    output_dir: str
    records_path: str
    simulation_spec: SimulationSpec | None
    validation_report: ValidationReport
    script: str
    model_path: str
    error: str
    retry_count: int
    retry_history: list[dict[str, Any]]
    simulation_plan: SimulationPlan | None
    tool_results: list[dict[str, Any]]
    expert_feedback: list[str]
    pending_expert_questions: list[str]
    expert_round_count: int
    max_expert_rounds: int
    awaiting_expert_feedback: bool
    human_loop_events: list[dict[str, Any]]
    needs_regeneration: bool
    expert_feedback_action: Literal[
        "none",
        "classify",
        "continue",
        "patch_plan",
        "regenerate_plan",
        "manual_review",
    ]
    expert_feedback_interpretation: str | None
    plan_patch: list[dict[str, Any]] | None
    patch_confidence: Literal["high", "medium", "low"] | None
    patch_reason: str | None
    patch_error: str | None
    resolved_expert_items: list[dict[str, Any]]
    capability_repair_errors: list[str]
    # P0-D5A expert-feedback semantics + skeleton blocker routing.
    capability_blocker_summary: dict[str, Any]
    expert_question_groups: list[dict[str, Any]]
    expert_feedback_decision: dict[str, Any]
    expert_assumption_acknowledgements: list[dict[str, Any]]
    workflow_outcome: dict[str, Any]
    raw_llm_outputs: list[str]
    candidate_payload: dict[str, Any] | None
    plan_artifacts: list[str]
    hard_count_constraints: str
    pin_count_mismatch_context: str
    openmc_api_docs: list[dict[str, str]]
    few_shot_examples: list[dict[str, str]]
    verbose: bool
    investigation_trace: list[dict[str, Any]]
    investigation_findings: str
    retrieval_context: dict[str, Any]
    retrieval_prompt: str
    grep_evidence: list[dict[str, Any]]
    graph_context: dict[str, Any]
    rag_evidence: list[dict[str, Any]]
    patch_failure_count: int
    trace: dict[str, Any]
    planning_mode_decision: dict[str, Any]
    plan_build_state: dict[str, Any]
    # Optional accepted source state for isolated runtime evaluation. Normal
    # production planning never provides this field.
    accepted_plan_build_state: dict[str, Any]
    incremental_execution_result: dict[str, Any]
    plan_loop_mode: str
    plan_loop_policy: dict[str, Any]
    plan_loop_outcome: dict[str, Any]
    plan_loop_artifacts: list[str]
    # Facts-gate human confirmation is intentionally separate from the older
    # capability-expert free-text protocol.
    plan_human_questions: list[dict[str, Any]]
    plan_human_answers: dict[str, dict[str, Any]]
    plan_awaiting_human: bool
    plan_resume_requested: bool
    plan_human_interactive: bool
    incremental_regeneration_pending: bool
    incremental_patch_repair_accepted: bool
    use_incremental_executor: bool
    allow_monolithic_fallback_for_incremental_failure: bool
    requirement_resolution: dict[str, Any]
    semantic_audit_input_summary: dict[str, Any]
    semantic_audit_result: dict[str, Any]
    semantic_audit_findings: list[dict[str, Any]]
    repair_proposal_input_summary: dict[str, Any]
    repair_proposal_result: dict[str, Any]
    repair_proposal_status: str
    repair_resolved_issue_codes: list[str]
    repair_remaining_issue_codes: list[str]
    repair_new_issue_codes: list[str]
    run_supervisor_input_summary: dict[str, Any]
    run_supervisor_result: dict[str, Any]
    run_supervisor_action: str
    run_supervisor_route_override: str
    supervisor_action_history: list[dict[str, Any]]
    supervisor_decision_count: int
    supervisor_retry_count_by_patch: dict[str, int]
    supervisor_no_progress_count: int
    runtime_failures: list[dict[str, Any]]
    runtime_primary_failure: dict[str, Any]
    runtime_repair_request: dict[str, Any]
    runtime_repair_proposal: dict[str, Any]
    runtime_repair_evaluation: dict[str, Any]
    runtime_repair_count: int
    runtime_repair_applied: bool
    runtime_repair_attempts_by_fingerprint: dict[str, int]
    runtime_repair_candidate_hashes: list[str]
    runtime_repair_mode: str
    runtime_diagnosis_input: dict[str, Any]
    runtime_diagnosis_result: dict[str, Any]
    runtime_validated_diagnosis: dict[str, Any]
    runtime_llm_proposal: dict[str, Any]
    runtime_llm_proposal_validation: dict[str, Any]
    runtime_llm_evaluation: dict[str, Any]
    runtime_llm_diagnosis_count: int
    runtime_llm_proposal_count: int
    runtime_committed_repair_count: int
    runtime_reexecution_count: int
    runtime_llm_candidate_hashes: list[str]
    runtime_iteration_count: int
    runtime_transient_retry_count: int
    runtime_failure_history: list[dict[str, Any]]
    runtime_repair_history: list[dict[str, Any]]
    runtime_iteration_history: list[dict[str, Any]]
    runtime_supervisor_history: list[dict[str, Any]]
    runtime_no_progress_count: int
    runtime_execution_succeeded: bool
    runtime_last_plan_hash: str
    runtime_last_build_state_hash: str
    runtime_policy_summary: dict[str, Any]
    runtime_final_disposition: str
    runtime_last_committed_failure_fingerprint: str
    runtime_supervisor_result: dict[str, Any]
    runtime_user_cancelled: bool


InvestigationLlmFn = Callable[[str], StructuredOutputResult]
RetrievalRootsResolver = Callable[[GraphState], list[Path]]


def build_graph(
    generate_spec: GenerateSpecFn = generate_structured_output,
    repair_spec: RepairSpecFn = repair_structured_output,
    max_retries: int = 3,
    checkpoint_path: str | Path | None = None,
    checkpointer: Any | None = None,
):
    if checkpoint_path is not None and checkpointer is not None:
        raise ValueError("Use either checkpoint_path or checkpointer, not both")
    if checkpoint_path is not None:
        checkpointer = _build_sqlite_checkpointer(checkpoint_path)

    graph = StateGraph(GraphState)
    graph.add_node("receive_requirement", _receive_requirement)
    graph.add_node("generate_spec", _make_generate_spec_node(generate_spec))
    graph.add_node("validate_spec", _make_validate_spec_node(max_retries))
    graph.add_node("repair_spec", _make_repair_spec_node(repair_spec, max_retries))
    graph.add_node("render_script", _render_script)
    graph.add_node("save_record", _save_record)

    graph.add_edge(START, "receive_requirement")
    graph.add_edge("receive_requirement", "generate_spec")
    graph.add_edge("generate_spec", "validate_spec")
    graph.add_conditional_edges(
        "validate_spec",
        _make_validation_router(max_retries),
        {
            "render": "render_script",
            "repair": "repair_spec",
            "stop": "save_record",
        },
    )
    graph.add_edge("repair_spec", "validate_spec")
    graph.add_edge("render_script", "save_record")
    graph.add_edge("save_record", END)
    return graph.compile(checkpointer=checkpointer)


def build_plan_graph(
    generate_plan: GeneratePlanFn = functools.partial(
        generate_structured_output, normalizer=normalize_capability_report
    ),
    repair_plan: RepairPlanFn = functools.partial(
        repair_structured_output, normalizer=normalize_capability_report
    ),
    *,
    export_xml_tool: ExportXmlToolFn = export_xml,
    plot_tool: PlotToolFn = run_geometry_plots,
    smoke_test_tool: SmokeTestToolFn = run_smoke_test,
    geometry_debug_tool: GeometryDebugToolFn = run_geometry_debug,
    retrieve_docs: RetrieveOpenMCDocsFn = retrieve_openmc_context,
    select_examples: SelectFewShotsFn = select_few_shots,
    investigation_llm: InvestigationLlmFn | None = None,
    retrieval_roots_resolver: RetrievalRootsResolver | None = None,
    retrieval_tool_dispatch: dict[str, Callable[..., ToolResult]] | None = None,
    retrieval_tool_specs: list[ToolSpec] | None = None,
    investigation_max_iterations: int = 4,
    enable_openmc_source_root: bool = False,
    enable_plots: bool = True,
    enable_smoke_test: bool = True,
    max_retries: int = 3,
    checkpoint_path: str | Path | None = None,
    checkpointer: Any | None = None,
    retrieval_policy: RetrievalPolicy | None = None,
    knowledge_graph_path: str | Path | None = None,
    patch_llm_client: Callable[[str], str] | None = None,
    patch_repair_llm_client: Any | None = None,
    use_incremental_executor: bool = True,
    allow_monolithic_fallback_for_incremental_failure: bool = False,
    reference_patch_policy: str = "off",
    material_policy: Any = None,
    plan_loop_policy: Any = None,
    plan_reviewer_client: Any = None,
    plan_repair_client: Any = None,
    universes_generation_mode: str = "auto",
    universe_fragment_max_tokens: int | None = None,
    large_patch_safe_output_ratio: float = 0.6,
    strict_structured_patch_output: bool = False,
    enable_semantic_audit: bool = False,
    semantic_audit_mode: Literal["off", "warning_only", "strict_evaluation"] = "warning_only",
    semantic_audit_client: Any | None = None,
    semantic_audit_model: str | None = None,
    semantic_audit_allow_fallback: bool = True,
    enable_llm_repair_proposer: bool = False,
    llm_repair_mode: Literal["off", "proposal_only", "validate_only", "apply_if_safe"] = "proposal_only",
    llm_repair_client: Any | None = None,
    llm_repair_model: str | None = None,
    llm_repair_allow_fallback: bool = True,
    llm_repair_max_proposals: int = 1,
    enable_run_supervisor: bool = False,
    run_supervisor_mode: Literal["off", "advisory", "controlled_route"] = "advisory",
    run_supervisor_client: Any | None = None,
    run_supervisor_model: str | None = None,
    run_supervisor_allow_fallback: bool = True,
    run_supervisor_max_decisions: int = 5,
    run_supervisor_max_patch_retries: int = 2,
    run_supervisor_max_no_progress: int = 2,
    enable_llm_runtime_repair: bool = False,
    llm_runtime_repair_mode: Literal["off", "diagnose_only", "validate_only", "apply_if_safe"] = "diagnose_only",
    runtime_diagnostician_client: Any | None = None,
    runtime_diagnostician_model: str | None = None,
    runtime_diagnostician_allow_fallback: bool = False,
    runtime_patch_proposer_client: Any | None = None,
    runtime_patch_proposer_model: str | None = None,
    runtime_patch_proposer_allow_fallback: bool = False,
    enable_runtime_supervisor: bool = False,
    runtime_supervisor_client: Any | None = None,
    runtime_supervisor_model: str | None = None,
    runtime_supervisor_allow_fallback: bool = False,
    runtime_loop_budget: Any | None = None,
    plan_investigation_config: Any = None,
    plan_investigation_client: Any = None,
    plan_investigation_registry: Any = None,
    plan_investigation_policy_registry: Any = None,
    plan_investigation_output_dir: str | Path | None = None,
    checkpoint_callback: Any = None,
    facts_action_callback: Any = None,
):
    if checkpoint_path is not None and checkpointer is not None:
        raise ValueError("Use either checkpoint_path or checkpointer, not both")
    if checkpoint_path is not None:
        checkpointer = _build_sqlite_checkpointer(checkpoint_path)

    if retrieval_roots_resolver is None:
        retrieval_roots_resolver = _make_default_retrieval_roots_resolver(
            enable_openmc_source_root=enable_openmc_source_root
        )

    effective_retrieval_policy = retrieval_policy or RetrievalPolicy()
    if (
        knowledge_graph_path is not None
        and effective_retrieval_policy.knowledge_graph_path is None
    ):
        effective_retrieval_policy = effective_retrieval_policy.model_copy(
            update={"knowledge_graph_path": str(knowledge_graph_path)}
        )

    graph = StateGraph(GraphState)
    graph.add_node("receive_requirement", _receive_requirement)
    graph.add_node("retrieve_openmc_docs", _make_retrieve_openmc_docs_node(retrieve_docs))
    graph.add_node("select_few_shots", _make_select_few_shots_node(select_examples))
    graph.add_node(
        "generate_plan",
        _make_generate_plan_node(
            generate_plan,
            investigation_llm=investigation_llm,
            retrieval_roots_resolver=retrieval_roots_resolver,
            retrieval_tool_dispatch=retrieval_tool_dispatch,
            retrieval_tool_specs=retrieval_tool_specs,
            investigation_max_iterations=investigation_max_iterations,
            patch_llm_client=patch_llm_client,
            use_incremental_executor=use_incremental_executor,
            allow_monolithic_fallback_for_incremental_failure=allow_monolithic_fallback_for_incremental_failure,
            reference_patch_policy=reference_patch_policy,
            material_policy=material_policy,
            plan_loop_policy=plan_loop_policy,
            plan_reviewer_client=plan_reviewer_client,
            plan_repair_client=plan_repair_client,
            universes_generation_mode=universes_generation_mode,
            universe_fragment_max_tokens=universe_fragment_max_tokens,
            large_patch_safe_output_ratio=large_patch_safe_output_ratio,
            strict_structured_patch_output=strict_structured_patch_output,
            plan_investigation_config=plan_investigation_config,
            plan_investigation_client=plan_investigation_client,
            plan_investigation_registry=plan_investigation_registry,
            plan_investigation_policy_registry=plan_investigation_policy_registry,
            plan_investigation_output_dir=plan_investigation_output_dir,
            checkpoint_callback=checkpoint_callback,
            facts_action_callback=facts_action_callback,
        ),
    )
    graph.add_node(
        "validate_plan",
        _make_validate_plan_node(
            max_retries,
            patch_repair_llm_client=patch_repair_llm_client or patch_llm_client,
        ),
    )
    graph.add_node("repair_plan_format", _make_repair_plan_format_node(generate_plan, max_retries))
    graph.add_node("assess_capability", _make_assess_plan_capability_node(max_retries))
    graph.add_node("semantic_audit", _make_semantic_audit_node(
        enabled=enable_semantic_audit,
        mode=semantic_audit_mode,
        client=semantic_audit_client,
        model_name=semantic_audit_model,
        allow_fallback=semantic_audit_allow_fallback,
    ))
    graph.add_node("llm_repair_proposal", _make_llm_repair_proposal_node(
        enabled=enable_llm_repair_proposer,
        mode=llm_repair_mode,
        client=llm_repair_client,
        model_name=llm_repair_model,
        allow_fallback=llm_repair_allow_fallback,
        max_proposals=llm_repair_max_proposals,
    ))
    graph.add_node("run_supervisor", _make_run_supervisor_node(
        enabled=enable_run_supervisor,
        mode=run_supervisor_mode,
        client=run_supervisor_client,
        model_name=run_supervisor_model,
        allow_fallback=run_supervisor_allow_fallback,
        max_decisions=run_supervisor_max_decisions,
        max_patch_retries=run_supervisor_max_patch_retries,
        max_no_progress=run_supervisor_max_no_progress,
    ))
    graph.add_node("ask_expert", _ask_expert)
    graph.add_node("ask_plan_expert", _ask_plan_expert)
    graph.add_node("resume_plan_closed_loop", _resume_plan_closed_loop)
    graph.add_node(
        "execute_plan_retry",
        _make_execute_plan_retry_node(
            patch_llm_client=patch_llm_client,
            plan_reviewer_client=plan_reviewer_client,
            plan_repair_client=plan_repair_client,
        ),
    )
    graph.add_node("resume_plan_retry", _resume_plan_retry)
    graph.add_node("classify_expert_feedback", _classify_expert_feedback)
    graph.add_node("patch_plan_from_expert_feedback", _patch_plan_from_expert_feedback)
    graph.add_node("render_plan_script", _render_plan_script)
    graph.add_node(
        "execute_tools",
        _make_execute_tools_node(
            export_xml_tool=export_xml_tool,
            plot_tool=plot_tool,
            smoke_test_tool=smoke_test_tool,
            geometry_debug_tool=geometry_debug_tool,
            enable_plots=enable_plots,
            enable_smoke_test=enable_smoke_test,
        ),
    )
    graph.add_node(
        "reflect_plan",
        _make_reflect_plan_node(
            repair_plan,
            investigation_llm=investigation_llm,
            retrieval_roots_resolver=retrieval_roots_resolver,
            retrieval_tool_dispatch=retrieval_tool_dispatch,
            retrieval_tool_specs=retrieval_tool_specs,
            investigation_max_iterations=investigation_max_iterations,
            retrieval_policy=effective_retrieval_policy,
        ),
    )
    graph.add_node("save_record", _save_plan_record)
    graph.add_node(
        "classify_runtime_feedback",
        _make_classify_runtime_feedback_node(),
    )
    graph.add_node(
        "deterministic_runtime_repair",
        _make_deterministic_runtime_repair_node(
            max_runtime_repairs=(runtime_loop_budget or {}).get("max_deterministic_attempts", 3)
            if isinstance(runtime_loop_budget, dict) else 3,
        ),
    )
    graph.add_node(
        "llm_runtime_diagnose",
        _make_llm_runtime_diagnose_node(
            client=runtime_diagnostician_client,
            model_name=runtime_diagnostician_model,
            allow_fallback=runtime_diagnostician_allow_fallback,
            mode=llm_runtime_repair_mode,
        ),
    )
    graph.add_node(
        "llm_runtime_propose",
        _make_llm_runtime_propose_node(
            client=runtime_patch_proposer_client,
            model_name=runtime_patch_proposer_model,
            allow_fallback=runtime_patch_proposer_allow_fallback,
            mode=llm_runtime_repair_mode,
        ),
    )
    graph.add_node(
        "runtime_supervisor",
        _make_runtime_supervisor_node(
            client=runtime_supervisor_client,
            model_name=runtime_supervisor_model,
            allow_fallback=runtime_supervisor_allow_fallback,
            budget=runtime_loop_budget,
            enable_llm_runtime_repair=enable_llm_runtime_repair,
        ),
    )

    graph.add_edge(START, "receive_requirement")
    graph.add_edge("receive_requirement", "retrieve_openmc_docs")
    graph.add_edge("retrieve_openmc_docs", "select_few_shots")
    graph.add_edge("select_few_shots", "generate_plan")
    graph.add_conditional_edges(
        "generate_plan", _plan_generation_router,
        {"ask_plan_expert": "ask_plan_expert", "validate": "validate_plan", "execute_plan_retry": "execute_plan_retry", "stop": "save_record"},
    )
    graph.add_conditional_edges(
        "ask_plan_expert", _plan_human_router,
        {"resume": "resume_plan_closed_loop", "resume_plan_retry": "resume_plan_retry", "stop": "save_record"},
    )
    graph.add_edge("resume_plan_closed_loop", "generate_plan")
    graph.add_edge("execute_plan_retry", "generate_plan")
    graph.add_edge("resume_plan_retry", "generate_plan")
    graph.add_conditional_edges(
        "validate_plan",
        _make_plan_validation_router(max_retries),
        {
            "assess": "assess_capability",
            "reflect": "reflect_plan",
            "generate": "generate_plan",
            "repair_format": "repair_plan_format",
            "repair_validate": "validate_plan",
            "stop": "save_record",
        },
    )
    graph.add_edge("repair_plan_format", "validate_plan")
    graph.add_edge("reflect_plan", "validate_plan")
    graph.add_edge("assess_capability", "semantic_audit")
    graph.add_edge("semantic_audit", "llm_repair_proposal")
    graph.add_edge("llm_repair_proposal", "run_supervisor")
    graph.add_conditional_edges(
        "run_supervisor",
        _make_supervisor_aware_router(
            enable_supervisor=enable_run_supervisor,
            supervisor_mode=run_supervisor_mode,
        ),
        {
            "reflect": "reflect_plan",
            "ask": "ask_expert",
            "render": "render_plan_script",
            "generate": "generate_plan",
            "stop": "save_record",
        },
    )
    graph.add_conditional_edges(
        "ask_expert",
        _make_expert_feedback_router(),
        {
            "classify": "classify_expert_feedback",
            "render": "render_plan_script",
            "stop": "save_record",
        },
    )
    graph.add_conditional_edges(
        "classify_expert_feedback",
        _make_expert_feedback_action_router(),
        {
            "patch": "patch_plan_from_expert_feedback",
            "generate": "generate_plan",
            "render": "render_plan_script",
            "stop": "save_record",
        },
    )
    graph.add_conditional_edges(
        "patch_plan_from_expert_feedback",
        _make_plan_patch_router(),
        {
            "validate": "validate_plan",
            "generate": "generate_plan",
        },
    )
    graph.add_edge("render_plan_script", "execute_tools")
    graph.add_conditional_edges(
        "execute_tools",
        _make_plan_execution_router(
            max_retries,
            enable_runtime_supervisor=enable_runtime_supervisor,
        ),
        {
            "ask": "ask_expert",
            "reflect": "reflect_plan",
            "save": "save_record",
            "runtime_repair": "classify_runtime_feedback",
            "runtime_supervisor": "runtime_supervisor",
        },
    )
    graph.add_conditional_edges(
        "classify_runtime_feedback",
        _make_runtime_feedback_router(
            enable_llm_runtime_repair=enable_llm_runtime_repair,
            enable_runtime_supervisor=enable_runtime_supervisor,
        ),
        {
            "repair": "deterministic_runtime_repair",
            "llm_diagnose": "llm_runtime_diagnose",
            "runtime_supervisor": "runtime_supervisor",
            "save": "save_record",
        },
    )
    graph.add_conditional_edges(
        "deterministic_runtime_repair",
        _make_runtime_repair_result_router(
            enable_llm_runtime_repair=enable_llm_runtime_repair,
            enable_runtime_supervisor=enable_runtime_supervisor,
        ),
        {
            "render": "render_plan_script",
            "llm_diagnose": "llm_runtime_diagnose",
            "save": "save_record",
            "runtime_supervisor": "runtime_supervisor",
        },
    )
    graph.add_conditional_edges(
        "runtime_supervisor",
        _make_runtime_supervisor_router(),
        {
            "finish": "save_record",
            "deterministic": "deterministic_runtime_repair",
            "llm": "llm_runtime_diagnose",
            "retry": "render_plan_script",
            "ask": "ask_expert",
            "stop": "save_record",
        },
    )
    graph.add_conditional_edges(
        "llm_runtime_diagnose",
        _make_llm_diagnosis_router(),
        {
            "propose": "llm_runtime_propose",
            "save": "save_record",
        },
    )
    graph.add_conditional_edges(
        "llm_runtime_propose",
        _make_llm_proposal_router(),
        {
            "render": "render_plan_script",
            "save": "save_record",
        },
    )
    graph.add_edge("save_record", END)
    return graph.compile(checkpointer=checkpointer)



def _make_llm_repair_proposal_node(
    *,
    enabled: bool,
    mode: str,
    client: Any | None,
    model_name: str | None,
    allow_fallback: bool,
    max_proposals: int,
):
    def _llm_repair_proposal(state: GraphState) -> GraphState:
        if not enabled or mode == "off" or max_proposals <= 0:
            return {}
        plan = state.get("simulation_plan")
        if plan is None:
            return {}
        validation_report = state.get("validation_report")
        repair_mode = RepairProposalMode(mode)
        _progress(state, "llm_repair_proposal", f"starting mode={repair_mode.value}")
        context = {
            "requirement": state.get("requirement", ""),
            "metadata": {"workflow": "plan_graph"},
            "repair_artifact_dir": str(Path(state.get("output_dir", "data/runs")) / "repair_proposals"),
        }
        start = _trace_event_update(
            state,
            "llm_repair_proposal_started",
            summary="LLM repair proposal started",
            metadata={"mode": repair_mode.value, "model": model_name},
        )
        working_state = {**state, **start}
        result = run_repair_proposal_flow(
            plan=plan,
            validation_result=validation_report,
            audit_result=state.get("semantic_audit_result"),
            mode=repair_mode,
            client=client,
            model_name=model_name,
            allow_fallback=allow_fallback,
            context=context,
        )
        _progress(
            state,
            "llm_repair_proposal",
            f"completed status={result.status} fallback={result.fallback_used}",
        )
        operation_count = len(result.proposal.operations) if result.proposal else 0
        allowed_count = sum(1 for ev in result.operation_evaluations if ev.allowed)
        rejected_count = sum(1 for ev in result.operation_evaluations if not ev.allowed)
        unsafe_count = sum(1 for ev in result.operation_evaluations if ev.risk_level.value == "forbidden")
        meta = {
            "proposal_id": result.proposal_id,
            "mode": result.mode.value,
            "model": model_name,
            "source_issue_codes": result.proposal.source_issue_codes if result.proposal else [],
            "source_audit_finding_codes": result.proposal.source_audit_finding_codes if result.proposal else [],
            "operation_count": operation_count,
            "allowed_operation_count": allowed_count,
            "rejected_operation_count": rejected_count,
            "unsafe_operation_count": unsafe_count,
            "status": result.status,
            "resolved_issue_codes": result.resolved_issue_codes,
            "remaining_issue_codes": result.remaining_issue_codes,
            "new_issue_codes": result.new_issue_codes,
            "applied_to_clone": result.applied_to_clone,
            "applied_to_workflow_plan": result.applied_to_workflow_plan,
            "duration_ms": result.duration_ms,
            "fallback_used": result.fallback_used,
            "operation_evaluations": [ev.model_dump(mode="json") for ev in result.operation_evaluations],
        }
        generated = _trace_event_update(
            working_state,
            "llm_repair_proposal_generated",
            summary="LLM repair proposal generated",
            metadata=meta,
        )
        status_event = {
            "accepted": "llm_repair_proposal_accepted",
            "rejected": "llm_repair_proposal_rejected",
            "unsafe": "llm_repair_proposal_unsafe",
            "failed": "llm_repair_proposal_failed",
            "proposed": "llm_repair_proposal_generated",
        }.get(result.status, "llm_repair_proposal_generated")
        status_update = {} if status_event == "llm_repair_proposal_generated" else _trace_event_update(
            {**working_state, **generated},
            status_event,
            summary=f"LLM repair proposal {result.status}",
            metadata=meta,
        )
        fallback_update = {}
        if result.fallback_used:
            fallback_update = _trace_event_update(
                {**working_state, **generated, **status_update},
                "llm_repair_fallback_used",
                summary="LLM repair proposal used fallback",
                metadata=meta,
            )
        return {
            **start,
            **generated,
            **status_update,
            **fallback_update,
            "repair_proposal_input_summary": {
                "proposal_id": result.proposal_id,
                "mode": result.mode.value,
                "operation_count": operation_count,
            },
            "repair_proposal_result": result.model_dump(mode="json"),
            "repair_proposal_status": result.status,
            "repair_resolved_issue_codes": result.resolved_issue_codes,
            "repair_remaining_issue_codes": result.remaining_issue_codes,
            "repair_new_issue_codes": result.new_issue_codes,
        }
    return _llm_repair_proposal
def _make_semantic_audit_node(
    *,
    enabled: bool,
    mode: str,
    client: Any | None,
    model_name: str | None,
    allow_fallback: bool,
):
    def _semantic_audit(state: GraphState) -> GraphState:
        if not enabled or mode == "off":
            return {}
        _progress(state, "semantic_audit", f"starting mode={mode}")
        audit_input = build_semantic_audit_input(
            requirement=state.get("requirement", ""),
            resolved_requirement=(state.get("requirement_resolution") or {}).get("summary"),
            workflow_state=state,
        )
        start_update = _trace_event_update(
            state,
            "semantic_audit_started",
            summary="semantic plan audit started",
            metadata={"audit_id": audit_input.audit_id, "model": model_name},
        )
        working_state = {**state, **start_update}
        result = run_semantic_plan_audit(
            audit_input,
            mode=SemanticAuditMode(mode),
            client=client,
            model_name=model_name,
            allow_fallback=allow_fallback,
        )
        _progress(
            state,
            "semantic_audit",
            f"completed findings={len(result.findings)} fallback={result.fallback_used}",
        )
        artifacts = _write_semantic_audit_artifacts(working_state, audit_input, result)
        finding_codes = [finding.finding_code for finding in result.findings]
        severity_counts: dict[str, int] = {}
        for finding in result.findings:
            severity_counts[str(finding.severity.value)] = severity_counts.get(str(finding.severity.value), 0) + 1
        meta = {
            "audit_id": result.audit_id,
            "auditor": result.auditor,
            "model": result.model,
            "finding_count": len(result.findings),
            "finding_codes": finding_codes,
            "severity_counts": severity_counts,
            "fallback_used": result.fallback_used,
            "duration_ms": result.duration_ms,
            "artifact_paths": artifacts,
            "findings": [f.model_dump(mode="json") for f in result.findings],
            "mode": result.mode.value,
        }
        completed = _trace_event_update(
            working_state,
            "semantic_audit_completed",
            summary="semantic plan audit completed",
            metadata=meta,
        )
        fallback_update = {}
        if result.fallback_used:
            fallback_update = _trace_event_update(
                {**working_state, **completed},
                "semantic_audit_fallback_used",
                summary="semantic plan audit used deterministic fallback",
                metadata=meta,
            )
        return {
            **start_update,
            **completed,
            **fallback_update,
            "semantic_audit_input_summary": result.input_summary,
            "semantic_audit_result": result.model_dump(mode="json"),
            "semantic_audit_findings": [f.model_dump(mode="json") for f in result.findings],
            "plan_artifacts": _append_many_plan_artifacts(state.get("plan_artifacts", []), artifacts),
        }
    return _semantic_audit


def _write_semantic_audit_artifacts(
    state: GraphState,
    audit_input: Any,
    result: Any,
) -> list[str]:
    try:
        output_dir = Path(state.get("output_dir", "data/runs")) / "semantic_audit"
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path = output_dir / "audit_input.json"
        result_path = output_dir / "audit_result.json"
        findings_path = output_dir / "findings.json"
        _write_json_file(input_path, audit_input.model_dump(mode="json"))
        _write_json_file(result_path, result.model_dump(mode="json"))
        _write_json_file(findings_path, [f.model_dump(mode="json") for f in result.findings])
        return [str(input_path), str(result_path), str(findings_path)]
    except Exception:
        return []


def _append_many_plan_artifacts(paths: list[str], new_paths: list[str]) -> list[str]:
    updated = list(paths or [])
    for path in new_paths:
        if path not in updated:
            updated.append(path)
    return updated


# ---------------------------------------------------------------------------
# Run supervisor node + router
# ---------------------------------------------------------------------------

SUPERVISOR_ROUTE_MAP: dict[str, str] = {
    "continue_to_render": "render",
    "continue_patch_generation": "generate",
    "retry_patch": "generate",
    "request_human_confirmation": "ask",
    "downgrade_to_skeleton": "stop",
    "stop": "stop",
}


def _make_run_supervisor_node(
    *,
    enabled: bool,
    mode: str,
    client: Any | None,
    model_name: str | None,
    allow_fallback: bool,
    max_decisions: int,
    max_patch_retries: int,
    max_no_progress: int,
):
    def _run_supervisor(state: GraphState) -> GraphState:
        if not enabled or mode == "off":
            return {}

        from openmc_agent.run_supervisor import (
            RunSupervisorMode as RSMode,
            run_supervisor_decision,
            write_run_supervisor_artifacts,
        )
        from openmc_agent.run_supervisor_policy import (
            DEFAULT_RUN_SUPERVISOR_CONFIG,
            build_run_supervisor_input,
            detect_no_progress,
        )
        from openmc_agent.run_supervisor_prompts import build_run_supervisor_prompt

        decision_count = state.get("supervisor_decision_count", 0)
        _progress(
            state,
            "run_supervisor",
            f"starting mode={mode} decision={decision_count + 1}/{max_decisions}",
        )
        if decision_count >= max_decisions:
            return _trace_event_update(
                state,
                "workflow_failed",
                summary="run supervisor: max decisions reached",
                metadata={"decision_count": decision_count, "max": max_decisions},
            )

        retry_count_by_patch = dict(state.get("supervisor_retry_count_by_patch") or {})
        action_history = list(state.get("supervisor_action_history") or [])
        prev_no_progress = state.get("supervisor_no_progress_count", 0)
        prev_fingerprint = None
        if action_history:
            prev_fingerprint = action_history[-1].get("state_fingerprint")

        config = {
            **DEFAULT_RUN_SUPERVISOR_CONFIG,
            "max_decisions": max_decisions,
            "max_patch_retries": max_patch_retries,
            "max_no_progress_steps": max_no_progress,
        }

        supervisor_input = build_run_supervisor_input(
            state,
            config=config,
            decision_count=decision_count,
            retry_count_by_patch=retry_count_by_patch,
            no_progress_count=prev_no_progress,
            action_history=action_history,
        )

        # Track no-progress.
        no_progress = detect_no_progress(supervisor_input.state_fingerprint, action_history)
        if no_progress >= max_no_progress:
            return _trace_event_update(
                state,
                "workflow_failed",
                summary="run supervisor: no progress threshold reached",
                metadata={
                    "no_progress_count": no_progress,
                    "max": max_no_progress,
                    "state_fingerprint": supervisor_input.state_fingerprint,
                },
            )

        prompt = build_run_supervisor_prompt(supervisor_input)
        start_update = _trace_event_update(
            state,
            "run_supervisor_started",
            summary="run supervisor decision started",
            metadata={
                "decision_id": supervisor_input.decision_id,
                "model": model_name,
                "mode": mode,
                "decision_count": decision_count,
                "allowed_actions": [a.value for a in supervisor_input.allowed_actions],
                "state_fingerprint": supervisor_input.state_fingerprint,
            },
        )
        working_state = {**state, **start_update}

        result = run_supervisor_decision(
            supervisor_input,
            mode=RSMode(mode),
            client=client,
            model_name=model_name,
            allow_fallback=allow_fallback,
        )
        _progress(
            state,
            "run_supervisor",
            "completed "
            f"action={result.final_action.value if result.final_action else 'none'} "
            f"vetoed={result.vetoed} fallback={result.fallback_used}",
        )

        # Update retry tracking.
        final_action = result.final_action
        if final_action and final_action.value == "retry_patch" and result.proposed_decision:
            target = result.proposed_decision.target_patch_type
            if target:
                retry_count_by_patch[target] = retry_count_by_patch.get(target, 0) + 1

        # Record action in history.
        new_history_entry = {
            "decision_id": result.decision_id,
            "proposed_action": result.proposed_decision.action.value if result.proposed_decision else None,
            "final_action": final_action.value if final_action else None,
            "target_patch_type": result.proposed_decision.target_patch_type if result.proposed_decision else None,
            "accepted": result.accepted,
            "vetoed": result.vetoed,
            "veto_reasons": result.veto_reasons,
            "fallback_used": result.fallback_used,
            "state_fingerprint": result.state_fingerprint,
        }
        updated_history = [*action_history, new_history_entry]

        # Compute route override for controlled_route mode.
        route_override = ""
        if mode == "controlled_route" and final_action:
            route_override = SUPERVISOR_ROUTE_MAP.get(final_action.value, "")

        # Write artifacts.
        try:
            output_dir = Path(state.get("output_dir", "data/runs")) / "run_supervisor"
            write_run_supervisor_artifacts(
                str(output_dir),
                supervisor_input,
                result,
                prompt=prompt,
            )
        except Exception:
            pass

        meta = {
            "decision_id": result.decision_id,
            "mode": result.mode.value,
            "proposed_action": result.proposed_decision.action.value if result.proposed_decision else None,
            "final_action": final_action.value if final_action else None,
            "target_patch_type": result.proposed_decision.target_patch_type if result.proposed_decision else None,
            "accepted": result.accepted,
            "executed": result.executed,
            "vetoed": result.vetoed,
            "veto_reasons": result.veto_reasons,
            "fallback_used": result.fallback_used,
            "model": result.model,
            "supervisor": result.supervisor,
            "confidence": result.proposed_decision.confidence if result.proposed_decision else None,
            "state_fingerprint": result.state_fingerprint,
            "decision_count": decision_count + 1,
            "retry_count_by_patch": retry_count_by_patch,
            "no_progress_count": no_progress,
            "allowed_actions": [a.value for a in supervisor_input.allowed_actions],
            "duration_ms": result.duration_ms,
        }

        event_type = "run_supervisor_decision_accepted"
        if result.vetoed:
            event_type = "run_supervisor_decision_vetoed"
        elif result.fallback_used:
            event_type = "run_supervisor_fallback_used"

        completed = _trace_event_update(
            working_state,
            event_type,
            summary=f"run supervisor: {final_action.value if final_action else 'none'}"
            + (" (vetoed)" if result.vetoed else "")
            + (" (fallback)" if result.fallback_used else ""),
            metadata=meta,
        )

        return {
            **start_update,
            **completed,
            "run_supervisor_input_summary": supervisor_input.model_dump(mode="json"),
            "run_supervisor_result": result.model_dump(mode="json"),
            "run_supervisor_action": final_action.value if final_action else "",
            "run_supervisor_route_override": route_override,
            "supervisor_action_history": updated_history,
            "supervisor_decision_count": decision_count + 1,
            "supervisor_retry_count_by_patch": retry_count_by_patch,
            "supervisor_no_progress_count": no_progress,
        }

    return _run_supervisor


def _make_supervisor_aware_router(
    *,
    enable_supervisor: bool,
    supervisor_mode: str,
):
    """Wrap the original capability router with supervisor override support."""

    original_router = _make_plan_capability_assessment_router()

    def _route(state: GraphState) -> str:
        if state.get("incremental_regeneration_pending"):
            return "generate"
        if enable_supervisor and supervisor_mode == "controlled_route":
            override = state.get("run_supervisor_route_override")
            if override and override in {"reflect", "ask", "render", "generate", "stop"}:
                return override
        return original_router(state)

    return _route


def _build_sqlite_checkpointer(path: str | Path) -> SqliteSaver:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(checkpoint_path), check_same_thread=False)
    # Resume payloads contain our Pydantic schema instances. Explicitly allow
    # this module before LangGraph makes unregistered MessagePack types fatal.
    saver = SqliteSaver(
        conn,
        serde=JsonPlusSerializer(
            allowed_msgpack_modules=[
                ("openmc_agent.schemas", "SimulationPlan"),
                ("openmc_agent.schemas", "ValidationReport"),
            ],
        ),
    )
    saver.setup()
    return saver


def _receive_requirement(state: GraphState) -> GraphState:
    _progress(state, "receive_requirement", "reading and normalizing user requirement")
    requirement = state.get("requirement", "").strip()
    if not requirement:
        _progress(state, "receive_requirement", "failed: requirement is empty")
        return {"error": "requirement is required"}

    # Resolve local file references so feature detection sees the file content.
    # This fixes the case where a requirement only says "described in <file>"
    # without inlining the structural keywords (17x17, guide tube, spacer grid).
    from openmc_agent.requirement_resolver import (
        resolve_requirement_references,
        resolved_requirement_summary,
    )

    # Input files are almost always specified relative to the invocation
    # directory (cwd), not the per-run output_dir. Try cwd first, then fall
    # back to output_dir.
    resolved = resolve_requirement_references(requirement, base_dir=None)
    if not resolved.referenced_files:
        output_dir = state.get("output_dir")
        if output_dir:
            resolved = resolve_requirement_references(
                requirement, base_dir=output_dir,
            )
    resolved_summary = resolved_requirement_summary(resolved)
    # Use the resolved requirement for downstream planning/LLM calls so the
    # file content is visible to feature detection AND patch generation.
    effective_requirement = resolved.resolved_requirement or requirement

    _progress(
        state,
        "receive_requirement",
        f"received {len(requirement)} chars; resolved to {len(effective_requirement)} chars; "
        f"referenced_files={resolved.referenced_files}",
    )

    # Phase 0: decide planning mode (monolithic vs incremental).
    # Pass the resolved requirement via plan_context so the mode detector uses
    # the inlined file content for feature detection without itself reading files.
    decision = should_use_incremental_planning(
        effective_requirement,
        retry_history=state.get("retry_history"),
        plan_context={"resolved_requirement": effective_requirement},
    )
    updates: GraphState = {
        "requirement": effective_requirement,
        "hard_count_constraints": _extract_hard_count_constraints(effective_requirement),
        "planning_mode_decision": decision.model_dump(mode="json"),
        "requirement_resolution": resolved_summary,
    }
    _progress(
        state,
        "receive_requirement",
        f"planning mode={decision.mode} triggers={decision.triggers}",
    )
    if decision.mode == "incremental":
        # Phase 7D: benchmark_id/variant are NOT extracted here — that would
        # require benchmark-specific text matching.  Instead, the executor
        # extracts them from the FactsPatch content (which the LLM generates
        # from the requirement).  This keeps graph.py benchmark-agnostic.
        # Evaluation runners may supply an already accepted incremental source
        # state. Keep that state intact so it still passes through this graph's
        # validation, renderer, execution, and runtime-supervisor nodes.
        supplied_state = state.get("accepted_plan_build_state")
        if supplied_state:
            build_state = PlanBuildState.model_validate(supplied_state)
        else:
            build_state = initialize_plan_build_state(
                requirement=effective_requirement,
                decision=decision,
            )
        if not supplied_state:
            build_state.add_event(
                event_type="planning.incremental_recommended_but_not_executed",
                message=(
                    "incremental planning selected; executor will synthesize input-driven structural patches"
                ),
                data={
                    "planning_mode": "incremental",
                    "reference_patch_policy": "off",
                    "monolithic_reflect_plan_allowed": False,
                    "requirement_resolution": resolved_summary,
                },
            )
        else:
            updates["accepted_plan_build_state"] = build_state.model_dump(mode="json")
        updates["plan_build_state"] = build_state.model_dump(mode="json")
    return updates


def _make_generate_spec_node(generate_spec: GenerateSpecFn):
    def _generate_spec(state: GraphState) -> GraphState:
        if state.get("error"):
            return {}

        model = state.get("model", "openai:gpt-4o")
        _progress(state, "generate_spec", f"calling LLM model={model}")
        result = generate_spec(
            requirement=state["requirement"],
            schema=SimulationSpec,
            model=model,
        )
        if not result.ok or result.value is None:
            _progress(state, "generate_spec", f"failed: {result.error}")
            return {
                "simulation_spec": None,
                "error": result.error or "failed to generate SimulationSpec",
            }
        _progress(state, "generate_spec", f"generated SimulationSpec name={result.value.name!r}")
        return {"simulation_spec": result.value}

    return _generate_spec


def _make_retrieve_openmc_docs_node(retrieve_docs: RetrieveOpenMCDocsFn):
    def _retrieve_openmc_docs(state: GraphState) -> GraphState:
        if state.get("error"):
            return {}
        _progress(state, "retrieve_openmc_docs", "retrieving local OpenMC API context")
        try:
            docs = retrieve_docs(state["requirement"])
        except Exception as exc:
            _progress(state, "retrieve_openmc_docs", f"failed: {exc}")
            return {"openmc_api_docs": []}
        _progress(state, "retrieve_openmc_docs", f"retrieved {len(docs)} API document(s)")
        return {"openmc_api_docs": docs}

    return _retrieve_openmc_docs


def _make_select_few_shots_node(select_examples: SelectFewShotsFn):
    def _select_few_shots(state: GraphState) -> GraphState:
        if state.get("error"):
            return {}
        _progress(state, "select_few_shots", "selecting modeling few-shot examples")
        try:
            examples = select_examples(state["requirement"])
        except Exception as exc:
            _progress(state, "select_few_shots", f"failed: {exc}")
            return {"few_shot_examples": []}
        _progress(state, "select_few_shots", f"selected {len(examples)} example(s)")
        return {"few_shot_examples": examples}

    return _select_few_shots


def _make_default_retrieval_roots_resolver(
    *, enable_openmc_source_root: bool = False
) -> RetrievalRootsResolver:
    """Build the default retrieval-roots resolver from GraphState.

    Roots cover the four evidence sources: the agent source tree, the run's
    output directory (rendered model.py / XML), the data/knowledge dir, and
    (optionally) the installed OpenMC library source.
    """

    def _resolve(state: GraphState) -> list[Path]:
        roots: list[Path] = []
        repo_root = Path(__file__).resolve().parent.parent
        roots.append(repo_root / "openmc_agent")
        output_dir = state.get("output_dir")
        if output_dir:
            roots.append(Path(output_dir))
        roots.append(repo_root / "data")
        if enable_openmc_source_root:
            try:
                import openmc as _openmc  # type: ignore[import-not-found]

                roots.append(Path(_openmc.__file__).resolve().parent)
            except Exception:
                pass
        return [root for root in roots if root.exists()]

    return _resolve


def _error_catalog_hints_for(errors: list[str]) -> list[dict[str, Any]]:
    """Pull matching error_catalog entries to seed the investigation prompt.

    ``ERROR_CATALOG`` entries are plain dicts, but their ``knowledge_refs`` /
    ``repair_hints`` elements are Pydantic models, so access is by attribute.
    """
    try:
        from openmc_agent.error_catalog import ERROR_CATALOG
    except Exception:
        return []
    hints: list[dict[str, Any]] = []
    for code, entry in ERROR_CATALOG.items():
        message = entry.get("message", "")
        schema_path = entry.get("schema_path", "")
        hit = any(
            (bool(message) and message[:40] in err)
            or (bool(schema_path) and schema_path in err)
            for err in errors
        )
        if not hit:
            continue
        retrieval_queries = [
            query
            for ref in entry.get("knowledge_refs", [])
            if (query := getattr(ref, "retrieval_query", None))
        ]
        repair_hints = [
            {
                "action": getattr(hint, "action", ""),
                "message": getattr(hint, "message", ""),
                "target_path": getattr(hint, "target_path", None),
                "example_patch": getattr(hint, "example_patch", None),
            }
            for hint in entry.get("repair_hints", [])
        ]
        hints.append(
            {
                "error_code": code,
                "schema_path": schema_path,
                "retrieval_queries": retrieval_queries,
                "repair_hints": repair_hints,
            }
        )
        if len(hints) >= 5:
            break
    return hints


def _run_investigation_safely(
    state: GraphState,
    *,
    phase: Literal["generate", "reflect"],
    task_brief: str,
    plan_summary: str,
    investigation_llm: InvestigationLlmFn | None,
    retrieval_roots_resolver: RetrievalRootsResolver | None,
    retrieval_tool_dispatch: dict[str, Callable[..., ToolResult]] | None,
    retrieval_tool_specs: list[ToolSpec] | None,
    investigation_max_iterations: int,
    error_catalog_hints: list[dict[str, Any]] | None,
) -> RetrievalOutcome | None:
    """Run the retrieval loop, returning None when disabled or when it fails.

    Never raises: investigation is best-effort and must not block the main
    generate/reflect flow.
    """
    if investigation_llm is None or retrieval_roots_resolver is None:
        return None
    roots = retrieval_roots_resolver(state)
    if not roots:
        return None
    _progress(state, "investigation", f"starting {phase} retrieval loop")
    try:
        outcome = run_retrieval_loop(
            phase=phase,
            task_brief=task_brief,
            plan_summary=plan_summary,
            roots=roots,
            investigation_llm=investigation_llm,
            tool_dispatch=retrieval_tool_dispatch,
            tool_specs=retrieval_tool_specs,
            max_iterations=investigation_max_iterations,
            error_catalog_hints=error_catalog_hints,
        )
    except Exception as exc:  # pragma: no cover - defensive
        _progress(state, "investigation", f"failed: {exc}")
        return None
    patch_ops = len(outcome.patch) if outcome.patch else 0
    _progress(
        state,
        "investigation",
        f"finished iterations={len(outcome.trace)} ok={outcome.ok} patch_ops={patch_ops}",
    )
    return outcome


def _investigation_state_updates(outcome: RetrievalOutcome | None) -> dict[str, Any]:
    if outcome is None:
        return {}
    return {
        "investigation_trace": outcome.trace,
        "investigation_findings": outcome.findings,
    }


PatchLlmClient = Callable[[str], str]


def _write_incremental_artifacts(state: GraphState, exec_result: Any) -> list[str]:
    """Save incremental patch artifacts for diagnosis.

    Returns list of artifact file paths for plan_artifacts summary.
    """
    output_dir = Path(state.get("output_dir", "data/runs"))
    inc_dir = output_dir / "incremental"
    artifact_paths: list[str] = []
    try:
        inc_dir.mkdir(parents=True, exist_ok=True)
        for stale_file in (
            inc_dir / "plan_build_state.json",
            inc_dir / "incremental_execution_result.json",
        ):
            if stale_file.exists():
                stale_file.unlink()
        for stale_dir_name in ("valid_patches", "invalid_patches", "patch_attempts"):
            stale_dir = inc_dir / stale_dir_name
            if stale_dir.exists():
                shutil.rmtree(stale_dir)
        bs_path = inc_dir / "plan_build_state.json"
        _write_json_file(bs_path, exec_result.state.model_dump(mode="json"))
        artifact_paths.append(str(bs_path))
        ier_path = inc_dir / "incremental_execution_result.json"
        _write_json_file(ier_path, {
            "ok": exec_result.ok,
            "summary": exec_result.summary,
            "issues": [i.model_dump(mode="json") for i in exec_result.issues],
        })
        artifact_paths.append(str(ier_path))
        # Save valid and invalid patches.
        valid_dir = inc_dir / "valid_patches"
        invalid_dir = inc_dir / "invalid_patches"
        for env in exec_result.state.patches.values():
            target = valid_dir if env.status == "valid" else invalid_dir
            target.mkdir(parents=True, exist_ok=True)
            p = target / f"{env.patch_type}.json"
            _write_json_file(p, env.content)
            artifact_paths.append(str(p))
        # Phase 7C: save patch attempt raw/prompt/issues from metadata.
        attempts_meta = exec_result.state.metadata.get("patch_attempt_artifacts", {})
        if attempts_meta:
            att_dir = inc_dir / "patch_attempts"
            att_dir.mkdir(parents=True, exist_ok=True)
            for att_key, att_data in attempts_meta.items():
                raw = att_data.get("raw_text", "")
                if raw:
                    rp = att_dir / f"{att_key}_raw.txt"
                    rp.write_text(raw, encoding="utf-8")
                    artifact_paths.append(str(rp))
                prompt = att_data.get("prompt_text", "")
                if prompt:
                    pp = att_dir / f"{att_key}_prompt.txt"
                    pp.write_text(prompt, encoding="utf-8")
                    artifact_paths.append(str(pp))
                issues = att_data.get("issues", [])
                if issues:
                    ip = att_dir / f"{att_key}_issues.json"
                    _write_json_file(ip, issues)
                    artifact_paths.append(str(ip))
        # P0-NEW: save material composition report for traceability.
        mat_rep = getattr(exec_result.state, "material_composition_report", None)
        if mat_rep:
            mr_path = inc_dir / "material_composition_report.json"
            _write_json_file(mr_path, mat_rep)
            artifact_paths.append(str(mr_path))
        species_rep = getattr(exec_result.state, "material_species_resolution_report", None)
        if species_rep:
            sr_path = inc_dir / "material_species_resolution_report.json"
            _write_json_file(sr_path, species_rep)
            artifact_paths.append(str(sr_path))
    except Exception:
        pass  # artifact writing must not block the workflow
    return artifact_paths


def _run_incremental_plan_generation(
    state: GraphState,
    *,
    patch_llm_client: PatchLlmClient,
    allow_fallback: bool = False,
    reference_patch_policy: str = "off",
    material_policy: Any = None,
    plan_loop_policy: Any = None,
    plan_reviewer_client: Any = None,
    plan_repair_client: Any = None,
    universes_generation_mode: str = "auto",
    universe_fragment_max_tokens: int | None = None,
    large_patch_safe_output_ratio: float = 0.6,
    strict_structured_patch_output: bool = False,
    plan_investigation_config: Any = None,
    plan_investigation_client: Any = None,
    plan_investigation_registry: Any = None,
    plan_investigation_policy_registry: Any = None,
    plan_investigation_output_dir: str | Path | None = None,
    checkpoint_callback: Any = None,
    facts_action_callback: Any = None,
) -> GraphState:
    """Run the incremental patch executor and inject the assembled plan.

    This replaces the monolithic LLM full-plan call when mode=incremental.
    On success, the assembled SimulationPlan is stored in ``simulation_plan``
    and the existing validation/capability/renderer pipeline takes over.
    On failure, a structured error with patch-level diagnostics is returned.
    """
    from openmc_agent.plan_builder.executor import run_incremental_planning
    from openmc_agent.plan_builder.state import (
        PlanBuildState as _PlanBuildState,
        initialize_plan_build_state as _init_state,
    )
    from openmc_agent.plan_builder.mode import (
        PlanningModeDecision as _PlanningModeDecision,
    )

    requirement = state.get("requirement", "")
    _progress(state, "generate_plan", "incremental mode: running patch executor")

    # Reconstruct or initialize PlanBuildState from graph state.
    pmd_dict = state.get("planning_mode_decision") or {}
    build_state_dict = state.get("plan_build_state")

    if build_state_dict:
        try:
            build_state = _PlanBuildState.model_validate(build_state_dict)
        except Exception:
            decision = _PlanningModeDecision.model_validate(pmd_dict) if pmd_dict else None
            build_state = _init_state(requirement, decision) if decision else _PlanBuildState(
                state_id="pbs_incremental", requirement_text=requirement,
            )
    else:
        decision = _PlanningModeDecision.model_validate(pmd_dict) if pmd_dict else None
        build_state = _init_state(requirement, decision) if decision else _PlanBuildState(
            state_id="pbs_incremental", requirement_text=requirement,
        )

    # Accepted-fixture evaluation must exercise the production downstream
    # workflow without regenerating its source patches.
    if state.get("accepted_plan_build_state") and build_state.assembled_plan is not None:
        plan = SimulationPlan.model_validate(build_state.assembled_plan)
        artifact_paths = _write_final_simulation_plan(state, plan)
        return {
            "plan_build_state": build_state.model_dump(mode="json"),
            "simulation_plan": plan,
            "simulation_spec": plan.model_spec,
            "incremental_execution_result": {
                "ok": True,
                "summary": {"accepted_fixture": True},
                "issues": [],
                "monolithic_fallback_attempted": False,
                "planning_mode": "incremental",
                "reference_patch_policy": reference_patch_policy,
                "monolithic_reflect_plan_allowed": False,
            },
            "plan_artifacts": artifact_paths,
            "needs_regeneration": False,
            "error": "",
            **_trace_event_update(
                state,
                "plan_generated",
                summary="accepted incremental source state loaded for runtime evaluation",
                plan=plan,
                metadata={"accepted_fixture": True, "planning_mode": "incremental"},
            ),
        }

    # Propagate selected gold few-shot cases into each patch prompt.
    few_shot_case_ids = [
        ex.get("gold_case_id")
        for ex in state.get("few_shot_examples", [])
        if isinstance(ex, dict) and ex.get("gold_case_id")
    ]

    exec_result = run_incremental_planning(
        requirement=requirement,
        state=build_state,
        llm_client=patch_llm_client,
        max_patch_attempts=2,
        strict=True,
        reference_patch_policy=reference_patch_policy,
        few_shot_case_ids=few_shot_case_ids,
        material_policy=material_policy,
        plan_loop_policy=plan_loop_policy,
        plan_loop_output_dir=state.get("output_dir"),
        plan_reviewer_client=plan_reviewer_client,
        plan_repair_client=plan_repair_client,
        universes_generation_mode=universes_generation_mode,
        universe_fragment_max_tokens=universe_fragment_max_tokens,
        large_patch_safe_output_ratio=large_patch_safe_output_ratio,
        strict_structured_patch_output=strict_structured_patch_output,
        plan_investigation_config=plan_investigation_config,
        plan_investigation_client=plan_investigation_client,
        plan_investigation_registry=plan_investigation_registry,
        plan_investigation_policy_registry=plan_investigation_policy_registry,
        plan_investigation_output_dir=plan_investigation_output_dir,
        checkpoint_callback=checkpoint_callback,
        facts_action_callback=facts_action_callback,
    )

    # Phase 7B/7C: save incremental artifacts for diagnosis (before state_updates).
    inc_artifact_paths = _write_incremental_artifacts(state, exec_result)

    # Serialize build state back into graph state regardless of outcome.
    state_updates: dict[str, Any] = {
        "plan_build_state": exec_result.state.model_dump(mode="json"),
        "incremental_execution_result": {
            "ok": exec_result.ok,
            "summary": exec_result.summary,
            "issues": [i.model_dump(mode="json") for i in exec_result.issues],
            "monolithic_fallback_attempted": False,
            "planning_mode": "incremental",
            "reference_patch_policy": reference_patch_policy,
            "monolithic_reflect_plan_allowed": bool(allow_fallback),
        },
        "plan_loop_mode": build_state.plan_loop_mode.value,
        "plan_loop_policy": build_state.plan_loop_policy,
        "plan_loop_outcome": exec_result.plan_loop_outcome or {},
        "plan_loop_artifacts": list(build_state.plan_loop_artifacts),
        "plan_human_questions": [question.model_dump(mode="json") for question in build_state.plan_human_questions.values()],
        "plan_human_answers": {key: answer.model_dump(mode="json") for key, answer in build_state.plan_human_answers.items()},
        "plan_awaiting_human": any(stage.status.value == "awaiting_human" for stage in build_state.plan_loop_stages.values()),
        "plan_resume_requested": False,
        "incremental_regeneration_pending": False,
        "plan_artifacts": list(state.get("plan_artifacts", [])) + inc_artifact_paths,
    }

    if exec_result.ok and exec_result.assembled_plan:
        # Parse assembled plan dict into SimulationPlan model.
        try:
            plan = SimulationPlan.model_validate(exec_result.assembled_plan)
        except Exception as exc:
            _progress(state, "generate_plan", f"assembled plan schema invalid: {exc}")
            state_updates.update({
                "simulation_plan": None,
                "error": f"incremental.assembled_plan_schema_invalid: {exc}",
                **_trace_event_update(
                    state,
                    "plan_generated",
                    summary="incremental assembled plan failed schema validation",
                    metadata={
                        "success": False,
                        "reason": "assembled_plan_schema_invalid",
                        "planning_mode": "incremental",
                    },
                ),
            })
            return state_updates

        _progress(
            state,
            "generate_plan",
            f"incremental plan assembled: {len(plan.complex_model.materials)} materials, "
            f"{len(plan.complex_model.universes)} universes",
        )
        artifact_paths = _write_final_simulation_plan(state, plan)
        state_updates.update({
            "simulation_plan": plan,
            "simulation_spec": plan.model_spec,
            "plan_artifacts": artifact_paths,
            "needs_regeneration": False,
            "error": "",
            **_trace_event_update(
                state,
                "plan_generated",
                summary=(
                    f"incremental plan assembled "
                    f"({len(build_state.get_valid_patches())} valid patches)"
                ),
                plan=plan,
                metadata={
                    "success": True,
                    "planning_mode": "incremental",
                    "reference_patch_policy": reference_patch_policy,
                    "monolithic_reflect_plan_allowed": bool(allow_fallback),
                    "patch_order": [t.patch_type for t in build_state.component_tasks],
                    "valid_patch_count": len(build_state.get_valid_patches()),
                    "assembly_ok": True,
                },
            ),
        })
        return state_updates

    if exec_result.ok and exec_result.summary.get("stopped_after_gate"):
        stopped_gate = str(exec_result.summary["stopped_after_gate"])
        _progress(
            state,
            "generate_plan",
            f"incremental planning stopped after {stopped_gate} gate accepted",
        )
        state_updates.update({
            "simulation_plan": None,
            "needs_regeneration": False,
            "error": "",
            **_trace_event_update(
                state,
                "plan_gate_stopped",
                summary=f"incremental planning stopped after {stopped_gate} gate accepted",
                metadata={
                    "success": True,
                    "planning_mode": "incremental",
                    "stopped_after_gate": stopped_gate,
                    "valid_patch_types": exec_result.summary.get("valid_patch_types", []),
                    "valid_patch_count": exec_result.summary.get("valid_patch_count", 0),
                },
            ),
        })
        return state_updates

    # Incremental execution failed.
    error_codes = [i.code for i in exec_result.issues if i.severity == "error"]
    _progress(
        state,
        "generate_plan",
        f"incremental execution failed: {error_codes}",
    )

    # Phase-3B retry requests are executable work, not terminal planning
    # failures.  Preserve the serialized build state without setting the
    # graph-level ``error`` so _plan_generation_router can enter
    # execute_plan_retry.  That node applies only the Python-selected owner
    # patch repair, then loops back here for a fresh controlled-gate review.
    # Treating this intermediate state as an error used to bypass the retry
    # router entirely and left pending retry requests stranded in artifacts.
    if exec_result.state.plan_retry_pending_request_ids:
        state_updates.update({
            "simulation_plan": None,
            "error": "",
            **_trace_event_update(
                state,
                "plan_retry_pending",
                summary="incremental gate requested executable Phase-3B retry",
                metadata={
                    "planning_mode": "incremental",
                    "pending_retry_request_ids": list(exec_result.state.plan_retry_pending_request_ids),
                    "error_codes": error_codes,
                },
            ),
        })
        return state_updates

    if allow_fallback:
        _progress(
            state,
            "generate_plan",
            "monolithic fallback enabled; falling back to full-plan generation",
        )
        state_updates.update({
            **_trace_event_update(
                state,
                "plan_generated",
                summary="incremental failed; monolithic fallback enabled",
                metadata={
                    "success": False,
                    "planning_mode": "incremental",
                    "fallback": "monolithic",
                    "error_codes": error_codes,
                },
            ),
        })
        # Return empty so the caller falls through to monolithic.
        return {**state_updates, "_fallback_to_monolithic": True}

    exec_result.state.add_event(
        event_type="incremental.monolithic_fallback_disabled_for_complex_case",
        message="monolithic reflect_plan fallback disabled for incremental complex case",
        data={
            "planning_mode": "incremental",
            "reference_patch_policy": reference_patch_policy,
            "monolithic_reflect_plan_allowed": False,
        },
    )
    state_updates["plan_build_state"] = exec_result.state.model_dump(mode="json")
    state_updates["incremental_execution_result"]["issues"].append({
        "code": "incremental.monolithic_fallback_disabled_for_complex_case",
        "severity": "info",
        "message": "monolithic reflect_plan fallback disabled for incremental complex case",
        "patch_type": None,
        "patch_id": None,
        "path": None,
    })
    state_updates.update({
        "simulation_plan": None,
        "error": f"incremental.execution_failed: {'; '.join(error_codes[:3])}",
        **_trace_event_update(
            state,
            "plan_generated",
            summary=f"incremental execution failed: {error_codes[:3]}",
            metadata={
                "success": False,
                "planning_mode": "incremental",
                "failed_patch_type": exec_result.summary.get("failed_patch_type"),
                "valid_patch_types": [
                    e.patch_type for e in build_state.patches.values()
                    if e.status == "valid"
                ],
                "error_codes": error_codes,
            },
        ),
    })
    return state_updates


def _make_generate_plan_node(
    generate_plan: GeneratePlanFn,
    *,
    investigation_llm: InvestigationLlmFn | None = None,
    retrieval_roots_resolver: RetrievalRootsResolver | None = None,
    retrieval_tool_dispatch: dict[str, Callable[..., ToolResult]] | None = None,
    retrieval_tool_specs: list[ToolSpec] | None = None,
    investigation_max_iterations: int = 4,
    patch_llm_client: Callable[[str], str] | None = None,
    use_incremental_executor: bool = True,
    allow_monolithic_fallback_for_incremental_failure: bool = False,
    reference_patch_policy: str = "off",
    material_policy: Any = None,
    plan_loop_policy: Any = None,
    plan_reviewer_client: Any = None,
    plan_repair_client: Any = None,
    universes_generation_mode: str = "auto",
    universe_fragment_max_tokens: int | None = None,
    large_patch_safe_output_ratio: float = 0.6,
    strict_structured_patch_output: bool = False,
    plan_investigation_config: Any = None,
    plan_investigation_client: Any = None,
    plan_investigation_registry: Any = None,
    plan_investigation_policy_registry: Any = None,
    plan_investigation_output_dir: str | Path | None = None,
    enable_semantic_audit: bool = False,
    semantic_audit_mode: Literal["off", "warning_only", "strict_evaluation"] = "warning_only",
    semantic_audit_client: Any | None = None,
    semantic_audit_model: str | None = None,
    semantic_audit_allow_fallback: bool = True,
    enable_llm_repair_proposer: bool = False,
    llm_repair_mode: Literal["off", "proposal_only", "validate_only", "apply_if_safe"] = "proposal_only",
    llm_repair_client: Any | None = None,
    llm_repair_model: str | None = None,
    llm_repair_allow_fallback: bool = True,
    llm_repair_max_proposals: int = 1,
    checkpoint_callback: Any = None,
    facts_action_callback: Any = None,
):
    def _generate_plan(state: GraphState) -> GraphState:
        if state.get("error"):
            return {}

        # Phase 6: route to incremental executor when mode=incremental.
        pmd = state.get("planning_mode_decision") or {}
        if (
            pmd.get("mode") == "incremental"
            and use_incremental_executor
            and state.get("accepted_plan_build_state")
        ):
            return _run_incremental_plan_generation(
                state,
                patch_llm_client=patch_llm_client or (lambda _prompt: ""),
                allow_fallback=False,
                reference_patch_policy=reference_patch_policy,
                material_policy=material_policy,
                plan_loop_policy=plan_loop_policy,
                plan_reviewer_client=plan_reviewer_client,
                plan_repair_client=plan_repair_client,
            )
        if (
            pmd.get("mode") == "incremental"
            and use_incremental_executor
            and patch_llm_client is not None
        ):
            inc_result = _run_incremental_plan_generation(
                state,
                patch_llm_client=patch_llm_client,
                allow_fallback=allow_monolithic_fallback_for_incremental_failure,
                reference_patch_policy=reference_patch_policy,
                material_policy=material_policy,
                plan_loop_policy=plan_loop_policy,
                plan_reviewer_client=plan_reviewer_client,
                plan_repair_client=plan_repair_client,
                universes_generation_mode=universes_generation_mode,
                universe_fragment_max_tokens=universe_fragment_max_tokens,
                large_patch_safe_output_ratio=large_patch_safe_output_ratio,
                strict_structured_patch_output=strict_structured_patch_output,
                plan_investigation_config=plan_investigation_config,
                plan_investigation_client=plan_investigation_client,
                plan_investigation_registry=plan_investigation_registry,
                plan_investigation_policy_registry=plan_investigation_policy_registry,
                plan_investigation_output_dir=plan_investigation_output_dir,
                checkpoint_callback=checkpoint_callback,
                facts_action_callback=facts_action_callback,
            )
            # If fallback was requested, strip marker and continue to monolithic.
            if inc_result.pop("_fallback_to_monolithic", False):
                _progress(state, "generate_plan", "continuing to monolithic path")
            else:
                return inc_result

        # Phase 7: auto-construct patch client from model name when incremental.
        if (
            pmd.get("mode") == "incremental"
            and use_incremental_executor
            and patch_llm_client is None
        ):
            model_name = state.get("model", "openai:gpt-4o")
            _auto_construct_error: str | None = None
            try:
                from openmc_agent.plan_builder.llm_adapter import make_patch_llm_client
                auto_client = make_patch_llm_client(model_name=model_name)
                _progress(
                    state,
                    "generate_plan",
                    f"auto-constructed patch LLM client from model={model_name}",
                )
                inc_result = _run_incremental_plan_generation(
                    state,
                    patch_llm_client=auto_client,
                    allow_fallback=False,
                    reference_patch_policy=reference_patch_policy,
                    material_policy=material_policy,
                    plan_loop_policy=plan_loop_policy,
                    plan_reviewer_client=plan_reviewer_client,
                    plan_repair_client=plan_repair_client,
                    universes_generation_mode=universes_generation_mode,
                    universe_fragment_max_tokens=universe_fragment_max_tokens,
                    large_patch_safe_output_ratio=large_patch_safe_output_ratio,
                    strict_structured_patch_output=strict_structured_patch_output,
                    plan_investigation_config=plan_investigation_config,
                    plan_investigation_client=plan_investigation_client,
                    plan_investigation_registry=plan_investigation_registry,
                    plan_investigation_policy_registry=plan_investigation_policy_registry,
                    plan_investigation_output_dir=plan_investigation_output_dir,
                    checkpoint_callback=checkpoint_callback,
                    facts_action_callback=facts_action_callback,
                )
                if inc_result.pop("_fallback_to_monolithic", False):
                    _progress(
                        state,
                        "generate_plan",
                        "incremental fallback requested; continuing",
                    )
                else:
                    return inc_result
            except Exception as exc:
                _auto_construct_error = str(exc)
                _progress(
                    state,
                    "generate_plan",
                    f"incremental patch client construction failed: {exc}",
                )

            # Phase 7D: If incremental mode was selected but the executor
            # could not run (construction failed or connectivity issue),
            # do NOT silently fall through to monolithic — monolithic will
            # re-encounter the 25K JSON truncation that incremental was
            # designed to avoid.  Return a structured failure instead.
            if _auto_construct_error or pmd.get("mode") == "incremental":
                error_msg = (
                    f"incremental.patch_client_unavailable: "
                    f"{_auto_construct_error or 'incremental executor failed'}"
                )
                _progress(
                    state,
                    "generate_plan",
                    f"NOT falling back to monolithic for incremental mode; "
                    f"returning structured failure",
                )
                return {
                    "simulation_plan": None,
                    "error": error_msg,
                    **_trace_event_update(
                        state,
                        "plan_generated",
                        summary=(
                            f"incremental patch client unavailable: "
                            f"{_auto_construct_error or 'executor failed'}"
                        ),
                        metadata={
                            "success": False,
                            "planning_mode": "incremental",
                            "reason": "patch_client_unavailable",
                            "auto_construct_error": _auto_construct_error,
                            "monolithic_fallback_attempted": False,
                        },
                    ),
                }

        model = state.get("model", "openai:gpt-4o")
        _progress(state, "generate_plan", f"calling LLM model={model}")
        events = list(state.get("human_loop_events", []))
        if state.get("expert_feedback"):
            events.append(
                {
                    "event": "expert_feedback_consumption_prompt_applied",
                    "round": state.get("expert_round_count", 0),
                    "reason": "generation prompt includes expert feedback consumption rules",
                    "action": "generate_plan",
                }
            )
        investigation_outcome = _run_investigation_safely(
            state,
            phase="generate",
            task_brief=state.get("requirement", ""),
            plan_summary="",
            investigation_llm=investigation_llm,
            retrieval_roots_resolver=retrieval_roots_resolver,
            retrieval_tool_dispatch=retrieval_tool_dispatch,
            retrieval_tool_specs=retrieval_tool_specs,
            investigation_max_iterations=investigation_max_iterations,
            error_catalog_hints=None,
        )
        requirement = _augmented_plan_requirement(state)
        if (
            investigation_outcome is not None
            and investigation_outcome.ok
            and investigation_outcome.findings
        ):
            requirement = (
                f"{requirement}\n\n"
                "Investigation findings (from codebase retrieval; verify before relying on):\n"
                f"{_truncate_text(investigation_outcome.findings, 2000)}\n"
            )
        result = generate_plan(
            requirement=requirement,
            schema=SimulationPlan,
            model=model,
        )
        artifact_paths = _write_plan_generation_artifacts(
            state,
            phase="generate_plan",
            result=result,
            retry_count=state.get("retry_count", 0),
        )
        if not result.ok or result.value is None:
            _progress(state, "generate_plan", f"failed: {result.error}")
            return {
                "simulation_plan": None,
                "simulation_spec": None,
                "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
                "candidate_payload": result.candidate_payload,
                "plan_artifacts": artifact_paths,
                "error": result.error or "failed to generate SimulationPlan",
                "human_loop_events": events,
                **_investigation_state_updates(investigation_outcome),
                **_trace_event_update(
                    state,
                    "plan_generated",
                    summary="SimulationPlan generation failed",
                    metadata={"success": False, "error": result.error},
                ),
            }
        _progress(
            state,
            "generate_plan",
            (
                f"generated SimulationPlan name={_plan_name(result.value)!r}, "
                f"plots={len(result.value.plot_specs)}, "
                f"smoke_test={result.value.execution_check.enabled}"
            ),
        )
        return {
            "simulation_plan": result.value,
            "simulation_spec": result.value.model_spec,
            "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
            "candidate_payload": result.candidate_payload,
            "plan_artifacts": _write_final_simulation_plan(
                state,
                result.value,
                existing_paths=artifact_paths,
            ),
            "needs_regeneration": False,
            "expert_feedback_action": "none",
            "human_loop_events": events,
            **_investigation_state_updates(investigation_outcome),
            **_trace_event_update(
                state,
                "plan_generated",
                summary=f"generated SimulationPlan name={_plan_name(result.value)!r}",
                plan=result.value,
                metadata={"success": True},
            ),
        }

    return _generate_plan


def _make_repair_plan_format_node(generate_plan: GeneratePlanFn, max_retries: int):
    def _repair_plan_format(state: GraphState) -> GraphState:
        retry_count = state.get("retry_count", 0)
        if retry_count >= max_retries:
            return {"retry_count": retry_count}

        model = state.get("model", "openai:gpt-4o")
        _progress(
            state,
            "repair_plan_format",
            f"calling LLM JSON-format repair retry={retry_count + 1}/{max_retries}",
        )
        result = generate_plan(
            requirement=_build_format_repair_requirement(state),
            schema=SimulationPlan,
            model=model,
        )
        artifact_paths = _write_plan_generation_artifacts(
            state,
            phase="repair_plan_format",
            result=result,
            retry_count=retry_count + 1,
        )
        if not result.ok or result.value is None:
            _progress(state, "repair_plan_format", f"failed: {result.error}")
            return {
                "simulation_plan": None,
                "simulation_spec": None,
                "retry_count": retry_count + 1,
                "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
                "candidate_payload": result.candidate_payload,
                "plan_artifacts": artifact_paths,
                "error": result.error or "failed to repair SimulationPlan JSON format",
            }
        _progress(state, "repair_plan_format", "format repair produced a SimulationPlan")
        return {
            "simulation_plan": result.value,
            "simulation_spec": result.value.model_spec,
            "retry_count": retry_count + 1,
            "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
            "candidate_payload": result.candidate_payload,
            "plan_artifacts": _write_final_simulation_plan(
                state,
                result.value,
                existing_paths=artifact_paths,
            ),
            "error": "",
        }

    return _repair_plan_format


def _make_repair_spec_node(repair_spec: RepairSpecFn, max_retries: int):
    def _repair_spec(state: GraphState) -> GraphState:
        report = state.get("validation_report")
        spec = state.get("simulation_spec")
        retry_count = state.get("retry_count", 0)
        if spec is None or report is None or report.is_valid or retry_count >= max_retries:
            return {"retry_count": retry_count}

        _progress(state, "repair_spec", f"calling LLM repair retry={retry_count + 1}/{max_retries}")
        result = repair_spec(
            requirement=state["requirement"],
            schema=SimulationSpec,
            model=state.get("model", "openai:gpt-4o"),
            previous_spec=spec,
            validation_errors=report.errors,
        )
        if not result.ok or result.value is None:
            _progress(state, "repair_spec", f"failed: {result.error}")
            return {
                "retry_count": retry_count + 1,
                "error": result.error or "failed to repair SimulationSpec",
            }
        _progress(state, "repair_spec", "repair produced a new SimulationSpec")
        return {
            "simulation_spec": result.value,
            "retry_count": retry_count + 1,
            "error": "",
        }

    return _repair_spec


def _make_validate_spec_node(max_retries: int):
    def _validate_spec(state: GraphState) -> GraphState:
        _progress(state, "validate_spec", "validating SimulationSpec")
        spec = state.get("simulation_spec")
        retry_count = state.get("retry_count", 0)
        if spec is None:
            report = ValidationReport(
                is_valid=False,
                errors=[state.get("error", "SimulationSpec is missing")],
            )
        else:
            report = validate_simulation_spec(spec)

        history = list(state.get("retry_history", []))
        history.append(
            {
                "requirement": state.get("requirement", ""),
                "retry_count": retry_count,
                "spec": spec.model_dump(mode="json") if spec is not None else None,
                "validation_errors": report.errors,
                "fix_suggestion": (
                    "Ask the model to repair the spec using the validation errors."
                    if not report.is_valid and retry_count < max_retries
                    else ""
                ),
            }
        )

        if not report.is_valid:
            _progress(state, "validate_spec", f"failed with {len(report.errors)} error(s)")
            return {
                "validation_report": report,
                "retry_history": history,
                "error": "; ".join(report.errors),
                **_trace_event_update(
                    state,
                    "validation_completed",
                    summary=f"SimulationSpec validation failed with {len(report.errors)} error(s)",
                    report=report,
                ),
            }
        _progress(state, "validate_spec", "passed")
        return {
            "validation_report": report,
            "retry_history": history,
            "error": "",
            **_trace_event_update(
                state,
                "validation_completed",
                summary="SimulationSpec validation passed",
                report=report,
            ),
        }

    return _validate_spec


def _write_validation_repair_artifact(
    output_dir: str | Path | None,
    name: str,
    payload: Any,
) -> str | None:
    """Best-effort, JSON-only repair artifact writer."""
    try:
        root = Path(output_dir or "data/runs") / "validation_repair"
        root.mkdir(parents=True, exist_ok=True)
        path = root / name
        _write_json_file(path, payload.model_dump(mode="json") if hasattr(payload, "model_dump") else payload)
        return str(path)
    except Exception:
        return None


def _plan_build_state_for_validation_repair(value: Any) -> PlanBuildState | None:
    try:
        if isinstance(value, PlanBuildState):
            return value
        if isinstance(value, dict) and value.get("patches"):
            return PlanBuildState.model_validate(value)
    except Exception:
        return None
    return None


def _invoke_patch_repair_llm(client: Any, request: Any, prompt: str) -> str | dict[str, Any]:
    """Support the explicit repair protocol and JSON-only callable test clients."""
    schema = PatchRepairModelOutput.model_json_schema()
    if hasattr(client, "propose_patch_repair"):
        return client.propose_patch_repair(request, prompt=prompt, json_schema=schema)
    # The normal real-model patch adapter exposes JSON mode under this method.
    # Prefer it over its backwards-compatible plain callable interface so a
    # real repair proposal is constrained to the proposal object as well as by
    # the prompt.  Plain fake clients remain supported below.
    if hasattr(client, "generate_patch_json"):
        return client.generate_patch_json(
            prompt=prompt,
            patch_type=request.target_patch_type,
            json_schema=schema,
        )
    if callable(client):
        return client(prompt)
    raise TypeError("patch repair LLM client is unavailable")


def _patch_repair_output_mode_metadata(client: Any) -> dict[str, Any]:
    """Keep adapter capability fallback visible in repair artifacts."""
    return {
        "requested_output_mode": getattr(client, "last_output_mode_requested", "protocol_or_plain_prompt"),
        "actual_output_mode": getattr(client, "last_output_mode_used", "protocol_or_plain_prompt"),
        "structured_fallback_used": bool(getattr(client, "last_output_fallback_used", False)),
        "fallback_reasons": list(getattr(client, "last_output_fallback_reasons", [])),
    }


def _needs_patch_repair_schema_correction(normalization: Any) -> bool:
    """Only malformed/missing operations warrant one format-only retry."""
    raw = getattr(normalization, "raw_output", None)
    if not isinstance(raw, dict):
        return False
    if "operations" not in raw:
        return True
    return any("operations" in error for error in getattr(normalization, "errors", []))


def _resolve_validation_patch_repair_llm_client(
    state: GraphState,
    configured_client: Any | None,
) -> Any | None:
    """Use the configured repair client or construct the normal patch adapter lazily.

    Incremental patch generation constructs its adapter from ``state['model']``
    inside ``generate_plan``.  Validation happens later, so a graph created
    without an injected test client must repeat that construction here rather
    than silently skipping P0-D repair and going straight to regeneration.
    """
    if configured_client is not None:
        return configured_client
    try:
        from openmc_agent.plan_builder.llm_adapter import make_patch_llm_client

        return make_patch_llm_client(model_name=state.get("model", "openai:gpt-4o"))
    except Exception:
        return None


def _try_incremental_validation_patch_repair(
    *,
    state: GraphState,
    report: ValidationReport,
    target_patch_types: list[str],
    llm_client: Any | None,
) -> tuple[PlanBuildState | None, Any | None, dict[str, Any]]:
    """Evaluate at most two patch edits without mutating the workflow state."""
    build_state = _plan_build_state_for_validation_repair(state.get("plan_build_state"))
    if build_state is None:
        return build_state, None, {"status": "unavailable"}
    eligible = [i for i in report.issues if i.severity == "error"]
    if any(i.requires_human_confirmation or i.route_hint in {"ask_expert", "retrieval"} for i in eligible):
        return build_state, None, {"status": "requires_human_confirmation"}
    for target_patch_type in target_patch_types:
        policies = [policy_for_issue_code(issue.code) for issue in eligible]
        policy = next((p for p in policies if p is not None and p.owner_patch_type == target_patch_type), None)
        policy_source = "registry" if policy is not None else "heuristic_fallback"
        allowed = list(policy.allowed_path_patterns) if policy else ["/**"]
        forbidden = list(policy.forbidden_path_patterns) if policy else []
        fingerprint_report = report
        from openmc_agent.plan_builder.validation_repair import compute_validation_issue_fingerprint
        fingerprint = compute_validation_issue_fingerprint(fingerprint_report, target_patch_type=target_patch_type)
        attempts = build_state.validation_repair_attempts_by_fingerprint.get(fingerprint, 0)
        if attempts >= 2:
            continue
        request = build_patch_repair_request(
            state=build_state, report=report, target_patch_type=target_patch_type,
            allowed_path_patterns=allowed, forbidden_path_patterns=forbidden, attempt_index=attempts,
        )
        if request is None:
            continue
        # The pin-map oracle is intentionally before any repair-model call. It
        # is only allowed to edit a uniquely proven base default and still uses
        # the ordinary clone-only acceptance gate below.
        if target_patch_type == "pin_map" and any(i.code == "lattice.pin_count_mismatch" for i in eligible):
            target = next(
                (p for p in build_state.patches.values() if p.patch_type == "pin_map" and p.status == "valid"),
                None,
            )
            plan_payload = build_state.assembled_plan or (
                state.get("simulation_plan").model_dump(mode="json")
                if hasattr(state.get("simulation_plan"), "model_dump") else state.get("simulation_plan")
            )
            if target is not None and isinstance(plan_payload, dict):
                try:
                    diagnosis = diagnose_pin_map_count_mismatch(
                        state=build_state,
                        plan=SimulationPlan.model_validate(plan_payload), report=report,
                        target_patch=parse_patch_content("pin_map", target.content),
                    )
                except Exception as exc:
                    diagnosis = None
                    build_state.add_event(
                        "planning.pin_map_repair_ambiguous", "pin-map repair diagnosis failed",
                        {"reason": str(exc)},
                    )
                if diagnosis is not None:
                    _write_validation_repair_artifact(
                        state.get("output_dir"), f"pin_map_diagnosis_{attempts}.json", diagnosis,
                    )
                    build_state.add_event(
                        "planning.pin_map_repair_diagnosed", "pin-map count mismatch diagnosed",
                        diagnosis.model_dump(mode="json"),
                    )
                    semantic_context = diagnosis.model_dump(mode="json")
                    # Keep only the semantic facts needed by a generic repair
                    # model when the proof is not unique.
                    request = request.model_copy(update={"semantic_context": semantic_context})
                    if diagnosis.deterministic_repair_available:
                        proposal = PatchRepairProposal(
                            repair_id=request.repair_id,
                            target_patch_type="pin_map",
                            operations=diagnosis.deterministic_operations,
                            rationale=(
                                "The default lattice universe was an axial profile replacement; "
                                "the expected base universe has an equal and opposite count delta "
                                "matching all default positions."
                            ),
                            confidence=1.0,
                        )
                        _write_validation_repair_artifact(
                            state.get("output_dir"), f"pin_map_deterministic_proposal_{attempts}.json", proposal,
                        )
                        build_state.add_event(
                            "planning.pin_map_deterministic_repair_proposed",
                            "deterministic pin-map repair proposed", proposal.model_dump(mode="json"),
                        )
                        evaluation = evaluate_patch_repair_proposal(
                            state=build_state, request=request, proposal=proposal,
                            requirement=state.get("requirement", ""),
                        )
                        if evaluation.candidate_hash:
                            build_state.validation_repair_candidate_hashes.setdefault(fingerprint, []).append(evaluation.candidate_hash)
                        _write_validation_repair_artifact(state.get("output_dir"), f"evaluation_{attempts}.json", evaluation)
                        if evaluation.candidate_preview is not None:
                            _write_validation_repair_artifact(
                                state.get("output_dir"), f"pin_map_candidate_preview_{attempts}.json", evaluation.candidate_preview,
                            )
                            build_state.add_event(
                                "planning.pin_map_candidate_preflight", "pin-map candidate preview completed",
                                evaluation.candidate_preview,
                            )
                        if evaluation.repaired_patch is not None:
                            _write_validation_repair_artifact(state.get("output_dir"), f"patch_after_{attempts}.json", evaluation.repaired_patch)
                        if evaluation.validation_report_after is not None:
                            _write_validation_repair_artifact(state.get("output_dir"), f"plan_validation_after_{attempts}.json", evaluation.validation_report_after)
                        if evaluation.accepted:
                            commit_accepted_patch_repair(build_state, evaluation, request)
                            build_state.add_event(
                                "planning.pin_map_deterministic_repair_accepted",
                                "deterministic pin-map repair accepted", evaluation.model_dump(mode="json"),
                            )
                            return build_state, evaluation, {"status": "accepted", "strategy": "deterministic_pin_map_count_repair"}
                        build_state.add_event(
                            "planning.pin_map_deterministic_repair_rejected",
                            "deterministic pin-map repair rejected", evaluation.model_dump(mode="json"),
                        )
                        return build_state, evaluation, {"status": evaluation.status, "strategy": "deterministic_pin_map_count_repair"}
                    build_state.add_event(
                        "planning.pin_map_repair_ambiguous", "pin-map repair requires semantic LLM fallback",
                        {"reasons": diagnosis.reasons},
                    )
        # The component-profile / shoulder-gap oracle is also before any
        # repair-model call.  It produces a deterministic multi-patch bundle
        # (universes + axial_layers) when the defect is uniquely provable.
        if (
            target_patch_type == "axial_layers"
            and any(i.code == "assembly3d.component_profile_as_material_slab" for i in eligible)
        ):
            plan_payload = build_state.assembled_plan or (
                state.get("simulation_plan").model_dump(mode="json")
                if hasattr(state.get("simulation_plan"), "model_dump") else state.get("simulation_plan")
            )
            if isinstance(plan_payload, dict):
                try:
                    assembled_plan = SimulationPlan.model_validate(plan_payload)
                except Exception:
                    assembled_plan = None
                if assembled_plan is not None:
                    slab_issues = [
                        i for i in eligible
                        if i.code == "assembly3d.component_profile_as_material_slab"
                    ]
                    last_bundle_eval = None
                    any_bundle_accepted = False
                    layer_idx = 0
                    for slab_issue in slab_issues:
                        layer_idx += 1
                        # Extract layer_id from schema_path
                        parts = (slab_issue.schema_path or "").split(".")
                        layer_id = next(
                            (p for p in reversed(parts) if p not in ("fill", "fill_type", "")),
                            None,
                        )
                        if layer_id is None or layer_id == "axial_layers":
                            continue
                        # Re-assemble the plan from current patches for each
                        # iteration, since a prior bundle may have committed.
                        try:
                            _parsed_current = [
                                parse_patch_content(e.patch_type, e.content)
                                for e in build_state.patches.values() if e.status == "valid"
                            ]
                            _assembly_current = assemble_simulation_plan_from_patches(_parsed_current, strict=True)
                            if _assembly_current.ok and _assembly_current.plan is not None:
                                assembled_plan = _assembly_current.plan
                        except Exception:
                            pass
                        try:
                            cp_diagnosis = diagnose_component_profile_slab(
                                state=build_state, plan=assembled_plan, layer_id=layer_id,
                            )
                        except Exception as exc:
                            cp_diagnosis = None
                            build_state.add_event(
                                "planning.component_profile_slab_diagnosed",
                                "component-profile slab diagnosis failed",
                                {"layer_id": layer_id, "reason": str(exc)},
                            )
                        if cp_diagnosis is None:
                            continue
                        _write_validation_repair_artifact(
                            state.get("output_dir"),
                            f"component_profile_diagnosis_{attempts}_{layer_idx}.json",
                            cp_diagnosis,
                        )
                        build_state.add_event(
                            "planning.component_profile_slab_diagnosed",
                            "component-profile slab diagnosed",
                            cp_diagnosis.model_dump(mode="json"),
                        )
                        if not cp_diagnosis.deterministic_repair_available:
                            build_state.add_event(
                                "planning.component_profile_dependency_repair_required",
                                "component-profile repair is ambiguous; deferring to LLM",
                                {"reasons": cp_diagnosis.reasons},
                            )
                            continue
                        bundle = propose_shoulder_gap_repair_bundle(
                            state=build_state, diagnosis=cp_diagnosis,
                        )
                        if bundle is None:
                            continue
                        _write_validation_repair_artifact(
                            state.get("output_dir"),
                            f"shoulder_gap_bundle_{attempts}_{layer_idx}.json",
                            bundle,
                        )
                        build_state.add_event(
                            "planning.shoulder_gap_bundle_proposed",
                            "deterministic shoulder-gap repair bundle proposed",
                            bundle.model_dump(mode="json"),
                        )
                        _axial_before = next(
                            (p for p in build_state.patches.values()
                             if p.patch_type == "axial_layers" and p.status == "valid"),
                            None,
                        )
                        _universes_before = next(
                            (p for p in build_state.patches.values()
                             if p.patch_type == "universes" and p.status == "valid"),
                            None,
                        )
                        if _axial_before:
                            _write_validation_repair_artifact(
                                state.get("output_dir"),
                                f"axial_layers_before_{attempts}_{layer_idx}.json",
                                _axial_before.content,
                            )
                        if _universes_before:
                            _write_validation_repair_artifact(
                                state.get("output_dir"),
                                f"universes_before_{attempts}_{layer_idx}.json",
                                _universes_before.content,
                            )
                        # Re-validate the current state as the "before" report
                        _current_report = validate_simulation_plan(
                            assembled_plan, requirement=state.get("requirement", ""),
                        )
                        bundle_eval = evaluate_shoulder_gap_repair_bundle(
                            state=build_state,
                            proposal=bundle,
                            diagnosis=cp_diagnosis,
                            report_before=_current_report,
                            requirement=state.get("requirement", ""),
                        )
                        _write_validation_repair_artifact(
                            state.get("output_dir"),
                            f"shoulder_gap_bundle_evaluation_{attempts}_{layer_idx}.json",
                            bundle_eval,
                        )
                        if bundle_eval.validation_report_after:
                            _write_validation_repair_artifact(
                                state.get("output_dir"),
                                f"validation_after_{attempts}_{layer_idx}.json",
                                bundle_eval.validation_report_after,
                            )
                        if bundle_eval.capability_after:
                            _write_validation_repair_artifact(
                                state.get("output_dir"),
                                f"capability_after_{attempts}_{layer_idx}.json",
                                bundle_eval.capability_after,
                            )
                        if bundle_eval.accepted:
                            commit_accepted_repair_bundle(build_state, bundle, bundle_eval)
                            _axial_after = next(
                                (p for p in build_state.patches.values()
                                 if p.patch_type == "axial_layers" and p.status == "valid"),
                                None,
                            )
                            _universes_after = next(
                                (p for p in build_state.patches.values()
                                 if p.patch_type == "universes" and p.status == "valid"),
                                None,
                            )
                            if _axial_after:
                                _write_validation_repair_artifact(
                                    state.get("output_dir"),
                                    f"axial_layers_after_{attempts}_{layer_idx}.json",
                                    _axial_after.content,
                                )
                            if _universes_after:
                                _write_validation_repair_artifact(
                                    state.get("output_dir"),
                                    f"universes_after_{attempts}_{layer_idx}.json",
                                    _universes_after.content,
                                )
                            build_state.add_event(
                                "planning.shoulder_gap_bundle_accepted",
                                "deterministic shoulder-gap repair bundle accepted",
                                bundle_eval.model_dump(mode="json"),
                            )
                            last_bundle_eval = bundle_eval
                            any_bundle_accepted = True
                        else:
                            build_state.add_event(
                                "planning.shoulder_gap_bundle_rejected",
                                "deterministic shoulder-gap repair bundle rejected",
                                bundle_eval.model_dump(mode="json"),
                            )
                    if any_bundle_accepted and last_bundle_eval is not None:
                        # Re-assemble the final plan from the committed state
                        try:
                            _parsed_final = [
                                parse_patch_content(e.patch_type, e.content)
                                for e in build_state.patches.values() if e.status == "valid"
                            ]
                            _assembly_final = assemble_simulation_plan_from_patches(_parsed_final, strict=True)
                            if _assembly_final.ok and _assembly_final.plan is not None:
                                last_bundle_eval.repaired_plan = _assembly_final.plan.model_dump(mode="json")
                                _final_report = validate_simulation_plan(
                                    _assembly_final.plan, requirement=state.get("requirement", ""),
                                )
                                last_bundle_eval.validation_report_after = _final_report.model_dump(mode="json")
                        except Exception:
                            pass
                        return build_state, last_bundle_eval, {
                            "status": "accepted",
                            "strategy": "deterministic_shoulder_gap_repair",
                        }
        # The grid-loading / lattice-transform oracle is also before any
        # repair-model call.  It diagnoses replacement_universe_missing and
        # spacer_grid_transformation_misuse, then produces a deterministic
        # bundle to remove redundant grid transformations or migrate them to
        # axial overlays.
        _lattice_codes = {
            "lattice_transform.replacement_universe_missing",
            "lattice_transform.cell_id_used_as_universe",
            "assembly3d.spacer_grid_transformation_misuse",
            "lattice_transform.source_universe_missing",
        }
        if (
            target_patch_type == "axial_layers"
            and any(i.code in _lattice_codes for i in eligible)
        ):
            plan_payload = build_state.assembled_plan or (
                state.get("simulation_plan").model_dump(mode="json")
                if hasattr(state.get("simulation_plan"), "model_dump") else state.get("simulation_plan")
            )
            if isinstance(plan_payload, dict):
                try:
                    assembled_plan = SimulationPlan.model_validate(plan_payload)
                except Exception:
                    assembled_plan = None
                if assembled_plan is not None:
                    lattice_issues = [
                        i for i in eligible if i.code in _lattice_codes
                    ]
                    try:
                        grid_diagnosis = diagnose_grid_loading_failure(
                            state=build_state, plan=assembled_plan, issues=lattice_issues,
                        )
                    except Exception as exc:
                        grid_diagnosis = None
                        build_state.add_event(
                            "planning.grid_loading_diagnosed",
                            "grid loading diagnosis failed",
                            {"reason": str(exc)},
                        )
                    if grid_diagnosis is not None:
                        _write_validation_repair_artifact(
                            state.get("output_dir"),
                            f"lattice_loading_diagnosis_{attempts}.json",
                            grid_diagnosis,
                        )
                        build_state.add_event(
                            "planning.grid_transformation_diagnosed",
                            "grid transformation diagnosed",
                            grid_diagnosis.model_dump(mode="json"),
                        )
                        if grid_diagnosis.deterministic_repair_available:
                            grid_bundle = propose_grid_migration_repair_bundle(
                                state=build_state, diagnosis=grid_diagnosis,
                            )
                            if grid_bundle is not None:
                                _write_validation_repair_artifact(
                                    state.get("output_dir"),
                                    f"grid_overlay_migration_bundle_{attempts}.json",
                                    grid_bundle,
                                )
                                build_state.add_event(
                                    "planning.grid_overlay_migration_proposed",
                                    "deterministic grid migration bundle proposed",
                                    grid_bundle.model_dump(mode="json"),
                                )
                                _axial_before = next(
                                    (p for p in build_state.patches.values()
                                     if p.patch_type == "axial_layers" and p.status == "valid"),
                                    None,
                                )
                                _overlays_before = next(
                                    (p for p in build_state.patches.values()
                                     if p.patch_type == "axial_overlays" and p.status == "valid"),
                                    None,
                                )
                                if _axial_before:
                                    _write_validation_repair_artifact(
                                        state.get("output_dir"),
                                        f"axial_layers_before_grid_{attempts}.json",
                                        _axial_before.content,
                                    )
                                if _overlays_before:
                                    _write_validation_repair_artifact(
                                        state.get("output_dir"),
                                        f"axial_overlays_before_grid_{attempts}.json",
                                        _overlays_before.content,
                                    )
                                _current_report = validate_simulation_plan(
                                    assembled_plan, requirement=state.get("requirement", ""),
                                )
                                grid_eval = evaluate_grid_migration_repair_bundle(
                                    state=build_state,
                                    proposal=grid_bundle,
                                    diagnosis=grid_diagnosis,
                                    report_before=_current_report,
                                    requirement=state.get("requirement", ""),
                                )
                                _write_validation_repair_artifact(
                                    state.get("output_dir"),
                                    f"grid_migration_evaluation_{attempts}.json",
                                    grid_eval,
                                )
                                if grid_eval.accepted:
                                    commit_accepted_repair_bundle(build_state, grid_bundle, grid_eval)
                                    _axial_after = next(
                                        (p for p in build_state.patches.values()
                                         if p.patch_type == "axial_layers" and p.status == "valid"),
                                        None,
                                    )
                                    _overlays_after = next(
                                        (p for p in build_state.patches.values()
                                         if p.patch_type == "axial_overlays" and p.status == "valid"),
                                        None,
                                    )
                                    if _axial_after:
                                        _write_validation_repair_artifact(
                                            state.get("output_dir"),
                                            f"axial_layers_after_grid_{attempts}.json",
                                            _axial_after.content,
                                        )
                                    if _overlays_after:
                                        _write_validation_repair_artifact(
                                            state.get("output_dir"),
                                            f"axial_overlays_after_grid_{attempts}.json",
                                            _overlays_after.content,
                                        )
                                    build_state.add_event(
                                        "planning.grid_overlay_migration_accepted",
                                        "deterministic grid migration accepted",
                                        grid_eval.model_dump(mode="json"),
                                    )
                                    try:
                                        _parsed_final = [
                                            parse_patch_content(e.patch_type, e.content)
                                            for e in build_state.patches.values() if e.status == "valid"
                                        ]
                                        _assembly_final = assemble_simulation_plan_from_patches(_parsed_final, strict=True)
                                        if _assembly_final.ok and _assembly_final.plan is not None:
                                            grid_eval.repaired_plan = _assembly_final.plan.model_dump(mode="json")
                                            _final_report = validate_simulation_plan(
                                                _assembly_final.plan, requirement=state.get("requirement", ""),
                                            )
                                            grid_eval.validation_report_after = _final_report.model_dump(mode="json")
                                    except Exception:
                                        pass
                                    return build_state, grid_eval, {
                                        "status": "accepted",
                                        "strategy": "deterministic_grid_migration",
                                    }
                                else:
                                    build_state.add_event(
                                        "planning.grid_overlay_migration_rejected",
                                        "deterministic grid migration rejected",
                                        grid_eval.model_dump(mode="json"),
                                    )
        if llm_client is None:
            return build_state, None, {"status": "unavailable"}
        _write_validation_repair_artifact(state.get("output_dir"), "diagnosis.json", {
            "target_patch_type": target_patch_type,
            "issue_fingerprint": fingerprint,
            "policy_source": policy_source,
            "issues": request.issues,
            "previous_patch_hash": request.previous_patch_hash,
        })
        request_path = _write_validation_repair_artifact(
            state.get("output_dir"), f"request_{attempts}.json", request
        )
        _write_validation_repair_artifact(state.get("output_dir"), f"patch_before_{attempts}.json", request.previous_patch_content)
        _write_validation_repair_artifact(state.get("output_dir"), f"plan_validation_before_{attempts}.json", report)
        build_state.add_event(
            "planning.validation_patch_repair_started",
            "validation-driven patch repair started",
            {"target_patch_type": target_patch_type, "issue_fingerprint": fingerprint, "policy_source": policy_source},
        )
        raw: str | dict[str, Any] | None = None
        try:
            raw = _invoke_patch_repair_llm(llm_client, request, build_patch_repair_prompt(request))
        except Exception as exc:
            build_state.validation_repair_attempts_by_fingerprint[fingerprint] = attempts + 1
            evaluation = PatchRepairEvaluation(
                accepted=False,
                status="failed",
                issues_before=sorted({issue.code for issue in eligible}),
                issues_after=sorted({issue.code for issue in eligible}),
                resolved_issue_codes=[],
                introduced_issue_codes=[],
                issue_fingerprint_before=fingerprint,
                reasons=[f"repair LLM invocation failed: {exc}"],
            )
            _write_validation_repair_artifact(state.get("output_dir"), f"evaluation_{attempts}.json", evaluation)
            build_state.add_event("planning.validation_patch_repair_rejected", "patch repair LLM invocation failed", evaluation.model_dump(mode="json"))
            return build_state, evaluation, {"status": "failed", "reason": str(exc), "request_path": request_path}
        _write_validation_repair_artifact(
            state.get("output_dir"), f"raw_response_{attempts}.json", {"raw_response": raw}
        )
        normalization = normalize_patch_repair_model_output(raw, request=request)
        output_metadata = _patch_repair_output_mode_metadata(llm_client)
        format_correction_count = 0
        if not normalization.ok and _needs_patch_repair_schema_correction(normalization):
            format_correction_count = 1
            correction_prompt = build_patch_repair_schema_correction_prompt(
                request,
                previous_raw_output=normalization.raw_output,
            )
            try:
                correction_raw = _invoke_patch_repair_llm(llm_client, request, correction_prompt)
                _write_validation_repair_artifact(
                    state.get("output_dir"),
                    f"raw_response_{attempts}_format_correction_1.json",
                    {"raw_response": correction_raw},
                )
                normalization = normalize_patch_repair_model_output(correction_raw, request=request)
                output_metadata = _patch_repair_output_mode_metadata(llm_client)
            except Exception as exc:
                normalization.errors.append(f"schema-only correction invocation failed: {exc}")
            build_state.metadata.setdefault("validation_repair_format_correction_counts", {})[
                fingerprint
            ] = format_correction_count
        raw_fields = sorted(normalization.raw_output) if normalization.raw_output else []
        fields_defaulted = [warning.rsplit(".", 1)[0] for warning in normalization.warnings]
        normalization_artifact = {
            **output_metadata,
            "fields_present": raw_fields,
            "fields_defaulted": fields_defaulted,
            "warnings": normalization.warnings,
            "errors": normalization.errors,
            "schema_correction_attempted": bool(format_correction_count),
            "format_correction_count": format_correction_count,
        }
        _write_validation_repair_artifact(
            state.get("output_dir"), f"normalization_{attempts}.json", normalization_artifact
        )
        _write_validation_repair_artifact(
            state.get("output_dir"),
            f"normalized_proposal_{attempts}.json",
            normalization.normalized_output or {},
        )
        if not normalization.ok or normalization.proposal is None:
            _write_validation_repair_artifact(
                state.get("output_dir"),
                f"proposal_{attempts}.json",
                {"raw_response": raw, "schema_error": normalization.errors},
            )
            evaluation = PatchRepairEvaluation(
                accepted=False,
                status="rejected_schema",
                issues_before=sorted({issue.code for issue in eligible}),
                issues_after=sorted({issue.code for issue in eligible}),
                resolved_issue_codes=[],
                introduced_issue_codes=[],
                issue_fingerprint_before=fingerprint,
                reasons=normalization.errors or ["repair proposal normalization failed"],
            )
            _write_validation_repair_artifact(state.get("output_dir"), f"evaluation_{attempts}.json", evaluation)
            build_state.add_event("planning.validation_patch_repair_rejected", "patch repair proposal schema rejected", evaluation.model_dump(mode="json"))
            return build_state, evaluation, {
                "status": "rejected_schema", "reason": "; ".join(normalization.errors),
                "request_path": request_path, "format_correction_count": format_correction_count,
            }
        proposal = normalization.proposal
        _write_validation_repair_artifact(state.get("output_dir"), f"proposal_{attempts}.json", proposal)
        build_state.add_event(
            "planning.validation_patch_repair_proposed", "patch repair proposal received",
            {
                "target_patch_type": target_patch_type,
                "issue_fingerprint": fingerprint,
                "normalization_warnings": normalization.warnings,
                "format_correction_count": format_correction_count,
            },
        )
        evaluation = evaluate_patch_repair_proposal(
            state=build_state, request=request, proposal=proposal, requirement=state.get("requirement", ""),
        )
        build_state.validation_repair_attempts_by_fingerprint[fingerprint] = attempts + 1
        if evaluation.candidate_hash:
            build_state.validation_repair_candidate_hashes.setdefault(fingerprint, []).append(evaluation.candidate_hash)
        _write_validation_repair_artifact(state.get("output_dir"), f"evaluation_{attempts}.json", evaluation)
        if evaluation.candidate_preview is not None:
            _write_validation_repair_artifact(
                state.get("output_dir"), f"pin_map_candidate_preview_{attempts}.json", evaluation.candidate_preview,
            )
            build_state.add_event(
                "planning.pin_map_candidate_preflight", "pin-map candidate preview completed",
                evaluation.candidate_preview,
            )
        if evaluation.repaired_patch is not None:
            _write_validation_repair_artifact(state.get("output_dir"), f"patch_after_{attempts}.json", evaluation.repaired_patch)
        if evaluation.repaired_plan is not None:
            _write_validation_repair_artifact(
                state.get("output_dir"),
                f"plan_validation_after_{attempts}.json",
                evaluation.validation_report_after or {"issues": evaluation.issues_after},
            )
        if evaluation.accepted:
            commit_accepted_patch_repair(build_state, evaluation, request)
            build_state.add_event("planning.validation_patch_repair_accepted", "patch repair accepted", evaluation.model_dump(mode="json"))
            return build_state, evaluation, {
                "status": "accepted", "policy_source": policy_source,
                "format_correction_count": format_correction_count,
            }
        event = "planning.validation_patch_repair_no_progress" if evaluation.status in {"rejected_no_improvement", "rejected_duplicate_candidate"} else "planning.validation_patch_repair_rejected"
        build_state.add_event(event, "patch repair rejected", evaluation.model_dump(mode="json"))
        # A no-progress candidate cannot improve on a second identical prompt;
        # retain the candidate hash and immediately fall through to regeneration.
        if evaluation.status in {"rejected_no_improvement", "rejected_duplicate_candidate"}:
            return build_state, evaluation, {
                "status": evaluation.status, "policy_source": policy_source,
                "format_correction_count": format_correction_count,
            }
        return build_state, evaluation, {
            "status": evaluation.status, "policy_source": policy_source,
            "format_correction_count": format_correction_count,
        }
    return build_state, None, {"status": "budget_exhausted"}


def _make_validate_plan_node(max_retries: int, *, patch_repair_llm_client: Any | None = None):
    def _validate_plan(state: GraphState) -> GraphState:
        _progress(state, "validate_plan", "validating SimulationPlan")
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        retry_count = state.get("retry_count", 0)
        if plan is None:
            report = ValidationReport(
                is_valid=False,
                errors=[state.get("error", "SimulationPlan is missing")],
            )
        else:
            report = validate_simulation_plan(
                plan, requirement=state.get("requirement", "")
            )

        history = list(state.get("retry_history", []))
        history.append(
            {
                "requirement": state.get("requirement", ""),
                "retry_count": retry_count,
                "plan": plan.model_dump(mode="json") if plan is not None else None,
                "validation_errors": report.errors,
                "fix_suggestion": (
                    (
                        "Non-executable complex-only plan used a supported_renderer other than "
                        "'none'; set capability_report.supported_renderer='none' and "
                        "executable_subsystems=[]."
                        if any(
                            "non-executable complex-only plans must use supported_renderer='none'"
                            in err
                            for err in report.errors
                        )
                        else "Ask the model to repair the plan using validation and execution errors."
                    )
                    if not report.is_valid and retry_count < max_retries
                    else ""
                ),
            }
        )

        if not report.is_valid:
            _progress(state, "validate_plan", f"failed with {len(report.errors)} error(s)")
            if (
                plan is None
                and retry_count < max_retries
                and _is_incremental_patch_generation_failure(state, report)
                and not _incremental_gate_outcome_is_terminal(state)
                and not _incremental_patch_generation_retry_exhausted(state)
            ):
                dependency_repair = _incremental_dependency_repair_metadata(state)
                if dependency_repair is not None:
                    repaired_build_state, invalidated_patch_types = (
                        _prepare_incremental_dependency_regeneration(
                            state.get("plan_build_state"), dependency_repair
                        )
                    )
                    _progress(
                        state,
                        "validate_plan",
                        "incremental dependency failure; scheduling targeted universe regeneration",
                    )
                    return {
                        "simulation_plan": None,
                        "simulation_spec": None,
                        "validation_report": report,
                        "retry_history": history,
                        "retry_count": retry_count + 1,
                        "error": "",
                        "incremental_regeneration_pending": True,
                        "plan_build_state": repaired_build_state,
                        "requirement": state.get("requirement", ""),
                        **_trace_event_update(
                            state,
                            "incremental_dependency_regeneration_scheduled",
                            summary=(
                                "incremental axial dependency failure; preserving "
                                "upstream patches for targeted universe regeneration"
                            ),
                            report=report,
                            metadata={
                                "retry_count": retry_count + 1,
                                "missing_universe_ids": dependency_repair.get(
                                    "missing_universe_ids", []
                                ),
                                "invalidated_patch_types": invalidated_patch_types,
                            },
                        ),
                    }
                _progress(
                    state,
                    "validate_plan",
                    "incremental patch generation failed; preserving valid patches for targeted resume",
                )
                return {
                    "simulation_plan": None,
                    "simulation_spec": None,
                    "validation_report": report,
                    "retry_history": history,
                    "retry_count": retry_count + 1,
                    "error": "",
                    "incremental_regeneration_pending": True,
                    "plan_build_state": state.get("plan_build_state", {}),
                    # The incremental executor restores valid envelopes from
                    # plan_build_state and will regenerate only the failed or
                    # invalid patch.  Do not append a full-plan correction
                    # transcript to the source requirement: that polluted
                    # subsequent patch prompts and encouraged broad replans.
                    "requirement": state.get("requirement", ""),
                    **_trace_event_update(
                        state,
                        "incremental_regeneration_scheduled",
                        summary=(
                            "incremental patch generation failed; scheduling "
                            "a targeted patch resume"
                        ),
                        report=report,
                        metadata={
                            "retry_count": retry_count + 1,
                            "reason": "patch_generation_failure",
                            "resume_strategy": "preserve_valid_patches",
                        },
                    ),
                }
            if (
                plan is not None
                and _incremental_reflect_plan_disabled(state)
                and retry_count < max_retries
            ):
                target_patch_types = _incremental_plan_repair_patch_targets(report)
                repair_llm_client = _resolve_validation_patch_repair_llm_client(
                    state,
                    patch_repair_llm_client,
                )
                repaired_build_state, repair_evaluation, repair_meta = _try_incremental_validation_patch_repair(
                    state=state,
                    report=report,
                    target_patch_types=target_patch_types,
                    llm_client=repair_llm_client,
                )
                if repair_evaluation is not None and repair_evaluation.accepted and repair_evaluation.repaired_plan:
                    repaired_plan = SimulationPlan.model_validate(repair_evaluation.repaired_plan)
                    return {
                        "simulation_plan": repaired_plan,
                        "simulation_spec": repaired_plan.model_spec,
                        "validation_report": report,
                        "retry_history": history,
                        "retry_count": retry_count,
                        "incremental_patch_repair_accepted": True,
                        "incremental_regeneration_pending": False,
                        "plan_build_state": repaired_build_state.model_dump(mode="json") if repaired_build_state else state.get("plan_build_state"),
                        **_trace_event_update(state, "planning.validation_patch_repair_accepted", summary="validation patch repair accepted; revalidating assembled clone", report=report, plan=repaired_plan, metadata=repair_meta),
                    }
                repair_build_state = _plan_build_state_with_repair_request(
                    repaired_build_state.model_dump(mode="json") if repaired_build_state else state.get("plan_build_state"),
                    report=report,
                    target_patch_types=target_patch_types,
                )
                if target_patch_types and repaired_build_state is not None:
                    from openmc_agent.plan_builder.validation_repair import compute_validation_issue_fingerprint
                    regeneration_fingerprint = compute_validation_issue_fingerprint(
                        report, target_patch_type=target_patch_types[0]
                    )
                    regeneration_count = repaired_build_state.validation_full_patch_regenerations_by_fingerprint.get(
                        regeneration_fingerprint, 0
                    )
                    if regeneration_count >= 1:
                        return {
                            "simulation_plan": plan,
                            "simulation_spec": plan.model_spec,
                            "validation_report": report,
                            "retry_history": history,
                            "error": "; ".join(report.errors),
                            "plan_build_state": repaired_build_state.model_dump(mode="json"),
                            **_trace_event_update(
                                state,
                                "planning.validation_patch_repair_fallback_to_regeneration",
                                summary="validation patch repair and one full regeneration budget exhausted",
                                report=report,
                                plan=plan,
                                metadata={"issue_fingerprint": regeneration_fingerprint, "full_patch_regeneration_budget": 1},
                            ),
                        }
                    repaired_build_state.validation_full_patch_regenerations_by_fingerprint[
                        regeneration_fingerprint
                    ] = regeneration_count + 1
                    repair_build_state = _plan_build_state_with_repair_request(
                        repaired_build_state.model_dump(mode="json"),
                        report=report,
                        target_patch_types=target_patch_types,
                    )
                if target_patch_types and repair_build_state is not None:
                    if repaired_build_state is not None:
                        repaired_build_state.add_event(
                            "planning.validation_patch_repair_fallback_to_regeneration",
                            "patch edit did not pass acceptance; using targeted full-patch regeneration",
                            {"target_patch_types": target_patch_types, "patch_repair_status": repair_meta.get("status")},
                        )
                        repair_build_state = repaired_build_state.model_dump(mode="json")
                        repair_build_state = _plan_build_state_with_repair_request(
                            repair_build_state, report=report, target_patch_types=target_patch_types
                        )
                    _progress(
                        state,
                        "validate_plan",
                        "incremental plan needs targeted patch repair: "
                        f"{target_patch_types}",
                    )
                    return {
                        "simulation_plan": None,
                        "simulation_spec": None,
                        "validation_report": report,
                        "retry_history": history,
                        "retry_count": retry_count + 1,
                        "error": "",
                        "incremental_regeneration_pending": True,
                        "plan_build_state": repair_build_state,
                        "requirement": _incremental_targeted_repair_requirement(
                            state.get("requirement", ""), report, target_patch_types
                        ),
                        "pin_count_mismatch_context": _pin_count_mismatch_context(
                            {**state, "validation_report": report}
                        ),
                        **_trace_event_update(
                            state,
                            "incremental_regeneration_scheduled",
                            summary=(
                                "incremental plan validation failed; scheduling "
                                "targeted patch repair with validator diagnostics"
                            ),
                            report=report,
                            plan=plan,
                            metadata={
                                "retry_count": retry_count + 1,
                                "issue_codes": [issue.code for issue in report.issues],
                                "target_patch_types": target_patch_types,
                                "repair_mode": "targeted_patch_regeneration",
                                "patch_repair_status": repair_meta.get("status"),
                            },
                        ),
                    }
                _progress(
                    state,
                    "validate_plan",
                    "incremental plan needs regeneration; returning validation diagnostics to patch planner",
                )
                return {
                    "simulation_plan": None,
                    "simulation_spec": None,
                    "validation_report": report,
                    "retry_history": history,
                    "retry_count": retry_count + 1,
                    "error": "",
                    "incremental_regeneration_pending": True,
                    # A completed incremental state caches valid patches. Start a
                    # fresh patch sequence so the LLM can correct this plan.
                    "plan_build_state": {},
                    "requirement": _incremental_regeneration_requirement(
                        state.get("requirement", ""), report
                    ),
                    "pin_count_mismatch_context": _pin_count_mismatch_context(
                        {**state, "validation_report": report}
                    ),
                    **_trace_event_update(
                        state,
                        "incremental_regeneration_scheduled",
                        summary=(
                            "incremental plan validation failed; scheduling a fresh "
                            "patch-planner pass with validator diagnostics"
                        ),
                        report=report,
                        plan=plan,
                        metadata={
                            "retry_count": retry_count + 1,
                            "issue_codes": [issue.code for issue in report.issues],
                        },
                    ),
                }
            invalid_plan = _plan_with_validation_failure_capability(plan, report)
            state_plan = (
                invalid_plan
                if retry_count >= max_retries
                else plan
            )
            artifact_paths = (
                _write_final_simulation_plan(state, invalid_plan)
                if invalid_plan is not None
                else state.get("plan_artifacts", [])
            )
            updates: GraphState = {
                "simulation_plan": state_plan,
                "simulation_spec": state_plan.model_spec if state_plan is not None else None,
                "validation_report": report,
                "retry_history": history,
                "error": "; ".join(report.errors),
                "pin_count_mismatch_context": _pin_count_mismatch_context(
                    {**state, "validation_report": report}
                ),
                "plan_artifacts": artifact_paths,
                **_trace_event_update(
                    state,
                    "validation_completed",
                    summary=f"SimulationPlan validation failed with {len(report.errors)} error(s)",
                    report=report,
                    plan=invalid_plan or plan,
                ),
            }
            if plan is None and _plan_generation_needs_expert_question(report.errors):
                updates["pending_expert_questions"] = _plan_generation_expert_questions(
                    report.errors,
                    state.get("raw_llm_outputs", []),
                )
            return updates
        _progress(state, "validate_plan", "passed")
        artifact_paths = (
            _write_final_simulation_plan(state, plan)
            if plan is not None
            else state.get("plan_artifacts", [])
        )
        return {
            "validation_report": report,
            "retry_history": history,
            "simulation_plan": plan,
            "simulation_spec": plan.model_spec if plan is not None else None,
            "plan_artifacts": artifact_paths,
            "error": "",
            "incremental_patch_repair_accepted": False,
            **_trace_event_update(
                state,
                "validation_completed",
                summary="SimulationPlan validation passed",
                report=report,
                plan=plan,
            ),
        }

    return _validate_plan


def _plan_generation_needs_expert_question(errors: list[str]) -> bool:
    text = "\n".join(errors).lower()
    return "could not parse model response" in text or "could not validate model response" in text


def _incremental_regeneration_requirement(
    requirement: str,
    report: ValidationReport,
) -> str:
    """Return a fresh incremental-planning request with actionable diagnostics."""
    requirement = _strip_incremental_repair_suffix(requirement)
    errors = "\n".join(f"- {error}" for error in report.errors)
    hints = "\n".join(f"- {hint}" for hint in report.suggestions)
    return (
        f"{requirement}\n\n"
        "=== Incremental planner correction required ===\n"
        "The previous patch sequence assembled an invalid SimulationPlan. Rebuild "
        "all patches from the source requirement and correct the following validator "
        "findings. Preserve only facts supported by the input; do not merely change "
        "expected_counts to hide a wrong universe pattern.\n"
        f"Validation errors:\n{errors}\n"
        f"Repair guidance:\n{hints or '- Reconcile the generated patch fields with the errors above.'}"
    )


def _incremental_targeted_repair_requirement(
    requirement: str,
    report: ValidationReport,
    patch_types: list[str],
) -> str:
    """Return a targeted patch-repair prompt suffix for incremental regeneration."""
    requirement = _strip_incremental_repair_suffix(requirement)
    errors = "\n".join(f"- {error}" for error in report.errors)
    issue_details = "\n".join(
        "- "
        + "; ".join(
            part for part in (
                f"code={issue.code}",
                f"path={issue.schema_path}" if issue.schema_path else "",
                issue.message,
            )
            if part
        )
        for issue in report.issues
        if issue.severity == "error"
        or issue.route_hint in {"auto_repair", "reflect_plan"}
    )
    return (
        f"{requirement}\n\n"
        "=== Incremental planner correction required ===\n"
        "The previous patch sequence assembled an invalid SimulationPlan. Regenerate "
        "only the affected patch type(s) requested by the executor, preserving valid "
        "upstream patches and facts supported by the input.\n"
        f"Target patch roots: {', '.join(patch_types)}\n"
        f"Validation errors:\n{errors}\n"
        f"Structured issue details:\n{issue_details or '- unavailable'}"
    )


def _strip_incremental_repair_suffix(requirement: str) -> str:
    marker = "=== Incremental planner correction required ==="
    if marker not in requirement:
        return requirement
    return requirement.split(marker, 1)[0].rstrip()


def _incremental_plan_repair_patch_targets(report: ValidationReport) -> list[str]:
    """Map plan-level validation issues back to likely incremental patch roots."""
    targets: set[str] = set()
    for issue in report.issues:
        if issue.severity != "error" and issue.route_hint not in {
            "auto_repair",
            "reflect_plan",
        }:
            continue
        code = issue.code.lower()
        path = (issue.schema_path or "").lower()
        text = f"{code} {path} {issue.message.lower()}"

        if "axial_overlays" in text:
            targets.add("axial_overlays")
        elif "lattice_loadings" in text or "axial_layers" in text:
            targets.add("axial_layers")
        elif (
            "pin_count" in text
            or "universe_pattern" in text
            or "complex_model.lattices" in text
            or "outer_universe" in text
        ):
            targets.add("pin_map")
        elif "complex_model.universes" in text or "complex_model.cells" in text:
            targets.add("universes")
        elif "complex_model.regions" in text or "complex_model.surfaces" in text:
            targets.add("universes")
        elif "material" in text or "complex_model.materials" in text:
            targets.add("materials")
        elif "settings" in text or "source" in text or "plot_specs" in text:
            targets.add("settings")
        elif "model_spec" in text or code.startswith("plan.model"):
            targets.add("facts")

    canonical_order = [
        "facts",
        "materials",
        "universes",
        "pin_map",
        "axial_layers",
        "axial_overlays",
        "settings",
    ]
    return [patch_type for patch_type in canonical_order if patch_type in targets]


def _plan_build_state_with_repair_request(
    build_state: Any,
    *,
    report: ValidationReport,
    target_patch_types: list[str],
) -> dict[str, Any] | None:
    if not build_state:
        return None
    if hasattr(build_state, "model_dump"):
        payload = build_state.model_dump(mode="json")
    elif isinstance(build_state, dict):
        payload = dict(build_state)
    else:
        return None
    # Only a persisted incremental state with actual patch envelopes can be
    # targeted.  Placeholder/stale graph data must take the fresh-generation
    # fallback rather than masquerading as a repairable cache.
    if not payload.get("patches"):
        return None
    metadata = dict(payload.get("metadata") or {})
    metadata["plan_validation_repair"] = {
        "target_patch_types": list(target_patch_types),
        "issues": [
            issue.model_dump(mode="json")
            for issue in report.issues
            if issue.severity == "error"
            or issue.route_hint in {"auto_repair", "reflect_plan"}
        ],
        "errors": list(report.errors),
    }
    payload["metadata"] = metadata
    return payload


def _plan_generation_expert_questions(
    errors: list[str],
    raw_outputs: list[str],
) -> list[str]:
    questions = [
        "The LLM did not return a valid SimulationPlan JSON object after retries. "
        "Review the modeling request or provide expert feedback that reduces large "
        "lattice/core patterns into explicit assumptions or confirmed pattern ids."
    ]
    latest_raw = raw_outputs[-1] if raw_outputs else ""
    if _looks_like_truncated_json(latest_raw, "\n".join(errors)):
        questions.append(
            "The response appears truncated while emitting a large lattice pattern. "
            "Confirm whether oversized universe_pattern/rings arrays may be omitted "
            "from the first pass and recorded as human-confirmation TODO items."
        )
    return questions


def _make_validation_router(max_retries: int):
    def _route(state: GraphState) -> str:
        report = state.get("validation_report")
        if report is not None and report.is_valid:
            return "render"
        if state.get("simulation_spec") is not None and state.get("retry_count", 0) < max_retries:
            return "repair"
        return "stop"

    return _route


def _make_plan_validation_router(max_retries: int):
    def _route(state: GraphState) -> str:
        report = state.get("validation_report")
        if state.get("incremental_patch_repair_accepted"):
            return "repair_validate"
        if report is not None and report.is_valid and _coerce_simulation_plan(state.get("simulation_plan")) is not None:
            return "assess"
        if state.get("retry_count", 0) >= max_retries:
            return "stop"
        if state.get("incremental_regeneration_pending"):
            return "generate"
        if _incremental_reflect_plan_disabled(state):
            return "stop"
        if _coerce_simulation_plan(state.get("simulation_plan")) is not None:
            return "reflect"
        if state.get("raw_llm_outputs"):
            return "repair_format"
        return "stop"

    return _route


def _make_plan_capability_router():
    def _route(state: GraphState) -> str:
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        report = state.get("validation_report")
        if report is None or not report.is_valid or plan is None:
            return "stop"
        # skeleton / exportable / runnable all produce a model.py; only 'none' stops.
        if plan.capability_report.renderability == "none":
            return "stop"
        return "render"

    return _route


def _make_expert_feedback_router():
    def _route(state: GraphState) -> str:
        decision = state.get("expert_feedback_decision") or {}
        if isinstance(decision, dict) and decision.get("action") == "abort":
            return "stop"
        if state.get("expert_feedback_action") == "classify":
            return "classify"
        return _make_plan_capability_router()(state)

    return _route


def _make_expert_feedback_action_router():
    def _route(state: GraphState) -> str:
        action = state.get("expert_feedback_action", "none")
        if action == "patch_plan":
            return "patch"
        if action == "regenerate_plan" or state.get("needs_regeneration"):
            return "generate"
        return _make_plan_capability_router()(state)

    return _route


def _make_plan_patch_router():
    def _route(state: GraphState) -> str:
        if state.get("expert_feedback_action") == "regenerate_plan" or state.get("patch_error"):
            return "generate"
        return "validate"

    return _route


def _make_plan_execution_router(max_retries: int, *, enable_runtime_supervisor: bool = False):
    def _route(state: GraphState) -> str:
        report = state.get("validation_report")
        if enable_runtime_supervisor:
            if report is not None and report.is_valid:
                return "runtime_supervisor"
            return "runtime_repair"
        if report is not None and report.is_valid:
            return "save"
        # Check for deterministic runtime repair opportunity.
        if (
            state.get("runtime_repair_count", 0) == 0
            and not state.get("runtime_repair_applied", False)
            and _has_runtime_repairable_failure(state)
        ):
            return "runtime_repair"
        if report is not None and _report_should_ask_expert(report):
            return "ask"
        if report is not None and not _report_should_reflect(report):
            return "save"
        if _incremental_reflect_plan_disabled(state):
            return "save"
        if (
            _coerce_simulation_plan(state.get("simulation_plan")) is not None
            and state.get("retry_count", 0) < max_retries
        ):
            return "reflect"
        return "save"

    return _route


def _has_runtime_repairable_failure(state: GraphState) -> bool:
    """Check if any tool result failure is deterministically repairable."""
    from openmc_agent.runtime_repair_policy import get_repair_policy

    # Check validation_report issues (parsed from stdout/stderr).
    report = state.get("validation_report")
    if report is not None and not report.is_valid:
        for issue in report.issues:
            code = issue.code
            policy = get_repair_policy(code)
            if (
                policy is not None
                and policy.deterministic_repair_supported
                and code not in (
                    "runtime.cross_sections_missing",
                    "runtime.cross_sections_invalid",
                )
            ):
                return True

    # Also check tool_results for structured issues.
    tool_results = state.get("tool_results", [])
    for tr in tool_results:
        if isinstance(tr, dict) and not tr.get("ok", True):
            for issue in tr.get("issues", []):
                if isinstance(issue, dict):
                    code = issue.get("code", "")
                else:
                    code = getattr(issue, "code", "")
                policy = get_repair_policy(code)
                if (
                    policy is not None
                    and policy.deterministic_repair_supported
                    and code not in (
                        "runtime.cross_sections_missing",
                        "runtime.cross_sections_invalid",
                    )
                ):
                    return True
    return False


def _make_runtime_feedback_router(*, enable_llm_runtime_repair: bool = False, enable_runtime_supervisor: bool = False):
    def _route(state: GraphState) -> str:
        from openmc_agent.runtime_repair_policy import get_repair_policy

        primary = state.get("runtime_primary_failure")
        if not primary:
            return "runtime_supervisor" if enable_runtime_supervisor else "save"
        if enable_runtime_supervisor:
            return "runtime_supervisor"
        code = primary.get("primary_issue_code", "")
        policy = get_repair_policy(code)
        if (
            policy is not None
            and policy.deterministic_repair_supported
            and state.get("runtime_repair_count", 0) == 0
        ):
            return "repair"
        # LLM diagnosis path: plan_fixable + LLM enabled + diagnosis supported.
        if (
            enable_llm_runtime_repair
            and policy is not None
            and policy.llm_diagnosis_supported
            and state.get("runtime_llm_diagnosis_count", 0) == 0
        ):
            return "llm_diagnose"
        return "save"

    return _route


def _make_runtime_repair_result_router(
    *,
    enable_llm_runtime_repair: bool = False,
    enable_runtime_supervisor: bool = False,
):
    def _route(state: GraphState) -> str:
        if state.get("runtime_repair_applied", False):
            return "render"
        if enable_runtime_supervisor:
            return "runtime_supervisor"
        # If deterministic repair rejected and LLM enabled, try LLM diagnosis.
        if (
            enable_llm_runtime_repair
            and state.get("runtime_llm_diagnosis_count", 0) == 0
        ):
            return "llm_diagnose"
        return "save"

    return _route


def _make_llm_diagnosis_router():
    def _route(state: GraphState) -> str:
        validated = state.get("runtime_validated_diagnosis")
        if not validated:
            return "save"
        if isinstance(validated, dict):
            if validated.get("proposal_allowed") and state.get("runtime_llm_proposal_count", 0) == 0:
                return "propose"
        return "save"

    return _route


def _make_llm_proposal_router():
    def _route(state: GraphState) -> str:
        # Check if proposal was accepted and committed.
        if state.get("runtime_repair_applied", False):
            return "render"
        return "save"

    return _route


def _make_plan_capability_assessment_router():
    def _route(state: GraphState) -> str:
        # assess_capability injects validation errors for LLM-fixable structural
        # defects (missing-cell references, pin-count mismatches, bad radii). Send
        # those back to reflect_plan; material confirmations and clean plans
        # proceed to ask_expert, which is a no-op when there is nothing to ask.
        report = state.get("validation_report")
        if (
            report is not None
            and not report.is_valid
            and state.get("capability_repair_errors")
            and _coerce_simulation_plan(state.get("simulation_plan")) is not None
        ):
            if _incremental_reflect_plan_disabled(state):
                return "ask"
            return "reflect"
        return "ask"

    return _route


def _incremental_reflect_plan_disabled(state: GraphState) -> bool:
    inc = state.get("incremental_execution_result")
    if not isinstance(inc, dict):
        return False
    return (
        inc.get("planning_mode") == "incremental"
        and inc.get("monolithic_reflect_plan_allowed") is False
    )


def _is_incremental_patch_generation_failure(
    state: GraphState, report: ValidationReport
) -> bool:
    """Detect whether the incremental executor failed at patch-generation level.

    Returns True only when the failure occurred *before* assembly (i.e. no plan
    was produced), so the caller can schedule a fresh incremental pass instead
    of dead-ending the workflow.
    """
    inc = state.get("incremental_execution_result")
    if not isinstance(inc, dict):
        return False
    if inc.get("planning_mode") != "incremental":
        return False
    error_text = " ".join(report.errors).lower()
    return (
        "patch_generation_failed" in error_text
        or "incremental.execution_failed" in error_text
        or "incremental.assembled_plan_schema_invalid" in error_text
    )


def _incremental_gate_outcome_is_terminal(state: GraphState) -> bool:
    """Return whether a controlled plan gate intentionally stopped execution.

    Incremental patch failures are normally resumable.  A Facts/Placement gate
    in ``blocked`` or ``awaiting_human`` is different: retrying
    ``generate_plan`` without an accepted revision or typed human answer only
    reuses the same envelopes and can bypass the gate.  The executor is the
    primary barrier; this graph-level check keeps generic graph retries from
    turning that terminal result into a full downstream regeneration.
    """
    outcome = state.get("plan_loop_outcome")
    if isinstance(outcome, dict) and outcome.get("status") in {"blocked", "awaiting_human"}:
        return bool(outcome.get("active_gate_id"))
    build_state = state.get("plan_build_state")
    if not isinstance(build_state, dict):
        return False
    if build_state.get("plan_loop_mode") != "controlled":
        return False
    stages = build_state.get("plan_loop_stages")
    if not isinstance(stages, dict):
        return False
    return any(
        isinstance(stage, dict)
        and stage.get("status") in {"blocked", "awaiting_human"}
        for stage in stages.values()
    )


def _incremental_patch_generation_retry_exhausted(state: GraphState) -> bool:
    """Prevent controlled-mode Graph retries from reopening an exhausted patch.

    ``generate_patch`` already retries the current patch with its schema
    diagnostics.  Once that bounded local retry has failed, a generic
    ``generate_plan`` loop only spends calls on the same target again.  The
    checkpoint remains resumable, but a controlled invocation must return the
    owner-aware failure instead of performing a third whole-workflow pass.
    Off and advisory retain their historical Graph retry semantics.
    """
    build_state = state.get("plan_build_state")
    mode = state.get("plan_loop_mode")
    if not mode and isinstance(build_state, dict):
        mode = build_state.get("plan_loop_mode")
    if mode != "controlled":
        return False
    incremental = state.get("incremental_execution_result")
    if not isinstance(incremental, dict):
        return False
    summary = incremental.get("summary")
    if not isinstance(summary, dict):
        return False
    # A duplicate candidate has no new information even if it also exposed a
    # dependency issue.  A typed dependency request, on the other hand,
    # legitimately changes the upstream context and must still take the
    # existing bounded owner-replay route.
    if summary.get("no_progress"):
        return True
    if not summary.get("patch_generation_exhausted"):
        return False
    return not (
        isinstance(build_state, dict)
        and isinstance(build_state.get("metadata"), dict)
        and isinstance(
            build_state["metadata"].get("incremental_dependency_repair"), dict
        )
    )


def _incremental_dependency_repair_metadata(state: GraphState) -> dict[str, Any] | None:
    """Return a persisted patch-dependency repair request, if one exists."""
    build_state = state.get("plan_build_state")
    if not isinstance(build_state, dict):
        return None
    metadata = build_state.get("metadata")
    if not isinstance(metadata, dict):
        return None
    repair = metadata.get("incremental_dependency_repair")
    if not isinstance(repair, dict):
        return None
    missing = repair.get("missing_universe_ids")
    if not isinstance(missing, list) or not all(isinstance(item, str) for item in missing):
        return None
    return repair


def _prepare_incremental_dependency_regeneration(
    build_state_data: Any,
    repair: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Persist a real, bounded dependency invalidation before graph replay.

    The legacy axial-universe path previously emitted a retry request but left
    the old valid ``UniversesPatch`` intact.  A subsequent incremental run
    therefore skipped it and retried the same invalid axial patch.  This
    helper invalidates only the universe owner and its true dependents; the
    failed axial producer is included explicitly because its replacement-ID
    reference is the origin of this retry, even though that relationship is
    dynamic rather than a static dependency-graph edge.
    """
    if not isinstance(build_state_data, dict):
        return {}, []
    try:
        from openmc_agent.plan_builder.dependency_graph import (
            DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH,
        )

        build_state = PlanBuildState.model_validate(build_state_data)
        targets = DEFAULT_PLAN_PATCH_DEPENDENCY_GRAPH.transitive_dependents(
            ["universes"]
        )
        for patch_type in ("axial_layers", "axial_overlays"):
            if patch_type not in targets:
                targets.append(patch_type)
        invalidated = build_state.invalidate_patch_types(
            targets,
            reason="axial replacement-universe dependency retry",
            issues=[
                {
                    "code": "lattice_transform.replacement_universe_missing",
                    "actual": universe_id,
                }
                for universe_id in repair.get("missing_universe_ids", [])
                if isinstance(universe_id, str)
            ],
        )
        # Preserve a typed request for the executor, including the failed
        # consumer, so it supplies the exact IDs to the regenerated universe
        # prompt and then resumes only the invalidated downstream path.
        request = build_state.metadata.get("plan_validation_repair")
        if isinstance(request, dict):
            request["target_patch_types"] = ["universes", "axial_layers"]
        build_state.add_event(
            "planning.incremental_dependency_regeneration_prepared",
            "invalidated universe owner and downstream patches for axial dependency retry",
            {
                "missing_universe_ids": list(repair.get("missing_universe_ids", [])),
                "invalidated_patch_types": targets,
                "invalidated_patch_ids": invalidated,
            },
        )
        return build_state.model_dump(mode="json"), targets
    except Exception:
        # Never discard the persisted retry request merely because an older
        # checkpoint cannot be reconstructed.  The executor will retain its
        # previous safe failure behaviour and report the original issue.
        return build_state_data, []


def _make_assess_plan_capability_node(max_retries: int):
    def _assess_plan_capability(state: GraphState) -> GraphState:
        _progress(state, "assess_capability", "checking executor support for structured plan")
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        if plan is None:
            return {}

        capability = _capability_for_plan(plan)
        # P0-D5 minimal alignment: surface axial-loading structural defects that
        # can_render's static checks miss, so the real blocker is visible before
        # the expert panel asks material questions. Applied at the graph assess
        # node only (not in _capability_for_plan) so other callers are unaffected.
        probe_issues = _probe_axial_materialization_blockers(plan)
        if probe_issues and capability.renderability in {"exportable", "runnable"}:
            probe_codes = [issue.code for issue in probe_issues]
            capability = capability.model_copy(update={
                "renderability": "skeleton",
                "is_executable": False,
                "issues": [*capability.issues, *probe_issues],
                "reasons": [
                    *capability.reasons,
                    "axial lattice materialization would fail (pre-render probe): "
                    + "; ".join(probe_codes),
                ],
            })
        updated_plan = plan.model_copy(update={"capability_report": capability})
        report = state.get("validation_report") or ValidationReport(is_valid=True)
        warnings = [
            warning
            for warning in report.warnings
            if warning != "Complex OpenMC IR was generated, but this executor version cannot render it yet."
        ]
        suggestions = [
            suggestion
            for suggestion in report.suggestions
            if suggestion
            != "Review complex_model and capability_report before implementing a renderer for this subsystem."
        ]

        if capability.renderability == "none":
            message = "; ".join(capability.reasons) or "no renderer can handle this plan"
            if message not in warnings:
                warnings.append(message)
            suggestions.append(
                "Use complex_model as reviewed IR before adding a renderer for this subsystem."
            )
        elif capability.renderability == "skeleton":
            warnings.append(
                "Renderer produced a review-only model.py skeleton; the model is NOT executable."
            )
            suggestions.append(
                "Fill the gaps listed in capability_report.json and TODO.md before exporting XML."
            )

        capability_warnings = [
            warning for warning in capability.warnings if warning not in warnings
        ]
        warnings.extend(capability_warnings)

        # Structural plan defects (a universe pointing at a missing cell, a pin-count
        # mismatch, a bad radius) are LLM typos, not missing expert facts. Route them
        # back to the LLM via reflect_plan instead of asking the expert, who can only
        # supply material values. Injecting them as validation errors makes the existing
        # reflect_plan path fire automatically; the assess_capability conditional edge
        # then picks reflect vs ask_expert based on retry_count.
        events = list(state.get("human_loop_events", []))
        retry_count = state.get("retry_count", 0)
        repair_issues: list[ValidationIssue] = (
            _capability_self_repair_errors(capability)
            if capability.renderability in {"none", "skeleton"}
            else []
        )
        repair_errors = [issue.message for issue in repair_issues]
        existing_issues = list(report.issues)
        deterministic_repair_available = bool(
            repair_issues
            and auto_repair_lattice_structure(
                updated_plan,
                repair_issues,
                requirement=state.get("requirement", ""),
            )
        )
        inject_invalid = bool(repair_errors) and (
            retry_count < max_retries or deterministic_repair_available
        )
        if inject_invalid:
            events.append(
                {
                    "event": "capability_structure_errors_delegated_to_reflect",
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "errors": repair_errors,
                    "reason": (
                        "structural plan defects are agent-fixable; routing to reflect_plan "
                        "instead of asking the expert"
                    ),
                }
            )
            updated_report = ValidationReport(
                is_valid=False,
                errors=list(repair_errors),
                warnings=warnings,
                suggestions=suggestions,
                issues=[*existing_issues, *repair_issues],
            )
        else:
            updated_report = ValidationReport(
                is_valid=report.is_valid,
                errors=report.errors,
                warnings=warnings,
                suggestions=suggestions,
                issues=existing_issues,
            )

        _progress(
            state,
            "assess_capability",
            (
                f"renderer={capability.supported_renderer} "
                f"renderability={capability.renderability}"
                + (f"; {len(repair_errors)} self-repair error(s)" if repair_errors else "")
            ),
        )
        updates: GraphState = {
            "simulation_plan": updated_plan,
            "simulation_spec": updated_plan.model_spec,
            "validation_report": updated_report,
            "capability_repair_errors": repair_errors,
            "human_loop_events": events,
            "plan_artifacts": _write_final_simulation_plan(state, updated_plan),
            **_trace_event_update(
                state,
                "capability_assessed",
                summary=(
                    f"renderer={capability.supported_renderer} "
                    f"renderability={capability.renderability}"
                ),
                report=updated_report,
                capability=capability,
                plan=updated_plan,
                metadata={
                    "unsupported_subsystems": capability.unsupported_subsystems,
                    "required_human_confirmations": capability.required_human_confirmations,
                    "self_repair_error_count": len(repair_errors),
                },
            ),
        }
        if inject_invalid and _incremental_reflect_plan_disabled(state):
            _progress(
                state,
                "assess_capability",
                "incremental capability failure needs regeneration; returning diagnostics to patch planner",
            )
            updates.update(
                {
                    "simulation_plan": None,
                    "simulation_spec": None,
                    "retry_count": retry_count + 1,
                    "error": "",
                    "incremental_regeneration_pending": True,
                    "plan_build_state": {},
                    "requirement": _incremental_regeneration_requirement(
                        state.get("requirement", ""), updated_report
                    ),
                }
            )
        return updates

    return _assess_plan_capability


def _plan_generation_router(state: GraphState) -> str:
    """Route any typed plan-gate wait state into the shared interrupt.

    Priority:
      1. plan Gate awaiting_human (existing reviewer gate interrupt),
      2. retry awaiting_human (typed human question from the retry loop),
      3. retry_pending (Phase-3B executable retry has work),
      4. validate.
    """
    build_state = state.get("plan_build_state") or {}
    incremental_result = state.get("incremental_execution_result") or {}
    if isinstance(incremental_result, dict):
        summary = incremental_result.get("summary") or {}
        if isinstance(summary, dict) and summary.get("stopped_after_gate"):
            return "stop"
    stages = build_state.get("plan_loop_stages", {}) if isinstance(build_state, dict) else {}
    awaiting_plan_gate = isinstance(stages, dict) and any(
        isinstance(stage, dict) and stage.get("status") == "awaiting_human"
        for stage in stages.values()
    )
    if awaiting_plan_gate:
        return "ask_plan_expert"
    # Phase-3B: if there are pending retry requests and the loop is not in
    # advisory/off mode, route into execute_plan_retry instead of validate.
    pending_retry_ids = build_state.get("plan_retry_pending_request_ids", []) if isinstance(build_state, dict) else []
    plan_loop_mode = (build_state.get("plan_loop_mode") or "off") if isinstance(build_state, dict) else "off"
    if pending_retry_ids and plan_loop_mode == "controlled":
        return "execute_plan_retry"
    return "validate"


def _plan_human_router(state: GraphState) -> str:
    """Distinguish gate-driven human questions from retry-driven ones.

    A retry human question sets ``plan_retry_human_resume=True`` so the
    answer is routed through ``resume_plan_retry`` (which consumes the typed
    answer and re-enters the retry loop) rather than
    ``resume_plan_closed_loop`` (which re-runs the incremental planner).
    """
    if state.get("plan_retry_human_resume"):
        return "resume_plan_retry"
    return "resume" if state.get("plan_resume_requested") else "stop"


def _make_execute_plan_retry_node(
    *,
    patch_llm_client: Any | None,
    plan_reviewer_client: Any | None,
    plan_repair_client: Any | None,
) -> Callable[[GraphState], GraphState]:
    """Bind real planning clients to the Phase-3B retry node.

    Retry requests created by controlled plan gates must regenerate only the
    Python-selected owner patch.  The former global node invoked
    ``execute_plan_retry_loop`` without a producer registry or clients, so a
    request was recorded but could never produce a candidate.  Capturing the
    same real patch/reviewer/repair clients used by ``generate_plan`` keeps
    retry execution within the campaign's unified LLM recorder and budget.
    """

    def _execute(state: GraphState) -> GraphState:
        build_state_dict = state.get("plan_build_state") or {}
        if not isinstance(build_state_dict, dict):
            return {"plan_build_state": build_state_dict}
        from openmc_agent.plan_builder.state import PlanBuildState
        from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
        from openmc_agent.plan_builder.closed_loop.retry_candidate_producers import (
            default_producer_registry,
        )
        from openmc_agent.plan_builder.closed_loop.retry_controller import (
            execute_plan_retry_loop,
        )
        from openmc_agent.plan_builder.executor import build_generation_context_from_state
        from openmc_agent.plan_builder.patch_generator import generate_patch

        build_state = PlanBuildState.model_validate(build_state_dict)
        policy_dict = build_state.plan_loop_policy or {}
        policy = PlanClosedLoopPolicy.model_validate(policy_dict)
        registry = default_producer_registry(
            generate_patch_fn=generate_patch,
            build_context_fn=build_generation_context_from_state,
        )
        outcome = execute_plan_retry_loop(
            state=build_state,
            policy=policy,
            producer_registry=registry,
            requirement=str(state.get("requirement") or ""),
            proposer_client=patch_llm_client,
            reviewer_client=plan_reviewer_client,
            repair_client=plan_repair_client,
            # Commit one owner patch then return to generate_plan for the
            # authoritative downstream rebuild and controlled Gate replay.
            # Processing multiple stale requests against the same pre-replay
            # snapshot can otherwise re-block an invalidated stage before
            # the planner sees the owner change.
            max_rounds=1,
        )
        return {
            "plan_build_state": build_state.model_dump(mode="json"),
            "plan_retry_outcome": outcome.model_dump(mode="json"),
            # Clear the intermediate incremental failure so the graph can
            # return to generate_plan after owner-patch commit/invalidation.
            "error": "",
        }

    return _execute


def _execute_plan_retry(state: GraphState) -> GraphState:
    """Run one pass of the executable retry loop.

    The actual producer/acceptance/downstream/gate-replay injection is done
    inside the executor; this node simply triggers it and records the outcome.
    The node never clears ``plan_build_state`` or mutates the requirement.
    """
    build_state_dict = state.get("plan_build_state") or {}
    if not isinstance(build_state_dict, dict):
        return {"plan_build_state": build_state_dict}
    from openmc_agent.plan_builder.state import PlanBuildState
    from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanLoopMode
    from openmc_agent.plan_builder.closed_loop.retry_controller import execute_plan_retry_loop

    build_state = PlanBuildState.model_validate(build_state_dict)
    policy_dict = build_state.plan_loop_policy or {}
    mode = (policy_dict.get("mode") or build_state.plan_loop_mode or PlanLoopMode.OFF).value if hasattr((policy_dict.get("mode") or build_state.plan_loop_mode or PlanLoopMode.OFF), "value") else str(policy_dict.get("mode") or build_state.plan_loop_mode or "off")
    policy = PlanClosedLoopPolicy(**{**policy_dict, "mode": mode})
    if policy.mode is PlanLoopMode.OFF or policy.mode is PlanLoopMode.ADVISORY:
        outcome = execute_plan_retry_loop(state=build_state, policy=policy)
    else:
        # Controlled mode: the producer/acceptance injection is wired by the
        # executor; this graph node is a lightweight entry point that records
        # the outcome.  The executor's own retry path remains authoritative.
        outcome = execute_plan_retry_loop(state=build_state, policy=policy)
    return {
        "plan_build_state": build_state.model_dump(mode="json"),
        "plan_retry_outcome": outcome.model_dump(mode="json"),
    }


def _resume_plan_retry(state: GraphState) -> GraphState:
    """Consume a typed human answer and re-enter the retry loop.

    Marks the prior question's request as superseded so the loop cannot
    accidentally consume a stale answer bound to a different input hash.
    """
    build_state_dict = state.get("plan_build_state") or {}
    if not isinstance(build_state_dict, dict):
        return {"plan_build_state": build_state_dict, "plan_retry_human_resume": False}
    from openmc_agent.plan_builder.state import PlanBuildState
    from openmc_agent.plan_builder.closed_loop.retry_models import RetryRequestLifecycle

    build_state = PlanBuildState.model_validate(build_state_dict)
    answers = state.get("plan_human_answers", {}) or {}
    for question_id, answer in answers.items():
        request_id = build_state.plan_retry_human_questions.get(question_id, {}).get("retry_request_id") if isinstance(build_state.plan_retry_human_questions.get(question_id, None), dict) else None
        if request_id and request_id in build_state.plan_retry_requests:
            req = build_state.plan_retry_requests[request_id]
            req.lifecycle = RetryRequestLifecycle.SUPERSEDED
            if request_id in build_state.plan_retry_pending_request_ids:
                build_state.plan_retry_pending_request_ids.remove(request_id)
            build_state.add_event("planning.retry_human_answered", "typed human answer consumed; request superseded", {"request_id": request_id, "question_id": question_id})
    return {
        "plan_build_state": build_state.model_dump(mode="json"),
        "plan_retry_human_resume": False,
        "plan_resume_requested": False,
    }


def _ask_plan_expert(state: GraphState) -> GraphState:
    """Issue a typed plan-gate interrupt without touching expert_feedback."""
    build_state = state.get("plan_build_state") or {}
    raw_questions = build_state.get("plan_human_questions", {}) if isinstance(build_state, dict) else {}
    questions = list(raw_questions.values()) if isinstance(raw_questions, dict) else []
    unanswered = [q for q in questions if isinstance(q, dict) and q.get("question_id") not in state.get("plan_human_answers", {})]
    if not unanswered:
        return {"plan_human_questions": questions, "plan_awaiting_human": False, "plan_resume_requested": False}
    if not state.get("plan_human_interactive", False):
        return {"plan_human_questions": unanswered, "plan_awaiting_human": True, "plan_resume_requested": False}
    payload = interrupt({
        "kind": "plan_gate_human_confirmation",
        "questions": unanswered,
        "instruction": "Submit a typed HumanPlanAnswer for each question_id; do not provide free-text patch JSON.",
    })
    answer_payload = payload.get("answers", payload) if isinstance(payload, dict) else {}
    if not isinstance(answer_payload, dict):
        return {"plan_human_questions": unanswered, "plan_awaiting_human": True, "plan_resume_requested": False}
    return {
        "plan_human_questions": unanswered,
        "plan_human_answers": answer_payload,
        "plan_awaiting_human": False,
        "plan_resume_requested": True,
    }


def _resume_plan_closed_loop(state: GraphState) -> GraphState:
    """Persist validated typed answers into PlanBuildState before regeneration."""
    from openmc_agent.plan_builder.closed_loop.models import HumanPlanAnswer
    from openmc_agent.plan_builder.state import PlanBuildState

    try:
        build_state = PlanBuildState.model_validate(state.get("plan_build_state") or {})
    except Exception:
        return {"plan_resume_requested": False, "plan_awaiting_human": True}
    answers = state.get("plan_human_answers", {})
    if not isinstance(answers, dict):
        return {"plan_resume_requested": False, "plan_awaiting_human": True}
    invalid: list[str] = []
    for question_id, raw in answers.items():
        try:
            answer = HumanPlanAnswer.model_validate({**raw, "question_id": question_id} if isinstance(raw, dict) else raw)
            question = build_state.plan_human_questions.get(question_id)
            if question is None:
                raise ValueError("unknown question")
            option_ids = {option.option_id for option in question.options}
            if answer.selected_option_id is not None and answer.selected_option_id not in option_ids:
                raise ValueError("invalid option")
            build_state.plan_human_answers[question_id] = answer
        except Exception:
            invalid.append(str(question_id))
    if invalid:
        return {"plan_build_state": build_state.model_dump(mode="json"), "plan_awaiting_human": True, "plan_resume_requested": False}
    return {
        "plan_build_state": build_state.model_dump(mode="json"),
        "plan_awaiting_human": False,
        "plan_resume_requested": False,
    }


def _ask_expert(state: GraphState) -> GraphState:
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    plan_update = {"simulation_plan": plan} if plan is not None else {}
    questions = _pending_expert_questions(state)
    max_rounds = state.get("max_expert_rounds", 0)
    round_count = state.get("expert_round_count", 0)
    events = list(state.get("human_loop_events", []))

    if not questions:
        trace_update = _trace_event_update(
            state,
            "ask_expert_completed",
            summary="no expert questions pending",
            metadata={"question_count": 0, "feedback_present": False},
        )
        return {
            **plan_update,
            "pending_expert_questions": [],
            "awaiting_expert_feedback": False,
            "needs_regeneration": False,
            "expert_feedback_action": "none",
            "human_loop_events": state.get("human_loop_events", []),
            **trace_update,
        }

    if round_count >= max_rounds:
        _progress(
            state,
            "ask_expert",
            (
                f"expert questions remain but max rounds reached "
                f"({round_count}/{max_rounds})"
            ),
        )
        events.append(
            {
                "event": "expert_questions_not_asked",
                "round": round_count,
                "questions": questions,
                "reason": "max_expert_rounds reached or expert loop disabled",
            }
        )
        start_update = _trace_event_update(
            state,
            "ask_expert_started",
            summary=f"{len(questions)} expert question(s) pending",
            metadata={
                "question_count": len(questions),
                "questions": questions,
                "feedback_present": False,
            },
        )
        completed_update = _trace_event_update(
            {**state, **start_update},
            "ask_expert_completed",
            summary="expert questions not asked; max rounds reached or disabled",
            metadata={
                "question_count": len(questions),
                "questions": questions,
                "feedback_present": False,
                "requires_human_confirmation_count": len(questions),
            },
        )
        return {
            **plan_update,
            "pending_expert_questions": questions,
            "awaiting_expert_feedback": False,
            "needs_regeneration": False,
            "expert_feedback_action": "none",
            "human_loop_events": events,
            **completed_update,
        }

    prompt_round = round_count + 1
    _progress(
        state,
        "ask_expert",
        f"interrupting for expert feedback round {prompt_round}/{max_rounds}",
    )
    # Surface the *real* execution blocker (structural/environment) in the panel
    # header so material assumptions cannot mask it. Computed deterministically;
    # never asks an LLM.
    blocker_payload = _capability_blocker_payload(state)
    question_group_ids = [
        g.get("question_id") for g in state.get("expert_question_groups", [])
        if isinstance(g, dict)
    ]
    # Record classification + grouping as TRACE events (observability), not as
    # human-loop interaction events, so they never shift human_loop_events order.
    classified_update = _trace_event_update(
        state,
        "capability_blockers_classified",
        summary=(
            f"primary_blockers={blocker_payload.get('primary_blocker_codes', [])} "
            f"has_blocking={blocker_payload.get('has_blocking_issue')}"
        ),
        metadata={
            "primary_blocker_codes": blocker_payload.get("primary_blocker_codes", []),
            "has_blocking_issue": blocker_payload.get("has_blocking_issue"),
            "structural_issue_not_visible_to_validate_plan": blocker_payload.get(
                "structural_issue_not_visible_to_validate_plan"
            ),
        },
        round_index=prompt_round,
    )
    grouped_update = _trace_event_update(
        {**state, **classified_update},
        "expert_questions_grouped",
        summary=f"{len(state.get('expert_question_groups', []))} question group(s)",
        metadata={
            "group_count": len(state.get("expert_question_groups", [])),
            "group_ids": question_group_ids,
        },
        round_index=prompt_round,
    )
    start_update = _trace_event_update(
        {**state, **grouped_update},
        "ask_expert_started",
        summary=f"asking {len(questions)} expert question(s)",
        metadata={
            "question_count": len(questions),
            "questions": questions,
            "group_count": len(state.get("expert_question_groups", [])),
            "primary_blocker_codes": blocker_payload.get("primary_blocker_codes", []),
        },
        round_index=prompt_round,
    )
    resume_payload = interrupt(
        {
            "kind": "expert_feedback_request",
            "round": prompt_round,
            "max_rounds": max_rounds,
            "questions": questions,
            "capability_blockers": blocker_payload,
            "instruction": (
                "Provide expert feedback that fills, corrects, or explicitly defers "
                "the missing modeling facts."
            ),
        }
    )
    feedback_items, should_continue = _feedback_from_resume_payload(resume_payload)
    event = {
        "event": "expert_feedback_received",
        "round": prompt_round,
        "questions": questions,
        "feedback": feedback_items,
        "should_continue": should_continue,
    }
    events.append(event)
    resolved_items = _update_resolved_expert_items_from_feedback(
        state=state,
        questions=questions,
        feedback_items=feedback_items,
        round_index=prompt_round,
    )
    if resolved_items:
        events.append(
            {
                "event": "expert_feedback_resolved_items_extracted",
                "round": prompt_round,
                "items": resolved_items,
                "reason": "bound current pending questions to the expert resume payload",
            }
        )

    decision_action = _decision_action_from_resume_payload(resume_payload)
    if decision_action:
        # Explicit :command decision. Build the decision deterministically,
        # create run-level acknowledgements (never mutating plan physics), and
        # record a trace event. Material assumptions are NOT forwarded to any
        # repair LLM: only the structural blocker codes are carried.
        from openmc_agent.expert_feedback import (
            ExpertFeedbackDecision,
            build_assumption_acknowledgements,
        )
        acknowledged_ids = (
            question_group_ids
            if decision_action == "accept_assumptions_for_this_run"
            else []
        )
        deferred_ids = (
            question_group_ids
            if decision_action == "defer_confirmations"
            else []
        )
        decision = ExpertFeedbackDecision(
            action=decision_action,  # type: ignore[arg-type]
            feedback_items=feedback_items,
            acknowledged_question_ids=acknowledged_ids,
            deferred_question_ids=deferred_ids,
            reason=f"explicit expert command {decision_action!r}",
        )
        event_name = {
            "accept_assumptions_for_this_run": "expert_assumptions_accepted_for_run",
            "accept_review_only": "expert_review_only_accepted",
            "defer_confirmations": "expert_confirmations_deferred",
            "abort": "expert_feedback_aborted",
            "continue_repair": "structural_blocker_routed_before_human_confirmation",
            "provide_corrections": "expert_feedback_classified",
        }.get(decision_action, "expert_feedback_classified")
        events.append(
            {
                "event": event_name,
                "round": prompt_round,
                "action": decision_action,
                "question_ids": question_group_ids,
                "reason": decision.reason,
            }
        )
        acks = build_assumption_acknowledgements(
            decision,
            question_ids=question_group_ids,
            round_index=prompt_round,
            plan_hash=_plan_hash(_coerce_simulation_plan(state.get("simulation_plan"))),
        )
        completed_update = _trace_event_update(
            {**state, **start_update},
            "ask_expert_completed",
            summary=f"expert decision -> {decision_action}",
            metadata={
                "question_count": len(questions),
                "expert_feedback_action": decision_action,
                "acknowledgement_count": len(acks),
            },
            round_index=prompt_round,
        )
        return {
            **plan_update,
            "pending_expert_questions": [] if decision_action != "defer_confirmations" else questions,
            "expert_round_count": prompt_round,
            "awaiting_expert_feedback": False,
            "needs_regeneration": False,
            "expert_feedback_action": "continue",
            "expert_feedback_decision": decision.model_dump(mode="json"),
            "expert_assumption_acknowledgements": [
                a.model_dump(mode="json") for a in acks
            ],
            "resolved_expert_items": resolved_items,
            "human_loop_events": events,
            **completed_update,
        }

    if not should_continue or not feedback_items:
        # Empty feedback no longer means a vague "accept the current artifact and
        # continue". It is an explicit, traceable decision: defer confirmations
        # for a runnable model, or accept the review-only skeleton for a blocked
        # one (outcome=BLOCKED_REVIEW_ONLY, never a silent success).
        from openmc_agent.expert_feedback import (
            interpret_empty_feedback,
            build_assumption_acknowledgements,
        )
        renderability = blocker_payload.get("renderability", "runnable")
        has_blocking = bool(blocker_payload.get("has_blocking_issue"))
        decision = interpret_empty_feedback(
            renderability=renderability,
            has_blocking_issue=has_blocking,
        )
        event_name = (
            "expert_review_only_accepted"
            if decision.action == "accept_review_only"
            else "expert_confirmations_deferred"
        )
        events.append(
            {
                "event": event_name,
                "round": prompt_round,
                "action": decision.action,
                "reason": decision.reason,
                "renderability": renderability,
                "has_blocking_issue": has_blocking,
            }
        )
        acks = build_assumption_acknowledgements(
            decision,
            question_ids=question_group_ids,
            round_index=prompt_round,
            plan_hash=_plan_hash(_coerce_simulation_plan(state.get("simulation_plan"))),
        )
        completed_update = _trace_event_update(
            {**state, **start_update},
            "ask_expert_completed",
            summary=f"expert feedback empty -> {decision.action}",
            metadata={
                "question_count": len(questions),
                "questions": questions,
                "feedback_present": bool(feedback_items),
                "expert_feedback_action": decision.action,
                "requires_human_confirmation_count": len(questions),
            },
            round_index=prompt_round,
        )
        # Routing falls through to render (the renderer emits a review-only
        # skeleton for blocked models; execute_tools then skips). The
        # workflow_outcome label is what distinguishes this from a real PASS.
        return {
            **plan_update,
            "pending_expert_questions": questions,
            "expert_round_count": prompt_round,
            "awaiting_expert_feedback": False,
            "needs_regeneration": False,
            "expert_feedback_action": "continue",
            "expert_feedback_decision": decision.model_dump(mode="json"),
            "expert_assumption_acknowledgements": [
                a.model_dump(mode="json") for a in acks
            ],
            "resolved_expert_items": resolved_items,
            "human_loop_events": events,
            **completed_update,
        }

    feedback = list(state.get("expert_feedback", []))
    feedback.extend(feedback_items)
    completed_update = _trace_event_update(
        {**state, **start_update},
        "ask_expert_completed",
        summary=f"received {len(feedback_items)} expert feedback item(s)",
        metadata={
            "question_count": len(questions),
            "questions": questions,
            "feedback_present": True,
            "feedback_count": len(feedback_items),
        },
        round_index=prompt_round,
    )
    return {
        **plan_update,
        "expert_feedback": feedback,
        "pending_expert_questions": [],
        "expert_round_count": prompt_round,
        "awaiting_expert_feedback": False,
        "needs_regeneration": False,
        "expert_feedback_action": "classify",
        "resolved_expert_items": resolved_items,
        "human_loop_events": events,
        "error": "",
        **completed_update,
    }


def _classify_expert_feedback(state: GraphState) -> GraphState:
    feedback = _latest_expert_feedback(state)
    events = list(state.get("human_loop_events", []))
    round_index = state.get("expert_round_count", 0)
    if not feedback:
        action = "continue"
        reason = "empty or missing expert feedback"
        confidence = "high"
    else:
        action, reason, confidence = _classify_feedback_text(
            feedback,
            _coerce_simulation_plan(state.get("simulation_plan")),
        )

    event = {
        "event": "expert_feedback_classified",
        "round": round_index,
        "action": action,
        "reason": reason,
        "confidence": confidence,
        "feedback": feedback,
    }
    events.append(event)
    if action == "regenerate_plan":
        events.append(
            {
                "event": "expert_feedback_regeneration_selected",
                "round": round_index,
                "action": action,
                "reason": reason,
                "confidence": confidence,
            }
        )
    elif action == "continue":
        events.append(
            {
                "event": "expert_feedback_continue_selected",
                "round": round_index,
                "action": action,
                "reason": reason,
                "confidence": confidence,
            }
        )

    return {
        "expert_feedback_action": action,
        "expert_feedback_interpretation": reason,
        "patch_confidence": confidence,
        "patch_reason": reason,
        "needs_regeneration": action == "regenerate_plan",
        "human_loop_events": events,
    }


def _patch_plan_from_expert_feedback(state: GraphState) -> GraphState:
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    feedback = _latest_expert_feedback(state)
    events = list(state.get("human_loop_events", []))
    round_index = state.get("expert_round_count", 0)
    if plan is None:
        error = "cannot patch because SimulationPlan is missing"
        events.extend(
            [
                {
                    "event": "plan_patch_failed",
                    "round": round_index,
                    "error": error,
                    "reason": error,
                },
                {
                    "event": "patch_failed_fallback_to_regeneration",
                    "round": round_index,
                    "reason": error,
                    "action": "regenerate_plan",
                },
            ]
        )
        return {
            "patch_error": error,
            "expert_feedback_action": "regenerate_plan",
            "needs_regeneration": True,
            "human_loop_events": events,
        }

    patches, reason, confidence = _build_plan_patches(plan, feedback, state)
    events.append(
        {
            "event": "plan_patch_generated",
            "round": round_index,
            "reason": reason,
            "action": "patch_plan",
            "confidence": confidence,
            "patch": patches,
        }
    )
    if not patches:
        error = "expert feedback could not be mapped to a safe SimulationPlan field"
        events.extend(
            [
                {
                    "event": "plan_patch_failed",
                    "round": round_index,
                    "reason": reason,
                    "error": error,
                },
                {
                    "event": "patch_failed_fallback_to_regeneration",
                    "round": round_index,
                    "reason": error,
                    "action": "regenerate_plan",
                },
            ]
        )
        return {
            "plan_patch": patches,
            "patch_confidence": confidence,
            "patch_reason": reason,
            "patch_error": error,
            "expert_feedback_action": "regenerate_plan",
            "needs_regeneration": True,
            "human_loop_events": events,
        }

    try:
        patched_payload = _apply_json_patches(plan.model_dump(mode="json"), patches)
        patched_payload = _normalize_capability_report_for_plan_validation(patched_payload)
        patched_plan = SimulationPlan.model_validate(patched_payload)
    except Exception as exc:
        error = str(exc)
        events.extend(
            [
                {
                    "event": "plan_patch_failed",
                    "round": round_index,
                    "reason": reason,
                    "error": error,
                    "patch": patches,
                },
                {
                    "event": "patch_failed_fallback_to_regeneration",
                    "round": round_index,
                    "reason": "patched plan failed schema validation",
                    "action": "regenerate_plan",
                },
            ]
        )
        return {
            "plan_patch": patches,
            "patch_confidence": confidence,
            "patch_reason": reason,
            "patch_error": error,
            "expert_feedback_action": "regenerate_plan",
            "needs_regeneration": True,
            "human_loop_events": events,
        }

    events.append(
        {
            "event": "plan_patch_applied",
            "round": round_index,
            "reason": reason,
            "action": "patch_plan",
            "confidence": confidence,
            "patch": patches,
        }
    )
    return {
        "simulation_plan": patched_plan,
        "simulation_spec": patched_plan.model_spec,
        "plan_patch": patches,
        "patch_confidence": confidence,
        "patch_reason": reason,
        "patch_error": "",
        "expert_feedback_action": "none",
        "needs_regeneration": False,
        "human_loop_events": events,
        "plan_artifacts": _write_final_simulation_plan(state, patched_plan),
        "error": "",
    }


def _render_script(state: GraphState) -> GraphState:
    _progress(state, "render_script", "rendering OpenMC Python model.py")
    start_update = _trace_event_update(
        state,
        "render_started",
        summary="rendering SimulationSpec to model.py",
    )
    report = state.get("validation_report")
    spec = state.get("simulation_spec")
    if spec is None or report is None or not report.is_valid:
        return start_update

    script = render_openmc_script(spec)
    script_report = validate_openmc_script(script, spec)
    if not script_report.is_valid:
        _progress(state, "render_script", f"failed script validation: {script_report.errors}")
        completed_update = _trace_event_update(
            {**state, **start_update},
            "render_completed",
            summary="SimulationSpec render failed script validation",
            report=script_report,
            metadata={"success": False},
        )
        return {
            "validation_report": script_report,
            "error": "; ".join(script_report.errors),
            **completed_update,
        }

    output_dir = Path(state.get("output_dir", "data/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    _clean_stale_render_artifacts(output_dir)
    model_path = output_dir / "model.py"
    model_path.write_text(script, encoding="utf-8")
    _progress(state, "render_script", f"wrote {model_path}")
    completed_update = _trace_event_update(
        {**state, **start_update},
        "render_completed",
        summary=f"wrote {model_path}",
        report=report,
        metadata={"success": True, "model_path": str(model_path)},
    )
    return {"script": script, "model_path": str(model_path), **completed_update}


def _render_plan_script(state: GraphState) -> GraphState:
    _progress(state, "render_plan_script", "rendering OpenMC Python model.py from SimulationPlan")
    start_update = _trace_event_update(
        state,
        "render_started",
        summary="rendering SimulationPlan to model.py",
    )
    output_dir = Path(state.get("output_dir", "data/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    # Always start from a clean render-output set so a non-exportable run never
    # leaves a prior run's model.py / XML / optimistic capability_report.json.
    _clean_stale_render_artifacts(output_dir)

    report = state.get("validation_report")
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    if plan is None or report is None or not report.is_valid:
        _progress(state, "render_plan_script", "skipped: plan is not valid")
        _write_non_executable_marker(output_dir, report, plan)
        return start_update

    renderer, capability = choose_renderer(plan)
    if renderer is None or capability.renderability == "none":
        _progress(state, "render_plan_script", "skipped: no renderer for this plan")
        _write_non_executable_marker(output_dir, report, plan, capability)
        completed_update = _trace_event_update(
            {**state, **start_update},
            "render_completed",
            summary="render skipped: no renderer",
            capability=capability,
            metadata={"success": False, "reason": "no renderer"},
        )
        return {
            "simulation_plan": plan,
            "plan_artifacts": _write_final_simulation_plan(state, plan),
            **completed_update,
        }

    result = renderer.render(plan, output_dir)
    # Rendering may defensively downgrade an apparently exportable plan to a
    # skeleton if executor/script validation fails. The render result is the
    # final authority for sidecars and downstream tool execution.
    plan = plan.model_copy(update={"capability_report": result.capability})
    _write_capability_sidecar(output_dir, result.capability)

    if result.errors:
        _progress(state, "render_plan_script", f"renderer failed: {result.errors}")
        result_report = ValidationReport(
            is_valid=False,
            errors=result.errors,
        )
        completed_update = _trace_event_update(
            {**state, **start_update},
            "render_completed",
            summary="renderer failed",
            report=result_report,
            capability=result.capability,
            plan=plan,
            metadata={
                "success": False,
                "renderer": result.renderer_name,
                "errors": result.errors,
            },
        )
        return {
            "simulation_plan": plan,
            "simulation_spec": plan.model_spec,
            "validation_report": result_report,
            "plan_artifacts": _write_final_simulation_plan(state, plan),
            "error": "; ".join(result.errors),
            **completed_update,
        }

    model_path = output_dir / "model.py"
    _progress(
        state,
        "render_plan_script",
        f"wrote {model_path} (renderer={result.renderer_name}, "
        f"renderability={result.renderability})",
    )
    if result.renderability in {"skeleton", "none"}:
        _enrich_skeleton_todo(output_dir, plan, state)
    completed_update = _trace_event_update(
        {**state, **start_update},
        "render_completed",
        summary=(
            f"wrote {model_path} with renderer={result.renderer_name} "
            f"renderability={result.renderability}"
        ),
        report=report,
        capability=result.capability,
        plan=plan,
        metadata={
            "success": True,
            "renderer": result.renderer_name,
            "model_path": str(model_path),
        },
    )
    return {
        "simulation_plan": plan,
        "simulation_spec": plan.model_spec,
        "script": result.script,
        "model_path": str(model_path),
        "plan_artifacts": _write_final_simulation_plan(state, plan),
        **completed_update,
    }


def _make_execute_tools_node(
    *,
    export_xml_tool: ExportXmlToolFn,
    plot_tool: PlotToolFn,
    smoke_test_tool: SmokeTestToolFn,
    geometry_debug_tool: GeometryDebugToolFn,
    enable_plots: bool,
    enable_smoke_test: bool,
):
    def _execute_tools(state: GraphState) -> GraphState:
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        model_path = state.get("model_path")
        if plan is None or not model_path:
            return {}

        renderability = plan.capability_report.renderability
        output_dir = Path(state.get("output_dir", "data/runs"))
        results: list[ToolResult] = []

        if renderability not in {"exportable", "runnable"}:
            _progress(
                state,
                "execute_tools",
                (
                    f"skipping export/run: renderability={renderability}, "
                    "model is not executable"
                ),
            )
            report = _execution_report_from_tool_results(results)
            return {
                "tool_results": [result.model_dump() for result in results],
                "validation_report": report,
                "error": "",
                **_trace_event_update(
                    state,
                    "export_xml_completed",
                    summary="export skipped because model is not executable",
                    report=report,
                    capability=plan.capability_report,
                    metadata={"success": False, "skipped": True, "renderability": renderability},
                ),
            }

        # ---- Stage 1: export_xml ----
        _progress(state, "execute_tools", "running export_xml")
        export_result = export_xml_tool(Path(model_path))
        results.append(export_result)
        _progress(state, "execute_tools", f"export_xml ok={export_result.ok}")
        trace_update = _trace_event_update(
            state,
            "export_xml_completed",
            summary=f"export_xml ok={export_result.ok}",
            report=ValidationReport.from_issues(
                export_result.issues,
                is_valid=export_result.ok,
            ),
            capability=plan.capability_report,
            metadata={
                "success": export_result.ok,
                "returncode": export_result.returncode,
                "error": export_result.error,
                "artifacts": export_result.artifacts,
            },
        )

        # Export failure: stop — no geometry debug, no smoke.
        if not export_result.ok:
            report = _execution_report_from_tool_results(results)
            return {
                "tool_results": [result.model_dump() for result in results],
                "validation_report": report,
                "error": "; ".join(report.errors),
                **trace_update,
            }

        # ---- Stage 2: optional plots ----
        if enable_plots and export_result.ok:
            _progress(state, "execute_tools", f"running run_geometry_plots requested_specs={len(plan.plot_specs)}")
            results.append(plot_tool(output_dir))
            _progress(state, "execute_tools", f"run_geometry_plots ok={results[-1].ok}")
        elif not enable_plots:
            _progress(state, "execute_tools", "skipping run_geometry_plots because --plot is disabled")

        # ---- Stage 3: geometry debug (isolated subdirectory) ----
        _progress(state, "execute_tools", "running run_geometry_debug")
        geom_debug_result = geometry_debug_tool(output_dir, plan)
        results.append(geom_debug_result)
        _progress(state, "execute_tools", f"run_geometry_debug ok={geom_debug_result.ok}")
        trace_update = _trace_event_update(
            {**state, **trace_update},
            "geometry_debug_completed",
            summary=f"run_geometry_debug ok={geom_debug_result.ok}",
            report=ValidationReport.from_issues(
                geom_debug_result.issues,
                is_valid=geom_debug_result.ok,
            ),
            capability=plan.capability_report,
            metadata={
                "success": geom_debug_result.ok,
                "returncode": geom_debug_result.returncode,
                "error": geom_debug_result.error,
                "artifacts": geom_debug_result.artifacts,
                "stage": "geometry_debug",
            },
        )

        # Geometry-debug failure: do not run smoke test.
        if not geom_debug_result.ok:
            _progress(
                state,
                "execute_tools",
                "skipping run_smoke_test: geometry_debug reported errors",
            )
            report = _execution_report_from_tool_results(results)
            history = list(state.get("retry_history", []))
            if history:
                history[-1]["tool_results"] = [result.model_dump() for result in results]
                history[-1]["execution_errors"] = report.errors
            return {
                "tool_results": [result.model_dump() for result in results],
                "validation_report": report,
                "retry_history": history,
                "error": "; ".join(report.errors),
                **trace_update,
            }

        # ---- Stage 4: smoke test (only if geometry debug passed) ----
        if (
            enable_smoke_test
            and export_result.ok
            and geom_debug_result.ok
            and renderability == "runnable"
            and plan.execution_check.enabled
        ):
            # Pre-flight source/settings check: do not waste a smoke run (and
            # risk a segfault cascade) when the source box provably misses the
            # active fuel region or the fuel is non-fissionable.
            from openmc_agent.source_settings import (
                source_bounds_for_plan,
                validate_source_settings,
            )
            from openmc_agent.geometry_bounds import validate_bounds_consistency

            source_issues = validate_source_settings(plan)
            if plan.complex_model is not None:
                src = source_bounds_for_plan(plan.complex_model)
                if src is not None:
                    src_tuple = (src.x_min, src.x_max, src.y_min, src.y_max, src.z_min, src.z_max)
                    source_issues.extend(validate_bounds_consistency(
                        plan.complex_model, source_bounds=src_tuple,
                        plot_bounds=_plot_bounds_metadata(plan),
                    ))
            blocking_source = [i for i in source_issues if i.severity == "error"]
            if blocking_source:
                _progress(
                    state,
                    "execute_tools",
                    f"skipping run_smoke_test: {len(blocking_source)} blocking source issue(s)",
                )
                source_report = ValidationReport.from_issues(source_issues, is_valid=False)
                # Inject a synthetic smoke_test ToolResult with the blocking source
                # issues so the runtime classifier can route to deterministic repair.
                from openmc_agent.tools import ToolResult
                source_tr = ToolResult(
                    name="run_smoke_test",
                    ok=False,
                    command=[],
                    returncode=1,
                    stdout="",
                    stderr="; ".join(i.message for i in blocking_source),
                    error="source/settings pre-flight blocked smoke test",
                    issues=blocking_source,
                    artifacts=[],
                )
                results.append(source_tr)
                trace_update = _trace_event_update(
                    {**state, **trace_update},
                    "smoke_test_completed",
                    summary="run_smoke_test skipped: source/settings pre-flight failed",
                    report=source_report,
                    metadata={
                        "success": False,
                        "skipped": True,
                        "source_issue_codes": [i.code for i in blocking_source],
                    },
                )
                return {
                    "validation_report": source_report,
                    "tool_results": [result.model_dump() for result in results],
                    **trace_update,
                }

            settings = plan.execution_check.settings
            _progress(
                state,
                "execute_tools",
                (
                    "running run_smoke_test "
                    f"batches={settings.batches} inactive={settings.inactive} "
                    f"particles={settings.particles}"
                ),
            )
            results.append(smoke_test_tool(output_dir, plan))
            _progress(state, "execute_tools", f"run_smoke_test ok={results[-1].ok}")
            smoke_result = results[-1]
            trace_update = _trace_event_update(
                {**state, **trace_update},
                "smoke_test_completed",
                summary=f"run_smoke_test ok={smoke_result.ok}",
                report=ValidationReport.from_issues(
                    smoke_result.issues,
                    is_valid=smoke_result.ok,
                ),
                capability=plan.capability_report,
                metadata={
                    "success": smoke_result.ok,
                    "returncode": smoke_result.returncode,
                    "error": smoke_result.error,
                },
            )
        elif not enable_smoke_test:
            _progress(state, "execute_tools", "skipping run_smoke_test because --smoke-test is disabled")
        elif renderability != "runnable":
            _progress(
                state,
                "execute_tools",
                f"skipping run_smoke_test: renderability={renderability} (not runnable)",
            )

        report = _execution_report_from_tool_results(results)
        _progress(
            state,
            "execute_tools",
            f"tool checks {'passed' if report.is_valid else 'failed'} with {len(report.errors)} error(s)",
        )
        history = list(state.get("retry_history", []))
        if history:
            history[-1]["tool_results"] = [result.model_dump() for result in results]
            history[-1]["execution_errors"] = report.errors

        return {
            "tool_results": [result.model_dump() for result in results],
            "validation_report": report,
            "retry_history": history,
            "error": "; ".join(report.errors),
            **trace_update,
        }

    return _execute_tools


# --------------------------------------------------------------------------- #
# Runtime feedback classification + deterministic repair nodes (R2/R3)
# --------------------------------------------------------------------------- #


def _make_classify_runtime_feedback_node():
    def _classify(state: GraphState) -> GraphState:
        """Classify runtime tool failures and route to repair or stop."""
        from openmc_agent.runtime_feedback import (
            RuntimeFailure,
            RuntimeFailureClass,
            classify_runtime_tool_results,
        )
        from openmc_agent.runtime_repair_policy import get_repair_policy

        tool_result_dicts = state.get("tool_results", [])
        if not tool_result_dicts:
            return {}

        # Reconstruct ToolResult objects for classification.
        tool_results: list[ToolResult] = []
        for tr in tool_result_dicts:
            if isinstance(tr, dict):
                issues = [
                    ValidationIssue.model_validate(i) if isinstance(i, dict) else i
                    for i in tr.get("issues", [])
                ]
                tool_results.append(ToolResult(
                    name=tr.get("name", ""),
                    ok=tr.get("ok", False),
                    command=tr.get("command", []),
                    returncode=tr.get("returncode"),
                    stdout=tr.get("stdout", ""),
                    stderr=tr.get("stderr", ""),
                    artifacts=tr.get("artifacts", []),
                    error=tr.get("error", ""),
                    issues=issues,
                ))
            else:
                tool_results.append(tr)

        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        plan_hash = None
        if plan is not None:
            plan_hash = stable_json_hash_from_plan(plan)

        failures = classify_runtime_tool_results(
            tool_results,
            plan_hash=plan_hash,
            stage="execute_tools",
        )

        if not failures:
            return {}

        primary = failures[0]
        trace_update = _trace_event_update(
            state,
            "runtime_feedback_classified",
            summary=f"Primary: {primary.primary_issue_code} ({primary.classification.value})",
            metadata={
                "failure_count": len(failures),
                "primary_issue_code": primary.primary_issue_code,
                "classification": primary.classification.value,
                "fingerprint": primary.error_fingerprint,
            },
        )

        policy = get_repair_policy(primary.primary_issue_code)
        can_attempt_repair = (
            policy is not None
            and policy.deterministic_repair_supported
            and policy.classification
            in (RuntimeFailureClass.PLAN_FIXABLE,)
            and state.get("runtime_repair_count", 0) == 0
        )

        if not can_attempt_repair:
            return {
                "runtime_failures": [f.to_dict() for f in failures],
                "runtime_primary_failure": primary.to_dict(),
                **trace_update,
            }

        return {
            "runtime_failures": [f.to_dict() for f in failures],
            "runtime_primary_failure": primary.to_dict(),
            **trace_update,
        }

    return _classify


def _make_deterministic_runtime_repair_node(*, max_runtime_repairs: int = 1):
    def _repair(state: GraphState) -> GraphState:
        """Attempt one-shot deterministic runtime repair."""
        from openmc_agent.runtime_feedback import RuntimeFailure, RuntimeFailureClass
        from openmc_agent.runtime_repair import (
            RuntimeRepairEvaluation,
            build_runtime_repair_request,
            commit_accepted_runtime_repair,
            diagnose_source_runtime_failure,
            evaluate_runtime_repair_candidate,
            propose_source_binding_repair,
        )
        from openmc_agent.runtime_repair_policy import get_repair_policy

        primary_dict = state.get("runtime_primary_failure")
        if not primary_dict:
            return {}

        repair_count = state.get("runtime_repair_count", 0)
        if repair_count >= max_runtime_repairs:
            trace_update = _trace_event_update(
                state,
                "runtime_repair_budget_exhausted",
                summary=f"Runtime repair budget exhausted ({repair_count}/{max_runtime_repairs})",
            )
            return {**trace_update}

        # Reconstruct RuntimeFailure.
        failure = RuntimeFailure(
            failure_id=primary_dict.get("failure_id", ""),
            stage=primary_dict.get("stage", ""),
            tool_name=primary_dict.get("tool_name", ""),
            returncode=primary_dict.get("returncode"),
            primary_issue_code=primary_dict.get("primary_issue_code", ""),
            secondary_issue_codes=primary_dict.get("secondary_issue_codes", []),
            normalized_message=primary_dict.get("normalized_message", ""),
            raw_error_excerpt=primary_dict.get("raw_error_excerpt", ""),
            error_fingerprint=primary_dict.get("error_fingerprint", ""),
            plan_hash=primary_dict.get("plan_hash"),
            artifact_paths=primary_dict.get("artifact_paths", []),
            classification=RuntimeFailureClass(
                primary_dict.get("classification", "unknown")
            ),
            owner_patch_types=primary_dict.get("owner_patch_types", []),
            requires_human_confirmation=primary_dict.get(
                "requires_human_confirmation", False
            ),
            environment_only=primary_dict.get("environment_only", False),
            metadata=primary_dict.get("metadata", {}),
        )

        # Check fingerprint dedup.
        fingerprint = failure.error_fingerprint
        attempts = state.get("runtime_repair_attempts_by_fingerprint", {})
        if attempts.get(fingerprint, 0) > 0:
            trace_update = _trace_event_update(
                state,
                "runtime_repair_duplicate_candidate",
                summary=f"Fingerprint {fingerprint} already attempted",
            )
            return {
                "runtime_repair_count": repair_count + 1,
                **trace_update,
            }

        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        build_state = state.get("plan_build_state", {})
        output_dir = state.get("output_dir", "data/runs")

        # Build repair request.
        request_or_eval = build_runtime_repair_request(
            failure, plan, build_state, state.get("tool_results", []), output_dir,
        )

        if isinstance(request_or_eval, RuntimeRepairEvaluation):
            trace_update = _trace_event_update(
                state,
                "runtime_deterministic_repair_not_applicable",
                summary=f"Repair not applicable: {request_or_eval.disposition}",
                metadata={"disposition": request_or_eval.disposition},
            )
            return {
                "runtime_repair_evaluation": request_or_eval.model_dump(mode="json"),
                "runtime_repair_count": repair_count + 1,
                "runtime_repair_attempts_by_fingerprint": {
                    **attempts,
                    fingerprint: attempts.get(fingerprint, 0) + 1,
                },
                **trace_update,
            }

        request = request_or_eval

        # Source binding oracle.
        policy = get_repair_policy(failure.primary_issue_code)
        proposal_or_eval: Any = None
        if policy and policy.preferred_patch_type == "settings":
            diagnosis = diagnose_source_runtime_failure(failure, plan, build_state)
            proposal_or_eval = propose_source_binding_repair(request, diagnosis)

        if proposal_or_eval is None:
            trace_update = _trace_event_update(
                state,
                "runtime_deterministic_repair_not_applicable",
                summary=f"No oracle for {failure.primary_issue_code}",
            )
            return {
                "runtime_repair_count": repair_count + 1,
                **trace_update,
            }

        from openmc_agent.runtime_repair import DeterministicRuntimeRepairProposal

        if isinstance(proposal_or_eval, RuntimeRepairEvaluation):
            trace_update = _trace_event_update(
                state,
                "runtime_deterministic_repair_not_applicable",
                summary=f"Oracle declined: {proposal_or_eval.disposition}",
            )
            return {
                "runtime_repair_evaluation": proposal_or_eval.model_dump(mode="json"),
                "runtime_repair_count": repair_count + 1,
                **trace_update,
            }

        proposal = proposal_or_eval

        _trace_event_update(
            {**state},
            "runtime_deterministic_repair_proposed",
            summary=f"Proposed: {proposal.deterministic_rule_id}",
        )

        # Evaluate on clone.
        prior_hashes = state.get("runtime_repair_candidate_hashes", [])
        evaluation = evaluate_runtime_repair_candidate(
            request, proposal, build_state,
            output_dir=output_dir,
            prior_candidate_hashes=prior_hashes,
        )

        if evaluation.accepted:
            # Commit to source state.
            updated_state = commit_accepted_runtime_repair(
                request, proposal, evaluation, build_state,
            )
            trace_update = _trace_event_update(
                state,
                "runtime_deterministic_repair_accepted",
                summary=f"Accepted: {proposal.deterministic_rule_id}",
                metadata={
                    "changed_paths": proposal.changed_paths,
                    "resolved": evaluation.resolved_issue_codes,
                },
            )
            return {
                "plan_build_state": updated_state.model_dump(mode="json"),
                "simulation_plan": SimulationPlan.model_validate(updated_state.assembled_plan),
                "runtime_repair_applied": True,
                "runtime_repair_count": repair_count + 1,
                "runtime_committed_repair_count": state.get("runtime_committed_repair_count", 0) + 1,
                "runtime_reexecution_count": state.get("runtime_reexecution_count", 0) + 1,
                "runtime_last_committed_failure_fingerprint": fingerprint,
                "runtime_repair_evaluation": evaluation.model_dump(mode="json"),
                "runtime_repair_candidate_hashes": [
                    *prior_hashes, evaluation.patch_hash_after,
                ] if evaluation.patch_hash_after else prior_hashes,
                "runtime_repair_attempts_by_fingerprint": {
                    **attempts,
                    fingerprint: attempts.get(fingerprint, 0) + 1,
                },
                "tool_results": [],  # Clear stale results
                "validation_report": ValidationReport(is_valid=True),
                "error": "",
                **trace_update,
            }

        trace_update = _trace_event_update(
            state,
            "runtime_deterministic_repair_rejected",
            summary=f"Rejected: {evaluation.disposition}",
            metadata={"reasons": evaluation.reasons},
        )
        return {
            "runtime_repair_evaluation": evaluation.model_dump(mode="json"),
            "runtime_repair_count": repair_count + 1,
            "runtime_repair_attempts_by_fingerprint": {
                **attempts,
                fingerprint: attempts.get(fingerprint, 0) + 1,
            },
            **trace_update,
        }

    return _repair


def _stable_json_hash_from_plan(plan: Any) -> str:
    if hasattr(plan, "model_dump"):
        import hashlib
        import json
        raw = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return "none"


stable_json_hash_from_plan = _stable_json_hash_from_plan


# --------------------------------------------------------------------------- #
# LLM Runtime diagnostician + constrained patch proposer nodes (R4)
# --------------------------------------------------------------------------- #


def _make_llm_runtime_diagnose_node(
    *,
    client: Any | None = None,
    model_name: str | None = None,
    allow_fallback: bool = False,
    mode: str = "diagnose_only",
):
    def _diagnose(state: GraphState) -> GraphState:
        """Run LLM diagnosis on the primary runtime failure."""
        from openmc_agent.runtime_feedback import RuntimeFailure, RuntimeFailureClass
        from openmc_agent.runtime_diagnostician import (
            FakeRuntimeDiagnosticianClient,
            RuntimeDiagnosis,
            build_runtime_diagnosis_input,
            make_runtime_diagnostician_client,
            validate_runtime_diagnosis,
        )
        from openmc_agent.runtime_feedback import RuntimeFailure
        from openmc_agent.runtime_repair_policy import get_repair_policy

        primary_dict = state.get("runtime_primary_failure")
        if not primary_dict:
            return {}

        diag_count = state.get("runtime_llm_diagnosis_count", 0)
        if diag_count >= 1:
            return {}

        # Reconstruct failure.
        failure = RuntimeFailure(
            failure_id=primary_dict.get("failure_id", ""),
            stage=primary_dict.get("stage", ""),
            tool_name=primary_dict.get("tool_name", ""),
            returncode=primary_dict.get("returncode"),
            primary_issue_code=primary_dict.get("primary_issue_code", ""),
            secondary_issue_codes=primary_dict.get("secondary_issue_codes", []),
            normalized_message=primary_dict.get("normalized_message", ""),
            raw_error_excerpt=primary_dict.get("raw_error_excerpt", ""),
            error_fingerprint=primary_dict.get("error_fingerprint", ""),
            classification=RuntimeFailureClass(
                primary_dict.get("classification", "unknown")
            ),
        )

        policy = get_repair_policy(failure.primary_issue_code)
        if policy is None or not policy.llm_diagnosis_supported:
            return {
                "runtime_llm_diagnosis_count": diag_count + 1,
            }

        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        build_state = state.get("plan_build_state", {})
        output_dir = state.get("output_dir", "data/runs")

        diagnosis_input = build_runtime_diagnosis_input(
            failure, plan, build_state,
            state.get("tool_results", []),
            output_dir,
        )

        trace_update = _trace_event_update(
            state,
            "runtime_llm_diagnosis_started",
            summary=f"Diagnosing {failure.primary_issue_code}",
            metadata={"mode": mode},
        )

        # Get client.
        actual_client = client
        fallback_used = False
        if actual_client is None:
            if allow_fallback:
                actual_client = FakeRuntimeDiagnosticianClient()
                fallback_used = True
            else:
                trace_update2 = _trace_event_update(
                    {**state, **trace_update},
                    "runtime_llm_client_unavailable",
                    summary="No diagnostician client available",
                )
                return {
                    "runtime_llm_diagnosis_count": diag_count + 1,
                    **trace_update2,
                }

        # Call client.
        try:
            from openmc_agent.runtime_diagnostician_prompts import (
                build_runtime_diagnosis_prompt,
            )
            prompt = build_runtime_diagnosis_prompt(diagnosis_input)
            schema = RuntimeDiagnosis.model_json_schema()
            raw = actual_client.diagnose(
                diagnosis_input, prompt=prompt, json_schema=schema,
            )
        except Exception as exc:
            return {
                "runtime_llm_diagnosis_count": diag_count + 1,
                **_trace_event_update(
                    {**state, **trace_update},
                    "runtime_llm_diagnosis_invalid",
                    summary=f"Client error: {exc}",
                ),
            }

        # Parse and validate diagnosis.
        import json as _json
        try:
            data = _json.loads(raw) if isinstance(raw, str) else raw
            diagnosis = RuntimeDiagnosis.model_validate(data)
        except Exception as exc:
            return {
                "runtime_llm_diagnosis_count": diag_count + 1,
                **_trace_event_update(
                    {**state, **trace_update},
                    "runtime_llm_diagnosis_invalid",
                    summary=f"Parse error: {exc}",
                ),
            }

        validated = validate_runtime_diagnosis(
            diagnosis, failure, build_state, policy=policy,
        )

        if not validated.accepted:
            return {
                "runtime_llm_diagnosis_count": diag_count + 1,
                "runtime_diagnosis_result": diagnosis.model_dump(mode="json"),
                "runtime_validated_diagnosis": validated.model_dump(mode="json"),
                **_trace_event_update(
                    {**state, **trace_update},
                    "runtime_llm_diagnosis_no_safe_repair",
                    summary=f"Rejected: {'; '.join(validated.rejection_codes)}",
                ),
            }

        return {
            "runtime_llm_diagnosis_count": diag_count + 1,
            "runtime_diagnosis_input": diagnosis_input,
            "runtime_diagnosis_result": diagnosis.model_dump(mode="json"),
            "runtime_validated_diagnosis": validated.model_dump(mode="json"),
            **_trace_event_update(
                {**state, **trace_update},
                "runtime_llm_diagnosis_validated",
                summary=f"Validated: {diagnosis.repair_kind}, proposal_allowed={validated.proposal_allowed}",
            ),
        }

    return _diagnose


def _make_llm_runtime_propose_node(
    *,
    client: Any | None = None,
    model_name: str | None = None,
    allow_fallback: bool = False,
    mode: str = "validate_only",
):
    def _propose(state: GraphState) -> GraphState:
        """Generate a constrained LLM patch proposal from validated diagnosis."""
        from openmc_agent.runtime_diagnostician import ValidatedRuntimeDiagnosis
        from openmc_agent.runtime_patch_proposer import (
            FakeRuntimePatchProposerClient,
            LLMRuntimeRepairProposal,
            make_runtime_patch_proposer_client,
            validate_llm_runtime_proposal,
        )

        validated_dict = state.get("runtime_validated_diagnosis")
        if not validated_dict:
            return {}

        proposal_count = state.get("runtime_llm_proposal_count", 0)
        if proposal_count >= 1:
            return {}

        validated = ValidatedRuntimeDiagnosis.model_validate(validated_dict)
        if not validated.proposal_allowed:
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
            }

        build_state = state.get("plan_build_state", {})
        # Get current patch content for target.
        target_content: dict[str, Any] = {}
        patches = build_state.get("patches", {}) if isinstance(build_state, dict) else {}
        target_id = validated.target_patch_id
        for pid, penv in patches.items():
            if pid == target_id or (isinstance(penv, dict) and penv.get("patch_type") == validated.target_patch_type):
                target_content = penv.get("content", {}) if isinstance(penv, dict) else {}
                break

        proposal_input = {
            "diagnosis_id": validated.failure_id,
            "failure_id": validated.failure_id,
            "target_patch_type": validated.target_patch_type,
            "target_patch_id": validated.target_patch_id,
            "repair_kind": validated.repair_kind,
            "allowed_paths": validated.deterministically_allowed_paths,
            "forbidden_paths": validated.deterministically_forbidden_paths,
            "current_patch": target_content,
            "max_mutating_operations": 4,
        }

        actual_client = client
        if actual_client is None:
            if allow_fallback:
                actual_client = FakeRuntimePatchProposerClient()
            else:
                return {
                    "runtime_llm_proposal_count": proposal_count + 1,
                    **_trace_event_update(
                        state,
                        "runtime_llm_client_unavailable",
                        summary="No proposer client available",
                    ),
                }

        try:
            from openmc_agent.runtime_patch_prompts import (
                build_runtime_patch_proposal_prompt,
            )
            prompt = build_runtime_patch_proposal_prompt(proposal_input)
            schema = LLMRuntimeRepairProposal.model_json_schema()
            raw = actual_client.propose(
                proposal_input, prompt=prompt, json_schema=schema,
            )
        except Exception as exc:
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_invalid",
                    summary=f"Client error: {exc}",
                ),
            }

        import json as _json
        try:
            data = _json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            data = raw if isinstance(raw, dict) else {}

        # Statically validate.
        validation = validate_llm_runtime_proposal(
            data, validated, target_content,
        )

        if not validation.accepted:
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                "runtime_llm_proposal": data if isinstance(data, dict) else {},
                "runtime_llm_proposal_validation": validation.model_dump(mode="json"),
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_rejected",
                    summary=f"Rejected: {'; '.join(validation.rejection_codes[:3])}",
                ),
            }

        proposal = validation.proposal
        if proposal is None:
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_rejected",
                    summary="Static validation produced no proposal",
                ),
            }

        # A static LLM proposal is never sufficient. Reuse the deterministic
        # clone evaluator, which renders and executes the isolated candidate.
        if mode != "apply_if_safe":
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                "runtime_llm_proposal": proposal.model_dump(mode="json"),
                "runtime_llm_proposal_validation": validation.model_dump(mode="json"),
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_accepted",
                    summary="Static validation passed; mode does not permit commit",
                ),
            }

        from openmc_agent.runtime_feedback import RuntimeFailure, RuntimeFailureClass
        from openmc_agent.runtime_repair import (
            DeterministicRuntimeRepairProposal,
            RuntimeRepairEvaluation,
            build_runtime_repair_request,
            commit_accepted_runtime_repair,
            evaluate_runtime_repair_candidate,
        )

        primary = state.get("runtime_primary_failure", {})
        failure = RuntimeFailure(
            failure_id=primary.get("failure_id", ""),
            stage=primary.get("stage", "execute_tools"),
            tool_name=primary.get("tool_name", ""),
            returncode=primary.get("returncode"),
            primary_issue_code=primary.get("primary_issue_code", ""),
            secondary_issue_codes=primary.get("secondary_issue_codes", []),
            normalized_message=primary.get("normalized_message", ""),
            raw_error_excerpt=primary.get("raw_error_excerpt", ""),
            error_fingerprint=primary.get("error_fingerprint", ""),
            classification=RuntimeFailureClass(primary.get("classification", "unknown")),
        )
        request_or_eval = build_runtime_repair_request(
            failure,
            _coerce_simulation_plan(state.get("simulation_plan")),
            build_state,
            state.get("tool_results", []),
            state.get("output_dir"),
            target_patch_type=validated.target_patch_type,
            target_patch_id=validated.target_patch_id,
            allow_llm_policy=True,
        )
        if isinstance(request_or_eval, RuntimeRepairEvaluation):
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                "runtime_llm_evaluation": request_or_eval.model_dump(mode="json"),
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_rejected",
                    summary=f"Candidate request rejected: {request_or_eval.disposition}",
                ),
            }

        repair_proposal = DeterministicRuntimeRepairProposal(
            proposal_id=proposal.proposal_id,
            request_id=request_or_eval.request_id,
            target_patch_type=proposal.target_patch_type,
            operations=proposal.operations,
            diagnosis=proposal.rationale,
            changed_paths=[
                op["path"] for op in proposal.operations
                if op.get("op") in {"add", "replace", "remove"}
            ],
            expected_effect=proposal.expected_effect,
            provenance="llm_runtime_patch_proposer",
            confidence=proposal.confidence,
            deterministic_rule_id=proposal.repair_kind,
        )
        evaluation = evaluate_runtime_repair_candidate(
            request_or_eval,
            repair_proposal,
            build_state,
            output_dir=state.get("output_dir"),
            prior_candidate_hashes=state.get("runtime_llm_candidate_hashes", []),
        )
        if not evaluation.accepted:
            return {
                "runtime_llm_proposal_count": proposal_count + 1,
                "runtime_llm_proposal": proposal.model_dump(mode="json"),
                "runtime_llm_proposal_validation": validation.model_dump(mode="json"),
                "runtime_llm_evaluation": evaluation.model_dump(mode="json"),
                **_trace_event_update(
                    state,
                    "runtime_llm_proposal_rejected",
                    summary=f"Candidate OpenMC acceptance failed: {evaluation.disposition}",
                    metadata={"artifacts": evaluation.artifacts},
                ),
            }

        committed = commit_accepted_runtime_repair(
            request_or_eval, repair_proposal, evaluation, build_state,
        )
        return {
            "runtime_llm_proposal_count": proposal_count + 1,
            "runtime_llm_proposal": proposal.model_dump(mode="json") if proposal else {},
            "runtime_llm_proposal_validation": validation.model_dump(mode="json"),
            "runtime_llm_evaluation": evaluation.model_dump(mode="json"),
            "plan_build_state": committed.model_dump(mode="json"),
            "simulation_plan": SimulationPlan.model_validate(committed.assembled_plan),
            "runtime_repair_applied": True,
            "runtime_committed_repair_count": state.get("runtime_committed_repair_count", 0) + 1,
            "runtime_reexecution_count": state.get("runtime_reexecution_count", 0) + 1,
            **_trace_event_update(
                state,
                "runtime_llm_repair_committed",
                summary=f"Candidate accepted and committed: {proposal.repair_kind}",
                metadata={"artifacts": evaluation.artifacts},
            ),
        }

    return _propose


def _make_runtime_supervisor_node(
    *,
    client: Any | None = None,
    model_name: str | None = None,
    allow_fallback: bool = False,
    budget: Any | None = None,
    enable_llm_runtime_repair: bool = False,
):
    def _supervise(state: GraphState) -> GraphState:
        """Choose a bounded post-execution action without altering patches."""
        from openmc_agent.runtime_repair import stable_json_hash
        from openmc_agent.runtime_repair_policy import get_repair_policy
        from openmc_agent.runtime_supervisor import (
            RuntimeIterationState,
            RuntimeLoopBudget,
            run_runtime_supervisor_decision,
            write_runtime_iteration_manifest,
        )
        from openmc_agent.runtime_supervisor_policy import build_runtime_supervisor_input

        runtime_budget = (
            budget if isinstance(budget, RuntimeLoopBudget)
            else RuntimeLoopBudget.model_validate(budget or {})
        )
        primary = state.get("runtime_primary_failure") or {}
        repeated_after_commit = bool(
            primary
            and primary.get("error_fingerprint")
            and primary.get("error_fingerprint")
            == state.get("runtime_last_committed_failure_fingerprint")
        )
        policy = get_repair_policy(primary.get("primary_issue_code", "")) if primary else None
        policy_summary = {
            "deterministic_repair_supported": bool(policy and policy.deterministic_repair_supported),
            "llm_diagnosis_supported": bool(enable_llm_runtime_repair and policy and policy.llm_diagnosis_supported),
            "llm_proposal_supported": bool(enable_llm_runtime_repair and policy and policy.llm_proposal_supported),
        }
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        plan_hash = stable_json_hash_from_plan(plan) if plan else "none"
        build_hash = stable_json_hash(state.get("plan_build_state", {}))
        supervisor_state = {
            **state,
            "runtime_policy_summary": policy_summary,
            "runtime_execution_succeeded": bool(
                state.get("validation_report") and state["validation_report"].is_valid
            ),
            "runtime_last_plan_hash": plan_hash,
            "runtime_last_build_state_hash": build_hash,
            "runtime_no_progress_count": (
                max(2, state.get("runtime_no_progress_count", 0))
                if repeated_after_commit
                else state.get("runtime_no_progress_count", 0)
            ),
        }
        supervisor_input = build_runtime_supervisor_input(supervisor_state, budget=runtime_budget)
        result = run_runtime_supervisor_decision(
            supervisor_input,
            client=client,
            model_name=model_name,
            allow_fallback=allow_fallback,
        )
        action = result.final_action.value if result.final_action else "stop"
        transient_retries = state.get("runtime_transient_retry_count", 0)
        if action == "retry_same_plan":
            transient_retries += 1
        iteration = RuntimeIterationState(
            iteration=state.get("runtime_iteration_count", 0),
            plan_hash_before_execution=plan_hash,
            build_state_hash=build_hash,
            primary_failure_fingerprint=primary.get("error_fingerprint"),
            primary_issue_code=primary.get("primary_issue_code"),
            supervisor_action=action,
            execution_result="success" if supervisor_state["runtime_execution_succeeded"] else "failure",
        )
        artifacts = write_runtime_iteration_manifest(
            state.get("output_dir", "data/runs"), iteration, supervisor_input, result,
            final_disposition=action,
        )
        history = [*state.get("runtime_supervisor_history", []), {
            "action": action,
            "state_fingerprint": result.state_fingerprint,
            "vetoed": result.vetoed,
        }]
        event = "runtime_supervisor_decision_accepted" if result.accepted else "runtime_supervisor_decision_vetoed"
        retry_reset: dict[str, Any] = {}
        if action == "retry_same_plan":
            # Transient retry: the plan schema is valid, only the runtime tool
            # failed. Reset the stale tool-level validation_report so
            # render_plan_script does not skip re-rendering.
            retry_reset = {
                "validation_report": ValidationReport(is_valid=True),
                "error": "",
            }
        return {
            "runtime_policy_summary": policy_summary,
            "runtime_execution_succeeded": supervisor_state["runtime_execution_succeeded"],
            "runtime_last_plan_hash": plan_hash,
            "runtime_last_build_state_hash": build_hash,
            "runtime_supervisor_result": result.model_dump(mode="json"),
            "runtime_supervisor_history": history,
            "runtime_iteration_count": state.get("runtime_iteration_count", 0) + 1,
            "runtime_transient_retry_count": transient_retries,
            "runtime_final_disposition": action,
            "runtime_no_progress_count": supervisor_state["runtime_no_progress_count"],
            **retry_reset,
            **_trace_event_update(
                state,
                event,
                summary=f"Runtime supervisor: {action}",
                metadata={"state_fingerprint": result.state_fingerprint, "artifacts": artifacts},
            ),
        }

    return _supervise


def _make_runtime_supervisor_router():
    def _route(state: GraphState) -> str:
        result = state.get("runtime_supervisor_result", {})
        action = result.get("final_action") if isinstance(result, dict) else None
        return {
            "finish_success": "finish",
            "attempt_deterministic_repair": "deterministic",
            "attempt_llm_repair": "llm",
            "retry_same_plan": "retry",
            "request_human_confirmation": "ask",
            "stop": "stop",
        }.get(action, "stop")
    return _route


# Number of consecutive patch failures (auto-repair or investigation) after which
# reflect_plan stops trying patches and regenerates the whole SimulationPlan.
PATCH_FALLBACK_THRESHOLD = 2


def _make_reflect_plan_node(
    repair_plan: RepairPlanFn,
    *,
    investigation_llm: InvestigationLlmFn | None = None,
    retrieval_roots_resolver: RetrievalRootsResolver | None = None,
    retrieval_tool_dispatch: dict[str, Callable[..., ToolResult]] | None = None,
    retrieval_tool_specs: list[ToolSpec] | None = None,
    investigation_max_iterations: int = 4,
    retrieval_policy: RetrievalPolicy | None = None,
):
    def _reflect_plan(state: GraphState) -> GraphState:
        plan = _coerce_simulation_plan(state.get("simulation_plan"))
        report = state.get("validation_report")
        retry_count = state.get("retry_count", 0)
        patch_failures = state.get("patch_failure_count", 0)
        if plan is None or report is None or report.is_valid:
            return {"retry_count": retry_count, "patch_failure_count": patch_failures}

        reflect_start_update = _trace_event_update(
            state,
            "reflect_plan_started",
            summary="reflect_plan handling invalid SimulationPlan",
            report=report,
            plan=plan,
            metadata={"llm_called": False, "patch_failure_count": patch_failures},
        )
        trace_state: GraphState = {**state, **reflect_start_update}

        # Patch sources, in priority order: (1) deterministic auto-repair of
        # uniquely-solvable id-reference typos -- no LLM call; (2) an
        # investigation patch from the retrieval loop. Whole-plan regeneration
        # below is the last resort, used only when no patch applies or recent
        # patches keep failing to validate.
        auto_patch: list[dict[str, Any]] | None = None
        if patch_failures < PATCH_FALLBACK_THRESHOLD:
            auto_attempt_update = _trace_event_update(
                trace_state,
                "auto_repair_attempted",
                summary="attempting deterministic lattice auto-repair",
                report=report,
                plan=plan,
                metadata={
                    "attempted_issue_codes": [
                        issue.code for issue in [*plan.capability_report.issues, *report.issues]
                    ],
                    "patch_failure_count": patch_failures,
                },
            )
            trace_state = {**trace_state, **auto_attempt_update}
            auto_patch = auto_repair_lattice_structure(
                plan,
                issues=[*plan.capability_report.issues, *report.issues],
                requirement=state.get("requirement", ""),
            )

        if auto_patch is not None and patch_failures < PATCH_FALLBACK_THRESHOLD:
            try:
                patched_payload = _apply_json_patches(
                    plan.model_dump(mode="json"), auto_patch
                )
                patched_payload = _normalize_capability_report_for_plan_validation(
                    patched_payload
                )
                patched_plan = SimulationPlan.model_validate(patched_payload)
                _progress(
                    state,
                    "reflect_plan",
                    f"applied {len(auto_patch)} patch op(s) via deterministic auto-repair",
                )
                auto_completed_update = _trace_event_update(
                    trace_state,
                    "auto_repair_completed",
                    summary="deterministic auto-repair succeeded",
                    report=report,
                    plan=patched_plan,
                    metadata={
                        "success": True,
                        "patch_count": len(auto_patch),
                        "changed_paths": [patch.get("path") for patch in auto_patch],
                    },
                )
                reflect_completed_update = _trace_event_update(
                    {**trace_state, **auto_completed_update},
                    "reflect_plan_completed",
                    summary="reflect_plan completed via deterministic auto-repair without LLM",
                    report=report,
                    plan=patched_plan,
                    metadata={
                        "llm_called": False,
                        "plan_changed": True,
                        "patch_count": len(auto_patch),
                    },
                )
                return {
                    "simulation_plan": patched_plan,
                    "simulation_spec": patched_plan.model_spec,
                    "retry_count": retry_count + 1,
                    "plan_patch": auto_patch,
                    "patch_confidence": "high",
                    "patch_reason": "deterministic auto-repair",
                    "patch_error": "",
                    "error": "",
                    "patch_failure_count": patch_failures,
                    "plan_artifacts": _write_final_simulation_plan(state, patched_plan),
                    **reflect_completed_update,
                }
            except Exception as exc:
                patch_failures += 1
                _progress(
                    state,
                    "reflect_plan",
                    f"deterministic auto-repair patch failed: {exc}; "
                    f"patch_failure_count={patch_failures}",
                )
                auto_completed_update = _trace_event_update(
                    trace_state,
                    "auto_repair_completed",
                    summary="deterministic auto-repair failed to apply",
                    report=report,
                    plan=plan,
                    metadata={
                        "success": False,
                        "patch_count": len(auto_patch or []),
                        "failure_reason": str(exc),
                    },
                )
                trace_state = {**trace_state, **auto_completed_update}

        retrieval_started_update = _trace_event_update(
            trace_state,
            "retrieval_started",
            summary="running retrieval orchestrator for reflect_plan",
            report=report,
            plan=plan,
            metadata={"policy": "default", "issue_count": len(report.issues)},
        )
        trace_state = {**trace_state, **retrieval_started_update}
        retrieval_context = _retrieval_context_for_report(report, policy=retrieval_policy)
        kg_summary = retrieval_context.knowledge_graph_summary or {}
        retrieval_completed_update = _trace_event_update(
            trace_state,
            "retrieval_completed",
            summary=retrieval_context.summary or "retrieval completed",
            report=report,
            retrieval_context=retrieval_context,
            plan=plan,
            metadata={
                "warnings": retrieval_context.warnings,
                "knowledge_graph_attempted": bool(kg_summary.get("attempted", False)),
                "knowledge_graph_loaded": bool(kg_summary.get("loaded", False)),
                "knowledge_graph_node_count": int(kg_summary.get("node_count", 0) or 0),
                "knowledge_graph_edge_count": int(kg_summary.get("edge_count", 0) or 0),
                "knowledge_graph_warning_count": len(
                    retrieval_context.knowledge_graph_warnings
                ),
            },
        )
        trace_state = {**trace_state, **retrieval_completed_update}
        state_with_evidence: GraphState = {
            **state,
            **trace_state,
            **_retrieval_state_updates(retrieval_context),
        }
        base_requirement = _build_reflection_requirement(state_with_evidence)
        plan_summary = json.dumps(plan.model_dump(mode="json"), ensure_ascii=False)
        hints = _error_catalog_hints_for(report.errors)
        investigation_outcome = _run_investigation_safely(
            state_with_evidence,
            phase="reflect",
            task_brief=base_requirement,
            plan_summary=_truncate_text(plan_summary, 4000),
            investigation_llm=investigation_llm,
            retrieval_roots_resolver=retrieval_roots_resolver,
            retrieval_tool_dispatch=retrieval_tool_dispatch,
            retrieval_tool_specs=retrieval_tool_specs,
            investigation_max_iterations=investigation_max_iterations,
            error_catalog_hints=hints,
        )

        investigation_patch = (
            investigation_outcome.patch
            if investigation_outcome is not None
            and investigation_outcome.ok
            and investigation_outcome.patch
            else None
        )
        candidate_patch = investigation_patch

        if candidate_patch is not None and patch_failures < PATCH_FALLBACK_THRESHOLD:
            patch_source = "investigation"
            try:
                patched_payload = _apply_json_patches(
                    plan.model_dump(mode="json"), candidate_patch
                )
                patched_payload = _normalize_capability_report_for_plan_validation(
                    patched_payload
                )
                patched_plan = SimulationPlan.model_validate(patched_payload)
                _progress(
                    state,
                    "reflect_plan",
                    f"applied {len(candidate_patch)} patch op(s) via {patch_source}",
                )
                reflect_completed_update = _trace_event_update(
                    state_with_evidence,
                    "reflect_plan_completed",
                    summary=f"reflect_plan completed via {patch_source} patch",
                    report=report,
                    retrieval_context=retrieval_context,
                    plan=patched_plan,
                    metadata={
                        "llm_called": False,
                        "plan_changed": True,
                        "patch_count": len(candidate_patch),
                        "changed_paths": [patch.get("path") for patch in candidate_patch],
                    },
                )
                return {
                    "simulation_plan": patched_plan,
                    "simulation_spec": patched_plan.model_spec,
                    "retry_count": retry_count + 1,
                    "plan_patch": candidate_patch,
                    "patch_confidence": "high",
                    "patch_reason": (
                        _truncate_text(investigation_outcome.findings, 500)
                        or "investigation patch"
                    ),
                    "patch_error": "",
                    "error": "",
                    "patch_failure_count": patch_failures,
                    "plan_artifacts": _write_final_simulation_plan(state, patched_plan),
                    **_retrieval_state_updates(retrieval_context),
                    **_investigation_state_updates(investigation_outcome),
                    **reflect_completed_update,
                }
            except Exception as exc:
                patch_failures += 1
                _progress(
                    state,
                    "reflect_plan",
                    f"{patch_source} patch failed: {exc}; "
                    f"patch_failure_count={patch_failures}",
                )

        # Fallback: regenerate the whole plan, enriched by findings when available.
        reflection_requirement = base_requirement
        if patch_failures > 0:
            reflection_requirement += (
                "\n\nNote: deterministic / investigation patches were attempted but "
                "did not validate. Carefully re-check id references (cell.fill_id, "
                "universe.cell_ids, lattice.universe_pattern, region.surface_ids, "
                "core.axial_layers.fill.id, core.axial_layers.loading_id, "
                "lattice_loadings.base_lattice_id) "
                "against the defined ids rather than guessing.\n"
            )
        if (
            investigation_outcome is not None
            and investigation_outcome.ok
            and investigation_outcome.findings
        ):
            reflection_requirement = (
                f"{reflection_requirement}\n\n"
                "Investigation findings (verified against codebase):\n"
                f"{_truncate_text(investigation_outcome.findings, 2000)}\n"
            )
        _progress(
            state,
            "reflect_plan",
            f"calling LLM reflection retry={retry_count + 1}, "
            f"patch_failures={patch_failures}",
        )
        result = repair_plan(
            requirement=reflection_requirement,
            schema=SimulationPlan,
            model=state.get("model", "openai:gpt-4o"),
            previous_spec=plan,
            validation_errors=report.errors,
        )
        artifact_paths = _write_plan_generation_artifacts(
            state,
            phase="reflect_plan",
            result=result,
            retry_count=retry_count + 1,
        )
        if not result.ok or result.value is None:
            _progress(state, "reflect_plan", f"failed: {result.error}")
            reflect_completed_update = _trace_event_update(
                state_with_evidence,
                "reflect_plan_completed",
                summary="LLM reflection failed",
                report=report,
                retrieval_context=retrieval_context,
                plan=plan,
                metadata={
                    "llm_called": True,
                    "plan_changed": False,
                    "failure_reason": result.error,
                },
            )
            return {
                "retry_count": retry_count + 1,
                "patch_failure_count": patch_failures,
                **_retrieval_state_updates(retrieval_context),
                "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
                "plan_artifacts": artifact_paths,
                "error": result.error or "failed to repair SimulationPlan",
                **_investigation_state_updates(investigation_outcome),
                **reflect_completed_update,
            }
        _progress(state, "reflect_plan", "reflection produced a new SimulationPlan")
        reflect_completed_update = _trace_event_update(
            state_with_evidence,
            "reflect_plan_completed",
            summary="LLM reflection produced a new SimulationPlan",
            report=report,
            retrieval_context=retrieval_context,
            plan=result.value,
            metadata={"llm_called": True, "plan_changed": True},
        )
        return {
            "simulation_plan": result.value,
            "simulation_spec": result.value.model_spec,
            "retry_count": retry_count + 1,
            "patch_failure_count": patch_failures,
            **_retrieval_state_updates(retrieval_context),
            "raw_llm_outputs": _append_raw_llm_output(state, result.raw_response),
            "plan_artifacts": _write_final_simulation_plan(
                state,
                result.value,
                existing_paths=artifact_paths,
            ),
            "error": "",
            "incremental_patch_repair_accepted": False,
            **_investigation_state_updates(investigation_outcome),
            **reflect_completed_update,
        }

    return _reflect_plan


def _save_record(state: GraphState) -> GraphState:
    _progress(state, "save_record", "appending simulation record")
    report = state.get("validation_report") or ValidationReport(
        is_valid=False,
        errors=[state.get("error", "unknown graph error")],
    )
    records_path = state.get("records_path", "data/runs/simulation_runs.jsonl")
    append_simulation_record(
        requirement=state.get("requirement", ""),
        model=state.get("model", ""),
        simulation_spec=state.get("simulation_spec"),
        validation_report=report,
        path=records_path,
        model_path=state.get("model_path"),
        error=state.get("error", ""),
        retry_count=state.get("retry_count", 0),
        retry_history=state.get("retry_history", []),
        pending_expert_questions=state.get("pending_expert_questions", []),
        human_loop_events=state.get("human_loop_events", []),
        investigation_trace=state.get("investigation_trace", []),
        plan_artifacts=state.get("plan_artifacts", []),
    )
    _progress(state, "save_record", "record saved")
    return _trace_event_update(
        state,
        "workflow_failed" if state.get("error") else "workflow_completed",
        summary="SimulationSpec workflow saved",
        report=report,
        metadata={
            "model_path": state.get("model_path"),
            "retry_count": state.get("retry_count", 0),
        },
    )


def _save_plan_record(state: GraphState) -> GraphState:
    _progress(state, "save_record", "appending SimulationPlan run record")
    report = state.get("validation_report") or ValidationReport(
        is_valid=False,
        errors=[state.get("error", "unknown graph error")],
    )
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    records_path = state.get("records_path", "data/runs/simulation_runs.jsonl")
    append_simulation_record(
        requirement=state.get("requirement", ""),
        model=state.get("model", ""),
        simulation_spec=plan.model_spec if plan is not None else None,
        validation_report=report,
        path=records_path,
        simulation_plan=plan.model_dump(mode="json") if plan is not None else None,
        model_path=state.get("model_path"),
        error=state.get("error", ""),
        retry_count=state.get("retry_count", 0),
        retry_history=state.get("retry_history", []),
        pending_expert_questions=state.get("pending_expert_questions", []),
        human_loop_events=state.get("human_loop_events", []),
        investigation_trace=state.get("investigation_trace", []),
        plan_artifacts=state.get("plan_artifacts", []),
    )
    _progress(state, "save_record", "record saved")
    output_dir = Path(state.get("output_dir", "data/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    outcome = _compute_workflow_outcome(state, plan, report)
    state["workflow_outcome"] = outcome
    _write_expert_feedback_artifacts(output_dir, state, outcome)
    if outcome.get("status") == "blocked_review_only":
        blocked_update = _trace_event_update(
            state,
            "workflow_blocked_review_only",
            summary=(
                "review-only skeleton kept; OpenMC not attempted; "
                f"blockers={outcome.get('reason_codes')}"
            ),
            report=report,
            plan=plan,
            metadata=outcome,
        )
        return {**blocked_update, "workflow_outcome": outcome}
    return _trace_event_update(
        state,
        "workflow_failed" if state.get("error") else "workflow_completed",
        summary="SimulationPlan workflow saved",
        report=report,
        plan=plan,
        metadata={
            "model_path": state.get("model_path"),
            "retry_count": state.get("retry_count", 0),
            "pending_expert_questions": state.get("pending_expert_questions", []),
            "workflow_outcome": outcome,
        },
    )


def _enrich_skeleton_todo(output_dir: Path, plan: SimulationPlan, state: GraphState) -> None:
    """Append the real blocker codes + next step to the review-only skeleton TODO.

    Avoids a generic 'fill the gaps' note: the reviewer sees exactly which
    structural code forced the skeleton, which confirmations are still open,
    which assumptions were accepted for this run, and whether the next step is
    an Agent repair or an expert fact.
    """
    import contextlib

    from openmc_agent.capability_blockers import classify_capability_blockers

    try:
        summary = classify_capability_blockers(plan, state.get("validation_report"))
    except Exception:
        return
    todo_path = output_dir / "TODO.md"
    section_lines = [
        "",
        "## P0-D5A blocker summary",
        f"- primary_blocker_codes: {', '.join(summary.primary_blocker_codes) or '(none)'}",
        f"- structural_issue_not_visible_to_validate_plan: "
        f"{summary.structural_issue_not_visible_to_validate_plan}",
        f"- open human confirmations: {len(plan.capability_report.required_human_confirmations)}",
        f"- accepted run-level assumptions: "
        f"{len(state.get('expert_assumption_acknowledgements') or [])}",
    ]
    if summary.structural_agent_fixable:
        section_lines.append(
            "- next step: Agent repair (structural defect; does NOT need an expert fact)."
        )
    elif summary.human_fact_required:
        section_lines.append(
            "- next step: expert fact required (human_fact_required issue present)."
        )
    elif summary.environment_required:
        section_lines.append("- next step: environment/runtime fix required.")
    else:
        section_lines.append("- next step: review the skeleton; no blocking issue classified.")
    with contextlib.suppress(Exception):
        with todo_path.open("a", encoding="utf-8") as handle:
            handle.write("\n".join(section_lines) + "\n")


def _compute_workflow_outcome(
    state: GraphState,
    plan: SimulationPlan | None,
    report: ValidationReport,
) -> dict[str, Any]:
    """Classify the final workflow status, separating review-only skeletons from
    real failures. A skeleton blocked by a structural/environment defect is
    ``blocked_review_only`` (kept for review, OpenMC not attempted) -- never a
    context-less ``FAIL``. Internal ``ok`` stays False so CI does not treat a
    skeleton as a success.
    """
    from openmc_agent.capability_blockers import classify_capability_blockers

    renderability = plan.capability_report.renderability if plan is not None else "none"
    decision = state.get("expert_feedback_decision") or {}
    user_decision = decision.get("action") if isinstance(decision, dict) else None
    model_artifact_generated = bool(state.get("model_path"))
    # execute_tools skips export/run for skeleton/none; runnable/exportable ran.
    export_ran = any(
        isinstance(tr, dict) and tr.get("name") == "export_xml"
        for tr in state.get("tool_results", [])
    )
    openmc_execution_attempted = renderability in {"exportable", "runnable"} and export_ran

    reason_codes: list[str] = []
    if plan is not None:
        summary = classify_capability_blockers(plan, report)
        reason_codes = list(summary.primary_blocker_codes)
        state["capability_blocker_summary"] = summary.model_dump(mode="json")

    if renderability in {"skeleton", "none"}:
        status = "blocked_review_only"
    elif openmc_execution_attempted and report.is_valid:
        status = "ok"
    else:
        status = "fail"
    return {
        "status": status,
        "reason_codes": reason_codes,
        "renderability": renderability,
        "model_artifact_generated": model_artifact_generated,
        "openmc_execution_attempted": openmc_execution_attempted,
        "user_decision": user_decision,
    }


def _write_expert_feedback_artifacts(
    output_dir: Path,
    state: GraphState,
    outcome: dict[str, Any],
) -> None:
    """Persist the P0-D5A expert-feedback artifacts next to the run record."""
    import contextlib

    def _write(name: str, payload: object) -> None:
        with contextlib.suppress(Exception):
            (output_dir / name).write_text(
                json.dumps(payload, ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

    _write("workflow_outcome.json", outcome)
    _write("capability_blocker_summary.json", state.get("capability_blocker_summary") or {})
    _write("expert_question_groups.json", state.get("expert_question_groups") or [])
    _write("expert_feedback_decision.json", state.get("expert_feedback_decision") or {})
    _write(
        "expert_assumption_acknowledgements.json",
        state.get("expert_assumption_acknowledgements") or [],
    )


def _execution_report_from_tool_results(results: list[ToolResult]) -> ValidationReport:
    issues: list[ValidationIssue] = []
    legacy_errors: list[str] = []
    legacy_warnings: list[str] = []
    for result in results:
        if not result.ok:
            message = result.error or result.stderr or result.stdout or "tool failed"
            legacy_errors.append(f"{result.name} failed: {message}")
            issues.extend(result.issues)
        diagnostics = parse_openmc_output(result.stdout, result.stderr)
        issues.extend(diagnostics.issues)
        legacy_warnings.extend(diagnostics.warnings)
    report = ValidationReport.from_issues(_dedupe_validation_issues(issues))
    return report.model_copy(
        update={
            "is_valid": not (legacy_errors or report.errors),
            "errors": [*legacy_errors, *report.errors],
            "warnings": [*report.warnings, *legacy_warnings],
        }
    )


def _report_should_reflect(report: ValidationReport) -> bool:
    if not report.issues:
        return True
    reflect_routes = {"auto_repair", "reflect_plan", "retrieval"}
    blocking_routes = {"ask_expert", "manual_review", "capability_downgrade"}
    if any(issue.route_hint in reflect_routes for issue in report.issues):
        return True
    if all(issue.route_hint in blocking_routes for issue in report.issues):
        return False
    return True


def _report_should_ask_expert(report: ValidationReport) -> bool:
    return bool(report.issues) and all(
        issue.route_hint == "ask_expert" or issue.requires_human_confirmation
        for issue in report.issues
    )


def _plan_with_validation_failure_capability(
    plan: SimulationPlan | None,
    report: ValidationReport,
) -> SimulationPlan | None:
    if plan is None:
        return None
    existing = plan.capability_report
    supported_renderer = existing.supported_renderer
    if supported_renderer == "none" and plan.complex_model is not None:
        supported_renderer = plan.complex_model.kind if plan.complex_model.kind in {"assembly", "core", "triso"} else "none"
    capability = existing.model_copy(
        update={
            "renderability": "skeleton" if supported_renderer != "none" else "none",
            "is_executable": False,
            "supported_renderer": supported_renderer,
            "executable_subsystems": [],
            "reasons": list(dict.fromkeys([*existing.reasons, *report.errors])),
            "issues": _dedupe_validation_issues([*existing.issues, *report.issues]),
        }
    )
    return plan.model_copy(update={"capability_report": capability})


def _dedupe_validation_issues(issues: list[ValidationIssue]) -> list[ValidationIssue]:
    seen: set[tuple[str, str, str | None]] = set()
    deduped: list[ValidationIssue] = []
    for issue in issues:
        key = (issue.code, issue.message, issue.schema_path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(issue)
    return deduped


def _structured_issue_context(issues: list[ValidationIssue]) -> str:
    if not issues:
        return ""
    payload: list[dict[str, Any]] = []
    for issue in issues[:12]:
        payload.append(
            {
                "code": issue.code,
                "severity": issue.severity,
                "message": issue.message,
                "schema_path": issue.schema_path,
                "route_hint": issue.route_hint,
                "concept_id": issue.concept_id,
                "grep_patterns": issue.grep_patterns,
                "repair_hints": [
                    {
                        "action": hint.action,
                        "message": hint.message,
                        "target_path": hint.target_path,
                        "example_patch": hint.example_patch,
                    }
                    for hint in issue.repair_hints
                ],
                "requires_retrieval": issue.requires_retrieval,
                "requires_human_confirmation": issue.requires_human_confirmation,
            }
        )
    return (
        "\n[Validation Issues]\n"
        "Use these stable codes and paths; do not guess missing physical facts.\n"
        f"{json.dumps(payload, ensure_ascii=False, indent=2)}\n"
    )


def _retrieval_context_for_report(
    report: ValidationReport | None,
    policy: RetrievalPolicy | None = None,
) -> RetrievalContext:
    if report is None or not report.issues:
        return RetrievalContext()
    return gather_retrieval_context_for_issues(report.issues, policy=policy)


def _retrieval_state_updates(context: RetrievalContext) -> dict[str, Any]:
    graph_context = context.graph_context or GraphContext()
    return {
        "retrieval_context": context.model_dump(mode="json"),
        "retrieval_prompt": format_retrieval_context(context),
        "grep_evidence": [item.model_dump(mode="json") for item in context.grep_evidence],
        "graph_context": graph_context.model_dump(mode="json"),
        "rag_evidence": [item.model_dump(mode="json") for item in context.rag_evidence],
    }


def _coerce_grep_evidence(raw_items: list[Any]) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    for raw in raw_items:
        if isinstance(raw, RetrievedEvidence):
            evidence.append(raw)
            continue
        if isinstance(raw, dict):
            try:
                evidence.append(RetrievedEvidence.model_validate(raw))
            except Exception:
                continue
    return evidence


def _coerce_rag_evidence(raw_items: list[Any]) -> list[RetrievedEvidence]:
    return _coerce_grep_evidence(raw_items)


def _coerce_graph_context(raw_item: Any) -> GraphContext:
    if isinstance(raw_item, GraphContext):
        return raw_item
    if isinstance(raw_item, dict):
        try:
            return GraphContext.model_validate(raw_item)
        except Exception:
            return GraphContext()
    return GraphContext()


def _repair_constraints_context() -> str:
    return (
        "\n[Repair Constraints]\n"
        "- Only change fields directly related to the validation issues.\n"
        "- Do not modify confirmed fields or expert feedback unless the issue targets them.\n"
        "- Do not invent material density, nuclide composition, benchmark facts, or cross section paths.\n"
        "- Treat grep evidence as locator context, not as final physics or nuclear-data truth.\n"
        "- Treat graph context as relationship metadata and retrieval routing hints, not as final physics facts.\n"
        "- Use RAG evidence only as documentation context for API usage, syntax, and explanations.\n"
        "- RAG evidence must not be used to invent nuclear data paths, material densities, compositions, or benchmark constants.\n"
        "- If an issue requires human confirmation, preserve that requirement even when RAG evidence explains the topic.\n"
    )


def _build_reflection_requirement(state: GraphState) -> str:
    tool_results = state.get("tool_results", [])
    expert_feedback = state.get("expert_feedback", [])
    repair_errors = state.get("capability_repair_errors", [])
    report = state.get("validation_report")
    issue_context = _structured_issue_context(report.issues if report else [])
    retrieval_prompt = state.get("retrieval_prompt") or format_retrieval_context(
        retrieval_context_from_raw(state.get("retrieval_context"))
    )
    if not retrieval_prompt:
        retrieval_prompt = format_retrieval_context(
            RetrievalContext(
                grep_evidence=_coerce_grep_evidence(state.get("grep_evidence", [])),
                graph_context=_coerce_graph_context(state.get("graph_context")),
                rag_evidence=_coerce_rag_evidence(state.get("rag_evidence", [])),
            )
        )
    structure_guidance = ""
    if repair_errors:
        structure_guidance = (
            "\nThe plan has STRUCTURAL defects the LLM must fix itself -- these are NOT "
            "questions for a human expert. Fix reference consistency and counts in the JSON:\n"
            "- Every universe.cell_ids / control-rod / reflector id reference must match an "
            "existing cell/material/region id. Rename the reference or add the missing object "
            "with that exact id.\n"
            "- A lattice pin-count mismatch means fill_universe + overrides positions do not "
            "sum to expected_counts; recompute the override (row, col) positions so each "
            "universe count matches expected_counts exactly.\n"
            "- Keep material values, geometry dimensions, and physical facts unchanged.\n"
            f"Structural errors to fix now: {repair_errors}\n"
        )
    pin_count_context = _pin_count_mismatch_context(state)
    return (
        f"{state.get('requirement', '')}\n\n"
        "The current SimulationPlan failed during OpenMC expert-style execution checks. "
        "Return a corrected SimulationPlan JSON object only. Do not modify Python code directly.\n"
        f"{structure_guidance}"
        f"{_hard_count_constraints_context(state)}"
        f"{pin_count_context}"
        f"{issue_context}"
        f"{retrieval_prompt}"
        f"{_repair_constraints_context()}"
        f"Validation and execution errors: {state.get('error', '')}\n"
        f"Tool results: {_compact_tool_results(tool_results)}\n"
        f"Human expert feedback: {expert_feedback}"
    )


def _build_format_repair_requirement(state: GraphState) -> str:
    raw_outputs = state.get("raw_llm_outputs", [])
    latest_raw = raw_outputs[-1] if raw_outputs else ""
    candidate_payload = state.get("candidate_payload")
    candidate_context = _candidate_payload_context(candidate_payload)
    schema_guidance = _format_repair_schema_guidance(
        state.get("error", ""),
        candidate_payload,
    )
    truncation_guidance = (
        "\nThe previous response appears truncated or too large. Do NOT attempt to "
        "reproduce large 17x17, 34x34, assembly, or core universe_pattern arrays. "
        "Return a minimal valid review-only SimulationPlan instead: keep lattice "
        "shape/pitch/ids, set oversized or uncertain universe_pattern/rings to [], "
        "and add explicit requires_human_confirmation items such as "
        "'rect lattice universe_pattern is missing'.\n"
        if _looks_like_truncated_json(latest_raw, state.get("error", ""))
        else ""
    )
    return (
        f"{_augmented_plan_requirement(state)}\n\n"
        "The previous model response could not be parsed or validated as a "
        "SimulationPlan JSON object. Return a corrected SimulationPlan JSON object only. "
        "Preserve the reactor modeling facts from the case requirement and do not invent "
        "missing physical data.\n"
        f"{schema_guidance}"
        f"{truncation_guidance}"
        f"Parse/validation error: {state.get('error', '')}\n"
        f"{candidate_context}"
        f"Previous raw response: {_truncate_text(latest_raw, 4000)}"
    )


def _candidate_payload_context(candidate_payload: Any) -> str:
    if not isinstance(candidate_payload, dict):
        return ""
    return (
        "Parsed candidate JSON is available. Repair this candidate locally instead of "
        "regenerating unrelated fields:\n"
        f"{_truncate_text(json.dumps(candidate_payload, ensure_ascii=False, indent=2), 4000)}\n"
    )


def _format_repair_schema_guidance(error: str, candidate_payload: Any) -> str:
    if not _is_missing_cell_fill_id_error(error):
        return ""

    paths = _missing_cell_fill_id_paths(candidate_payload)
    path_text = ", ".join(paths) if paths else "complex_model.cells[*].fill_id"
    return (
        "\n[Validation Issues]\n"
        "- code: cell.fill_id.missing\n"
        f"  schema_path: {path_text}\n"
        "  message: fill_id is required unless fill_type is void.\n"
        "  route_hint: reflect_plan\n\n"
        "[Repair Hints]\n"
        "- For every non-void cell, set fill_id to an already defined material, universe, "
        "or lattice id matching fill_type.\n"
        "- If and only if the cell is intentionally empty or outside the modeled domain, "
        "set fill_type='void' and leave fill_id null.\n"
        "- Do not invent material density, nuclide composition, benchmark facts, or cross "
        "section paths while repairing this schema error.\n"
        "- Preserve unrelated materials, dimensions, lattice maps, expert feedback, and "
        "confirmed reactor facts.\n\n"
    )


def _is_missing_cell_fill_id_error(error: str) -> bool:
    lowered = error.lower()
    return (
        "fill_id is required unless fill_type is void" in lowered
        or ("complex_model.cells" in lowered and "fill_id" in lowered)
    )


def _missing_cell_fill_id_paths(candidate_payload: Any) -> list[str]:
    if not isinstance(candidate_payload, dict):
        return []
    complex_model = candidate_payload.get("complex_model")
    if not isinstance(complex_model, dict):
        return []
    cells = complex_model.get("cells")
    if not isinstance(cells, list):
        return []

    paths: list[str] = []
    for index, cell in enumerate(cells):
        if not isinstance(cell, dict):
            continue
        fill_type = cell.get("fill_type", "material")
        if fill_type != "void" and not cell.get("fill_id"):
            paths.append(f"complex_model.cells[{index}].fill_id")
    return paths


def _looks_like_truncated_json(raw: str, error: str) -> bool:
    lowered = error.lower()
    if "unterminated string" in lowered:
        return True
    if raw.count("{") > raw.count("}") or raw.count("[") > raw.count("]"):
        return True
    tail = raw.rstrip()
    return bool(tail) and tail[-1] not in {"}", "]", "`"}


# Nuclear-data path is an environment config, not a modeling fact. When
# OPENMC_CROSS_SECTIONS already points at a readable cross_sections.xml, the
# LLM-written "must be set by the user" confirmation is stale noise and must
# not be re-asked by ask_expert. Reading an existing environment value is not
# the same as inventing a path, so the human-confirmation safety boundary
# (no fabricated nuclear-data paths) still holds.
_CROSS_SECTIONS_CONFIRMATION_MARKERS = (
    "cross_section",
    "cross-section",
    "cross sections library",
    "openmc_cross_sections",
    "nuclear data path",
    "nuclear-data path",
    "nuclear data library",
    "nuclear-data library",
)


def _is_cross_sections_confirmation(text: str) -> bool:
    lowered = (text or "").lower()
    return any(marker in lowered for marker in _CROSS_SECTIONS_CONFIRMATION_MARKERS)


def _cross_sections_env_available() -> bool:
    path = os.environ.get("OPENMC_CROSS_SECTIONS")
    return bool(path) and os.path.isfile(path)


def _cross_sections_question_resolved_by_env(text: str) -> bool:
    return _cross_sections_env_available() and _is_cross_sections_confirmation(text)


def _plan_hash(plan: SimulationPlan | None) -> str:
    """Stable hash of the plan's modeling fields for acknowledgement scoping.

    Acknowledgements are scoped to a plan hash so a changed plan (different
    materials/composition) cannot blindly reuse a prior run's accept/defer.
    """
    import hashlib

    if plan is None:
        return ""
    try:
        payload = plan.model_dump(mode="json", exclude={"capability_report"})
    except Exception:
        return ""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()[:16]


def _capability_blocker_payload(state: GraphState) -> dict[str, Any]:
    """Build the panel-header blocker view from the current plan + report."""
    from openmc_agent.capability_blockers import classify_capability_blockers

    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    if plan is None:
        return {"renderability": "none", "is_executable": False, "has_blocking_issue": False}
    report = state.get("validation_report")
    summary = classify_capability_blockers(plan, report)
    state["capability_blocker_summary"] = summary.model_dump(mode="json")
    blockers: list[dict[str, Any]] = []
    for issue in summary.structural_agent_fixable:
        blockers.append({
            "code": issue.code,
            "message": issue.message,
            "schema_path": issue.schema_path,
            "route_type": "agent-fixable",
        })
    for issue in summary.environment_required:
        blockers.append({
            "code": issue.code,
            "message": issue.message,
            "schema_path": issue.schema_path,
            "route_type": "environment",
        })
    for issue in summary.human_fact_required:
        blockers.append({
            "code": issue.code,
            "message": issue.message,
            "schema_path": issue.schema_path,
            "route_type": "human-required",
        })
    return {
        "renderability": summary.renderability,
        "is_executable": summary.is_executable,
        "has_blocking_issue": summary.has_blocking_issue,
        "primary_blocker_codes": summary.primary_blocker_codes,
        "blocking_issues": blockers,
        "structural_issue_not_visible_to_validate_plan": (
            summary.structural_issue_not_visible_to_validate_plan
        ),
    }


def _pending_expert_questions(state: GraphState) -> list[str]:
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    if plan is None:
        return []

    questions: list[str] = []
    capability = plan.capability_report
    for item in capability.required_human_confirmations:
        if is_structural_error_confirmation(item):
            # LLMs sometimes record structural defects (pin-count mismatch,
            # missing-universe refs) here; those are agent-fixable, not expert
            # questions, and re-asking them can trigger regenerate_plan.
            continue
        if _cross_sections_question_resolved_by_env(item):
            continue
        questions.append(f"Please provide or confirm: {item}")

    report = state.get("validation_report")
    if report is not None:
        # Runtime issues (e.g. runtime.cross_sections_missing) come from a real
        # OpenMC failure, not an LLM guess: even when OPENMC_CROSS_SECTIONS is
        # set the run may still fail (stale path, subprocess env not inherited,
        # hdf5 mismatch), so these must stay routed to ask_expert. Do NOT
        # env-suppress them -- only capability-stage confirmations are suppressed.
        for issue in report.issues:
            if issue.route_hint == "ask_expert" or issue.requires_human_confirmation:
                questions.append(f"Please provide or confirm: [{issue.code}] {issue.message}")

    if capability.renderability in {"none", "skeleton"}:
        for issue in capability.issues:
            if issue.route_hint == "ask_expert" or issue.requires_human_confirmation:
                if _cross_sections_question_resolved_by_env(issue.message):
                    continue
                questions.append(f"Please provide or confirm: [{issue.code}] {issue.message}")

    # Material assumptions are merged into per-material confirmable groups so the
    # panel no longer shows a wall of near-duplicate questions. Structural /
    # environment blockers are deliberately NOT turned into expert questions:
    # they surface in the panel header (see capability_blocker_summary) and route
    # to repair / review-only instead of asking the expert for physical facts.
    from openmc_agent.expert_feedback import group_expert_questions
    from openmc_agent.capability_blockers import classify_capability_blockers

    summary = classify_capability_blockers(plan, report)
    groups = group_expert_questions(plan, summary)
    for group in groups:
        questions.append(group.prompt)

    questions = list(dict.fromkeys(q for q in questions if q.strip()))
    return _filter_already_resolved_questions(questions, state)

    questions = list(dict.fromkeys(q for q in questions if q.strip()))
    return _filter_already_resolved_questions(questions, state)[:8]


def _filter_already_resolved_questions(
    questions: list[str],
    state: GraphState,
) -> list[str]:
    events = state.setdefault("human_loop_events", [])
    resolved_items = _coerce_resolved_expert_items(state.get("resolved_expert_items", []))
    if not resolved_items:
        return questions

    kept: list[str] = []
    for question in questions:
        match = _resolved_match_for_question(question, resolved_items)
        if match is None:
            kept.append(question)
            continue
        event_name = (
            "expert_question_filtered_as_declined"
            if match.status == "declined"
            else "expert_question_filtered_as_resolved"
        )
        events.append(
            {
                "event": event_name,
                "round": state.get("expert_round_count", 0),
                "question": question,
                "answer": match.answer,
                "reason": match.reason or "matched prior expert feedback semantically",
                "semantic_keys": match.semantic_keys,
            }
        )
    return kept


def _coerce_resolved_expert_items(items: Any) -> list[ResolvedExpertItem]:
    resolved: list[ResolvedExpertItem] = []
    for item in items or []:
        try:
            resolved.append(
                item if isinstance(item, ResolvedExpertItem) else ResolvedExpertItem.model_validate(item)
            )
        except Exception:
            continue
    return resolved


def _update_resolved_expert_items_from_feedback(
    *,
    state: GraphState,
    questions: list[str],
    feedback_items: list[str],
    round_index: int,
) -> list[dict[str, Any]]:
    existing = _coerce_resolved_expert_items(state.get("resolved_expert_items", []))
    answer = "\n".join(feedback_items).strip()
    updated = list(existing)
    for question in questions:
        if not answer:
            status: Literal["resolved", "declined", "unknown"] = "declined"
            reason = "expert submitted empty feedback for this round"
        elif _question_answered_by_feedback(question, answer):
            status = "resolved"
            reason = "expert feedback overlaps the question by field keywords, values, or units"
        else:
            status = "unknown"
            reason = "feedback received, but this question was not clearly answered"
        updated.append(
            ResolvedExpertItem(
                question=question,
                answer=answer,
                kind=_expert_question_kind(question),
                status=status,
                source_round=round_index,
                semantic_keys=_semantic_keys(f"{question}\n{answer}"),
                reason=reason,
            )
        )
    return [item.model_dump(mode="json") for item in updated]


def _resolved_match_for_question(
    question: str,
    resolved_items: list[ResolvedExpertItem],
) -> ResolvedExpertItem | None:
    kind = _expert_question_kind(question)
    # Structural/capability questions (missing-cell references, pin-count
    # mismatches, unsupported subsystems) are plan-internal defects that a
    # material-level expert answer can never resolve. Only an identical-text
    # match may de-duplicate them; semantic matching against a prior material
    # answer would silently swallow the gap and leave the model stuck at
    # skeleton -- the over-correction of the earlier "re-ask" bug.
    structural_kind = kind in {"capability_reason", "capability_warning", "unknown"}
    for item in resolved_items:
        if item.status not in {"resolved", "declined"}:
            continue
        # Identical question text always wins (covers declined + exact re-asks).
        if _normalized_text(question) == _normalized_text(item.question):
            return item
        if item.status == "declined":
            continue
        if structural_kind:
            continue
        if _question_answered_by_feedback(question, item.answer):
            return item
        question_keys = set(_semantic_keys(question))
        item_keys = set(item.semantic_keys)
        # Empty question_keys cannot support a semantic claim. Guard against the
        # 0 >= 0 vacuous-truth trap that previously matched structural errors
        # (which extract no keywords) against unrelated material answers.
        if not question_keys:
            continue
        if question_keys.issubset(item_keys):
            return item
        if len(question_keys & item_keys) >= min(2, len(question_keys)):
            return item
    return None


def _expert_question_kind(question: str) -> str:
    lowered = question.lower()
    if "assumption" in lowered:
        return "assumption"
    if "renderability gap" in lowered:
        return "capability_reason"
    if "warning" in lowered:
        return "capability_warning"
    if "confirm" in lowered:
        return "confirmation"
    return "unknown"


def _question_answered_by_feedback(question: str, feedback: str) -> bool:
    question_norm = _normalized_text(question)
    feedback_norm = _normalized_text(feedback)
    if not question_norm or not feedback_norm:
        return False
    if question_norm in feedback_norm or feedback_norm in question_norm:
        return True

    question_keys = set(_semantic_keys(question))
    feedback_keys = set(_semantic_keys(feedback))
    if not question_keys or not feedback_keys:
        return False
    if question_keys & feedback_keys and (
        feedback_keys & _value_like_semantic_keys(feedback)
        or {"benchmark", "default", "acceptable", "confirmed"} & feedback_keys
    ):
        return True
    return len(question_keys & feedback_keys) >= min(2, len(question_keys))


def _semantic_keys(text: str) -> list[str]:
    lowered = text.lower()
    keys: list[str] = []
    patterns = {
        "fuel temperature": [
            r"fuel\s+temperature",
            r"temperature",
            r"temperature_k",
            r"燃料温度",
            r"温度",
        ],
        "boundary condition": [
            r"boundary\s+(condition|type)",
            r"boundary_type",
            r"boundary",
            r"边界条件",
            r"边界",
        ],
        "density": [r"density", r"density_value", r"密度"],
        "enrichment": [r"enrichment", r"u-?235", r"富集", r"富集度"],
        "composition": [r"composition", r"isotope", r"nuclide", r"同位素", r"组分"],
        "packing fraction": [r"packing\s+fraction", r"packing_fraction", r"填充"],
        "triso": [r"triso"],
        "pitch": [r"pitch", r"栅距"],
        "radius": [r"radius", r"半径"],
        "benchmark": [r"benchmark", r"基准"],
        "default": [r"default", r"默认"],
        "acceptable": [r"acceptable", r"可以", r"同意", r"确认"],
        "confirmed": [r"confirm", r"confirmed", r"按"],
        "reflective": [r"reflective", r"反射"],
        "vacuum": [r"vacuum", r"真空"],
        "periodic": [r"periodic", r"周期"],
    }
    for key, regexes in patterns.items():
        if any(re.search(pattern, lowered) for pattern in regexes):
            keys.append(key)
    for number in re.findall(r"\b\d+(?:\.\d+)?\s*(?:k|g/cm3|kg/m3|cm|%)?\b", lowered):
        keys.append(re.sub(r"\s+", " ", number.strip()))
    return list(dict.fromkeys(keys))


def _value_like_semantic_keys(text: str) -> set[str]:
    keys = set(_semantic_keys(text))
    return {
        key
        for key in keys
        if re.search(r"\d", key)
        or key in {"reflective", "vacuum", "periodic", "benchmark", "default"}
    }


def _normalized_text(text: str) -> str:
    return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", " ", text.lower()).strip()


def _latest_expert_feedback(state: GraphState) -> str:
    feedback = state.get("expert_feedback", [])
    return feedback[-1].strip() if feedback else ""


def _classify_feedback_text(
    feedback: str,
    plan: SimulationPlan | None,
) -> tuple[str, str, Literal["high", "medium", "low"]]:
    lowered = feedback.lower()
    regenerate_patterns = [
        r"\brebuild\b",
        r"\bregenerate\b",
        r"\brestart\b",
        r"not\s+be\s+a",
        r"wrong\s+(requirement|task|model)",
        r"full[-\s]?core",
        r"whole\s+core",
        r"assembly",
        r"fixed\s+source",
        r"criticality",
        r"c5g7",
        r"triso",
        r"pebble",
        r"重新",
        r"重建",
        r"理解错",
        r"不是",
        r"全堆芯",
        r"组件",
        r"固定源",
        r"临界",
    ]
    if any(re.search(pattern, lowered) for pattern in regenerate_patterns):
        return (
            "regenerate_plan",
            "expert feedback appears to change the modeling intent, topology, benchmark, or physics task",
            "high",
        )
    if not _semantic_keys(feedback):
        return (
            "manual_review",
            "expert feedback has no safely mappable field, value, or confirmation keyword",
            "low",
        )
    if plan is None:
        return (
            "regenerate_plan",
            "no existing SimulationPlan is available for a local patch",
            "high",
        )
    return (
        "patch_plan",
        "expert feedback appears to provide or confirm local plan fields",
        "medium",
    )


def _build_plan_patches(
    plan: SimulationPlan,
    feedback: str,
    state: GraphState,
) -> tuple[list[dict[str, Any]], str, Literal["high", "medium", "low"]]:
    patches: list[dict[str, Any]] = []
    payload = plan.model_dump(mode="json")
    keys = set(_semantic_keys(feedback))

    if plan.model_spec is not None:
        spec_patches = _build_model_spec_patches(payload, feedback, keys)
        patches.extend(spec_patches)
    if plan.complex_model is not None:
        complex_patches = _build_complex_model_patches(payload, feedback, keys)
        patches.extend(complex_patches)

    patches.extend(_confirmation_removal_patches(payload, state))
    patches = _dedupe_patches(patches)
    if not patches:
        return [], "no safe local field path matched the expert feedback", "low"
    value_patches = [patch for patch in patches if patch.get("op") != "remove"]
    confidence: Literal["high", "medium", "low"] = "high" if value_patches else "medium"
    return patches, "generated minimal JSON Patch operations from expert feedback", confidence


def _build_model_spec_patches(
    payload: dict[str, Any],
    feedback: str,
    keys: set[str],
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    model_spec = payload.get("model_spec") or {}
    fuel = (((model_spec.get("pin_cell") or {}).get("fuel")) or {})
    if "fuel temperature" in keys and fuel:
        temperature = _extract_temperature_k(feedback)
        if temperature is not None:
            patches.append(
                {
                    "op": "replace" if fuel.get("temperature_k") is not None else "add",
                    "path": "/model_spec/pin_cell/fuel/temperature_k",
                    "value": temperature,
                }
            )
    if "density" in keys and fuel:
        density = _extract_density(feedback)
        if density is not None:
            patches.append(
                {
                    "op": "replace",
                    "path": "/model_spec/pin_cell/fuel/density_value",
                    "value": density,
                }
            )
            unit = _extract_density_unit(feedback) or fuel.get("density_unit") or "g/cm3"
            patches.append(
                {
                    "op": "replace",
                    "path": "/model_spec/pin_cell/fuel/density_unit",
                    "value": unit,
                }
            )
    return patches


def _build_complex_model_patches(
    payload: dict[str, Any],
    feedback: str,
    keys: set[str],
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    complex_model = payload.get("complex_model") or {}
    materials = complex_model.get("materials") or []
    fuel_index = _find_material_index(materials, feedback)
    if fuel_index is not None:
        material = materials[fuel_index]
        if "fuel temperature" in keys:
            temperature = _extract_temperature_k(feedback)
            if temperature is not None:
                patches.append(
                    {
                        "op": "replace" if material.get("temperature_k") is not None else "add",
                        "path": f"/complex_model/materials/{fuel_index}/temperature_k",
                        "value": temperature,
                    }
                )
        if "density" in keys:
            density = _extract_density(feedback)
            if density is not None:
                patches.append(
                    {
                        "op": "replace" if material.get("density_value") is not None else "add",
                        "path": f"/complex_model/materials/{fuel_index}/density_value",
                        "value": density,
                    }
                )
                unit = _extract_density_unit(feedback) or material.get("density_unit") or "g/cm3"
                patches.append(
                    {
                        "op": "replace" if material.get("density_unit") is not None else "add",
                        "path": f"/complex_model/materials/{fuel_index}/density_unit",
                        "value": unit,
                    }
                )

    boundary = _extract_boundary_type(feedback)
    if boundary is not None and "boundary condition" in keys:
        for index, surface in enumerate(complex_model.get("surfaces") or []):
            if surface.get("boundary_type") is not None:
                patches.append(
                    {
                        "op": "replace",
                        "path": f"/complex_model/surfaces/{index}/boundary_type",
                        "value": boundary,
                    }
                )
        for index, assembly in enumerate(complex_model.get("assemblies") or []):
            if assembly.get("boundary") is not None:
                patches.append(
                    {
                        "op": "replace",
                        "path": f"/complex_model/assemblies/{index}/boundary",
                        "value": boundary,
                    }
                )
    return patches


def _confirmation_removal_patches(
    payload: dict[str, Any],
    state: GraphState,
) -> list[dict[str, Any]]:
    patches: list[dict[str, Any]] = []
    resolved_items = [
        item
        for item in _coerce_resolved_expert_items(state.get("resolved_expert_items", []))
        if item.status == "resolved"
    ]
    if not resolved_items:
        return patches
    _append_remove_matches(
        patches,
        payload.get("capability_report", {}).get("required_human_confirmations", []),
        "/capability_report/required_human_confirmations",
        resolved_items,
    )
    _append_remove_matches(
        patches,
        payload.get("expert_assumptions", []),
        "/expert_assumptions",
        resolved_items,
    )
    complex_model = payload.get("complex_model") or {}
    _append_remove_matches(
        patches,
        complex_model.get("requires_human_confirmation", []),
        "/complex_model/requires_human_confirmation",
        resolved_items,
    )
    for index, material in enumerate(complex_model.get("materials") or []):
        _append_remove_matches(
            patches,
            material.get("requires_human_confirmation", []),
            f"/complex_model/materials/{index}/requires_human_confirmation",
            resolved_items,
        )
    for index, lattice in enumerate(complex_model.get("lattices") or []):
        _append_remove_matches(
            patches,
            lattice.get("requires_human_confirmation", []),
            f"/complex_model/lattices/{index}/requires_human_confirmation",
            resolved_items,
        )
    for index, triso in enumerate(complex_model.get("trisos") or []):
        _append_remove_matches(
            patches,
            triso.get("requires_human_confirmation", []),
            f"/complex_model/trisos/{index}/requires_human_confirmation",
            resolved_items,
        )
    for index, pebble in enumerate(complex_model.get("pebbles") or []):
        _append_remove_matches(
            patches,
            pebble.get("requires_human_confirmation", []),
            f"/complex_model/pebbles/{index}/requires_human_confirmation",
            resolved_items,
        )
    return patches


def _append_remove_matches(
    patches: list[dict[str, Any]],
    items: list[Any],
    base_path: str,
    resolved_items: list[ResolvedExpertItem],
) -> None:
    for index in range(len(items) - 1, -1, -1):
        item_text = str(items[index])
        question = f"Please provide or confirm: {item_text}"
        if _resolved_match_for_question(question, resolved_items) is not None:
            patches.append({"op": "remove", "path": f"{base_path}/{index}"})


def _find_material_index(materials: list[dict[str, Any]], feedback: str) -> int | None:
    if not materials:
        return None
    lowered = feedback.lower()
    if "fuel" in lowered or "燃料" in lowered:
        for index, material in enumerate(materials):
            material_text = f"{material.get('id', '')} {material.get('name', '')}".lower()
            if "fuel" in material_text or "uo2" in material_text:
                return index
    if len(materials) == 1:
        return 0
    return None


def _extract_temperature_k(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*k\b", text.lower())
    return float(match.group(1)) if match else None


def _extract_density(text: str) -> float | None:
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:g\s*/\s*cm3|g/cm3|g/cc|kg\s*/\s*m3|kg/m3)", text.lower())
    return float(match.group(1)) if match else None


def _extract_density_unit(text: str) -> str | None:
    lowered = text.lower()
    if re.search(r"kg\s*/\s*m3|kg/m3", lowered):
        return "kg/m3"
    if re.search(r"g\s*/\s*cm3|g/cm3|g/cc", lowered):
        return "g/cm3"
    if "atom/b-cm" in lowered:
        return "atom/b-cm"
    return None


def _extract_boundary_type(text: str) -> str | None:
    lowered = text.lower()
    for boundary in ("reflective", "vacuum", "periodic", "transmission", "white"):
        if boundary in lowered:
            return boundary
    if "反射" in text:
        return "reflective"
    if "真空" in text:
        return "vacuum"
    if "周期" in text:
        return "periodic"
    return None


def _dedupe_patches(patches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for patch in patches:
        deduped[(patch.get("op", ""), patch.get("path", ""))] = patch
    return list(deduped.values())


def _apply_json_patches(payload: Any, patches: list[dict[str, Any]]) -> Any:
    updated = json.loads(json.dumps(payload))
    for patch in patches:
        op = patch.get("op")
        path = patch.get("path")
        if not isinstance(path, str) or not path.startswith("/"):
            raise ValueError(f"invalid JSON Patch path: {path!r}")
        parent, key = _json_pointer_parent(updated, path)
        if op in {"add", "replace"}:
            value = patch.get("value")
            if isinstance(parent, list):
                index = int(key)
                if op == "add" and index == len(parent):
                    parent.append(value)
                else:
                    parent[index] = value
            else:
                parent[key] = value
        elif op == "remove":
            if isinstance(parent, list):
                del parent[int(key)]
            else:
                parent.pop(key, None)
        else:
            raise ValueError(f"unsupported JSON Patch op: {op!r}")
    return updated


def _normalize_capability_report_for_plan_validation(payload: dict[str, Any]) -> dict[str, Any]:
    """Reset locally assessed renderer capability before validating patched plans.

    ``assess_capability`` may write skeleton/assembly capability back onto a
    complex-only plan for sidecars and routing. The schema intentionally rejects
    that shape as an LLM-authored plan. A local expert patch should validate the
    structural plan first, then let ``assess_capability`` recompute capability.
    """
    if payload.get("model_spec") is not None or payload.get("complex_model") is None:
        return payload
    capability = dict(payload.get("capability_report") or {})
    if capability.get("is_executable") is False:
        capability["renderability"] = "none"
        capability["supported_renderer"] = "none"
        capability["executable_subsystems"] = []
        payload["capability_report"] = capability
    return payload


def _json_pointer_parent(payload: Any, path: str) -> tuple[Any, str]:
    parts = [_decode_json_pointer_part(part) for part in path.strip("/").split("/")]
    current = payload
    for part in parts[:-1]:
        if isinstance(current, list):
            current = current[int(part)]
        else:
            current = current[part]
    return current, parts[-1]


def _decode_json_pointer_part(part: str) -> str:
    return part.replace("~1", "/").replace("~0", "~")


def _feedback_from_resume_payload(payload: Any) -> tuple[list[str], bool]:
    if payload is None:
        return [], False
    if isinstance(payload, str):
        text = payload.strip()
        return ([text] if text else []), bool(text)
    if isinstance(payload, list):
        items = [str(item).strip() for item in payload if str(item).strip()]
        return items, bool(items)
    if isinstance(payload, dict):
        should_continue = bool(payload.get("should_continue", True))
        raw_feedback = (
            payload.get("expert_feedback")
            or payload.get("feedback")
            or payload.get("text")
            or payload.get("answer")
        )
        if isinstance(raw_feedback, list):
            items = [str(item).strip() for item in raw_feedback if str(item).strip()]
        elif raw_feedback is None:
            items = []
        else:
            text = str(raw_feedback).strip()
            items = [text] if text else []
        return items, should_continue
    text = str(payload).strip()
    return ([text] if text else []), bool(text)


_VALID_DECISION_ACTIONS = frozenset(
    {
        "accept_assumptions_for_this_run",
        "provide_corrections",
        "defer_confirmations",
        "continue_repair",
        "accept_review_only",
        "abort",
    }
)


def _decision_action_from_resume_payload(payload: Any) -> str | None:
    """Extract an explicit ``:command`` decision action from the resume payload."""
    if not isinstance(payload, dict):
        return None
    action = payload.get("decision_action")
    if isinstance(action, str) and action in _VALID_DECISION_ACTIONS:
        return action
    return None


def _coerce_simulation_plan(value: Any) -> SimulationPlan | None:
    if value is None:
        return None
    if isinstance(value, SimulationPlan):
        if isinstance(value.capability_report, dict) or isinstance(value.complex_model, dict):
            return _construct_simulation_plan_from_payload(dict(value.__dict__))
        return value
    if isinstance(value, dict):
        try:
            return SimulationPlan.model_validate(value)
        except Exception:
            return _construct_simulation_plan_from_payload(value)
    return None


def _construct_simulation_plan_from_payload(payload: dict[str, Any]) -> SimulationPlan:
    return SimulationPlan.model_construct(
        schema_version=payload.get("schema_version", "simulation_plan.v1"),
        model_spec=(
            SimulationSpec.model_validate(payload["model_spec"])
            if payload.get("model_spec") is not None
            else None
        ),
        complex_model=(
            ComplexModelSpec.model_validate(payload["complex_model"])
            if payload.get("complex_model") is not None
            else None
        ),
        capability_report=RenderCapabilityReport.model_validate(
            payload.get("capability_report") or {}
        ),
        plot_specs=[
            PlotSpec.model_validate(item) for item in payload.get("plot_specs", [])
        ],
        execution_check=ExecutionCheckSpec.model_validate(
            payload.get("execution_check") or {}
        ),
        expert_assumptions=list(payload.get("expert_assumptions", [])),
        expert_feedback=[
            ExpertFeedback.model_validate(item)
            for item in payload.get("expert_feedback", [])
        ],
    )


_SELF_REPAIRABLE_CAPABILITY_PATTERNS = (
    # Reference-consistency defects from renderer diagnostics: a universe/cell/
    # region/control-rod/reflector pointing at an id that does not exist.
    r"references missing (cells|universes|materials|surfaces|regions?)",
    # Pin-count / lattice-shape defects.
    r"pin counts? (do not match|mismatch)",
    r"expected_counts",
    r"shape .* does not match",
    r"(universe_pattern )?rows have unequal lengths",
    # Cylinder geometry defects.
    r"radius must be (positive|less than pitch)",
    r"non-numeric radius",
)


# Stable codes for plan defects the agent can fix itself (plan typos) vs. facts
# only an expert can supply. Code-based matching replaces the fragile regex over
# free-text renderer messages above; the regex stays as a legacy fallback for
# renderers that do not yet emit structured issues.
SELF_REPAIRABLE_CODES = frozenset({
    "lattice.universe_ref_missing",
    "lattice.shape_pattern_mismatch",
    "lattice.pattern_ragged_rows",
    "lattice.pin_count_mismatch",
    "cell.material_ref_missing",
    "cell.region_ref_missing",
    "cell.universe_ref_missing",
    "cell.lattice_ref_missing",
    "core.lattice_ref_missing",
    "universe.cell_ref_missing",
    "region.surface_ref_missing",
    "axial_layer.fill_ref_missing",
    "axial_layer.loading_ref_missing",
    "lattice_loading.base_ref_missing",
    "lattice_loading.override_universe_ref_missing",
    "surface.cylinder_radius_invalid",
    "material.mixed_percent_type",
    # A component-profile layer must preserve the assembly's tube paths; a
    # material slab is a planner structural error, not a missing expert fact.
    "assembly3d.component_profile_as_material_slab",
})


def _capability_self_repair_errors(
    capability: RenderCapabilityReport,
) -> list[ValidationIssue]:
    """Issues the agent can fix itself (plan typos) vs. facts only an expert can supply.

    Prefers structured ``capability.issues`` filtered by :data:`SELF_REPAIRABLE_CODES`;
    falls back to the legacy regex over ``reasons`` / ``required_human_confirmations``
    for renderers that do not yet emit structured issues. A missing density or
    composition is a real gap the expert must fill, so those codes stay out of
    :data:`SELF_REPAIRABLE_CODES` and route to ask_expert.
    """
    structured = [
        issue
        for issue in capability.issues
        if issue.code in SELF_REPAIRABLE_CODES
    ]
    if structured:
        return list(structured)
    candidates: list[str] = []
    if capability.renderability in {"none", "skeleton"}:
        candidates.extend(capability.reasons)
    # Pin-count mismatches are also recorded as soft human confirmations on the
    # lattice spec; they are still count/override errors the agent can fix.
    candidates.extend(capability.required_human_confirmations)
    repaired_texts = [
        text
        for text in candidates
        if any(
            re.search(pattern, text, re.IGNORECASE)
            for pattern in _SELF_REPAIRABLE_CAPABILITY_PATTERNS
        )
    ]
    return [
        ValidationIssue(severity="error", code="legacy.self_repairable", message=text)
        for text in dict.fromkeys(repaired_texts)
    ]


def _probe_axial_materialization_blockers(plan: SimulationPlan) -> list[ValidationIssue]:
    """Defensive assertion: surface any lattice-loading defects the shared
    validator may have missed.

    The primary detection is now :func:`lattice_loading_structural_issues`
    called from :func:`validate_simulation_plan`.  This probe runs the full
    materialization as a dry-run defensively; if the shared validator already
    caught everything, this returns an empty list.  It never duplicates issues
    the validator already reported.
    """
    from openmc_agent.lattice_loading_validation import (
        lattice_loading_structural_issues,
    )

    model = plan.complex_model
    if model is None or model.core is None or not getattr(model.core, "axial_layers", None):
        return []

    # Issues already visible to the shared validator — these are the primary
    # detection source and must not be re-injected by this probe.
    shared_issues = lattice_loading_structural_issues(model)
    shared_codes = {i.code for i in shared_issues}

    try:
        from openmc_agent.lattice_transform import materialize_axial_lattice_transformations
        _new_spec, issues, _meta = materialize_axial_lattice_transformations(model)
    except Exception:
        return []
    # Only return issues whose code the shared validator did NOT already cover.
    # Dedup by code (not code+path) because the compose engine and the shared
    # validator annotate schema_paths differently.
    return [
        issue for issue in issues
        if issue.severity == "error"
        and issue.code not in shared_codes
    ]


def _capability_for_plan(plan: SimulationPlan) -> RenderCapabilityReport:
    """Use the renderer registry as the single source of truth for capability."""
    _renderer, report = choose_renderer(plan)
    # Merge plan-level human confirmations (e.g. complex_model.requires_human_confirmation)
    # on top of the renderer-level report so the sidecar records everything.
    confirmations = list(
        dict.fromkeys(
            [
                *report.required_human_confirmations,
                *_plan_human_confirmations(plan),
            ]
        )
    )
    return report.model_copy(update={"required_human_confirmations": confirmations})


def _write_capability_sidecar(output_dir: Path, capability: RenderCapabilityReport) -> None:
    """Keep capability_report.json consistent with the authoritative plan report."""
    path = output_dir / "capability_report.json"
    path.write_text(
        json.dumps(capability.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# Regenerable render outputs. Removing them at the start of a render node lets a
# skeleton / non-exportable run overwrite a prior exportable run's model.py, XML
# and optimistic capability_report.json, so the on-disk state always matches the
# current run. Run records (simulation_plan.json, transcript.json,
# plan_artifacts/, checkpoints.sqlite, inspect_runs.jsonl) are NOT in this set.
_RENDER_ARTIFACT_NAMES: tuple[str, ...] = (
    "model.py",
    "smoke_model.py",
    "materials.xml",
    "geometry.xml",
    "settings.xml",
    "tallies.xml",
    "plots.xml",
    "capability_report.json",
    "TODO.md",
)


def _clean_stale_render_artifacts(output_dir: Path) -> None:
    """Remove regenerable render outputs left over from a prior run.

    Guarantees the output directory reflects the CURRENT run: when this run is
    non-exportable, no previous run's model.py / XML / exportable
    capability_report.json remains to masquerade as a successful result. Run
    records that are appended-to rather than regenerated are preserved.
    """
    import shutil

    for name in _RENDER_ARTIFACT_NAMES:
        path = output_dir / name
        if path.exists() or path.is_symlink():
            try:
                path.unlink()
            except OSError:
                pass
    # OpenMC run outputs and plot images.
    for pattern in ("statepoint.*.h5", "summary.*.h5"):
        for path in output_dir.glob(pattern):
            try:
                path.unlink()
            except OSError:
                pass
    plots_dir = output_dir / "plots"
    if plots_dir.exists():
        try:
            shutil.rmtree(plots_dir)
        except OSError:
            pass


def _plot_bounds_metadata(plan: SimulationPlan) -> list[dict]:
    """Project plan.plot_specs into the dict shape the bounds validator expects."""
    out: list[dict] = []
    for i, p in enumerate(plan.plot_specs):
        origin = p.origin or (0.0, 0.0, 0.0)
        width = p.width_cm or (0.0, 0.0)
        out.append({
            "id": f"plot_{i}",
            "basis": getattr(p, "basis", "xy"),
            "origin": {"x": float(origin[0]), "y": float(origin[1] if len(origin) > 1 else 0.0)},
            "width": {"x": float(width[0]), "y": float(width[1] if len(width) > 1 else width[0])},
        })
    return out


def _write_non_executable_marker(
    output_dir: Path,
    report: "ValidationReport | None",
    plan: "SimulationPlan | None",
    capability: RenderCapabilityReport | None = None,
) -> None:
    """Write an honest NOT_EXECUTABLE capability_report.json + TODO.md.

    Used when the render node is skipped (plan invalid / no renderer) so that a
    prior exportable run's optimistic sidecars cannot mask the current run's
    failure. ``capability`` may be supplied (e.g. renderer returned 'none');
    otherwise a minimal non-executable report is derived from the validation
    issues.
    """
    if capability is None:
        issues = list(report.issues) if report is not None and report.issues else []
        error_messages = [iss.message for iss in issues if iss.severity == "error"]
        capability = RenderCapabilityReport(
            renderability="none",
            is_executable=False,
            supported_renderer="none",
            executable_subsystems=[],
            reasons=error_messages or ["plan validation failed; render skipped"],
            issues=issues,
        )
    _write_capability_sidecar(output_dir, capability)

    lines = [
        "# TODO — OpenMC model not executable",
        "",
        f"Renderability: {capability.renderability}",
        "Status: NOT_EXECUTABLE",
        "",
        "## Blocking reasons",
    ]
    for reason in capability.reasons:
        lines.append(f"- {reason}")
    (output_dir / "TODO.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plan_human_confirmations(plan: SimulationPlan) -> list[str]:
    confirmations = list(plan.capability_report.required_human_confirmations)
    model = plan.complex_model
    if model is None:
        return confirmations
    confirmations.extend(model.requires_human_confirmation)
    for material in model.materials:
        confirmations.extend(
            f"material {material.id}: {item}"
            for item in material.requires_human_confirmation
        )
    for lattice in model.lattices:
        confirmations.extend(
            f"lattice {lattice.id}: {item}"
            for item in lattice.requires_human_confirmation
        )
    for triso in model.trisos:
        confirmations.extend(
            f"TRISO {triso.id}: {item}"
            for item in triso.requires_human_confirmation
        )
    for pebble in model.pebbles:
        confirmations.extend(
            f"pebble {pebble.id}: {item}"
            for item in pebble.requires_human_confirmation
        )
    return list(dict.fromkeys(confirmations))


_COUNT_CONSTRAINT_KEYWORDS = (
    "expected_counts",
    "pin count",
    "pin-count",
    "棒位",
    "总数",
    "共有",
    "检查",
    "燃料棒",
    "导向管",
    "裂变室",
    "MOX",
    "UO2",
    "mox",
    "uo2",
)


def _extract_hard_count_constraints(requirement: str, *, limit: int = 18) -> str:
    """Extract bounded source lines that look like hard count constraints."""
    lines: list[str] = []
    for raw_line in requirement.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        has_digit = bool(re.search(r"\d", line))
        has_count_unit = bool(
            re.search(r"(个|根|位置|count|counts|total|expected)", line, re.IGNORECASE)
        )
        has_keyword = any(keyword in line for keyword in _COUNT_CONSTRAINT_KEYWORDS)
        if has_digit and has_count_unit and has_keyword:
            lines.append(line)
        if len(lines) >= limit:
            break
    return "\n".join(dict.fromkeys(lines))


def _hard_count_constraints_context(state: GraphState) -> str:
    constraints = state.get("hard_count_constraints") or _extract_hard_count_constraints(
        state.get("requirement", "")
    )
    if not constraints:
        return ""
    return (
        "\n[Hard Count Constraints From Input]\n"
        "Treat these source-text counts as hard constraints. For any lattice that "
        "uses these counts, encode them in expected_counts and ensure the expanded "
        "universe_pattern matches them exactly. During repair, prefer fixing "
        "fill_universe/overrides/universe_pattern; change expected_counts only if "
        "the original input count was transcribed incorrectly.\n"
        f"{_truncate_text(constraints, 2500)}\n"
    )


def _core_lattice_naming_guidance() -> str:
    """Static rule injected into generation/reflection prompts.

    The C5G7 regression: the LLM wrote 'uo2_assembly_univ' in core_lattice but
    set assembly.id to 'uo2_assy'. The core renderer looks up assemblies by id,
    so any mismatch blocks export and loops reflect_plan. auto_repair now
    unifies the names deterministically; stating the rule up front keeps the
    LLM from introducing the mismatch in the first place.
    """
    return (
        "\n[Core Lattice Naming]\n"
        "Each assembly slot in a core lattice's universe_pattern MUST equal the\n"
        "id of an AssemblySpec -- the wrapper universe id is exactly assembly.id.\n"
        "Do NOT invent names like '<lattice_id>_univ' or '<assembly>_univ' that\n"
        "differ from assembly.id; the core renderer looks up assemblies by id and\n"
        "any mismatch blocks export.\n"
    )


def _pin_count_mismatch_context(state: GraphState) -> str:
    report = state.get("validation_report")
    if report is None:
        return ""
    mismatches = [
        issue
        for issue in report.issues
        if issue.code == "lattice.pin_count_mismatch"
    ]
    if not mismatches:
        return ""
    plan = _coerce_simulation_plan(state.get("simulation_plan"))
    requirement = state.get("requirement", "")
    lines: list[str] = []
    for issue in mismatches[:8]:
        lines.append(f"- {issue.schema_path or '<unknown path>'}: {issue.message}")
        location = _pin_count_mismatch_location(plan, issue, requirement)
        if location:
            lines.append(location)
    return (
        "\n[Pin Count Mismatch Evidence]\n"
        "The current IR expands to counts that disagree with expected_counts. "
        "Re-read the input rows/regions and correct the lattice map before any "
        "render/export step.\n"
        + "\n".join(lines)
        + "\n"
    )


def _lattice_id_from_schema_path(schema_path: str | None) -> str | None:
    """Extract the lattice id from ``complex_model.lattices.<id>.universe_pattern``."""
    if not schema_path:
        return None
    parts = schema_path.split(".")
    try:
        return parts[parts.index("lattices") + 1]
    except (ValueError, IndexError):
        return None


def _pin_count_mismatch_location(
    plan: SimulationPlan | None,
    issue: ValidationIssue,
    requirement: str,
) -> str:
    """Pinpoint the exact cells to rewrite using the requirement's canonical pin map.

    A count diff alone ('mox7 -2, mox87 +2') does not tell the LLM which of 289
    positions are wrong, so repeated reflections return a byte-identical wrong
    pattern. When the input document carries a canonical pin map for this
    lattice, compare it cell by cell and list the mis-positioned coordinates
    alongside the authoritative rows.
    """
    if plan is None or plan.complex_model is None or not requirement:
        return ""
    lattice_id = _lattice_id_from_schema_path(issue.schema_path)
    if not lattice_id:
        return ""
    lattice = next(
        (lat for lat in plan.complex_model.lattices if lat.id == lattice_id),
        None,
    )
    if lattice is None or not lattice.universe_pattern:
        return ""
    canonical = extract_canonical_pin_map(requirement, lattice_id)
    if canonical is None:
        return ""
    diffs = lattice_cell_mismatches(lattice.universe_pattern, canonical.rows)
    total = sum(len(row) for row in lattice.universe_pattern)
    parts: list[str] = []
    if diffs:
        if len(diffs) <= max(8, total // 4):
            cell_lines = [
                f"  R{row:02d}C{col:02d}: expected {expected!r}, got {actual!r}"
                for row, col, expected, actual in diffs[:64]
            ]
            parts.append(
                "Exact cells whose universe differs from the canonical pin map "
                "(row/col are 1-indexed; rewrite each to the expected universe):\n"
                + "\n".join(cell_lines)
            )
        else:
            parts.append(
                f"{len(diffs)}/{total} cells differ from the canonical map -- the "
                "pattern is broadly wrong; rebuild universe_pattern row by row "
                "from the canonical map below rather than patching individual cells."
            )
    parts.append(
        "Canonical pin map (transcribe row by row from R01; do NOT infer from "
        "symmetry or from the count diff):\n" + canonical.raw_text.strip()
    )
    return "\n".join(parts)


def _requirement_with_expert_feedback(state: GraphState) -> str:
    requirement = state["requirement"]
    feedback = state.get("expert_feedback", [])
    if not feedback:
        return requirement
    resolved_items = _coerce_resolved_expert_items(state.get("resolved_expert_items", []))
    resolved_context = ""
    if resolved_items:
        resolved_context = (
            "\n\nResolved expert feedback items:\n"
            + "\n".join(
                (
                    f"- Question: {item.question}\n"
                    f"  Expert answer: {item.answer}\n"
                    f"  Resolution: {item.status}; semantic_keys={item.semantic_keys}"
                )
                for item in resolved_items
                if item.status in {"resolved", "declined"}
            )
        )
    return (
        f"{requirement}\n\n"
        "Human expert feedback that should guide the structured SimulationPlan:\n"
        + "\n".join(f"- {item}" for item in feedback)
        + resolved_context
        + "\n\n"
        "Expert feedback consumption rules (IMPORTANT):\n"
        "- Treat expert feedback as authoritative unless it conflicts with the original requirement or validated OpenMC constraints.\n"
        "- If expert feedback answers a previous confirmation question, write the answer into the corresponding structured field.\n"
        "- Do not keep requires_human_confirmation entries for items already answered by expert feedback.\n"
        "- Do not keep expert_assumptions entries for assumptions already confirmed or corrected by expert feedback.\n"
        "- Do not ask the same expert question again in a later round.\n"
        "- If the feedback is in a different language from the question, infer the semantic match rather than relying on exact text.\n"
        "- If the feedback confirms use of benchmark/default/document values, represent that confirmation explicitly in the plan or remove the unresolved confirmation marker.\n"
        "- If a value remains genuinely unresolved after considering all expert feedback, then and only then keep requires_human_confirmation."
    )


def _augmented_plan_requirement(state: GraphState) -> str:
    base = _requirement_with_expert_feedback(state)
    docs = state.get("openmc_api_docs", [])
    few_shots = _enrich_few_shots_with_gold(state.get("few_shot_examples", []))
    parts = [
        base,
        _hard_count_constraints_context(state),
        _core_lattice_naming_guidance(),
        "",
        "OpenMC API context retrieved from local Python introspection and official docs references:",
        _compact_context(docs),
        "",
        "Few-shot modeling patterns to follow when relevant:",
        _compact_context(few_shots),
    ]
    return "\n".join(parts)


def _enrich_few_shots_with_gold(few_shots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Attach slim IR + digest summaries to selected gold few-shot cases.

    Non-gold entries (abstract outlines) pass through unchanged. Missing case
    data falls back to the original entry so a missing case never breaks the
    prompt.
    """
    from openmc_agent.few_shot_cases import load_monolithic_few_shot

    enriched: list[dict[str, Any]] = []
    for ex in few_shots:
        if not isinstance(ex, dict) or not ex.get("gold_case_id"):
            enriched.append(ex)
            continue
        try:
            gold = load_monolithic_few_shot(ex["gold_case_id"])
        except FileNotFoundError:
            enriched.append(ex)
            continue
        enriched_ex = dict(ex)
        enriched_ex["gold_ir_summary"] = json.dumps(gold["slim_ir"], ensure_ascii=False)
        enriched_ex["gold_digest_summary"] = gold.get("digest", "")
        enriched.append(enriched_ex)
    return enriched


_COMPACT_WHITELIST: frozenset[str] = frozenset({
    "symbol", "signature", "doc_summary", "official_url",
    "name", "requirement", "structured_outline",
    "gold_case_id", "gold_ir_summary", "gold_digest_summary",
})
_COMPACT_TRUNCATE: dict[str, int] = {
    "gold_ir_summary": 6000,
    "gold_digest_summary": 1500,
}
_COMPACT_DEFAULT_TRUNCATE: int = 700


def _compact_context(items: list[dict[str, Any]], *, limit: int = 6) -> str:
    if not items:
        return "[]"
    compact: list[dict[str, Any]] = []
    for item in items[:limit]:
        row: dict[str, Any] = {}
        for key, value in item.items():
            if key not in _COMPACT_WHITELIST:
                continue
            cap = _COMPACT_TRUNCATE.get(key, _COMPACT_DEFAULT_TRUNCATE)
            row[key] = _truncate_text(str(value), cap)
        compact.append(row)
    return str(compact)


def _plan_name(plan: SimulationPlan) -> str:
    if plan.model_spec is not None:
        return plan.model_spec.name
    if plan.complex_model is not None:
        return plan.complex_model.name
    return "unknown"


def _append_raw_llm_output(state: GraphState, raw_response: str) -> list[str]:
    outputs = list(state.get("raw_llm_outputs", []))
    if raw_response:
        outputs.append(raw_response)
    return outputs


def _trace_event_update(
    state: GraphState,
    event_type: str,
    *,
    summary: str = "",
    report: ValidationReport | None = None,
    capability: RenderCapabilityReport | None = None,
    retrieval_context: RetrievalContext | None = None,
    plan: SimulationPlan | None = None,
    metadata: dict[str, Any] | None = None,
    round_index: int | None = None,
) -> dict[str, Any]:
    """Return a state update with one appended trace event.

    Trace failures are intentionally swallowed so observability never changes
    workflow behavior.
    """
    try:
        recorder = TraceRecorder(config=TraceConfig(), trace=state.get("trace"))
        if recorder.trace.user_request_preview is None and state.get("requirement"):
            recorder.trace.user_request_preview = _truncate_text(
                state.get("requirement", ""), recorder.config.max_preview_chars
            )
        active_report = report or state.get("validation_report")
        issues = list(active_report.issues) if active_report is not None else []
        active_plan = plan or _coerce_simulation_plan(state.get("simulation_plan"))
        active_capability = capability
        if active_capability is None and active_plan is not None:
            active_capability = active_plan.capability_report
        event_metadata: dict[str, Any] = dict(metadata or {})
        if report is not None:
            event_metadata.update(summarize_validation_report(report))
        if capability is not None:
            event_metadata.update(summarize_capability_report(capability))
        if retrieval_context is not None:
            event_metadata["retrieval"] = summarize_retrieval_context_for_trace(
                retrieval_context
            )
        if plan is not None and recorder.config.capture_plan_preview:
            event_metadata["plan_preview"] = preview_plan(
                plan, recorder.config.max_preview_chars
            )
        recorder.add_event(
            event_type,  # type: ignore[arg-type]
            round_index=(
                round_index
                if round_index is not None
                else state.get("retry_count", 0)
            ),
            summary=summary,
            issue_codes=[issue.code for issue in issues],
            route_hints=[issue.route_hint for issue in issues if issue.route_hint],
            renderability=(
                active_capability.renderability if active_capability is not None else None
            ),
            supported_renderer=(
                active_capability.supported_renderer
                if active_capability is not None
                else None
            ),
            metadata=event_metadata,
        )
        if event_type in {"workflow_completed", "workflow_failed"}:
            if event_type == "workflow_failed" or state.get("error"):
                recorder.trace.final_status = "failed"
            elif active_capability is not None and active_capability.renderability == "skeleton":
                recorder.trace.final_status = "skeleton"
            elif active_report is not None and active_report.is_valid:
                recorder.trace.final_status = "valid"
            else:
                recorder.trace.final_status = "invalid"
            if active_capability is not None:
                recorder.trace.final_renderability = active_capability.renderability
                recorder.trace.final_supported_renderer = active_capability.supported_renderer
        return {"trace": recorder.export_json()}
    except Exception:
        return {}


def _write_plan_generation_artifacts(
    state: GraphState,
    *,
    phase: str,
    result: StructuredOutputResult[SimulationPlan],
    retry_count: int,
) -> list[str]:
    output_dir = Path(state.get("output_dir", "data/runs"))
    artifacts_dir = output_dir / "plan_artifacts"
    stage_index = _next_plan_artifact_index(state)
    stage_dir = artifacts_dir / f"{stage_index:03d}_{phase}"
    stage_dir.mkdir(parents=True, exist_ok=True)

    paths = list(state.get("plan_artifacts", []))
    stage_paths: list[str] = []
    if result.raw_response:
        raw_path = stage_dir / "raw_response.txt"
        raw_path.write_text(result.raw_response, encoding="utf-8")
        stage_paths.append(str(raw_path))
    if result.candidate_payload is not None:
        candidate_path = stage_dir / "candidate_plan.json"
        _write_json_file(candidate_path, result.candidate_payload)
        stage_paths.append(str(candidate_path))
    if result.value is not None:
        validated_path = stage_dir / "validated_plan.json"
        _write_json_file(validated_path, result.value.model_dump(mode="json"))
        stage_paths.append(str(validated_path))

    meta_path = stage_dir / "meta.json"
    _write_json_file(
        meta_path,
        {
            "phase": phase,
            "ok": result.ok,
            "error": result.error,
            "parse_notes": result.parse_notes or [],
            "retry_count": retry_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "artifacts": stage_paths,
        },
    )
    stage_paths.append(str(meta_path))
    paths.extend(stage_paths)
    return paths


def _write_final_simulation_plan(
    state: GraphState,
    plan: SimulationPlan,
    *,
    existing_paths: list[str] | None = None,
) -> list[str]:
    output_dir = Path(state.get("output_dir", "data/runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "simulation_plan.json"
    _write_json_file(plan_path, plan.model_dump(mode="json"))
    paths = existing_paths if existing_paths is not None else state.get("plan_artifacts", [])
    return _append_plan_artifact_path(paths, plan_path)


def _next_plan_artifact_index(state: GraphState) -> int:
    count = 0
    for path in state.get("plan_artifacts", []):
        artifact_path = Path(path)
        if artifact_path.name == "meta.json" and artifact_path.parent.parent.name == "plan_artifacts":
            count += 1
    return count


def _append_plan_artifact_path(paths: list[str], path: Path) -> list[str]:
    text = str(path)
    updated = list(paths)
    if text not in updated:
        updated.append(text)
    return updated


def _write_json_file(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _compact_tool_results(tool_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for result in tool_results:
        compact.append(
            {
                "name": result.get("name"),
                "ok": result.get("ok"),
                "returncode": result.get("returncode"),
                "stdout": _truncate_text(result.get("stdout", "")),
                "stderr": _truncate_text(result.get("stderr", "")),
                "error": _truncate_text(result.get("error", "")),
                "artifacts": result.get("artifacts", []),
            }
        )
    return compact


def _truncate_text(text: str, limit: int = 1200) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "...[truncated]"


def _progress(state: GraphState, node: str, message: str) -> None:
    _logger.info("[node:%s] %s", node, message)
