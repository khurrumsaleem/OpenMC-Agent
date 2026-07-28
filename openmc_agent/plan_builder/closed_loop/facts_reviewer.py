"""Independent Facts Critic invocation and strict Python normalization."""

from __future__ import annotations

from openmc_agent.structured_output import canonical_payload_hash

from typing import Any

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .facts_review_prompts import build_facts_review_prompt, build_facts_review_schema_retry_prompt
from .models import (
    FactsReviewFindingDraft, FactsReviewModelOutput, PlanClosedLoopPolicy, PlanEvidencePack, PlanFindingCategory,
    PlanFindingSeverity, PlanGateId, PlanReviewFinding, SourceExcerpt,
)
from .review_io import (
    StructuredReviewCallSpec,
    _ACCEPTED_REVIEW_STATUSES,
    run_structured_review_call,
)


class FactsReviewResult(AgentBaseModel):
    ok: bool = False
    findings: list[PlanReviewFinding] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    outputs: list[dict[str, Any]] = Field(default_factory=list)
    coverage_complete: bool = False
    reviewer_calls: int = 0
    schema_retries: int = 0
    error: str = ""
    raw_outputs: list[str] = Field(default_factory=list)
    call_metadata: list[dict[str, Any]] = Field(default_factory=list)
    failure_code: str = ""


_FACTS_RECORDING_METADATA_ROOTS = (
    "/missing_facts",
    "/assumptions",
    "/source_notes",
)


def _is_facts_recording_metadata_path(path: str) -> bool:
    return any(path == root or path.startswith(f"{root}/") for root in _FACTS_RECORDING_METADATA_ROOTS)


def _normalize_recording_metadata_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    # Metadata repairs record an evidence-backed gap; they never infer a value.
    paths = draft.affected_json_paths
    if not paths or not all(_is_facts_recording_metadata_path(path) for path in paths):
        return draft, None
    original = {
        "reason": "facts_metadata_recording",
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={"repairable_by_llm": True, "requires_human": False}), original


def _normalize_confirmation_not_error_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Downgrade self-contradictory reviewer confirmations.

    Real reviewers occasionally emit an error-severity finding whose own
    message explicitly says the condition is a coverage confirmation rather
    than an error.  Treating that as a non-repairable blocker prevents the
    revision loop from fixing genuinely remaining repairable findings.
    Keep the finding visible as a warning with provenance instead of
    suppressing the code.
    """
    if (
        draft.severity is not PlanFindingSeverity.ERROR
        or draft.category is not PlanFindingCategory.SOURCE_COVERAGE
        or draft.requires_human
        or draft.repairable_by_llm
    ):
        return draft, None
    message = draft.message.lower()
    confirmation_markers = (
        "recorded as a coverage confirmation",
        "not an error",
    )
    if not all(marker in message for marker in confirmation_markers):
        return draft, None
    original = {
        "reason": "facts_reviewer_confirmation_not_error",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={"severity": PlanFindingSeverity.WARNING}), original


def _normalize_downstream_material_scope_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Keep Materials/MU-owned fuel details from blocking the Facts Gate.

    Facts fuel variants identify source-declared variants and coarse attributes
    such as enrichment/density when supplied.  Isotope vectors, oxygen
    stoichiometry, and material composition are enforced by Materials /
    Material-Universe contracts.  A reviewer may still mention that downstream
    gap while looking at ``/fuel_variant_requirements``; preserve it as a
    warning with owner metadata instead of treating it as a Facts repair.
    """

    code = draft.code.strip().lower()
    paths = tuple(draft.affected_json_paths)
    message = draft.message.lower()
    material_detail_markers = (
        "isotope",
        "isotopic",
        "composition",
        "stoichiometry",
        "o-16",
        "u-234",
        "u-235",
        "u-236",
        "u-238",
    )
    is_known_code = code in {
        "fuel_variant_missing_isotope_composition",
        "facts.fuel_variant_missing_isotope_composition",
    }
    is_fuel_variant_material_detail = (
        any(path == "/fuel_variant_requirements" or path.startswith("/fuel_variant_requirements/") for path in paths)
        and any(marker in message for marker in material_detail_markers)
    )
    if not (is_known_code or is_fuel_variant_material_detail):
        return draft, None

    downstream_impact = list(dict.fromkeys([
        *draft.downstream_impact,
        "materials_contract",
        "material_universe_contract",
    ]))
    original = {
        "reason": "facts_downstream_material_scope",
        "owner_route": "materials",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
        "downstream_impact": downstream_impact,
    }), original


def _normalize_blank_operating_state_classification(
    draft: FactsReviewFindingDraft,
) -> tuple[FactsReviewFindingDraft, dict[str, Any] | None]:
    """Treat an explicitly blank operating state as a recorded base-state gap.

    VERA-style inputs may carry an empty operating_state field to mean the case
    has no named operating variant.  That is not a human-required Facts
    ambiguity when the source itself is explicitly blank.  Downstream code uses
    canonical base-state labels for control-state IDs; the Facts Gate should
    keep this visible without blocking.
    """

    code = draft.code.strip().lower()
    message = draft.message.lower()
    paths = tuple(draft.affected_json_paths)
    blank_markers = (
        '""',
        "blank",
        "empty",
        "omitted",
        "unspecified",
    )
    is_operating_state_code = code in {
        "missing_operating_state",
        "missing_operating_state_unrecorded",
        "facts.missing_operating_state",
    }
    is_selected_variant_path = any(
        path == "/selected_variant" or path.startswith("/selected_variant/")
        for path in paths
    )
    mentions_operating_state = "operating state" in message or "operating_state" in message
    mentions_blank_source = any(marker in message for marker in blank_markers)
    if not (is_operating_state_code and (is_selected_variant_path or mentions_operating_state) and mentions_blank_source):
        return draft, None

    original = {
        "reason": "facts_blank_operating_state_base_gap",
        "canonical_value": "base",
        "original_severity": draft.severity.value,
        "original_repairable_by_llm": draft.repairable_by_llm,
        "original_requires_human": draft.requires_human,
    }
    return draft.model_copy(update={
        "severity": PlanFindingSeverity.WARNING,
        "repairable_by_llm": False,
        "requires_human": False,
    }), original


def _is_excerpt_limited_unsupported_inference(draft: FactsReviewFindingDraft) -> bool:
    """Reject chunk-local unsupported-inference claims.

    Facts review can be split across source chunks.  A reviewer for one chunk
    may say a value is "not provided in this source excerpt" even though a
    later chunk contains the source-backed value.  That is not a valid
    whole-source ambiguity and must not be promoted to a human blocker.
    """

    code = draft.code.strip().lower()
    message = draft.message.lower()
    expected = str(draft.expected_value or "").lower()
    if draft.category is not PlanFindingCategory.UNSUPPORTED_INFERENCE:
        return False
    if draft.requires_human:
        return (
            "source excerpt" in message
            or "provided text" in message
            or "provided source excerpt" in message
            or "not provided in source excerpt" in expected
            or "not provided in the source excerpt" in expected
        )
    return code.startswith("unsupported_inference.") and (
        "source excerpt" in message
        or "provided text" in message
    )


def _normalize(output: FactsReviewModelOutput, pack: PlanEvidencePack) -> tuple[list[PlanReviewFinding], list[dict[str, Any]]]:
    evidence = {item.evidence_hash: item for item in pack.source_excerpts}
    facts = pack.relevant_patches.get("facts", {})
    accepted: list[PlanReviewFinding] = []
    rejected: list[dict[str, Any]] = []
    for index, draft in enumerate(output.findings):
        if not draft.code.strip():
            rejected.append({"code": "facts_review.invalid_finding_contract", "reason": "blank code"})
            continue
        unknown = set(draft.evidence_hashes) - set(evidence)
        if unknown:
            rejected.append({"code": "facts_review.unknown_evidence_hash", "finding_code": draft.code, "unknown": sorted(unknown)})
            continue
        normalized_paths: list[str] = []
        for p in draft.affected_json_paths:
            if p.startswith("facts_subset."):
                p = "/" + p[len("facts_subset."):]
            elif not p.startswith("/"):
                p = "/" + p
            normalized_paths.append(p)
        if normalized_paths != draft.affected_json_paths:
            draft = draft.model_copy(update={"affected_json_paths": normalized_paths})
        if any(not path.startswith("/") or path.startswith("/materials") or path.startswith("/universes") for path in draft.affected_json_paths):
            rejected.append({"code": "facts_review.path_out_of_scope", "finding_code": draft.code})
            continue
        draft, classification_override = _normalize_recording_metadata_classification(draft)
        if classification_override is None:
            draft, classification_override = _normalize_confirmation_not_error_classification(draft)
        if classification_override is None:
            draft, classification_override = _normalize_downstream_material_scope_classification(draft)
        if classification_override is None:
            draft, classification_override = _normalize_blank_operating_state_classification(draft)
        if _is_excerpt_limited_unsupported_inference(draft):
            rejected.append({
                "code": "facts_review.excerpt_limited_unsupported_inference",
                "finding_code": draft.code,
                "reason": "chunk-local unsupported-inference claim is not a whole-source human blocker",
            })
            output.findings[index] = draft.model_copy(update={"severity": PlanFindingSeverity.WARNING, "requires_human": False})
            continue
        output.findings[index] = draft
        if draft.requires_human and draft.repairable_by_llm:
            rejected.append({"code": "facts_review.invalid_finding_contract", "finding_code": draft.code})
            continue
        if draft.severity is PlanFindingSeverity.ERROR and draft.category is not PlanFindingCategory.PHYSICAL_AMBIGUITY:
            # An error must be both evidence-grounded and actionable against
            # a specific FactsPatch field.  This rejects self-contradictory
            # critic output such as "this is consistent; no issue" labelled
            # as an error with no affected path; accepting such a finding
            # would fail-close a valid plan without a repairable contract.
            if not draft.evidence_hashes or not draft.affected_json_paths:
                rejected.append({"code": "facts_review.invalid_finding_contract", "finding_code": draft.code})
                continue
        excerpts = [evidence[key] for key in draft.evidence_hashes]
        finding = PlanReviewFinding(
            gate_id=PlanGateId.FACTS, code=draft.code, severity=draft.severity,
            category=draft.category, message=draft.message, source_evidence=excerpts,
            affected_patch_types=["facts"], affected_json_paths=draft.affected_json_paths,
            repairable_by_llm=draft.repairable_by_llm, requires_human=draft.requires_human,
            confidence=draft.confidence,
            metadata={"expected_value": draft.expected_value, "current_value": draft.current_value,
                      "candidate_interpretations": [item.model_dump(mode="json") for item in draft.candidate_interpretations],
                      "downstream_impact": draft.downstream_impact,
                      **({"classification_override": classification_override} if classification_override else {})},
        )
        accepted.append(finding)
    # Finding identity excludes wording and unions evidence under the same semantic fingerprint.
    merged: dict[str, PlanReviewFinding] = {item.finding_id: item for item in accepted}
    return list(merged.values()), rejected


def _aggregate_coverage(outputs: list[dict[str, Any]], *, expected_stage_count: int | None = None) -> bool:
    """Aggregate review-stage outputs into a single coverage decision.

    Rules (fail-closed):

    * Every stage ``review_status`` must be in ``_ACCEPTED_REVIEW_STATUSES``.
    * No finding may have ``severity == "error"`` (blocking).
    * The number of outputs must match *expected_stage_count* when provided.
    """
    if not outputs:
        return False
    if expected_stage_count is not None and len(outputs) != expected_stage_count:
        return False
    statuses = [
        str(item.get("output", {}).get("review_status", "")).strip()
        for item in outputs
    ]
    if any(status not in _ACCEPTED_REVIEW_STATUSES for status in statuses):
        return False
    has_blocking = any(
        str(finding.get("severity", "")).lower() == "error"
        for item in outputs
        for finding in item.get("output", {}).get("findings", [])
    )
    return not has_blocking


def run_facts_review(*, evidence_packs: list[PlanEvidencePack], reviewer_client: Any, state: Any, policy: PlanClosedLoopPolicy) -> FactsReviewResult:
    # Phase 8B Step 3: stage-split path.
    if getattr(policy, "facts_review_stage_split", False) and evidence_packs:
        return _run_facts_review_staged(
            evidence_packs=evidence_packs,
            reviewer_client=reviewer_client,
            state=state,
            policy=policy,
        )
    result = FactsReviewResult()
    all_findings: list[PlanReviewFinding] = []
    all_rejected: list[dict[str, Any]] = []
    for pack in evidence_packs:
        call = run_structured_review_call(
            client=reviewer_client, initial_prompt=build_facts_review_prompt(pack),
            retry_prompt_builder=lambda raw, error: build_facts_review_schema_retry_prompt(pack, error, raw),
            output_model=FactsReviewModelOutput,
            call_spec=StructuredReviewCallSpec(
                role_id="facts_review", gate_id=PlanGateId.FACTS,
                schema_name="FactsReviewModelOutput", json_schema=FactsReviewModelOutput.model_json_schema(),
                artifact_prefix="facts_review",
            input_payload_hash=canonical_payload_hash(pack)
            ), state=state, stage=state.plan_loop_stages.get("plan_gate_facts"), policy=policy,
        )
        result.reviewer_calls += call.call_count
        result.schema_retries += call.schema_retry_count
        result.raw_outputs.extend(call.raw_outputs)
        for attempt in call.attempts:
            result.call_metadata.append({"pack_id": pack.evidence_pack_id, **attempt.model_dump(mode="json", exclude={"raw_text"})})
        if not call.ok or call.parsed_output is None:
            # Phase 8B Step 3: classify the failure precisely.
            result.error = f"facts_review.schema_invalid: {call.error_detail}"
            result.failure_code = _classify_review_failure(call)
            return result
        output = FactsReviewModelOutput.model_validate(call.parsed_output)
        findings, rejected = _normalize(output, pack)
        all_findings.extend(findings)
        all_rejected.extend(rejected)
        result.outputs.append({"pack_id": pack.evidence_pack_id, "output": output.model_dump(mode="json")})
    result.findings = list({finding.finding_id: finding for finding in all_findings}.values())
    result.rejected = all_rejected
    expected = {item.evidence_hash for pack in evidence_packs for item in pack.source_excerpts}
    reviewed: set[str] = set()
    for out_entry in result.outputs:
        reviewed.update(out_entry["output"].get("reviewed_evidence_hashes", []))
        for finding in out_entry["output"].get("findings", []):
            reviewed.update(finding.get("evidence_hashes", []))
    result.coverage_complete = (
        _aggregate_coverage(result.outputs, expected_stage_count=len(evidence_packs))
        and not any(item.severity is PlanFindingSeverity.ERROR for item in result.findings)
    )
    if not result.coverage_complete:
        result.failure_code = "facts_review.coverage_incomplete"
    result.ok = not result.error
    return result


# ---------------------------------------------------------------------------
# Phase 8B Step 3: staged review path
# ---------------------------------------------------------------------------


def _run_facts_review_staged(
    *,
    evidence_packs: list[PlanEvidencePack],
    reviewer_client: Any,
    state: Any,
    policy: PlanClosedLoopPolicy,
) -> FactsReviewResult:
    """Run per-topic stage calls instead of one monolithic per-pack call.

    Each stage sees only its FactsPatch subset + the source excerpts,
    producing a much smaller and more focused prompt.
    """

    from .facts_review_stages import (
        STAGE_ORDER,
        FactsReviewStageRequest,
        build_stage_review_prompt,
        build_stage_schema_retry_prompt,
        extract_facts_subset,
    )

    result = FactsReviewResult()
    all_findings: list[PlanReviewFinding] = []
    all_rejected: list[dict[str, Any]] = []

    # Use the first pack as the evidence source.  Stage-split mode does
    # not iterate per-pack; it iterates per-stage using the consolidated
    # evidence from the first pack.
    base_pack = evidence_packs[0]
    facts_patch = base_pack.relevant_patches.get("facts", {})
    base_excerpts = [
        s.model_dump(mode="json") if hasattr(s, "model_dump") else s
        for s in base_pack.source_excerpts
    ]
    confirmed_summary = base_pack.metadata.get("facts_summary", {})
    consistency_issues = base_pack.metadata.get("facts_consistency_issues", [])

    for stage in STAGE_ORDER:
        facts_subset = extract_facts_subset(facts_patch, stage)
        stage_request = FactsReviewStageRequest(
            stage=stage,
            target_fields=tuple(facts_subset.keys()),
            facts_subset=facts_subset,
            evidence_excerpts=base_excerpts,
            confirmed_facts_summary=confirmed_summary,
            consistency_issues=consistency_issues,
        )
        call = run_structured_review_call(
            client=reviewer_client,
            initial_prompt=build_stage_review_prompt(stage_request, base_pack),
            retry_prompt_builder=lambda raw, error, sr=stage_request, bp=base_pack: build_stage_schema_retry_prompt(sr, bp, error, raw),
            output_model=FactsReviewModelOutput,
            call_spec=StructuredReviewCallSpec(
                role_id="facts_review",
                gate_id=PlanGateId.FACTS,
                schema_name="FactsReviewModelOutput",
                json_schema=FactsReviewModelOutput.model_json_schema(),
                artifact_prefix=f"facts_review_{stage.value}",
                input_payload_hash=canonical_payload_hash(
                    {"stage": stage.value, "facts_subset": facts_subset}
                ),
            ),
            state=state,
            stage=state.plan_loop_stages.get("plan_gate_facts"),
            policy=policy,
        )
        result.reviewer_calls += call.call_count
        result.schema_retries += call.schema_retry_count
        result.raw_outputs.extend(call.raw_outputs)
        for attempt in call.attempts:
            result.call_metadata.append(
                {"stage": stage.value, **attempt.model_dump(mode="json", exclude={"raw_text"})}
            )
        if not call.ok or call.parsed_output is None:
            result.error = f"facts_review.schema_invalid[{stage.value}]: {call.error_detail}"
            result.failure_code = _classify_review_failure(call)
            return result
        output = FactsReviewModelOutput.model_validate(call.parsed_output)
        # Build a synthetic pack for _normalize that carries the same
        # source excerpts but a pruned facts subset.
        synthetic_pack = base_pack.model_copy(
            update={"relevant_patches": {"facts": facts_subset}}
        )
        findings, rejected = _normalize(output, synthetic_pack)
        all_findings.extend(findings)
        all_rejected.extend(rejected)
        result.outputs.append(
            {"stage": stage.value, "output": output.model_dump(mode="json")}
        )

    result.findings = list({finding.finding_id: finding for finding in all_findings}.values())
    result.rejected = all_rejected
    result.coverage_complete = _aggregate_coverage(result.outputs, expected_stage_count=len(STAGE_ORDER))
    if not result.coverage_complete:
        result.failure_code = "facts_review.coverage_incomplete"
    result.ok = not result.error
    return result


# ---------------------------------------------------------------------------
# Phase 8B Step 3: failure classification
# ---------------------------------------------------------------------------

# Phrases that indicate a free-text "approve" (no JSON structure).
_FREE_TEXT_APPROVE_PHRASES: tuple[str, ...] = (
    "looks good",
    "no issues",
    "no issues found",
    "everything looks correct",
    "i approve",
    "approved",
    "accepted",
    "no findings",
    "no errors",
    "all correct",
    "consistent with the source",
    "patch is correct",
)


def _classify_review_failure(call: Any) -> str:
    """Classify a failed structured review call precisely.

    Distinguishes:
    * ``facts_review.budget_exhausted`` — LLM budget ran out.
    * ``facts.reviewer_empty_response`` — all attempts returned empty content.
    * ``facts.reviewer_free_text_approve`` — prose-only "approve" without JSON.
    * ``facts_review.schema_invalid`` — JSON was present but malformed.
    """

    if call.error_code == "planning.closed_loop.budget_exhausted":
        return "facts_review.budget_exhausted"

    # If the call was successful, empty raw_text is a data propagation bug.
    if call.ok is True:
        return call.error_code or "facts_review.schema_invalid"

    # Inspect raw outputs captured at the provider boundary (P1 fix).
    # Fall back to attempt.raw_text for callers that haven't been updated.
    raw_texts = list(getattr(call, "raw_outputs", []) or [])
    if not raw_texts:
        raw_texts = [
            getattr(a, "raw_text", "") or "" for a in (call.attempts or [])
        ]
    # Empty response: all attempts produced empty content.
    if raw_texts and all(not text.strip() for text in raw_texts):
        return "facts.reviewer_empty_response"
    # Free-text approve: the response is short prose that matches an
    # approval phrase but contains no JSON structure.
    for text in raw_texts:
        lower = text.strip().lower()
        if lower and len(lower) < 200 and "{" not in lower:
            if any(phrase in lower for phrase in _FREE_TEXT_APPROVE_PHRASES):
                return "facts.reviewer_free_text_approve"
    return call.error_code or "facts_review.schema_invalid"
