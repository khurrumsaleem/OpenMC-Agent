"""Independent Placement Critic call and deterministic normalization."""

from __future__ import annotations

from openmc_agent.structured_output import canonical_payload_hash

from typing import Any

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .models import (
    PlanClosedLoopPolicy, PlanFindingSeverity, PlanGateId, PlanReviewFinding,
    PlacementEvidencePack, PlacementReviewModelOutput,
)
from .placement_issue_policy import placement_issue_owner
from .placement_review_prompts import build_placement_review_prompt, build_placement_review_schema_retry_prompt
from .review_io import StructuredReviewCallSpec, run_structured_review_call


class PlacementReviewResult(AgentBaseModel):
    ok: bool = False
    findings: list[PlanReviewFinding] = Field(default_factory=list)
    rejected: list[dict[str, Any]] = Field(default_factory=list)
    output: dict[str, Any] | None = None
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    reviewer_calls: int = 0
    schema_retries: int = 0
    coverage_complete: bool = False
    error_code: str = ""
    error_detail: str = ""


def _normalize(output: PlacementReviewModelOutput, pack: PlacementEvidencePack) -> tuple[list[PlanReviewFinding], list[dict[str, Any]]]:
    evidence = {item.ref_id: item for item in pack.evidence_items}
    row_ids = {row.requirement_id for row in pack.contract_matrix.rows}
    findings: list[PlanReviewFinding] = []
    rejected: list[dict[str, Any]] = []
    for draft in output.findings:
        if not draft.code.strip() or draft.requires_human and draft.repairable_by_llm:
            rejected.append({"code": "placement_review.invalid_finding_contract", "finding_code": draft.code})
            continue
        unknown_refs = set(draft.evidence_refs) - set(evidence)
        if unknown_refs:
            rejected.append({"code": "placement_review.unknown_evidence_ref", "finding_code": draft.code, "unknown": sorted(unknown_refs)})
            continue
        unknown_rows = set(draft.affected_contract_rows) - row_ids
        if unknown_rows:
            rejected.append({"code": "placement_review.unknown_contract_row", "finding_code": draft.code, "unknown": sorted(unknown_rows)})
            continue
        if any(not path.startswith("/") or path.startswith(("/facts", "/materials", "/settings", "/axial")) for path in draft.affected_json_paths):
            rejected.append({"code": "placement_review.path_out_of_scope", "finding_code": draft.code})
            continue
        owner = placement_issue_owner(draft.code)
        owners = owner.get("owner_patch_types", [])
        if not owners and draft.repairable_by_llm:
            rejected.append({"code": "placement_review.owner_out_of_scope", "finding_code": draft.code})
            continue
        if draft.severity is PlanFindingSeverity.ERROR and not draft.evidence_refs:
            rejected.append({"code": "placement_review.invalid_finding_contract", "finding_code": draft.code})
            continue
        findings.append(PlanReviewFinding(
            gate_id=PlanGateId.PLACEMENT, code=draft.code, severity=draft.severity,
            category=draft.category, message=draft.message, source_evidence=[],
            affected_patch_types=owners, affected_json_paths=draft.affected_json_paths,
            repairable_by_llm=draft.repairable_by_llm and bool(owners), requires_human=draft.requires_human,
            confidence=draft.confidence, metadata={
                "evidence_refs": draft.evidence_refs, "evidence_hashes": [evidence[ref].canonical_hash for ref in draft.evidence_refs],
                "affected_contract_rows": draft.affected_contract_rows, "expected_value": draft.expected_value,
                "current_value": draft.current_value, "candidate_interpretations": [item.model_dump(mode="json") for item in draft.candidate_interpretations],
                "downstream_impact": draft.downstream_impact,
            },
        ))
    return list({item.finding_id: item for item in findings}.values()), rejected


def run_placement_review(*, evidence_pack: PlacementEvidencePack, reviewer_client: Any, state: Any, policy: PlanClosedLoopPolicy) -> PlacementReviewResult:
    call = run_structured_review_call(
        client=reviewer_client, initial_prompt=build_placement_review_prompt(evidence_pack),
        retry_prompt_builder=lambda raw, error: build_placement_review_schema_retry_prompt(evidence_pack, error, raw),
        output_model=PlacementReviewModelOutput,
        call_spec=StructuredReviewCallSpec(role_id="placement_review", gate_id=PlanGateId.PLACEMENT, schema_name="PlacementReviewModelOutput", json_schema=PlacementReviewModelOutput.model_json_schema(), artifact_prefix="placement_review", input_payload_hash=canonical_payload_hash(evidence_pack)),
        state=state, stage=state.plan_loop_stages.get("plan_gate_placement"), policy=policy,
    )
    result = PlacementReviewResult(reviewer_calls=call.call_count, schema_retries=call.schema_retry_count, attempts=[item.model_dump(mode="json") for item in call.attempts], error_code=call.error_code, error_detail=call.error_detail)
    if not call.ok or call.parsed_output is None:
        return result
    output = PlacementReviewModelOutput.model_validate(call.parsed_output)
    findings, rejected = _normalize(output, evidence_pack)
    required_rows = {item.requirement_id for item in evidence_pack.contract_matrix.rows}
    required_refs = {item.ref_id for item in evidence_pack.evidence_items}
    reviewed_rows = set(output.reviewed_contract_row_ids)
    reviewed_refs = set(output.reviewed_evidence_refs)
    # ``complete_with_gaps`` means the reviewer completed coverage and found
    # semantic gaps; it must not be treated as a coverage failure.  The
    # row/ref lists are the deterministic proof of coverage and remain
    # fail-closed for incomplete or malformed reviewer outputs.
    result.coverage_complete = (
        output.review_status in {"complete", "complete_with_gaps"}
        and required_rows.issubset(reviewed_rows)
        and required_refs.issubset(reviewed_refs)
    )
    if not result.coverage_complete:
        result.error_code = "placement_review.coverage_incomplete"
    result.ok = result.coverage_complete
    result.findings, result.rejected, result.output = findings, rejected, output.model_dump(mode="json")
    return result
