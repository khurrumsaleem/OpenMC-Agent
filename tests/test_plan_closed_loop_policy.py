from __future__ import annotations

from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy, PlanReviewAction, PlanReviewFinding, PlanStageState
from openmc_agent.plan_builder.closed_loop.policy import compute_allowed_actions, gate_for_patch_type


def _finding(**updates):
    data = dict(gate_id="facts", code="x", severity="error", category="source_coverage", message="x", confidence=1.0)
    data.update(updates)
    return PlanReviewFinding(**data)


def test_action_policy_and_primary_gate_are_deterministic() -> None:
    stage = PlanStageState(stage_id="s", gate_id="facts")
    assert compute_allowed_actions(policy=PlanClosedLoopPolicy(), stage_state=stage, findings=[], deterministic_issues=[]) == []
    advisory = PlanClosedLoopPolicy(mode="advisory")
    assert [str(v.value) for v in compute_allowed_actions(policy=advisory, stage_state=stage, findings=[], deterministic_issues=[])] == ["approve"]
    assert [v.value for v in compute_allowed_actions(policy=advisory, stage_state=stage, findings=[_finding(requires_human=True)], deterministic_issues=[])] == ["fail_closed"]
    human = advisory.model_copy(update={"enable_human_gate": True})
    assert [v.value for v in compute_allowed_actions(policy=human, stage_state=stage, findings=[_finding(requires_human=True)], deterministic_issues=[])] == ["ask_human", "fail_closed"]
    assert [v.value for v in compute_allowed_actions(policy=advisory, stage_state=stage, findings=[_finding(repairable_by_llm=True)], deterministic_issues=[])] == ["revise_current_patch", "retry_dependency"]
    exhausted = stage.model_copy(update={"no_progress_count": 1})
    assert [v.value for v in compute_allowed_actions(policy=advisory, stage_state=exhausted, findings=[], deterministic_issues=[])] == ["fail_closed"]
    assert gate_for_patch_type("pin_map").value == "placement"


def test_clean_review_can_approve_on_final_review_round() -> None:
    policy = PlanClosedLoopPolicy(mode="controlled", max_review_rounds_per_gate=2)
    stage = PlanStageState(stage_id="s", gate_id="placement", review_count=2)

    assert compute_allowed_actions(
        policy=policy,
        stage_state=stage,
        findings=[],
        deterministic_issues=[],
    ) == [PlanReviewAction.APPROVE]

    assert compute_allowed_actions(
        policy=policy,
        stage_state=stage,
        findings=[_finding(repairable_by_llm=True)],
        deterministic_issues=[],
    ) == [PlanReviewAction.FAIL_CLOSED]
