import json

from openmc_agent.plan_builder.closed_loop.facts_evidence import build_facts_evidence_packs
from openmc_agent.plan_builder.closed_loop.facts_reviewer import run_facts_review
from openmc_agent.plan_builder.closed_loop.models import PlanClosedLoopPolicy
from openmc_agent.plan_builder.state import PlanBuildState


def test_reviewer_normalizes_only_evidence_backed_facts_findings() -> None:
    policy = PlanClosedLoopPolicy()
    packs = build_facts_evidence_packs(requirement_text="variant A\n", facts_patch={"patch_type": "facts"}, confirmed_facts={}, planning_metadata={}, policy=policy)
    evidence = packs[0].source_excerpts[0].evidence_hash
    payload = {"review_status": "complete", "reviewed_evidence_hashes": [evidence], "coverage_summary": {}, "findings": [{"code": "facts.variant_missing", "severity": "warning", "category": "source_coverage", "message": "missing", "evidence_hashes": [evidence], "affected_json_paths": ["/fuel_variant_requirements"], "repairable_by_llm": True, "requires_human": False, "confidence": 0.9}]}
    result = run_facts_review(evidence_packs=packs, reviewer_client=lambda _: json.dumps(payload), state=PlanBuildState(state_id="s", requirement_text="r"), policy=policy)
    assert result.ok and result.coverage_complete and result.findings[0].affected_patch_types == ["facts"]


def test_stage_reviewer_downgrades_downstream_fuel_composition_gap() -> None:
    policy = PlanClosedLoopPolicy(facts_review_stage_split=True)
    facts = {
        "patch_type": "facts",
        "fuel_variant_requirements": [{"variant_id": "region1"}, {"variant_id": "region2"}],
        "selected_variant": None,
        "material_roles": [],
        "missing_facts": [],
        "assumptions": [],
        "source_notes": [],
        "benchmark_id": "VERA4",
    }
    packs = build_facts_evidence_packs(
        requirement_text="Table P4-2 gives U-234/U-235/U-236/U-238 wt% and O-16 by UO2 stoichiometry.",
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash

    def reviewer(prompt: str) -> str:
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        findings = []
        if payload["stage"] == "fuel_variant":
            findings = [{
                "code": "FUEL_VARIANT_MISSING_ISOTOPE_COMPOSITION",
                "severity": "error",
                "category": "source_coverage",
                "message": "Table P4-2 provides full isotope composition and O-16 stoichiometry; fuel_variant_requirements do not contain isotope-level composition.",
                "evidence_hashes": [evidence],
                "affected_json_paths": ["/fuel_variant_requirements/0", "/fuel_variant_requirements/1"],
                "repairable_by_llm": True,
                "requires_human": False,
                "confidence": 0.95,
            }]
        return json.dumps({
            "review_status": "complete_with_gaps" if findings else "complete",
            "reviewed_evidence_hashes": [evidence],
            "coverage_summary": {},
            "findings": findings,
        })

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=reviewer,
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert result.coverage_complete
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.code == "FUEL_VARIANT_MISSING_ISOTOPE_COMPOSITION"
    assert finding.severity.value == "warning"
    assert finding.repairable_by_llm is False
    assert finding.requires_human is False
    assert finding.metadata["classification_override"]["owner_route"] == "materials"
    assert "material_universe_contract" in finding.metadata["downstream_impact"]


def test_stage_reviewer_downgrades_blank_operating_state_gap() -> None:
    policy = PlanClosedLoopPolicy(facts_review_stage_split=True)
    facts = {
        "patch_type": "facts",
        "fuel_variant_requirements": [{"variant_id": "fuel"}],
        "selected_variant": None,
        "missing_facts": [],
        "assumptions": [],
        "source_notes": [],
        "benchmark_id": "VERA4",
    }
    packs = build_facts_evidence_packs(
        requirement_text='The source operating_state field is explicitly blank ("").',
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash

    def reviewer(prompt: str) -> str:
        payload = json.loads(prompt.split("INPUT:\n", 1)[1])
        findings = []
        if payload["stage"] == "completeness":
            findings = [{
                "code": "MISSING_OPERATING_STATE",
                "severity": "error",
                "category": "source_coverage",
                "message": 'The operating state value is explicitly blank ("") in the source text and omitted in the FactsPatch.',
                "evidence_hashes": [evidence],
                "affected_json_paths": ["/selected_variant"],
                "repairable_by_llm": False,
                "requires_human": True,
                "confidence": 0.95,
            }]
        return json.dumps({
            "review_status": "complete_with_gaps" if findings else "complete",
            "reviewed_evidence_hashes": [evidence],
            "coverage_summary": {},
            "findings": findings,
        })

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=reviewer,
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert result.coverage_complete
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.code == "MISSING_OPERATING_STATE"
    assert finding.severity.value == "warning"
    assert finding.repairable_by_llm is False
    assert finding.requires_human is False
    assert finding.metadata["classification_override"]["canonical_value"] == "base"
