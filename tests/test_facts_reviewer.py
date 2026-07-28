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


def test_reviewer_rejects_source_note_schema_boundary_false_positive() -> None:
    policy = PlanClosedLoopPolicy()
    facts = {
        "patch_type": "facts",
        "missing_facts": [],
        "assumptions": [],
        "source_notes": [
            "Material densities: coolant density=0.743 g/cm3; absorber density=2.25 g/cm3.",
            "Absorber composition: B10=0.712 wt%; B11=3.170 wt%; O16=55.217 wt%; Si=40.901 wt%.",
        ],
    }
    packs = build_facts_evidence_packs(
        requirement_text="Table lists material densities and absorber composition.",
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_evidence_hashes": [evidence],
        "coverage_summary": {},
        "findings": [{
            "code": "MISSING_PYREX_COMPOSITION",
            "severity": "error",
            "category": "source_coverage",
            "message": "B10 0.712, B11 3.170, O16 55.217, and Si 40.901 are only recorded in source_notes, not dedicated structured fields.",
            "expected_value": {"B-10": 0.00712, "B-11": 0.0317, "O-16": 0.55217, "Si": 0.40901},
            "evidence_hashes": [evidence],
            "affected_json_paths": ["/"],
            "repairable_by_llm": True,
            "requires_human": False,
            "confidence": 0.9,
        }],
    }

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=lambda _: json.dumps(payload),
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert result.coverage_complete
    assert len(result.findings) == 1
    finding = result.findings[0]
    assert finding.code == "MISSING_PYREX_COMPOSITION"
    assert finding.severity.value == "warning"
    assert finding.repairable_by_llm is False
    assert finding.requires_human is False
    assert finding.metadata["classification_override"]["reason"] == "facts_source_note_schema_boundary_covered"


def test_reviewer_keeps_uncovered_source_note_boundary_gap_blocking() -> None:
    policy = PlanClosedLoopPolicy()
    facts = {
        "patch_type": "facts",
        "missing_facts": [],
        "assumptions": [],
        "source_notes": ["Material density: coolant density=0.743 g/cm3."],
    }
    packs = build_facts_evidence_packs(
        requirement_text="Table lists material densities.",
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_evidence_hashes": [evidence],
        "coverage_summary": {},
        "findings": [{
            "code": "MISSING_MATERIAL_DENSITIES",
            "severity": "error",
            "category": "source_coverage",
            "message": "Required material density 8.19 is only recorded in source_notes, not dedicated structured fields.",
            "evidence_hashes": [evidence],
            "affected_json_paths": ["/source_notes"],
            "repairable_by_llm": True,
            "requires_human": False,
            "confidence": 0.9,
        }],
    }

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=lambda _: json.dumps(payload),
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert not result.coverage_complete
    assert [finding.code for finding in result.findings] == ["MISSING_MATERIAL_DENSITIES"]


def test_reviewer_downgrades_schema_unrepresentable_downstream_detail_gaps() -> None:
    policy = PlanClosedLoopPolicy()
    facts = {
        "patch_type": "facts",
        "model_scope": "single_assembly",
        "assembly_count": 1,
        "fuel_variant_requirements": [{"variant_id": "fuel"}],
        "has_spacer_grids": True,
        "has_axial_geometry": True,
        "active_fuel_region_cm": [11.951, 377.711],
        "axial_domain_cm": [-55.0, 463.937],
        "missing_facts": [],
        "assumptions": [],
        "source_notes": [],
    }
    packs = build_facts_evidence_packs(
        requirement_text="Source includes pin radii, axial layer z ranges, and spacer grid geometry.",
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_evidence_hashes": [evidence],
        "coverage_summary": {},
        "findings": [
            {
                "code": "MISSING_STANDARD_PIN_RADII",
                "severity": "error",
                "category": "source_coverage",
                "message": "The facts patch omits standard pin radii such as pellet radius and clad outer radius.",
                "evidence_hashes": [evidence],
                "affected_json_paths": ["/"],
                "repairable_by_llm": True,
                "requires_human": False,
                "confidence": 0.9,
            },
            {
                "code": "MISSING_AXIAL_LAYERS",
                "severity": "error",
                "category": "source_coverage",
                "message": "has_axial_geometry is true and active_fuel_region_cm is present, but detailed axial layer z-ranges for nozzles, core plates, plenums, and spacer grids are omitted.",
                "evidence_hashes": [evidence],
                "affected_json_paths": ["/has_axial_geometry", "/active_fuel_region_cm", "/axial_domain_cm"],
                "repairable_by_llm": True,
                "requires_human": False,
                "confidence": 0.9,
            },
            {
                "code": "FACTS_PATCH_INCOMPLETE",
                "severity": "error",
                "category": "source_coverage",
                "message": "The missing_facts array is empty, but downstream-critical details such as axial z-ranges, standard pin radii, and spacer grid properties are omitted.",
                "evidence_hashes": [evidence],
                "affected_json_paths": ["/missing_facts"],
                "repairable_by_llm": True,
                "requires_human": False,
                "confidence": 0.9,
            },
        ],
    }

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=lambda _: json.dumps(payload),
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert result.coverage_complete
    assert {finding.code for finding in result.findings} == {
        "MISSING_STANDARD_PIN_RADII",
        "MISSING_AXIAL_LAYERS",
        "FACTS_PATCH_INCOMPLETE",
    }
    assert all(finding.severity.value == "warning" for finding in result.findings)
    assert all(finding.repairable_by_llm is False for finding in result.findings)
    assert {
        finding.metadata["classification_override"]["reason"]
        for finding in result.findings
    } == {"facts_downstream_detail_scope"}


def test_reviewer_keeps_hard_facts_contract_gap_blocking() -> None:
    policy = PlanClosedLoopPolicy()
    facts = {
        "patch_type": "facts",
        "model_scope": "unknown",
        "missing_facts": [],
        "assumptions": [],
        "source_notes": [],
    }
    packs = build_facts_evidence_packs(
        requirement_text="Source declares a multi-assembly benchmark with spacer grids.",
        facts_patch=facts,
        confirmed_facts={},
        planning_metadata={},
        policy=policy,
    )
    evidence = packs[0].source_excerpts[0].evidence_hash
    payload = {
        "review_status": "complete_with_gaps",
        "reviewed_evidence_hashes": [evidence],
        "coverage_summary": {},
        "findings": [{
            "code": "FACTS_PATCH_INCOMPLETE",
            "severity": "error",
            "category": "source_coverage",
            "message": "The facts patch is incomplete because it omits model_scope and assembly_count.",
            "evidence_hashes": [evidence],
            "affected_json_paths": ["/model_scope", "/assembly_count"],
            "repairable_by_llm": True,
            "requires_human": False,
            "confidence": 0.9,
        }],
    }

    result = run_facts_review(
        evidence_packs=packs,
        reviewer_client=lambda _: json.dumps(payload),
        state=PlanBuildState(state_id="s", requirement_text="r"),
        policy=policy,
    )

    assert result.ok
    assert not result.coverage_complete
    assert result.findings[0].severity.value == "error"
