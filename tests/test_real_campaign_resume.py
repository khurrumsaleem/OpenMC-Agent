"""Real campaign resume fingerprint tests."""

from openmc_agent.real_campaign_harness import (
    CampaignResumeFingerprint,
    RealCampaignCaseSpec,
    compute_resume_fingerprint,
    detect_provider_environment,
    make_five_gate_controlled_policy,
    _resumable_campaign_results,
)


def _fingerprint(**overrides) -> CampaignResumeFingerprint:
    base = dict(
        git_sha="g1", input_sha="i1", requirement_sha="r1",
        human_answer_sha="h1", model="ds:test", provider="ds",
        reasoning_effort="default", output_mode="auto",
        plan_policy_hash="p1",
        enabled_gates=("facts", "material_universe", "placement", "axial_geometry", "assembled_plan"),
        review_modes=("controlled", "controlled", "controlled", "controlled"),
        universes_generation_mode="auto",
        universe_fragment_max_tokens=None,
        large_patch_safe_output_ratio=0.6,
        strict_structured_patch_output=True,
        material_policy="strict",
        runtime_mode="deterministic",
        openmc_cross_sections_fingerprint="xs1",
    )
    base.update(overrides)
    return CampaignResumeFingerprint(**base)


def test_identical_fingerprints_have_no_mismatches():
    a = _fingerprint()
    b = _fingerprint()
    assert a.mismatches_against(b) == []


def test_git_sha_mismatch_is_flagged():
    a = _fingerprint()
    b = _fingerprint(git_sha="g2")
    assert "git_sha" in a.mismatches_against(b)


def test_input_sha_mismatch_is_flagged():
    a = _fingerprint()
    b = _fingerprint(input_sha="i2")
    assert "input_sha" in a.mismatches_against(b)


def test_model_mismatch_is_flagged():
    a = _fingerprint()
    b = _fingerprint(model="ds:other")
    assert "model" in a.mismatches_against(b)


def test_universes_mode_mismatch_is_flagged():
    a = _fingerprint(universes_generation_mode="auto")
    b = _fingerprint(universes_generation_mode="fragmented")
    assert "universes_generation_mode" in a.mismatches_against(b)


def test_strict_output_mismatch_is_flagged():
    a = _fingerprint(strict_structured_patch_output=True)
    b = _fingerprint(strict_structured_patch_output=False)
    assert "strict_structured_patch_output" in a.mismatches_against(b)


def test_openmc_cross_sections_mismatch_is_flagged():
    a = _fingerprint(openmc_cross_sections_fingerprint="xs1")
    b = _fingerprint(openmc_cross_sections_fingerprint="xs2")
    assert "openmc_cross_sections_fingerprint" in a.mismatches_against(b)


def test_human_answer_hash_mismatch_is_flagged():
    a = _fingerprint(human_answer_sha="h1")
    b = _fingerprint(human_answer_sha="h2")
    assert "human_answer_sha" in a.mismatches_against(b)


def test_enabled_gates_mismatch_is_flagged():
    a = _fingerprint(enabled_gates=("facts",))
    b = _fingerprint(enabled_gates=("facts", "material_universe"))
    assert "enabled_gates" in a.mismatches_against(b)


def test_compute_resume_fingerprint_uses_case_and_policy():
    case = RealCampaignCaseSpec(
        case_id="vera4", input_path="/tmp/in.md",
        operating_state="", benchmark_label="VERA4",
        model="ds:test", output_dir="/tmp/out",
    )
    env_status = detect_provider_environment("ds:test")
    policy = make_five_gate_controlled_policy()
    fp = compute_resume_fingerprint(
        case=case, env_status=env_status, policy=policy,
        universes_generation_mode="fragmented",
        universe_fragment_max_tokens=4000,
        large_patch_safe_output_ratio=0.6,
        strict_structured_patch_output=True,
        material_policy="strict",
        runtime_mode="deterministic",
        reasoning_effort="default",
        output_mode="auto",
        input_sha="abc", requirement_sha="def",
        human_answer_sha="", git_sha="g1",
    )
    assert fp.model == "ds:test"
    assert fp.provider == "ds"
    assert fp.universes_generation_mode == "fragmented"
    assert fp.universe_fragment_max_tokens == 4000


def test_resume_filters_failed_run_records_so_they_rerun():
    results = [
        {"run_id": "run_001", "final_disposition": "PLANNING_FAILURE"},
        {"run_id": "run_002", "final_disposition": "BLOCKED_BY_GATE:placement"},
        {"run_id": "run_003", "final_disposition": "STOP_AFTER_GATE_PASSED:placement"},
    ]
    resumable = _resumable_campaign_results(results)
    assert [item["run_id"] for item in resumable] == ["run_003"]
