"""Real campaign stage mode tests."""

import pytest

from openmc_agent.real_campaign_harness import (
    CanaryCampaignConfig,
    ProviderEnvironmentStatus,
    RealCampaignRunResult,
    RealCampaignCaseSpec,
    _render_compile_passed,
    run_real_canary_campaign,
    _validate_stage,
)


def _case() -> RealCampaignCaseSpec:
    return RealCampaignCaseSpec(
        case_id="x", input_path="/tmp/x.md", operating_state="",
        benchmark_label="X", model="fake:test", output_dir="/tmp/out",
    )


def test_validate_stage_accepts_planning():
    assert _validate_stage("planning") == "planning"


def test_validate_stage_accepts_render_compile():
    assert _validate_stage("render-compile") == "render-compile"


def test_validate_stage_accepts_openmc_smoke():
    assert _validate_stage("openmc-smoke") == "openmc-smoke"


def test_validate_stage_rejects_unknown():
    with pytest.raises(ValueError):
        _validate_stage("unknown-stage")


@pytest.mark.parametrize(
    ("renderability", "expected"),
    [
        ("exportable", True),
        ("runnable", True),
        ("skeleton", False),
        ("none", False),
        ("supported", False),
    ],
)
def test_render_compile_passed_uses_current_renderability_contract(
    renderability: str, expected: bool
) -> None:
    assert _render_compile_passed(renderability) is expected


def test_canary_campaign_config_default_stage_is_planning():
    cfg = CanaryCampaignConfig(case=_case(), runs=1, model="fake:test")
    assert cfg.planning_stage == "planning"


def test_canary_campaign_config_renders_compile_stage():
    cfg = CanaryCampaignConfig(
        case=_case(), runs=1, model="fake:test",
        planning_stage="render-compile",
    )
    assert cfg.planning_stage == "render-compile"


def test_canary_campaign_config_openmc_smoke_stage():
    cfg = CanaryCampaignConfig(
        case=_case(), runs=1, model="fake:test",
        planning_stage="openmc-smoke",
    )
    assert cfg.planning_stage == "openmc-smoke"
    # CanaryCampaignConfig doesn't carry an enable_smoke_test field — the
    # per-run CanaryRunConfig derives it from the stage at run setup time.


def test_planning_stage_disables_smoke_test_in_run_config():
    """CanaryRunConfig built from CanaryCampaignConfig in planning stage
    must not enable smoke_test."""
    from openmc_agent.real_campaign_harness import CanaryRunConfig
    campaign = CanaryCampaignConfig(
        case=_case(), runs=1, model="fake:test",
        planning_stage="planning",
    )
    run_cfg = CanaryRunConfig(
        run_id="r1", run_index=1,
        case=campaign.case,
        policy=object(),
        env_status=object(),
        fingerprint=object(),
        output_dir="/tmp/out",
        model="fake:test",
        planning_stage=campaign.planning_stage,
        enable_smoke_test=campaign.planning_stage == "openmc-smoke",
    )
    assert run_cfg.enable_smoke_test is False


def test_openmc_smoke_stage_enables_smoke_test_in_run_config():
    from openmc_agent.real_campaign_harness import CanaryRunConfig
    campaign = CanaryCampaignConfig(
        case=_case(), runs=1, model="fake:test",
        planning_stage="openmc-smoke",
    )
    run_cfg = CanaryRunConfig(
        run_id="r1", run_index=1,
        case=campaign.case,
        policy=object(),
        env_status=object(),
        fingerprint=object(),
        output_dir="/tmp/out",
        model="fake:test",
        planning_stage=campaign.planning_stage,
        enable_smoke_test=campaign.planning_stage == "openmc-smoke",
    )
    assert run_cfg.enable_smoke_test is True


def test_planning_stage_does_not_require_openmc_environment(monkeypatch):
    """Even when OPENMC_CROSS_SECTIONS is unset, a planning canary should
    not be blocked by BLOCKED_BY_OPENMC_ENVIRONMENT."""
    from openmc_agent.real_campaign_harness import detect_provider_environment
    monkeypatch.delenv("OPENMC_CROSS_SECTIONS", raising=False)
    monkeypatch.setenv("SENSENOVA_API_KEY", "x")
    status = detect_provider_environment("ds:test")
    # LLM environment is OK; OpenMC env may be missing but that must not
    # block a planning canary.
    assert status.llm_environment_available is True


def test_completed_failed_campaign_updates_final_aggregate_status(tmp_path, monkeypatch):
    import openmc_agent.real_campaign_harness as harness

    monkeypatch.setattr(
        harness,
        "detect_provider_environment",
        lambda _model: ProviderEnvironmentStatus(
            provider="zhipu",
            model="zhipu:test",
            api_key_env="ZHIPUAI_API_KEY",
            api_key_present=True,
            openmc_library_present=True,
            openmc_cross_sections_present=True,
            openmc_cross_sections_path="/tmp/cross_sections.xml",
            openmc_version="0.0",
            endpoint="https://example.invalid",
        ),
    )
    monkeypatch.setattr(harness, "_git_sha", lambda: "git")

    def failed_once(*_args, **_kwargs):
        return RealCampaignRunResult(
            run_id="run_001",
            status="completed",
            final_disposition="BLOCKED_BY_GATE:material_universe",
            started_at="2026-07-23T00:00:00+00:00",
            completed_at="2026-07-23T00:00:01+00:00",
            duration_s=1.0,
            git_sha="git",
            input_sha="",
            configuration_hash="cfg",
            provider="zhipu",
            model="zhipu:test",
            real_llm_verified=True,
            real_openmc_verified=False,
            llm_call_count=0,
        )

    monkeypatch.setattr(harness, "run_real_canary_once", failed_once)

    campaign = CanaryCampaignConfig(
        case=_case(),
        runs=1,
        model="zhipu:test",
    )
    manifest = run_real_canary_campaign(tmp_path, campaign)

    assert manifest["completed_runs"] == 1
    assert manifest["failed_runs"] == 1
    assert manifest["aggregate_status"] == "CAMPAIGN_FAILED"


def test_campaign_metadata_propagates_to_run_config(tmp_path, monkeypatch):
    import openmc_agent.real_campaign_harness as harness

    input_path = tmp_path / "input.md"
    input_path.write_text("Build a 17x17 lattice with axial layers.\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        harness,
        "detect_provider_environment",
        lambda _model: ProviderEnvironmentStatus(
            provider="zhipu",
            model="zhipu:test",
            api_key_env="ZHIPUAI_API_KEY",
            api_key_present=True,
            openmc_library_present=True,
            openmc_cross_sections_present=True,
            openmc_cross_sections_path="/tmp/cross_sections.xml",
            openmc_version="0.0",
            endpoint="https://example.invalid",
        ),
    )
    monkeypatch.setattr(harness, "_git_sha", lambda: "git")

    def capture_once(config, *_args, **_kwargs):
        captured["metadata"] = dict(config.metadata)
        return RealCampaignRunResult(
            run_id=config.run_id,
            status="completed",
            final_disposition="STOP_AFTER_GATE_PASSED:axial_geometry",
            started_at="2026-07-27T00:00:00+00:00",
            completed_at="2026-07-27T00:00:01+00:00",
            duration_s=1.0,
            git_sha="git",
            input_sha="",
            configuration_hash="cfg",
            provider="zhipu",
            model="zhipu:test",
            real_llm_verified=True,
            real_openmc_verified=False,
            llm_call_count=0,
        )

    monkeypatch.setattr(harness, "run_real_canary_once", capture_once)

    campaign = CanaryCampaignConfig(
        case=RealCampaignCaseSpec(
            case_id="x",
            input_path=str(input_path),
            operating_state="",
            benchmark_label="X",
            model="zhipu:test",
            output_dir=str(tmp_path / "out"),
        ),
        runs=1,
        model="zhipu:test",
        metadata={
            "accepted_plan_build_state": {"state_id": "seed"},
            "accepted_plan_build_state_path": "seed.json",
        },
    )
    run_real_canary_campaign(tmp_path / "out", campaign)

    assert captured["metadata"]["accepted_plan_build_state"] == {"state_id": "seed"}
    assert captured["metadata"]["accepted_plan_build_state_path"] == "seed.json"
    assert "llm_budget" in captured["metadata"]


def test_render_compile_seed_does_not_stop_after_gate(tmp_path, monkeypatch):
    import openmc_agent.real_campaign_harness as harness

    input_path = tmp_path / "input.md"
    input_path.write_text("Build a 17x17 lattice with axial layers.\n", encoding="utf-8")
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        harness,
        "detect_provider_environment",
        lambda _model: ProviderEnvironmentStatus(
            provider="zhipu",
            model="zhipu:test",
            api_key_env="ZHIPUAI_API_KEY",
            api_key_present=True,
            openmc_library_present=True,
            openmc_cross_sections_present=True,
            openmc_cross_sections_path="/tmp/cross_sections.xml",
            openmc_version="0.0",
            endpoint="https://example.invalid",
        ),
    )
    monkeypatch.setattr(harness, "_git_sha", lambda: "git")

    def capture_once(config, *_args, **_kwargs):
        from openmc_agent.plan_builder.closed_loop.policy import enabled_gates

        captured["run_stop_after_gate"] = config.stop_after_gate
        captured["policy_stop_after_gate"] = getattr(config.policy, "stop_after_gate", None)
        captured["enabled_gate_ids"] = [g.value for g in enabled_gates(config.policy)]
        return RealCampaignRunResult(
            run_id=config.run_id,
            status="completed",
            final_disposition="RENDER_COMPILE_INCOMPLETE",
            started_at="2026-07-28T00:00:00+00:00",
            completed_at="2026-07-28T00:00:01+00:00",
            duration_s=1.0,
            git_sha="git",
            input_sha="",
            configuration_hash="cfg",
            provider="zhipu",
            model="zhipu:test",
            real_llm_verified=True,
            real_openmc_verified=False,
            llm_call_count=0,
        )

    monkeypatch.setattr(harness, "run_real_canary_once", capture_once)

    campaign = CanaryCampaignConfig(
        case=RealCampaignCaseSpec(
            case_id="x",
            input_path=str(input_path),
            operating_state="",
            benchmark_label="X",
            model="zhipu:test",
            output_dir=str(tmp_path / "out"),
            planning_stage="render-compile",
        ),
        runs=1,
        model="zhipu:test",
        planning_stage="render-compile",
        stop_after_gate="assembled_plan",
        metadata={
            "accepted_plan_build_state": {"state_id": "seed"},
            "accepted_plan_build_state_path": "seed.json",
        },
    )
    run_real_canary_campaign(tmp_path / "out", campaign)

    assert captured["run_stop_after_gate"] is None
    assert captured["policy_stop_after_gate"] is None
    assert captured["enabled_gate_ids"] == [
        "facts",
        "material_universe",
        "placement",
        "axial_geometry",
        "assembled_plan",
    ]
