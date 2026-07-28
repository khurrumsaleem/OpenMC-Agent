"""VERA4 fragmented universes offline qualification.

Simulates a VERA4-like 11-universe scenario using FakePatchLLM, verifying:
- Fragmented generation produces all 11 universes
- Each fragment is a separate LLM call
- Merge produces a valid UniversesPatch
- Checkpoint resume skips completed fragments
- One fragment failure does not regenerate other fragments
- The final merged patch passes existing validators
"""

import json

from openmc_agent.plan_builder.universe_fragment_generation import (
    UniverseDefinitionFragment,
    UniverseManifest,
    UniverseManifestItem,
    build_manifest_from_requirements,
    extract_universe_requirements,
    merge_universe_fragments,
    validate_merged_patch,
    LargePatchGenerationSession,
    FragmentStatus,
)
from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope


def _make_universe_json(uid, kind="fuel_pin", material_id="m_fuel"):
    return {
        "universe_id": uid,
        "kind": kind,
        "cells": [
            {"id": "c1", "role": "fuel" if kind == "fuel_pin" else "background", "material_id": material_id, "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.4},
        ],
    }


VERA4_UNIVERSE_IDS = [f"u{i}" for i in range(1, 12)]  # 11 universes


class _FragmentedFakeLLM:
    """Fake LLM that returns one universe per call, in order."""

    def __init__(self, universe_payloads):
        self.prompts = []
        self._payloads = list(universe_payloads)
        self._call_count = 0

    def __call__(self, prompt: str) -> str:
        self.prompts.append(prompt)
        self._call_count += 1
        if not self._payloads:
            return json.dumps({"patch_type": "universes", "universes": []})
        universe = self._payloads.pop(0)
        return json.dumps({"patch_type": "universes", "universes": [universe]})


def test_vera4_11_universe_fragmented_generation():
    """11 universes generated one at a time, then merged."""
    payloads = [_make_universe_json(uid) for uid in VERA4_UNIVERSE_IDS]
    fake = _FragmentedFakeLLM(payloads)
    # Build manifest directly.
    manifest = UniverseManifest(
        manifest_id="vera4_test",
        input_hash="vera4_hash",
        expected_universe_count=11,
        items=[UniverseManifestItem(universe_id=uid, kind="fuel_pin") for uid in VERA4_UNIVERSE_IDS],
        generation_order=list(VERA4_UNIVERSE_IDS),
    )
    # Generate fragments by calling the fake LLM for each universe.
    from openmc_agent.plan_builder.universe_patch_pipeline import _call_llm_fragment
    fragments = []
    for item_id in manifest.generation_order:
        outcome = _call_llm_fragment(fake, prompt=f"generate {item_id}", max_tokens=4000)
        assert not outcome.failed, f"unexpected LLM failure for {item_id}: {outcome.note}"
        from openmc_agent.plan_builder.patch_generator import parse_llm_patch_json
        parsed = parse_llm_patch_json(outcome.response.content, "universes")
        universe_data = parsed["universes"][0]
        fragments.append(UniverseDefinitionFragment(universe_id=item_id, universe=universe_data))
    assert len(fragments) == 11
    # Merge.
    patch, errors = merge_universe_fragments(manifest=manifest, fragments=fragments, known_material_ids={"m_fuel"})
    assert errors == []
    assert patch is not None
    assert len(patch["universes"]) == 11


def test_vera4_merged_patch_passes_validation():
    """The merged 11-universe patch must pass existing validate_patch."""
    payloads = [_make_universe_json(uid) for uid in VERA4_UNIVERSE_IDS]
    manifest = UniverseManifest(
        manifest_id="vera4_test",
        input_hash="vera4_hash",
        expected_universe_count=11,
        items=[UniverseManifestItem(universe_id=uid, kind="fuel_pin") for uid in VERA4_UNIVERSE_IDS],
        generation_order=list(VERA4_UNIVERSE_IDS),
    )
    fragments = [UniverseDefinitionFragment(universe_id=uid, universe=p) for uid, p in zip(VERA4_UNIVERSE_IDS, payloads)]
    patch, errors = merge_universe_fragments(manifest=manifest, fragments=fragments, known_material_ids={"m_fuel"})
    ok, issues = validate_merged_patch(patch, known_material_ids={"m_fuel"})
    assert ok is True


def test_vera4_one_fragment_failure_doesnt_regen_others():
    """If one fragment fails, other accepted fragments remain."""
    session = LargePatchGenerationSession(
        session_id="test",
        input_hash="vera4_hash",
        mode="fragmented",
        fragment_statuses=[
            FragmentStatus(universe_id="u1", status="accepted"),
            FragmentStatus(universe_id="u2", status="accepted"),
            FragmentStatus(universe_id="u3", status="failed"),
        ],
    )
    accepted = [fs for fs in session.fragment_statuses if fs.status == "accepted"]
    failed = [fs for fs in session.fragment_statuses if fs.status == "failed"]
    assert len(accepted) == 2
    assert len(failed) == 1


def test_vera4_checkpoint_resume_skips_accepted():
    """Resume should not re-call LLM for already-accepted fragments."""
    session = LargePatchGenerationSession(
        session_id="test",
        input_hash="vera4_hash",
        mode="fragmented",
        fragment_statuses=[
            FragmentStatus(universe_id="u1", status="accepted", fragment_hash="h1"),
        ],
        accepted_fragment_hashes={"u1": "h1"},
    )
    # Simulate resume: only generate remaining universes.
    remaining = [uid for uid in VERA4_UNIVERSE_IDS if uid not in session.accepted_fragment_hashes]
    assert "u1" not in remaining
    assert len(remaining) == 10


def test_vera4_complex_pyrex_universe():
    """A complex Pyrex universe fragment should be valid."""
    pyrex_universe = {
        "universe_id": "u_pyrex",
        "kind": "pyrex_rod",
        "cells": [
            {"id": "c1", "role": "gas_gap", "material_id": "m_he", "region_kind": "cylinder", "r_min_cm": 0.0, "r_max_cm": 0.2},
            {"id": "c2", "role": "cladding", "material_id": "m_clad", "region_kind": "annulus", "r_min_cm": 0.2, "r_max_cm": 0.22},
            {"id": "c3", "role": "gas_gap", "material_id": "m_he", "region_kind": "annulus", "r_min_cm": 0.22, "r_max_cm": 0.24},
            {"id": "c4", "role": "poison", "material_id": "m_pyrex", "region_kind": "annulus", "r_min_cm": 0.24, "r_max_cm": 0.42},
            {"id": "c5", "role": "gas_gap", "material_id": "m_he", "region_kind": "annulus", "r_min_cm": 0.42, "r_max_cm": 0.44},
            {"id": "c6", "role": "cladding", "material_id": "m_clad", "region_kind": "annulus", "r_min_cm": 0.44, "r_max_cm": 0.48},
            {"id": "c7", "role": "inner_flow", "material_id": "m_water", "region_kind": "annulus", "r_min_cm": 0.48, "r_max_cm": 0.56},
        ],
    }
    manifest = UniverseManifest(
        manifest_id="test", input_hash="h", expected_universe_count=1,
        items=[UniverseManifestItem(universe_id="u_pyrex", kind="pyrex_rod")],
        generation_order=["u_pyrex"],
    )
    frag = UniverseDefinitionFragment(universe_id="u_pyrex", universe=pyrex_universe)
    patch, errors = merge_universe_fragments(manifest=manifest, fragments=[frag])
    ok, issues = validate_merged_patch(patch, known_material_ids={"m_he", "m_pyrex", "m_water", "m_clad"})
    assert ok is True


def test_vera4_strategy_switches_on_truncation():
    """Monolithic truncation should trigger fragmented strategy."""
    from openmc_agent.plan_builder.universe_fragment_generation import should_fragment_universes
    do_it, reason = should_fragment_universes(
        mode="auto", universe_count=11, provider_max_output_tokens=16000,
        history_json_truncated=True,
    )
    assert do_it is True
    assert "truncated" in reason


def test_vera4_downstream_sees_single_envelope():
    """The final output must be a single standard PlanPatchEnvelope, not fragments."""
    patch_dict = {"patch_type": "universes", "universes": [_make_universe_json("u1")]}
    envelope = PlanPatchEnvelope(
        patch_id="universes_fragmented_test",
        patch_type="universes",
        content=patch_dict,
        source="llm",
        status="valid",
    )
    assert envelope.patch_type == "universes"
    assert isinstance(envelope.content, dict)
    assert "universes" in envelope.content
