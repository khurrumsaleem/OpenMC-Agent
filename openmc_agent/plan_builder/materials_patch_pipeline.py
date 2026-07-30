"""Materials patch pipeline — fragmented generation with checkpoint/resume.

This mirrors the proven Universe fragment transaction (Step 4B-1) for the
Materials patch type.  When materials are complex (many requirements,
compound compositions, multiple fuel variants), the monolithic generator
truncates.  This pipeline:

1. Tries monolithic first (when mode=auto and within budget).
2. On truncation, switches to fragmented mode — one material per LLM call.
3. Each fragment is deterministically qualified before acceptance.
4. Accepted fragments are checkpointed; resume only regenerates corrupted ones.
5. Pure-Python merge assembles all accepted materials into a MaterialsPatch.
6. The merged patch is validated by the standard MaterialsPatch validator.

Wiring
------
The executor dispatcher (executor.py ~line 3994) routes ``materials`` patch
type to this pipeline when ``materials_generation_mode`` is not ``"off"``.
"""

from __future__ import annotations

import json
from typing import Any

from openmc_agent.plan_builder.state import PlanBuildState, PlanPatchEnvelope
from openmc_agent.plan_builder.patch_generator import PatchGenerationResult
from openmc_agent.plan_builder.llm_adapter import (
    PatchLLMResponse,
    normalize_patch_llm_response,
)
from openmc_agent.plan_builder.patches import get_patch_json_schema

from .materials_fragment_generation import (
    AcceptedMaterialFragmentRecord,
    MaterialDefinitionFragment,
    MaterialFragmentQualificationResult,
    MaterialFragmentStatus,
    MaterialManifest,
    MaterialManifestItem,
    MaterialsPatchGenerationSession,
    MaterialMergeResult,
    build_material_manifest,
    compute_manifest_item_contract_hash,
    merge_material_fragments_structured,
    qualify_material_fragment,
    should_fragment_materials,
    validate_material_manifest,
    verify_accepted_material_fragment,
)
from .material_requirements import (
    MaterialGenerationRequirementSet,
    extract_material_requirements_from_facts,
)
from .materials_patch_normalization import normalize_materials_patch_content


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def _get_session(state: PlanBuildState, input_hash: str) -> MaterialsPatchGenerationSession | None:
    sessions = state.metadata.get("large_patch_generation_sessions", {})
    if not isinstance(sessions, dict):
        return None
    session_data = sessions.get(f"materials:{input_hash}")
    if session_data is None:
        return None
    if isinstance(session_data, dict):
        return MaterialsPatchGenerationSession.model_validate(session_data)
    return session_data


def _save_session(state: PlanBuildState, session: MaterialsPatchGenerationSession) -> None:
    sessions = state.metadata.setdefault("large_patch_generation_sessions", {})
    sessions[f"materials:{session.input_hash}"] = session.model_dump(mode="json")


def _stale_session(state: PlanBuildState, input_hash: str) -> None:
    sessions = state.metadata.get("large_patch_generation_sessions", {})
    if isinstance(sessions, dict):
        for key in list(sessions.keys()):
            if key.startswith("materials:") and key != f"materials:{input_hash}":
                old = sessions[key]
                if isinstance(old, dict):
                    old["completed"] = False
                    old.setdefault("metadata", {})["stale"] = True


# ---------------------------------------------------------------------------
# LLM call classification (no silent except-pass)
# ---------------------------------------------------------------------------

class _FragmentLLMOutcome:
    def __init__(
        self,
        *,
        response: PatchLLMResponse | None,
        outcome_kind: str,
        exception: BaseException | None = None,
        exception_class: str = "",
        note: str = "",
    ) -> None:
        self.response = response
        self.outcome_kind = outcome_kind
        self.exception = exception
        self.exception_class = exception_class
        self.note = note

    @property
    def failed(self) -> bool:
        return self.outcome_kind != "ok" or self.response is None


def _call_llm_fragment(
    llm_client: Any,
    *,
    prompt: str,
    max_tokens: int | None = None,
) -> _FragmentLLMOutcome:
    schema = get_patch_json_schema("materials")

    if hasattr(llm_client, "generate_patch_json_with_meta"):
        try:
            resp = llm_client.generate_patch_json_with_meta(
                prompt=prompt, patch_type="materials",
                json_schema=schema, max_tokens=max_tokens,
            )
            if isinstance(resp, PatchLLMResponse):
                if not resp.content:
                    return _FragmentLLMOutcome(response=resp, outcome_kind="empty", note="empty content")
                return _FragmentLLMOutcome(response=resp, outcome_kind="ok")
            return _FragmentLLMOutcome(response=normalize_patch_llm_response(resp), outcome_kind="ok")
        except Exception as exc:
            return _FragmentLLMOutcome(
                response=None, outcome_kind="provider_exception",
                exception=exc, exception_class=type(exc).__name__,
                note=f"generate_patch_json_with_meta raised {type(exc).__name__}",
            )

    if hasattr(llm_client, "generate_patch_json"):
        try:
            raw = llm_client.generate_patch_json(
                prompt=prompt, patch_type="materials",
                json_schema=schema, max_tokens=max_tokens,
            )
            resp = PatchLLMResponse(
                content=raw or "",
                output_mode_used=str(getattr(llm_client, "last_output_mode_used", "structured")),
            )
            if not resp.content:
                return _FragmentLLMOutcome(response=resp, outcome_kind="empty", note="empty content")
            return _FragmentLLMOutcome(response=resp, outcome_kind="ok")
        except Exception as exc:
            return _FragmentLLMOutcome(
                response=None, outcome_kind="provider_exception",
                exception=exc, exception_class=type(exc).__name__,
                note=f"generate_patch_json raised {type(exc).__name__}",
            )

    try:
        raw = llm_client(prompt)
    except Exception as exc:
        return _FragmentLLMOutcome(
            response=None, outcome_kind="provider_exception",
            exception=exc, exception_class=type(exc).__name__,
            note=f"plain callable raised {type(exc).__name__}",
        )
    resp = normalize_patch_llm_response(raw)
    if not resp.content:
        return _FragmentLLMOutcome(response=resp, outcome_kind="empty", note="empty content")
    return _FragmentLLMOutcome(response=resp, outcome_kind="ok")


# ---------------------------------------------------------------------------
# Fragment prompt builder
# ---------------------------------------------------------------------------

def _build_material_fragment_prompt(
    item: MaterialManifestItem,
    *,
    requirement: str,
    prior_failures: list[str] | None = None,
) -> str:
    lines = [
        f"Generate ONLY the material definition for material_id={item.material_id}.",
        f"Role: {item.role}",
    ]
    if item.preferred_name:
        lines.append(f"Preferred name: {item.preferred_name}")
    if item.source_variant_id:
        lines.append(f"Fuel variant: {item.source_variant_id} (set source_variant_id to this value)")
    if item.localized_insert_requirement_id:
        lines.append(f"Localized insert requirement: {item.localized_insert_requirement_id}")
    if item.density_required:
        lines.append("Density is required — provide density_g_cm3 and density_status.")
    if item.composition_required:
        lines.append("Composition is required — provide transport-ready composition dict.")
    if item.mixture_required:
        lines.append(f"This is a mixture material. Reference component material_ids: {', '.join(item.mixture_component_ids)}")
        lines.append("Use mixture_components list (not composition dict).")
    lines.append(f"\nSource context:\n{requirement[:2000]}")
    lines.append(
        "\nOutput a single JSON object with keys: patch_type, materials (list with ONE material)."
    )
    lines.append("The material must have: material_id, name, role, density_g_cm3, density_status, composition, composition_basis, composition_status.")
    lines.append("Enum constraints (use EXACTLY one of the listed values):")
    lines.append("- density_status: confirmed | source_provided | library | approximate | needs_confirmation")
    lines.append("- composition_basis: atom_frac | weight_frac | atom_density_barn_cm | stoichiometric_ratio | ppm_by_weight | ppm_by_atom | unknown")
    lines.append("- composition_status: confirmed | approximate | needs_library | needs_confirmation | placeholder | derived_from_mixture")
    lines.append("IMPORTANT rules:")
    lines.append("- Do NOT use placeholder values (REPLACE, TBD, unknown) for material_id or composition.")
    lines.append("- Do NOT merge fuel variants into one material.")
    lines.append("- composition keys must be transport-ready element/nuclide names (e.g., U235, O16, Zr).")
    lines.append("- Chemical formulae (UO2, B4C, H2O) go in compound_components, NOT composition.")
    lines.append("- Set source_variant_id to match the fuel variant if this is a fuel material.")
    if prior_failures:
        lines.append(f"\nPrior failures to avoid:\n{chr(10).join(prior_failures[-3:])}")
    lines.append("\n```json")
    lines.append(json.dumps({
        "patch_type": "materials",
        "materials": [{
            "material_id": item.material_id,
            "name": "<descriptive_name>",
            "role": item.role,
            "density_g_cm3": 0.0,
            "density_status": "confirmed",
            "composition": {"<nuclide>": 0.0},
            "composition_basis": "atom_frac",
            "composition_status": "confirmed",
        }],
    }, indent=2))
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resume verification
# ---------------------------------------------------------------------------

def _verify_resume_fragments(
    *,
    session: MaterialsPatchGenerationSession,
    manifest: MaterialManifest,
) -> tuple[dict[str, AcceptedMaterialFragmentRecord], dict[str, MaterialFragmentQualificationResult]]:
    """Verify accepted fragments on resume; downgrade corrupted ones."""
    valid: dict[str, AcceptedMaterialFragmentRecord] = {}
    invalid: dict[str, MaterialFragmentQualificationResult] = {}
    all_ids = manifest.material_ids
    for mid, record in session.accepted_fragments.items():
        item = manifest.item_by_id(mid)
        if item is None:
            continue
        result = verify_accepted_material_fragment(record, item, all_ids)
        if result.ok:
            valid[mid] = record
        else:
            invalid[mid] = result
    return valid, invalid


def _downgrade_fragment(
    session: MaterialsPatchGenerationSession,
    mid: str,
    *,
    reason: str,
    qualification_result: MaterialFragmentQualificationResult | None = None,
) -> None:
    session.accepted_fragments.pop(mid, None)
    session.accepted_fragment_hashes.pop(mid, None)
    for fs in session.fragment_statuses:
        if fs.material_id == mid:
            fs.status = "pending"
            fs.fragment_hash = ""
            fs.qualification_status = "failed" if qualification_result and not qualification_result.ok else "pending"
            fs.metadata["downgrade_reason"] = reason
            break


# ---------------------------------------------------------------------------
# Single fragment generation + qualification
# ---------------------------------------------------------------------------

def _generate_and_qualify_one_fragment(
    *,
    item: MaterialManifestItem,
    llm_client: Any,
    session: MaterialsPatchGenerationSession,
    requirement: str,
    all_manifest_material_ids: set[str],
    effective_max_tokens: int,
    explicit_max_tokens: int | None,
    prior_failures: list[str],
    attempt_index: int,
) -> tuple[AcceptedMaterialFragmentRecord | None, MaterialFragmentQualificationResult | None, _FragmentLLMOutcome | None]:
    prompt = _build_material_fragment_prompt(item, requirement=requirement, prior_failures=prior_failures)
    max_tok = explicit_max_tokens or effective_max_tokens
    outcome = _call_llm_fragment(llm_client, prompt=prompt, max_tokens=max_tok)
    session.llm_call_count += 1

    if outcome.failed:
        prior_failures.append(f"llm_{outcome.outcome_kind}: {outcome.note}")
        return None, None, outcome

    raw_text = outcome.response.content if outcome.response else ""

    finish_reason = ""
    if outcome.response is not None:
        finish_reason = getattr(outcome.response, "finish_reason", "") or ""

    session.provider_telemetry.append({
        "material_id": item.material_id,
        "attempt": attempt_index,
        "finish_reason": finish_reason,
        "content_chars": len(raw_text),
        "output_mode": getattr(outcome.response, "output_mode_used", "") if outcome.response else "",
    })

    try:
        candidate = json.loads(raw_text)
    except Exception:
        from .patch_generator import parse_llm_patch_json
        try:
            candidate = parse_llm_patch_json(raw_text, "materials")
        except Exception:
            try:
                from .patch_generator import _looks_truncated
                if _looks_truncated(raw_text):
                    prior_failures.append("json_truncated")
                else:
                    prior_failures.append("json_parse_error")
            except Exception:
                prior_failures.append("json_parse_error")
            return None, None, outcome

    qualification = qualify_material_fragment(
        raw_fragment=candidate,
        manifest_item=item,
        all_manifest_material_ids=all_manifest_material_ids,
        attempt_index=attempt_index,
    )

    if not qualification.ok:
        for issue in qualification.issues:
            prior_failures.append(f"{issue['code']}: {issue.get('message', '')[:80]}")
        return None, qualification, outcome

    record = AcceptedMaterialFragmentRecord(
        material_id=item.material_id,
        material=qualification.canonical_material_data,
        fragment_hash=qualification.fragment_hash,
        manifest_contract_hash=qualification.manifest_contract_hash,
        qualification_status="passed",
        qualification_issues=qualification.issues,
        accepted_at_attempt=attempt_index,
    )
    return record, qualification, outcome


# ---------------------------------------------------------------------------
# Merged-patch validation
# ---------------------------------------------------------------------------

def _validate_merged_materials_patch(
    merged_patch: dict[str, Any],
) -> tuple[bool, list[dict[str, Any]]]:
    from .patches import MaterialsPatch, parse_patch_content, PatchParseError
    from .validators import validate_patch

    try:
        patch_obj = parse_patch_content("materials", merged_patch)
    except PatchParseError as exc:
        return False, [{"code": "patch_generation.merge_parse_failed", "message": str(exc)}]
    except Exception as exc:
        return False, [{"code": "patch_generation.merge_parse_failed", "message": f"{type(exc).__name__}: {exc}"}]
    val_result = validate_patch(patch_obj)
    if not val_result.ok:
        return False, [{"code": vi.code, "message": vi.message} for vi in val_result.issues]
    return True, []


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def generate_materials_patch(
    *,
    requirement: str,
    state: PlanBuildState,
    llm_client: Any,
    material_requirement_set: MaterialGenerationRequirementSet | None = None,
    mode: str = "auto",
    max_tokens: int | None = None,
    max_fragment_attempts: int = 2,
    max_total_llm_calls: int = 40,
    safe_output_ratio: float = 0.6,
    max_merge_replays: int = 2,
) -> PatchGenerationResult:
    """Generate a materials patch, using fragmentation when necessary.

    Transaction contract
    --------------------
    * A fragment only enters the accepted set after deterministic
      qualification against its manifest item.
    * Accepted fragments store data + canonical hash + manifest contract
      hash + qualification record; resume re-verifies all four.
    * Merge is pure Python and deterministic.
    * Fragment-scoped merge failures trigger a targeted replay.
    * Manifest / global failures fail closed.
    """
    from .patch_generator import generate_patch
    from .llm_adapter import LARGE_PATCH_MAX_TOKENS

    patch_type = "materials"
    effective_max = max_tokens or LARGE_PATCH_MAX_TOKENS.get(patch_type, 16000)

    # Resolve requirement set.
    if material_requirement_set is None:
        material_requirement_set = _load_requirement_set(state)
    if material_requirement_set is None or not material_requirement_set.requirements:
        # Fallback: derive from accepted Facts when the inventory-compilation
        # path did not populate a requirement set.  This mirrors how the
        # universes pipeline derives its requirements from facts and lets
        # fragmentation activate for large catalogs instead of monolithic
        # truncation.
        material_requirement_set = _derive_requirement_set_from_facts(state)
    if material_requirement_set is None or not material_requirement_set.requirements:
        # No requirements derivable → fall back to monolithic.
        return generate_patch(
            patch_type=patch_type, requirement=requirement, state=state,
            llm_client=llm_client, max_attempts=2, max_tokens=effective_max,
        )

    input_hash = material_requirement_set.requirement_set_hash

    # Checkpoint session.
    session = _get_session(state, input_hash)
    if session is None:
        _stale_session(state, input_hash)
        session = MaterialsPatchGenerationSession(
            session_id=f"session:{input_hash}",
            patch_type=patch_type,
            input_hash=input_hash,
            mode=mode,
            requirement_set_hash=input_hash,
        )

    # Strategy decision.
    history_truncated = any(
        "json_truncated" in str(a.get("issues", []))
        for a in getattr(state, "metadata", {}).get("patch_attempt_artifacts", {}).values()
        if isinstance(a, dict) and a.get("patch_type") == patch_type
    )

    material_count = len(material_requirement_set.requirements)
    # Strategy decision.  In ``auto`` mode we attempt the monolithic path first
    # (one LLM call) and only fall back to fragmentation on
    # truncation/parse/validation failure — this lets callers that feed a full
    # materials patch (tests, small catalogs) succeed, while large catalogs
    # that truncate still get fragmented.  Only explicit ``fragmented`` mode or
    # a known truncation history skips the monolithic attempt outright.
    do_fragment, fragment_reason = should_fragment_materials(
        mode=mode,
        material_count=material_count,
        provider_max_output_tokens=effective_max,
        history_json_truncated=history_truncated,
        safe_output_ratio=safe_output_ratio,
    )
    skip_monolithic = mode == "fragmented" or history_truncated
    do_fragment = skip_monolithic
    if skip_monolithic:
        session.strategy_transitions.append(
            {"from": "none", "to": "fragmented", "reason": fragment_reason}
        )

    if not skip_monolithic:
        session.strategy_transitions.append({"from": "none", "to": "monolithic", "reason": "auto_try_monolithic_first"})
        _save_session(state, session)
        result = generate_patch(
            patch_type=patch_type, requirement=requirement, state=state,
            llm_client=llm_client, max_attempts=2, max_tokens=effective_max,
        )
        if mode == "monolithic":
            session.completed = True
            _save_session(state, session)
            return result
        for attempt in result.attempts:
            if any(i.get("code") == "patch_generation.json_truncated" for i in attempt.issues):
                session.strategy_transitions.append({"from": "monolithic", "to": "fragmented", "reason": "json_truncated"})
                session.llm_call_count += len(result.attempts)
                do_fragment = True
                break
            if attempt.finish_reason and attempt.finish_reason.lower() in ("length", "max_tokens"):
                session.strategy_transitions.append({"from": "monolithic", "to": "fragmented", "reason": f"finish_reason={attempt.finish_reason}"})
                session.llm_call_count += len(result.attempts)
                do_fragment = True
                break
        # A monolithic result that parsed but failed validation (e.g. a
        # composition/contract error on a large catalog) should also fall back
        # to fragmentation rather than fail the whole gate.
        if not do_fragment and not result.ok:
            session.strategy_transitions.append({"from": "monolithic", "to": "fragmented", "reason": "monolithic_failed"})
            session.llm_call_count += len(result.attempts)
            do_fragment = True
        if not do_fragment:
            session.completed = True
            _save_session(state, session)
            return result

    # -----------------------------------------------------------------
    # Fragmented path.
    # -----------------------------------------------------------------
    session.mode = "fragmented"
    session.strategy_transitions.append({
        "from": "monolithic" if session.strategy_transitions else "none",
        "to": "fragmented",
        "reason": fragment_reason if not session.strategy_transitions else session.strategy_transitions[-1].get("reason", ""),
    })

    # Build or reuse manifest.
    if session.manifest is None:
        manifest = build_material_manifest(material_requirement_set)
        errors = validate_material_manifest(manifest, material_requirement_set)
        if errors:
            session.manifest_status = "failed"
            _save_session(state, session)
            return PatchGenerationResult(
                ok=False, patch_type=patch_type,
                issues=[{
                    "code": "patch_generation.manifest_validation_failed",
                    "severity": "error",
                    "message": f"manifest errors: {errors}",
                    "metadata": {"manifest_errors": errors},
                }],
            )
        session.manifest = manifest
        session.manifest_status = "accepted"
    else:
        manifest = session.manifest
        for item in manifest.items:
            if not item.contract_hash:
                item.recompute_contract_hash()

    all_manifest_ids = manifest.material_ids

    # Resume verification.
    valid_resume, invalid_resume = _verify_resume_fragments(session=session, manifest=manifest)
    for mid, result in invalid_resume.items():
        _downgrade_fragment(session, mid, reason=f"resume_verification_failed: {result.metadata.get('reason', 'qualification_failed')}", qualification_result=result)

    manifest_by_id = {item.material_id: item for item in manifest.items}

    def _fragment_status(mid: str) -> MaterialFragmentStatus:
        fs = next((fs for fs in session.fragment_statuses if fs.material_id == mid), None)
        if fs is None:
            fs = MaterialFragmentStatus(material_id=mid, status="pending")
            session.fragment_statuses.append(fs)
        return fs

    # -----------------------------------------------------------------
    # Generation phase: fill pending/failed fragments.
    # -----------------------------------------------------------------
    for mid in manifest.generation_order:
        fs = _fragment_status(mid)
        if fs.status == "accepted" and mid in valid_resume:
            continue
        if mid not in manifest_by_id:
            continue
        item = manifest_by_id[mid]
        prior_failures: list[str] = []

        record: AcceptedMaterialFragmentRecord | None = None
        for frag_attempt in range(max_fragment_attempts):
            if session.llm_call_count >= max_total_llm_calls:
                break
            record, qualification, outcome = _generate_and_qualify_one_fragment(
                item=item, llm_client=llm_client, session=session,
                requirement=requirement,
                all_manifest_material_ids=all_manifest_ids,
                effective_max_tokens=effective_max or 8000,
                explicit_max_tokens=max_tokens,
                prior_failures=prior_failures,
                attempt_index=frag_attempt,
            )
            if record is not None:
                break
            fs = _fragment_status(mid)
            fs.status = "pending"
            fs.llm_calls += 1
            if qualification is not None:
                fs.qualification_status = "failed"
                fs.qualification_issues = [issue.model_dump(mode="json") if hasattr(issue, "model_dump") else issue for issue in [qualification]]
            elif outcome is not None:
                fs.metadata["last_outcome_kind"] = outcome.outcome_kind
                fs.metadata["last_exception_class"] = outcome.exception_class

        if record is None:
            fs = _fragment_status(mid)
            fs.status = "failed"
            fs.issues = [{
                "code": "patch_generation.fragment_failed",
                "severity": "error",
                "message": f"material fragment {mid} failed after {max_fragment_attempts} attempts; last_failures={prior_failures[-3:]}",
            }]
            session.failed_fragment_issues[mid] = fs.issues
            _save_session(state, session)
            return PatchGenerationResult(
                ok=False, patch_type=patch_type,
                issues=[{
                    "code": "patch_generation.fragment_failed",
                    "severity": "error",
                    "message": f"material fragment {mid} failed qualification after {max_fragment_attempts} attempts",
                    "metadata": {"material_id": mid, "prior_failures": prior_failures[-3:]},
                }],
            )

        fs = _fragment_status(mid)
        fs.status = "accepted"
        fs.fragment_hash = record.fragment_hash
        fs.manifest_contract_hash = record.manifest_contract_hash
        fs.qualification_status = "passed"
        fs.qualification_issues = list(record.qualification_issues)
        fs.accepted_at_attempt = record.accepted_at_attempt
        fs.issues = []
        fs.metadata.pop("downgrade_reason", None)
        session.accepted_fragment_hashes[mid] = record.fragment_hash
        session.accepted_fragments[mid] = record
        _save_session(state, session)

    # -----------------------------------------------------------------
    # Merge phase (pure Python).
    # -----------------------------------------------------------------
    accepted_records: dict[str, AcceptedMaterialFragmentRecord] = dict(session.accepted_fragments)
    accepted_fragments: list[MaterialDefinitionFragment] = [
        MaterialDefinitionFragment(
            material_id=mid,
            material=record.material,
            fragment_hash=record.fragment_hash,
            manifest_contract_hash=record.manifest_contract_hash,
        )
        for mid, record in accepted_records.items()
    ]

    merge_result = _attempt_merge_with_replay(
        session=session,
        manifest=manifest,
        accepted_fragments=accepted_fragments,
        accepted_records=accepted_records,
        requirement=requirement,
        llm_client=llm_client,
        effective_max_tokens=effective_max or 8000,
        explicit_max_tokens=max_tokens,
        max_fragment_attempts=max_fragment_attempts,
        max_total_llm_calls=max_total_llm_calls,
        max_merge_replays=max_merge_replays,
        state=state,
    )

    if not merge_result.ok:
        _save_session(state, session)
        top_issue = _build_merge_failed_issue(merge_result)
        return PatchGenerationResult(ok=False, patch_type=patch_type, issues=[top_issue])

    merged_patch = merge_result.merged_patch or {}
    normalization = normalize_materials_patch_content(merged_patch, state=state)
    if normalization.changed:
        merged_patch = normalization.content
        state.metadata.setdefault("materials_deterministic_normalizations", [])
        state.metadata["materials_deterministic_normalizations"].extend(
            normalization.operations
        )
        session.metadata.setdefault("deterministic_normalizations", [])
        session.metadata["deterministic_normalizations"].extend(
            normalization.operations
        )

    ok, val_issues = _validate_merged_materials_patch(merged_patch)
    if not ok:
        _save_session(state, session)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type,
            issues=[{
                "code": "patch_generation.merge_failed",
                "severity": "error",
                "message": "merged MaterialsPatch failed standard validation",
                "metadata": {
                    "validation_issues": val_issues,
                    "merged_patch_hash": merge_result.merged_patch_hash,
                    "invalid_fragment_ids": merge_result.invalid_fragment_ids,
                },
            }],
        )

    import hashlib
    patch_hash = hashlib.sha256(
        json.dumps(merged_patch, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]

    envelope = PlanPatchEnvelope(
        patch_id=f"materials_fragmented_{patch_hash}",
        patch_type=patch_type,
        content=merged_patch,
        source="llm",
        status="valid",
        metadata={"deterministic_normalizations": normalization.operations}
        if normalization.changed else {},
    )
    session.completed = True
    session.merged_patch_hash = patch_hash
    _save_session(state, session)

    return PatchGenerationResult(
        ok=True, patch_type=patch_type, envelope=envelope,
        parsed_patch=merged_patch,
        attempts=[], issues=[],
    )


def _build_merge_failed_issue(merge_result: MaterialMergeResult) -> dict[str, Any]:
    return {
        "code": "patch_generation.merge_failed",
        "severity": "error",
        "message": f"material fragment merge failed: {len(merge_result.issues)} issue(s)",
        "metadata": {
            "invalid_fragment_ids": merge_result.invalid_fragment_ids,
            "accepted_fragment_ids": merge_result.accepted_fragment_ids,
            "manifest_id": merge_result.manifest_id,
            "issues": [issue.model_dump(mode="json") for issue in merge_result.issues],
        },
    }


def _attempt_merge_with_replay(
    *,
    session: MaterialsPatchGenerationSession,
    manifest: MaterialManifest,
    accepted_fragments: list[MaterialDefinitionFragment],
    accepted_records: dict[str, AcceptedMaterialFragmentRecord],
    requirement: str,
    llm_client: Any,
    effective_max_tokens: int,
    explicit_max_tokens: int | None,
    max_fragment_attempts: int,
    max_total_llm_calls: int,
    max_merge_replays: int,
    state: PlanBuildState,
) -> MaterialMergeResult:
    """Try the merge; replay only fragment-scoped failures."""
    merge_round = 0
    last_result: MaterialMergeResult | None = None

    while merge_round <= max_merge_replays:
        merge_result = merge_material_fragments_structured(
            manifest=manifest,
            accepted_fragments=accepted_fragments,
            accepted_records=accepted_records,
        )
        last_result = merge_result
        session.merge_history.append({
            "round": merge_round,
            "ok": merge_result.ok,
            "invalid_fragment_ids": merge_result.invalid_fragment_ids,
            "issues": [i.model_dump(mode="json") for i in merge_result.issues],
        })

        if merge_result.ok:
            return merge_result

        fragment_scoped = [
            i for i in merge_result.issues
            if i.retry_scope == "fragment" and i.material_id
        ]
        manifest_scoped = any(i.retry_scope == "manifest" for i in merge_result.issues)
        global_scoped = any(i.retry_scope == "global" for i in merge_result.issues)

        if manifest_scoped or global_scoped or not fragment_scoped:
            return merge_result

        if merge_round >= max_merge_replays:
            return merge_result

        invalid_ids = {i.material_id for i in fragment_scoped}
        all_manifest_ids = manifest.material_ids

        for mid in sorted(invalid_ids):
            if session.llm_call_count >= max_total_llm_calls:
                break
            item = manifest.item_by_id(mid)
            if item is None:
                continue
            _downgrade_fragment(session, mid, reason=f"merge_replay_round_{merge_round}")
            prior_failures: list[str] = []
            for frag_attempt in range(max_fragment_attempts):
                if session.llm_call_count >= max_total_llm_calls:
                    break
                record, _, _ = _generate_and_qualify_one_fragment(
                    item=item, llm_client=llm_client, session=session,
                    requirement=requirement,
                    all_manifest_material_ids=all_manifest_ids,
                    effective_max_tokens=effective_max_tokens,
                    explicit_max_tokens=explicit_max_tokens,
                    prior_failures=prior_failures,
                    attempt_index=frag_attempt,
                )
                if record is not None:
                    fs = next((f for f in session.fragment_statuses if f.material_id == mid), None)
                    if fs:
                        fs.status = "accepted"
                        fs.fragment_hash = record.fragment_hash
                        fs.manifest_contract_hash = record.manifest_contract_hash
                        fs.qualification_status = "passed"
                        fs.qualification_issues = list(record.qualification_issues)
                        fs.accepted_at_attempt = record.accepted_at_attempt
                    session.accepted_fragment_hashes[mid] = record.fragment_hash
                    session.accepted_fragments[mid] = record
                    accepted_records[mid] = record
                    break

        accepted_fragments = [
            MaterialDefinitionFragment(
                material_id=mid, material=rec.material,
                fragment_hash=rec.fragment_hash,
                manifest_contract_hash=rec.manifest_contract_hash,
            )
            for mid, rec in accepted_records.items()
        ]
        _save_session(state, session)
        merge_round += 1

    return last_result or MaterialMergeResult(ok=False)


# ---------------------------------------------------------------------------
# Requirement set loader
# ---------------------------------------------------------------------------

def _load_requirement_set(state: PlanBuildState) -> MaterialGenerationRequirementSet | None:
    raw = state.metadata.get("planning_material_requirement_set")
    if raw is None:
        return None
    if isinstance(raw, MaterialGenerationRequirementSet):
        return raw
    if isinstance(raw, dict):
        return MaterialGenerationRequirementSet.model_validate(raw)
    return None


def _derive_requirement_set_from_facts(state: PlanBuildState) -> MaterialGenerationRequirementSet | None:
    """Fallback: derive a material requirement set from the accepted FactsPatch.

    Used when the geometry-component inventory path did not populate
    ``planning_material_requirement_set``.  This lets the fragmented pipeline
    activate (one fragment per material role / fuel variant) so a large
    materials catalog does not hit a monolithic truncation failure.
    """

    from .patches import parse_patch_content

    facts_env = next(
        (e for e in state.patches.values() if e.patch_type == "facts" and e.status == "valid"),
        None,
    )
    if facts_env is None:
        return None
    try:
        facts_patch = parse_patch_content("facts", facts_env.content)
    except Exception:
        return None
    requirement_set = extract_material_requirements_from_facts(facts_patch)
    if not requirement_set.requirements:
        return None
    return requirement_set
