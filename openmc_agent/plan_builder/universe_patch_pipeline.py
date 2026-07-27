"""Top-level entry point for fragmented universe patch generation.

Provides ``generate_universes_patch`` which orchestrates the full pipeline:

    requirements
    → deterministic manifest (with per-item contract hash)
    → one-universe LLM fragments
    → deterministic fragment qualification against manifest contract
    → checkpoint with hash / contract / qualification integrity
    → deterministic structured merge (pure Python, no LLM)
    → merged patch validation
    → targeted fragment invalidation and replay when merge reports
      fragment-scoped issues
    → one authoritative UniversesPatch envelope

The executor calls this instead of ``generate_patch`` when the universes
patch is large enough to risk truncation.
"""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any

from .llm_adapter import PatchLLMResponse, normalize_patch_llm_response
from .patch_generator import PatchGenerationResult, PatchGenerationAttempt
from .patches import get_patch_json_schema
from .state import PlanBuildState, PlanPatchEnvelope
from .universe_fragment_generation import (
    AcceptedFragmentRecord,
    FragmentStatus,
    LargePatchGenerationSession,
    UniverseDefinitionFragment,
    UniverseManifest,
    UniverseManifestItem,
    UniverseGenerationRequirementSet,
    UniverseMergeResult,
    build_manifest_from_requirements,
    estimate_universes_output_size,
    extract_universe_requirements,
    merge_universe_fragments_structured,
    resolve_patch_output_budget,
    should_fragment_universes,
    validate_manifest,
    validate_merged_patch,
)
from .universe_fragment_qualification import (
    FragmentQualificationResult,
    qualify_universe_fragment,
    verify_accepted_fragment_record,
)


def _expected_localized_insert_universe_ids(facts_obj: Any | None) -> dict[str, list[str]]:
    """Return accepted Facts localized-insert expected universe IDs by row ID."""

    expected: dict[str, list[str]] = {}
    if facts_obj is None:
        return expected
    for req in getattr(facts_obj, "localized_insert_requirements", []) or []:
        req_id = getattr(req, "requirement_id", "")
        if not req_id:
            continue
        values = getattr(req, "expected_insert_universe_ids", []) or []
        ids = [str(item) for item in values if isinstance(item, str) and item]
        if ids:
            expected[str(req_id)] = list(dict.fromkeys(ids))
    return expected


def _localized_insert_ids_for_universe(universe: dict[str, Any]) -> set[str]:
    """Extract source-backed localized insert requirement IDs from universe metadata."""

    meta = universe.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    ids: set[str] = set()
    singular = meta.get("localized_insert_requirement_id")
    if isinstance(singular, str) and singular:
        ids.add(singular)
    values = meta.get("localized_insert_requirement_ids") or []
    values = values if isinstance(values, list) else [values]
    for value in values:
        if isinstance(value, str) and value:
            ids.add(value)
    source_values = meta.get("source_requirement_ids") or []
    source_values = source_values if isinstance(source_values, list) else [source_values]
    for value in source_values:
        if isinstance(value, str) and value.startswith("localized_insert:"):
            req_id = value.split(":", 1)[1]
            if req_id:
                ids.add(req_id)
    return ids


def materialize_localized_insert_universe_aliases(
    patch: dict[str, Any],
    *,
    facts_obj: Any | None,
) -> list[dict[str, Any]]:
    """Add exact-ID localized-insert universe aliases required by accepted Facts.

    Fragmented generation may create one radial/profile universe whose metadata
    says it covers a localized insert requirement, while downstream placement
    profiles reference the source-backed ``expected_insert_universe_ids`` from
    Facts.  The Universes patch must therefore expose those exact IDs as real
    universe IDs; metadata-only coverage is insufficient for Placement/Axial
    consumers.
    """

    expected_by_req = _expected_localized_insert_universe_ids(facts_obj)
    universes = patch.get("universes", [])
    if not expected_by_req or not isinstance(universes, list):
        return []

    existing_ids = {
        str(item.get("universe_id"))
        for item in universes
        if isinstance(item, dict) and item.get("universe_id")
    }
    sources_by_req: dict[str, dict[str, Any]] = {}
    for universe in universes:
        if not isinstance(universe, dict):
            continue
        for req_id in _localized_insert_ids_for_universe(universe):
            sources_by_req.setdefault(req_id, universe)

    issues: list[dict[str, Any]] = []
    aliases: list[dict[str, Any]] = []
    for req_id, expected_ids in expected_by_req.items():
        source = sources_by_req.get(req_id)
        for expected_id in expected_ids:
            if expected_id in existing_ids:
                continue
            if source is None:
                issues.append({
                    "code": "localized_insert.expected_universe_source_missing",
                    "severity": "error",
                    "message": (
                        f"no generated universe carries localized_insert_requirement_id "
                        f"'{req_id}' for expected universe '{expected_id}'"
                    ),
                    "requirement_id": req_id,
                    "universe_id": expected_id,
                })
                continue
            alias = deepcopy(source)
            source_id = str(source.get("universe_id") or "")
            alias["universe_id"] = expected_id
            meta = dict(alias.get("metadata") or {})
            meta["alias_of_universe_id"] = source_id
            meta["localized_insert_expected_universe_id"] = expected_id
            meta["localized_insert_requirement_id"] = req_id
            ids = list(meta.get("localized_insert_requirement_ids") or [])
            if req_id not in ids:
                ids.append(req_id)
            meta["localized_insert_requirement_ids"] = sorted(set(str(item) for item in ids if item))
            alias["metadata"] = meta
            aliases.append(alias)
            existing_ids.add(expected_id)

    if aliases:
        universes.extend(aliases)
    return issues


def _get_session(state: PlanBuildState, input_hash: str) -> LargePatchGenerationSession | None:
    """Retrieve a checkpoint session matching the input hash."""
    sessions = state.metadata.get("large_patch_generation_sessions", {})
    if not isinstance(sessions, dict):
        return None
    session_data = sessions.get(f"universes:{input_hash}")
    if session_data is None:
        return None
    if isinstance(session_data, dict):
        return LargePatchGenerationSession.model_validate(session_data)
    return session_data


def _save_session(state: PlanBuildState, session: LargePatchGenerationSession) -> None:
    """Persist a checkpoint session."""
    sessions = state.metadata.setdefault("large_patch_generation_sessions", {})
    sessions[f"universes:{session.input_hash}"] = session.model_dump(mode="json")


def _stale_session(state: PlanBuildState, input_hash: str) -> None:
    """Mark a session as stale (input hash changed)."""
    sessions = state.metadata.get("large_patch_generation_sessions", {})
    if isinstance(sessions, dict):
        for key in list(sessions.keys()):
            if key.startswith("universes:") and key != f"universes:{input_hash}":
                old = sessions[key]
                if isinstance(old, dict):
                    old["completed"] = False
                    old.setdefault("metadata", {})["stale"] = True


# ---------------------------------------------------------------------------
# LLM call classification
# ---------------------------------------------------------------------------


class _FragmentLLMOutcome:
    """Classified outcome of a single fragment LLM call.

    Distinguishes provider exceptions from schema exceptions and content
    validation failures so the caller can route diagnostics without
    silently swallowing errors.
    """

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
        self.outcome_kind = outcome_kind  # "ok" | "provider_exception" | "parse_exception" | "empty"
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
    """Call the LLM for a single universe fragment.

    Returns a classified :class:`_FragmentLLMOutcome`.  Provider exceptions
    are recorded once and never silently swallowed into a fallback LLM
    call.  ``generate_patch_json_with_meta`` is preferred; ``generate_patch_json``
    is the structured fallback; plain callable is the last-resort fallback
    so legacy fake clients keep working in tests.
    """
    schema = get_patch_json_schema("universes")

    # Path 1: structured metadata API.
    if hasattr(llm_client, "generate_patch_json_with_meta"):
        try:
            resp = llm_client.generate_patch_json_with_meta(
                prompt=prompt, patch_type="universes",
                json_schema=schema, max_tokens=max_tokens,
            )
            if isinstance(resp, PatchLLMResponse):
                if not resp.content:
                    return _FragmentLLMOutcome(
                        response=resp, outcome_kind="empty",
                        note="provider returned empty content",
                    )
                return _FragmentLLMOutcome(response=resp, outcome_kind="ok")
            # Some clients return a str; normalize.
            return _FragmentLLMOutcome(
                response=normalize_patch_llm_response(resp),
                outcome_kind="ok",
            )
        except Exception as exc:
            return _FragmentLLMOutcome(
                response=None, outcome_kind="provider_exception",
                exception=exc, exception_class=type(exc).__name__,
                note=f"generate_patch_json_with_meta raised {type(exc).__name__}",
            )

    # Path 2: structured JSON API without telemetry.
    if hasattr(llm_client, "generate_patch_json"):
        try:
            raw = llm_client.generate_patch_json(
                prompt=prompt, patch_type="universes",
                json_schema=schema, max_tokens=max_tokens,
            )
            resp = PatchLLMResponse(
                content=raw or "",
                output_mode_used=str(getattr(llm_client, "last_output_mode_used", "structured")),
            )
            if not resp.content:
                return _FragmentLLMOutcome(
                    response=resp, outcome_kind="empty",
                    note="provider returned empty content",
                )
            return _FragmentLLMOutcome(response=resp, outcome_kind="ok")
        except Exception as exc:
            return _FragmentLLMOutcome(
                response=None, outcome_kind="provider_exception",
                exception=exc, exception_class=type(exc).__name__,
                note=f"generate_patch_json raised {type(exc).__name__}",
            )

    # Path 3: plain callable (legacy FakePatchLLM).
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
        return _FragmentLLMOutcome(
            response=resp, outcome_kind="empty",
            note="plain callable returned empty content",
        )
    return _FragmentLLMOutcome(response=resp, outcome_kind="ok")


def _preflight_material_role_coverage(
    manifest: UniverseManifest,
    material_roles_by_id: dict[str, str],
) -> list[dict[str, Any]]:
    """Check that every required_material_role has at least one accepted material.

    Returns a list of issue dicts (empty = pass).  When any role is
    uncovered, the pipeline blocks deterministically with
    ``unavailable_material_role`` — no LLM call is wasted.
    """
    available_roles = set(material_roles_by_id.values())
    issues: list[dict[str, Any]] = []
    for item in manifest.items:
        for role in item.required_material_roles:
            if role not in available_roles:
                issues.append({
                    "code": "patch_generation.unavailable_material_role",
                    "severity": "error",
                    "message": (
                        f"universe '{item.universe_id}' requires material role "
                        f"'{role}' but no accepted material provides it"
                    ),
                    "metadata": {
                        "universe_id": item.universe_id,
                        "required_role": role,
                        "available_roles": sorted(available_roles),
                    },
                })
    return issues


def _build_role_binding_map(
    item: UniverseManifestItem,
    material_roles_by_id: dict[str, str],
    material_source_variants_by_id: dict[str, str | None] | None = None,
) -> dict[str, list[str]]:
    """Map each required_material_role to accepted material IDs with that role."""
    material_source_variants_by_id = material_source_variants_by_id or {}
    binding: dict[str, list[str]] = {}
    for role in item.required_material_roles:
        binding[role] = sorted(
            mid for mid, mrole in material_roles_by_id.items()
            if mrole == role
            and not (
                role == "fuel"
                and item.fuel_variant_id
                and material_source_variants_by_id.get(mid) is not None
                and material_source_variants_by_id.get(mid) != item.fuel_variant_id
            )
        )
    return binding


def _build_fragment_prompt(
    item: UniverseManifestItem,
    *,
    requirement: str,
    material_summary: str,
    prior_failures: list[str] | None = None,
    role_binding: dict[str, list[str]] | None = None,
) -> str:
    """Build a focused prompt for generating one universe."""
    lines = [
        f"Generate ONLY the universe definition for universe_id={item.universe_id}.",
        f"Kind: {item.kind}",
    ]
    if item.required_cell_roles:
        lines.append(f"Required cell roles: {', '.join(item.required_cell_roles)}")
    if item.required_material_ids:
        lines.append(f"Required material IDs: {', '.join(item.required_material_ids)}")
    if item.required_material_roles:
        lines.append(f"Required material roles: {', '.join(item.required_material_roles)}")
    if role_binding:
        lines.append("\nRole → material bindings (use these exact material_id values):")
        for role, mids in sorted(role_binding.items()):
            lines.append(f"  {role} → {', '.join(mids) if mids else '<NONE AVAILABLE — role unsatisfied>'}")
        lines.append("Each required role above MUST be referenced by at least one cell's material_id.")
    if item.fuel_variant_id:
        lines.append(f"Fuel variant: {item.fuel_variant_id}")
        lines.append(
            "For this fuel universe, every role='fuel' cell MUST use a fuel "
            "material whose source_variant_id matches this fuel variant; do "
            "not mix fuel materials from multiple variants inside one universe."
        )
    if item.protected_through_path_roles:
        lines.append(f"Protected through-path roles: {', '.join(item.protected_through_path_roles)}")
    lines.append(f"\nAvailable materials:\n{material_summary}")
    lines.append(f"\nSource context:\n{requirement[:2000]}")
    lines.append("\nOutput a single JSON object with keys: patch_type, universes (list with ONE universe).")
    lines.append("The universe must have: universe_id, kind, cells (list of cell layers).")
    lines.append("Each cell layer needs: id, role, material_id, region_kind, r_min_cm, r_max_cm.")
    lines.append(
        "IMPORTANT: every material_id MUST be one of the available materials listed above. "
        "Do NOT copy placeholder values such as REPLACE or <material_id>."
    )
    if prior_failures:
        lines.append(f"\nPrior failures to avoid:\n{chr(10).join(prior_failures[-3:])}")
    lines.append("\n```json")
    lines.append(json.dumps({
        "patch_type": "universes",
        "universes": [{
            "universe_id": item.universe_id,
            "kind": item.kind,
            "cells": [
                {
                    "id": "<unique_cell_id>",
                    "role": item.required_cell_roles[0] if item.required_cell_roles else "filler",
                    "material_id": "<one_of_the_materials_listed_above>",
                    "region_kind": "cylinder",
                    "r_min_cm": 0.0,
                    "r_max_cm": 0.4,
                }
            ],
        }],
    }, indent=2))
    lines.append("```")
    return "\n".join(lines)


def _build_schema_repair_prompt(
    item: UniverseManifestItem,
    *,
    requirement: str,
    material_summary: str,
    role_binding: dict[str, list[str]] | None = None,
    prior_failures: list[str] | None = None,
) -> str:
    """Focused schema-repair prompt for the second attempt.

    This prompt is sent when the first attempt failed qualification.
    It shows the exact JSON schema, the exact failures, and asks for
    a minimal repair — no narrative, no reasoning, just the corrected JSON.
    """
    lines = [
        f"SCHEMA REPAIR: fix the universe definition for universe_id={item.universe_id}.",
        f"Kind: {item.kind}",
        "",
        "The previous attempt failed. Fix ONLY the errors listed below.",
        "Output ONLY a JSON object — no prose, no explanation.",
        "",
    ]
    if item.required_cell_roles:
        lines.append(f"Required cell roles (each MUST appear in at least one cell): {', '.join(item.required_cell_roles)}")
    if item.required_material_roles:
        lines.append(f"Required material roles: {', '.join(item.required_material_roles)}")
    if item.fuel_variant_id:
        lines.append(f"Fuel variant: {item.fuel_variant_id}")
        lines.append(
            "Every role='fuel' cell MUST use a fuel material whose "
            "source_variant_id matches this fuel variant; do not mix fuel "
            "materials from multiple variants inside one universe."
        )
    if role_binding:
        lines.append("\nRole → material bindings (MUST use these exact material_id values):")
        for role, mids in sorted(role_binding.items()):
            if mids:
                lines.append(f"  {role} → {', '.join(mids)}")
            else:
                lines.append(f"  {role} → <NO MATERIAL AVAILABLE>")
        lines.append("Each required role MUST be referenced by at least one cell.")
    lines.append(f"\nAvailable materials:\n{material_summary}")
    if prior_failures:
        lines.append(f"\nERRORS TO FIX:")
        for failure in prior_failures[-5:]:
            lines.append(f"  - {failure}")
    lines.append("")
    lines.append("Required JSON schema (fill in real values):")
    example_material_id = "<material_id>"
    if role_binding and item.required_material_roles:
        bound_materials = role_binding.get(item.required_material_roles[0]) or []
        if bound_materials:
            example_material_id = bound_materials[0]
    lines.append("```json")
    lines.append(json.dumps({
        "patch_type": "universes",
        "universes": [{
            "universe_id": item.universe_id,
            "kind": item.kind,
            "cells": [
                {
                    "id": "<cell_1>",
                    "role": item.required_cell_roles[0] if item.required_cell_roles else "filler",
                    "material_id": example_material_id,
                    "region_kind": "cylinder",
                    "r_min_cm": 0.0,
                    "r_max_cm": 0.4,
                }
            ],
        }],
    }, indent=2))
    lines.append("```")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resume verification
# ---------------------------------------------------------------------------


def _verify_resume_fragments(
    *,
    session: LargePatchGenerationSession,
    manifest: UniverseManifest,
    known_material_ids: set[str],
    material_roles_by_id: dict[str, str],
    material_source_variants_by_id: dict[str, str | None],
) -> tuple[dict[str, AcceptedFragmentRecord], dict[str, FragmentQualificationResult]]:
    """Re-verify every accepted fragment record on resume.

    Returns ``(valid_records, invalid_results)``.  A record becomes
    invalid if its data is missing, hash drifted, manifest contract
    changed, or its qualification now fails against the current
    MaterialsPatch.  Invalid records are NOT deleted here; the caller
    downgrades them so only the affected fragments are regenerated.
    """
    valid: dict[str, AcceptedFragmentRecord] = {}
    invalid: dict[str, FragmentQualificationResult] = {}
    manifest_by_id = {item.universe_id: item for item in manifest.items}

    # Source of truth is the typed ``accepted_fragments`` dict.  Fall back
    # to the legacy ``metadata._accepted_fragments`` blob if needed so
    # older sessions can still resume.
    legacy_store = session.metadata.get("_accepted_fragments") or {}

    for fs in session.fragment_statuses:
        uid = fs.universe_id
        if fs.status != "accepted":
            continue
        record = session.accepted_fragments.get(uid)
        if record is None and uid in legacy_store:
            # Migrate legacy record on the fly.
            legacy = legacy_store[uid]
            if isinstance(legacy, dict):
                record = AcceptedFragmentRecord(
                    universe_id=uid,
                    universe=legacy.get("universe", {}),
                    fragment_hash=fs.fragment_hash,
                    manifest_contract_hash=fs.manifest_contract_hash,
                    qualification_status="passed" if fs.qualification_status == "passed" else "pending",
                    qualification_issues=list(fs.qualification_issues),
                    accepted_at_attempt=fs.accepted_at_attempt or 0,
                )
                session.accepted_fragments[uid] = record
        if record is None:
            invalid[uid] = FragmentQualificationResult(
                ok=False,
                universe_id=uid,
                manifest_contract_hash=manifest_by_id.get(uid, UniverseManifestItem(universe_id=uid)).contract_hash,
                issues=[],
                metadata={"reason": "qualification.record_missing"},
            )
            continue
        item = manifest_by_id.get(uid)
        if item is None:
            invalid[uid] = FragmentQualificationResult(
                ok=False,
                universe_id=uid,
                issues=[],
                metadata={"reason": "qualification.universe_not_in_manifest"},
            )
            continue
        result = verify_accepted_fragment_record(
            manifest_item=item,
            record=record,
            known_material_ids=known_material_ids,
            material_roles_by_id=material_roles_by_id,
            material_source_variants_by_id=material_source_variants_by_id,
        )
        if result.ok:
            valid[uid] = record
        else:
            invalid[uid] = result
    return valid, invalid


def _downgrade_fragment(
    session: LargePatchGenerationSession,
    universe_id: str,
    *,
    reason: str,
    qualification_result: FragmentQualificationResult | None = None,
) -> None:
    """Downgrade an accepted fragment to ``pending`` so it is regenerated.

    Other accepted fragments are NOT touched.
    """
    session.accepted_fragments.pop(universe_id, None)
    session.accepted_fragment_hashes.pop(universe_id, None)
    session.metadata.get("_accepted_fragments", {}).pop(universe_id, None)
    fs = next((fs for fs in session.fragment_statuses if fs.universe_id == universe_id), None)
    if fs is None:
        fs = FragmentStatus(universe_id=universe_id, status="pending")
        session.fragment_statuses.append(fs)
    fs.status = "pending"
    fs.fragment_hash = ""
    fs.qualification_status = "pending"
    fs.qualification_issues = []
    fs.accepted_at_attempt = None
    fs.metadata["downgrade_reason"] = reason
    if qualification_result is not None:
        fs.metadata["last_qualification_issues"] = [
            issue.model_dump(mode="json") for issue in qualification_result.issues
        ]


# ---------------------------------------------------------------------------
# Single-fragment generation + qualification
# ---------------------------------------------------------------------------


def _generate_and_qualify_one_fragment(
    *,
    item: UniverseManifestItem,
    llm_client: Any,
    session: LargePatchGenerationSession,
    requirement: str,
    material_summary: str,
    known_material_ids: set[str],
    material_roles_by_id: dict[str, str],
    material_source_variants_by_id: dict[str, str | None],
    effective_max_tokens: int,
    explicit_max_tokens: int | None,
    prior_failures: list[str],
    attempt_index: int,
) -> tuple[AcceptedFragmentRecord | None, FragmentQualificationResult | None, _FragmentLLMOutcome | None]:
    """Generate and qualify one fragment for ``item``.

    Returns ``(record, qualification_result, llm_outcome)``.  When the
    LLM call fails or qualification fails, ``record`` is ``None`` and the
    other return values carry the diagnostic.
    """
    # Build role → material_id binding for the prompt.
    role_binding = _build_role_binding_map(
        item,
        material_roles_by_id,
        material_source_variants_by_id,
    )

    # Attempt 0: full prompt.  Attempt 1+: focused schema-repair prompt.
    if attempt_index == 0:
        prompt = _build_fragment_prompt(
            item, requirement=requirement, material_summary=material_summary,
            prior_failures=prior_failures,
            role_binding=role_binding,
        )
    else:
        prompt = _build_schema_repair_prompt(
            item, requirement=requirement, material_summary=material_summary,
            role_binding=role_binding,
            prior_failures=prior_failures,
        )
    frag_max = resolve_patch_output_budget(
        explicit=explicit_max_tokens, fragment_mode=True,
        provider_max_output=effective_max_tokens,
    )
    session.llm_call_count += 1
    outcome = _call_llm_fragment(llm_client, prompt=prompt, max_tokens=frag_max)
    session.provider_telemetry.append({
        "universe_id": item.universe_id,
        "attempt": attempt_index,
        "outcome_kind": outcome.outcome_kind,
        "exception_class": outcome.exception_class,
        "note": outcome.note,
        "finish_reason": outcome.response.finish_reason if outcome.response else None,
        "output_mode_used": outcome.response.output_mode_used if outcome.response else "",
        "completion_tokens": outcome.response.completion_tokens if outcome.response else None,
        "reasoning_tokens": outcome.response.reasoning_tokens if outcome.response else None,
    })
    if outcome.failed:
        prior_failures.append(
            f"llm_call[{item.universe_id}] outcome={outcome.outcome_kind} "
            f"({outcome.exception_class or outcome.note})"
        )
        return None, None, outcome

    # Parse the fragment response.
    from .patch_generator import parse_llm_patch_json
    from .patches import PatchParseError

    try:
        parsed = parse_llm_patch_json(outcome.response.content or "", "universes")
        universes_list = parsed.get("universes", [])
        if not universes_list:
            raise ValueError("empty universes list in fragment response")
        if len(universes_list) > 1:
            raise ValueError(
                f"fragment response contains {len(universes_list)} universes; "
                "exactly one is required per fragment"
            )
        universe_data = universes_list[0]
    except (PatchParseError, ValueError, Exception) as exc:
        prior_failures.append(
            f"parse[{item.universe_id}] {type(exc).__name__}: {exc}"
        )
        # Record a synthetic qualification result so the failure is observable.
        from .universe_fragment_qualification import FragmentQualificationIssue
        synth = FragmentQualificationResult(
            ok=False, universe_id=item.universe_id,
            manifest_contract_hash=item.contract_hash,
            issues=[FragmentQualificationIssue(
                code="qualification.fragment_parse_failed",
                universe_id=item.universe_id,
                message=f"could not parse fragment: {type(exc).__name__}: {exc}",
                retryable=True,
            )],
        )
        return None, synth, outcome

    fragment = UniverseDefinitionFragment(
        universe_id=item.universe_id,
        universe=universe_data,
        manifest_contract_hash=item.contract_hash,
    )
    qualification = qualify_universe_fragment(
        manifest_item=item,
        fragment=fragment,
        known_material_ids=known_material_ids,
        material_roles_by_id=material_roles_by_id,
        material_source_variants_by_id=material_source_variants_by_id,
        qualification_attempt=attempt_index,
    )
    if not qualification.ok:
        prior_failures.extend(
            f"{issue.code}: {issue.message}" for issue in qualification.issues
            if issue.severity == "error"
        )
        return None, qualification, outcome

    record = AcceptedFragmentRecord(
        universe_id=item.universe_id,
        universe=qualification.canonical_universe_data,
        fragment_hash=qualification.fragment_hash,
        manifest_contract_hash=item.contract_hash,
        qualification_status="passed",
        qualification_issues=[
            issue.model_dump(mode="json") for issue in qualification.issues
        ],
        accepted_at_attempt=attempt_index,
    )
    return record, qualification, outcome


# ---------------------------------------------------------------------------
# Top-level entry
# ---------------------------------------------------------------------------


def generate_universes_patch(
    *,
    requirement: str,
    state: PlanBuildState,
    llm_client: Any,
    mode: str = "auto",
    max_tokens: int | None = None,
    max_fragment_attempts: int = 2,
    max_total_llm_calls: int = 40,
    safe_output_ratio: float = 0.6,
    strict_structured: bool = False,
    max_merge_replays: int = 2,
    inventory_universe_requirement_set: Any = None,
) -> PatchGenerationResult:
    """Generate a universes patch, using fragmentation when necessary.

    This function replaces ``generate_patch`` for the ``universes`` patch type.
    Other patch types continue to use ``generate_patch``.

    Transaction contract
    --------------------
    * A fragment only enters the accepted set after deterministic
      qualification against its manifest item.
    * Accepted fragments store data + canonical hash + manifest contract
      hash + qualification record; resume re-verifies all four.
    * Merge is pure Python and deterministic; structured issues attribute
      each failure to a fragment / manifest / global scope.
    * Fragment-scoped merge failures trigger a *targeted* replay that
      only regenerates the invalid fragments.  Other accepted fragments
      are never re-generated.
    * Manifest / global failures fail closed.

    Parameters
    ----------
    mode
        ``"auto"`` (default), ``"monolithic"``, or ``"fragmented"``.
    max_fragment_attempts
        Maximum retry attempts per individual universe fragment within
        a single generation transaction.
    max_total_llm_calls
        Global LLM call budget for this generation (prevents infinite fragmentation).
    max_merge_replays
        Upper bound on fragment-scoped merge replays inside one generation
        transaction.
    """
    from .patch_generator import generate_patch, _looks_truncated
    from .llm_adapter import LARGE_PATCH_MAX_TOKENS
    from .patches import parse_patch_content, PatchParseError

    patch_type = "universes"
    effective_max = max_tokens or LARGE_PATCH_MAX_TOKENS.get(patch_type)

    # Extract requirements from accepted upstream context.
    facts_env = next((e for e in state.patches.values() if e.patch_type == "facts" and e.status == "valid"), None)
    materials_env = next((e for e in state.patches.values() if e.patch_type == "materials" and e.status == "valid"), None)
    facts_obj = None
    materials_obj = None
    known_material_ids: set[str] = set()
    material_roles_by_id: dict[str, str] = {}
    material_source_variants_by_id: dict[str, str | None] = {}
    if facts_env is not None:
        try:
            facts_obj = parse_patch_content("facts", facts_env.content)
        except Exception:
            pass
    if materials_env is not None:
        try:
            materials_obj = parse_patch_content("materials", materials_env.content)
            for m in getattr(materials_obj, "materials", []):
                known_material_ids.add(m.material_id)
                if getattr(m, "role", None):
                    material_roles_by_id[m.material_id] = m.role
                material_source_variants_by_id[m.material_id] = getattr(
                    m, "source_variant_id", None
                )
        except Exception:
            pass

    if inventory_universe_requirement_set is not None:
        from .universe_fragment_generation import (
            convert_inventory_to_generation_requirements,
        )
        requirement_set = convert_inventory_to_generation_requirements(
            inventory_universe_requirement_set
        )
    else:
        requirement_set = extract_universe_requirements(
            facts=facts_obj, materials=materials_obj,
            canonical_task_plan=getattr(state, "canonical_task_plan", None),
            confirmed_records=getattr(state, "plan_confirmed_plan_fact_records", None),
        )

    # Check for existing checkpoint session.
    session = _get_session(state, requirement_set.input_hash)
    if session is None:
        _stale_session(state, requirement_set.input_hash)
        session = LargePatchGenerationSession(
            session_id=f"session:{requirement_set.input_hash}",
            patch_type=patch_type,
            input_hash=requirement_set.input_hash,
            mode=mode,  # type: ignore[arg-type]
            requirement_set_hash=requirement_set.input_hash,
        )

    material_summary = ""
    if materials_obj is not None:
        parts = []
        for m in getattr(materials_obj, "materials", []):
            parts.append(
                f"  - {m.material_id} (role={getattr(m, 'role', 'unknown')}, "
                f"density={getattr(m, 'density_g_cm3', '?')})"
            )
        material_summary = "\n".join(parts)

    # Decide strategy.
    history_truncated = any(
        "json_truncated" in str(a.get("issues", []))
        for a in getattr(state, "metadata", {}).get("patch_attempt_artifacts", {}).values()
        if isinstance(a, dict) and a.get("patch_type") == patch_type
    )

    do_fragment, fragment_reason = should_fragment_universes(
        mode=mode,  # type: ignore[arg-type]
        universe_count=max(len(requirement_set.requirements), 1),
        provider_max_output_tokens=effective_max,
        reasoning_enabled=False,
        history_json_truncated=history_truncated,
        safe_output_ratio=safe_output_ratio,
    )

    if not do_fragment:
        # Monolithic path: delegate to existing generate_patch.
        session.strategy_transitions.append({"from": "none", "to": "monolithic", "reason": fragment_reason})
        _save_session(state, session)
        result = generate_patch(
            patch_type=patch_type, requirement=requirement, state=state,
            llm_client=llm_client, max_attempts=2, max_tokens=effective_max,
        )
        # Check if monolithic was truncated → switch to fragmented.
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
        manifest = build_manifest_from_requirements(
            requirement_set, known_material_ids=known_material_ids,
        )
        manifest_errors = validate_manifest(manifest, requirement_set, known_material_ids=known_material_ids)
        if manifest_errors:
            session.manifest_status = "failed"
            _save_session(state, session)
            return PatchGenerationResult(
                ok=False, patch_type=patch_type,
                issues=[{
                    "code": "patch_generation.manifest_validation_failed",
                    "severity": "error",
                    "message": f"manifest errors: {manifest_errors}",
                    "metadata": {"manifest_errors": manifest_errors},
                }],
            )
        session.manifest = manifest
        session.manifest_status = "accepted"
    else:
        manifest = session.manifest
        # Recompute contract hashes defensively if they were persisted empty
        # (e.g., older sessions written before this field existed).
        for item in manifest.items:
            if not item.contract_hash:
                item.recompute_contract_hash()

    # Material-role preflight: verify every required_material_role has at
    # least one accepted material.  Without this the LLM is asked to
    # generate a universe referencing a role that cannot be satisfied —
    # guaranteed to fail qualification.  Block deterministically instead.
    preflight_issues = _preflight_material_role_coverage(
        manifest, material_roles_by_id,
    )
    if preflight_issues:
        _save_session(state, session)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type, issues=preflight_issues,
        )

    # Resume verification: downgrade corrupted accepted fragments BEFORE
    # generating anything new.  Only the corrupted ones are regenerated.
    valid_resume_records, invalid_resume_results = _verify_resume_fragments(
        session=session, manifest=manifest,
        known_material_ids=known_material_ids,
        material_roles_by_id=material_roles_by_id,
        material_source_variants_by_id=material_source_variants_by_id,
    )
    for uid, result in invalid_resume_results.items():
        _downgrade_fragment(
            session, uid,
            reason=f"resume_verification_failed: {result.metadata.get('reason', 'qualification_failed')}",
            qualification_result=result,
        )

    # Build the per-fragment status table.
    manifest_by_id = {item.universe_id: item for item in manifest.items}

    def _fragment_status(uid: str) -> FragmentStatus:
        fs = next((fs for fs in session.fragment_statuses if fs.universe_id == uid), None)
        if fs is None:
            fs = FragmentStatus(universe_id=uid, status="pending")
            session.fragment_statuses.append(fs)
        return fs

    # -----------------------------------------------------------------
    # Generation phase 1: fill any pending/failed fragments.
    # -----------------------------------------------------------------
    for item_id in manifest.generation_order:
        fs = _fragment_status(item_id)
        if fs.status == "accepted" and item_id in valid_resume_records:
            # Trust the resume-verified record.
            continue
        if item_id not in manifest_by_id:
            continue
        item = manifest_by_id[item_id]
        prior_failures: list[str] = []

        record: AcceptedFragmentRecord | None = None
        for frag_attempt in range(max_fragment_attempts):
            if session.llm_call_count >= max_total_llm_calls:
                break
            record, qualification, outcome = _generate_and_qualify_one_fragment(
                item=item, llm_client=llm_client, session=session,
                requirement=requirement, material_summary=material_summary,
                known_material_ids=known_material_ids,
                material_roles_by_id=material_roles_by_id,
                material_source_variants_by_id=material_source_variants_by_id,
                effective_max_tokens=effective_max or 8000,
                explicit_max_tokens=max_tokens,
                prior_failures=prior_failures,
                attempt_index=frag_attempt,
            )
            if record is not None:
                break
            # Record the failed attempt on the status table.
            fs = _fragment_status(item_id)
            fs.status = "pending"
            fs.llm_calls += 1
            if qualification is not None:
                fs.qualification_status = "failed"
                fs.qualification_issues = [
                    issue.model_dump(mode="json") for issue in qualification.issues
                ]
            elif outcome is not None:
                fs.metadata["last_outcome_kind"] = outcome.outcome_kind
                fs.metadata["last_exception_class"] = outcome.exception_class

        if record is None:
            fs = _fragment_status(item_id)
            fs.status = "failed"
            fs.issues = [{
                "code": "patch_generation.fragment_failed",
                "severity": "error",
                "message": (
                    f"fragment {item_id} failed after {max_fragment_attempts} attempts; "
                    f"last_failures={prior_failures[-3:]}"
                ),
            }]
            session.failed_fragment_issues[item_id] = fs.issues
            _save_session(state, session)
            return PatchGenerationResult(
                ok=False, patch_type=patch_type,
                issues=[{
                    "code": "patch_generation.fragment_failed",
                    "severity": "error",
                    "message": (
                        f"universe fragment {item_id} failed qualification after "
                        f"{max_fragment_attempts} attempts"
                    ),
                    "metadata": {
                        "universe_id": item_id,
                        "last_qualification_issues": fs.qualification_issues,
                        "prior_failures": prior_failures[-3:],
                    },
                }],
            )

        # Record accepted.
        fs = _fragment_status(item_id)
        fs.status = "accepted"
        fs.fragment_hash = record.fragment_hash
        fs.manifest_contract_hash = record.manifest_contract_hash
        fs.qualification_status = "passed"
        fs.qualification_issues = list(record.qualification_issues)
        fs.accepted_at_attempt = record.accepted_at_attempt
        fs.issues = []
        fs.metadata.pop("downgrade_reason", None)
        fs.metadata.pop("last_qualification_issues", None)
        fs.metadata.pop("last_outcome_kind", None)
        fs.metadata.pop("last_exception_class", None)
        session.accepted_fragment_hashes[item_id] = record.fragment_hash
        session.accepted_fragments[item_id] = record
        # Backward-compatible store so older tooling can still introspect.
        session.metadata.setdefault("_accepted_fragments", {})[item_id] = {
            "universe": record.universe,
        }
        _save_session(state, session)

    # -----------------------------------------------------------------
    # Merge phase (pure Python, structured).
    # -----------------------------------------------------------------
    accepted_records: dict[str, AcceptedFragmentRecord] = dict(session.accepted_fragments)
    accepted_fragments: list[UniverseDefinitionFragment] = [
        UniverseDefinitionFragment(
            universe_id=uid,
            universe=record.universe,
            fragment_hash=record.fragment_hash,
            manifest_contract_hash=record.manifest_contract_hash,
        )
        for uid, record in accepted_records.items()
    ]

    merge_result: UniverseMergeResult = _attempt_merge_with_replay(
        session=session,
        manifest=manifest,
        accepted_fragments=accepted_fragments,
        accepted_records=accepted_records,
        known_material_ids=known_material_ids,
        material_roles_by_id=material_roles_by_id,
        material_source_variants_by_id=material_source_variants_by_id,
        requirement=requirement,
        material_summary=material_summary,
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
        # Build the top-level failure issue with structured metadata so
        # the existing retry owner policy keeps routing to ``universes``.
        top_issue = _build_merge_failed_issue(merge_result)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type, issues=[top_issue],
        )

    # Validate merged patch using the existing UniversesPatch validator.
    ok, val_issues = validate_merged_patch(
        merge_result.merged_patch or {}, known_material_ids=known_material_ids,
    )
    if not ok:
        _save_session(state, session)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type,
            issues=[{
                "code": "patch_generation.merge_failed",
                "severity": "error",
                "message": "merged UniversesPatch failed standard validation",
                "metadata": {
                    "validation_issues": val_issues,
                    "merged_patch_hash": merge_result.merged_patch_hash,
                    "invalid_fragment_ids": merge_result.invalid_fragment_ids,
                    "manifest_id": merge_result.manifest_id,
                    "manifest_input_hash": merge_result.manifest_input_hash,
                },
            }],
        )

    # Stamp inventory metadata onto each generated universe so downstream
    # preflights can resolve geometry_profile_id and source_requirement_ids.
    # Without this, MU/inventory preflights cannot match universes to
    # inventory profiles (root cause of v9's 10 false-positive findings).
    manifest_by_uid = {item.universe_id: item for item in manifest.items}
    for universe in merge_result.merged_patch.get("universes", []):
        uid = universe.get("universe_id", "")
        m_item = manifest_by_uid.get(uid)
        if m_item is None:
            continue
        u_meta = dict(universe.get("metadata") or {})
        if m_item.base_path_component_profile_id:
            u_meta["geometry_profile_id"] = m_item.base_path_component_profile_id
        if m_item.source_requirement_ids:
            u_meta["source_requirement_ids"] = list(m_item.source_requirement_ids)
        for mk in ("component_kind", "profile_kind", "fuel_variant_id"):
            if mk in m_item.metadata:
                u_meta[mk] = m_item.metadata[mk]
        if m_item.fuel_variant_id:
            u_meta["fuel_variant_id"] = m_item.fuel_variant_id
        if m_item.localized_insert_requirement_id:
            u_meta["localized_insert_requirement_id"] = m_item.localized_insert_requirement_id
        localized_insert_ids = list(
            m_item.localized_insert_requirement_ids
            or m_item.metadata.get("localized_insert_requirement_ids", [])
            or []
        )
        if m_item.localized_insert_requirement_id and m_item.localized_insert_requirement_id not in localized_insert_ids:
            localized_insert_ids.append(m_item.localized_insert_requirement_id)
        if localized_insert_ids:
            u_meta["localized_insert_requirement_ids"] = sorted(
                set(str(item) for item in localized_insert_ids if item)
            )
        if u_meta:
            universe["metadata"] = u_meta

    alias_issues = materialize_localized_insert_universe_aliases(
        merge_result.merged_patch,
        facts_obj=facts_obj,
    )
    if alias_issues:
        _save_session(state, session)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type,
            issues=[{
                "code": "patch_generation.localized_insert_universe_alias_failed",
                "severity": "error",
                "message": "merged UniversesPatch could not materialize accepted Facts insert universe IDs",
                "metadata": {
                    "validation_issues": alias_issues,
                    "merged_patch_hash": merge_result.merged_patch_hash,
                    "manifest_id": merge_result.manifest_id,
                    "manifest_input_hash": merge_result.manifest_input_hash,
                },
            }],
        )

    ok_after_alias, alias_val_issues = validate_merged_patch(
        merge_result.merged_patch or {}, known_material_ids=known_material_ids,
    )
    if not ok_after_alias:
        _save_session(state, session)
        return PatchGenerationResult(
            ok=False, patch_type=patch_type,
            issues=[{
                "code": "patch_generation.localized_insert_universe_alias_invalid",
                "severity": "error",
                "message": "UniversesPatch failed validation after localized insert alias materialization",
                "metadata": {
                    "validation_issues": alias_val_issues,
                    "merged_patch_hash": merge_result.merged_patch_hash,
                    "manifest_id": merge_result.manifest_id,
                    "manifest_input_hash": merge_result.manifest_input_hash,
                },
            }],
        )

    # Create the authoritative PlanPatchEnvelope.
    import hashlib
    patch_hash = (
        merge_result.merged_patch_hash
        or hashlib.sha256(
            json.dumps(merge_result.merged_patch, sort_keys=True).encode()
        ).hexdigest()[:16]
    )
    envelope = PlanPatchEnvelope(
        patch_id=f"universes_fragmented_{patch_hash}",
        patch_type=patch_type,
        content=merge_result.merged_patch,
        source="llm",
        status="valid",
    )
    session.completed = True
    session.merged_patch_hash = patch_hash
    _save_session(state, session)

    return PatchGenerationResult(
        ok=True, patch_type=patch_type, envelope=envelope,
        parsed_patch=merge_result.merged_patch,
        attempts=[],
        issues=[],
    )


# ---------------------------------------------------------------------------
# Merge + targeted replay loop
# ---------------------------------------------------------------------------


def _attempt_merge_with_replay(
    *,
    session: LargePatchGenerationSession,
    manifest: UniverseManifest,
    accepted_fragments: list[UniverseDefinitionFragment],
    accepted_records: dict[str, AcceptedFragmentRecord],
    known_material_ids: set[str],
    material_roles_by_id: dict[str, str],
    material_source_variants_by_id: dict[str, str | None],
    requirement: str,
    material_summary: str,
    llm_client: Any,
    effective_max_tokens: int,
    explicit_max_tokens: int | None,
    max_fragment_attempts: int,
    max_total_llm_calls: int,
    max_merge_replays: int,
    state: PlanBuildState,
) -> UniverseMergeResult:
    """Try the merge; replay only fragment-scoped failures.

    Manifest and global merge failures fail closed.  Fragment-scoped
    failures replay the offending fragments within ``max_fragment_attempts``
    inside ``max_merge_replays`` rounds; accepted fragments that were not
    flagged stay untouched.
    """
    merge_round = 0
    last_result: UniverseMergeResult | None = None

    current_records = dict(accepted_records)
    current_fragments = list(accepted_fragments)

    while True:
        result = merge_universe_fragments_structured(
            manifest=manifest,
            fragments=current_fragments,
            known_material_ids=known_material_ids,
            known_material_roles_by_id=material_roles_by_id,
            qualification_records=current_records,
        )
        last_result = result
        session.merge_history.append({
            "round": merge_round,
            "ok": result.ok,
            "manifest_id": result.manifest_id,
            "manifest_input_hash": result.manifest_input_hash,
            "merged_patch_hash": result.merged_patch_hash,
            "issue_codes": [issue.code for issue in result.issues],
            "invalid_fragment_ids": list(result.invalid_fragment_ids),
            "issues": [issue.model_dump(mode="json") for issue in result.issues],
        })

        if result.ok:
            return result

        # If any issue is manifest or global scope → fail closed.
        blocking_scopes = {
            issue.retry_scope for issue in result.issues
            if issue.severity == "error" and issue.retry_scope in ("manifest", "global")
        }
        if blocking_scopes:
            return result

        # Otherwise, we have only fragment-scoped issues.  Check budget.
        if merge_round >= max_merge_replays:
            return result

        invalid_ids = list(result.invalid_fragment_ids)
        if not invalid_ids:
            # No specific fragment to replay; cannot recover.
            return result

        manifest_by_id = {item.universe_id: item for item in manifest.items}
        # Downgrade only the invalid fragments and regenerate them.
        any_replayed = False
        for uid in invalid_ids:
            item = manifest_by_id.get(uid)
            if item is None:
                continue
            # Clear the invalid record so it is regenerated.
            current_records.pop(uid, None)
            current_fragments = [f for f in current_fragments if f.universe_id != uid]
            _downgrade_fragment(
                session, uid,
                reason=f"merge_fragment_scoped_failure_round_{merge_round}",
            )

            prior_failures: list[str] = []
            new_record: AcceptedFragmentRecord | None = None
            for frag_attempt in range(max_fragment_attempts):
                if session.llm_call_count >= max_total_llm_calls:
                    break
                rec, qualification, outcome = _generate_and_qualify_one_fragment(
                    item=item, llm_client=llm_client, session=session,
                    requirement=requirement, material_summary=material_summary,
                    known_material_ids=known_material_ids,
                    material_roles_by_id=material_roles_by_id,
                    material_source_variants_by_id=material_source_variants_by_id,
                    effective_max_tokens=effective_max_tokens,
                    explicit_max_tokens=explicit_max_tokens,
                    prior_failures=prior_failures,
                    attempt_index=frag_attempt,
                )
                if rec is not None:
                    new_record = rec
                    break
            if new_record is None:
                # Could not recover this fragment within budget → fail closed
                # with the structured reason.
                return result
            current_records[uid] = new_record
            current_fragments.append(UniverseDefinitionFragment(
                universe_id=uid,
                universe=new_record.universe,
                fragment_hash=new_record.fragment_hash,
                manifest_contract_hash=new_record.manifest_contract_hash,
            ))
            # Persist the recovered record.
            fs = next((fs for fs in session.fragment_statuses if fs.universe_id == uid), None)
            if fs is None:
                fs = FragmentStatus(universe_id=uid)
                session.fragment_statuses.append(fs)
            fs.status = "accepted"
            fs.fragment_hash = new_record.fragment_hash
            fs.manifest_contract_hash = new_record.manifest_contract_hash
            fs.qualification_status = "passed"
            fs.qualification_issues = list(new_record.qualification_issues)
            fs.accepted_at_attempt = new_record.accepted_at_attempt
            session.accepted_fragments[uid] = new_record
            session.accepted_fragment_hashes[uid] = new_record.fragment_hash
            session.metadata.setdefault("_accepted_fragments", {})[uid] = {
                "universe": new_record.universe,
            }
            any_replayed = True

        if not any_replayed:
            return result

        merge_round += 1
        _save_session(state, session)

    # Unreachable: the loop above always returns.
    return last_result  # pragma: no cover


def _build_merge_failed_issue(merge_result: UniverseMergeResult) -> dict[str, Any]:
    """Build the top-level failure issue carrying structured metadata.

    Keeps ``code == "patch_generation.merge_failed"`` so the existing
    retry owner policy routes the request to the ``universes`` owner.
    """
    affected_paths = sorted({
        issue.json_path for issue in merge_result.issues if issue.json_path
    })
    fragment_hashes = sorted({
        issue.fragment_hash for issue in merge_result.issues if issue.fragment_hash
    })
    severity_scopes = sorted({
        issue.retry_scope for issue in merge_result.issues
        if issue.severity == "error"
    })
    return {
        "code": "patch_generation.merge_failed",
        "severity": "error",
        "message": (
            f"universe fragment merge failed: "
            f"{len(merge_result.issues)} issue(s), "
            f"invalid_fragment_ids={merge_result.invalid_fragment_ids}, "
            f"scopes={severity_scopes}"
        ),
        "metadata": {
            "merge_issue_codes": [issue.code for issue in merge_result.issues],
            "invalid_fragment_ids": merge_result.invalid_fragment_ids,
            "required_ids": [
                item.universe_id for item in (merge_result.manifest_id and [] or [])
            ],
            "affected_json_paths": affected_paths,
            "fragment_hashes": fragment_hashes,
            "manifest_id": merge_result.manifest_id,
            "manifest_input_hash": merge_result.manifest_input_hash,
            "retry_scopes": severity_scopes,
            "issues": [issue.model_dump(mode="json") for issue in merge_result.issues],
        },
    }


__all__ = ["generate_universes_patch"]
