"""Strict JSON Patch prompt for the facts-only revision role."""

from __future__ import annotations

import json
from typing import Any


def _target_schema_hint() -> dict[str, Any]:
    """Return the authoritative FactsPatch target schema.

    We project :class:`openmc_agent.plan_builder.patches.FactsPatch` to its
    JSON schema and trim ``$defs`` / titles to keep the prompt compact
    (~6KB instead of ~10KB).  This is the source of truth — hand-writing
    a hint risks going stale when patch fields change (e.g. when a field
    is a typed object instead of a plain string, the LLM must know the
    exact nested shape).
    """
    from openmc_agent.plan_builder.patches import FactsPatch

    schema = FactsPatch.model_json_schema()
    # Drop noisy metadata that bloats the prompt without helping the model.
    for key in ("title", "description", "$defs"):
        schema.pop(key, None)
    defs = FactsPatch.model_json_schema().get("$defs", {})
    for def_name, def_value in list(defs.items()):
        for key in ("title",):
            def_value.pop(key, None)
    if defs:
        schema["$defs"] = defs
    return schema


# Worked example that shows the LLM the EXACT expected response shape.
# This is reactor-neutral: the values are placeholders, not benchmark data.
_WORKED_EXAMPLE = """{
  "proposal_id": "facts_revision_001",
  "confidence": 0.9,
  "rationale": "Corrected model scope and missing fields based on evidence.",
  "operations": [
    {"op": "replace", "path": "/model_scope", "value": "multi_assembly_core"},
    {"op": "replace", "path": "/assembly_count", "value": 9},
    {"op": "replace", "path": "/has_spacer_grids", "value": true}
  ],
  "resolved_finding_ids": []
}"""


def build_facts_revision_prompt(*, facts_patch: dict, findings: list[dict], evidence: list[dict], allowed_paths: list[str], confirmed_facts: dict) -> str:
    # Phase 8B Step 3: list the required coverage fields explicitly so the
    # LLM does not omit them.  This is reactor-neutral — the fields are
    # the same structural slots every FactsPatch must fill, regardless of
    # reactor type.  Empty-but-required fields are flagged so the LLM
    # knows it MUST provide operations for them.
    required_fields = _required_coverage_fields_with_status(facts_patch)
    return (
        "You are the Facts Revision Agent. Your task is to produce a single JSON "
        "FactsRevisionProposal object containing a list of RFC6902 JSON Patch "
        "operations (add / replace / remove) that fix the supplied findings.\n\n"
        "CRITICAL: Your output is a JSON Patch proposal, NOT a FactsPatch. "
        "You are NOT producing the corrected FactsPatch itself — you are "
        "producing a list of edit operations. Each operation has the shape "
        '{"op": "replace", "path": "/field_name", "value": <new value>}. '
        "Do not output the underlying FactsPatch shape. "
        "Do not output the input payload back.\n\n"
        "CLOSURE RULE: address every supplied blocking finding in this round. "
        "List every finding ID you resolve in resolved_finding_ids. Do not claim "
        "a finding is resolved unless the operations make the source-backed change. "
        "Never invent a material, geometry, or reactor fact when the supplied "
        "evidence does not support it. Do not invent FactsPatch fields that "
        "are not in the schema, such as /operating_conditions; source-backed "
        "operating-state values belong in /source_notes unless a schema field "
        "explicitly exists below. If a finding asks for a source-backed fact "
        "that has no dedicated FactsPatch field, resolve it by adding or "
        "replacing a compact /source_notes entry, /assumptions entry, or an "
        "existing nested source_note field. Preserve existing notes unless "
        "the evidence directly contradicts them. Prefer explicit key=value or "
        "semicolon-separated note text for downstream recovery (for example: "
        "density/name/value/unit/source; composition/name/isotope/fraction; "
        "axial interval/name/z_min/z_max/unit; repeated structure/count/role). "
        "When multiple supplied findings cover the same source table or section, "
        "repair the whole source-backed topic in one proposal rather than only "
        "the first mentioned scalar. If evidence does not support a requested "
        "fact, add a specific /missing_facts entry instead of leaving the gap "
        "implicit.\n\n"
        "The output object must have these fields: proposal_id (string), "
        "confidence (float 0..1), rationale (string), operations (array of "
        "{op, path, value} objects), and resolved_finding_ids (array of "
        "strings, may be empty).\n\n"
        "REQUIRED COVERAGE FIELDS — your operations MUST address every field "
        "listed as MISSING or EMPTY below.  An incomplete repair that leaves "
        "a required field empty will be rejected automatically.\n"
        + json.dumps(required_fields, ensure_ascii=False)
        + "\n\nWorked example (placeholder values, NOT a real answer):\n"
        + _WORKED_EXAMPLE
        + "\n\nThe target FactsPatch JSON schema is provided ONLY so you "
        "know which values are valid at each path (use EXACT enum string "
        "values, EXACT tuple shapes for tuple fields, and EXACT object "
        "shapes for nested-object fields). You still produce operations, "
        "not a FactsPatch object:\n"
        + json.dumps(_target_schema_hint(), ensure_ascii=False)
        + "\n\nInput payload (do NOT echo this back):\n"
        + json.dumps({"facts_patch": facts_patch, "findings": findings, "evidence": evidence,
                      "allowed_paths": allowed_paths, "confirmed_facts": confirmed_facts}, ensure_ascii=False)
    )


def _required_coverage_fields_with_status(facts_patch: dict) -> list[dict]:
    """Return per-field status so the LLM sees which fields are empty."""

    from .facts_revision import _is_empty_coverage_value, required_coverage_paths_for_patch

    statuses: list[dict] = []
    for path in required_coverage_paths_for_patch(facts_patch):
        key = path.lstrip("/")
        value = facts_patch.get(key)
        is_empty = _is_empty_coverage_value(value)
        statuses.append({
            "path": path,
            "status": "MISSING" if is_empty else "present",
            "current_value": value,
        })
    return statuses
