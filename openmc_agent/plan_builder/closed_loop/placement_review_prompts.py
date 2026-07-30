"""Strict prompt contract for the independent Placement Critic."""

from __future__ import annotations

import json

from .models import PlacementEvidencePack, PlacementReviewModelOutput


def build_placement_review_prompt(pack: PlacementEvidencePack) -> str:
    return """You are an Independent Placement Contract Critic, not a patch generator, renderer, or supervisor.

Review the accepted Facts placement contract against the supplied placement patches:
requirement -> universe/profile -> intent -> assembly scope -> core-layout instance.
Only report semantic binding omissions or conflicts not already mechanically computed. Do not recalculate coordinate counts, multiplicity, profile references, universe existence, duplicate coordinates, host subsets, anchors, or control-state string equality: Python has supplied those deterministic results.
When a row supplies actual_profile_resolution, treat profile relative_z_* values as profile-frame coordinates. Compare intent z_min_cm/z_max_cm with resolved_profile_interval_cm only after anchor translation, and do not flag an intent envelope/profile extent difference unless the accepted Facts contract explicitly requires the profile to fill that envelope.
For source_scope="single_assembly", actual_assembly_type_ids=["single_assembly"] is the implicit sole assembly scope and is already accepted by deterministic binding when static_binding_status="pass"; do not report it as an assembly-type token mismatch.
Use only supplied evidence references. Do not use external reactor knowledge, infer from names, modify patches, output actions, RFC6902, Markdown, or reasoning.
An error needs accepted-facts or deterministic evidence. Ambiguity alone may require human confirmation and cannot be repairable.

Return only one JSON object conforming exactly to this schema:
""" + json.dumps(PlacementReviewModelOutput.model_json_schema(), ensure_ascii=False, sort_keys=True) + "\nINPUT:\n" + json.dumps(pack.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)


def build_placement_review_schema_retry_prompt(pack: PlacementEvidencePack, error: str, raw_output: str) -> str:
    return build_placement_review_prompt(pack) + "\nThe previous output was invalid. Return exactly one schema-valid JSON object, no prose.\nSCHEMA_ERROR:\n" + error + "\nPREVIOUS_OUTPUT:\n" + raw_output
