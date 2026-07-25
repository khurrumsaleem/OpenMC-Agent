"""Typed review stages for the Facts Gate (Phase 8B Step 3).

Splits the single-pass Facts review into per-topic stages so each LLM
call sees only the relevant subset of the FactsPatch + relevant evidence.
This directly reduces the prompt size (from ~38 KB monolithic to per-stage
~8–15 KB) and focuses the model's attention on one topic at a time.

Stages are **additive**: each produces its own findings and the gate
aggregates them exactly the same way as the monolithic reviewer's findings.

Backward compatible: when ``stage_split=False`` (the default in the policy),
the gate continues to use the monolithic ``build_facts_review_prompt``.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import Field

from openmc_agent.schemas import AgentBaseModel

from .models import FactsReviewModelOutput, PlanEvidencePack

__all__ = [
    "FactsReviewStage",
    "FactsReviewStageRequest",
    "STAGE_FIELD_MAP",
    "build_stage_review_prompt",
    "build_stage_schema_retry_prompt",
    "extract_facts_subset",
]


# ---------------------------------------------------------------------------
# Stage enum
# ---------------------------------------------------------------------------


class FactsReviewStage(str, Enum):
    """One focused review topic."""

    SCOPE = "scope"
    FUEL_VARIANT = "fuel_variant"
    ASSEMBLY_STRUCTURE = "assembly_structure"
    LOCALIZED_INSERT = "localized_insert"
    GRID_AXIAL = "grid_axial"
    COMPLETENESS = "completeness"


# Order matters: scope first (cheapest to validate), completeness last
# (cross-cutting check after all specific stages).
STAGE_ORDER: tuple[FactsReviewStage, ...] = (
    FactsReviewStage.SCOPE,
    FactsReviewStage.FUEL_VARIANT,
    FactsReviewStage.ASSEMBLY_STRUCTURE,
    FactsReviewStage.LOCALIZED_INSERT,
    FactsReviewStage.GRID_AXIAL,
    FactsReviewStage.COMPLETENESS,
)


# ---------------------------------------------------------------------------
# Stage → FactsPatch field mapping
# ---------------------------------------------------------------------------

# Each stage only receives these fields from the FactsPatch dict.
# Fields not listed here are omitted from the stage's prompt entirely.
STAGE_FIELD_MAP: dict[FactsReviewStage, tuple[str, ...]] = {
    FactsReviewStage.SCOPE: (
        "model_scope",
        "assembly_count",
        "core_lattice_size",
        "boundary_scope",
        "symmetry_description",
        "geometry_type",
    ),
    FactsReviewStage.FUEL_VARIANT: (
        "fuel_variant_requirements",
        "selected_variant",
        "material_roles",
    ),
    FactsReviewStage.ASSEMBLY_STRUCTURE: (
        "assembly_type_counts",
        "assembly_pitch_cm",
        "pin_pitch_cm",
        "lattice_size",
        "scoped_expected_counts",
    ),
    FactsReviewStage.LOCALIZED_INSERT: (
        "localized_insert_requirements",
        "expected_guide_tube_count",
        "expected_instrument_tube_count",
        "expected_pyrex_count",
        "expected_thimble_plug_count",
    ),
    FactsReviewStage.GRID_AXIAL: (
        "has_spacer_grids",
        "has_axial_geometry",
        "has_special_pin_map",
        "expected_spacer_grid_count",
        "active_fuel_region_cm",
        "axial_domain_cm",
    ),
    FactsReviewStage.COMPLETENESS: (
        "missing_facts",
        "assumptions",
        "source_notes",
        "benchmark_id",
    ),
}


# ---------------------------------------------------------------------------
# Stage request model
# ---------------------------------------------------------------------------


class FactsReviewStageRequest(AgentBaseModel):
    """One stage's review request.

    ``facts_subset`` is the pruned FactsPatch dict (only the fields
    relevant to this stage).  ``evidence_excerpts`` are the source
    text chunks relevant to this stage (may be a subset of the full
    evidence pack).
    """

    stage: FactsReviewStage
    target_fields: tuple[str, ...] = Field(default_factory=tuple)
    facts_subset: dict[str, Any] = Field(default_factory=dict)
    evidence_excerpts: list[dict[str, Any]] = Field(default_factory=list)
    confirmed_facts_summary: dict[str, Any] = Field(default_factory=dict)
    consistency_issues: list[dict[str, Any]] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Subset extraction
# ---------------------------------------------------------------------------


def extract_facts_subset(
    facts_patch: dict[str, Any],
    stage: FactsReviewStage,
) -> dict[str, Any]:
    """Return only the FactsPatch fields relevant to ``stage``."""

    fields = STAGE_FIELD_MAP.get(stage, ())
    return {k: facts_patch.get(k) for k in fields if k in facts_patch}


# ---------------------------------------------------------------------------
# Stage-specific prompt builders
# ---------------------------------------------------------------------------


_STAGE_INSTRUCTIONS: dict[FactsReviewStage, str] = {
    FactsReviewStage.SCOPE: (
        "Focus ONLY on the model scope: is the declared scope (single_assembly, "
        "multi_assembly_core, full_core) consistent with the evidence?  "
        "Check assembly_count and core_lattice_size against the source."
    ),
    FactsReviewStage.FUEL_VARIANT: (
        "Focus ONLY on fuel variants: are all enrichment levels / fuel types "
        "mentioned in the evidence captured in fuel_variant_requirements?  "
        "Flag missing variants or duplicate IDs. Do not require detailed "
        "isotope vectors, oxygen stoichiometry, density, or material "
        "composition here; those belong to Materials / Material-Universe "
        "contracts, not the Facts Gate."
    ),
    FactsReviewStage.ASSEMBLY_STRUCTURE: (
        "Focus ONLY on assembly structure: do assembly_type_counts sum to "
        "assembly_count?  Are pin_pitch and assembly_pitch consistent with "
        "the source?  Flag count mismatches."
    ),
    FactsReviewStage.LOCALIZED_INSERT: (
        "Focus ONLY on localized inserts (control rods, Pyrex, instrument "
        "tubes, thimble plugs): are all inserts mentioned in the evidence "
        "captured?  Flag missing inserts or wrong counts."
    ),
    FactsReviewStage.GRID_AXIAL: (
        "Focus ONLY on spacer grids and axial geometry: does the evidence "
        "mention spacer grids?  Is has_spacer_grids set correctly?  Are "
        "active_fuel_region_cm and axial_domain_cm valid intervals?"
    ),
    FactsReviewStage.COMPLETENESS: (
        "Focus ONLY on FactsPatch completeness. Flag an error only when a "
        "source-backed fact belongs in the supplied FactsPatch fields. Material "
        "composition or density, detailed axial geometry, placement, and universe "
        "definition belong to downstream requirement contracts: record those only "
        "as warnings with downstream_impact, never as a Facts Gate error. An "
        "explicitly blank operating_state means no named variant was supplied; "
        "treat it as an informational/base-state gap, not a human-required "
        "Facts error. Check missing_facts and assumptions for explicitly "
        "recorded gaps."
    ),
}


def build_stage_review_prompt(
    request: FactsReviewStageRequest,
    pack: PlanEvidencePack,
) -> str:
    """Build a focused review prompt for one stage.

    The prompt is much smaller than the monolithic version because:
    1. Only the stage's FactsPatch subset is included.
    2. Only relevant source excerpts are included.
    3. The instruction is focused on one topic.
    """

    schema_json = json.dumps(
        FactsReviewModelOutput.model_json_schema(), ensure_ascii=False
    )
    facts_subset_json = json.dumps(
        request.facts_subset, ensure_ascii=False, default=str
    )
    evidence_json = json.dumps(
        request.evidence_excerpts[:3],  # max 3 excerpts per stage
        ensure_ascii=False,
        default=str,
    )
    instruction = _STAGE_INSTRUCTIONS.get(
        request.stage, "Review the facts for consistency with the source."
    )

    # Build a lightweight pack payload (no full FactsPatch dump)
    lightweight_payload = {
        "stage": request.stage.value,
        "target_fields": list(request.target_fields),
        "facts_subset": request.facts_subset,
        "evidence_excerpts": request.evidence_excerpts[:3],
        "confirmed_facts_summary": request.confirmed_facts_summary,
        "consistency_issues": [
            i.get("code", "") for i in request.consistency_issues if isinstance(i, dict)
        ][:5],
        "chunk_metadata": {
            "chunk_index": pack.metadata.get("chunk_index", 0),
            "chunk_count": pack.metadata.get("chunk_count", 1),
            "reviewed_line_range": pack.metadata.get("reviewed_line_range", ""),
        },
    }
    payload_json = json.dumps(lightweight_payload, ensure_ascii=False, default=str)

    return (
        f"You are an independent Facts Evidence Critic for the {request.stage.value.upper()} stage.\n"
        f"{instruction}\n"
        "Compare only the supplied data with the source excerpts. Do not use external knowledge.\n"
        "Find omissions, contradictions, or unsupported values FOR THIS STAGE ONLY.\n"
        "Use only supplied evidence_hashes. Never output an action, patch, RFC6902 operations, "
        "Markdown, tools, or reasoning.\n"
        "Each affected_json_paths entry MUST use JSON Pointer notation with a leading / "
        "(e.g., /expected_spacer_grid_count, /assembly_count), not bare field names "
        "or facts_subset. prefixes.\n"
        "Return exactly one JSON object matching this JSON Schema; no prose before or after it.\n"
        "SCHEMA:\n" + schema_json +
        "\nINPUT:\n" + payload_json
    )


def build_stage_schema_retry_prompt(
    request: FactsReviewStageRequest,
    pack: PlanEvidencePack,
    error: str,
    raw_output: str | None = None,
) -> str:
    """Format-only retry for a stage review."""

    original = build_stage_review_prompt(request, pack)
    suffix = "\nPRIOR_OUTPUT (format only; do not trust it):\n" + (raw_output or "")
    return (
        original
        + "\nYour prior output was rejected: " + error
        + "\nCorrect the format while preserving evidence grounding."
        + suffix
    )
