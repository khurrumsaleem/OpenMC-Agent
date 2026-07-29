"""Incremental planning mode decision (Phase 0).

Decides whether a requirement should use the future incremental (patch-based)
plan builder or stay on the existing monolithic LLM full-plan generation path.

Design constraints
------------------
* **No benchmark facts.**  Only generic reactor vocabulary (axial layers, spacer
  grids, special pin types, lattice size signals, ...) is pattern-matched.
* **Backward compatible.**  Simple 2D assembly requirements always stay on the
  monolithic path; nothing changes unless the requirement or retry history
  carries strong incremental triggers.
* **No side effects.**  The function is pure: it reads its arguments and returns
  a :class:`PlanningModeDecision`.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import Field

from openmc_agent.assembly3d_guard import detect_assembly_3d_features
from openmc_agent.schemas import AgentBaseModel


# ---------------------------------------------------------------------------
# Trigger vocabulary -- generic reactor engineering terms only.
# ---------------------------------------------------------------------------

_SPECIAL_PIN_MAP_TERMS: tuple[str, ...] = (
    "pyrex",
    "pyrex rod",
    "thimble plug",
    "burnable poison",
    "ba rod",
    "gadolinia",
    "ifba",
    "waba",
    "guide tube",
    "instrument tube",
    "control rod guide",
    "wet annular burnable absorber",
    # Chinese
    "可燃毒物",
    "导向管",
    "测量管",
    " instrumentation tube",
    "pyrex 棒",
    "塞管",
)

_BENCHMARK_VARIANT_TERMS: tuple[str, ...] = (
    "vera",
    "vera3",
    "vera 3",
    "3a",
    "3b",
    "variant",
    "c5g7",
    "c5g7-mox",
    "benchmark variant",
    # Chinese
    "基准",
    "变体",
)

_MULTI_UNIVERSE_TERMS: tuple[str, ...] = (
    "multiple universe",
    "multiple pin type",
    "multiple pin universe",
    "different pin type",
    "several universe",
    "fuel pin universe",
    "guide tube universe",
    "instrument tube universe",
    "pyrex universe",
    "plug universe",
)

_LARGE_LATTICE_PATTERN: tuple[re.Pattern[str], ...] = (
    # Match "NNxNN" lattice dimensions, but NOT the fractional part of a
    # decimal dimension like "21.50x21.50" (where "50x21" would otherwise be
    # misread as a 50-dim lattice). The lookarounds require the matched digit
    # groups to be bounded by non-digit, non-dot characters.
    re.compile(r"(?<![0-9.])(\d{2,})\s*[x×]\s*(\d{2,})(?![0-9.])", re.IGNORECASE),
)

_LARGE_LATTICE_THRESHOLD: int = 20

_LARGE_JSON_OUTPUT_THRESHOLD: int = 12_000

_JSON_PARSE_ERROR_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"expecting\s+['\"]?,?['\"]?\s*delimiter", re.IGNORECASE),
    re.compile(r"could not parse model response", re.IGNORECASE),
    re.compile(r"unterminated string", re.IGNORECASE),
    re.compile(r"json\.jsondecodeerror", re.IGNORECASE),
    re.compile(r"expecting (?:value|property|comma)", re.IGNORECASE),
)

_REPAIR_LOST_AXIAL_CODE: str = "assembly3d.axial_layers_required"

# Event / trigger codes (stable strings for downstream analytics).
TRIGGER_FEATURE_3D_AXIAL: str = "feature.3d_axial_geometry"
TRIGGER_FEATURE_SPACER_GRID: str = "feature.spacer_grid"
TRIGGER_FEATURE_SPECIAL_PIN_MAP: str = "feature.special_pin_map"
TRIGGER_FEATURE_MULTIPLE_VARIANTS: str = "feature.multiple_variants"
TRIGGER_FEATURE_LARGE_LATTICE: str = "feature.large_lattice"
TRIGGER_FEATURE_MULTI_ASSEMBLY: str = "feature.multi_assembly_core"
TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR: str = "history.large_json_parse_error"
TRIGGER_HISTORY_REPAIR_LOST_AXIAL: str = "history.repair_lost_axial_layers"
TRIGGER_HISTORY_REPEATED_AXIAL_CONTRACT: str = "history.repeated_axial_contract_violation"
TRIGGER_OVERRIDE_FORCE_INCREMENTAL: str = "override.force_incremental"


class PlanningModeDecision(AgentBaseModel):
    """Result of :func:`should_use_incremental_planning`.

    Attributes
    ----------
    mode
        ``"monolithic"`` keeps the existing single-LLM-call path; ``"incremental"``
        signals the future patch-based builder should take over.
    reasons
        Human-readable explanation lines (may be shown in prompts / transcripts).
    triggers
        Stable trigger codes (see ``TRIGGER_*`` constants).  Empty for a plain
        monolithic decision.
    confidence
        Heuristic confidence in the decision (0.0–1.0).
    feature_summary
        Structured summary of detected features for downstream consumption.
    """

    mode: Literal["monolithic", "incremental"] = "monolithic"
    reasons: list[str] = Field(default_factory=list)
    triggers: list[str] = Field(default_factory=list)
    confidence: float = 1.0
    feature_summary: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal detectors
# ---------------------------------------------------------------------------


def _detect_special_pin_map(text_lower: str) -> list[str]:
    return [t for t in _SPECIAL_PIN_MAP_TERMS if t in text_lower]


def _detect_benchmark_variants(text_lower: str) -> list[str]:
    return [t for t in _BENCHMARK_VARIANT_TERMS if t in text_lower]


def _detect_multiple_universes(text_lower: str) -> list[str]:
    return [t for t in _MULTI_UNIVERSE_TERMS if t in text_lower]


def _detect_large_lattice(text: str) -> int | None:
    """Return the largest lattice dimension found, or None."""
    max_dim = 0
    for pattern in _LARGE_LATTICE_PATTERN:
        for match in pattern.finditer(text):
            try:
                d1 = int(match.group(1))
                d2 = int(match.group(2))
                max_dim = max(max_dim, d1, d2)
            except (ValueError, IndexError):
                continue
    return max_dim if max_dim >= _LARGE_LATTICE_THRESHOLD else None


# Reactor-neutral cues that a requirement describes a core composed of
# MULTIPLE assemblies (not a single-assembly model). Detecting this early is
# what schedules the assembly_catalog + core_layout patches; without it a
# multi-assembly core is built on the single-assembly task order and the
# assembler later fails with assembly.missing_patch.
_MULTI_ASSEMBLY_KEYWORDS: tuple[str, ...] = (
    "multi-assembly", "multi assembly", "multiple assembl",
    "full core", "full-core", "core lattice", "core layout",
    "assembly array", "assembly lattice", "assemblies arranged",
    "多组件", "多个组件", "组件阵列", "全堆", "整堆", "全堆芯",
)

# These are source-feature cues, not reactor templates: they only preserve an
# explicit localized/movable-object contract for the Facts stage.  No geometry,
# coordinate, material, or benchmark value is inferred here.
_LOCALIZED_INSERT_TERMS: tuple[str, ...] = (
    "control rod", "control element", "movable absorber", "absorber insert",
    "rod cluster", "rcca", "局部插入", "控制棒", "吸收体",
)
# "N assemblies" (N >= 2) — English count.
_MULTI_ASSEMBLY_COUNT_RE = re.compile(
    r"(\d+)\s*(?:assemblies|fuel\s*assemblies|assembly\s*types)",
    re.IGNORECASE,
)
# Chinese count of assemblies, e.g. "九个燃料组件" / "3 个组件" (>=2).
_CN_ASSEMBLY_COUNT_RE = re.compile(r"([二三四五六七八九2-9])\s*个\s*(?:燃料)?组件")


def _detect_multi_assembly_core(text: str) -> bool:
    """Return True if the requirement describes a multi-assembly core.

    Reactor-neutral: matches generic multi-assembly / core-lattice vocabulary
    (a core assembled from multiple fuel assemblies), never a single reactor
    type. A single-assembly model (one pin lattice, no core-level placement)
    returns False so it stays on the simpler single-assembly task order.
    """
    low = text.lower()
    if any(k in low for k in _MULTI_ASSEMBLY_KEYWORDS):
        return True
    for m in _MULTI_ASSEMBLY_COUNT_RE.finditer(low):
        try:
            if int(m.group(1)) >= 2:
                return True
        except ValueError:
            continue
    return bool(_CN_ASSEMBLY_COUNT_RE.search(text))


def _analyze_retry_history(
    retry_history: list[Any] | None,
) -> tuple[list[str], list[str]]:
    """Extract history-based triggers and reasons.

    Returns ``(triggers, reasons)``.
    """
    triggers: list[str] = []
    reasons: list[str] = []
    if not retry_history:
        return triggers, reasons

    combined_errors: list[str] = []
    axial_contract_count = 0
    repair_format_count = 0

    for entry in retry_history:
        if not isinstance(entry, dict):
            continue
        errors = entry.get("validation_errors") or []
        if isinstance(errors, str):
            errors = [errors]
        for err in errors:
            if isinstance(err, str):
                combined_errors.append(err)

        # Count repair_format phases
        fix_suggestion = entry.get("fix_suggestion") or ""
        if isinstance(fix_suggestion, str) and "repair" in fix_suggestion.lower():
            repair_format_count += 1

        # Count assembly3d.axial_layers_required occurrences
        plan_data = entry.get("plan")
        if isinstance(plan_data, dict):
            cm = plan_data.get("complex_model") or {}
            core = (cm.get("core") or {}) if isinstance(cm, dict) else {}
            has_axial = bool(core.get("axial_layers")) or bool(core.get("axial_overlays"))
            if not has_axial and any(
                _REPAIR_LOST_AXIAL_CODE in str(e) for e in errors
            ):
                axial_contract_count += 1

    error_blob = "\n".join(combined_errors)

    # Large JSON parse error
    has_json_parse_error = any(
        pattern.search(error_blob) for pattern in _JSON_PARSE_ERROR_PATTERNS
    )
    if has_json_parse_error:
        triggers.append(TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR)
        reasons.append(
            "retry history contains JSON parse errors on large plan output"
        )

    # Repair lost axial layers
    if axial_contract_count > 0:
        triggers.append(TRIGGER_HISTORY_REPAIR_LOST_AXIAL)
        reasons.append(
            f"retry history shows {axial_contract_count} round(s) where "
            "axial_layers were lost after format repair"
        )
    elif axial_contract_count == 0 and repair_format_count >= 2 and any(
        _REPAIR_LOST_AXIAL_CODE in err for err in combined_errors
    ):
        triggers.append(TRIGGER_HISTORY_REPEATED_AXIAL_CONTRACT_VIOLATION)
        reasons.append(
            "retry history shows repeated assembly3d.axial_layers_required "
            "after format repair attempts"
        )

    return triggers, reasons


def _analyze_plan_context(
    plan_context: dict[str, Any] | None,
) -> tuple[list[str], list[str], bool]:
    """Extract context-based triggers.

    Returns ``(triggers, reasons, force_override)`` where ``force_override``
    is True when the context forces a specific mode.
    """
    triggers: list[str] = []
    reasons: list[str] = []
    force_incremental = False

    if not plan_context:
        return triggers, reasons, False

    if plan_context.get("force_incremental_planning"):
        triggers.append(TRIGGER_OVERRIDE_FORCE_INCREMENTAL)
        reasons.append("plan_context explicitly forces incremental planning")
        force_incremental = True

    # Large prior output
    raw_output_length = plan_context.get("raw_output_length")
    if isinstance(raw_output_length, int) and raw_output_length > _LARGE_JSON_OUTPUT_THRESHOLD:
        if TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR not in triggers:
            triggers.append(TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR)
            reasons.append(
                f"prior LLM raw output was {raw_output_length} chars "
                f"(>{_LARGE_JSON_OUTPUT_THRESHOLD}), exceeding reliable threshold"
            )

    return triggers, reasons, force_incremental


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def should_use_incremental_planning(
    requirement: str,
    feature_flags: Any | None = None,
    retry_history: list[Any] | None = None,
    plan_context: dict[str, Any] | None = None,
) -> PlanningModeDecision:
    """Decide whether to use incremental or monolithic plan generation.

    Parameters
    ----------
    requirement
        The user/benchmark requirement text.
    feature_flags
        Optional pre-computed feature flags (e.g. from
        :func:`~openmc_agent.assembly3d_guard.detect_assembly_3d_features`).
        If ``None``, the function calls the detector internally.
    retry_history
        Optional retry history from a prior monolithic attempt.  Each entry is
        expected to be a dict with ``validation_errors``, ``plan``, etc.
    plan_context
        Optional context dict with keys like ``force_incremental_planning``,
        ``force_monolithic_planning``, ``raw_output_length``, etc.

    Returns
    -------
    PlanningModeDecision
        The planning mode decision with triggers, reasons, and feature summary.
    """
    # --- Feature detection ---------------------------------------------------
    if feature_flags is not None and hasattr(feature_flags, "has_axial_geometry"):
        flags = feature_flags
    else:
        flags = detect_assembly_3d_features(requirement)

    text_lower = requirement.lower()
    triggers: list[str] = []
    reasons: list[str] = []
    feature_summary: dict[str, Any] = {
        "has_axial_geometry": flags.has_axial_geometry,
        "has_spacer_grid": flags.has_spacer_grid,
        "has_explicit_z_ranges": flags.has_explicit_z_ranges,
        "has_axial_components": flags.has_axial_components,
    }

    # 1. 3D axial geometry
    if flags.has_axial_geometry:
        triggers.append(TRIGGER_FEATURE_3D_AXIAL)
        reasons.append(
            "requirement describes 3D axial geometry "
            f"({', '.join(flags.matched_terms[:5]) or 'axial signals'})"
        )
        feature_summary["axial_matched_terms"] = list(flags.matched_terms)

    # 2. Spacer grids
    if flags.has_spacer_grid:
        triggers.append(TRIGGER_FEATURE_SPACER_GRID)
        reasons.append("requirement mentions spacer/support grids")

    # 3. Special pin map / variants
    special_pins = _detect_special_pin_map(text_lower)
    feature_summary["has_special_pin_map"] = bool(special_pins)
    if special_pins:
        triggers.append(TRIGGER_FEATURE_SPECIAL_PIN_MAP)
        reasons.append(
            f"requirement mentions special pin types ({', '.join(special_pins[:4])})"
        )
        feature_summary["special_pin_terms"] = special_pins

    # 4. Benchmark variants
    variants = _detect_benchmark_variants(text_lower)
    feature_summary["has_benchmark_variant"] = bool(variants)
    if variants:
        triggers.append(TRIGGER_FEATURE_MULTIPLE_VARIANTS)
        reasons.append(
            f"requirement references benchmark variants ({', '.join(variants[:3])})"
        )
        feature_summary["benchmark_variant_terms"] = variants

    # 5. Multiple universes
    multi_univ = _detect_multiple_universes(text_lower)
    feature_summary["has_multiple_universes"] = bool(multi_univ)
    if multi_univ:
        if TRIGGER_FEATURE_SPECIAL_PIN_MAP not in triggers:
            triggers.append(TRIGGER_FEATURE_SPECIAL_PIN_MAP)
            reasons.append(
                f"requirement implies multiple universe types ({', '.join(multi_univ[:3])})"
            )

    # 6. Large lattice
    large_dim = _detect_large_lattice(requirement)
    feature_summary["large_lattice_dimension"] = large_dim
    if large_dim is not None:
        triggers.append(TRIGGER_FEATURE_LARGE_LATTICE)
        reasons.append(
            f"requirement mentions a {large_dim}x{large_dim} or larger lattice"
        )

    # 6b. Localized and potentially multi-segment insert contracts.  Preserve
    # evidence so Facts can represent an unknown profile rather than erasing
    # the requirement merely because a numeric anchor is absent.
    localized_terms = [term for term in _LOCALIZED_INSERT_TERMS if term in text_lower]
    feature_summary["has_localized_insert"] = bool(localized_terms)
    feature_summary["localized_insert_terms"] = localized_terms
    feature_summary["has_control_state"] = bool(localized_terms)
    feature_summary["has_multi_segment_localized_insert"] = bool(localized_terms and flags.has_axial_geometry)

    # 7. Multi-assembly core (core composed of multiple fuel assemblies).
    # This is the trigger that schedules assembly_catalog + core_layout;
    # without it a multi-assembly requirement is mis-routed onto the
    # single-assembly task order and assembly later fails (missing patches).
    is_multi_assembly = _detect_multi_assembly_core(requirement)
    feature_summary["multi_assembly_core"] = is_multi_assembly
    if is_multi_assembly:
        triggers.append(TRIGGER_FEATURE_MULTI_ASSEMBLY)
        reasons.append(
            "requirement describes a multi-assembly core "
            "(multiple assemblies / core-level placement)"
        )

    # --- History analysis ----------------------------------------------------
    hist_triggers, hist_reasons = _analyze_retry_history(retry_history)
    triggers.extend(hist_triggers)
    reasons.extend(hist_reasons)

    # --- Context overrides ---------------------------------------------------
    ctx_triggers, ctx_reasons, force_inc = _analyze_plan_context(plan_context)
    triggers.extend(ctx_triggers)
    reasons.extend(ctx_reasons)

    # --- Final decision ------------------------------------------------------
    # force_monolithic override: respect unless a safety condition forces inc.
    force_monolithic = bool(plan_context and plan_context.get("force_monolithic_planning"))

    if force_inc:
        mode: Literal["monolithic", "incremental"] = "incremental"
        confidence = 1.0
    elif force_monolithic:
        mode = "monolithic"
        confidence = 1.0
        # Still record triggers for observability but override mode.
        reasons.append("plan_context forces monolithic planning; overriding triggers")
    elif triggers:
        mode = "incremental"
        confidence = min(0.99, 0.5 + 0.15 * len(triggers))
    else:
        mode = "monolithic"
        confidence = 1.0
        reasons.append("no incremental triggers detected; simple plan expected")

    # Deduplicate triggers and reasons preserving order.
    triggers = list(dict.fromkeys(triggers))
    reasons = list(dict.fromkeys(reasons))

    return PlanningModeDecision(
        mode=mode,
        reasons=reasons,
        triggers=triggers,
        confidence=round(confidence, 4),
        feature_summary=feature_summary,
    )


__all__ = [
    "PlanningModeDecision",
    "should_use_incremental_planning",
    "TRIGGER_FEATURE_3D_AXIAL",
    "TRIGGER_FEATURE_SPACER_GRID",
    "TRIGGER_FEATURE_SPECIAL_PIN_MAP",
    "TRIGGER_FEATURE_MULTIPLE_VARIANTS",
    "TRIGGER_FEATURE_LARGE_LATTICE",
    "TRIGGER_FEATURE_MULTI_ASSEMBLY",
    "TRIGGER_HISTORY_LARGE_JSON_PARSE_ERROR",
    "TRIGGER_HISTORY_REPAIR_LOST_AXIAL",
    "TRIGGER_HISTORY_REPEATED_AXIAL_CONTRACT_VIOLATION",
    "TRIGGER_OVERRIDE_FORCE_INCREMENTAL",
]
