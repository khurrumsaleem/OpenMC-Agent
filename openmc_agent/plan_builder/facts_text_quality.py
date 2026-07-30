"""High-precision text-quality checks for FactsPatch scalar fields."""

from __future__ import annotations

from typing import Any


FACTS_TEXT_LEAK_CODE = "facts.reasoning_text_leaked"

_SCALAR_FIELD_LIMITS: dict[str, int] = {
    "benchmark_id": 128,
    "selected_variant": 128,
    "geometry_type": 128,
    "boundary_scope": 256,
    "symmetry_description": 512,
}

_REASONING_LEAK_MARKERS: tuple[str, ...] = (
    "now generate json",
    "output only json",
    "trailing commas",
    "schema allows",
    "not in schema",
    "to avoid errors",
    "as placeholder",
    "we need to",
    "we include",
    "we omit",
    "we set",
    "set boundary_scope",
)


def find_facts_scalar_text_leaks(facts_patch: dict[str, Any]) -> list[dict[str, Any]]:
    """Return likely prompt/reasoning leakage in short FactsPatch scalar slots.

    These fields are identifiers or compact descriptions. Long strings are not
    rejected by length alone; they must also carry multiple instruction/reasoning
    markers, which keeps the check reactor-neutral and avoids treating ordinary
    source-backed notes as errors.
    """

    issues: list[dict[str, Any]] = []
    for field_name, soft_limit in _SCALAR_FIELD_LIMITS.items():
        value = facts_patch.get(field_name)
        if not isinstance(value, str):
            continue
        text = value.strip()
        lower = text.lower().replace("_", " ").replace("-", " ")
        marker_count = sum(1 for marker in _REASONING_LEAK_MARKERS if marker in lower)
        if len(text) <= soft_limit or marker_count < 2:
            continue
        issues.append(
            {
                "code": FACTS_TEXT_LEAK_CODE,
                "severity": "error",
                "blocking": True,
                "path": f"/{field_name}",
                "owner_patch_type": "facts",
                "repairable_by_llm": True,
                "requires_human": False,
                "marker_count": marker_count,
                "char_count": len(text),
                "actual_preview": text[:200],
            }
        )
    return issues
