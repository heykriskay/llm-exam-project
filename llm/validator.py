"""Check the LLM's suggestions with code before the user sees them.

The LLM picks and explains; code decides what is valid:
  - the id must exist AND be among the recipes we actually gave it (else: invented)
  - the recipe must still pass the hard filters (else: rule violation)
  - no duplicates, at most n results
Without RAG there are no ids to check, so suggestions are returned as 'unverified'.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from rag.filters import HardFilters, violations


@dataclass
class Validation:
    valid: list[tuple[int, str]] = field(default_factory=list)
    unverified: list[tuple[str, str]] = field(default_factory=list)
    invented: list[str] = field(default_factory=list)
    rule_violations: list[str] = field(default_factory=list)
    duplicates: int = 0
    intro: str = ""
    parse_error: str = ""


def parse_suggestions(raw: str) -> tuple[str, list[dict], str]:
    """-> (intro, suggestions, error). Tolerates code fences around the JSON."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return "", [], f"Could not parse JSON: {e}"
    if not isinstance(data, dict) or not isinstance(data.get("suggestions"), list):
        return "", [], "JSON has no 'suggestions' list"
    return str(data.get("intro", "")), [s for s in data["suggestions"] if isinstance(s, dict)], ""


def _as_int(value) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def validate(raw: str, candidate_ids: set[int], recipes: dict[int, dict],
             filters: HardFilters, n: int, use_rag: bool = True) -> Validation:
    v = Validation()
    v.intro, suggestions, v.parse_error = parse_suggestions(raw)

    if not use_rag:
        v.unverified = [(str(s.get("name", "")).strip(), str(s.get("reason", "")).strip())
                        for s in suggestions][:n]
        return v

    seen: set[int] = set()
    for s in suggestions:
        rid = _as_int(s.get("id"))
        label = f"{s.get('id')} ({s.get('name', '?')})"
        if rid is None or rid not in candidate_ids or rid not in recipes:
            v.invented.append(label)
            continue
        if rid in seen:
            v.duplicates += 1
            continue
        broken = violations(recipes[rid], filters)
        if broken:
            v.rule_violations.append(f"{label}: {', '.join(broken)}")
            continue
        seen.add(rid)
        v.valid.append((rid, str(s.get("reason", "")).strip()))
    v.valid = v.valid[:n]
    return v
