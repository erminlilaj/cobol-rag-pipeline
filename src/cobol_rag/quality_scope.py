"""Shared quality evidence taxonomy for planning and typed execution.

Quality categories are additive obligations, not competing intents. Semantic
tasks remain authoritative; this resolver is the common text fallback.
"""
from __future__ import annotations

import re

QUALITY_CATEGORIES = (
    "commented_code", "unreachable_code", "unused_copybooks", "review_copybooks",
)


def quality_tasks_for_plan(plan: object) -> tuple[str, ...]:
    """Keep required claim-level obligations when the envelope is incomplete."""
    tasks = set(getattr(plan, "tasks", ()) or ())
    for claim in getattr(plan, "subtasks", ()) or ():
        if getattr(claim, "required", True):
            tasks.update(getattr(claim, "tasks", ()) or ())
    named = tuple(category for category in QUALITY_CATEGORIES if category in tasks)
    # A request quantified over the whole subject -- "summarize everything
    # unused or unreachable" -- asks for the taxonomy, not for whichever
    # category the planner happened to name first. Both signals already exist
    # on the plan: result_scope is the contract that defeats truncation
    # elsewhere, and a summarize operation is a request for the whole picture.
    # Reading them here is what separates covering these phrasings from
    # covering the concept; a request that names its categories is unaffected,
    # because the categories are additive and it already asked for them.
    if named and (
        getattr(plan, "result_scope", None) == "all"
        or "summarize" in set(getattr(plan, "operations", ()) or ())
    ):
        return QUALITY_CATEGORIES
    return named


def quality_categories_named(question: str) -> tuple[str, ...]:
    text = re.sub(r"[-_\u2010-\u2015]+", " ", question.lower())
    copybook = bool(re.search(r"\b(?:copy\s*books?|copies|copy)\b", text))
    broad_code = bool(re.search(r"\b(?:unused|dead)\s+code\b", text))
    requested: set[str] = set()
    if broad_code or re.search(r"\bcomment(?:ed|s)?\b", text):
        requested.add("commented_code")
    if broad_code or re.search(r"\bunreachable\b", text):
        requested.add("unreachable_code")
    if copybook:
        # Review candidates are not proof of non-use. An unused-copybook
        # question needs both statuses, but review-only requests stay narrow.
        if re.search(r"\b(?:unused|dead|proven)\b", text) or not re.search(r"\breview\b", text):
            requested.add("unused_copybooks")
        requested.add("review_copybooks")
    if not requested:
        requested.update(QUALITY_CATEGORIES[:2])
    return tuple(category for category in QUALITY_CATEGORIES if category in requested)
