"""Rank evidence capabilities by meaning instead of by question wording.

The semantic planner is a local 8B model. When it returns an unusable plan the
system previously fell back to an empty one, which selects no capability at all
and degrades into generic retrieval plus free-form generation. That failure is
not a wording problem, so it cannot be fixed by adding more question patterns.

Each capability is described here in natural language by what it answers. A
question is embedded with the same model that already indexes the corpus and
compared against those descriptions, so paraphrases land near the capability
that owns them without any phrase being enumerated. Deterministic entity scope
still decides which capabilities are eligible: this module ranks meaning, it
never invents a program or an identifier.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

# What each capability answers, written as meaning rather than as keywords.
CAPABILITY_DESCRIPTORS: dict[str, str] = {
    "variable_inventory": (
        "The catalogue of the variables, fields and data items a program declares. "
        "Answers which variables or fields exist, how many of them there are, naming "
        "or sampling a few of them, listing them all, and which of the declared "
        "variables are flagged as controlling execution, whenever the question is "
        "about the program's variables in general rather than one named variable."
    ),
    "variable_access": (
        "Where one specific named variable is defined, set, written, modified, "
        "computed, read, tested or checked, with the paragraphs and source lines "
        "for that exact identifier."
    ),
    "variable_lineage": (
        "How a value moves out of one named variable into other variables, the "
        "chain of transfers through intermediate fields, and where it finally lands."
    ),
    "literal_assignment": (
        "Hard-coded literal values, forced values and constants that the program "
        "assigns to fields. Answers which fixed values are set anywhere in the "
        "program, listing every literal together with the field it is moved into "
        "and the line where that value is forced."
    ),
    "condition_outcome": (
        "Business rules and decisions: the conditions the program tests, what "
        "happens when a condition is or is not met, and the resulting action."
    ),
    "control_flow": (
        "The order in which the program executes. Answers what runs first, how "
        "control passes from one paragraph to the next, which branches are taken, "
        "and how execution finally terminates or hands control away. A step by step "
        "walk through the program's execution path from entry to end."
    ),
    "paragraph_evidence": (
        "What one specific named paragraph does or executes, and the places from "
        "which that paragraph is referenced or performed."
    ),
    "pagination_evidence": (
        "How the program moves between pages of a result list, paging forward and "
        "backward through rows shown on the screen."
    ),
    "call_evidence": (
        "Which external programs this program calls, links to or transfers control "
        "to, including the call type and the parameters or COMMAREA passed."
    ),
    "call_context": (
        "What the program prepares immediately before one specific call, and what "
        "it inspects, tests or does with the result once that call returns control."
    ),
    "cics_evidence": (
        "The CICS commands the program issues, such as sending and receiving maps, "
        "linking, transferring control, returning or abending, and where each occurs."
    ),
    "copybook_evidence": (
        "The copybooks or COPY members the program includes, which of them are "
        "actually referenced by evidence, and which appear unused or need review."
    ),
    "db2_evidence": (
        "The DB2 tables the program accesses and the SQL INCLUDE members it uses."
    ),
    "jcl_evidence": (
        "The JCL jobs, steps, DD names and datasets associated with the program."
    ),
    "quality_evidence": (
        "Dead code findings: commented-out code, unreachable paragraphs, and "
        "copybooks with no reference evidence in the analyzed artifacts."
    ),
    "program_summary": (
        "A short technical identification of the program itself: what kind of "
        "program it is, the technology it runs on, and its overall scale."
    ),
    "source_metrics": (
        "The size of the program measured in lines of code, paragraphs or statements."
    ),
    "artifact_inventory": (
        "Which analysis artifacts and evidence files are available for the program."
    ),
    "screen_lineage": (
        "How a value reaches a field displayed on a screen or map."
    ),
}

# Capabilities that describe one named entity. Without a resolved identifier they
# have nothing to answer about, so deterministic scope removes them from ranking.
ENTITY_REQUIRED_CAPABILITIES = frozenset({
    "variable_access", "variable_lineage", "call_context", "paragraph_evidence",
    "screen_lineage",
})

# The claim each capability verifies when the planner supplies no explicit tasks.
CAPABILITY_DEFAULT_TASKS: dict[str, tuple[str, ...]] = {
    "variable_inventory": ("variable_inventory",),
    "variable_access": ("variable_definition", "variable_reads", "variable_writes"),
    "variable_lineage": ("variable_lineage",),
    "literal_assignment": ("literal_assignments",),
    "condition_outcome": ("business_rules",),
    "control_flow": ("complete_program_flow",),
    "paragraph_evidence": ("paragraph_body",),
    "pagination_evidence": ("pagination_logic",),
    "call_evidence": ("external_calls",),
    "call_context": ("call_context",),
    "cics_evidence": ("cics_operations",),
    "copybook_evidence": ("copybook_inventory",),
    "db2_evidence": ("db2_tables",),
    "jcl_evidence": ("jcl_datasets",),
    "quality_evidence": ("commented_code", "unreachable_code"),
    "program_summary": ("program_summary",),
    "source_metrics": ("source_metrics",),
    "artifact_inventory": ("artifact_inventory",),
    "screen_lineage": ("screen_lineage",),
}

# A match is only acted on when it is both similar enough and clearly ahead of the
# runner-up, so a question that belongs to no capability stays unrouted instead of
# being forced into the nearest one.
MIN_CAPABILITY_SCORE = 0.55
MIN_CAPABILITY_MARGIN = 0.015


@dataclass(frozen=True)
class CapabilityMatch:
    capability: str
    score: float
    margin: float

    @property
    def confident(self) -> bool:
        return self.score >= MIN_CAPABILITY_SCORE and self.margin >= MIN_CAPABILITY_MARGIN


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def eligible_capabilities(
    *,
    entity_types: Iterable[str] = (),
    available: Iterable[str] | None = None,
) -> tuple[str, ...]:
    """Restrict ranking to capabilities the resolved scope can actually support."""
    resolved = {value for value in entity_types if value}
    candidates = set(available if available is not None else CAPABILITY_DESCRIPTORS)
    identifier_types = {"variable", "unknown_identifier", "call", "paragraph", "copybook"}
    if not (resolved & identifier_types):
        candidates -= ENTITY_REQUIRED_CAPABILITIES
    if resolved & {"variable", "unknown_identifier"}:
        # An exact identifier is authoritative: the per-variable evidence answers
        # it, and the whole-program catalogue would replace it with a broad list.
        candidates.discard("variable_inventory")
    return tuple(sorted(candidates))


def rank_capabilities(
    question_vector: Sequence[float],
    descriptor_vectors: dict[str, Sequence[float]],
    *,
    allowed: Iterable[str] | None = None,
) -> tuple[CapabilityMatch, ...]:
    """Order capabilities by semantic closeness to the question."""
    permitted = set(allowed) if allowed is not None else set(descriptor_vectors)
    scored = sorted(
        (
            (cosine_similarity(question_vector, vector), capability)
            for capability, vector in descriptor_vectors.items()
            if capability in permitted
        ),
        key=lambda item: (-item[0], item[1]),
    )
    if not scored:
        return ()
    matches: list[CapabilityMatch] = []
    for index, (score, capability) in enumerate(scored):
        runner_up = scored[index + 1][0] if index + 1 < len(scored) else 0.0
        matches.append(
            CapabilityMatch(capability, score, score - runner_up if index == 0 else 0.0)
        )
    return tuple(matches)


class CapabilityRouter:
    """Embed capability descriptions once, then rank questions against them."""

    def __init__(self, embed_text: Callable[[str], Sequence[float]]) -> None:
        self._embed_text = embed_text
        self._descriptor_vectors: dict[str, Sequence[float]] | None = None

    def descriptor_vectors(self) -> dict[str, Sequence[float]]:
        if self._descriptor_vectors is None:
            self._descriptor_vectors = {
                capability: self._embed_text(description)
                for capability, description in CAPABILITY_DESCRIPTORS.items()
            }
        return self._descriptor_vectors

    def rank(
        self,
        question: str,
        *,
        allowed: Iterable[str] | None = None,
    ) -> tuple[CapabilityMatch, ...]:
        text = (question or "").strip()
        if not text:
            return ()
        return rank_capabilities(
            self._embed_text(text), self.descriptor_vectors(), allowed=allowed,
        )


_ROUTER_CACHE: dict[tuple[str, str], CapabilityRouter] = {}


def router_for(config: Any) -> CapabilityRouter:
    """Reuse one router per configured embedding endpoint so descriptors embed once."""
    key = (str(config.embedding.model), str(config.embedding.base_url))
    router = _ROUTER_CACHE.get(key)
    if router is None:
        from cobol_rag.index import build_embedder

        embedder = build_embedder(config)
        router = CapabilityRouter(embedder.get_query_embedding)
        _ROUTER_CACHE[key] = router
    return router
