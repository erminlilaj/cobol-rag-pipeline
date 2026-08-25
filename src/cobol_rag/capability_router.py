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
    # Queue commands were missing from this list, so asking whether the program
    # writes to a queue drifted to the nearest storage-shaped capability, JCL
    # datasets, and came back reporting an absence that was true of JCL but not
    # of the question. A capability has to name every command family it holds.
    "cics_evidence": (
        "Every CICS command the program executes: map send and receive, queue writes "
        "and reads to temporary storage, program link and transfer, syncpoint, return "
        "and abend, with the paragraph and line of each."
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
    # An address, not a meaning. Ranking cannot resolve one, so this descriptor
    # exists to advertise the capability and to keep line questions away from
    # capabilities that would answer them by similarity.
    "source_line_lookup": (
        "The exact text at a physical source address: a specific line number or "
        "range of line numbers in a program or copybook member, shown verbatim."
    ),
}

# What a question can want to know about one named variable, described by meaning.
#
# Reads and writes are deliberately absent. Measurement showed embedding
# similarity ranks them almost interchangeably, and inverts them on questions
# about where a value comes from: "which statement produces the value" scored
# reads above writes. They are the same topic with opposite polarity, and
# similarity does not carry polarity. Direction is decided separately.
VARIABLE_ASPECT_DESCRIPTORS: dict[str, str] = {
    "variable_definition": (
        "Where the field is declared and what kind of storage it lives in, its "
        "origin and the section that defines it."
    ),
    "variable_lineage": (
        "What this field's value eventually reaches, which destination fields or "
        "screen positions receive it downstream, and where the trail ends."
    ),
    # The inbound direction of lineage. Without it, asking what a field is built
    # from could only be answered by its write sites, which name the statements
    # but never the fields that contribute the value.
    "variable_composition": (
        "Which variables contribute to this field's content, the pieces "
        "gathered from elsewhere that together form its value."
    ),
    "control_outcome": (
        "What the program does as a consequence of the value. Which branch is "
        "taken, what happens when the condition holds, the resulting action."
    ),
    "literal_assignments": (
        "The specific hard-coded constant values that are forced into the field."
    ),
    "variable_comparison": (
        "How two different fields relate to each other, comparing one against another."
    ),
}

# Calibrated on a small sample: correctly matched aspects scored 0.51 to 0.76
# with a margin of 0.05 or more over the next aspect, while questions belonging
# to no aspect stayed below. Re-measure before trusting these on a new corpus.
MIN_ASPECT_SCORE = 0.45
MIN_ASPECT_MARGIN = 0.05
# A second aspect is only a second request when it would have been confident on
# its own. Measured on the same sample: genuine pairs sat at 0.59 and above,
# while the runner-up behind a single-aspect question stayed near 0.52.
MIN_COMPOUND_ASPECT_SCORE = 0.55
# Companions are read off the leader, so a weak leader cannot introduce them: a
# flat cluster just above the floor means the question matched nothing clearly,
# not that it asked for three things at once. Measured leaders of real compound
# questions sat at 0.62 and above; a three-way tie of unrelated aspects led at
# 0.56 and produced a claim the evidence could not support.
MIN_COMPOUND_LEADER_SCORE = 0.60


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
        self._aspect_vectors: dict[str, Sequence[float]] | None = None

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

    def aspect_vectors(self) -> dict[str, Sequence[float]]:
        if self._aspect_vectors is None:
            self._aspect_vectors = {
                aspect: self._embed_text(description)
                for aspect, description in VARIABLE_ASPECT_DESCRIPTORS.items()
            }
        return self._aspect_vectors

    def rank_aspects(self, question: str) -> tuple[CapabilityMatch, ...]:
        """Rank what a question wants to know about a named variable."""
        text = (question or "").strip()
        if not text:
            return ()
        return rank_capabilities(self._embed_text(text), self.aspect_vectors())


def confident_aspects(matches: tuple[CapabilityMatch, ...]) -> tuple[str, ...]:
    """Keep the aspects a question clearly asked for.

    A question about reads or writes belongs to no aspect here, and must come
    back empty rather than being assigned the nearest unrelated one.

    One question can ask about more than one aspect: "what values can it take,
    and how does each affect execution" is asking about both the assigned
    constants and the resulting control flow. Requiring the leader to beat the
    runner-up rejected exactly those, because the second thing being asked is
    what closes the gap. A runner-up is therefore treated as a second request
    when it stands on its own well above the bar, and as noise otherwise.
    """
    if not matches:
        return ()
    best = matches[0]
    if best.score < MIN_ASPECT_SCORE:
        return ()
    companions = tuple(
        match.capability
        for match in matches[1:]
        if best.score >= MIN_COMPOUND_LEADER_SCORE
        and match.score >= MIN_COMPOUND_ASPECT_SCORE
        and best.score - match.score <= MIN_ASPECT_MARGIN
    )
    if companions:
        return (best.capability, *companions)
    if best.margin < MIN_ASPECT_MARGIN:
        return ()
    return (best.capability,)


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
