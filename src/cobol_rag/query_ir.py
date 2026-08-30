"""Typed queries: what is being asked, separated from how it was phrased.

The planner used to record a question as a flat bag of entities plus a task
label, which cannot express three things the questions actually asked for:

* **roles** -- that one paragraph is the source of an edge and another is its
  target. With only a bag, "under what condition does A reach B" was
  indistinguishable from "what reaches A and B", and the handler that answers
  edge questions accepted a single target, so a question naming both endpoints
  matched nothing and fell through to free generation.
* **predicates** -- "paragraphs nothing jumps to" is a property of the graph
  (no incoming edges), not an entity that can be named and looked up.
* **relations over programs** -- which set operation a two-program question
  wants.

Because the plan could not hold these, the code went back and re-read the
question text, matching phrase tables. That is why one preposition decided
whether a question was answered: "only in" was a difference and "but not by"
was nothing at all.

Here a question compiles to a typed query whose fields carry that structure.
Recognition is structural wherever it can be -- how many programs are named,
how many graph nodes, where they sit relative to the relation word -- so a
paraphrase that names the same things compiles to the same query.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence


# --------------------------------------------------------------------------
# Query types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GraphEdges:
    """Edges of a control-flow graph, optionally pinned at either end."""

    program: str
    source: str | None = None
    target: str | None = None
    direction: str = "incoming"      # used only when one end is pinned
    projection: str = "all"          # all | condition

    @property
    def kind(self) -> str:
        return "graph_edges"


@dataclass(frozen=True)
class GraphPredicate:
    """Paragraphs selected by a property of the graph rather than by name."""

    program: str
    predicate: str                   # unreferenced | terminal

    @property
    def kind(self) -> str:
        return "graph_predicate"


@dataclass(frozen=True)
class SetRelation:
    """A set operation over the same entity type in several programs."""

    programs: tuple[str, ...]
    relation: str                    # intersection | difference | union | comparison
    entity_type: str | None = None

    @property
    def kind(self) -> str:
        return "set_relation"


@dataclass(frozen=True)
class FieldProjection:
    """One recorded field of one named entity, in one or more programs.

    Asking whether two programs agree about a field is the same projection run
    twice, not a different question, so it is one query with several programs
    rather than a separate comparison type.
    """

    program: str
    entity: str
    field: str                       # origin | ...
    programs: tuple[str, ...] = ()

    @property
    def kind(self) -> str:
        return "field_projection"


@dataclass(frozen=True)
class ScreenField:
    """A field of the program's screen, and where the program touches it."""

    program: str
    field: str

    @property
    def kind(self) -> str:
        return "screen_field"


@dataclass(frozen=True)
class Inventory:
    """Everything of one kind in a program, optionally filtered by a property.

    "Which maps does it send" and "which of them control flow" are the same
    shape: a kind of thing, and a property that narrows it.
    """

    program: str
    entity_type: str
    property_filter: str | None = None

    @property
    def kind(self) -> str:
        return "inventory"


Query = (
    GraphEdges | GraphPredicate | SetRelation | FieldProjection | ScreenField | Inventory
)


# --------------------------------------------------------------------------
# Structural cues
# --------------------------------------------------------------------------

# English negation. A closed grammatical class, not a vocabulary of phrasings:
# any way of saying a relation does not hold uses one of these.
_NEGATION = re.compile(r"(?<![a-z])(?:no|not|never|nothing|none|n't|without)(?![a-z])", re.I)

# A relation between paragraphs, however written. Used only to decide that the
# question is about flow at all; which flow it means is decided structurally.
_FLOW_RELATION = re.compile(
    r"(?<![a-z])(?:reach\w*|lead\w*|jump\w*|goe?s?|go|went|call\w*|perform\w*|"
    r"branch\w*|transfer\w*|flow\w*|enter\w*|exit\w*|arrive\w*|target\w*|"
    r"invoke\w*|transition\w*)(?![a-z])",
    re.I,
)

# The field a question projects out of an entity. Origin is the only one the
# artifacts record under several names, so it is spelled out here.
_ORIGIN_FIELD = re.compile(
    r"(?<![a-z])(?:declar\w*|defin\w*|origin|comes?\s+from|belongs?\s+to|"
    r"which\s+cop\w*|what\s+cop\w*)(?![a-z])",
    re.I,
)

_COPYBOOK_WORD = re.compile(r"(?<![a-z])cop\w*(?![a-z])", re.I)
_SCREEN_WORD = re.compile(r"(?<![a-z])(?:screen|map|bms|display|field)s?(?![a-z])", re.I)


ENTITY_TYPE_NAMES: tuple[str, ...] = (
    "paragraph", "variable", "copybook", "call", "map", "mapset", "field", "program",
)


def entity_type_named(question: str, known: Sequence[str] = ENTITY_TYPE_NAMES) -> str | None:
    """Which kind of thing a question is about, read from the word for it.

    A question that says "copybooks" names its own entity type; nothing has to
    be taught which phrasings mean which type, and a type added to the registry
    is understood as soon as it exists.
    """
    lowered = question.lower()
    best: tuple[int, str] | None = None
    for name in known:
        if re.search(rf"(?<![a-z]){re.escape(name)}s?(?![a-z])", lowered):
            if best is None or len(name) > best[0]:
                best = (len(name), name)
    return best[1] if best else None


def _positions(question: str, names: Sequence[str]) -> dict[str, int]:
    """Where each name is written, so roles can be read off word order."""
    found: dict[str, int] = {}
    upper = question.upper()
    for name in names:
        match = re.search(rf"(?<![A-Z0-9-]){re.escape(name.upper())}(?![A-Z0-9-])", upper)
        if match:
            found[name] = match.start()
    return found


def _relation_between(question: str, first_end: int, second_start: int) -> str | None:
    """Whether the two mentions are joined by 'and also' or by 'but not'.

    Read from the span that starts after the first name and ends with the
    clause carrying the second, rather than from the sentence as a whole: a
    negation before both names belongs to something else, while one attached to
    the second name is the relation ("what A uses that B never uses").
    """
    if second_start <= first_end:
        return None
    tail = question[second_start:]
    boundary = re.search(r"[.?!;]", tail)
    span = question[first_end : second_start + (boundary.end() if boundary else len(tail))]
    if _NEGATION.search(span):
        return "difference"
    if re.search(r"(?<![a-z])(?:both|and|same|also|shared|common)(?![a-z])", span, re.I):
        return "intersection"
    return None


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------

# Properties of a record that a question can narrow an inventory by. Names the
# field in the evidence, so adding one is a statement about the data rather
# than about phrasing.
_PROPERTY_FILTERS: tuple[tuple[str, str], ...] = (
    (r"(?<![a-z])controls?\s+(?:the\s+)?flow(?![a-z])", "controls_flow"),
    (r"(?<![a-z])control\s+flow(?![a-z])", "controls_flow"),
    (r"(?<![a-z])flow[- ]controlling(?![a-z])", "controls_flow"),
)


def property_filter_named(question: str) -> str | None:
    for pattern, field in _PROPERTY_FILTERS:
        if re.search(pattern, question, re.I):
            return field
    return None


def compile_query(
    question: str,
    *,
    program: str | None,
    programs: Sequence[str] = (),
    paragraphs: Sequence[str] = (),
    variables: Sequence[str] = (),
    graph_nodes: Sequence[str] = (),
    screen_fields: Sequence[str] = (),
    entity_type: str | None = None,
    inherited_entity_type: str | None = None,
    allow_inventory: bool = False,
) -> Query | None:
    """Compile a question into a typed query, or None to leave it alone.

    Returns None rather than guessing. Everything the existing planner already
    answers keeps its current path; this adds the shapes that had none.
    """
    named_programs = tuple(dict.fromkeys(p for p in programs if p))

    # Whether two programs agree about a field of one named entity is that
    # field, read in each. Routed before the set branch because the subject is
    # the entity, not the programs' inventories.
    if variables and _ORIGIN_FIELD.search(question) and _COPYBOOK_WORD.search(question):
        return FieldProjection(
            program=(program or (named_programs[0] if named_programs else "")),
            entity=variables[0].upper(),
            field="origin",
            programs=named_programs,
        )

    # Two programs and one entity type is a set question whatever words joined
    # them. Refusing the ones whose phrasing was unrecognised was the single
    # largest source of unanswered comparison questions.
    if len(named_programs) > 1:
        where = _positions(question, named_programs)
        relation = None
        if len(where) > 1:
            ordered = sorted(where.items(), key=lambda kv: kv[1])
            first, second = ordered[0], ordered[1]
            relation = _relation_between(question, first[1] + len(first[0]), second[1])
        return SetRelation(
            programs=named_programs,
            # A two-program question with no readable relation is still
            # answerable as a comparison, which shows both sides and asserts
            # nothing the evidence does not carry.
            relation=relation or "comparison",
            entity_type=entity_type or entity_type_named(question),
        )

    if not program:
        return None

    # A named field that the screen artifact records is a screen question, and
    # the corpus decides that rather than the wording: the program's own list of
    # screen fields is what makes the name one.
    screen = {str(f).upper() for f in screen_fields}
    if screen and _SCREEN_WORD.search(question):
        for name in variables:
            if name.upper() in screen:
                return ScreenField(program=program, field=name.upper())

    nodes = {str(n).upper() for n in graph_nodes}
    named_nodes = [p.upper() for p in paragraphs if p.upper() in nodes]
    named_nodes = list(dict.fromkeys(named_nodes))

    # Both endpoints named: the question is about the edges between them, and
    # which way round is a property of where they sit, not of the verb used.
    if len(named_nodes) >= 2:
        where = _positions(question, named_nodes)
        if len(where) >= 2:
            ordered = sorted(where.items(), key=lambda kv: kv[1])
            first, second = ordered[0], ordered[1]
            source, target = first[0], second[0]
            # Word order alone gets the roles backwards under passive voice:
            # "A is reached from B" names A first and B is the source. The
            # preposition in front of a name is what marks its role, so it is
            # read rather than assumed.
            if _preceded_by_source_marker(question, second[1]):
                source, target = target, source
            return GraphEdges(
                program=program,
                source=source,
                target=target,
                projection="condition" if _asks_for_condition(question) else "all",
            )

    # A flow question that names no paragraph but negates the relation is
    # asking for the paragraphs the relation does not hold for.
    if not named_nodes and _FLOW_RELATION.search(question) and _NEGATION.search(question):
        return GraphPredicate(program=program, predicate="unreferenced")

    # An inventory of a kind of thing, narrowed by a property if one is named.
    # Offered only where the planner produced no route of its own, so this
    # fills a gap rather than overriding something that already works.
    if allow_inventory:
        wanted = entity_type_named(question) or inherited_entity_type
        if wanted:
            return Inventory(
                program=program,
                entity_type=wanted,
                property_filter=property_filter_named(question),
            )

    # Which copybook declares a named variable: a field of that variable, not
    # the program's copybook inventory. Answering it from the inventory was a
    # substitution the response contract had to reject, leaving no answer.
    return None


# A name introduced by "from" or "by" is where control comes from, whichever
# side of the sentence it sits on.
_SOURCE_MARKER = re.compile(r"(?<![a-z])(?:from|by|out\s+of)\s+$", re.I)


def _preceded_by_source_marker(question: str, position: int) -> bool:
    return bool(_SOURCE_MARKER.search(question[max(0, position - 24):position]))


def _asks_for_condition(question: str) -> bool:
    return bool(
        re.search(r"(?<![a-z])(?:condition|when|guard\w*|under\s+what|why)(?![a-z])", question, re.I)
    )
