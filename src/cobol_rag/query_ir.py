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
    in_paragraph: str | None = None

    @property
    def kind(self) -> str:
        return "inventory"


@dataclass(frozen=True)
class CopybookRole:
    """Why a copybook is included, read from how the program uses it."""

    program: str
    copybook: str

    @property
    def kind(self) -> str:
        return "copybook_role"


@dataclass(frozen=True)
class UnusedCode:
    """Everything unused in one program: paragraphs, copybooks, commented code."""

    program: str

    @property
    def kind(self) -> str:
        return "unused_code"


@dataclass(frozen=True)
class CorpusReferences:
    """Which analyzed programs reference a name, and in what role.

    The subject is the corpus rather than one program, so this cannot be
    answered from any single program's artifacts however complete they are.
    """

    entity: str
    relation: str | None = None

    @property
    def kind(self) -> str:
        return "corpus_references"


Query = (
    GraphEdges | GraphPredicate | SetRelation | FieldProjection | ScreenField
    | Inventory | CopybookRole | CorpusReferences | UnusedCode
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
_UNUSED_QUESTION = re.compile(
    r"(?<![a-z])(?:unused|unreferenced|dead|unreachable|never\s+(?:used|reached|called)|"
    r"left\s+over|leftover|obsolete)(?![a-z])",
    re.I,
)
_SCREEN_WORD = re.compile(r"(?<![a-z])(?:screen|map|bms|display|field)s?(?![a-z])", re.I)
# Asking what something is for, rather than for a list of them.
# A question whose subject is the corpus: it asks which programs do something,
# or who does something to a named thing. The plural "programs" and the passive
# or interrogative subject are what mark it, not any particular verb.
_CORPUS_SUBJECT = re.compile(
    # Adjectives may sit between the determiner and "programs" -- "which other
    # analyzed programs" -- so allow a few words rather than one fixed one.
    r"(?<![a-z])(?:which|what|any|all|how\s+many)\s+(?:\w+\s+){0,3}programs?(?![a-z])"
    r"|(?<![a-z])who\s+(?:calls|uses|includes|references)(?![a-z])"
    r"|(?<![a-z])(?:called|used|included|referenced)\s+by\s+(?:which|what)(?![a-z])",
    re.I,
)

# The role a corpus question is asking about, named by its own verb.
_CORPUS_RELATIONS: tuple[tuple[str, str], ...] = (
    (r"(?<![a-z])call(?:s|ed|ing)?(?![a-z])", "calls"),
    (r"(?<![a-z])includ(?:e|es|ed|ing)(?![a-z])", "includes"),
    (r"(?<![a-z])us(?:e|es|ed|ing)(?![a-z])", "includes"),
    (r"(?<![a-z])declar(?:e|es|ed|ing)(?![a-z])", "declares"),
    (r"(?<![a-z])defin(?:e|es|ed|ing)(?![a-z])", "defines"),
)


def corpus_relation_named(question: str) -> str | None:
    for pattern, relation in _CORPUS_RELATIONS:
        if re.search(pattern, question, re.I):
            return relation
    return None


_PURPOSE_QUESTION = re.compile(
    r"(?<![a-z])(?:what\s+is\s+\S+\s+(?:for|used\s+for)|purpose|role|why\s+is\s+\S+\s+"
    r"(?:cop\w+|includ\w+|used)|what\s+does\s+\S+\s+do|tell\s+me\s+about)(?![a-z])",
    re.I,
)


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
    (r"(?<![a-z])literal\s+length(?![a-z])", "literal_length"),
    (r"(?<![a-z])length\s+literal(?![a-z])", "literal_length"),
    (r"(?<![a-z])(?:temporary[- ]storage|ts\s+queue|tsq)(?![a-z])", "ts_queue"),
)

# Entity types a question can name that are not registry entities. CICS
# operations are recorded per program rather than as named things, so they need
# their own word.
_OPERATION_WORD = re.compile(
    r"(?<![a-z])(?:cics\s+)?(?:command|operation|statement)s?(?![a-z])", re.I
)
_CALL_WORD = re.compile(r"(?<![a-z])(?:call|link|xctl)s?(?![a-z])", re.I)


def property_filter_named(question: str) -> str | None:
    for pattern, field in _PROPERTY_FILTERS:
        if re.search(pattern, question, re.I):
            return field
    return None



def paragraph_scope(
    question: str, graph_nodes: Sequence[str], program: str | None = None
) -> str | None:
    """A paragraph the question restricts the answer to, rather than asks about.

    "Which variables does LINK-PD0UTI01 modify" names a paragraph as a scope,
    not as the subject. Without reading it, the question was answered with
    every variable in the program.

    The program's own name is excluded. Statements before the first paragraph
    label belong to an implicit paragraph named after the program, so the
    program is a node in its own graph -- and "in PDCBVC" then reads as a
    restriction to that one paragraph rather than as naming the program.
    """
    excluded = {str(program).upper()} if program else set()
    nodes = {str(n).upper() for n in graph_nodes} - excluded
    upper = question.upper()
    for name in sorted(nodes, key=len, reverse=True):
        if re.search(rf"(?<![A-Z0-9-]){re.escape(name)}(?![A-Z0-9-])", upper):
            return name
    return None



def _names_the_actor(question: str, entity: str) -> bool:
    """Whether the named program is doing the calling rather than being called.

    "Which programs call PDCBVC" and "which programs does PDCBVC call" ask
    opposite questions and differ only in where the name sits relative to the
    verb. Reading the plural "programs" without reading that reversed the
    second one: a request for a program's own outgoing calls came back as the
    list of programs that call it, which is a different fact and was empty.
    """
    upper = question.upper()
    name = re.escape(entity.upper())
    # The name is the grammatical subject: "does X call", "X calls", "X uses".
    if re.search(rf"\bDOES\s+{name}\b", upper):
        return True
    if re.search(rf"(?<![A-Z0-9-]){name}(?![A-Z0-9-])\s+(?:CALLS?|USES?|INCLUDES?)\b", upper):
        return True
    # Passive with an agent: "called by X", "used by X".
    # "called by both A and B" puts a quantifier between the preposition and
    # the name, and the name is still the actor.
    if re.search(
        rf"\b(?:CALLED|USED|INCLUDED|REFERENCED)\s+BY\s+(?:\w+\s+){{0,2}}{name}\b", upper
    ):
        return True
    return False


def compile_query(
    question: str,
    *,
    program: str | None,
    programs: Sequence[str] = (),
    paragraphs: Sequence[str] = (),
    variables: Sequence[str] = (),
    graph_nodes: Sequence[str] = (),
    screen_fields: Sequence[str] = (),
    copybooks: Sequence[str] = (),
    corpus_entity: str | None = None,
    capability: str | None = None,
    entity_type: str | None = None,
    inherited_entity_type: str | None = None,
    allow_inventory: bool = False,
) -> Query | None:
    """Compile a question into a typed query, or None to leave it alone.

    Returns None rather than guessing. Everything the existing planner already
    answers keeps its current path; this adds the shapes that had none.
    """
    named_programs = tuple(dict.fromkeys(p for p in programs if p))

    # What the router judged the question to mean, decided by comparing it with
    # each capability's description rather than with a list of phrasings. It
    # leads; the patterns below only fill in a query's fields or stand in when
    # the router had no confident view.
    if capability == "unused_code" and program:
        return UnusedCode(program=program)
    if capability == "copybook_role" and program and copybooks:
        upper = question.upper()
        for name in sorted({str(c).upper() for c in copybooks}, key=len, reverse=True):
            if re.search(rf"(?<![A-Z0-9-]){re.escape(name)}(?![A-Z0-9-])", upper):
                return CopybookRole(program=program, copybook=name)
    if capability == "screen_field" and program and screen_fields:
        upper = question.upper()
        for name in sorted({str(f).upper() for f in screen_fields}, key=len, reverse=True):
            if re.search(rf"(?<![A-Z0-9-]){re.escape(name)}(?![A-Z0-9-])", upper):
                return ScreenField(program=program, field=name)
    if (
        capability == "corpus_references"
        and corpus_entity
        and len(named_programs) < 2
        and not _names_the_actor(question, corpus_entity)
    ):
        return CorpusReferences(
            entity=corpus_entity.upper(), relation=corpus_relation_named(question)
        )

    # A question about the corpus is answered from the corpus. Resolving it to
    # one program is what made "which programs use PDRUTI01" a clarification --
    # the copybook being in several programs is the answer, not an ambiguity --
    # and what made "which program calls PDCBVC" answer with PDCBVC's own
    # outgoing calls, the only direction a program-scoped capability has.
    # Naming two analyzed programs makes the question a comparison between
    # them, not a lookup of what refers to one of them.
    if (
        corpus_entity
        and len(named_programs) < 2
        and _CORPUS_SUBJECT.search(question)
        and not _names_the_actor(question, corpus_entity)
    ):
        return CorpusReferences(
            entity=corpus_entity.upper(),
            relation=corpus_relation_named(question),
        )

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
    # A set question needs the programs to be what the question is about. A
    # question that names a second program only in passing -- "what is this
    # copybook for, and which other programs use it" -- is not a comparison,
    # and treating it as one took every part of every multi-program question.
    if len(named_programs) > 1 and not _PURPOSE_QUESTION.search(question) and not (
        corpus_entity and _CORPUS_SUBJECT.search(question)
    ):
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
    # A named copybook the question asks the purpose of. Checked before the
    # inventory branch, which would otherwise answer "what is PDRTWA2 for" with
    # the program's list of copybooks.
    if copybooks and _PURPOSE_QUESTION.search(question):
        upper = question.upper()
        for name in sorted({str(c).upper() for c in copybooks}, key=len, reverse=True):
            if re.search(rf"(?<![A-Z0-9-]){re.escape(name)}(?![A-Z0-9-])", upper):
                return CopybookRole(program=program, copybook=name)

    # "Unused code or copy" is one question spanning three artifacts. Answered
    # from whichever the planner picked, it reported no unused copybooks while
    # proven-unreachable paragraphs sat unmentioned in the graph.
    if _UNUSED_QUESTION.search(question):
        return UnusedCode(program=program)

    predicate = property_filter_named(question)
    scope_paragraph = paragraph_scope(question, graph_nodes, program)
    wanted = entity_type_named(question) or inherited_entity_type
    if wanted is None and _OPERATION_WORD.search(question):
        wanted = "cics_operation"
    elif wanted is None and _CALL_WORD.search(question):
        wanted = "call"

    # A narrowed set is answered whenever the narrowing is readable, even where
    # the planner had a route: returning the unnarrowed set is the failure this
    # exists to prevent, and it looks like a successful answer.
    if wanted and (predicate or scope_paragraph):
        # A paragraph that is the subject of a flow question is not a scope.
        if not (scope_paragraph and wanted == "paragraph"):
            return Inventory(
                program=program,
                entity_type=wanted,
                property_filter=predicate,
                in_paragraph=scope_paragraph,
            )

    if allow_inventory and wanted:
        return Inventory(
            program=program,
            entity_type=wanted,
            property_filter=predicate,
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


# ---------------------------------------------------------------------------
# Compound questions
#
# A question can ask several things at once, and the parts often need different
# capabilities: "what is this copybook for, and which other programs use it" is
# a role question and a corpus question. Compiling to one query answered the
# first part and dropped the second without saying so, which reads as a
# complete answer to a question that was only half read.
# ---------------------------------------------------------------------------

# Clause boundaries strong enough to separate questions. Plain "and" is not one
# of them: it joins operands far more often than clauses ("PDCBVC and PDB305"),
# so it splits only before a word that starts a new question.
_PART_BOUNDARY = re.compile(
    r",\s*and\s+|;\s*|\?\s+|"
    r"\s+and\s+(?=(?:which|what|who|where|when|how|does|do|did|is|are|was|were|"
    r"can|could|should|if|whether)\b)",
    re.IGNORECASE,
)


def split_question(question: str) -> list[str]:
    """The separate things a question asks, in the order asked."""
    parts = [part.strip(" ,;?") for part in _PART_BOUNDARY.split(question or "")]
    return [part for part in parts if len(part.split()) >= 2]


def compile_queries(question: str, **context: Any) -> list[tuple[str, Query | None]]:
    """Every part of a question with the query it compiles to, in order.

    A part that compiles to nothing is kept with a None rather than dropped.
    Dropping it is what made a half-read question look fully answered: the
    missing part left no trace, so the reply was indistinguishable from one
    that had answered everything asked.
    """
    whole = compile_query(question, **context)
    parts = split_question(question)
    if len(parts) < 2:
        return [(question, whole)] if whole else []

    compiled: list[tuple[str, Query | None]] = []
    seen: list[Query] = []
    for part in parts:
        # Each part is compiled against the same resolved context, so a part
        # that names nothing of its own ("and which programs use it") still
        # sees the entities the question as a whole resolved.
        query = compile_query(part, **context)
        if query is not None and query in seen:
            continue
        if query is not None:
            seen.append(query)
        compiled.append((part, query))
    if whole is not None and whole not in seen:
        compiled.insert(0, (question, whole))
    return compiled
