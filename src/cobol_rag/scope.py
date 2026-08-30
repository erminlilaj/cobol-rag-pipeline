from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from collections.abc import Sequence
from typing import Any

from cobol_rag.final_scripts_answers import find_final_scripts_root


@dataclass(frozen=True)
class EntityReference:
    program: str
    entity_type: str
    value: str
    entity_key: str


@dataclass(frozen=True)
class QueryScope:
    program: str | None = None
    programs: tuple[str, ...] = ()
    entity_type: str | None = None
    entity_value: str | None = None
    entity_key: str | None = None
    entities: tuple[EntityReference, ...] = ()
    intent: str | None = None
    confidence: float = 0.0
    program_source: str = "unresolved"
    entity_source: str = "unresolved"
    ambiguous: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def entity_values(self) -> tuple[str, ...]:
        if self.entities:
            return tuple(entity.value for entity in self.entities)
        return (self.entity_value,) if self.entity_value else ()

    @property
    def entity_keys(self) -> tuple[str, ...]:
        if self.entities:
            return tuple(entity.entity_key for entity in self.entities if entity.entity_key)
        return (self.entity_key,) if self.entity_key else ()


@dataclass
class SessionState:
    current_program: str | None = None
    current_programs: list[str] = field(default_factory=list)
    current_entity_type: str | None = None
    current_entity_value: str | None = None
    current_entity_key: str | None = None
    current_entities: list[EntityReference] = field(default_factory=list)
    # ``current_entities`` is the last resolved scope.  It may contain several
    # comparison targets, so it is not safe to interpret its first item as what
    # a later singular pronoun means.  Focus is set only when one target is
    # unambiguous; the result set remains available for plural follow-ups.
    focused_entity: EntityReference | None = None
    last_result_entities: list[EntityReference] = field(default_factory=list)
    last_capabilities: list[str] = field(default_factory=list)
    current_intent: str | None = None
    current_domain: str | None = None
    current_tasks: list[str] = field(default_factory=list)
    response_language: str = "en"
    last_sources: list[str] = field(default_factory=list)
    pending_clarification: str | None = None
    current_plan: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def update(
        self,
        scope: QueryScope,
        source_ids: list[str],
        plan: dict[str, Any] | None = None,
    ) -> None:
        if scope.program:
            self.current_program = scope.program
        self.current_programs = list(scope.programs or ((scope.program,) if scope.program else ()))
        resolved_entities = list(scope.entities)
        if not resolved_entities and scope.entity_value:
            resolved_entities = [
                EntityReference(
                    program=scope.program or "",
                    entity_type=scope.entity_type or "unknown",
                    value=scope.entity_value,
                    entity_key=scope.entity_key or "",
                )
            ]
        if resolved_entities:
            primary = resolved_entities[0]
            self.current_entities = resolved_entities
            self.last_result_entities = list(resolved_entities)
            self.focused_entity = primary if len(resolved_entities) == 1 else None
            self.current_entity_type = primary.entity_type
            self.current_entity_value = primary.value
            self.current_entity_key = primary.entity_key
        elif scope.entity_source == "unresolved":
            self.clear_entities()
        if scope.intent:
            self.current_intent = scope.intent
        self.last_sources = source_ids[-8:]
        self.pending_clarification = scope.reason if scope.ambiguous else None
        if plan is not None:
            self.current_plan = dict(plan)
            self.current_domain = str(plan.get("domain") or "") or None
            self.current_tasks = [str(value) for value in plan.get("tasks", [])]
            self.last_capabilities = [
                str(item.get("capability"))
                for item in plan.get("subtasks", [])
                if isinstance(item, dict) and str(item.get("capability", "")).strip()
            ]
            language = str(plan.get("response_language") or "").strip().lower()
            if language:
                self.response_language = language

    def reset(self) -> None:
        self.current_program = None
        self.current_programs.clear()
        self.clear_entities()
        self.current_intent = None
        self.current_domain = None
        self.current_tasks.clear()
        self.last_capabilities.clear()
        self.response_language = "en"
        self.last_sources.clear()
        self.pending_clarification = None
        self.current_plan.clear()

    def clear_entities(self) -> None:
        self.current_entity_type = None
        self.current_entity_value = None
        self.current_entity_key = None
        self.current_entities.clear()
        self.focused_entity = None
        self.last_result_entities.clear()


def _programs_holding(
    entities: Sequence[EntityReference], tokens: Sequence[str],
) -> dict[str, set[str]]:
    """Which analyzed programs hold each identifier, across the whole corpus.

    Driven entirely by the catalogue, so it stays correct as programs are added:
    nothing here knows how many there are or what they are called.
    """
    found: dict[str, set[str]] = {}
    for token in tokens:
        upper = token.upper()
        holders = {
            entity.program
            for entity in entities
            if entity.program
            and (entity.value == upper or entity.value.startswith(upper + "-"))
        }
        if holders:
            found[token] = holders
    return found


def resolve_query_scope(
    question: str,
    *,
    intent: str | None = None,
    state: SessionState | None = None,
    final_scripts_root: Path | None = None,
    target_program: str | None = None,
) -> QueryScope:
    root = final_scripts_root or find_final_scripts_root()
    programs, entities = _catalogue(str(root.resolve()) if root else "")
    upper = question.upper()

    mentioned_programs = [
        program for program in programs if _contains_identifier(upper, program)
    ]
    globally_named_entities = [
        entity for entity in entities if _contains_identifier(upper, entity.value)
    ]
    entity_programs = tuple(dict.fromkeys(
        entity.program for entity in globally_named_entities
        if entity.program not in {"__GLOBAL__", "GLOBAL"}
    ))
    # Naming two analyzed programs is itself the request to consider both. The
    # relation between them is read where it can be and defaults to a
    # comparison, which shows each side's evidence and asserts nothing beyond
    # it. Requiring a recognised phrase before accepting the question refused
    # "copybooks used by A but not by B" over the preposition alone, while a
    # comparison would have answered it and every paraphrase of it.
    multi_program_comparison = len(mentioned_programs) > 1

    program: str | None = None
    program_source = "unresolved"
    selected_program = str(target_program or "").strip().upper()
    if mentioned_programs:
        program = mentioned_programs[0]
        program_source = "question_multi" if multi_program_comparison else "question"
    elif selected_program:
        if selected_program not in programs:
            return QueryScope(
                intent=intent,
                ambiguous=True,
                reason=f"Selected program `{selected_program}` is not present in the analyzed corpus.",
            )
        program = selected_program
        program_source = "target"
    elif state and state.current_program in programs:
        program = state.current_program
        program_source = "session"
    elif len(programs) == 1:
        program = programs[0]
        program_source = "single_program"
    elif len(entity_programs) == 1:
        program = entity_programs[0]
        program_source = "unique_entity"
    elif len(entity_programs) > 1:
        return QueryScope(
            programs=entity_programs,
            intent=intent,
            ambiguous=True,
            reason=(
                "The named COBOL entity exists in multiple analyzed programs "
                f"({', '.join(entity_programs)}). Name the one you mean."
            ),
        )
    elif len(programs) > 1:
        return QueryScope(
            intent=intent,
            ambiguous=True,
            reason="The corpus contains multiple programs and the question does not identify one.",
        )

    eligible_entities = [
        entity
        for entity in entities
        if program is None or entity.program in set(mentioned_programs or [program]) | {"__GLOBAL__", "GLOBAL"}
    ]
    candidates = [
        entity for entity in eligible_entities if _contains_identifier(upper, entity.value)
    ]
    # A program name can also appear in analyzer output as its entry paragraph.
    # In ordinary questions ("describe PDCBVC", "calls in PDCBVC") that token is
    # the program selector, not a paragraph selector.  Treat it as a paragraph only
    # when the user explicitly says so; otherwise it can poison semantic planning by
    # turning a program-wide request into an entity request.
    candidates = [
        entity
        for entity in candidates
        if entity.value not in programs
        or bool(re.search(
            rf"\bparagraph\s+{re.escape(entity.value)}\b", upper, flags=re.IGNORECASE,
        ))
    ]
    candidates = list({entity.entity_key: entity for entity in candidates}.values())
    candidates.sort(key=lambda item: (upper.find(item.value), -len(item.value)))

    resolved_entities: list[EntityReference] = list(candidates)
    ambiguous_aliases: list[tuple[str, list[EntityReference]]] = []
    unresolved_references: list[str] = []
    resolved_values = {entity.value for entity in resolved_entities}
    analyzed_programs = set(programs)
    for token in _explicit_identifier_tokens(question):
        if token == program or token in resolved_values or token in analyzed_programs:
            continue
        alias_matches = [
            entity for entity in eligible_entities if entity.value.startswith(token + "-")
        ]
        alias_matches = list({entity.entity_key: entity for entity in alias_matches}.values())
        if len(alias_matches) == 1:
            resolved_entities.append(alias_matches[0])
            resolved_values.add(alias_matches[0].value)
        elif len(alias_matches) > 1:
            ambiguous_aliases.append((token, alias_matches))
        elif "-" in token:
            resolved_entities.append(
                EntityReference(
                    program=program or "",
                    entity_type="unknown_identifier",
                    value=token,
                    entity_key=f"{program or 'UNKNOWN'}|UNKNOWN|{token}",
                )
            )
            resolved_values.add(token)
        elif len(token) >= 4:
            # A hyphen-less name that matches nothing in the corpus is usually a
            # program the user believes is analyzed. Silently dropping it answered
            # about whichever program happened to be selected instead.
            unresolved_references.append(token)

    if ambiguous_aliases:
        token, matches = ambiguous_aliases[0]
        examples = ", ".join(entity.value for entity in matches[:4])
        return QueryScope(
            program=program,
            entities=tuple(resolved_entities),
            intent=intent,
            confidence=0.35,
            program_source=program_source,
            entity_source="question_unresolved",
            ambiguous=True,
            reason=(
                f"Identifier `{token}` matches multiple analyzed entities"
                f" ({examples}). Use the exact COBOL identifier."
            ),
        )

    # An unresolved bare word only means the question is unanswerable when the
    # question named nothing else. A message that already names something the
    # corpus holds has a subject, and refusing it because a second word looked
    # like a name answers nothing: "is the PERFORM of READ-TAB-SEMAF
    # conditional" resolved the paragraph and was still rejected over the verb.
    # Deciding this by whether anything resolved needs no list of words to
    # ignore, so a COBOL keyword the list never anticipated behaves correctly.
    if unresolved_references and resolved_entities:
        unresolved_references = []

    if unresolved_references:
        named = ", ".join(f"`{value}`" for value in unresolved_references[:3])
        # The search above is scoped to the selected program, so "absent" here
        # only ever meant "absent from that program". Claiming the corpus does
        # not hold it is a stronger statement than the lookup supports, and it
        # was false as soon as a second program existed: with PDCBVC selected,
        # a map belonging to PDB305 was reported as not present anywhere.
        elsewhere = _programs_holding(entities, unresolved_references)
        if elsewhere:
            located = "; ".join(
                f"`{token}` is in {', '.join(sorted(found))}"
                for token, found in elsewhere.items()
            )
            in_scope = f" not in {program}" if program else " not in the selected program"
            return QueryScope(
                program=program,
                programs=tuple(sorted({p for found in elsewhere.values() for p in found})),
                entities=tuple(resolved_entities),
                intent=intent,
                confidence=0.3,
                program_source=program_source,
                entity_source="question_unresolved",
                ambiguous=True,
                reason=(
                    f"{named} is{in_scope}. {located}. "
                    "Name that program to ask about it."
                ),
            )
        return QueryScope(
            program=program,
            programs=tuple(mentioned_programs) if mentioned_programs else (),
            entities=tuple(resolved_entities),
            intent=intent,
            confidence=0.3,
            program_source=program_source,
            entity_source="question_unresolved",
            ambiguous=True,
            reason=(
                f"{named} is not present in the analyzed corpus. "
                "Name an analyzed program or the exact COBOL identifier you want inspected."
            ),
        )

    has_unknown = any(entity.entity_type == "unknown_identifier" for entity in resolved_entities)
    entity_source = (
        "question_unresolved" if has_unknown else "question" if resolved_entities else "unresolved"
    )
    if not resolved_entities and state and _should_reuse_state_entities(question, intent, state):
        resolved_entities = _state_entities_for_followup(question, state)
        if not resolved_entities and _is_singular_entity_reference(question):
            return QueryScope(
                program=program,
                programs=((program,) if program else ()),
                intent=intent,
                confidence=0.35,
                program_source=program_source,
                entity_source="session_ambiguous",
                ambiguous=True,
                reason=(
                    "The previous result contains more than one entity, so the singular "
                    "reference is ambiguous. Name the exact COBOL identifier."
                ),
            )
        entity_source = "session"

    entity = resolved_entities[0] if resolved_entities else None

    confidence = 0.0
    if program_source in {"question", "question_multi", "target"}:
        confidence += 0.45
    elif program_source in {"session", "single_program", "unique_entity"}:
        confidence += 0.30
    if entity_source == "question":
        confidence += 0.50
    elif entity_source == "question_unresolved":
        confidence += 0.25
    elif entity_source == "session":
        confidence += 0.30
    if intent:
        confidence += 0.05

    scope_programs = tuple(mentioned_programs) if mentioned_programs else ((program,) if program else ())
    return QueryScope(
        program=program,
        programs=scope_programs,
        entity_type=entity.entity_type if entity else None,
        entity_value=entity.value if entity else None,
        entity_key=entity.entity_key or None if entity else None,
        entities=tuple(resolved_entities),
        intent=intent,
        confidence=min(confidence, 1.0),
        program_source=program_source,
        entity_source=entity_source,
    )



# Set operations over two programs' evidence. "Which copybooks are shared by A
# and B" is a comparison; requiring the word "compare" rejected it and answered
# "ask for a comparison or choose one program", which is what it had just asked.
_SET_RELATION_WORDS: tuple[tuple[str, str], ...] = (
    (r"\b(?:shared|common|in both|both programs|same in both)\b", "intersection"),
    (r"\b(?:only in|unique to|specific to|not in|absent from|missing from)\b", "difference"),
    (r"\b(?:compare|comparison|difference|differences|versus|vs\.?)\b", "comparison"),
    (r"\b(?:across both|combined|together|in total across)\b", "union"),
)


def set_relation_in(question: str) -> str | None:
    """Which set operation over programs the question asks for, or None."""
    lowered = question.lower()
    for pattern, relation in _SET_RELATION_WORDS:
        if re.search(pattern, lowered):
            return relation
    return None


def _looks_like_comparison(question: str) -> bool:
    return set_relation_in(question) is not None

def contextualize_question(question: str, scope: QueryScope) -> str:
    additions: list[str] = []
    upper = question.upper()
    if scope.program and not _contains_identifier(upper, scope.program):
        additions.append(f"Program: {scope.program}.")
    for entity_value in scope.entity_values:
        if not _contains_identifier(upper, entity_value):
            additions.append(f"Entity: {entity_value}.")
    if not additions:
        return question
    return question.rstrip() + "\nResolved context: " + " ".join(additions)


# A pronoun points back at a previously discussed entity only when it stands in
# for one. Before a present participle it is the subject of a progressive verb
# instead — "how its going", "hows it going" — which is small talk, not a
# question about the last variable. Inheriting an entity there answers a
# greeting with dataflow evidence. "being" is excepted because it is the
# progressive auxiliary of a passive: "where is it being used" does refer back.
_PROGRESSIVE_AFTER_PRONOUN = r"(?!\s+(?!being\b)\w+ing\b)"
_PRONOUN_REFERENCE = re.compile(rf"\b(?:it|them)\b{_PROGRESSIVE_AFTER_PRONOUN}", re.IGNORECASE)
# A possessive refers back only when it qualifies a noun: "its callers", "their
# parameters". Bare "its" before a participle is the misspelt contraction.
_POSSESSIVE_REFERENCE = re.compile(r"\b(?:its|their)\s+(?!\w+ing\b)\w+\b", re.IGNORECASE)
_NAMED_REFERENCE = re.compile(
    r"\b(this one|that one|(?:the|that) same (?:variable|field|call|paragraph)|"
    r"the variable|the field|the call|that call|this call|that paragraph|those paragraphs|"
    r"previously discussed variable|previous variable)\b",
    re.IGNORECASE,
)

_SINGULAR_ENTITY_REFERENCE = re.compile(
    r"\b(?:it|its|this one|that one|(?:the|that) same (?:variable|field|call|paragraph)|"
    r"the variable|the field|the call|that call|this call|that paragraph|"
    r"previously discussed variable|previous variable)\b",
    re.IGNORECASE,
)
_PLURAL_ENTITY_REFERENCE = re.compile(
    r"\b(?:them|their|those|these|those two|the two|both|those paragraphs)\b",
    re.IGNORECASE,
)


def _is_singular_entity_reference(question: str) -> bool:
    return bool(_SINGULAR_ENTITY_REFERENCE.search(question)) and not bool(
        _PLURAL_ENTITY_REFERENCE.search(question)
    )


def _state_entities_for_followup(
    question: str,
    state: SessionState,
) -> list[EntityReference]:
    """Resolve singular and plural references from structured session state."""
    if _is_singular_entity_reference(question):
        if state.focused_entity is not None:
            return [state.focused_entity]
        if len(state.last_result_entities) == 1:
            return list(state.last_result_entities)
        # Backward compatibility for sessions created before focused memory was
        # introduced (and for tests/clients that populate the legacy fields).
        if not state.last_result_entities and state.current_entity_value:
            return [EntityReference(
                program=state.current_program or "",
                entity_type=state.current_entity_type or "unknown",
                value=state.current_entity_value,
                entity_key=state.current_entity_key or "",
            )]
        return []
    if _PLURAL_ENTITY_REFERENCE.search(question):
        return list(state.last_result_entities or state.current_entities)
    return list(state.current_entities)


# A number after "line" is an address, not a quantity. Both forms are required
# to carry a digit so that asking for the source line *field* ("include the
# source line") never resolves to an address.
_LINE_RANGE = re.compile(
    r"\blines?\s+(\d{1,6})\s*(?:-|–|—|to|through|thru|until)\s*(\d{1,6})\b", re.IGNORECASE,
)
_LINE_SINGLE = re.compile(r"\blines?\s+(\d{1,6})\b", re.IGNORECASE)
_SOURCE_MEMBER = re.compile(r"\b([A-Z][A-Z0-9-]{1,30}\.(?:CBL|CPY|COB))\b", re.IGNORECASE)
# "five lines around", "3 lines either side" — a window, not an address.
_CONTEXT_WINDOW = re.compile(
    r"\b(\d{1,3})\s+lines?\s+(?:of\s+)?(?:context|around|either side|before and after|surrounding)",
    re.IGNORECASE,
)


# One address expression: "227", "10-20", "100 to 110".
_ADDRESS_SPAN = re.compile(
    r"(\d{1,6})\s*(?:(?:-|–|—|to|through|thru|until)\s*(\d{1,6}))?", re.IGNORECASE,
)
# The separators that continue a list of addresses after the "line(s)" anchor.
# No leading "^": .match(text, pos) already anchors at pos, while "^" would only
# ever match at the start of the whole question.
_ADDRESS_SEPARATOR = re.compile(r"\s*(?:,|;|&|\+|and\b|ed?\b)\s*", re.IGNORECASE)
_LINE_ANCHOR = re.compile(
    r"\b(?:lines?|line[ae])\b\s*(?:number|no\.?|#)?\s*", re.IGNORECASE,
)


def source_addresses_in(question: str) -> tuple[dict[str, Any], ...]:
    """Read every physical source address a question names, in the order asked.

    The single-address reader this replaces stopped at the first match, so
    "line 227 and 229" quietly became line 227 alone -- the caller had no way to
    tell a one-address question from a two-address question it had truncated.
    Every address the question names is returned, and a caller that cannot
    resolve one is expected to say so rather than drop it.
    """
    text = question or ""
    member = _SOURCE_MEMBER.search(text)
    window = _CONTEXT_WINDOW.search(text)
    context = int(window.group(1)) if window else 0
    consumed_by_window = set()
    if window:
        consumed_by_window = set(range(window.start(), window.end()))

    addresses: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for anchor in _LINE_ANCHOR.finditer(text):
        if anchor.start() in consumed_by_window:
            # "3 lines of context" states a window, not an address to fetch.
            continue
        cursor = anchor.end()
        while True:
            span = _ADDRESS_SPAN.match(text, cursor)
            if not span:
                break
            start = int(span.group(1))
            end = int(span.group(2)) if span.group(2) else start
            cursor = span.end()
            if start >= 1 and end >= 1:
                if end < start:
                    start, end = end, start
                if (start, end) not in seen:
                    seen.add((start, end))
                    addresses.append({
                        "line_start": start,
                        "line_end": end,
                        "source_file": member.group(1).upper() if member else None,
                        "context_before": context,
                        "context_after": context,
                    })
            separator = _ADDRESS_SEPARATOR.match(text, cursor)
            if not separator:
                break
            cursor = separator.end()
    return tuple(addresses)


def source_address_in(question: str) -> dict[str, Any] | None:
    """Read the first physical source address in a question, or None.

    Kept for callers that genuinely want a single address; anything answering a
    user's question should use source_addresses_in so a second address is not
    silently discarded.
    """
    addresses = source_addresses_in(question)
    return addresses[0] if addresses else None


def source_address_entity(program: str | None, address: dict[str, Any]) -> EntityReference:
    """Represent an address as a typed entity so the plan can carry it."""
    start, end = address["line_start"], address["line_end"]
    value = str(start) if start == end else f"{start}-{end}"
    member = address.get("source_file")
    if member:
        value = f"{member}:{value}"
    return EntityReference(
        program=program or "",
        entity_type="source_address",
        value=value,
        entity_key=f"{program or ''}|SOURCE_ADDRESS|{value}",
    )


def named_identifiers_in(question: str) -> tuple[str, ...]:
    """COBOL identifiers written out in a message, in their original casing.

    Exposed so the planner can tell "list them" from "list the copybooks in
    PDRTWA2": a message that names something is starting a new subject, not
    continuing the previous one.
    """
    return _explicit_identifier_tokens(question)


def refers_to_previous_turn(question: str) -> bool:
    """True when a message continues the last one instead of starting fresh.

    Shared with the planner deliberately. This module treated "list them" as a
    continuation while the planner's own narrower test did not, so the session
    topic was never inherited and "how many variables are in PDCBVC?" -> "list
    them" answered with the copybooks. Two views of what a follow-up is are one
    more than the system can keep consistent.
    """
    return _looks_like_followup(question)


def _looks_like_followup(question: str) -> bool:
    q = question.lower().strip()
    return bool(
        _PRONOUN_REFERENCE.search(q)
        or _POSSESSIVE_REFERENCE.search(q)
        or _NAMED_REFERENCE.search(q)
        or _PLURAL_ENTITY_REFERENCE.search(q)
        or re.match(r"^(?:and\s+)?(?:where|what|how|why|when)\s+else\b", q)
        or re.match(r"^(?:and\s+)?(?:what|how)\s+about\b", q)
        or re.match(
            r"^(?:and\s+)?(?:there (?:is|are) more|more\b|continue\b|show (?:me )?the rest\b|"
            r"list the rest\b|you missed\b)",
            q,
        )
    )


def _should_reuse_state_entities(
    question: str,
    intent: str | None,
    state: SessionState,
) -> bool:
    has_remembered_entity = bool(
        state.focused_entity or state.last_result_entities or state.current_entity_value
    )
    if not has_remembered_entity or not _looks_like_followup(question):
        return False
    remembered = (
        state.focused_entity
        or (state.last_result_entities[0] if state.last_result_entities else None)
    )
    entity_type = state.current_entity_type or (
        remembered.entity_type if remembered is not None else "unknown"
    )
    if intent in {None, "general"}:
        return True
    compatible_types = {
        "variable_dataflow": {"variable", "unknown_identifier"},
        "static_values": {"variable", "unknown_identifier"},
        "external_programs": {"call", "unknown_identifier"},
        "copybooks": {"copybook", "unknown_identifier"},
        "control_flow": {"paragraph", "variable", "unknown_identifier"},
        "business_rules": {"paragraph", "variable", "unknown_identifier"},
        "ui_navigation": {"paragraph", "variable", "unknown_identifier"},
        "program_summary": {"variable", "unknown_identifier"},
    }
    return entity_type in compatible_types.get(intent, set())


_NON_ENTITY_IDENTIFIER_TOKENS = {
    "ABEND", "ASKTIME", "CALL", "CICS", "COBOL", "COMMAREA", "COPY",
    "DATA", "DB2", "DIVISION", "END-EXEC", "EXEC", "FORMATTIME", "INCLUDE",
    "JCL", "LINK", "LINKAGE", "ONLY", "PARAGRAPH", "PARAGRAPHS", "PROCEDURE",
    "LENGTH", "READQ", "RECEIVE", "RETURN", "SECTION", "SEND", "SOURCE", "SQL",
    "SYNCPOINT", "WORKING-STORAGE", "WRITEQ", "XCTL",
    # COBOL/SQL syntax named in technical questions is a qualifier, not an
    # identifier the corpus must contain.  Keeping these in the typed grammar
    # prevents requests such as "show the USING parameter" from being rejected
    # as questions about an unknown variable called USING.
    "AND", "CONTENT", "EQUAL", "FROM", "GIVING", "INTO", "MOVE", "REFERENCE",
    "RETURNING", "SELECT", "THEN", "THRU", "USING", "WHEN",
    # Requested output formats and presentation words are not COBOL entities.
    "ARRAY", "COUNT", "CSV", "JSON", "OBJECT", "XML", "YAML",
}


def _explicit_identifier_tokens(question: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)*\b", question):
        if len(token) < 3 or token in _NON_ENTITY_IDENTIFIER_TOKENS or token in tokens:
            continue
        tokens.append(token)
    return tuple(tokens)


def _contains_identifier(text: str, identifier: str) -> bool:
    return bool(
        identifier
        and re.search(
            rf"(?<![A-Z0-9-]){re.escape(identifier.upper())}(?![A-Z0-9-])",
            text,
        )
    )


# MAP('X') / MAPSET('Y') inside an EXEC CICS statement. The platform repo
# carries the same two patterns in its corpus-registry writer; keep them in step.
_MAP_NAME = re.compile(r"\bMAP\s*\(\s*['\"]([A-Z0-9$#@-]{1,8})['\"]\s*\)", re.IGNORECASE)
_MAPSET_NAME = re.compile(r"\bMAPSET\s*\(\s*['\"]([A-Z0-9$#@-]{1,8})['\"]\s*\)", re.IGNORECASE)


@lru_cache(maxsize=8)
def _catalogue(root_text: str) -> tuple[tuple[str, ...], tuple[EntityReference, ...]]:
    if not root_text:
        return (), ()
    root = Path(root_text)
    if not root.exists():
        return (), ()

    registry = _read_json(root / "corpus.registry.json")
    if isinstance(registry, dict) and isinstance(registry.get("programs"), list):
        registry_programs: set[str] = set()
        registry_entities: dict[tuple[str, str, str], EntityReference] = {}
        for item in registry["programs"]:
            if not isinstance(item, dict):
                continue
            program = str(item.get("program", "")).strip().upper()
            if not program:
                continue
            registry_programs.add(program)
            for entity in item.get("entities", []):
                if not isinstance(entity, dict):
                    continue
                entity_type = str(entity.get("type", "unknown")).strip().lower()
                value = str(entity.get("value", "")).strip().upper()
                if not value:
                    continue
                entity_key = str(entity.get("entity_key", "")).strip() or f"{program}|{entity_type.upper()}|{value}"
                _add_entity(registry_entities, program, entity_type, value, entity_key)
        if registry_programs:
            return tuple(sorted(registry_programs)), tuple(registry_entities.values())

    programs: set[str] = set()
    entities: dict[tuple[str, str, str], EntityReference] = {}
    json_paths = sorted(root.rglob("*.json"))
    for path in json_paths:
        filename = path.name
        variable_prefix = "dataflow.variable."
        if filename.startswith(variable_prefix) and filename.endswith(".json"):
            variable = filename[len(variable_prefix) : -5].upper()
            program = _program_from_path_or_payload(path, root)
            if program:
                programs.add(program)
                _add_entity(entities, program, "variable", variable, f"{program}|VARIABLE|{variable}")
            continue

        if filename.startswith("architecture.call.") and filename.endswith(".json"):
            parts = filename[:-5].split(".")
            if len(parts) >= 4:
                encoded_type, target = parts[-2].upper(), parts[-1].upper()
                call_type = "XCTL" if "XCTL" in encoded_type else "LINK" if "LINK" in encoded_type else "CALL"
                program = _program_from_path_or_payload(path, root)
                if program:
                    programs.add(program)
                    _add_entity(entities, program, "call", target, f"{program}|{target}|{call_type}")
            continue

        if filename not in {
            "architecture.copybooks.json",
            "architecture.cics_operations.json",
            "controlflow.cfg.json",
        } and programs:
            continue
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        program = str(payload.get("program") or "").strip().upper()
        if program:
            programs.add(program)
        if filename == "architecture.copybooks.json" and program:
            content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
            names: list[str] = []
            for value in content.values():
                if isinstance(value, list):
                    names.extend(str(item) for item in value if isinstance(item, str))
                elif isinstance(value, dict):
                    for nested in value.values():
                        if isinstance(nested, list):
                            names.extend(str(item) for item in nested if isinstance(item, str))
            for name in names:
                value = name.strip().upper()
                if value:
                    _add_entity(entities, program, "copybook", value, f"{program}|COPYBOOK|{value}")
        if filename == "architecture.cics_operations.json" and program:
            # BMS map and mapset names appear only inside CICS statements, so
            # without this branch a question naming PDCBVC1 resolves to no
            # entity and is refused as absent from the corpus. Kept in step with
            # _write_corpus_registry in the platform repo, which is the path
            # that runs wherever a corpus registry exists.
            content = payload.get("content") if isinstance(payload.get("content"), dict) else {}
            operations = content.get("operations") or []
            for operation in operations if isinstance(operations, list) else []:
                if not isinstance(operation, dict):
                    continue
                statement = str(operation.get("statement", ""))
                for entity_type, pattern in (("map", _MAP_NAME), ("mapset", _MAPSET_NAME)):
                    for match in pattern.finditer(statement):
                        value = match.group(1).strip().upper()
                        if value:
                            _add_entity(
                                entities, program, entity_type, value,
                                f"{program}|{entity_type.upper()}|{value}",
                            )

        if filename == "controlflow.cfg.json" and program:
            nodes = payload.get("nodes") or []
            for node in nodes:
                value = str(node.get("id") if isinstance(node, dict) else node).strip().upper()
                if value:
                    _add_entity(entities, program, "paragraph", value, f"{program}|PARAGRAPH|{value}")

    if not programs and root.name:
        programs.add(root.name.upper())
    return tuple(sorted(programs)), tuple(entities.values())


def _program_from_path_or_payload(path: Path, root: Path) -> str | None:
    payload = _read_json(path)
    if isinstance(payload, dict) and payload.get("program"):
        return str(payload["program"]).strip().upper()
    if root.name:
        return root.name.upper()
    return None


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _add_entity(
    entities: dict[tuple[str, str, str], EntityReference],
    program: str,
    entity_type: str,
    value: str,
    entity_key: str,
) -> None:
    ref = EntityReference(program, entity_type, value, entity_key)
    entities[(program, entity_type, value)] = ref
