from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
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
        self.response_language = "en"
        self.last_sources.clear()
        self.pending_clarification = None
        self.current_plan.clear()

    def clear_entities(self) -> None:
        self.current_entity_type = None
        self.current_entity_value = None
        self.current_entity_key = None
        self.current_entities.clear()


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
    multi_program_comparison = len(mentioned_programs) > 1 and _looks_like_comparison(question)
    if len(mentioned_programs) > 1 and not multi_program_comparison:
        return QueryScope(
            programs=tuple(mentioned_programs),
            intent=intent,
            ambiguous=True,
            reason="More than one analyzed program is named; ask for a comparison or choose one program.",
        )

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
                "The named COBOL entity exists in multiple analyzed programs. "
                "Name the target program explicitly."
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

    if unresolved_references:
        named = ", ".join(f"`{value}`" for value in unresolved_references[:3])
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
        resolved_entities = list(state.current_entities)
        if not resolved_entities and state.current_entity_value:
            resolved_entities = [
                EntityReference(
                    program=state.current_program or program or "",
                    entity_type=state.current_entity_type or "unknown",
                    value=state.current_entity_value,
                    entity_key=state.current_entity_key or "",
                )
            ]
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



def _looks_like_comparison(question: str) -> bool:
    return bool(re.search(r"\b(?:compare|comparison|difference|differences|versus|vs\.?)\b", question.lower()))

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
    r"\b(this one|that one|the same (?:variable|field|call|paragraph)|"
    r"the variable|the field|the call|that call|this call|that paragraph|those paragraphs|"
    r"previously discussed variable|previous variable)\b",
    re.IGNORECASE,
)


def _looks_like_followup(question: str) -> bool:
    q = question.lower().strip()
    return bool(
        _PRONOUN_REFERENCE.search(q)
        or _POSSESSIVE_REFERENCE.search(q)
        or _NAMED_REFERENCE.search(q)
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
    if not state.current_entity_value or not _looks_like_followup(question):
        return False
    entity_type = state.current_entity_type or "unknown"
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
    "READQ", "RECEIVE", "RETURN", "SECTION", "SEND", "SOURCE", "SQL",
    "SYNCPOINT", "WORKING-STORAGE", "WRITEQ", "XCTL",
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


@lru_cache(maxsize=8)
def _catalogue(root_text: str) -> tuple[tuple[str, ...], tuple[EntityReference, ...]]:
    if not root_text:
        return (), ()
    root = Path(root_text)
    if not root.exists():
        return (), ()

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

        if filename not in {"architecture.copybooks.json", "controlflow.cfg.json"} and programs:
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
        if filename == "controlflow.cfg.json" and program:
            nodes = payload.get("nodes") or []
            for node in nodes:
                value = str(node.get("id") if isinstance(node, dict) else node).strip().upper()
                if value and value != program:
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
