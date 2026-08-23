from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Any

from cobol_rag.scope import EntityReference, QueryScope, SessionState


@dataclass(frozen=True)
class EvidenceSubtask:
    """One independently executable and verifiable claim requested by the user."""

    claim_id: str
    description: str
    capability: str
    tasks: tuple[str, ...] = ()
    entity_values: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()
    required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QueryPlan:
    route: str = "technical"
    category: str = "single_source"
    domain: str = "general"
    tasks: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    response_language: str = "en"
    response_language_source: str = "default"
    intent: str = "general"
    program: str | None = None
    programs: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    entities: tuple[EntityReference, ...] = ()
    operations: tuple[str, ...] = ()
    excluded_operations: tuple[str, ...] = ()
    source_domains: tuple[str, ...] = ()
    include_types: tuple[str, ...] = ()
    exclude_types: tuple[str, ...] = ()
    divisions: tuple[str, ...] = ()
    sections: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()
    only_requested_fields: bool = False
    condition_terms: tuple[str, ...] = ()
    negative_condition: bool = False
    result_scope: str = "default"
    explicit_followup: bool = False
    requires_comparison: bool = False
    requires_clarification: bool = False
    confidence: float = 0.0
    planner_source: str = "deterministic"
    subtasks: tuple[EvidenceSubtask, ...] = ()

    @property
    def entity_values(self) -> tuple[str, ...]:
        return tuple(entity.value for entity in self.entities)

    def entity_values_for(self, *entity_types: str) -> tuple[str, ...]:
        allowed = set(entity_types)
        return tuple(entity.value for entity in self.entities if entity.entity_type in allowed)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PlanContractValidation:
    passed: bool
    reasons: tuple[str, ...] = ()


_OPERATION_PATTERNS = {
    "LINK": (r"(?<![A-Z0-9-])LINK(?![A-Z0-9-])",),
    "XCTL": (r"(?<![A-Z0-9-])XCTL(?![A-Z0-9-])",),
    "CALL": (r"\bCOBOL\s+CALL\b", r"\bCALL\s+(?:STATEMENTS?|OPERATIONS?)\b"),
    "RETURN": (r"(?<![A-Z0-9-])RETURN(?![A-Z0-9-])",),
    "ABEND": (r"(?<![A-Z0-9-])ABEND(?![A-Z0-9-])",),
    "SEND": (r"(?<![A-Z0-9-])SEND(?![A-Z0-9-])",),
    "RECEIVE": (r"(?<![A-Z0-9-])RECEIVE(?![A-Z0-9-])",),
    "WRITEQ": (r"(?<![A-Z0-9-])WRITEQ(?![A-Z0-9-])",),
    "READQ": (r"(?<![A-Z0-9-])READQ(?![A-Z0-9-])",),
    "SYNCPOINT": (r"(?<![A-Z0-9-])SYNCPOINT(?![A-Z0-9-])",),
    "ASKTIME": (r"(?<![A-Z0-9-])ASKTIME(?![A-Z0-9-])",),
    "FORMATTIME": (r"(?<![A-Z0-9-])FORMATTIME(?![A-Z0-9-])",),
}

_VARIABLE_PRODUCTION_CUE_PATTERN = (
    r"\b(?:writ(?:e|es|ing|ten)|modif(?:y|ies|ied|ying|ication|ications)|"
    r"set|sets|setting|assign(?:s|ed|ing)?|values?|"
    r"produc(?:e|es|ed|ing)|comput(?:e|es|ed|ing)|calculat(?:e|es|ed|ing)|"
    r"deriv(?:e|es|ed|ing)|generat(?:e|es|ed|ing)|"
    r"results? from|resulting from)\b"
)

_VARIABLE_CONSUMPTION_CUE_PATTERN = (
    r"\b(?:read(?:s|ing)?|test(?:s|ed|ing)?|check(?:s|ed|ing)?|"
    r"us(?:e|es|ed|ing|age)|inspect(?:s|ed|ing)?|verif(?:y|ies|ied|ying)|"
    r"examin(?:e|es|ed|ing))\b"
)

_CALL_AFTER_CUE_PATTERN = (
    r"\bafter\b|\bafterward\b|\bfollowing the call\b|\bfollows? the call\b|"
    r"\bonce\b.{0,40}\b(?:returns?|responds?|completes?|comes? back)\b|"
    r"\breturns? control\b|\bcontrol(?:\s+is)? (?:returned|back)\b|\bcomes? back\b|"
    r"\bgets? control back\b|\bresults? of the call\b|\bresponse of the call\b|"
    r"\bwhat comes back\b|\bwhat happens next\b|"
    r"\bwhat (?:happens|occurs|does (?:it|the program|the call) do)\b.{0,60}"
    r"\b(?:return|respond|complete)s?\b"
)

_CALL_BEFORE_CUE_PATTERN = (
    r"\bbefore\b|\bprior to\b|\bleading up to\b|\bahead of\b|"
    r"\bin preparation for\b|\bbeforehand\b|\bsetup before\b|"
    r"\bgets? ready (?:for|to)\b"
)

_DIVISIONS = (
    "IDENTIFICATION DIVISION",
    "ENVIRONMENT DIVISION",
    "DATA DIVISION",
    "PROCEDURE DIVISION",
)

_SECTIONS = (
    "CONFIGURATION SECTION",
    "INPUT-OUTPUT SECTION",
    "FILE SECTION",
    "WORKING-STORAGE SECTION",
    "LOCAL-STORAGE SECTION",
    "LINKAGE SECTION",
)

_OUTPUT_FIELD_PATTERNS = {
    "name": (r"\bnames?\b",),
    "target": (r"\btarget(?: program)?\b",),
    "paragraph": (r"\bparagraphs?\b",),
    "commarea": (r"\bcommarea\b",),
    # "lines of code" and "how many lines" are size measurements, not a request for
    # the source line where something happens.
    "source_line": (
        r"\bsource lines?\b",
        r"\bline numbers?\b",
        r"(?<!how many )(?<!number of )\blines?\b(?!\s+of\s+code)",
    ),
    "division": (r"\bdivisions?\b",),
    "section": (r"\bsections?\b",),
    "exact_statement": (r"\bexact (?:cobol )?statements?\b",),
    "parameters": (r"\bparameters?\b", r"\barguments?\b"),
    "length": (r"\blength\b",),
    "condition": (r"\bconditions?\b",),
    "action": (r"\bactions?\b",),
    "variables": (r"\bvariables?\b",),
    "artifact": (r"\bartifacts?\b", r"\bsource locations?\b"),
    "origin": (r"\borigin\b", r"\bdefined (?:in|where)\b", r"\bdeclaration\b"),
    "read_sites": (r"\bread(?:s| sites?)?\b", r"\btested\b", r"\binspect(?:s|ed)?\b"),
    "write_sites": (r"\bwrite(?:s| sites?)?\b", r"\bmodified\b", r"\bchanged\b", r"\bset\b"),
    "control_usage": (r"\bcontrols? (?:flow|execution)\b", r"\bcontrol usage\b"),
    "line_count": (
        r"\bhow many (?:physical |source |code )?lines?\b",
        r"\bnumber of (?:physical |source |code )?lines?\b",
        r"\b(?:loc|line count)\b",
    ),
}

_INTENT_SOURCE_DOMAINS = {
    "artifact_inventory": ("artifact_inventory",),
    "variable_inventory": ("dataflow.used_variables",),
    "variable_dataflow": ("dataflow.variable",),
    "copybooks": ("architecture.copybooks",),
    "business_rules": ("business_rule",),
    "external_programs": ("architecture.call_parameters",),
    "control_flow": ("controlflow.cfg", "architecture.cics_operations"),
    "cics_operations": ("architecture.cics_operations",),
    "static_values": ("dataflow.literal_assignments",),
    "dead_code": ("quality.dead_code", "program.comments", "architecture.unused_copybooks"),
    "db2_sql": ("architecture.sqlinclude", "architecture.db2_table"),
    "datasets_tables": ("architecture.db2_table", "jcl.file_io"),
    "program_summary": ("program.summary",),
    "source_metrics": ("program.summary", "program.comments"),
}

_SEMANTIC_OPERATIONS = {
    "describe", "exists", "locate", "list", "trace", "compare", "summarize",
    "explain_condition", "find_reads", "find_writes", "show_context",
}

_ENGLISH_LANGUAGE_MARKERS = frozenset({
    "about", "and", "answer", "are", "can", "could", "do", "does",
    "explain", "good", "hello", "hey", "hi", "how", "is", "list", "morning",
    "night", "please", "reply", "show", "speak", "sure", "tell", "thank",
    "thanks", "the", "what", "where", "which", "why", "write", "you", "your",
})
_ITALIAN_LANGUAGE_MARKERS = frozenset({
    "bene", "buongiorno", "buonanotte", "buonasera", "certo", "che", "chi",
    "ciao", "come", "cosa", "dove", "grazie", "il", "la", "le", "perche",
    "perché", "potresti", "puoi", "quale", "quali", "rispondi", "scrivi",
    "sono", "spiega", "stai", "sto", "tu", "usa",
})


def detect_message_language(message: str) -> str | None:
    """Recognize clear English or Italian wording without interpreting the request."""
    english_score, italian_score = language_marker_scores(message)
    if english_score == italian_score:
        return None
    return "en" if english_score > italian_score else "it"


def language_marker_scores(message: str) -> tuple[int, int]:
    """Return English and Italian marker counts for monolingual reply validation."""
    tokens = re.findall(r"[^\W\d_]+", message.casefold(), flags=re.UNICODE)
    if not tokens:
        return 0, 0
    english_score = sum(token in _ENGLISH_LANGUAGE_MARKERS for token in tokens)
    italian_score = sum(token in _ITALIAN_LANGUAGE_MARKERS for token in tokens)
    return english_score, italian_score


def resolve_response_language(
    message: str,
    state: SessionState | None = None,
) -> tuple[str, str]:
    """Resolve reply language by explicit request, current message, then session."""
    explicit = _explicit_response_language(message)
    if explicit:
        return explicit, "explicit_request"
    detected = detect_message_language(message)
    if detected:
        return detected, "message"
    session_language = str(state.response_language if state else "").strip().lower()
    if re.fullmatch(r"[a-z]{2,3}", session_language):
        return session_language, "session"
    return "en", "default"


def _explicit_response_language(message: str) -> str | None:
    normalized = " ".join(message.casefold().split())
    language_names = {
        "english": "en", "inglese": "en", "italian": "it", "italiano": "it",
    }
    standalone = re.fullmatch(
        r"(english|inglese|italian|italiano)(?:\s+(?:please|per favore))?[.!?]*",
        normalized,
    )
    if standalone:
        return language_names[standalone.group(1)]

    request_patterns = (
        r"^(?:please\s+)?(?:answer|reply|respond|speak|write|continue)"
        r"(?:\s+(?:to\s+)?me)?\s+in\s+(english|italian)\b",
        r"\b(?:can|could|would|will)\s+you\s+(?:please\s+)?"
        r"(?:answer|reply|respond|speak|write|continue)(?:\s+(?:to\s+)?me)?"
        r"\s+in\s+(english|italian)\b",
        r"^(?:per\s+favore\s+)?(?:rispondi|scrivi|parla|continua)"
        r"(?:\s+(?:con|a)\s+me)?\s+(?:in\s+)?(inglese|italiano)\b",
        r"\b(?:puoi|potresti|vorresti)\s+(?:per\s+favore\s+)?"
        r"(?:rispondere|scrivere|parlare|continuare)(?:\s+(?:con|a)\s+me)?"
        r"\s+(?:in\s+)?(inglese|italiano)\b",
        r"^(?:switch|change)\s+(?:the\s+reply\s+language\s+)?to\s+(english|italian)\b",
        r"^(?:passa|cambia)(?:\s+la\s+lingua)?\s+(?:a|in)\s+(inglese|italiano)\b",
    )
    for pattern in request_patterns:
        match = re.search(pattern, normalized)
        if match:
            return language_names[match.group(1)]
    return None

ALLOWED_PLAN_DOMAINS = {
    "program_structure", "dataflow", "control_flow", "integration", "quality",
    "multi_source", "conversation", "general",
}

ALLOWED_PLAN_TASKS = {
    "artifact_inventory", "program_summary", "paragraph_references", "paragraph_body",
    "division_section", "variable_inventory",
    "variable_definition", "variable_reads", "variable_writes",
    "literal_assignments", "variable_comparison", "business_rules",
    "complete_program_flow", "path_from_paragraph", "condition_outcome",
    "external_calls", "call_context", "cics_operations", "copybook_inventory",
    "copybook_usage", "direct_usage_examples", "db2_tables", "sql_includes",
    "jcl_datasets", "commented_code", "unreachable_code", "unused_copybooks",
    "review_copybooks", "pagination_logic", "screen_lineage", "variable_lineage",
    "control_outcome", "variable_composition", "call_option_usage",
    "lineage_terminal", "source_metrics",
}

ALLOWED_PLAN_RELATIONS = {
    "referenced_by", "contains", "starts_at", "ends_at", "before", "after",
    "condition_causes", "reads", "writes", "calls", "uses", "compares",
    "separate_categories", "example_per_item",
}

ALLOWED_EVIDENCE_CAPABILITIES = {
    "artifact_inventory", "program_summary", "source_metrics",
    "variable_inventory",
    "paragraph_evidence", "variable_access", "literal_assignment",
    "variable_lineage", "condition_outcome", "control_flow",
    "call_evidence", "call_context", "cics_evidence", "copybook_evidence",
    "db2_evidence", "jcl_evidence", "quality_evidence",
    "pagination_evidence", "screen_lineage",
}

_CAPABILITY_TASKS = {
    "artifact_inventory": ("artifact_inventory",),
    "program_summary": ("program_summary",),
    "source_metrics": ("source_metrics",),
    "variable_inventory": ("variable_inventory",),
    "paragraph_evidence": ("paragraph_references", "paragraph_body", "division_section"),
    "variable_access": ("variable_definition", "variable_reads", "variable_writes", "variable_comparison"),
    "literal_assignment": ("literal_assignments",),
    "variable_lineage": ("variable_lineage", "variable_composition", "call_option_usage", "lineage_terminal"),
    "condition_outcome": ("business_rules", "condition_outcome", "control_outcome"),
    "control_flow": ("complete_program_flow", "path_from_paragraph"),
    "call_evidence": ("external_calls",),
    "call_context": ("call_context",),
    "cics_evidence": ("cics_operations",),
    "copybook_evidence": ("copybook_inventory", "copybook_usage", "direct_usage_examples"),
    "db2_evidence": ("db2_tables", "sql_includes"),
    "jcl_evidence": ("jcl_datasets",),
    "quality_evidence": ("commented_code", "unreachable_code", "unused_copybooks", "review_copybooks"),
    "pagination_evidence": ("pagination_logic",),
    "screen_lineage": ("screen_lineage",),
}

# Entity types that are exact identifiers copied from the user's question, and so
# must survive into any answer claiming to be about them.
NAMED_IDENTIFIER_ENTITY_TYPES = frozenset({"call", "variable", "unknown_identifier"})

_CAPABILITY_ENTITY_TYPES = {
    "paragraph_evidence": {"paragraph"},
    "variable_access": {"variable", "unknown_identifier"},
    "literal_assignment": {"variable", "unknown_identifier"},
    "variable_lineage": {"variable", "unknown_identifier"},
    "condition_outcome": {"variable", "paragraph", "unknown_identifier"},
    "control_flow": {"paragraph"},
    "call_evidence": {"call"},
    "call_context": {"call"},
    "cics_evidence": {"paragraph"},
    "copybook_evidence": {"copybook"},
    "screen_lineage": {"variable", "unknown_identifier"},
}

_INTENT_DOMAIN = {
    "artifact_inventory": "program_structure",
    "program_summary": "program_structure",
    "variable_inventory": "dataflow",
    "variable_dataflow": "dataflow",
    "static_values": "dataflow",
    "business_rules": "control_flow",
    "control_flow": "control_flow",
    "external_programs": "integration",
    "cics_operations": "integration",
    "copybooks": "integration",
    "db2_sql": "integration",
    "datasets_tables": "integration",
    "dead_code": "quality",
    "ui_navigation": "program_structure",
    "source_metrics": "program_structure",
    "general": "general",
}


def build_query_plan(
    question: str,
    scope: QueryScope,
    *,
    intent: str | None = None,
    state: SessionState | None = None,
) -> QueryPlan:
    q = question.lower()
    upper = question.upper()
    resolved_intent = intent or scope.intent or "general"
    explicit_followup = _is_explicit_followup(q)
    if explicit_followup and resolved_intent == "general" and state and state.current_intent:
        resolved_intent = state.current_intent

    if any(term in q for term in ("return value", "value returned")):
        resolved_intent = "variable_dataflow"
    elif _is_source_metrics_question(q):
        resolved_intent = "source_metrics"
    elif (
        resolved_intent in {"general", "control_flow", "ui_navigation"}
        and any(term in q for term in ("pagination", "page navigation"))
    ):
        resolved_intent = "control_flow"
    elif _is_condition_effect_question(q) and resolved_intent in {
        "general", "variable_dataflow", "business_rules", "control_flow"
    }:
        resolved_intent = "business_rules"
    elif (
        re.search(r"\b(?:call|calls|called)\b", q)
        and any(entity.entity_type == "call" for entity in scope.entities)
    ) or any(term in q for term in ("commarea", "target program")):
        resolved_intent = "external_programs"
    elif (
        resolved_intent in {"general", "datasets_tables", "db2_sql"}
        and any(term in q for term in ("sql include", "sqlinclude", "db2"))
    ):
        resolved_intent = "db2_sql"
    elif (
        resolved_intent in {"general", "copybooks"}
        and ("copy statement" in q or "copybook" in q or "copy book" in q)
    ):
        resolved_intent = "copybooks"

    entity_types = {entity.entity_type for entity in scope.entities}
    has_exact_variable = "variable" in entity_types
    # Program and entity resolution is authoritative. The semantic model may refine
    # what evidence is wanted, but an exact variable must not be diverted to a
    # whole-program control-flow or literal-assignment handler. Condition/outcome
    # questions deliberately remain business-rule questions.
    if (
        has_exact_variable
        and not _is_condition_effect_question(q)
        and resolved_intent not in {
            "program_summary", "business_rules", "artifact_inventory",
            "source_metrics", "copybooks", "dead_code",
        }
    ):
        resolved_intent = "variable_dataflow"
    if resolved_intent == "general" and entity_types:
        if entity_types <= {"variable", "unknown_identifier"}:
            resolved_intent = "variable_dataflow"
        elif entity_types <= {"call", "unknown_identifier"}:
            resolved_intent = "external_programs"
        elif entity_types <= {"copybook", "unknown_identifier"}:
            resolved_intent = "copybooks"

    operations = tuple(
        operation
        for operation, patterns in _OPERATION_PATTERNS.items()
        if any(re.search(pattern, upper, flags=re.IGNORECASE) for pattern in patterns)
    )
    if any(term in q for term in ("return value", "value returned")):
        operations = tuple(operation for operation in operations if operation != "RETURN")

    semantic_operations: list[str] = []
    if re.search(r"\b(?:compare|comparison|difference|differences|versus|vs\.?)\b", q):
        semantic_operations.append("compare")
    if re.search(r"\b(?:does|do|is|are)\b.{0,80}\b(?:exist|exists|present|available)\b", q):
        semantic_operations.append("exists")
    if re.search(r"\b(?:what is|what are|describe|explain|tell me about)\b", q) or re.search(
        r"\bhow\b.{0,80}\b(?:use|uses|used|handle|handles)\b", q
    ):
        semantic_operations.append("describe")
    if re.search(r"\b(?:where|locate|location|which paragraph|source line)\b", q):
        semantic_operations.append("locate")
    if re.search(r"\b(?:read|reads|test|tests|tested|check|checks|checked|inspect|inspects|inspected|used|uses|usage)\b", q):
        semantic_operations.append("find_reads")
    if re.search(r"\b(?:write|writes|written|modified|modify|changed|set|assigned)\b", q):
        semantic_operations.append("find_writes")
    if re.search(r"\b(?:list|show every|show all)\b", q):
        semantic_operations.append("list")
    if re.search(r"\b(?:summary|summarize|overview)\b", q):
        semantic_operations.append("summarize")
    if _is_condition_effect_question(q):
        semantic_operations.append("explain_condition")
    operations = _unique((*operations, *semantic_operations))
    if any(value.upper() == value for value in operations):
        operations = tuple(value for value in operations if value != "list")

    negative_text, positive_text = _split_negative_constraints(q)
    excluded_operations = tuple(
        operation
        for operation, patterns in _OPERATION_PATTERNS.items()
        if any(re.search(pattern, negative_text.upper(), flags=re.IGNORECASE) for pattern in patterns)
    )
    operations = tuple(operation for operation in operations if operation not in excluded_operations)
    include_types: list[str] = []
    exclude_types: list[str] = []
    type_aliases = {
        "sql_include": ("sql include", "sqlinclude"),
        "db2_table": ("db2 table", "db2 tables", "table", "tables"),
        "dependency": ("dependency", "dependencies"),
        "copy_statement": ("copy statement", "copy statements"),
    }
    for evidence_type, aliases in type_aliases.items():
        if any(alias in negative_text for alias in aliases):
            exclude_types.append(evidence_type)
        if any(alias in positive_text for alias in aliases):
            include_types.append(evidence_type)

    if resolved_intent == "db2_sql":
        if "sql_include" in include_types and "db2_table" not in include_types:
            exclude_types.append("db2_table")
        if "db2_table" in include_types and "sql_include" not in include_types:
            exclude_types.append("sql_include")

    divisions = tuple(division for division in _DIVISIONS if division in upper)
    sections = tuple(section for section in _SECTIONS if section in upper)
    output_fields = tuple(
        field
        for field, patterns in _OUTPUT_FIELD_PATTERNS.items()
        if any(re.search(pattern, q) for pattern in patterns)
    )
    only_requested_fields = bool(
        re.search(
            r"(?:return|provide|give|include|show)\s+only\s+(?:their\s+|the\s+)?"
            r"(?:names?|targets?|paragraphs?|source|lines?|divisions?|sections?|parameters?|commarea)",
            q,
        )
        or re.search(
            r"only\s+include\s+(?:their\s+|the\s+)?"
            r"(?:names?|targets?|paragraphs?|source|lines?|divisions?|sections?|parameters?|commarea)",
            q,
        )
    )

    condition_terms = _condition_terms(question) if _is_condition_effect_question(q) else ()
    negative_condition = bool(
        _is_condition_effect_question(q)
        and re.search(r"\b(?:neither|otherwise|not equal|not equals|is not|isn.t|not =)\b", q)
    )
    result_scope = "all" if _requests_exhaustive_results(q) else "default"
    if explicit_followup and state and state.current_plan:
        previous = state.current_plan
        if not operations:
            operations = tuple(str(value) for value in previous.get("operations", []))
        if not include_types and resolved_intent == str(previous.get("intent", "")):
            include_types = [str(value) for value in previous.get("include_types", [])]
        if not exclude_types and resolved_intent == str(previous.get("intent", "")):
            exclude_types = [str(value) for value in previous.get("exclude_types", [])]
        if result_scope == "default":
            previous_scope = str(previous.get("result_scope", "default"))
            if previous_scope in {"default", "all"}:
                result_scope = previous_scope

    programs = tuple(getattr(scope, "programs", ()) or ((scope.program,) if scope.program else ()))
    requires_comparison = "compare" in operations or len(programs) > 1
    category = "multi_source_comparison" if requires_comparison else "single_source"
    source_domains = _INTENT_SOURCE_DOMAINS.get(resolved_intent, ())
    if resolved_intent == "program_summary" and scope.entities:
        source_domains = _unique((*source_domains, "dataflow.variable"))
        category = "multi_source_synthesis"
    unresolved_previous_variable = bool(
        re.search(r"\b(?:previously discussed|previous|earlier) variable\b", q)
        and "without" not in q
        and not any(entity.entity_type == "variable" for entity in scope.entities)
    )
    confidence = 0.95 if resolved_intent != "general" else 0.45
    if not operations:
        confidence = min(confidence, 0.7)
    domain = _INTENT_DOMAIN.get(resolved_intent, "general")
    tasks, relations = _fallback_tasks_and_relations(q, resolved_intent)
    if any(entity.entity_type == "call" for entity in scope.entities):
        if re.search(r"\bbefore\b", q):
            relations = _unique((*relations, "before"))
        if re.search(r"\bafter\b", q):
            relations = _unique((*relations, "after"))
    response_language, response_language_source = resolve_response_language(question, state)

    base_plan = QueryPlan(
        route="technical",
        category=category,
        domain=domain,
        tasks=tasks,
        relations=relations,
        response_language=response_language,
        response_language_source=response_language_source,
        intent=resolved_intent,
        program=scope.program,
        programs=programs,
        entities=scope.entities,
        operations=_unique(operations),
        excluded_operations=_unique(excluded_operations),
        source_domains=_unique(source_domains),
        include_types=_unique(include_types),
        exclude_types=_unique(exclude_types),
        divisions=divisions,
        sections=sections,
        output_fields=output_fields,
        only_requested_fields=only_requested_fields,
        condition_terms=condition_terms,
        negative_condition=negative_condition,
        result_scope=result_scope,
        explicit_followup=explicit_followup,
        requires_comparison=requires_comparison,
        requires_clarification=unresolved_previous_variable,
        confidence=confidence,
        planner_source="deterministic",
    )
    return replace(base_plan, subtasks=derive_evidence_subtasks(base_plan))


def plan_needs_semantic_refinement(question: str, plan: QueryPlan) -> bool:
    # The LLM is the primary semantic planner. Deterministic preprocessing only
    # supplies verified program/entity scope and literal constraints.
    return plan.route == "technical" and not plan.requires_clarification


def merge_semantic_plan(plan: QueryPlan, update: dict[str, Any]) -> QueryPlan:
    allowed_routes = {"technical", "conversational", "unclear", "out_of_scope"}
    allowed_categories = {
        "single_source", "multi_source_synthesis", "multi_source_comparison",
        "clarification", "conversational", "out_of_scope",
    }
    route = str(update.get("route", plan.route)).strip().lower()
    if route not in allowed_routes:
        route = plan.route
    category = str(update.get("category", plan.category)).strip().lower()
    if category not in allowed_categories:
        category = plan.category
    proposed_intent = str(update.get("intent", plan.intent)).strip().lower().replace("-", "_").replace(" ", "_")
    if proposed_intent == "datasets":
        proposed_intent = "datasets_tables"
    allowed_intents = set(_INTENT_SOURCE_DOMAINS) | {"ui_navigation", "source_metrics", "general"}
    intent = proposed_intent if proposed_intent in allowed_intents else plan.intent
    proposed_operations = [
        str(value).strip().lower()
        for value in update.get("operations", [])
        if str(value).strip().lower() in _SEMANTIC_OPERATIONS
    ]
    # An exclusion removes real evidence from the answer, so it may only come from
    # the user's own words. The deterministic layer extracts exclusions from
    # explicit negative phrasing; a planner-invented one silently drops calls the
    # user asked to see, which is indistinguishable from missing analysis.
    excluded_operations = plan.excluded_operations
    explicit_operations = [
        value for value in plan.operations
        if value.upper() == value and value not in excluded_operations
    ]
    operations = _unique((*explicit_operations, *proposed_operations))
    allowed_domains = {domain for values in _INTENT_SOURCE_DOMAINS.values() for domain in values}
    source_domains = _unique((*_INTENT_SOURCE_DOMAINS.get(intent, ()), *(
        str(value) for value in update.get("source_domains", []) if str(value) in allowed_domains
    )))
    domain = str(update.get("domain", _INTENT_DOMAIN.get(intent, plan.domain))).strip().lower()
    if domain not in ALLOWED_PLAN_DOMAINS:
        domain = _INTENT_DOMAIN.get(intent, plan.domain)
    semantic_tasks = _unique(value for value in update.get("tasks", []) if str(value) in ALLOWED_PLAN_TASKS)
    semantic_relations = _unique(value for value in update.get("relations", []) if str(value) in ALLOWED_PLAN_RELATIONS)
    tasks = _unique((*plan.tasks, *semantic_tasks))
    relations = _unique((*plan.relations, *semantic_relations))
    has_exact_variable = bool(plan.entity_values_for("variable"))
    if has_exact_variable and plan.intent == "variable_dataflow":
        # The LLM remains the semantic guide, but cannot replace verified variable
        # scope with an incompatible program-wide task.
        variable_tasks = {
            "variable_definition", "variable_reads", "variable_writes",
            "variable_comparison", "variable_lineage", "literal_assignments",
            "control_outcome", "variable_composition", "call_option_usage",
            "lineage_terminal",
        }
        tasks = _unique(task for task in tasks if task in variable_tasks)
        if not tasks:
            tasks = ("variable_definition", "variable_reads", "variable_writes")
        intent = "variable_dataflow"
        domain = "dataflow"
        source_domains = _unique(("dataflow.variable", *(
            value for value in source_domains
            if value in {
                "dataflow.literal_assignments", "business_rule",
                "architecture.call_parameters", "integration.paragraph_context",
            }
        )))
    if plan.intent == "source_metrics":
        # Source-size questions are a verified program-metrics capability. Keep the
        # semantic router as the guide, but do not allow a vague summary/general
        # label to discard the explicit count request.
        intent = "source_metrics"
        domain = "program_structure"
        tasks = ("source_metrics",)
        source_domains = _INTENT_SOURCE_DOMAINS["source_metrics"]
    # Language is a deterministic conversation contract. The semantic planner writes
    # the reply, but it cannot override an explicit/current-message/session decision.
    response_language = plan.response_language
    output_fields = _unique((*plan.output_fields, *(
        str(value) for value in update.get("output_fields", []) if str(value) in _OUTPUT_FIELD_PATTERNS
    )))
    requires_comparison = bool(
        plan.requires_comparison or update.get("requires_comparison")
        or "compare" in operations or len(plan.programs) > 1
    )
    if requires_comparison:
        category = "multi_source_comparison"
    try:
        confidence = max(0.0, min(float(update.get("confidence", plan.confidence)), 1.0))
    except (TypeError, ValueError):
        confidence = plan.confidence
    merged = replace(
        plan, route=route, category=category, domain=domain,
        tasks=tasks,
        relations=relations,
        response_language=response_language,
        response_language_source=plan.response_language_source,
        intent=intent, operations=operations,
        excluded_operations=excluded_operations,
        source_domains=source_domains or _INTENT_SOURCE_DOMAINS.get(intent, ()),
        output_fields=output_fields, requires_comparison=requires_comparison,
        result_scope=plan.result_scope,
        requires_clarification=bool(plan.requires_clarification or update.get("requires_clarification")),
        confidence=confidence, planner_source="hybrid_llm",
    )
    semantic_subtasks = _parse_semantic_subtasks(merged, update.get("subtasks"))
    return replace(
        merged,
        subtasks=_ensure_subtask_coverage(merged, semantic_subtasks),
    )


def derive_evidence_subtasks(plan: QueryPlan) -> tuple[EvidenceSubtask, ...]:
    """Convert the global plan into independently verifiable evidence claims."""
    remaining = list(plan.tasks)
    subtasks: list[EvidenceSubtask] = []
    for capability, supported_tasks in _CAPABILITY_TASKS.items():
        selected = tuple(task for task in remaining if task in supported_tasks)
        if not selected:
            continue
        remaining = [task for task in remaining if task not in selected]
        entities = _entities_for_capability(plan, capability)
        subtasks.append(
            EvidenceSubtask(
                claim_id=f"claim_{len(subtasks) + 1}",
                description=_subtask_description(capability, entities, plan.program),
                capability=capability,
                tasks=selected,
                entity_values=entities,
                relations=_relations_for_capability(plan.relations, capability),
                source_domains=_source_domains_for_capability(plan, capability),
                output_fields=plan.output_fields,
            )
        )
    for task in remaining:
        subtasks.append(
            EvidenceSubtask(
                claim_id=f"claim_{len(subtasks) + 1}",
                description=f"Execute and verify requested task {task}",
                capability="program_summary",
                tasks=(task,),
                entity_values=plan.entity_values,
                relations=plan.relations,
                source_domains=plan.source_domains,
                output_fields=plan.output_fields,
            )
        )

    call_values = plan.entity_values_for("call")
    temporal = tuple(value for value in plan.relations if value in {"before", "after"})
    if call_values and temporal and not any(task.capability == "call_context" for task in subtasks):
        subtasks.append(
            EvidenceSubtask(
                claim_id=f"claim_{len(subtasks) + 1}",
                description=_subtask_description("call_context", call_values, plan.program),
                capability="call_context",
                tasks=("call_context",),
                entity_values=call_values,
                relations=temporal,
                source_domains=("architecture.call_parameters",),
                output_fields=_unique((*plan.output_fields, "source_line")),
            )
        )
    return tuple(subtasks)


def plan_for_subtask(plan: QueryPlan, subtask: EvidenceSubtask) -> QueryPlan:
    """Create a narrow executable plan whose contract covers only one claim."""
    selected_values = set(subtask.entity_values)
    entities = tuple(
        entity for entity in plan.entities
        if not selected_values or entity.value in selected_values
    )
    intent, domain = _capability_route(subtask.capability, entities)
    return replace(
        plan,
        route="technical",
        category=("multi_source_comparison" if len(plan.programs) > 1 else "single_source"),
        domain=domain,
        intent=intent,
        tasks=subtask.tasks,
        relations=subtask.relations,
        entities=entities,
        source_domains=subtask.source_domains or plan.source_domains,
        output_fields=subtask.output_fields,
        requires_comparison=("compare" in plan.operations or len(plan.programs) > 1),
        requires_clarification=False,
        subtasks=(),
    )


def _parse_semantic_subtasks(
    plan: QueryPlan,
    raw_subtasks: Any,
) -> tuple[EvidenceSubtask, ...]:
    if not isinstance(raw_subtasks, list):
        return ()
    available_entities = {entity.value for entity in plan.entities}
    allowed_sources = {source for values in _INTENT_SOURCE_DOMAINS.values() for source in values}
    parsed: list[EvidenceSubtask] = []
    for index, raw in enumerate(raw_subtasks[:12], start=1):
        if not isinstance(raw, dict):
            continue
        capability = str(raw.get("capability", "")).strip().lower()
        if capability not in ALLOWED_EVIDENCE_CAPABILITIES:
            continue
        supported = set(_CAPABILITY_TASKS[capability])
        raw_tasks = raw.get("tasks", [])
        if isinstance(raw_tasks, str):
            raw_tasks = [raw_tasks]
        tasks = _unique(
            str(task) for task in raw_tasks
            if str(task) in ALLOWED_PLAN_TASKS and str(task) in supported
        )
        if not tasks:
            tasks = tuple(task for task in plan.tasks if task in supported)
        if not tasks:
            tasks = (_CAPABILITY_TASKS[capability][0],)
        raw_entities = raw.get("entity_values", [])
        if isinstance(raw_entities, str):
            raw_entities = [raw_entities]
        entities = _unique(
            str(value) for value in raw_entities
            if str(value) in available_entities
        )
        if not entities:
            entities = _entities_for_capability(plan, capability)
        raw_relations = raw.get("relations", [])
        if isinstance(raw_relations, str):
            raw_relations = [raw_relations]
        requested_relations = _unique(
            str(value) for value in raw_relations
            if str(value) in ALLOWED_PLAN_RELATIONS
        ) or plan.relations
        relations = _relations_for_capability(requested_relations, capability)
        raw_sources = raw.get("source_domains", [])
        if isinstance(raw_sources, str):
            raw_sources = [raw_sources]
        capability_sources = set(_source_domains_for_capability(plan, capability))
        sources = _unique(
            str(value) for value in raw_sources
            if str(value) in allowed_sources and str(value) in capability_sources
        )
        raw_fields = raw.get("output_fields", [])
        if isinstance(raw_fields, str):
            raw_fields = [raw_fields]
        fields = _unique(str(value) for value in raw_fields if str(value) in _OUTPUT_FIELD_PATTERNS)
        description = str(raw.get("description") or "").strip()[:500]
        parsed.append(
            EvidenceSubtask(
                claim_id=f"claim_{len(parsed) + 1}",
                description=description or _subtask_description(capability, entities, plan.program),
                capability=capability,
                tasks=tasks,
                entity_values=entities,
                relations=relations,
                source_domains=sources or _source_domains_for_capability(plan, capability),
                output_fields=fields or plan.output_fields,
                required=bool(raw.get("required", True)),
            )
        )
    return tuple(parsed)


def _ensure_subtask_coverage(
    plan: QueryPlan,
    semantic_subtasks: tuple[EvidenceSubtask, ...],
) -> tuple[EvidenceSubtask, ...]:
    fallback = derive_evidence_subtasks(plan)
    if not semantic_subtasks:
        return fallback
    merged = list(semantic_subtasks)
    for required in fallback:
        matching = [item for item in merged if item.capability == required.capability]
        covered_tasks = {task for item in matching for task in item.tasks}
        covered_entities = {value for item in matching for value in item.entity_values}
        missing_tasks = tuple(task for task in required.tasks if task not in covered_tasks)
        missing_entities = tuple(value for value in required.entity_values if value not in covered_entities)
        if missing_tasks or (required.entity_values and missing_entities):
            merged.append(
                replace(
                    required,
                    claim_id=f"claim_{len(merged) + 1}",
                    tasks=missing_tasks or required.tasks,
                    entity_values=missing_entities or required.entity_values,
                )
            )
    merged = _remove_subsumed_subtasks(merged)
    return tuple(replace(item, claim_id=f"claim_{index}") for index, item in enumerate(merged, start=1))


def _remove_subsumed_subtasks(
    subtasks: list[EvidenceSubtask],
) -> list[EvidenceSubtask]:
    """Remove duplicate claims and narrower write claims already proven as literals."""
    unique: list[EvidenceSubtask] = []
    seen: set[tuple[Any, ...]] = set()
    for item in subtasks:
        key = (
            item.capability, item.tasks, item.entity_values, item.relations,
            item.source_domains, item.output_fields, item.required,
        )
        if key not in seen:
            seen.add(key)
            unique.append(item)

    literal_entity_sets = [
        set(item.entity_values) for item in unique
        if item.capability == "literal_assignment" and item.entity_values
    ]
    unique = [
        item for item in unique
        if not (
            item.capability == "variable_access"
            and set(item.tasks) <= {"variable_writes"}
            and item.entity_values
            and any(set(item.entity_values) <= values for values in literal_entity_sets)
        )
    ]
    return _merge_whole_program_capability_claims(unique)


def _merge_whole_program_capability_claims(
    subtasks: list[EvidenceSubtask],
) -> list[EvidenceSubtask]:
    """Collapse repeated whole-program claims that share one evidence capability.

    Without an entity to distinguish them, several claims against the same
    capability are one claim about the same artifact family rather than
    independently verifiable requests. Merging their tasks preserves every
    requested phase while removing the duplicate execution and generation pass.
    """
    merged: list[EvidenceSubtask] = []
    index_by_capability: dict[str, int] = {}
    for item in subtasks:
        if item.entity_values:
            merged.append(item)
            continue
        existing_index = index_by_capability.get(item.capability)
        if existing_index is None:
            index_by_capability[item.capability] = len(merged)
            merged.append(item)
            continue
        existing = merged[existing_index]
        merged[existing_index] = replace(
            existing,
            tasks=_unique((*existing.tasks, *item.tasks)),
            relations=_unique((*existing.relations, *item.relations)),
            source_domains=_unique((*existing.source_domains, *item.source_domains)),
            output_fields=_unique((*existing.output_fields, *item.output_fields)),
            required=existing.required or item.required,
        )
    return merged


def _entities_for_capability(plan: QueryPlan, capability: str) -> tuple[str, ...]:
    allowed_types = _CAPABILITY_ENTITY_TYPES.get(capability)
    if not allowed_types:
        return ()
    return tuple(
        entity.value for entity in plan.entities
        if entity.entity_type in allowed_types
    )


def _relations_for_capability(
    relations: tuple[str, ...], capability: str,
) -> tuple[str, ...]:
    allowed = {
        "paragraph_evidence": {"referenced_by", "contains"},
        "variable_access": {"reads", "writes", "compares"},
        "literal_assignment": {"writes"},
        "variable_lineage": {"reads", "writes", "uses", "compares", "before", "after"},
        "condition_outcome": {"condition_causes"},
        "control_flow": {"starts_at", "ends_at", "condition_causes"},
        "call_evidence": {"calls", "uses"},
        "call_context": {"before", "after", "calls", "uses"},
        "copybook_evidence": {"uses", "example_per_item"},
        "quality_evidence": {"separate_categories"},
    }.get(capability, set(ALLOWED_PLAN_RELATIONS))
    return tuple(value for value in relations if value in allowed)


def _source_domains_for_capability(plan: QueryPlan, capability: str) -> tuple[str, ...]:
    mapping = {
        "artifact_inventory": ("artifact_inventory",),
        "program_summary": ("program.summary",),
        "source_metrics": ("program.summary", "program.comments"),
        "variable_inventory": ("dataflow.used_variables",),
        "paragraph_evidence": ("controlflow.cfg",),
        "variable_access": ("dataflow.variable",),
        "literal_assignment": ("dataflow.literal_assignments", "dataflow.variable"),
        "variable_lineage": ("dataflow.variable", "architecture.call_parameters"),
        "condition_outcome": ("business_rule", "dataflow.variable"),
        "control_flow": ("controlflow.cfg", "architecture.cics_operations"),
        "call_evidence": ("architecture.call_parameters",),
        "call_context": ("architecture.call_parameters", "dataflow.variable"),
        "cics_evidence": ("architecture.cics_operations",),
        "copybook_evidence": ("architecture.copybooks",),
        "db2_evidence": ("architecture.db2_table", "architecture.sqlinclude"),
        "jcl_evidence": ("jcl.file_io",),
        "quality_evidence": ("quality.dead_code", "program.comments", "architecture.unused_copybooks"),
        "pagination_evidence": ("controlflow.cfg", "dataflow.variable"),
        "screen_lineage": ("dataflow.variable",),
    }
    return mapping.get(capability, plan.source_domains)


def _capability_route(
    capability: str,
    entities: tuple[EntityReference, ...],
) -> tuple[str, str]:
    mapping = {
        "artifact_inventory": ("artifact_inventory", "program_structure"),
        "program_summary": ("program_summary", "program_structure"),
        "source_metrics": ("source_metrics", "program_structure"),
        "variable_inventory": ("variable_inventory", "dataflow"),
        "paragraph_evidence": ("control_flow", "control_flow"),
        "variable_access": ("variable_dataflow", "dataflow"),
        "literal_assignment": ("static_values", "dataflow"),
        "variable_lineage": ("variable_dataflow", "dataflow"),
        "condition_outcome": ("business_rules", "control_flow"),
        "control_flow": ("control_flow", "control_flow"),
        "call_evidence": ("external_programs", "integration"),
        "call_context": ("external_programs", "integration"),
        "cics_evidence": ("cics_operations", "integration"),
        "copybook_evidence": ("copybooks", "integration"),
        "db2_evidence": ("db2_sql", "integration"),
        "jcl_evidence": ("datasets_tables", "integration"),
        "quality_evidence": ("dead_code", "quality"),
        "pagination_evidence": ("control_flow", "control_flow"),
        "screen_lineage": ("ui_navigation", "program_structure"),
    }
    intent, domain = mapping.get(capability, ("general", "general"))
    return intent, domain


def _subtask_description(
    capability: str,
    entities: tuple[str, ...],
    program: str | None,
) -> str:
    target = ", ".join(entities) if entities else (program or "the selected program")
    return f"Verify {capability.replace('_', ' ')} evidence for {target}"


def validate_plan_answer(plan: QueryPlan, answer: str) -> PlanContractValidation:
    if not answer.strip():
        return PlanContractValidation(False, ("empty_answer",))
    lowered = answer.lower()
    reasons: list[str] = []

    forbidden_markers = {
        "db2_table": ("db2 table ",),
        "sql_include": ("sql include", "sql includes"),
        "dependency": ("dependency", "dependencies"),
    }
    for excluded in plan.exclude_types:
        if any(marker in lowered for marker in forbidden_markers.get(excluded, ())):
            reasons.append(f"excluded_type_present:{excluded}")

    if plan.divisions and "copy statements" in lowered:
        allowed = {division.lower() for division in plan.divisions}
        for line in answer.splitlines():
            if not line.lstrip().startswith("-"):
                continue
            present = {division.lower() for division in _DIVISIONS if division.lower() in line.lower()}
            if present and present.isdisjoint(allowed):
                reasons.append("outside_requested_division")
                break

    # Evidence offered as being about a named identifier must mention that
    # identifier. This holds for every capability, not just the two that read
    # entities most often: a claim scoped to one entity that comes back with
    # program-wide records is not evidence about that entity, whatever intent
    # the planner chose. Matching is a substring test on purpose, so a group
    # item is satisfied by any of its qualified children while a question about
    # a specific field is not satisfied by the group alone.
    for entity in plan.entities:
        if entity.entity_type in NAMED_IDENTIFIER_ENTITY_TYPES and entity.value.lower() not in lowered:
            reasons.append(f"missing_requested_entity:{entity.value}")

    # Program-level capabilities describe a whole program rather than a location in
    # it, so requiring a source line from them rejects a correct answer for a field
    # the capability never renders.
    program_level_tasks = {
        "program_summary", "source_metrics", "artifact_inventory", "variable_inventory",
    }
    renders_source_lines = not plan.tasks or not set(plan.tasks) <= program_level_tasks
    if (
        "source_line" in plan.output_fields
        and renders_source_lines
        and not (re.search(r"\blines?\b", lowered) or "line unavailable" in lowered)
    ):
        reasons.append("missing_requested_field:source_line")

    required_sections = {
        "variable_reads": ("tested/read at:", "read sites:"),
        "variable_writes": ("modified at:", "write sites:"),
        "variable_lineage": ("lineage:", "downstream lineage:"),
        "control_outcome": ("resulting control-flow actions:",),
        "variable_composition": ("construction/context evidence:",),
        "call_option_usage": ("cics/call option usage:",),
        "lineage_terminal": ("lineage completion:",),
    }
    for task, markers in required_sections.items():
        if task in plan.tasks and not any(marker in lowered for marker in markers):
            reasons.append(f"missing_requested_section:{task}")
    if "control_usage" in plan.output_fields and "control-flow use:" not in lowered:
        reasons.append("missing_requested_field:control_usage")
    if "line_count" in plan.output_fields and not re.search(
        r"\b\d+\s+(?:total physical source lines|loc|lines? of code)\b", lowered,
    ):
        reasons.append("missing_requested_field:line_count")

    if plan.result_scope == "all":
        if re.search(r"\bshowing (?:only )?(?:the )?(?:first|top)\b", lowered):
            reasons.append("exhaustive_result_truncated")
        if "literal_assignments" in plan.tasks:
            count_match = re.search(r":\s*(\d+)\s+matching item\(s\)\.", answer)
            if not count_match:
                reasons.append("missing_exhaustive_result_count")
            else:
                returned = sum(1 for line in answer.splitlines() if line.startswith("- line "))
                if returned != int(count_match.group(1)):
                    reasons.append(
                        f"exhaustive_result_count_mismatch:{returned}/{count_match.group(1)}"
                    )

    return PlanContractValidation(not reasons, tuple(dict.fromkeys(reasons)))



def _fallback_tasks_and_relations(q: str, intent: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Capture literal task/relation constraints that a semantic planner may not drop."""
    tasks: list[str] = []
    relations: list[str] = []

    if intent == "control_flow":
        if "pagination" in q or "page navigation" in q:
            tasks.append("pagination_logic")
        if "referenc" in q or "mention" in q:
            tasks.append("paragraph_references")
            relations.append("referenced_by")
        if "what statements" in q or ("what does" in q and "paragraph" in q):
            tasks.append("paragraph_body")
            relations.append("contains")
        if re.search(r"\b(?:from|starting at|starts? at)\b", q):
            tasks.append("path_from_paragraph")
            relations.append("starts_at")
        if not tasks:
            tasks.append("complete_program_flow")
    elif intent == "variable_dataflow":
        lifecycle = bool(re.search(
            r"\b(?:lifecycle|life cycle|trace|lineage|data movement|data flow|"
            r"intermediate|originated|transferred|feeds?|toward|through)\b", q
        ))
        if re.search(r"\bcompare|comparison|versus|vs\.?\b", q):
            tasks.append("variable_comparison")
            relations.append("compares")
        if lifecycle:
            tasks.append("variable_lineage")
        if re.search(
            r"\b(?:complete lifecycle|affects? (?:the )?control flow|controls? execution|"
            r"resulting control flow|subsequent control flow)\b",
            q,
        ) or re.search(r"\bwhat does (?:each |the )?value mean\b", q):
            tasks.append("control_outcome")
            relations.append("condition_causes")
        if re.search(r"\b(?:construct(?:ed|ion)?|built|build|composed|comes? from|data movement)\b", q):
            tasks.append("variable_composition")
        if re.search(r"\b(?:cics link|link operation|exact statement|commarea|length option|controls? the .*link)\b", q):
            tasks.append("call_option_usage")
        if re.search(r"\b(?:toward|destination|terminal|displayed|output field|intermediate variable)\b", q):
            tasks.append("lineage_terminal")
        if lifecycle or re.search(r"\b(?:what is|defined|definition|exist|origin)\b", q):
            tasks.append("variable_definition")
        if lifecycle or re.search(_VARIABLE_CONSUMPTION_CUE_PATTERN, q):
            tasks.append("variable_reads")
        if lifecycle or re.search(_VARIABLE_PRODUCTION_CUE_PATTERN, q):
            tasks.append("variable_writes")
        if "literal" in q:
            tasks.append("literal_assignments")
        if not tasks:
            tasks.extend(["variable_definition", "variable_reads", "variable_writes"])
    elif intent == "variable_inventory":
        tasks.append("variable_inventory")
    elif intent == "static_values":
        tasks.append("literal_assignments")
    elif intent == "external_programs":
        before_cue = bool(re.search(_CALL_BEFORE_CUE_PATTERN, q))
        after_cue = bool(re.search(_CALL_AFTER_CUE_PATTERN, q))
        if before_cue or after_cue:
            tasks.append("call_context")
            if before_cue:
                relations.append("before")
            if after_cue:
                relations.append("after")
        else:
            tasks.append("external_calls")
    elif intent == "copybooks":
        tasks.append("copybook_inventory")
        if any(term in q for term in ("usage", "used", "actually reference", "contribute variables")):
            tasks.append("copybook_usage")
        if "for each" in q or "each one" in q or "example" in q:
            tasks.append("direct_usage_examples")
            relations.append("example_per_item")
    elif intent in {"datasets_tables", "db2_sql"}:
        if "db2" in q or "table" in q:
            tasks.append("db2_tables")
        if "sql include" in q or "sqlinclude" in q:
            tasks.append("sql_includes")
        if "jcl" in q or "dataset" in q:
            tasks.append("jcl_datasets")
        if "separat" in q:
            relations.append("separate_categories")
    elif intent == "dead_code":
        if "comment" in q:
            tasks.append("commented_code")
        if "unreachable" in q:
            tasks.append("unreachable_code")
        if "unused" in q and ("copybook" in q or "copy book" in q or "copy" in q):
            tasks.extend(["unused_copybooks", "review_copybooks"])
        if "review" in q:
            tasks.append("review_copybooks")
        if not tasks:
            tasks.extend(["commented_code", "unreachable_code"])
        if "separat" in q:
            relations.append("separate_categories")
    elif intent == "business_rules":
        tasks.append("condition_outcome" if _is_condition_effect_question(q) else "business_rules")
    elif intent == "source_metrics":
        tasks.append("source_metrics")

    if "for each" in q and "example_per_item" not in relations:
        relations.append("example_per_item")
    return _unique(tasks), _unique(relations)


def _split_negative_constraints(q: str) -> tuple[str, str]:
    spans: list[tuple[int, int]] = []
    for match in re.finditer(
        r"(?:do not include|don.t include|exclude|excluding|without|but not)\s+([^.;?]+)",
        q,
    ):
        spans.append(match.span())
    negative = " ".join(q[start:end] for start, end in spans)
    positive_chars = list(q)
    for start, end in spans:
        positive_chars[start:end] = " " * (end - start)
    return negative, "".join(positive_chars)


def _is_condition_effect_question(q: str) -> bool:
    return bool(
        re.search(r"\bwhat (?:happens|occurs) (?:when|if)\b", q)
        or re.search(r"\bresult (?:when|if|of)\b", q)
        or re.search(r"\bwhat does .+ do (?:when|if)\b", q)
        or "neither" in q
        or "otherwise" in q
    )


def _is_source_metrics_question(q: str) -> bool:
    return bool(
        re.search(
            r"\b(?:how many|number of|count(?: of)?)\b.{0,40}\b(?:lines?|loc|lines? of code)\b",
            q,
        )
        or re.search(r"\b(?:loc|line count)\b", q)
    )


def _condition_terms(question: str) -> tuple[str, ...]:
    upper = question.upper()
    terms: list[str] = []
    terms.extend(re.findall(r"'([^']+)'", upper))
    neither = re.search(r"\bNEITHER\s+([A-Z0-9-]+)\s+NOR\s+([A-Z0-9-]+)", upper)
    if neither:
        terms.extend(neither.groups())
    terms.extend(re.findall(r"(?:=|EQUALS?|EQUAL TO)\s*'?([A-Z0-9-]+)", upper))
    return _unique(term.strip() for term in terms if term.strip())


def _is_explicit_followup(q: str) -> bool:
    return bool(
        re.search(
            r"^(?:and\s+)?(?:where|what|how|why|when)\b.*"
            r"\b(?:it|them|that call|this call|that variable|this variable|those paragraphs)\b",
            q,
        )
        or re.match(r"^(?:and\s+)?(?:where|what|how|why|when)\s+else\b", q)
        or re.match(r"^(?:and\s+)?(?:what|how)\s+about\b", q)
        or re.match(
            r"^(?:and\s+)?(?:there (?:is|are) more|more\b|continue\b|show (?:me )?the rest\b|"
            r"list the rest\b|you missed\b)",
            q,
        )
    )


def _requests_exhaustive_results(q: str) -> bool:
    return bool(
        re.search(
            r"\b(?:all|every|every single|complete list|entire list|the rest|remaining)\b",
            q,
        )
        or re.match(r"^(?:and\s+)?(?:there (?:is|are) more|more\b|continue\b|you missed\b)", q)
    )


def _unique(values: Any) -> tuple[str, ...]:
    return tuple(dict.fromkeys(str(value) for value in values if str(value)))
