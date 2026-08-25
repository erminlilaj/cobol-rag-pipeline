from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field, replace
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
class ResponseContract:
    """Machine-checkable presentation requirements from the user message."""

    format: str = "default"
    max_sentences: int | None = None
    max_lines: int | None = None
    exact_lines: int | None = None
    max_words: int | None = None
    exact_item_count: int | None = None
    only_requested_content: bool = False
    yes_no_first: bool = False


@dataclass(frozen=True)
class QueryPlan:
    route: str = "technical"
    category: str = "single_source"
    domain: str = "general"
    tasks: tuple[str, ...] = ()
    relations: tuple[str, ...] = ()
    response_language: str = "en"
    response_language_source: str = "default"
    response_contract: ResponseContract = field(default_factory=ResponseContract)
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
    # Zero-based position for continuation requests such as "show me the rest".
    # Scope (all/default) describes completeness; offset describes which part of
    # that result set the current turn should render.
    result_offset: int = 0
    explicit_followup: bool = False
    requires_comparison: bool = False
    requires_clarification: bool = False
    confidence: float = 0.0
    # Confidence owned by typed parsing before any semantic planner update. This
    # is the authority boundary; semantic self-confidence must not promote its
    # own interpretation into an immutable deterministic fact.
    authority_confidence: float = 0.0
    planner_source: str = "deterministic"
    subtasks: tuple[EvidenceSubtask, ...] = ()
    policy_rejections: tuple[str, ...] = ()

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

_NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}


def _contract_number(value: str) -> int | None:
    value = value.strip().lower()
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def parse_response_contract(question: str) -> ResponseContract:
    """Extract output-shape instructions independently from COBOL semantics."""
    q = question.lower()
    number = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"

    output_format = "default"
    if re.search(r"\bjson\s+array\b", q):
        output_format = "json_array"
    elif re.search(r"\bjson\b", q):
        output_format = "json"
    elif re.search(r"\b(?:only\s+(?:the\s+)?(?:number|count)|(?:number|count)\s+only)\b", q):
        output_format = "count"
    elif re.search(r"\b(?:bullet(?:\s+points?)?|bullets?)\b", q):
        output_format = "bullets"
    elif re.search(rf"\b(?:in\s+)?{number}\s+sentences?\b|\bone[- ]sentence\b", q):
        output_format = "sentence"
    elif re.search(
        r"\b(?:how many|number of|count of)\b.{0,45}"
        r"\b(?:variables?|fields?|data items?|calls?|programs?|copybooks?|"
        r"operations?|assignments?|forced values?)\b",
        q,
    ) and not re.search(
        r"\b(?:summary|summarize|summarise|overview|describe|explain|list|show)\b"
        r".{0,100}\b(?:and|also|plus|then)\b.{0,100}"
        r"\b(?:how many|number of|count of)\b",
        q,
    ):
        # A count-only contract applies to the complete answer.  In a compound
        # request such as "summarize the program and tell me how many variables"
        # the count is one evidence claim, not permission to discard the summary.
        output_format = "count"

    max_sentences: int | None = None
    sentence_match = re.search(
        rf"\b(?:in\s+|at\s+most\s+|no\s+more\s+than\s+|maximum\s+)?({number})\s+sentences?\b"
        r"|\bone[- ]sentence\b",
        q,
    )
    if sentence_match:
        max_sentences = _contract_number(sentence_match.group(1) or "one")

    # A request such as "explain it in two lines" describes the shape of the
    # answer.  It must not be confused with a request for COBOL source lines.
    max_lines: int | None = None
    exact_lines: int | None = None
    line_match = re.search(
        rf"\b(?:in|within|using|as|at\s+most|no\s+more\s+than|maximum|max)\s+"
        rf"(?:exactly\s+)?({number})\s+lines?\b"
        rf"|\b({number})\s+lines?\s+(?:only|maximum|max)\b",
        q,
    )
    if line_match:
        max_lines = _contract_number(line_match.group(1) or line_match.group(2))
        # Natural phrasing such as "in three lines" requests an exact shape.
        # Explicit upper-bound wording remains a maximum instead.
        if not re.search(
            rf"\b(?:within|at\s+most|no\s+more\s+than|maximum|max)\s+(?:exactly\s+)?{number}\s+lines?\b"
            rf"|\b{number}\s+lines?\s+(?:maximum|max)\b",
            q,
        ):
            exact_lines = max_lines

    max_words: int | None = None
    word_match = re.search(
        r"(?:\b(?:in|within|under|maximum|max|at\s+most|no\s+more\s+than)\s+|<=\s*|≤\s*)"
        r"(\d+)\s+words?\b|\b(\d+)\s+words?\s+or\s+(?:fewer|less)\b",
        q,
    )
    if word_match:
        max_words = int(word_match.group(1) or word_match.group(2))

    exact_item_count: int | None = None
    item_match = re.search(
        rf"\b(?:exactly\s+(?:the\s+)?(?:first\s+)?|(?:the\s+)?first\s+)({number})\s+"
        r"(?:(?:literal|forced|outgoing|external)\s+)?"
        r"(?:bullet(?:\s+points?)?|bullets?|items?|results?|values?|assignments?|calls?|programs?|copybooks?)\b",
        q,
    )
    if item_match:
        exact_item_count = _contract_number(item_match.group(1))
    else:
        named_count = re.search(
            rf"\b(?:name|list|show|give(?:\s+me)?)\s+(?:exactly\s+)?({number})\s+"
            r"(?:variables?|fields?|data items?|calls?|programs?|copybooks?|"
            r"operations?|assignments?|forced values?)\b",
            q,
        )
        if named_count:
            exact_item_count = _contract_number(named_count.group(1))
    if exact_item_count is None and output_format == "bullets":
        bullet_match = re.search(rf"\b({number})\s+bullet(?:\s+points?)?\b", q)
        if bullet_match:
            exact_item_count = _contract_number(bullet_match.group(1))

    only_requested_content = bool(
        re.search(
            r"\b(?:only|nothing else|no extra (?:text|content|explanation)|without (?:extra|additional) (?:text|content|explanation))\b",
            q,
        )
    )
    yes_no_first = bool(
        re.search(r"\b(?:yes\s*(?:/|or)\s*no|start with yes or no|answer yes or no)\b", q)
    )
    return ResponseContract(
        format=output_format,
        max_sentences=max_sentences,
        max_lines=max_lines,
        exact_lines=exact_lines,
        max_words=max_words,
        exact_item_count=exact_item_count,
        only_requested_content=only_requested_content,
        yes_no_first=yes_no_first,
    )


def _is_named_program_overview_request(question: str, program: str) -> bool:
    """True only when the whole request is a short description of one program.

    Anchoring the semantic request prevents phrases such as "explain PDCBVC's
    control flow" from being collapsed into a generic program summary.
    """
    match = re.match(
        rf"^\s*(?:what\s+is|explain|describe|summarize|summarise|tell\s+me\s+about)\s+"
        rf"(?:the\s+)?(?:program\s+)?{re.escape(program)}\b",
        question,
        flags=re.IGNORECASE,
    )
    if not match:
        return False
    tail = question[match.end():].strip().strip(".?!").strip()
    if not tail:
        return True
    number = r"(?:\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)"
    constraint = (
        rf"(?:briefly|concisely|only|please|"
        rf"in\s+(?:exactly\s+)?{number}\s+(?:sentences?|lines?)(?:\s+only)?(?:\s+please)?|"
        rf"(?:using|within|under|at\s+most|no\s+more\s+than|maximum|max)\s+"
        rf"{number}\s+words?(?:\s+or\s+(?:fewer|less))?)"
    )
    return all(
        re.fullmatch(constraint, part.strip(), flags=re.IGNORECASE) is not None
        for part in re.split(r"\s+and\s+", tail, flags=re.IGNORECASE)
    )

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
    r"\bgets? control back\b|"
    r"\bresults?\s+of\s+(?:the\s+)?(?:[a-z0-9-]+\s+)?call\b|"
    r"\bresponse\s+of\s+(?:the\s+)?(?:[a-z0-9-]+\s+)?call\b|"
    r"\bwhat comes back\b|\bwhat happens next\b|"
    r"\bwhat (?:happens|occurs|does (?:it|the program|the call) do)\b.{0,60}"
    r"\b(?:return|respond|complete)s?\b"
)

_CALL_BEFORE_CUE_PATTERN = (
    r"\bbefore\b|\bprior to\b|\bleading up to\b|\bahead of\b|"
    r"\bin preparation for\b|\bbeforehand\b|\bsetup before\b|"
    r"\bgets? ready (?:for|to)\b"
)


def is_pagination_question(question: str) -> bool:
    """Recognize the concept of moving through result pages, not one phrase."""
    q = question.lower()
    return bool(
        re.search(r"\b(?:pagination|page navigation|paging)\b", q)
        or re.search(r"\b(?:page|move|go)\s+(?:through|between)\s+(?:the\s+)?results?\b", q)
        or re.search(r"\b(?:move|go|step|navigate)s?\s+(?:through|between)\s+(?:the\s+)?result\s+pages?\b", q)
        or re.search(r"\b(?:next|previous|prior)\s+(?:result\s+)?page\b", q)
    )


def is_complete_program_flow_question(question: str) -> bool:
    """Recognize an end-to-end program walk rather than a named start node."""
    q = question.lower()
    return bool(
        (
            re.search(r"\bwalk\s+(?:me\s+)?through\b", q)
            and re.search(r"\b(?:start|beginning|entry)\b", q)
            and re.search(r"\b(?:finish|end|termination|exit)\b", q)
        )
        or re.search(
            r"\bfrom\s+(?:its\s+|the\s+)?(?:start|beginning|entry(?: point)?)\s+"
            r"(?:through|to|until)\s+(?:its\s+|the\s+)?"
            r"(?:finish|end|termination|exit)\b",
            q,
        )
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
    "call_type": (r"\bcall types?\b", r"\btypes? of calls?\b"),
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
    "control_usage": (
        r"\bcontrols? (?:the )?(?:flow|execution)\b",
        r"\bcontrol usage\b",
    ),
    "line_count": (
        r"\bhow many (?:physical |source |code )?lines?\b",
        r"\bnumber of (?:physical |source |code )?lines?\b",
        r"\b(?:loc|line count)\b",
    ),
    "item_count": (
        r"\bhow many\b.{0,45}\b(?:variables?|fields?|data items?|calls?|programs?|"
        r"copybooks?|operations?|assignments?|forced values?)\b",
        r"\b(?:number|count) of\b.{0,45}\b(?:variables?|fields?|data items?|calls?|"
        r"programs?|copybooks?|operations?|assignments?|forced values?)\b",
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

    # Naming a language is itself the request, whatever verb surrounds it. The
    # patterns above enumerate verbs, so "facciamo" and "parliamo" were invisible
    # and a session could not be steered out of a language by ordinary phrasing.
    # A near match also survives the misspelling people actually type. Only
    # standalone word tokens are considered, so a COBOL identifier such as
    # ENGLISH-FLAG is never read as a language preference.
    return _named_response_language(normalized)


# 0.85 measured against both requests and COBOL decoys: 0.90 loses "englese",
# 0.80 admits more than it needs to.
_LANGUAGE_NAME_CUTOFF = 0.85
_STANDALONE_WORD = re.compile(r"(?<![A-Za-z0-9_-])[A-Za-z]{5,12}(?![A-Za-z0-9_-])")
_RESPONSE_LANGUAGE_NAMES = {
    "english": "en", "inglese": "en", "anglais": "en",
    "italian": "it", "italiano": "it", "italien": "it",
}


def _named_response_language(normalized: str) -> str | None:
    for token in _STANDALONE_WORD.findall(normalized):
        exact = _RESPONSE_LANGUAGE_NAMES.get(token)
        if exact:
            return exact
        near = difflib.get_close_matches(
            token, _RESPONSE_LANGUAGE_NAMES, n=1, cutoff=_LANGUAGE_NAME_CUTOFF,
        )
        if near:
            return _RESPONSE_LANGUAGE_NAMES[near[0]]
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
    "source_line_lookup": ("source_lines",),
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


# Below the 0.9 authority gate in merge_semantic_plan, so a topical intent
# informs the plan without vetoing the planner's reading of the question.
_TOPICAL_INTENT_CONFIDENCE = 0.7


def build_query_plan(
    question: str,
    scope: QueryScope,
    *,
    intent: str | None = None,
    intent_basis: str = "explicit",
    state: SessionState | None = None,
) -> QueryPlan:
    q = question.lower()
    upper = question.upper()
    resolved_intent = intent or scope.intent or "general"
    detected_intent = resolved_intent
    explicit_followup = _is_explicit_followup(q)
    if explicit_followup and resolved_intent == "general" and state and state.current_intent:
        resolved_intent = state.current_intent

    named_program_request = bool(
        scope.program
        and not scope.entities
        and (
            _is_named_program_overview_request(question, scope.program)
            or re.search(
                rf"\b(?:(?:do\s+)?you\s+have|have\s+you\s+got|is\s+there)\b"
                rf".{{0,45}}\b(?:file|program|source)\b.{{0,25}}"
                rf"\b(?:called|named)?\s*{re.escape(scope.program)}\b",
                question,
                flags=re.IGNORECASE,
            )
        )
    )

    if named_program_request:
        # Exact program resolution is authoritative.  Broad words such as
        # "file" must not redirect this request to JCL dataset evidence.
        resolved_intent = "program_summary"
    elif any(term in q for term in ("return value", "value returned")):
        resolved_intent = "variable_dataflow"
    elif (
        resolved_intent == "general"
        and scope.program
        and not scope.entities
        and re.search(
            rf"\b(?:describe|summarize|summarise)\s+(?:the\s+)?(?:program\s+)?{re.escape(scope.program)}\b",
            question,
            flags=re.IGNORECASE,
        )
    ):
        resolved_intent = "program_summary"
    elif resolved_intent == "general" and re.search(r"\bliteral assignments?\b", q):
        resolved_intent = "static_values"
    elif _is_source_metrics_question(q):
        resolved_intent = "source_metrics"
    elif (
        resolved_intent in {"general", "control_flow", "ui_navigation"}
        and is_pagination_question(question)
    ):
        resolved_intent = "control_flow"
    elif _is_condition_effect_question(q) and resolved_intent in {
        "general", "variable_dataflow", "business_rules", "control_flow"
    }:
        resolved_intent = "business_rules"
    elif (
        re.search(r"\b(?:call|calls|called|calling)\b", q)
        and (
            any(entity.entity_type == "call" for entity in scope.entities)
            or any(term in q for term in (
                "outgoing call", "external call", "external program", "call type",
                "called by", "programs called",
            ))
            or bool(
                scope.program
                and re.search(r"\b(?:which|what|list|show)\b.{0,45}\bprograms?\b", q)
            )
        )
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
    # Presentation language is not a COBOL operation. In particular, phrases such
    # as "return only target and call type" used to filter calls to EXEC CICS
    # RETURN and produce an empty but apparently valid answer.
    presentation_return = bool(
        re.match(r"^\s*return\s+(?:every|all|exactly|only|the|a|an|\d+)\b", q)
        and not re.search(r"\b(?:exec\s+cics|cics)\s+return\b", q)
    )
    if presentation_return or any(
        term in q for term in (
            "return value", "value returned", "return only", "return a json", "return json",
        )
    ):
        operations = tuple(operation for operation in operations if operation != "RETURN")

    semantic_operations: list[str] = []
    if re.search(r"\b(?:compare|comparison|difference|differences|versus|vs\.?)\b", q):
        semantic_operations.append("compare")
    if re.search(r"\b(?:does|do|is|are)\b.{0,80}\b(?:exist|exists|present|available)\b", q):
        semantic_operations.append("exists")
    if re.search(r"\b(?:(?:do\s+)?you\s+have|have\s+you\s+got|is\s+there)\b", q):
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
            r"(?:names?|targets?|call types?|paragraphs?|source|lines?|divisions?|sections?|parameters?|commarea)",
            q,
        )
        or re.search(
            r"only\s+include\s+(?:their\s+|the\s+)?"
            r"(?:names?|targets?|call types?|paragraphs?|source|lines?|divisions?|sections?|parameters?|commarea)",
            q,
        )
        or re.search(
            r"with\s+only\s+(?:their\s+|the\s+)?"
            r"(?:names?|targets?|call types?|paragraphs?|source|lines?|divisions?|sections?|parameters?|commarea)",
            q,
        )
    )

    condition_terms = _condition_terms(question) if _is_condition_effect_question(q) else ()
    negative_condition = bool(
        _is_condition_effect_question(q)
        and re.search(r"\b(?:neither|otherwise|not equal|not equals|is not|isn.t|not =)\b", q)
    )
    response_contract = parse_response_contract(question)
    if (
        resolved_intent == "variable_inventory"
        and response_contract.exact_item_count is None
        and re.search(r"\b(?:sample|example|examples|a few|some)\b", q)
    ):
        # "A sample" is intentionally bounded.  Ten is a presentation default,
        # not a stored answer: the formatter still selects the values from the
        # current program's generated catalogue.
        response_contract = replace(response_contract, exact_item_count=10)
    result_scope = "all" if _requests_exhaustive_results(q) else "default"
    if (
        resolved_intent == "variable_inventory"
        and not re.search(r"\b(?:sample|few|some|example|examples)\b", q)
        and response_contract.format != "count"
        and response_contract.exact_item_count is None
        and re.search(r"\b(?:what|which)\b.{0,55}\b(?:variables?|fields?|data items?)\b", q)
    ):
        result_scope = "all"
    result_offset = 0
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
        if re.search(r"\b(?:the rest|remaining|continue)\b", q):
            result_scope = "all"
            previous_contract = previous.get("response_contract", {})
            previous_offset = int(previous.get("result_offset", 0) or 0)
            previous_count = (
                previous_contract.get("exact_item_count")
                if isinstance(previous_contract, dict)
                else None
            )
            result_offset = previous_offset + int(previous_count or 25)

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
    if intent_basis == "topical" and resolved_intent == detected_intent:
        # The intent rested on a common English noun rather than on anything this
        # layer can verify, so it stays below the authority threshold and the
        # semantic planner is free to overrule it. If one of the explicit rules
        # above re-derived the intent, resolved_intent no longer matches what the
        # keyword pass produced and full confidence is kept.
        confidence = min(confidence, _TOPICAL_INTENT_CONFIDENCE)
    if not operations:
        confidence = min(confidence, 0.7)
    domain = _INTENT_DOMAIN.get(resolved_intent, "general")
    tasks, relations = _fallback_tasks_and_relations(q, resolved_intent)
    if scope.program and not scope.entities:
        requested_program_tasks: list[str] = []
        if re.search(r"\b(?:summary|summarize|summarise|overview)\b", q):
            requested_program_tasks.append("program_summary")
        if re.search(
            r"\b(?:how many|number of|count of)\b.{0,45}"
            r"\b(?:variables?|fields?|data items?)\b",
            q,
        ):
            requested_program_tasks.append("variable_inventory")
        tasks = _unique((*tasks, *requested_program_tasks))
        if {"program_summary", "variable_inventory"} <= set(tasks):
            # Intent is the primary purpose; tasks are the complete claim set.
            # Preserve both so the semantic planner cannot answer just the last
            # clause of a compound program-wide request.
            resolved_intent = "program_summary"
            domain = "multi_source"
            category = "multi_source_synthesis"
            source_domains = _unique((
                *_INTENT_SOURCE_DOMAINS["program_summary"],
                *_INTENT_SOURCE_DOMAINS["variable_inventory"],
            ))
    if any(entity.entity_type == "call" for entity in scope.entities):
        if re.search(r"\bbefore\b", q):
            relations = _unique((*relations, "before"))
        if re.search(r"\bafter\b", q):
            relations = _unique((*relations, "after"))
    response_language, response_language_source = resolve_response_language(question, state)
    if response_contract.max_lines is not None:
        output_fields = tuple(field for field in output_fields if field != "source_line")

    base_plan = QueryPlan(
        route="technical",
        category=category,
        domain=domain,
        tasks=tasks,
        relations=relations,
        response_language=response_language,
        response_language_source=response_language_source,
        response_contract=response_contract,
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
        result_offset=result_offset,
        explicit_followup=explicit_followup,
        requires_comparison=requires_comparison,
        requires_clarification=unresolved_previous_variable,
        confidence=confidence,
        authority_confidence=confidence,
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
    policy_rejections: list[str] = list(plan.policy_rejections)
    strict_authority = bool(
        plan.route == "technical"
        and plan.intent != "general"
        and plan.confidence >= 0.9
        and not plan.requires_clarification
    )
    route = str(update.get("route", plan.route)).strip().lower()
    if route not in allowed_routes:
        route = plan.route
    if strict_authority and route != plan.route:
        policy_rejections.append(f"route_not_authorized:{route}")
        route = plan.route
    category = str(update.get("category", plan.category)).strip().lower()
    if category not in allowed_categories:
        category = plan.category
    if strict_authority and category != plan.category:
        policy_rejections.append(f"category_not_authorized:{category}")
        category = plan.category
    proposed_intent = str(update.get("intent", plan.intent)).strip().lower().replace("-", "_").replace(" ", "_")
    if proposed_intent == "datasets":
        proposed_intent = "datasets_tables"
    allowed_intents = set(_INTENT_SOURCE_DOMAINS) | {"ui_navigation", "source_metrics", "general"}
    intent = proposed_intent if proposed_intent in allowed_intents else plan.intent
    # A high-confidence deterministic intent comes from explicit vocabulary and
    # verified scope (for example "outgoing calls" or "program summary"). The
    # LLM may decompose it into more claims, but may not downgrade or replace the
    # primary capability with a merely nearby one.
    semantic_intent_conflict = bool(
        plan.intent != "general"
        and plan.confidence >= 0.9
        and intent != plan.intent
        and not plan.requires_comparison
    )
    if semantic_intent_conflict:
        policy_rejections.append(f"intent_not_authorized:{intent}")
        intent = plan.intent
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
    if strict_authority:
        for operation in proposed_operations:
            if operation not in plan.operations:
                policy_rejections.append(f"operation_not_authorized:{operation}")
        operations = plan.operations
    else:
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
    if strict_authority and domain != plan.domain:
        policy_rejections.append(f"domain_not_authorized:{domain}")
        domain = plan.domain
    semantic_tasks = _unique(value for value in update.get("tasks", []) if str(value) in ALLOWED_PLAN_TASKS)
    semantic_relations = _unique(value for value in update.get("relations", []) if str(value) in ALLOWED_PLAN_RELATIONS)
    # When the semantic model proposes an incompatible primary intent, discard its
    # task decomposition as well as the intent label. Keeping (for example)
    # variable-read tasks under a verified program-summary intent creates a plan
    # that is internally inconsistent even though the visible intent looks right.
    if strict_authority:
        for task in semantic_tasks:
            if task not in plan.tasks:
                policy_rejections.append(f"task_not_authorized:{task}")
        for relation in semantic_relations:
            if relation not in plan.relations:
                policy_rejections.append(f"relation_not_authorized:{relation}")
        tasks = plan.tasks
        relations = plan.relations
    else:
        tasks = plan.tasks if semantic_intent_conflict else _unique((*plan.tasks, *semantic_tasks))
        relations = plan.relations if semantic_intent_conflict else _unique((*plan.relations, *semantic_relations))
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
    semantic_output_fields = _unique(
        str(value) for value in update.get("output_fields", [])
        if str(value) in _OUTPUT_FIELD_PATTERNS
    )
    if strict_authority:
        for output_field in semantic_output_fields:
            if output_field not in plan.output_fields:
                policy_rejections.append(f"output_field_not_authorized:{output_field}")
        output_fields = plan.output_fields
        source_domains = plan.source_domains
    else:
        output_fields = _unique((*plan.output_fields, *semantic_output_fields))
    requires_comparison = bool(
        plan.requires_comparison or update.get("requires_comparison")
        or "compare" in operations or len(plan.programs) > 1
    )
    if strict_authority and requires_comparison != plan.requires_comparison:
        policy_rejections.append("comparison_not_authorized")
        requires_comparison = plan.requires_comparison
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
        requires_clarification=(
            plan.requires_clarification
            if strict_authority
            else bool(plan.requires_clarification or update.get("requires_clarification"))
        ),
        confidence=(plan.confidence if strict_authority else confidence),
        planner_source="hybrid_llm",
        policy_rejections=_unique(policy_rejections),
    )
    semantic_subtasks = (
        () if semantic_intent_conflict
        else _parse_semantic_subtasks(merged, update.get("subtasks"))
    )
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
    merged = _remove_subsumed_subtasks(subtasks)
    return tuple(
        replace(item, claim_id=f"claim_{index}")
        for index, item in enumerate(merged, start=1)
    )


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
        # Presentation constraints belong to the user's request and remain
        # immutable across planning. Evidence-stage validation deliberately
        # ignores them via ``validate_evidence_answer`` below; the final renderer
        # is the only stage that enforces them.
        response_contract=plan.response_contract,
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
        # A subtask must implement at least one task in the merged query plan.
        # This rejects internally inconsistent LLM decompositions such as a DB2
        # evidence claim for a variable declaration merely because the user said
        # "include its declaration". Genuine compound plans keep all requested
        # tasks in plan.tasks, so their multiple capabilities remain eligible.
        if plan.tasks and supported.isdisjoint(plan.tasks):
            continue
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
                # Model-proposed fields may add useful detail, but they cannot
                # erase fields parsed from the user's words (for example the
                # item_count half of a summary-plus-count request).
                output_fields=_unique((*plan.output_fields, *fields)),
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
    # Access and lineage for the same exact variable are one evidence join, not
    # two independent claims.  Keeping them separate makes the answer executor
    # run the same variable artifact twice and often produces duplicated blocks.
    merged_variables: list[EvidenceSubtask] = []
    variable_index: dict[tuple[str, ...], int] = {}
    for item in unique:
        if item.capability not in {"variable_access", "variable_lineage"} or not item.entity_values:
            merged_variables.append(item)
            continue
        key = tuple(sorted(item.entity_values))
        existing_index = variable_index.get(key)
        if existing_index is None:
            variable_index[key] = len(merged_variables)
            merged_variables.append(item)
            continue
        existing = merged_variables[existing_index]
        merged_variables[existing_index] = replace(
            existing,
            capability="variable_access",
            tasks=_unique((*existing.tasks, *item.tasks)),
            relations=_unique((*existing.relations, *item.relations)),
            source_domains=_unique((*existing.source_domains, *item.source_domains)),
            output_fields=_unique((*existing.output_fields, *item.output_fields)),
            required=existing.required or item.required,
        )
    return _merge_whole_program_capability_claims(merged_variables)


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
        "source_line_lookup": ("source_lines", "program_structure"),
    }
    intent, domain = mapping.get(capability, ("general", "general"))
    return intent, domain


def _subtask_description(
    capability: str,
    entities: tuple[str, ...],
    program: str | None,
) -> str:
    target = ", ".join(entities) if entities else (program or "the selected program")
    descriptions = {
        "artifact_inventory": "Inspect analyzed artifacts",
        "program_summary": "Summarize the program",
        "source_metrics": "Measure source size",
        "variable_inventory": "Inspect the variable catalogue",
        "paragraph_evidence": "Inspect paragraph structure",
        "variable_access": "Trace variable access",
        "literal_assignment": "Inspect literal assignments",
        "variable_lineage": "Trace variable lineage",
        "condition_outcome": "Trace condition outcomes",
        "control_flow": "Trace control flow",
        "call_evidence": "Inspect outgoing calls",
        "call_context": "Inspect call context",
        "cics_evidence": "Inspect CICS operations",
        "copybook_evidence": "Inspect copybooks",
        "db2_evidence": "Inspect DB2 and SQL evidence",
        "jcl_evidence": "Inspect JCL dataset evidence",
        "quality_evidence": "Inspect quality evidence",
        "pagination_evidence": "Trace pagination",
        "screen_lineage": "Trace screen-field lineage",
    }
    action = descriptions.get(capability, f"Inspect {capability.replace('_', ' ')}")
    return f"{action} for {target}"


# An exhaustive answer has to say how much it returned. Renderers state that
# either as an item count or as a coverage ratio; both are a completeness claim
# the reader can check, so either satisfies the contract. Only the item-count
# form can be cross-checked against the listed lines, so the count comparison
# stays attached to it.
_EXHAUSTIVE_ITEM_COUNT = re.compile(r":\s*(\d+)\s+matching item\(s\)\.")
_EXHAUSTIVE_COVERAGE = re.compile(
    r"\b\d+\s*/\s*\d+\s+(?:\w+\s+)*?(?:site|item|rule|entry)\(s\)\s+returned\b"
    r"|\b\d+\s+matching unique direct-evidence rule\(s\)",
    re.IGNORECASE,
)


def _answer_word_count(answer: str) -> int:
    without_code = re.sub(r"`[^`]*`", " ", answer)
    return len(re.findall(r"\b[\w]+(?:[-'][\w]+)*\b", without_code, flags=re.UNICODE))


def _answer_sentence_count(answer: str) -> int:
    without_code = re.sub(r"`[^`]*`", "", answer).strip()
    if not without_code:
        return 0
    return len([
        part for part in re.split(r"(?<=[.!?])(?:\s+|$)", without_code)
        if re.search(r"\w", part)
    ])


def _answer_item_count(answer: str, output_format: str) -> int | None:
    if output_format in {"json", "json_array"}:
        try:
            payload = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            return None
        if isinstance(payload, list):
            return len(payload)
        return len(payload) if isinstance(payload, dict) else None
    bullets = [
        line for line in answer.splitlines()
        if re.match(r"^\s*(?:[-*]|\d+[.)])\s+", line)
    ]
    return len(bullets) if bullets else None


def validate_plan_answer(plan: QueryPlan, answer: str) -> PlanContractValidation:
    if not answer.strip():
        return PlanContractValidation(False, ("empty_answer",))
    lowered = answer.lower()
    reasons: list[str] = []
    response = plan.response_contract

    if response.format in {"json", "json_array"}:
        try:
            parsed_json = json.loads(answer)
        except (json.JSONDecodeError, TypeError):
            reasons.append("invalid_requested_json")
        else:
            if response.format == "json_array" and not isinstance(parsed_json, list):
                reasons.append("requested_json_array_not_returned")
    if response.format == "count" and not re.fullmatch(r"\s*\d+\s*", answer):
        reasons.append("requested_count_only_not_returned")
    if response.max_sentences is not None:
        if _answer_sentence_count(answer) > response.max_sentences:
            reasons.append(
                f"max_sentences_exceeded:{_answer_sentence_count(answer)}/{response.max_sentences}"
            )
        if response.max_sentences == 1 and len([line for line in answer.splitlines() if line.strip()]) > 1:
            reasons.append("one_sentence_requires_single_line")
    if response.max_lines is not None:
        actual_lines = len([line for line in answer.splitlines() if line.strip()])
        if actual_lines > response.max_lines:
            reasons.append(f"max_lines_exceeded:{actual_lines}/{response.max_lines}")
    if response.exact_lines is not None:
        actual_lines = len([line for line in answer.splitlines() if line.strip()])
        if actual_lines != response.exact_lines:
            reasons.append(f"exact_lines_mismatch:{actual_lines}/{response.exact_lines}")
    if response.max_words is not None and _answer_word_count(answer) > response.max_words:
        reasons.append(f"max_words_exceeded:{_answer_word_count(answer)}/{response.max_words}")
    if response.exact_item_count is not None:
        actual_items = _answer_item_count(answer, response.format)
        if actual_items != response.exact_item_count:
            reasons.append(
                f"exact_item_count_mismatch:{actual_items if actual_items is not None else 0}/"
                f"{response.exact_item_count}"
            )
    if response.yes_no_first and not re.match(r"\s*(?:yes|no)\b", answer, flags=re.IGNORECASE):
        reasons.append("missing_yes_no_prefix")

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

    # Each capability renders in its own vocabulary, so a section is satisfied by
    # any wording that carries the substance. A control outcome stated as a rule
    # action ("Action: JUMP -> ABEND00") is the same finding as one under a
    # "Resulting control-flow actions:" heading, and rejecting it discarded a
    # cited, line-accurate answer the retrieval layer had already found.
    required_sections = {
        "variable_reads": ("tested/read at:", "read sites:"),
        "variable_writes": ("modified at:", "write sites:"),
        "variable_lineage": ("lineage:", "downstream lineage:"),
        "control_outcome": (
            "resulting control-flow actions:", "control-flow use:", "action:",
        ),
        "variable_composition": ("construction/context evidence:",),
        "call_option_usage": ("cics/call option usage:",),
        "lineage_terminal": ("lineage completion:",),
    }
    for task, markers in required_sections.items():
        if task in plan.tasks and not any(marker in lowered for marker in markers):
            reasons.append(f"missing_requested_section:{task}")
    if "control_usage" in plan.output_fields:
        control_usage_markers = (
            "control-flow use:",
            "controls flow",
        )
        if not any(marker in lowered for marker in control_usage_markers):
            reasons.append("missing_requested_field:control_usage")
    if "line_count" in plan.output_fields and not re.search(
        r"\b\d+\s+(?:total physical source lines|loc|lines? of code)\b", lowered,
    ):
        reasons.append("missing_requested_field:line_count")

    if plan.result_scope == "all":
        if re.search(r"\bshowing (?:only )?(?:the )?(?:first|top)\b", lowered):
            reasons.append("exhaustive_result_truncated")
        if "literal_assignments" in plan.tasks:
            count_match = _EXHAUSTIVE_ITEM_COUNT.search(answer)
            if count_match:
                returned = sum(1 for line in answer.splitlines() if line.startswith("- line "))
                if returned != int(count_match.group(1)):
                    reasons.append(
                        f"exhaustive_result_count_mismatch:{returned}/{count_match.group(1)}"
                    )
            elif not _EXHAUSTIVE_COVERAGE.search(answer):
                reasons.append("missing_exhaustive_result_count")

    return PlanContractValidation(not reasons, tuple(dict.fromkeys(reasons)))


def validate_evidence_answer(plan: QueryPlan, answer: str) -> PlanContractValidation:
    """Validate one evidence claim without applying final presentation limits.

    A three-line contract applies to the composed answer, not independently to
    every evidence claim. Keeping the contract on every subplan preserves the
    authority boundary while this narrow validation view prevents false claim
    failures during evidence collection.
    """
    return validate_plan_answer(replace(plan, response_contract=ResponseContract()), answer)


def execution_strategy_for_plan(plan: QueryPlan) -> str:
    """Select the least expensive execution mode that can satisfy the plan."""
    if plan.route != "technical":
        return "conversational" if plan.route == "conversational" else "clarification"
    required = tuple(subtask for subtask in plan.subtasks if subtask.required)
    # Comparing two entities inside one typed capability is still a direct
    # structured lookup. Agentic decomposition is reserved for multiple evidence
    # capabilities or multiple programs, where independent claims must be joined.
    if len(plan.programs) > 1 or len(required) > 1:
        return "agentic"
    if required:
        return "single_claim"
    return "standard_rag"



def _fallback_tasks_and_relations(q: str, intent: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Capture literal task/relation constraints that a semantic planner may not drop."""
    tasks: list[str] = []
    relations: list[str] = []

    if intent == "control_flow":
        if is_pagination_question(q):
            tasks.append("pagination_logic")
        elif is_complete_program_flow_question(q):
            tasks.append("complete_program_flow")
        if "referenc" in q or "mention" in q:
            tasks.append("paragraph_references")
            relations.append("referenced_by")
        if "what statements" in q or ("what does" in q and "paragraph" in q):
            tasks.append("paragraph_body")
            relations.append("contains")
        if not tasks and re.search(r"\b(?:from|starting at|starts? at)\b", q):
            tasks.append("path_from_paragraph")
            relations.append("starts_at")
        if not tasks:
            tasks.append("complete_program_flow")
    elif intent == "variable_dataflow":
        lifecycle = bool(re.search(
            r"\b(?:lifecycle|life cycle|trace|lineage|data movement|data flow|"
            r"intermediate|originated|transferred|feeds?|toward|through|"
            r"everything(?: about)?|all about)\b", q
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
    elif intent == "program_summary":
        tasks.append("program_summary")
    elif intent == "artifact_inventory":
        tasks.append("artifact_inventory")
    elif intent == "cics_operations":
        tasks.append("cics_operations")

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
