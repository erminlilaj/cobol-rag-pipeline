from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.final_scripts_answers import answer_from_final_scripts
from cobol_rag.index import configure_llamaindex, open_index
from cobol_rag.question_router import preflight_entity_answer
from cobol_rag.retrieve import RetrievalResult, retrieve, _detect_intent


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    answer: str
    sources: list[RetrievalResult]


class QueryError(Exception):
    """Raised when answer generation fails after retrieval succeeds."""


def answer_query(
    question: str,
    config: AppConfig,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
) -> QueryAnswer:
    current_question = _current_question(question)
    local_answer = _try_local_answer(current_question)
    if local_answer:
        return QueryAnswer(question=question, answer=local_answer, sources=[])

    if not _is_cobol_question(current_question):
        return QueryAnswer(
            question=question,
            answer=_answer_general_question(question, current_question, config),
            sources=[],
        )

    final_scripts_answer = answer_from_final_scripts(current_question)
    metadata_answer = _try_program_metadata_answer(current_question)
    entity_answer = preflight_entity_answer(current_question)
    sources: list[RetrievalResult] = []
    if _rag_runtime_available(config.embedding.base_url):
        try:
            sources = retrieve(current_question, config=config, top_k=top_k, chunk_types=chunk_types)
        except Exception:
            sources = []
    if not sources and not _rag_runtime_available(config.embedding.base_url):
        fallback_answer = final_scripts_answer or metadata_answer or entity_answer
        if fallback_answer:
            return QueryAnswer(
                question=question,
                answer=_maybe_polish_structured_answer(current_question, fallback_answer, config),
                sources=[],
            )
        return QueryAnswer(
            question=question,
            answer=(
                "I cannot search the RAG index right now because the configured Ollama service or embedding model "
                f"`{config.embedding.model}` is not responding. Start Ollama and make sure the embedding model is "
                "available, then try the COBOL question again."
            ),
            sources=[],
        )
    if _is_dead_code_question(current_question):
        try:
            targeted_sources = retrieve(
                f"{current_question} program.comments commented_out_code classification_counts dead_code unused_copybooks",
                config=config,
                top_k=max(top_k or config.retrieval.top_k, 12),
                chunk_types=chunk_types,
            )
            sources = _merge_sources(targeted_sources, sources)
        except Exception:
            pass
    if not sources:
        fallback_answer = final_scripts_answer or metadata_answer or entity_answer
        if fallback_answer:
            return QueryAnswer(
                question=question,
                answer=_maybe_polish_structured_answer(current_question, fallback_answer, config),
                sources=[],
            )
        return QueryAnswer(
            question=question,
            answer="I could not find relevant indexed sources for this question.",
            sources=[],
        )

    sources = _merge_sources(
        _privileged_evidence_sources(current_question, final_scripts_answer, metadata_answer),
        sources,
    )

    grounding_error = _validate_retrieved_evidence(current_question, sources)
    if grounding_error:
        return QueryAnswer(question=question, answer=grounding_error, sources=sources)

    sufficiency_error = _validate_intent_evidence(current_question, sources)
    if sufficiency_error:
        return QueryAnswer(question=question, answer=sufficiency_error, sources=sources)

    resources = open_index(config)
    prompt = _build_prompt(question=current_question, sources=sources)
    try:
        response = resources.runtime.llm.complete(prompt)
    except Exception as error:
        return QueryAnswer(
            question=question,
            answer=_offline_fallback_answer(config.llm.model, sources),
            sources=sources,
        )
    answer_text = str(response.text).strip()
    if _looks_off_evidence_answer(current_question, answer_text, sources):
        return QueryAnswer(
            question=question,
            answer=_grounded_fallback_answer(current_question, sources),
            sources=sources,
        )
    return QueryAnswer(
        question=question,
        answer=answer_text,
        sources=sources,
    )


def _append_rag_context_note(answer: str, sources: list[RetrievalResult]) -> str:
    """Show retrieval provenance when structured evidence is used as a fallback."""
    if not sources or "Sources used:" in answer:
        return answer
    return _append_provenance_note(answer, sources)


def _is_guardrail_answer(answer: str) -> bool:
    text = answer.lower()
    return (
        "do not have indexed" in text
        or "not indexed as a standalone program" in text
        or "do not have indexed analysis" in text
    )


def _should_prefer_final_scripts(question: str) -> bool:
    q = question.lower()
    if "business rule" in q or "business rules" in q:
        return False
    deterministic_terms = (
        "what is",
        "what does",
        "overview",
        "summary",
        "purpose",
        "copybook",
        "copy book",
        "copy member",
        "abend",
        "commarea",
        "communication area",
        "linking to",
        "link to",
        "call",
        "calls",
        "parameter",
        "parameters",
        "variable",
        "defined",
        "modified",
        "used",
        "where is",
        "where does",
        "line",
        "lines",
        "count",
        "how many",
        "dead code",
        "unused code",
        "commented",
    )
    return any(term in q for term in deterministic_terms)


def _validate_retrieved_evidence(question: str, sources: list[RetrievalResult]) -> str | None:
    """Refuse answers when the retrieved evidence does not ground named entities."""
    entities = _question_entities(question)
    if not entities:
        return None

    haystack = _evidence_haystack(sources)
    missing = [
        entity
        for entity in entities
        if not _entity_present_in_text(entity, haystack)
    ]
    if not missing:
        return None

    program = _program_from_sources(sources) or _program_from_question(question) or "the indexed program"
    formatted = ", ".join(f"`{entity}`" for entity in missing)
    return (
        f"I do not have indexed evidence for {formatted} in {program}. "
        "The retrieved chunks do not explicitly mention the requested entity, so I will not infer an answer from "
        "similar-looking control-flow or dataflow evidence."
    )


def _looks_off_evidence_answer(question: str, answer: str, sources: list[RetrievalResult]) -> bool:
    if not answer.strip():
        return True
    text = answer.lower()
    forbidden_patterns = (
        "https://",
        "http://",
        "github.com/",
        "git clone",
        "git checkout",
        "git branch",
        "new branch",
        "install cobol-rekt",
        "cobol-rekt init",
        "cobol-rekt add",
        "cobol-rekt report",
        "code snippet you provided",
        "larger program or system",
        "specific details of the program are not provided",
        "i don't understand what you mean by",
        "i do not understand what you mean by",
        "can you provide more context",
        "official website",
        "package manager",
    )
    if any(pattern in text for pattern in forbidden_patterns):
        return True

    if _external_links_or_markdown_links(answer):
        return True

    return False


def _external_links_or_markdown_links(answer: str) -> bool:
    return bool(re.search(r"\[[^\]]+\]\(https?://", answer) or re.search(r"https?://\S+", answer))


def _privileged_evidence_sources(
    question: str,
    final_scripts_answer: str | None,
    metadata_answer: str | None,
) -> list[RetrievalResult]:
    """Expose final_scripts output as RAG evidence instead of returning it directly."""
    sources: list[RetrievalResult] = []
    program = _program_from_question(question) or "PDCBVC"
    if final_scripts_answer and not _is_guardrail_answer(final_scripts_answer):
        sources.append(
            RetrievalResult(
                score=1.0,
                text=(
                    "Privileged structured evidence from final_scripts.\n"
                    "Use this as evidence, not as a prewritten answer.\n\n"
                    f"{final_scripts_answer}"
                ),
                metadata={
                    "program": program,
                    "source_system": "mapa_hamza",
                    "chunk_type": "privileged.final_scripts",
                    "source_chunk_type": "privileged.final_scripts",
                    "title": f"{program} final_scripts evidence",
                    "source_id": f"final_scripts:{content_hash_text(question)}",
                    "coverage_dimension": "privileged_structured_evidence",
                },
            )
        )
    if metadata_answer:
        sources.append(
            RetrievalResult(
                score=1.0,
                text=f"Privileged metadata evidence from final_scripts.\n\n{metadata_answer}",
                metadata={
                    "program": program,
                    "source_system": "mapa_hamza",
                    "chunk_type": "privileged.metadata",
                    "source_chunk_type": "privileged.metadata",
                    "title": f"{program} metadata evidence",
                    "source_id": f"metadata:{content_hash_text(question)}",
                    "coverage_dimension": "privileged_structured_evidence",
                },
            )
        )
    return sources


def content_hash_text(text: str) -> str:
    import hashlib

    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _validate_intent_evidence(question: str, sources: list[RetrievalResult]) -> str | None:
    intent = _detect_intent(question)
    haystack = _evidence_haystack(sources)
    chunk_types = {_chunk_type(source) for source in sources}
    program = _program_from_sources(sources) or _program_from_question(question) or "the indexed program"

    def missing(reason: str) -> str:
        return (
            f"I do not have enough indexed evidence to answer this from RAG for {program}. "
            f"Missing evidence: {reason}. I will not guess from unrelated retrieved chunks."
        )

    if intent == "program_summary":
        if not ("program.summary" in chunk_types or "program_summary" in chunk_types or "PRIVILEGED STRUCTURED EVIDENCE" in haystack):
            return missing("a `program.summary` or equivalent structured overview chunk")
    if intent == "error_paths":
        if "ABEND" in question.upper() and "ABEND00" not in haystack:
            return missing("retrieved error/control-flow evidence mentioning `ABEND00`")
        if not (chunk_types & {"error_path", "quality.error_paths.rich", "controlflow.cfg", "privileged.final_scripts"}):
            return missing("an `error_path`, `controlflow.cfg`, or structured final_scripts error-path source")
    if intent == "control_flow":
        q = question.lower()
        if any(term in q for term in ("page", "pages", "browse result", "npagt")):
            if not all(term in haystack for term in ("CALCOLA-NPAG", "NPAGT")):
                return missing("pagination facts mentioning `CALCOLA-NPAG` and `NPAGT`")
        if any(term in q for term in ("select", "selected", "row", "progressivo", "sceltai")):
            if not any(term in haystack for term in ("BROWSE-FASE2-SEL", "SCELTAI")):
                return missing("selection-validation facts mentioning `BROWSE-FASE2-SEL` or `SCELTAI`")
        if any(term in q for term in ("twcob-funzione", "twcob-id-sistema", "semaf", "pxcsemaf", "ip")):
            if not any(term in haystack for term in ("READ-TAB-SEMAF", "PXCSEMAF-STATUS", "XCTL-LIV4")):
                return missing("semaphore-flow facts mentioning `READ-TAB-SEMAF`, `PXCSEMAF-STATUS`, or `XCTL-LIV4`")
        if "enter" in q:
            if not any(term in haystack for term in ("DFHENTER", "BROWSE-FASE2-ENTER")):
                return missing("ENTER-key flow facts mentioning `DFHENTER` or `BROWSE-FASE2-ENTER`")
    if intent == "external_programs":
        if not (chunk_types & {"architecture.call_parameters", "architecture.calls", "architecture.call", "call_contract", "privileged.final_scripts"}):
            return missing("call/COMMAREA evidence such as `architecture.call_parameters` or `call_contract`")
    if intent == "variable_dataflow":
        if not (chunk_types & {"dataflow.variable", "dataflow.used_variables", "privileged.final_scripts"}):
            return missing("variable dataflow evidence")
    return None


def _question_entities(question: str) -> list[str]:
    """Extract COBOL-like entities that must be grounded in retrieved evidence."""
    ignored = {
        "ABEND",
        "ABOUT",
        "AREA",
        "ARE",
        "CALL",
        "CALCULATE",
        "CALCULATES",
        "CODE",
        "COBOL",
        "COMMAREA",
        "CONDITION",
        "CONDITIONS",
        "COPYBOOK",
        "DOES",
        "FIELD",
        "FIELDS",
        "FROM",
        "FUNCTION",
        "HAPPENS",
        "HOW",
        "KEY",
        "KEYS",
        "LEAD",
        "LINK",
        "PATH",
        "PATHS",
        "PRESSED",
        "PRESSES",
        "PRESSING",
        "PROGRESSIVO",
        "PROGRAM",
        "READ",
        "RESULT",
        "SCREEN",
        "SELECTS",
        "TABLE",
        "THE",
        "THERE",
        "TOTAL",
        "USED",
        "USER",
        "VARIABLE",
        "WHAT",
        "WHEN",
        "WHERE",
        "WHICH",
        "WITH",
        "WRITE",
        "WRITTEN",
    }
    entities: list[str] = []

    patterns = [
        r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b",
        r"\bABEND\d+\b",
        r"\bPF\d+\b",
        r"\b(?:PD|PX|PR|PB|DFH|SQL|TWCOB|M1|W|FUNZ|SCELTA)[A-Z0-9]{1,}\b",
    ]
    for pattern in patterns:
        entities.extend(re.findall(pattern, question.upper()))

    lower_targets = []
    for pattern in (
        r"\b(?:lead|leads|go|goes|transfer|transfers|xctl)\s+to\s+([a-z][a-z0-9_-]{2,})\b",
        r"\b(?:variable|field|paragraph|program|copybook|table)\s+([a-z][a-z0-9_-]{2,})\b",
        r"\b(?:about|for)\s+([a-z][a-z0-9_-]{2,})\b",
    ):
        lower_targets.extend(match.upper() for match in re.findall(pattern, question))

    entities.extend(lower_targets)

    unique: list[str] = []
    seen: set[str] = set()
    for entity in entities:
        cleaned = entity.strip().strip("`'\".,:;()[]{}").upper()
        if not cleaned or cleaned in ignored:
            continue
        if len(cleaned) <= 2 and not cleaned.startswith("PF"):
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            unique.append(cleaned)
    return unique[:12]


def _evidence_haystack(sources: list[RetrievalResult]) -> str:
    parts: list[str] = []
    for source in sources:
        parts.append(source.text)
        parts.extend(str(value) for value in source.metadata.values() if value is not None)
    return "\n".join(parts).upper()


def _entity_present_in_text(entity: str, haystack: str) -> bool:
    if "-" in entity:
        pattern = rf"(?<![A-Z0-9-]){re.escape(entity)}(?![A-Z0-9-])"
        prefix_pattern = rf"(?<![A-Z0-9-]){re.escape(entity)}-[A-Z0-9]+"
        if re.search(prefix_pattern, haystack):
            return True
    else:
        pattern = rf"(?<![A-Z0-9]){re.escape(entity)}(?![A-Z0-9])"
    return re.search(pattern, haystack) is not None


def _current_question(question: str) -> str:
    marker = "Current question:"
    if marker in question:
        return question.rsplit(marker, 1)[-1].strip()
    return question.strip()


@lru_cache(maxsize=8)
def _rag_runtime_available(base_url: str) -> bool:
    url = base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=0.75) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def _try_local_answer(question: str) -> str | None:
    text = question.strip().lower()
    normalized = text.strip(" ?.!").strip()
    if text in {"hi", "hello", "hey", "ciao", "salve", "buongiorno", "good morning", "good afternoon"}:
        return (
            "Hi. I am the COBOL RAG assistant for this workspace. I can help you inspect the indexed COBOL analysis. "
            "Try asking about called programs, COMMAREA parameters, forced values, DB2 tables, copybooks, or screen fields."
        )
    if normalized in {"what are you", "who are you", "what can you do", "help", "how can you help"}:
        return (
            "I am a local COBOL RAG assistant for this workspace. I search the indexed analysis artifacts and use them "
            "to answer questions about programs, calls, COMMAREA parameters, variables, DB2 tables, copybooks, comments, "
            "and control flow. If Ollama is running, I can generate a polished answer; if not, I can still show the best "
            "retrieved evidence."
        )
    if text in {"thanks", "thank you", "ok", "okay"}:
        return "You are welcome. Send me a COBOL question whenever you are ready."
    return None


def _is_cobol_question(question: str) -> bool:
    text = question.lower()
    cobol_terms = {
        "cobol",
        "pdc",
        "pdcbvc",
        "program",
        "paragraph",
        "section",
        "copybook",
        "copy book",
        "commarea",
        "cics",
        "xctl",
        "link",
        "db2",
        "sql",
        "dataset",
        "jcl",
        "variable",
        "screen",
        "map",
        "mapset",
        "field",
        "hardcoded",
        "forced value",
        "unused",
        "dead code",
        "commented code",
        "commented-out",
        "commented out",
        "literal",
        "call",
        "called",
        "table",
        "control flow",
        "dataflow",
        "data flow",
        "working-storage",
        "linkage",
        "business rule",
        "pf key",
        "pf keys",
        "path",
        "paths",
        "function key",
        "eibaid",
        "provenance",
        "conflict",
    }
    if any(term in text for term in cobol_terms):
        return True
    if re.search(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", question):
        return True
    return any(
        token.startswith(("W", "PD", "PB", "PR", "PX", "TWCOB", "SQL", "FUNZ", "M1", "SCELTA"))
        for token in re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", question.upper())
    )


def _answer_general_question(full_question: str, current_question: str, config: AppConfig) -> str:
    prompt = f"""You are a helpful local assistant inside a COBOL RAG workspace.

Answer the user's general question normally. If the user asks about this repository or indexed COBOL code,
say they should ask a code-specific question so you can use RAG evidence.

Question:
{current_question}

Conversation context, if any:
{full_question if full_question != current_question else "none"}

Answer:
"""
    try:
        runtime = configure_llamaindex(config)
        response = runtime.llm.complete(prompt)
    except Exception:
        return (
            "I can answer general questions too, but the configured Ollama model "
            f"`{config.llm.model}` is not responding right now. For COBOL questions I can still search the indexed "
            "evidence and show useful snippets."
        )
    return str(response.text).strip()


def _maybe_polish_structured_answer(question: str, evidence_answer: str, config: AppConfig) -> str:
    if not config.answers.llm_polish_final_scripts:
        return evidence_answer

    prompt = f"""You are the configured COBOL RAG LLM: {config.llm.model}.

Rewrite the evidence answer below into a clear, concise answer for the user.

Rules:
- Use only the evidence answer. Do not add facts, assumptions, COBOL general knowledge, or guessed meanings.
- Preserve concrete names, line numbers, citations, counts, and "not indexed" / "not evidenced" limitations exactly.
- If the evidence answer says something is missing or unknown, keep that limitation.
- Keep the answer compact and practical.

Question:
{question}

Evidence answer:
{evidence_answer}

Final answer:
"""
    try:
        runtime = configure_llamaindex(config)
        response = runtime.llm.complete(prompt)
    except Exception:
        return evidence_answer

    text = str(response.text).strip()
    if not text:
        return evidence_answer
    if not _polish_preserves_structured_evidence(evidence_answer, text):
        return evidence_answer
    return text


def _polish_preserves_structured_evidence(evidence_answer: str, polished_answer: str) -> bool:
    required = _protected_evidence_terms(evidence_answer)
    if not required:
        return True

    haystack = polished_answer.upper()
    missing = [term for term in required if term.upper() not in haystack]
    if missing:
        return False
    return not _looks_like_identifier_corruption(evidence_answer, polished_answer)


def _protected_evidence_terms(text: str) -> list[str]:
    terms: list[str] = []
    terms.extend(re.findall(r"`([^`]{3,80})`", text))
    terms.extend(re.findall(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", text))
    terms.extend(re.findall(r"\b(?:PD|PX|PR|DFH|TWCOB|SQL|M1|W)[A-Z0-9-]{2,}\b", text))
    terms.extend(re.findall(r"\bABEND00\b", text))
    terms.extend(re.findall(r"\bline\s+\d+\b", text, flags=re.IGNORECASE))

    ignored = {"UNKNOWN", "WORKING-STORAGE", "COMMAREA", "COPY", "CICS", "CALL", "JUMP", "LINK"}
    unique: list[str] = []
    seen: set[str] = set()
    for term in terms:
        cleaned = term.strip()
        if not cleaned or cleaned.upper() in ignored or cleaned in seen:
            continue
        seen.add(cleaned)
        unique.append(cleaned)
    return unique[:120]


def _looks_like_identifier_corruption(evidence_answer: str, polished_answer: str) -> bool:
    original_terms = {term.upper() for term in _protected_evidence_terms(evidence_answer)}
    polished_terms = set(re.findall(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", polished_answer.upper()))
    suspicious = [
        term
        for term in polished_terms
        if term not in original_terms and (
            term.startswith("BROWN")
            or term.startswith("BROWSE") and not any(source.startswith(term) or term.startswith(source) for source in original_terms)
        )
    ]
    return bool(suspicious)


def _try_program_metadata_answer(question: str) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("how many", "number of", "count")):
        return None

    program = _program_from_question(question)
    if not program:
        return None

    comments_payload = _load_final_script_comments_payload(program)
    if comments_payload and any(term in q for term in ("line", "lines", "loc", "code lines")):
        metrics = comments_payload.get("metrics", {})
        total_lines = metrics.get("total_lines")
        comment_count = comments_payload.get("count")
        commented_out = comments_payload.get("classification_counts", {}).get("commented_out_code")
        if total_lines is not None:
            details = [f"{program} has {total_lines} total source lines."]
            if comment_count is not None:
                details.append(f"The comments artifact also reports {comment_count} comment lines.")
            if commented_out is not None:
                details.append(f"{commented_out} of those are classified as commented-out code.")
            details.append("Source: `program.comments.json` metrics.")
            return " ".join(details)

    return None


def _try_dead_code_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    if not _is_dead_code_question(question):
        return None

    evidence = [
        source
        for source in sources
        if source.metadata.get("chunk_type") in {"dead_code", "unused_copybooks", "commented_out_code"}
        or _has_commented_code_evidence(source.text)
    ]
    if not evidence:
        return (
            "I cannot confirm unused code or unused COPY members from the current indexed evidence. "
            "The retrieved chunks for this question are not explicit dead-code/unused-copy analysis chunks, "
            "so I will not infer that PDCBVC has unused code from unrelated call/copybook evidence.\n\n"
            "What is missing: a dedicated artifact such as `dead_code`, `unused_copybooks`, or "
            "`commented_out_code` for PDCBVC."
        )

    lines = ["Dead/unused-code evidence found:"]
    listed_any = False
    full_comment_items = _load_commented_code_items(question, sources)
    if full_comment_items:
        lines.append("commented_out_code:")
        lines.extend(f"- line {item['line']}: {item['text']}" for item in full_comment_items[:20])
        listed_any = True

    for source in evidence:
        chunk_type = source.metadata.get("chunk_type", "source")
        if _has_commented_code_evidence(source.text):
            items = _commented_code_items(source.text)
            if items:
                if listed_any:
                    continue
                lines.append("commented_out_code:")
                lines.extend(f"- line {item['line']}: {item['text']}" for item in items[:12])
                listed_any = True
                continue
        lines.append(f"{chunk_type}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=6))
    if "unused copy" in question.lower() or "/copy" in question.lower():
        lines.append(
            "Unused COPY members: no dedicated unused-copy analysis was found in the current index. "
            "The index can show COPY members used/missing, but not prove unused COPY members yet."
        )
    return "\n".join(lines)


def _is_dead_code_question(question: str) -> bool:
    q = question.lower()
    return any(
        term in q
        for term in (
            "unused",
            "dead code",
            "unused code",
            "unused copy",
            "inactive",
            "commented-out",
            "commented out",
            "unreachable",
        )
    )


def _try_business_rules_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("business rule", "business rules", "rule ", "rules ", "br-")):
        return None
    evidence = [source for source in sources if _chunk_type(source) in {"business_rule", "business_rule.rag"}]
    if not evidence:
        return None
    lines = ["Business-rule evidence found:"]
    for source in evidence[:8]:
        title = source.metadata.get("title") or source.metadata.get("chunk_id") or _chunk_type(source)
        lines.append(f"\n{title}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=5))
    return _append_provenance_note("\n".join(lines), evidence)


def _try_ui_navigation_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("pf key", "pf keys", "function key", "function keys", "enter", "eibaid", "navigation")):
        return None
    evidence = [source for source in sources if _chunk_type(source) in {"ui.cics.navigation", "screen.key_dispatch"}]
    if not evidence:
        return None
    lines = ["UI/navigation evidence found:"]
    for source in evidence[:6]:
        title = source.metadata.get("title") or source.metadata.get("chunk_id") or _chunk_type(source)
        lines.append(f"\n{title}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=8))
    return _append_provenance_note("\n".join(lines), evidence)


def _try_variable_dataflow_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("dataflow", "data flow", "read", "write", "where is", "where are", "variable")):
        return None
    requested = _extract_identifier(question)
    evidence = [
        source
        for source in sources
        if _chunk_type(source) in {"dataflow.variable", "dataflow.used_variables"}
        and (not requested or requested.upper() in (source.text + " " + str(source.metadata)).upper())
    ]
    if not evidence:
        return None
    subject = f" for `{requested.upper()}`" if requested else ""
    lines = [f"Variable dataflow evidence{subject}:"]
    for source in evidence[:6]:
        title = source.metadata.get("title") or source.metadata.get("variable") or source.metadata.get("chunk_id") or _chunk_type(source)
        lines.append(f"\n{title}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=8))
    return _append_provenance_note("\n".join(lines), evidence)


def _try_sql_includes_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("sql include", "sql includes", "sqlca", "db2 include")):
        return None
    evidence = [source for source in sources if _chunk_type(source) == "architecture.sqlinclude"]
    if not evidence:
        return None
    lines = ["SQL include evidence found:"]
    for source in evidence[:8]:
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=5))
    return _append_provenance_note("\n".join(lines), evidence)


def _try_jcl_file_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("jcl", "dataset", "datasets", "file i/o", "file io", "batch job", "job ")):
        return None
    evidence = [
        source
        for source in sources
        if _chunk_type(source).startswith("jcl.")
        or _chunk_type(source) in {"global.jcl_program_map.summary", "file_io", "jcl.file_io"}
    ]
    if not evidence:
        return None
    lines = ["JCL/file evidence found:"]
    for source in evidence[:8]:
        title = source.metadata.get("title") or source.metadata.get("chunk_id") or _chunk_type(source)
        lines.append(f"\n{title}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=6))
    return _append_provenance_note("\n".join(lines), evidence)


def _try_conflict_provenance_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("conflict", "provenance", "source system", "trust", "confidence", "why different", "limitation")):
        return None
    evidence = [
        source
        for source in sources
        if _chunk_type(source) in {"integration.conflicts", "integration.entity_link"}
        or source.metadata.get("coverage_dimension") in {"conflict_report", "quality_confidence"}
    ]
    if not evidence:
        return None
    lines = ["Conflict/provenance evidence found:"]
    for source in evidence[:6]:
        title = source.metadata.get("title") or source.metadata.get("chunk_id") or _chunk_type(source)
        lines.append(f"\n{title}:")
        lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=6))
    return _append_provenance_note("\n".join(lines), evidence)


def _chunk_type(source: RetrievalResult) -> str:
    raw = (
        source.metadata.get("source_chunk_type")
        or source.metadata.get("original_chunk_type")
        or source.metadata.get("chunk_type")
        or source.metadata.get("type")
        or ""
    )
    chunk_type = str(raw)
    for prefix in ("cobol_rekt.", "mapa_hamza.", "mapa."):
        if chunk_type.startswith(prefix):
            chunk_type = chunk_type[len(prefix):]
            break
    if chunk_type.startswith("screen_"):
        return chunk_type.replace("_", ".")
    if chunk_type.startswith("dataflow_"):
        return chunk_type.replace("_", ".", 1)
    return chunk_type


def _extract_identifier(question: str) -> str | None:
    matches = re.findall(r"\b[A-Z][A-Z0-9]+(?:-[A-Z0-9]+)+\b", question.upper())
    return matches[0] if matches else None


def _append_provenance_note(answer: str, sources: list[RetrievalResult]) -> str:
    by_source: dict[str, set[str]] = {}
    for source in sources:
        source_system = str(source.metadata.get("source_system") or "indexed")
        by_source.setdefault(source_system, set()).add(_chunk_type(source) or str(source.metadata.get("chunk_type", "source")))
    if not by_source:
        return answer
    lines = [answer.rstrip(), "", "Sources used:"]
    for source_system, chunk_types in sorted(by_source.items()):
        label = {
            "mapa_hamza": "MAPA/Hamza",
            "cobol_rekt": "cobol-rekt",
            "cobol-rekt": "cobol-rekt",
            "integration": "integration",
            "combined": "integration",
        }.get(source_system, source_system)
        lines.append(f"- {label}: {', '.join(sorted(chunk_types))}")
    if any(source.metadata.get("entity_key") for source in sources):
        lines.append("Confidence note: linked evidence shares an exact cross-source entity key.")
    return "\n".join(lines)


def _merge_sources(*groups: list[RetrievalResult]) -> list[RetrievalResult]:
    merged: list[RetrievalResult] = []
    seen: set[tuple[str, str]] = set()
    for group in groups:
        for source in group:
            key = (
                str(source.metadata.get("source_id", "")),
                str(source.metadata.get("chunk_index", source.text[:80])),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(source)
    return merged


def _has_commented_code_evidence(text: str) -> bool:
    return "classification: commented_out_code" in text or "classification_counts.commented_out_code" in text


def _commented_code_items(text: str) -> list[dict[str, str]]:
    by_index: dict[str, dict[str, str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        match = re.match(r"comments\[(\d+)\]\.([^:]+):\s*(.*)", line)
        if not match:
            continue
        index, field, value = match.groups()
        item = by_index.setdefault(index, {"line": "?", "text": "", "classification": ""})
        value = value.strip().strip('"')
        if field == "classification":
            item["classification"] = value
        elif field == "line":
            item["line"] = value
        elif field == "text_raw":
            item["text"] = value
        elif field == "text" and not item.get("text"):
            item["text"] = value
    return [
        item
        for _index, item in sorted(by_index.items(), key=lambda pair: int(pair[0]))
        if item.get("classification") == "commented_out_code" and item.get("text")
    ]


def _load_commented_code_items(question: str, sources: list[RetrievalResult]) -> list[dict[str, str]]:
    program = _program_from_question(question) or _program_from_sources(sources)
    if not program:
        return []

    items: list[dict[str, str]] = []
    for source in sources:
        source_path = source.metadata.get("source_path")
        if not source_path:
            continue
        path = Path(source_path)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists() or path.suffix.lower() != ".jsonl":
            continue
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                doc = json.loads(line)
                if doc.get("program") != program or doc.get("type") != "program.comments":
                    continue
                items.extend(_commented_code_items(doc.get("text", "")))
        except (OSError, json.JSONDecodeError):
            continue

    final_script_items = _load_final_script_commented_code_items(program)
    return _unique_comment_items([*items, *final_script_items])


def _load_final_script_commented_code_items(program: str) -> list[dict[str, str]]:
    payload = _load_final_script_comments_payload(program)
    if not payload:
        return []
    items = []
    for comment in payload.get("comments", []):
        if comment.get("classification") != "commented_out_code":
            continue
        items.append(
            {
                "line": str(comment.get("line", "?")),
                "text": str(comment.get("text_raw") or comment.get("text") or "").strip(),
            }
        )
    return [item for item in items if item.get("text")]


def _load_final_script_comments_payload(program: str) -> dict | None:
    comments_path = (
        Path.cwd().parent
        / "control_flow"
        / "artifacts"
        / "final"
        / "final_scripts"
        / "program.comments"
        / "program.comments.json"
    )
    if not comments_path.exists():
        return []
    try:
        payload = json.loads(comments_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("program") != program:
        return None
    return payload


def _program_from_question(question: str) -> str | None:
    ignored = {
        "IS",
        "THERE",
        "ANY",
        "UNUSED",
        "CODE",
        "COPY",
        "THIS",
        "PROGRAM",
        "DEAD",
        "COMMENTED",
        "OUT",
    }
    candidates = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", question.upper())
        if token not in ignored
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _program_from_sources(sources: list[RetrievalResult]) -> str | None:
    for source in sources:
        program = source.metadata.get("program")
        if program and program != "__GLOBAL__":
            return str(program)
    return None


def _unique_comment_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    by_line: dict[str, dict[str, str]] = {}
    for item in items:
        line = item.get("line", "")
        text = item.get("text", "")
        if not line or not text:
            continue
        existing = by_line.get(line)
        if existing is None or len(text) > len(existing.get("text", "")):
            by_line[line] = item
    return sorted(by_line.values(), key=lambda item: int(item.get("line", "0")) if item.get("line", "").isdigit() else 0)


def _first_nonempty_lines(text: str, limit: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:limit]


def _offline_fallback_answer(model: str, sources: list[RetrievalResult]) -> str:
    lines = [
        f"I found relevant indexed sources, but the configured Ollama model `{model}` did not answer.",
        "Here are the best retrieved snippets so you can still inspect the evidence:",
        "",
    ]
    for index, source in enumerate(sources[:4], start=1):
        chunk_type = source.metadata.get("chunk_type", "source")
        source_id = source.metadata.get("source_id", f"source-{index}")
        preview = " ".join(source.text.split())
        if len(preview) > 420:
            preview = preview[:420].rstrip() + "..."
        lines.append(f"{index}. `{chunk_type}` `{source_id}`")
        lines.append(preview)
        lines.append("")
    lines.append("Start Ollama and make sure the configured model is available to get a generated answer.")
    return "\n".join(lines).strip()


def _grounded_fallback_answer(question: str, sources: list[RetrievalResult]) -> str:
    facts = _extract_grounding_facts(sources, per_source_limit=10)
    if facts:
        lines = [
            "I found indexed evidence for this question, but the generated prose was not grounded enough, so here are the supported facts:",
            "",
        ]
        fact_lines = [line for line in facts.splitlines() if line.strip()]
        lines.extend(fact_lines[:32])
        lines.append("")
        lines.append(_compact_provenance_line(sources))
        return "\n".join(lines).strip()
    return _offline_fallback_answer("configured LLM", sources)


def _compact_provenance_line(sources: list[RetrievalResult]) -> str:
    by_source: dict[str, set[str]] = {}
    for source in sources:
        source_system = str(source.metadata.get("source_system") or "indexed")
        by_source.setdefault(source_system, set()).add(_chunk_type(source) or str(source.metadata.get("chunk_type", "source")))
    if not by_source:
        return "Sources used: indexed evidence."
    parts = []
    for source_system, chunk_types in sorted(by_source.items()):
        label = {
            "mapa_hamza": "MAPA/Hamza",
            "cobol_rekt": "cobol-rekt",
            "cobol-rekt": "cobol-rekt",
            "integration": "integration",
            "combined": "integration",
        }.get(source_system, source_system)
        parts.append(f"{label}: {', '.join(sorted(chunk_types))}")
    return "Sources used: " + "; ".join(parts) + "."


def _build_prompt(question: str, sources: list[RetrievalResult]) -> str:
    intent = _detect_intent(question)
    entities = _question_entities(question)
    facts = _extract_grounding_facts(sources)
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        source_id = source.metadata.get("source_id", f"source-{index}")
        source_path = source.metadata.get("source_path", "")
        chunk_type = source.metadata.get("chunk_type") or source.metadata.get("source_chunk_type") or ""
        source_system = source.metadata.get("source_system") or "indexed"
        title = source.metadata.get("title") or ""
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_id: {source_id}",
                    f"source_path: {source_path}",
                    f"source_system: {source_system}",
                    f"chunk_type: {chunk_type}",
                    f"title: {title}",
                    "text:",
                    source.text,
                ]
            )
        )

    context = "\n\n".join(context_blocks)
    return f"""You are answering a question using only the retrieved sources below.

Rules:
- Answer only from the retrieved sources.
- If the sources do not contain the answer, say that the indexed sources do not contain enough information.
- If the user names a program, variable, paragraph, copybook, file, or table that is not present in the sources,
  say it is not present in the indexed evidence. Do not explain it from general COBOL knowledge.
- Treat exact identifiers as mandatory grounding. Do not answer about a similar identifier.
- For control-flow/path questions, list only edges, conditions, or targets explicitly present in the sources.
- For variable/field questions, mention only reads, writes, controlling conditions, or origins explicitly present.
- For calculations, include the exact operands/formula only if the retrieved facts or sources show them.
- Prefer the compact extracted facts below over raw flattened JSON text when they are available.
- If a source is marked `privileged.final_scripts` or `privileged.metadata`, treat it as the highest-trust
  structured evidence, but still write a fresh answer rather than copying it verbatim.
- If raw sources conflict with privileged structured evidence, prefer the privileged structured evidence and mention
  the conflict only when it is explicit in the retrieved sources.
- Do not include external URLs, Git commands, installation steps, or generic tool documentation unless those exact
  facts appear in the retrieved sources.
- Do not say the user provided a code snippet. The only context you have is the retrieved evidence.
- Keep the answer concise.
- Mention source ids inline when useful, but do not invent source ids.

Question:
{question}

Detected intent:
{intent}

Requested entities that must be grounded:
{", ".join(entities) if entities else "none"}

Compact extracted facts:
{facts if facts else "none"}

Retrieved sources:
{context}

Answer:
"""


def _extract_grounding_facts(sources: list[RetrievalResult], per_source_limit: int = 16) -> str:
    fact_lines: list[str] = []
    fact_patterns = (
        "content.condition",
        "content.action",
        "content.target",
        "content.source",
        "content.edge",
        "content.edges",
        "content.variable",
        "content.variables_read",
        "content.variables_modified",
        "content.read",
        "content.write",
        "content.calls",
        "content.command",
        "content.paragraph",
        "content.formula",
        "content.evidence",
        "condition:",
        "target:",
        "action:",
        "edge:",
        "evidence:",
    )
    code_patterns = (
        " IF ",
        " GO TO ",
        " PERFORM ",
        " MOVE ",
        " DIVIDE ",
        " ADD ",
        " EXEC SQL",
        " EXEC CICS",
        " CICSLINK",
        " CICSXCTL",
        " CALL ",
    )

    for index, source in enumerate(sources, start=1):
        source_id = source.metadata.get("source_id", f"source-{index}")
        chunk_type = source.metadata.get("chunk_type") or source.metadata.get("source_chunk_type") or "source"
        picked: list[str] = []
        if str(chunk_type).startswith("privileged."):
            lines = _first_nonempty_lines(source.text, limit=36)
            if lines:
                fact_lines.append(f"[{index}] {chunk_type} {source_id}")
                fact_lines.extend(f"- {line}" for line in lines)
            continue
        for raw_line in source.text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            upper = f" {line.upper()} "
            if any(pattern in lower for pattern in fact_patterns) or any(pattern in upper for pattern in code_patterns):
                picked.append(line)
            if len(picked) >= per_source_limit:
                break
        if picked:
            fact_lines.append(f"[{index}] {chunk_type} {source_id}")
            fact_lines.extend(f"- {line}" for line in picked)
    return "\n".join(fact_lines[:180])
