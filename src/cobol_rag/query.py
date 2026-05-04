from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.final_scripts_answers import answer_from_final_scripts
from cobol_rag.index import open_index
from cobol_rag.retrieve import RetrievalResult, retrieve


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
    conversation_history: str | None = None,
) -> QueryAnswer:
    final_scripts_answer = answer_from_final_scripts(question)
    if final_scripts_answer:
        return QueryAnswer(question=question, answer=final_scripts_answer, sources=[])

    metadata_answer = _try_program_metadata_answer(question)
    if metadata_answer:
        return QueryAnswer(question=question, answer=metadata_answer, sources=[])

    sources = retrieve(question, config=config, top_k=top_k, chunk_types=chunk_types)
    if not sources:
        return QueryAnswer(
            question=question,
            answer="I could not find relevant indexed sources for this question.",
            sources=[],
        )

    direct_answer = (
        _try_dead_code_answer(question, sources)
        or _try_static_values_answer(question, sources)
        or _try_external_programs_answer(question, sources)
        or _try_datasets_tables_answer(question, sources)
        or _try_comments_answer(question, sources)
        or _try_program_summary_answer(question, sources)
        or _try_copybook_answer(question, sources)
    )
    if direct_answer:
        return QueryAnswer(question=question, answer=direct_answer, sources=sources)

    resources = open_index(config)
    system_prompt = _load_system_prompt(config)
    prompt = _build_prompt(
        question=question,
        sources=sources,
        system_prompt=system_prompt,
        conversation_history=conversation_history,
    )
    try:
        response = resources.runtime.llm.complete(prompt)
    except Exception as error:
        raise QueryError(
            "Answer generation failed after retrieval succeeded. "
            "Check that the configured LLM is available in Ollama and fits in memory."
        ) from error
    return QueryAnswer(
        question=question,
        answer=str(response.text).strip(),
        sources=sources,
    )


def _load_system_prompt(config: AppConfig) -> str:
    raw = config.answers.system_prompt_path
    if not raw:
        return ""
    path = Path(raw)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def _try_program_metadata_answer(question: str) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("how many", "number of", "count")):
        return None
    if not any(term in q for term in ("line", "lines", "loc", "code lines")):
        return None

    program = _program_from_question(question)
    if not program:
        return None
    payload = _load_final_script_comments_payload(program)
    if not payload:
        return None

    total_lines = payload.get("metrics", {}).get("total_lines")
    if total_lines is None:
        return None
    comment_count = payload.get("count")
    commented_out = payload.get("classification_counts", {}).get("commented_out_code")
    details = [f"{program} has {total_lines} total source lines."]
    if comment_count is not None:
        details.append(f"The comments artifact also reports {comment_count} comment lines.")
    if commented_out is not None:
        details.append(f"{commented_out} of those are classified as commented-out code.")
    details.append("Source: `program.comments.json` metrics.")
    return " ".join(details)


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
        return None
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
        "HOW",
        "MANY",
        "LINES",
        "LINE",
        "NUMBER",
        "COUNT",
    }
    candidates = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", question.upper())
        if token not in ignored
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _build_prompt(
    question: str,
    sources: list[RetrievalResult],
    system_prompt: str = "",
    conversation_history: str | None = None,
) -> str:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        meta = source.metadata
        source_id = meta.get("source_id", f"source-{index}")
        source_path = meta.get("source_path", "")
        chunk_type = meta.get("chunk_type", "")
        program = meta.get("program", "")
        parse_quality = meta.get("parse_quality", "")
        header_parts = [f"[Source {index}]", f"source_id: {source_id}"]
        if source_path:
            header_parts.append(f"source_path: {source_path}")
        if chunk_type:
            header_parts.append(f"chunk_type: {chunk_type}")
        if program:
            header_parts.append(f"program: {program}")
        if parse_quality:
            header_parts.append(f"parse_quality: {parse_quality}")
        header_parts.append("text:")
        header_parts.append(source.text)
        context_blocks.append("\n".join(header_parts))

    context = "\n\n".join(context_blocks)

    prefix = f"{system_prompt}\n\n" if system_prompt else ""
    history_block = ""
    if conversation_history:
        history_block = (
            "Conversation history, for resolving follow-up wording only. "
            "Do not treat it as indexed evidence:\n"
            f"{conversation_history}\n\n"
        )
    return f"""{prefix}Question:
{question}

{history_block}\
Retrieved sources:
{context}

Answer:
"""


def _try_dead_code_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("unused", "dead code", "inactive", "commented-out", "commented out", "unreachable")):
        return None

    evidence = [
        source for source in sources
        if source.metadata.get("chunk_type") in {"dead_code", "unused_copybooks", "commented_out_code"}
    ]
    if not evidence:
        return (
            "The indexed sources do not contain enough explicit dead-code or unused-copy evidence "
            "to answer this safely. I will not infer that there is no unused code from unrelated chunks."
        )

    lines = ["Dead/unused-code evidence found:"]
    for source in evidence:
        chunk_type = source.metadata.get("chunk_type", "source")
        first_lines = _first_nonempty_lines(source.text, limit=6)
        lines.append(f"{chunk_type}:")
        lines.extend(f"- {line}" for line in first_lines)
    return "\n".join(lines)


def _try_static_values_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("forced value", "forced values", "static value", "static values", "hardcoded", "hard-coded")):
        return None

    static_sources = [
        source for source in sources
        if source.metadata.get("chunk_type") in {"static_values", "dataflow.literal_assignments"}
    ]
    if not static_sources:
        return "The retrieved sources do not contain a static/forced-values chunk for this question."

    value_lines = []
    for source in static_sources:
        if source.metadata.get("chunk_type") == "dataflow.literal_assignments":
            value_lines.extend(_assignment_lines(source.text))
            continue
        for line in source.text.splitlines():
            clean = line.strip()
            if clean.startswith("- "):
                value_lines.append(clean)
    if not value_lines:
        return "The static/forced-values chunk was retrieved, but it does not list individual values."

    return "Forced/static values found:\n" + "\n".join(_unique_static_value_lines(value_lines))


def _try_external_programs_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("outside program", "outside programs", "external program", "external programs", "external call", "external calls", "called program", "called programs", "with parameters", "commarea")):
        return None

    call_sources = [
        source for source in sources
        if source.metadata.get("chunk_type") in {"external_program_calls", "architecture.call_parameters"}
    ]
    if call_sources:
        lines = []
        for source in call_sources:
            if source.metadata.get("chunk_type") == "architecture.call_parameters":
                lines.extend(_call_parameter_lines(source.text))
                continue
            for line in source.text.splitlines():
                clean = line.strip()
                if clean.startswith("- "):
                    lines.append(_clean_external_call_line(clean))
        if "commarea" in q:
            lines = [line for line in lines if "commarea" in line.lower()]
        if lines:
            return "External program calls:\n" + "\n".join(_unique_preserving_order(lines))
        if "commarea" in q:
            return "The retrieved external-program chunk does not list any calls with COMMAREA."

    fallback = [
        source for source in sources
        if source.metadata.get("chunk_type") in {"cics_operations", "dependencies"}
    ]
    if fallback:
        return (
            "The retrieved sources contain dependency/program-transfer evidence, but no dedicated "
            "`external_program_calls` chunk with parameters. Relevant source text:\n"
            + "\n".join(_first_nonempty_lines("\n".join(source.text for source in fallback), limit=8))
        )
    return "The retrieved sources do not contain external-program call evidence."


def _assignment_lines(text: str) -> list[str]:
    lines: list[str] = []
    for sentence in re.split(r"(?<=\.)\s+", text.replace("\n", " ")):
        clean = sentence.strip()
        if " gets " in clean and " line " in clean:
            lines.append(f"- {clean}")
    return lines


def _call_parameter_lines(text: str) -> list[str]:
    lines: list[str] = []
    compact = " ".join(text.split())
    match = re.search(r"Embedding:\s*(.*?)(?:\s*Metadata:|$)", compact)
    call_text = match.group(1) if match else compact
    for sentence in re.split(r"(?<=\.)\s+", call_text):
        clean = sentence.strip()
        if " uses " in clean and ("via" in clean or "CALL" in clean or "CICS" in clean):
            lines.append(f"- {clean}")
    return lines


def _try_datasets_tables_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("dataset", "datasets", "table", "tables", "file", "files", "mapset", "mapsets", "queue", "queues", "transaction id", "resources")):
        return None

    resource_sources = [
        source for source in sources
        if source.metadata.get("chunk_type") == "datasets_tables_resources"
    ]
    if resource_sources:
        lines = []
        for source in resource_sources:
            for line in source.text.splitlines():
                clean = line.strip()
                if clean and not clean.lower().startswith("datasets, tables"):
                    lines.append(clean)
        if lines:
            return "Datasets, tables, and resources:\n" + "\n".join(_unique_preserving_order(lines))

    fallback = [source for source in sources if source.metadata.get("chunk_type") == "dependencies"]
    if fallback:
        return (
            "The retrieved sources contain dependency evidence, but no dedicated "
            "`datasets_tables_resources` chunk. Relevant source text:\n"
            + "\n".join(_first_nonempty_lines("\n".join(source.text for source in fallback), limit=8))
        )
    return "The retrieved sources do not contain dataset/table/resource evidence."


def _try_comments_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if "comment" not in q:
        return None

    comment_sources = [
        source for source in sources
        if source.metadata.get("chunk_type") in {"commented_out_code", "comments"}
    ]
    if comment_sources:
        lines = ["Comment/commented-code evidence found:"]
        for source in comment_sources:
            chunk_type = source.metadata.get("chunk_type", "source")
            lines.append(f"{chunk_type}:")
            lines.extend(f"- {line}" for line in _first_nonempty_lines(source.text, limit=8))
        return "\n".join(lines)

    return (
        "The retrieved sources do not contain a dedicated comments or commented-out-code chunk, "
        "so I cannot list program comments safely from the current index."
    )


def _try_program_summary_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("program about", "what is the program", "what this program", "what does this program", "purpose", "overview", "summary")):
        return None

    summary_source = next(
        (source for source in sources if source.metadata.get("chunk_type") == "program_summary"),
        None,
    )
    if summary_source is None:
        return None

    lines = _summary_lines(summary_source.text, limit=5)
    if not lines:
        return "The program summary chunk was retrieved, but it does not contain summary text."
    return "Program summary:\n" + "\n".join(lines)


def _try_copybook_answer(question: str, sources: list[RetrievalResult]) -> str | None:
    q = question.lower()
    if "copybook" not in q and "copy book" not in q:
        return None

    if any(term in q for term in ("which line", "what line", "line number", "lines are", "lines mention", "mentioned")):
        return (
            "The retrieved copybook sources do not contain source line numbers for copybook mentions. "
            "They can report copybook names, resolved/stubbed counts, and limitations, but not exact COPY statement lines yet."
        )

    if any(term in q for term in ("parameter", "parameters", "field", "fields", "from copybook", "from copybooks")):
        return (
            "The retrieved copybook sources do not contain copybook field/parameter extraction. "
            "They only provide copybook usage and resolution status. A dedicated copybook-variables chunk is needed to answer this safely."
        )

    facts_text = "\n".join(source.text for source in sources)
    copybooks_used = _extract_list_fact(facts_text, "copybooks_used")
    stubbed_copybooks = _extract_stubbed_copybooks(facts_text)
    total = _extract_int_fact(facts_text, "total_copybooks")
    resolved = _extract_int_fact(facts_text, "resolved_copybooks")
    stubbed_count = _extract_int_fact(facts_text, "stubbed_copybook_count")

    if not any([copybooks_used, stubbed_copybooks, total, resolved, stubbed_count]):
        return None

    lines = []
    count_parts = []
    if total is not None:
        count_parts.append(f"{total} total")
    if resolved is not None:
        count_parts.append(f"{resolved} resolved/found")
    if stubbed_count is not None:
        count_parts.append(f"{stubbed_count} stubbed")
    if count_parts:
        lines.append("Copybook status: " + ", ".join(count_parts) + ".")

    if copybooks_used:
        lines.append("Copybooks listed as used: " + ", ".join(copybooks_used) + ".")
    if stubbed_copybooks:
        lines.append("Stubbed copybooks: " + ", ".join(stubbed_copybooks) + ".")

    if "stubbed" in facts_text.lower() or "degraded" in facts_text.lower():
        lines.append("Limitation: the retrieved analysis reports degraded parse quality and stubbed copybooks.")

    return "\n".join(lines)


def _extract_int_fact(text: str, field: str) -> int | None:
    match = re.search(rf"{re.escape(field)}:\s*(\d+)", text)
    if not match:
        return None
    return int(match.group(1))


def _extract_list_fact(text: str, field: str) -> list[str]:
    match = re.search(rf"{re.escape(field)}:\s*([^\n]+)", text)
    if not match:
        return []
    return _unique_preserving_order(
        item.strip().strip(".")
        for item in match.group(1).split(",")
        if item.strip()
    )


def _extract_stubbed_copybooks(text: str) -> list[str]:
    match = re.search(r"stubbed_copybooks:\s*([^\n]+)", text)
    if not match:
        return []
    return _unique_preserving_order(
        item.strip()
        for item in re.findall(r"(?:^|,\s*)([A-Z0-9$#@-]+(?: \[[^\]]+\])?):", match.group(1))
    )


def _unique_preserving_order(items) -> list[str]:
    seen = set()
    result = []
    for item in items:
        if not item or item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def _unique_static_value_lines(lines: list[str]) -> list[str]:
    by_name: dict[str, str] = {}
    order: list[str] = []
    for line in lines:
        name = _static_value_name(line)
        if name not in by_name:
            order.append(name)
            by_name[name] = line
            continue
        existing = by_name[name].lower()
        current = line.lower()
        if "category:" not in existing and "category:" in current:
            by_name[name] = line
    return [by_name[name] for name in order]


def _static_value_name(line: str) -> str:
    clean = line.removeprefix("- ").strip()
    if ":" not in clean:
        return clean
    return clean.split(":", 1)[0].strip()


def _first_nonempty_lines(text: str, limit: int) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[:limit]


def _summary_lines(text: str, limit: int) -> list[str]:
    lines = []
    for line in text.splitlines():
        clean = line.strip()
        if not clean:
            continue
        if clean.lower().startswith("structured facts from source json"):
            break
        lines.append(clean)
        if len(lines) >= limit:
            break
    return lines


def _clean_external_call_line(line: str) -> str:
    clean = line.replace(", target_source literal", "")
    clean = clean.replace(" target_source literal", "")
    return clean
