from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.final_scripts_answers import answer_from_final_scripts
from cobol_rag.index import configure_llamaindex, open_index
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
) -> QueryAnswer:
    current_question = _current_question(question)
    local_answer = _try_local_answer(current_question)
    if local_answer:
        return QueryAnswer(question=question, answer=local_answer, sources=[])

    final_scripts_answer = answer_from_final_scripts(current_question)
    if final_scripts_answer:
        return QueryAnswer(question=question, answer=final_scripts_answer, sources=[])

    metadata_answer = _try_program_metadata_answer(current_question)
    if metadata_answer:
        return QueryAnswer(question=question, answer=metadata_answer, sources=[])

    if not _is_cobol_question(current_question):
        return QueryAnswer(
            question=question,
            answer=_answer_general_question(question, current_question, config),
            sources=[],
        )

    try:
        sources = retrieve(question, config=config, top_k=top_k)
    except Exception:
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
            )
            sources = _merge_sources(targeted_sources, sources)
        except Exception:
            pass
    if not sources:
        return QueryAnswer(
            question=question,
            answer="I could not find relevant indexed sources for this question.",
            sources=[],
        )

    direct_answer = _try_dead_code_answer(current_question, sources)
    if direct_answer:
        return QueryAnswer(question=question, answer=direct_answer, sources=sources)

    resources = open_index(config)
    prompt = _build_prompt(question=question, sources=sources)
    try:
        response = resources.runtime.llm.complete(prompt)
    except Exception as error:
        return QueryAnswer(
            question=question,
            answer=_offline_fallback_answer(config.llm.model, sources),
            sources=sources,
        )
    return QueryAnswer(
        question=question,
        answer=str(response.text).strip(),
        sources=sources,
    )


def _current_question(question: str) -> str:
    marker = "Current question:"
    if marker in question:
        return question.rsplit(marker, 1)[-1].strip()
    return question.strip()


def _try_local_answer(question: str) -> str | None:
    text = question.strip().lower()
    normalized = text.strip(" ?.!").strip()
    if text in {"hi", "hello", "hey", "ciao", "salve", "buongiorno", "good morning", "good afternoon"}:
        return (
            "Hi. I can help you inspect the indexed COBOL analysis. "
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
        "working-storage",
        "linkage",
    }
    return any(term in text for term in cobol_terms)


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


def _build_prompt(question: str, sources: list[RetrievalResult]) -> str:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        source_id = source.metadata.get("source_id", f"source-{index}")
        source_path = source.metadata.get("source_path", "")
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_id: {source_id}",
                    f"source_path: {source_path}",
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
- Keep the answer concise.
- Mention source ids inline when useful, but do not invent source ids.

Question:
{question}

Retrieved sources:
{context}

Answer:
"""
