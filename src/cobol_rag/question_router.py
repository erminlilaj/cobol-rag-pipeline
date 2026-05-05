from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cobol_rag.final_scripts_answers import find_final_scripts_root, program_from_question


@dataclass(frozen=True)
class EntityIndex:
    root: Path
    program: str | None
    programs: set[str]
    variables: set[str]
    paragraphs: set[str]
    copybooks: set[str]
    call_targets: set[str]
    db2_tables: set[str]
    sql_includes: set[str]
    screen_fields: set[str]
    evidence_terms: set[str]


def preflight_entity_answer(question: str) -> str | None:
    """Validate explicit COBOL entity references before retrieval/LLM generation.

    This is intentionally conservative: it only answers when the user names a
    concrete program/variable/paragraph/copybook/call target and the indexed
    final_scripts artifacts can prove whether that entity exists.
    """

    root = find_final_scripts_root()
    if root is None:
        return None

    programs = _core_programs_from_root(root)
    program_candidate = program_from_question(question)
    program = program_candidate if program_candidate in programs else _default_program(programs)
    index = _build_entity_index(root, program, programs)

    if program_candidate and program_candidate not in programs:
        answer = _answer_non_primary_program_reference(index, question, program_candidate)
        if answer:
            return answer

    tokens = _entity_tokens(question)
    if not tokens or not index.program:
        return None

    unknown_answer = _answer_unknown_named_entity(index, question, tokens)
    if unknown_answer:
        return unknown_answer

    return None


def _answer_non_primary_program_reference(
    index: EntityIndex,
    question: str,
    candidate: str,
) -> str | None:
    if not _looks_like_explicit_program_reference(question, candidate):
        return None

    secondary_lines = _secondary_entity_lines(index, candidate)
    if secondary_lines:
        lines = [
            f"`{candidate}` is not indexed as a standalone program in this RAG workspace.",
            f"Indexed program(s): {_join_or_none(sorted(index.programs))}.",
            f"But `{candidate}` does appear in the indexed evidence for `{index.program}`:",
            *secondary_lines,
            f"I can answer how `{index.program}` uses `{candidate}`, but to explain `{candidate}` internally you need to generate and index that program too.",
        ]
        return "\n".join(lines)

    if candidate in index.evidence_terms:
        return None

    lines = [f"I do not have indexed analysis for `{candidate}`."]
    if index.programs:
        lines.append(f"Indexed program(s) currently available: {_join_or_none(sorted(index.programs))}.")
    close = _closest(candidate, sorted(index.programs))
    if close:
        lines.append(f"Closest indexed program: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("Generate and index that program's analysis artifacts first, then ask again.")
    return "\n".join(lines)


def _answer_unknown_named_entity(index: EntityIndex, question: str, tokens: list[str]) -> str | None:
    known = (
        index.programs
        | index.variables
        | index.paragraphs
        | index.copybooks
        | index.call_targets
        | index.db2_tables
        | index.sql_includes
        | index.screen_fields
        | index.evidence_terms
    )
    q = question.lower()

    for token in tokens:
        if token in known or token in _IGNORED_ENTITY_TOKENS:
            continue
        if _looks_like_unknown_variable_question(q, token):
            return _unknown_variable_answer(index, token)
        if _looks_like_unknown_paragraph_question(q, token):
            return _unknown_paragraph_answer(index, token)
        if _looks_like_unknown_copybook_question(q, token):
            return _unknown_copybook_answer(index, token)
        if _looks_like_unknown_program_question(q, token):
            return _answer_non_primary_program_reference(index, question, token)
    return None


def _unknown_variable_answer(index: EntityIndex, variable: str) -> str:
    lines = [f"I do not have indexed dataflow evidence for variable `{variable}` in `{index.program}`."]
    close = _closest(variable, sorted(index.variables))
    if close:
        lines.append(f"Closest indexed variable: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("I will not infer a meaning from the name alone; regenerate/check the analysis if this variable should exist.")
    return "\n".join(lines)


def _unknown_paragraph_answer(index: EntityIndex, paragraph: str) -> str:
    lines = [f"I do not have indexed control-flow evidence for paragraph `{paragraph}` in `{index.program}`."]
    close = _closest(paragraph, sorted(index.paragraphs))
    if close:
        lines.append(f"Closest indexed paragraph: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("Known paragraph evidence comes from `controlflow.cfg` and related final_scripts artifacts.")
    return "\n".join(lines)


def _unknown_copybook_answer(index: EntityIndex, copybook: str) -> str:
    lines = [f"I do not have indexed COPY/member evidence for `{copybook}` in `{index.program}`."]
    close = _closest(copybook, sorted(index.copybooks))
    if close:
        lines.append(f"Closest indexed COPY member: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("Known COPY evidence comes from `architecture.copybooks`.")
    return "\n".join(lines)


def _secondary_entity_lines(index: EntityIndex, name: str) -> list[str]:
    lines: list[str] = []
    name = name.upper()
    calls = _calls(index.root, index.program)
    matching_calls = [call for call in calls if str(call.get("target", "")).upper() == name]
    for call in matching_calls[:3]:
        detail = [
            f"- call target: `{call.get('target')}` via {call.get('call_type', '?')}",
            f"in `{call.get('paragraph', '?')}` line {call.get('line_start', '?')}",
        ]
        params = call.get("parameters", [])
        if params:
            detail.append(f"parameters: {', '.join(str(item) for item in params)}")
        if call.get("commarea"):
            detail.append(f"COMMAREA={call.get('commarea')}")
        if call.get("length"):
            detail.append(f"LENGTH={call.get('length')}")
        lines.append("; ".join(detail) + ".")
    if name in index.copybooks:
        lines.append(f"- COPY member: `{name}` is listed in `architecture.copybooks`.")
    if name in index.db2_tables:
        lines.append(f"- DB2 table: `{name}` is listed in DB2 table artifacts.")
    if name in index.sql_includes:
        lines.append(f"- SQL include: `{name}` is listed in SQL include artifacts.")
    return lines


def _build_entity_index(root: Path, program: str | None, programs: set[str]) -> EntityIndex:
    return EntityIndex(
        root=root,
        program=program,
        programs=programs,
        variables=_variables(root, program),
        paragraphs=_paragraphs(root, program),
        copybooks=_copybooks(root, program),
        call_targets=_call_targets(root, program),
        db2_tables=_db2_tables(root, program),
        sql_includes=_sql_includes(root, program),
        screen_fields=_screen_fields(root, program),
        evidence_terms=_evidence_terms(root),
    )


def _core_programs_from_root(root: Path) -> set[str]:
    programs: set[str] = set()
    for relative in (
        "program_summary/program.summary.json",
        "program.comments/program.comments.json",
        "architecture.copybooks/architecture.copybooks.json",
        "architecture.call_parameters/architecture.call_parameters.json",
        "dataflow.literal_assignments/dataflow.literal_assignments.json",
        "dataflow.used_variables/dataflow.used_variables.json",
    ):
        payload = _read_json(root / relative)
        if isinstance(payload, dict):
            program = str(payload.get("program", "")).strip().upper()
            if program and program != "__GLOBAL__":
                programs.add(program)
    return programs


def _default_program(programs: set[str]) -> str | None:
    if len(programs) == 1:
        return next(iter(programs))
    return sorted(programs)[0] if programs else None


def _variables(root: Path, program: str | None) -> set[str]:
    if not program:
        return set()
    variables: set[str] = set()
    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program:
            variable = str(payload.get("content", {}).get("variable", "")).upper()
            if variable:
                variables.add(variable)
    used = _read_json(root / "dataflow.used_variables" / "dataflow.used_variables.json")
    if isinstance(used, dict) and str(used.get("program", "")).upper() == program:
        for item in used.get("variables", []):
            variable = str(item.get("variable", "")).upper()
            if variable:
                variables.add(variable)
    return variables


def _paragraphs(root: Path, program: str | None) -> set[str]:
    if not program:
        return set()
    names: set[str] = set()
    cfg = _read_json(root / "controlflow.cfg" / "controlflow.cfg.json")
    for edge in _cfg_edges(cfg):
        for key in ("from", "to"):
            value = str(edge.get(key, "")).upper()
            if value and value != program:
                names.add(value)
    literals = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if isinstance(literals, dict) and str(literals.get("program", "")).upper() == program:
        for item in literals.get("assignments", []):
            paragraph = str(item.get("paragraph", "")).upper()
            if paragraph:
                names.add(paragraph)
    for call in _calls(root, program):
        paragraph = str(call.get("paragraph", "")).upper()
        if paragraph:
            names.add(paragraph)
    return names


def _copybooks(root: Path, program: str | None) -> set[str]:
    payload = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != str(program or "").upper():
        return set()
    return {str(item).upper() for item in payload.get("content", {}).get("all", []) if str(item).strip()}


def _call_targets(root: Path, program: str | None) -> set[str]:
    return {str(call.get("target", "")).upper() for call in _calls(root, program) if call.get("target")}


def _calls(root: Path, program: str | None) -> list[dict[str, Any]]:
    payload = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != str(program or "").upper():
        return []
    return [call for call in payload.get("calls", []) if isinstance(call, dict)]


def _db2_tables(root: Path, program: str | None) -> set[str]:
    values: set[str] = set()
    for path in (root / "architecture.db2_table").glob("*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != str(program or "").upper():
            continue
        content = payload.get("content", {})
        table = str(content.get("table") or content.get("name") or "").upper()
        if table:
            values.add(table)
    return values


def _sql_includes(root: Path, program: str | None) -> set[str]:
    values: set[str] = set()
    for path in (root / "architecture.sqlinclude").glob("*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != str(program or "").upper():
            continue
        content = payload.get("content", {})
        include = str(content.get("include") or content.get("name") or "").upper()
        if include:
            values.add(include)
    return values


def _screen_fields(root: Path, program: str | None) -> set[str]:
    payload = _read_json(root / "screen_field_lineage" / "screen_field_lineage.json")
    values: set[str] = set()
    if isinstance(payload, dict) and str(payload.get("program", "")).upper() == str(program or "").upper():
        for item in payload.get("content", {}).get("fields", []):
            field = str(item.get("field", "")).upper()
            if field:
                values.add(field)
    return values


def _evidence_terms(root: Path) -> set[str]:
    terms: set[str] = set()
    for path in root.glob("**/*.json"):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", text):
            upper = token.upper()
            if upper not in _IGNORED_ENTITY_TOKENS:
                terms.add(upper)
    return terms


def _cfg_edges(payload: Any) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        return []
    edges = payload.get("edges", [])
    if isinstance(edges, list):
        return [edge for edge in edges if isinstance(edge, dict)]
    graph = payload.get("graph", {})
    if isinstance(graph, dict) and isinstance(graph.get("edges"), list):
        return [edge for edge in graph.get("edges", []) if isinstance(edge, dict)]
    return []


def _entity_tokens(question: str) -> list[str]:
    tokens = [
        token.upper()
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}\b", question)
        if token.upper() not in _IGNORED_ENTITY_TOKENS
    ]
    return list(dict.fromkeys(tokens))


def _looks_like_explicit_program_reference(question: str, candidate: str) -> bool:
    q = question.lower()
    c = candidate.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", q).strip()
    if normalized == c:
        return True
    if f"{c}.cbl" in q:
        return True
    markers = (f"file {c}", f"program {c}", f"code {c}", f"{c} file", f"{c} program")
    if any(marker in q for marker in markers):
        return True
    return c.startswith(("pd", "px", "pr")) and any(term in q for term in ("what does", "explain", "summarize"))


def _looks_like_unknown_variable_question(q: str, token: str) -> bool:
    if not _looks_like_variable_name(token):
        return False
    return any(
        term in q
        for term in (
            "what does",
            "what is",
            "what stores",
            "store",
            "stores",
            "variable",
            "field",
            "used",
            "modified",
            "read",
            "written",
            "set",
            "origin",
            "value",
            "calculated",
            "computed",
        )
    )


def _looks_like_unknown_paragraph_question(q: str, token: str) -> bool:
    if not _looks_like_paragraph_name(token):
        return False
    return any(term in q for term in ("paragraph", "section", "flow", "logic", "what does", "explain"))


def _looks_like_unknown_copybook_question(q: str, token: str) -> bool:
    if not token.startswith(("DFH", "PD", "PX", "PR")):
        return False
    return any(term in q for term in ("copy", "copybook", "copy member"))


def _looks_like_unknown_program_question(q: str, token: str) -> bool:
    if not token.startswith(("PD", "PX", "PR")):
        return False
    return any(term in q for term in ("file", "program", "code", "what does", "explain", "summarize"))


def _looks_like_variable_name(token: str) -> bool:
    if "-" in token:
        return True
    return token.startswith(("W", "PD", "TWCOB", "PX", "SQL", "FUNZ", "M1", "SCELTA"))


def _looks_like_paragraph_name(token: str) -> bool:
    return "-" in token and not _looks_like_variable_name(token)


def _closest(candidate: str, options: list[str]) -> str | None:
    if not options:
        return None
    distance, value = min((_edit_distance(candidate, option), option) for option in options)
    return value if distance <= max(2, len(candidate) // 4) else None


def _edit_distance(left: str, right: str) -> int:
    if left == right:
        return 0
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def _join_or_none(values: list[str]) -> str:
    return ", ".join(values) if values else "none"


def _read_json(path: Path) -> Any | None:
    candidates = [path]
    if len(path.parents) >= 2:
        flat_candidate = path.parent.parent / path.name
        if flat_candidate not in candidates:
            candidates.append(flat_candidate)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            import json

            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
    return None


_IGNORED_ENTITY_TOKENS = {
    "ABOUT",
    "AND",
    "ANY",
    "ARE",
    "BEFORE",
    "CALCULATED",
    "CBL",
    "CICS",
    "CODE",
    "COBOL",
    "CONTROL-FLOW",
    "COPY",
    "COPYBOOK",
    "DATA",
    "DB2",
    "DOES",
    "FILE",
    "FIELD",
    "FIELDS",
    "FOR",
    "FROM",
    "HOW",
    "IDENTIFY",
    "INSIDE",
    "INTERACTIONS",
    "INTO",
    "JCL",
    "LEAD",
    "LINE",
    "LINES",
    "MAINTAINED",
    "MAP",
    "MAPA",
    "MESSAGE",
    "PATH",
    "PATHS",
    "PRESSED",
    "PRESSES",
    "PRESSING",
    "PROGRAM",
    "QUESTION",
    "SQL",
    "STORE",
    "STORES",
    "THE",
    "THIS",
    "USED",
    "VARIABLE",
    "VARIABLES",
    "WHEN",
    "WHAT",
    "WHETHER",
    "WHERE",
    "WHICH",
    "WITH",
    "WRITTEN",
}
