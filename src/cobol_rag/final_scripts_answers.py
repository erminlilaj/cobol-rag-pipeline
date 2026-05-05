from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from cobol_rag.final_scripts_artifacts import (
    load_or_build_jcl_file_io,
    load_or_build_quality_dead_code,
    load_or_build_screen_field_lineage,
    load_or_build_unused_copybooks,
    program_has_jcl_evidence,
)


def answer_from_final_scripts(question: str) -> str | None:
    root = find_final_scripts_root()
    if root is None:
        return None

    unknown_program_answer = _answer_unknown_program_if_explicit(root, question)
    if unknown_program_answer:
        return unknown_program_answer

    program = _program_for_question(root, question)
    if program is None:
        return None

    q = question.lower()

    if _asks_about_copybooks(q) and "unused" in q:
        answer = _answer_copybooks(root, program, q)
        if answer:
            return answer

    if _asks_about_lines_or_counts(q):
        answer = _answer_counts(root, program, q)
        if answer:
            return answer

    if _asks_about_dead_or_commented_code(q):
        answer = _answer_commented_code(root, program, q)
        if answer:
            return answer

    if _asks_about_screen_field_lineage(q):
        answer = _answer_screen_field_lineage(root, program, question)
        if answer:
            return answer

    answer = _answer_structured_behavior(root, program, question)
    if answer:
        return answer

    variable_answer = _answer_variable_reference(root, program, question)
    if variable_answer:
        return variable_answer

    if _asks_about_calls(q):
        answer = _answer_calls(root, program)
        if answer:
            return answer

    if _asks_about_forced_values(q):
        answer = _answer_literal_assignments(root, program, q)
        if answer:
            return answer

    if _asks_about_copybooks(q):
        answer = _answer_copybooks(root, program, q)
        if answer:
            return answer

    if _asks_about_db2_or_sql(q):
        answer = _answer_db2_sql(root, program)
        if answer:
            return answer

    if _asks_about_datasets(q):
        return _answer_datasets(root, program)

    if _asks_about_ui_navigation(q):
        answer = _answer_ui_navigation(root, program)
        if answer:
            return answer

    if _asks_about_business_rules(q):
        answer = _answer_business_rules(root, program)
        if answer:
            return answer

    if _asks_about_program_overview(q, question, program):
        answer = _answer_program_overview(root, program)
        if answer:
            return answer

    return None


def find_final_scripts_root() -> Path | None:
    configured = os.environ.get("COBOL_RAG_FINAL_SCRIPTS_DIR")
    if configured:
        path = Path(configured).expanduser().resolve()
        if path.exists():
            return path

    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidates = [
            base / "final_scripts",
            base / "data" / "final_scripts",
            base / "control_flow" / "artifacts" / "final" / "final_scripts",
            base.parent / "control_flow" / "artifacts" / "final" / "final_scripts",
            base / "artifacts" / "final" / "final_scripts",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def program_from_question(question: str) -> str | None:
    ignored = {
        "ABOUT",
        "ANY",
        "BUSINESS",
        "CALL",
        "CALLS",
        "CODE",
        "COMMENTED",
        "COPY",
        "COPYBOOK",
        "COPYBOOKS",
        "COUNT",
        "DATASET",
        "DATASETS",
        "DEAD",
        "FILE",
        "FILES",
        "FORCED",
        "HOW",
        "LINE",
        "LINES",
        "MANY",
        "NUMBER",
        "OUT",
        "PARAMETER",
        "PARAMETERS",
        "PROGRAM",
        "PROGRAMS",
        "PRODUCE",
        "PRODUCED",
        "PRODUCES",
        "RULES",
        "TABLE",
        "TABLES",
        "THIS",
        "UNUSED",
        "USE",
        "USED",
        "USES",
        "VALUE",
        "VALUES",
        "WITH",
        "WHAT",
        "WHICH",
    }
    candidates = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", question.upper())
        if token not in ignored
    ]
    if not candidates:
        return None
    return max(candidates, key=len)


def _program_for_question(root: Path, question: str) -> str | None:
    candidate = program_from_question(question)
    core_programs = _core_programs_from_root(root)
    if candidate and candidate in core_programs:
        return candidate
    if candidate and program_has_jcl_evidence(root, candidate):
        return candidate
    return _primary_program_from_root(root) or candidate


def _answer_unknown_program_if_explicit(root: Path, question: str) -> str | None:
    candidate = program_from_question(question)
    if not candidate:
        return None
    if not _looks_like_explicit_program_reference(question, candidate):
        return None
    core_programs = _core_programs_from_root(root)
    if candidate in core_programs or program_has_jcl_evidence(root, candidate):
        return None
    if _token_exists_in_final_scripts(root, candidate):
        return None

    indexed = sorted(core_programs)
    close = _closest_program(candidate, indexed)
    lines = [f"I do not have indexed analysis for `{candidate}`."]
    if indexed:
        lines.append(f"Indexed program(s) currently available: {', '.join(indexed)}.")
    if close:
        lines.append(f"Closest indexed name: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("Generate and index that program's analysis artifacts first, then ask again.")
    return "\n".join(lines)


def _looks_like_explicit_program_reference(question: str, candidate: str) -> bool:
    q = question.lower()
    c = candidate.lower()
    normalized = re.sub(r"[^a-z0-9]+", " ", q).strip()
    if normalized == c:
        return True
    if f"{c}.cbl" in q:
        return True
    if any(marker in q for marker in (f"file {c}", f"program {c}", f"code {c}", f"{c} file", f"{c} program")):
        return True
    return c.startswith(("pd", "px", "pr")) and any(term in q for term in ("what does", "explain", "summarize"))


def _closest_program(candidate: str, programs: list[str]) -> str | None:
    if not programs:
        return None
    distances = sorted((_edit_distance(candidate, program), program) for program in programs)
    distance, program = distances[0]
    return program if distance <= 2 else None


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


def _default_program_from_root(root: Path) -> str | None:
    primary = _primary_program_from_root(root)
    if primary:
        return primary
    programs = _programs_from_root(root)
    if len(programs) == 1:
        return next(iter(programs))
    return None


def _primary_program_from_root(root: Path) -> str | None:
    programs = _core_programs_from_root(root)
    if not programs:
        return None
    return sorted(programs)[0]


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


def _programs_from_root(root: Path) -> set[str]:
    programs: set[str] = set()
    for relative in (
        "program_summary/program.summary.json",
        "program.comments/program.comments.json",
        "architecture.copybooks/architecture.copybooks.json",
        "architecture.call_parameters/architecture.call_parameters.json",
    ):
        payload = _read_json(root / relative)
        if isinstance(payload, dict):
            program = str(payload.get("program", "")).strip().upper()
            if program and program != "__GLOBAL__":
                programs.add(program)
    for path in root.glob("**/*.json"):
        payload = _read_json(path)
        if isinstance(payload, dict):
            program = str(payload.get("program", "")).strip().upper()
            if program and program != "__GLOBAL__":
                programs.add(program)
    return programs


def _token_exists_in_final_scripts(root: Path, token: str) -> bool:
    token = token.upper()
    if not token:
        return False
    pattern = re.compile(rf"\b{re.escape(token)}\b", flags=re.IGNORECASE)
    for path in root.glob("**/*.json"):
        try:
            if pattern.search(path.read_text(encoding="utf-8", errors="ignore")):
                return True
        except OSError:
            continue
    return False


def _asks_about_lines_or_counts(q: str) -> bool:
    return any(term in q for term in ("how many", "number of", "count", "loc", "lines"))


def _asks_about_dead_or_commented_code(q: str) -> bool:
    return any(term in q for term in ("unused code", "dead code", "unreachable", "commented-out", "commented out", "commented code"))


def _asks_about_calls(q: str) -> bool:
    return any(term in q for term in ("call", "calls", "outside program", "external program", "commarea", "parameter"))


def _asks_about_forced_values(q: str) -> bool:
    return any(term in q for term in ("forced value", "forced values", "literal", "hardcoded", "hard-coded", "static value"))


def _asks_about_copybooks(q: str) -> bool:
    return any(term in q for term in ("copybook", "copy book", "copy member", "copy members", "unused copy"))


def _asks_about_db2_or_sql(q: str) -> bool:
    return any(term in q for term in ("db2", "sql", "table", "tables", "sqlinclude", "sql include"))


def _asks_about_datasets(q: str) -> bool:
    return any(term in q for term in ("dataset", "datasets", "file io", "file i/o", "produce", "produces", "output file"))


def _asks_about_ui_navigation(q: str) -> bool:
    return any(term in q for term in ("pf key", "pfkey", "screen", "map", "navigation", "cics key", "eibaid"))


def _asks_about_screen_field_lineage(q: str) -> bool:
    has_screen_term = any(term in q for term in ("screen", "map", "field"))
    has_lineage_term = any(
        term in q
        for term in (
            "connected",
            "connection",
            "variable",
            "variables",
            "origin",
            "data origin",
            "computation",
            "computed",
            "calculated",
            "modified",
            "defined",
            "feeds",
        )
    )
    return has_screen_term and has_lineage_term


def _asks_about_business_rules(q: str) -> bool:
    return any(term in q for term in ("business rule", "business rules", "rules", "condition", "conditions"))


def _read_json(path: Path) -> Any | None:
    candidates = [path]
    if len(path.parents) >= 2:
        # Support both historical final_scripts layout:
        #   root/program_summary/program.summary.json
        # and generated pipeline layout:
        #   root/program.summary.json
        flat_candidate = path.parent.parent / path.name
        if flat_candidate not in candidates:
            candidates.append(flat_candidate)
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _asks_about_program_overview(q: str, question: str, program: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]+", " ", q).strip()
    if normalized == program.lower():
        return True
    if program.lower() not in q:
        return False
    return any(
        term in q
        for term in (
            "explain the code",
            "explain code",
            "explain the program",
            "describe the program",
            "summarize",
            "summary",
            "overview",
            "purpose",
            f"what is {program.lower()}",
            f"what does {program.lower()} do",
            f"what does the file {program.lower()} do",
            f"what does the program {program.lower()} do",
        )
    )


def _answer_program_overview(root: Path, program: str) -> str | None:
    summary = _summary_payload(root, program)
    if not summary:
        return None

    meta = summary.get("meta", {}) if isinstance(summary.get("meta"), dict) else {}
    loc = meta.get("loc") or _extract_approx_loc(summary)
    paragraphs = meta.get("paragraphs") or _extract_paragraph_count(summary)
    statements = meta.get("statements")

    lines = [f"{program} overview from the indexed analysis:"]
    if summary.get("content"):
        lines.append(f"- {summary.get('content')}")
    metric_parts = []
    if loc is not None:
        metric_parts.append(f"{loc} LOC")
    if statements is not None:
        metric_parts.append(f"{statements} statements")
    if paragraphs is not None:
        metric_parts.append(f"{paragraphs} paragraphs")
    if metric_parts:
        lines.append(f"- Size: {', '.join(str(part) for part in metric_parts)}.")

    calls = _calls(root, program)
    if calls:
        call_bits = []
        for call in calls[:8]:
            target = call.get("target", "?")
            call_type = call.get("call_type", "?")
            paragraph = call.get("paragraph", "?")
            call_bits.append(f"{target} ({call_type} in {paragraph})")
        lines.append(f"- External interactions: {len(calls)} outgoing call(s): {', '.join(call_bits)}.")

    copybooks = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    if isinstance(copybooks, dict) and copybooks.get("program") == program:
        all_copybooks = copybooks.get("content", {}).get("all", [])
        if all_copybooks:
            lines.append(f"- COPY usage: {len(all_copybooks)} COPY member(s): {', '.join(all_copybooks)}.")

    comments = _comments_payload(root, program)
    if comments:
        comment_count = comments.get("count")
        commented_out = comments.get("classification_counts", {}).get("commented_out_code")
        comment_bits = []
        if comment_count is not None:
            comment_bits.append(f"{comment_count} comment lines")
        if commented_out is not None:
            comment_bits.append(f"{commented_out} commented-out code/data item(s)")
        if comment_bits:
            lines.append(f"- Comment evidence: {', '.join(comment_bits)}.")

    _append_evidence(
        lines,
        [
            _cite("program_summary/program.summary.json", detail="program overview"),
            _cite("architecture.call_parameters/architecture.call_parameters.json", detail="outgoing calls"),
            _cite("architecture.copybooks/architecture.copybooks.json", detail="COPY usage"),
            _cite("program.comments/program.comments.json", detail="comment metrics"),
        ],
    )
    return "\n".join(lines)


def _cfg_edges(root: Path, program: str) -> list[dict[str, Any]]:
    payload = _read_json(root / "controlflow.cfg" / "controlflow.cfg.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return []
    return [edge for edge in payload.get("edges", []) if isinstance(edge, dict)]


def _literal_items(root: Path, program: str, paragraph: str | None = None) -> list[dict[str, Any]]:
    payload = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return []
    items = [item for item in payload.get("assignments", []) if isinstance(item, dict)]
    if paragraph:
        items = [item for item in items if item.get("paragraph") == paragraph]
    return items


def _call_by_target(root: Path, program: str, target: str) -> dict[str, Any] | None:
    for call in _calls(root, program):
        if str(call.get("target", "")).upper() == target.upper():
            return call
    return None


def _calls(root: Path, program: str) -> list[dict[str, Any]]:
    payload = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return []
    return [call for call in payload.get("calls", []) if isinstance(call, dict)]


def _variable_payload(root: Path, program: str, variable: str) -> dict[str, Any] | None:
    path = root / "dataflow.variable" / f"dataflow.variable.{variable.upper()}.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and payload.get("program") == program:
        return payload
    return None


def _site_lines(
    payload: dict[str, Any],
    site_key: str,
    *,
    limit: int = 6,
    indent: str = "",
) -> list[str]:
    sites = payload.get("content", {}).get("evidence", {}).get(site_key, [])
    lines: list[str] = []
    for site in sites[:limit]:
        line = site.get("line_start", "?")
        paragraph = site.get("paragraph", "?")
        statement = str(site.get("statement", "")).strip()
        citation = _site_citation(payload, line)
        suffix = f" [{citation}]" if citation else ""
        lines.append(f"{indent}- line {line} `{paragraph}`: {statement}{suffix}")
    if not lines:
        lines.append(f"{indent}- none")
    return lines


def _site_citation(payload: dict[str, Any], line: Any) -> str:
    variable = str(payload.get("content", {}).get("variable", "")).upper()
    if variable:
        return _cite(f"dataflow.variable/dataflow.variable.{variable}.json", line=line)
    return ""


def _cite(path: str, *, line: Any | None = None, detail: str | None = None) -> str:
    parts = [path]
    if line not in (None, "", -1, "?"):
        parts.append(f"line {line}")
    if detail:
        parts.append(detail)
    return " | ".join(parts)


def _append_evidence(lines: list[str], evidence: list[str], *, limit: int = 8) -> None:
    unique = []
    seen = set()
    for item in evidence:
        if not item or item in seen:
            continue
        seen.add(item)
        unique.append(item)
    if not unique:
        return
    lines.append("")
    lines.append("Evidence:")
    for item in unique[:limit]:
        lines.append(f"- {item}")


def _line_from_site_list(payload: dict[str, Any], key: str) -> Any | None:
    for site in payload.get("content", {}).get("evidence", {}).get(key, []):
        line = site.get("line_start")
        if isinstance(line, int) and line > 0:
            return line
    return None


def _variables_in_question(question: str) -> list[str]:
    ignored = {
        "PDCBVC", "COBOL", "CBL", "BROWSE-FASE1", "BROWSE-FASE2", "ENTER",
        "PF1", "PF2", "PF3", "PF4", "PF7", "PF8", "PF9", "CONTROL-FLOW", "WHEN", "WHETHER", "WHICH",
        "WHERE", "WHAT", "WITH", "WRITTEN", "PRESSES", "PRESSED", "PRESSING",
        "USER", "INTERACTIONS", "MAINTAINED", "CALCULATED",
    }
    names: list[str] = []
    for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", question.upper()):
        if token not in ignored and ("-" in token or token.startswith(("W", "PD", "TWCOB", "PX", "SQL"))):
            names.append(token)
    return list(dict.fromkeys(names))


def _known_variables(root: Path, program: str) -> set[str]:
    dataflow_dir = root / "dataflow.variable"
    return {
        path.name.removeprefix("dataflow.variable.").removesuffix(".json").upper()
        for path in dataflow_dir.glob("dataflow.variable.*.json")
        if path.is_file()
    }


def _variable_reference_tokens(question: str) -> list[str]:
    ignored = {
        "ABOUT",
        "COBOL",
        "CODE",
        "DOES",
        "FILE",
        "FUNCTION",
        "INTERACTIONS",
        "MAINTAINED",
        "METHOD",
        "PRESSED",
        "PRESSES",
        "PRESSING",
        "PROGRAM",
        "REPOSITORY",
        "STORES",
        "STORE",
        "USER",
        "WHAT",
        "WHEN",
        "WHETHER",
        "WHERE",
        "WHICH",
        "WRITTEN",
    }
    tokens = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", question.upper())
        if token not in ignored
    ]
    return list(dict.fromkeys(tokens))


def _looks_like_variable_reference_question(q: str) -> bool:
    return any(
        term in q
        for term in (
            "what does",
            "what is",
            "what stores",
            "stores",
            "store",
            "variable",
            "field",
            "calculated",
            "computed",
            "feed",
            "feeds",
            "set",
            "used",
            "modified",
            "read",
            "written",
            "origin",
            "value",
        )
    )


def _answer_variable_reference(root: Path, program: str, question: str) -> str | None:
    q = question.lower()
    if not _looks_like_variable_reference_question(q):
        return None

    known = _known_variables(root, program)
    if not known:
        return None

    tokens = _variable_reference_tokens(question)
    variables = [_variable_payload(root, program, token) for token in tokens if token in known]
    variables = [variable for variable in variables if variable]
    if variables:
        lines = [f"{program} variable evidence:"]
        for variable in variables[:4]:
            lines.append(_format_variable_lineage(program, variable))
        return "\n\n".join(lines)

    unknowns = [
        token
        for token in tokens
        if token not in {program, "PDCBVC", "COBOL", "CBL"} and _looks_like_variable_name(token)
    ]
    if not unknowns:
        return None
    unknown = unknowns[0]
    close = _closest_program(unknown, sorted(known))
    lines = [f"I do not have indexed dataflow evidence for variable `{unknown}` in `{program}`."]
    if close:
        lines.append(f"Closest indexed variable: `{close}`. Ask about `{close}` if that is what you meant.")
    lines.append("Check the variable name or regenerate the analysis artifacts if this variable should exist.")
    return "\n".join(lines)


def _looks_like_variable_name(token: str) -> bool:
    if "-" in token:
        return True
    return token.startswith(("W", "PD", "TWCOB", "PX", "SQL", "FUNZ", "M1", "SCELTA"))


def _paragraph_names(root: Path, program: str) -> set[str]:
    names: set[str] = set()
    for edge in _cfg_edges(root, program):
        for key in ("from", "to"):
            value = str(edge.get(key, "")).strip().upper()
            if value and value != program:
                names.add(value)
    for item in _literal_items(root, program):
        paragraph = str(item.get("paragraph", "")).strip().upper()
        if paragraph:
            names.add(paragraph)
    for call in _calls(root, program):
        paragraph = str(call.get("paragraph", "")).strip().upper()
        if paragraph:
            names.add(paragraph)
    return names


def _paragraphs_in_question(root: Path, program: str, question: str) -> list[str]:
    known = _paragraph_names(root, program)
    found = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", question.upper())
        if token in known
    ]
    return list(dict.fromkeys(found))


def _call_target_in_question(root: Path, program: str, question: str) -> str | None:
    targets = {str(call.get("target", "")).upper() for call in _calls(root, program)}
    tokens = set(re.findall(r"\b[A-Z][A-Z0-9]{3,}\b", question.upper()))
    matches = sorted(targets & tokens, key=len, reverse=True)
    return matches[0] if matches else None


def _comments_payload(root: Path, program: str) -> dict[str, Any] | None:
    payload = _read_json(root / "program.comments" / "program.comments.json")
    if isinstance(payload, dict) and payload.get("program") == program:
        return payload
    return None


def _summary_payload(root: Path, program: str) -> dict[str, Any] | None:
    payload = _read_json(root / "program_summary" / "program.summary.json")
    if isinstance(payload, dict) and payload.get("program") == program:
        return payload
    return None


def _answer_counts(root: Path, program: str, q: str) -> str | None:
    comments = _comments_payload(root, program)
    summary = _summary_payload(root, program)
    copybooks = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    calls = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    literals = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")

    if "copy" in q and isinstance(copybooks, dict):
        all_copybooks = copybooks.get("content", {}).get("all", [])
        lines = [f"{program} has {len(all_copybooks)} COPY members listed: {', '.join(all_copybooks)}."]
        _append_evidence(lines, [_cite("architecture.copybooks/architecture.copybooks.json", detail="content.all")])
        return "\n".join(lines)

    if ("call" in q or "external" in q or "outside" in q) and isinstance(calls, dict):
        call_items = calls.get("calls", [])
        lines = [f"{program} has {len(call_items)} outgoing calls in `architecture.call_parameters.json`."]
        _append_evidence(lines, [_cite("architecture.call_parameters/architecture.call_parameters.json", detail="calls")])
        return "\n".join(lines)

    if ("literal" in q or "forced" in q or "hardcoded" in q) and isinstance(literals, dict):
        items = literals.get("assignments", [])
        lines = [f"{program} has {len(items)} literal assignments in `dataflow.literal_assignments.json`."]
        _append_evidence(lines, [_cite("dataflow.literal_assignments/dataflow.literal_assignments.json", detail="assignments")])
        return "\n".join(lines)

    if comments and any(term in q for term in ("line", "lines", "loc", "code")):
        total_lines = comments.get("metrics", {}).get("total_lines")
        comment_count = comments.get("count")
        commented_out = comments.get("classification_counts", {}).get("commented_out_code")
        approx_loc = _extract_approx_loc(summary)
        paragraphs = _extract_paragraph_count(summary)
        parts: list[str] = []
        if total_lines is not None:
            parts.append(f"{program} has {total_lines} total physical source lines.")
        if approx_loc is not None:
            parts.append(f"`program.summary.json` estimates about {approx_loc} LOC.")
        if paragraphs is not None:
            parts.append(f"It reports about {paragraphs} paragraphs.")
        if comment_count is not None:
            parts.append(f"`program.comments.json` reports {comment_count} comment lines.")
        if commented_out is not None:
            parts.append(f"{commented_out} comments are classified as commented-out code.")
        if parts:
            lines = [" ".join(parts)]
            _append_evidence(
                lines,
                [
                    _cite("program.comments/program.comments.json", detail="metrics.total_lines/count/classification_counts"),
                    _cite("program_summary/program.summary.json", detail="content"),
                ],
            )
            return "\n".join(lines)

    return None


def _extract_approx_loc(summary: dict[str, Any] | None) -> int | None:
    if not summary:
        return None
    text = str(summary.get("content", ""))
    match = re.search(r"approximately\s+(\d+)\s+LOC", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _extract_paragraph_count(summary: dict[str, Any] | None) -> int | None:
    if not summary:
        return None
    text = str(summary.get("content", ""))
    match = re.search(r"(\d+)\s+paragraphs", text, flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _answer_commented_code(root: Path, program: str, q: str) -> str | None:
    artifact = load_or_build_quality_dead_code(root, program)
    content = artifact.get("content", {})
    commented = content.get("commented_out_code", [])
    if not isinstance(commented, list):
        commented = []
    reachability = content.get("cfg_reachability", {})
    if "dead code" in q or "unreachable" in q:
        lines = [
            f"{program} dead-code evidence from `quality.dead_code`:",
            f"- commented-out code/data: {len(commented)} item(s).",
            (
                f"- CFG reachability: {reachability.get('unreachable_nodes_count', 0)} unreachable "
                f"paragraph/node(s) among {reachability.get('nodes_count', 0)} CFG nodes."
            ),
            "- Limitation: static CFG reachability is not a runtime execution proof.",
        ]
    else:
        lines = [f"Commented-out code/data found in {program}: {len(commented)} item(s)."]
    for comment in commented[:20]:
        citation = comment.get("citation") or _cite("program.comments/program.comments.json", line=comment.get("line"))
        lines.append(f"- line {comment.get('line')}: {str(comment.get('text', '')).strip()} [{citation}]")
    if "copy" in q:
        copy_answer = _answer_copybooks(root, program, q)
        if copy_answer:
            lines.append("")
            lines.append(copy_answer)
    _append_evidence(
        lines,
        [
            _cite("quality.dead_code/quality.dead_code.json", detail="derived artifact"),
            _cite("program.comments/program.comments.json", detail="commented_out_code"),
            _cite("controlflow.cfg/controlflow.cfg.json", detail="CFG reachability"),
        ],
    )
    return "\n".join(lines)


def _answer_calls(root: Path, program: str) -> str | None:
    payload = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    calls = payload.get("calls", [])
    lines = [f"{program} outgoing calls with parameters:"]
    evidence: list[str] = []
    for call in calls:
        target = call.get("target", "?")
        call_type = call.get("call_type", "?")
        paragraph = call.get("paragraph", "?")
        line = call.get("line_start", "?")
        params = ", ".join(call.get("parameters", [])) or "no explicit parameter"
        details = [f"- {target}: {call_type} in {paragraph} line {line}; parameters: {params}"]
        if call.get("commarea"):
            details.append(f"COMMAREA={call.get('commarea')}")
        if call.get("length"):
            details.append(f"LENGTH={call.get('length')}")
        lines.append("; ".join(details) + ".")
        evidence.append(
            _cite(
                "architecture.call_parameters/architecture.call_parameters.json",
                line=call.get("line_start"),
                detail=str(target),
            )
        )
    _append_evidence(lines, evidence)
    return "\n".join(lines)


def _answer_literal_assignments(root: Path, program: str, q: str) -> str | None:
    payload = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    items = payload.get("assignments", [])
    if "commarea" in q or "parameter" in q:
        items = [item for item in items if item.get("call_commarea_field")]
    elif "screen" in q or "map" in q:
        items = [item for item in items if item.get("screen_or_map_field")]
    elif "control" in q or "flow" in q:
        items = [item for item in items if item.get("controls_flow")]
    lines = [f"{program} literal assignments: {len(items)} matching item(s)."]
    evidence: list[str] = []
    for item in items[:25]:
        tags = []
        if item.get("call_commarea_field"):
            tags.append("COMMAREA")
        if item.get("screen_or_map_field"):
            tags.append("screen/map")
        if item.get("controls_flow"):
            tags.append("controls flow")
        suffix = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"- line {item.get('line')} {item.get('paragraph')}: "
            f"{item.get('target_variable')} = {item.get('literal')}{suffix}"
        )
        evidence.append(
            _cite(
                "dataflow.literal_assignments/dataflow.literal_assignments.json",
                line=item.get("line"),
                detail=str(item.get("target_variable", "")),
            )
        )
    _append_evidence(lines, evidence)
    return "\n".join(lines)


def _answer_structured_behavior(root: Path, program: str, question: str) -> str | None:
    q = question.lower()
    if _asks_about_phase_decision(q):
        return _answer_phase_decision(root, program)
    if _asks_about_error_paths(q):
        return _answer_error_paths(root, program)
    if _asks_about_semaphore(q):
        return _answer_semaphore_flow(root, program)
    if _asks_about_browse_fase1_sequence(q):
        return _answer_paragraph_sequence(root, program, "BROWSE-FASE1", stop_at="SEND-PDCBVC1")
    call_target = _call_target_in_question(root, program, question)
    if call_target and _asks_about_call_details(q):
        return _answer_call_preparation(root, program, call_target)
    if _asks_about_variable_behavior(q) and _asks_for_specific_variable_trace(q):
        return _answer_variables_from_question(root, program, question)
    if _asks_about_pagination(q):
        return _answer_pagination(root, program)
    if _asks_about_row_selection(q):
        return _answer_row_selection(root, program)
    if _asks_about_pf_key_comparison(q):
        return _answer_pf_key_comparison(root, program)
    paragraphs = _paragraphs_in_question(root, program, question)
    if paragraphs and _asks_about_paragraph_behavior(q):
        return _answer_paragraph_behavior(root, program, paragraphs[0])
    key_answer = _answer_key_flow(root, program, question)
    if key_answer:
        return key_answer
    if _asks_about_variable_behavior(q):
        return _answer_variables_from_question(root, program, question)
    return None


def _asks_about_phase_decision(q: str) -> bool:
    return (
        "twcob-fase" in q
        and any(term in q for term in ("browse-fase1", "browse-fase2", "unexpected", "decide", "whether"))
    )


def _asks_about_semaphore(q: str) -> bool:
    return "semaphore" in q or "pdaggvip" in q or "pxcsemaf" in q


def _asks_about_browse_fase1_sequence(q: str) -> bool:
    return "browse-fase1" in q and any(term in q for term in ("sequence", "before", "map", "sent", "send"))


def _asks_about_call_preparation(q: str, target: str) -> bool:
    return target.lower() in q and any(term in q for term in ("prepare", "prepared", "parameter", "influence", "commarea"))


def _asks_about_call_details(q: str) -> bool:
    return any(
        term in q
        for term in (
            "call",
            "calls",
            "called",
            "parameter",
            "parameters",
            "passed",
            "prepare",
            "prepared",
            "commarea",
            "length",
            "link",
            "xctl",
        )
    )


def _asks_about_pagination(q: str) -> bool:
    return any(term in q for term in ("pagination", "page", "wctpag", "npagt", "npagina")) and any(
        term in q for term in ("enter", "pf7", "pf8", "wctpag", "npagt", "npagina")
    )


def _asks_about_row_selection(q: str) -> bool:
    return any(term in q for term in ("selects a row", "selected progressivo", "progressivo", "selected accounting voice"))


def _asks_about_pf_key_comparison(q: str) -> bool:
    return any(key in q for key in ("pf1", "pf2", "pf3", "pf4", "pf9")) and any(
        term in q for term in ("compare", "target", "xctl", "function key", "reset")
    )


def _asks_about_error_paths(q: str) -> bool:
    return any(
        term in q
        for term in (
            "error message",
            "abnormal",
            "abend",
            "abnormal termination",
            "failed service",
            "invalid function",
            "invalid key",
            "missing record",
            "invalid selection",
            "sql error",
            "sqlerror",
            "restriction",
        )
    )


def _asks_about_variable_behavior(q: str) -> bool:
    return any(
        term in q
        for term in (
            "how is",
            "where is",
            "what feeds",
            "who feeds",
            "calculated",
            "maintained",
            "set",
            "used",
            "read",
            "written",
            "modified",
            "origin",
            "value of",
            "feed",
            "feeds",
        )
    ) and bool(_variables_in_question(q))


def _asks_for_specific_variable_trace(q: str) -> bool:
    return any(term in q for term in ("where is", "set and used", "read and written", "written", "modified", "origin"))


def _asks_about_paragraph_behavior(q: str) -> bool:
    return any(
        term in q
        for term in (
            "what happens",
            "explain",
            "sequence",
            "flow",
            "logic",
            "do in",
            "does",
            "operations",
            "before",
            "after",
        )
    )


def _answer_phase_decision(root: Path, program: str) -> str | None:
    edges = _cfg_edges(root, program)
    phase_edges = [
        edge for edge in edges
        if edge.get("from") == program and "TWCOB-FASE" in str(edge.get("condition", ""))
    ]
    if not phase_edges:
        return None
    lines = [f"{program} dispatches the initial browse phase from `TWCOB-FASE` using CFG edges:"]
    for edge in phase_edges:
        lines.append(f"- `{edge.get('condition')}` -> `{edge.get('to')}` ({edge.get('evidence', '')})")
    variable = _variable_payload(root, program, "TWCOB-FASE")
    if variable:
        lines.append("")
        lines.append("Supporting variable evidence:")
        lines.extend(_site_lines(variable, "control_sites", limit=6))
    _append_evidence(
        lines,
        [
            _cite("controlflow.cfg/controlflow.cfg.json", detail="TWCOB-FASE dispatch edges"),
            _cite("dataflow.variable/dataflow.variable.TWCOB-FASE.json", detail="control_sites"),
        ],
    )
    return "\n".join(lines)


def _answer_semaphore_flow(root: Path, program: str) -> str | None:
    edges = _cfg_edges(root, program)
    read_edges = [edge for edge in edges if edge.get("to") == "READ-TAB-SEMAF"]
    closed_edges = [edge for edge in edges if "PXCSEMAF-STATUS" in str(edge.get("condition", ""))]
    abend_edges = [edge for edge in edges if edge.get("from") == "READ-TAB-SEMAF" and edge.get("to") == "ABEND00"]
    literals = _literal_items(root, program)
    semaf_literals = [
        item for item in literals
        if str(item.get("target_variable", "")).startswith("PXCSEMAF-")
        or str(item.get("literal", "")).upper() == "PDAGGVIP"
    ]
    call = _call_by_target(root, program, "PXRSEMAF")
    if not read_edges and not semaf_literals and not call:
        return None

    lines = [f"{program} semaphore `PDAGGVIP` flow:"]
    for edge in read_edges:
        condition = edge.get("condition") or "unconditional"
        lines.append(f"- `READ-TAB-SEMAF` is reached when `{condition}` ({edge.get('evidence', '')}).")
    if semaf_literals:
        lines.append("- `READ-TAB-SEMAF` prepares the semaphore request:")
        for item in semaf_literals[:8]:
            lines.append(
                f"  - line {item.get('line')} {item.get('paragraph')}: "
                f"{item.get('target_variable')} = {item.get('literal')}"
            )
    if call:
        params = ", ".join(call.get("parameters", [])) or "no explicit parameter"
        lines.append(
            f"- The service call is `{call.get('target')}` in `{call.get('paragraph')}` "
            f"line {call.get('line_start')} with parameter(s): {params}."
        )
    for edge in closed_edges:
        lines.append(
            f"- When `{edge.get('condition')}`, control goes to `{edge.get('to')}` "
            f"({edge.get('evidence', '')})."
        )
    for edge in abend_edges:
        lines.append(
            f"- If `{edge.get('condition')}`, `READ-TAB-SEMAF` performs `{edge.get('to')}` "
            f"({edge.get('evidence', '')})."
        )
    _append_evidence(
        lines,
        [
            _cite("controlflow.cfg/controlflow.cfg.json", detail="READ-TAB-SEMAF edges"),
            _cite("dataflow.literal_assignments/dataflow.literal_assignments.json", detail="PXCSEMAF assignments"),
            _cite("architecture.call_parameters/architecture.call_parameters.json", line=(call or {}).get("line_start"), detail="PXRSEMAF"),
        ],
    )
    return "\n".join(lines)


def _answer_paragraph_sequence(root: Path, program: str, paragraph: str, stop_at: str | None = None) -> str | None:
    edges = [edge for edge in _cfg_edges(root, program) if edge.get("from") == paragraph]
    if not edges:
        return None
    lines = [f"{program} `{paragraph}` sequence from `controlflow.cfg.json`:"]
    for edge in edges:
        condition = edge.get("condition")
        prefix = "conditional" if condition else "step"
        detail = f"{prefix}: `{edge.get('to')}` via {edge.get('type', '?')}"
        if condition:
            detail += f" when `{condition}`"
        evidence = edge.get("evidence")
        if evidence:
            detail += f" ({evidence})"
        lines.append(f"- {detail}.")
        if stop_at and edge.get("to") == stop_at:
            break
    _append_evidence(lines, [_cite("controlflow.cfg/controlflow.cfg.json", detail=f"{paragraph} outgoing edges")])
    return "\n".join(lines)


def _answer_call_preparation(root: Path, program: str, target: str) -> str | None:
    call = _call_by_target(root, program, target)
    if not call:
        return None
    lines = [
        f"{program} prepares `{target}` in `{call.get('paragraph')}` before the call:",
        (
            f"- call statement line {call.get('line_start')}: {call.get('call_type')} "
            f"COMMAREA={call.get('commarea', 'n/a')} LENGTH={call.get('length', 'n/a')}"
        ),
    ]

    key_variables = {
        f"{target}-FUNZIONE",
        f"{target}-TIPO-ESTRA",
        f"{target}-TIPO-VOCE",
        f"{target}-TIPO-GEST",
        f"{target}-TIPO-VARIAZ",
    }
    for detail in call.get("parameter_details", []):
        variables = detail.get("variables", [])
        selected = [
            variable for variable in variables
            if str(variable.get("variable", "")).upper() in key_variables
            or str(variable.get("variable", "")).upper().startswith(f"{target}-COD")
            or str(variable.get("variable", "")).upper().startswith(f"{target}-LIQUID")
        ]
        if not selected:
            selected = [
                variable for variable in variables
                if variable.get("writes_before_call") or variable.get("reads_before_call")
            ][:8]
        if not selected:
            field_prefix = detail.get("field_prefix")
            if field_prefix:
                lines.append(f"- parameter detail `{field_prefix}` has no per-field write evidence in this artifact.")
            continue
        lines.append("- prepared fields:")
        for variable in selected[:18]:
            name = variable.get("variable")
            writes = variable.get("writes_before_call", [])
            if writes:
                for site in writes[:5]:
                    lines.append(f"  - `{name}` line {site.get('line_start')}: {site.get('statement')}")
            else:
                lines.append(f"  - `{name}` has no write-before-call evidence in this artifact.")

    for name in ("TWCOB-VARCONT-NUMFUNZ", "TWCOB-FUNZIONE"):
        variable = _variable_payload(root, program, name)
        if variable:
            lines.append(f"- `{name}` control/read evidence:")
            lines.extend(_site_lines(variable, "control_sites", limit=8, indent="  "))
    _append_evidence(
        lines,
        [
            _cite("architecture.call_parameters/architecture.call_parameters.json", line=call.get("line_start"), detail=target),
            _cite("dataflow.variable/*.json", detail="write/read sites for call parameters"),
        ],
    )
    return "\n".join(lines)


def _answer_pagination(root: Path, program: str) -> str | None:
    variable = _variable_payload(root, program, "WCTPAG")
    edges = _cfg_edges(root, program)
    if not variable:
        return None
    lines = [f"{program} pagination is centered on `WCTPAG` and `TWCOB-VARCONT-NPAGINA`:"]
    lines.append("- writes/updates:")
    lines.extend(_site_lines(variable, "write_sites", limit=10, indent="  "))
    lines.append("- page-control reads:")
    control_sites = _site_lines(variable, "control_sites", limit=12, indent="  ")
    lines.extend(control_sites)
    ui_edges = []
    for edge in edges:
        condition = str(edge.get("condition", ""))
        source = edge.get("from")
        target = edge.get("to")
        if (
            source == "BROWSE-FASE2"
            and edge.get("to") != "BROWSE-FASE2-TASTOER"
            and any(key in condition for key in ("DFHENTER", "DFHPF7", "DFHPF8"))
        ):
            ui_edges.append(edge)
        elif source in {"BROWSE-FASE2-ENTER", "BROWSE-FASE2-PF7", "BROWSE-FASE2-PF8", "BROWSE-FASE2-VISUAL"} and (
            "WCTPAG" in condition or target in {"BROWSE-FASE2-VISUAL", "XCTL-LIV4"}
        ):
            ui_edges.append(edge)
    if ui_edges:
        lines.append("- user-interaction paths:")
        for edge in ui_edges[:12]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{edge.get('from')}` -> `{edge.get('to')}` when `{condition}` ({edge.get('evidence', '')})")
    _append_evidence(
        lines,
        [
            _cite("dataflow.variable/dataflow.variable.WCTPAG.json", detail="write/control sites"),
            _cite("controlflow.cfg/controlflow.cfg.json", detail="PF7/PF8/ENTER pagination edges"),
        ],
    )
    return "\n".join(lines)


def _answer_row_selection(root: Path, program: str) -> str | None:
    variables = [
        _variable_payload(root, program, name)
        for name in ("SCELTAI", "WPROGR", "WPROGREC", "WCTRIG", "WVOCE", "TWCOB-VARCONT-PROGVOCE")
    ]
    variables = [variable for variable in variables if variable]
    edges = [
        edge for edge in _cfg_edges(root, program)
        if edge.get("from") in {"BROWSE-FASE2-ENTER", "BROWSE-FASE2-SEL-10"}
        or edge.get("to") in {"BROWSE-FASE2-SEL", "BROWSE-FASE2-SEL-20", "BROWSE-FASE2-NOTFND"}
    ]
    if not variables and not edges:
        return None
    lines = [f"{program} row/progressivo selection flow:"]
    for edge in edges:
        condition = edge.get("condition") or "unconditional"
        lines.append(f"- `{edge.get('from')}` -> `{edge.get('to')}` when `{condition}` ({edge.get('evidence', '')})")
    for variable in variables:
        content = variable.get("content", {})
        lines.append(f"- `{content.get('variable')}` evidence:")
        lines.extend(_site_lines(variable, "read_sites", limit=4, indent="  "))
        lines.extend(_site_lines(variable, "write_sites", limit=4, indent="  "))
    _append_evidence(
        lines,
        [
            _cite("controlflow.cfg/controlflow.cfg.json", detail="row-selection edges"),
            _cite("dataflow.variable/*.json", detail="SCELTAI/WPROGR/TWA variable sites"),
        ],
    )
    return "\n".join(lines)


def _answer_pf_key_comparison(root: Path, program: str) -> str | None:
    nav = _read_json(root / "ui.cics.navigation" / "ui.cics.navigation.json")
    edges = _cfg_edges(root, program)
    if not isinstance(nav, dict) or nav.get("program") != program:
        return None
    wanted = {"DFHPF1", "DFHPF2", "DFHPF3", "DFHPF4", "DFHPF9"}
    actions = [
        action for action in nav.get("content", {}).get("actions", [])
        if action.get("key") in wanted
    ]
    lines = [f"{program} PF-key control flow:"]
    for action in actions:
        xctl = str(action.get("target", ""))
        reset = any(edge.get("from") == xctl and edge.get("to") == "RESET-TWA" for edge in edges)
        main = any(edge.get("from") == xctl and edge.get("to") == "XCTL-MAIN" for edge in edges)
        assignments = _literal_items(root, program, paragraph=xctl)
        assigned = ", ".join(
            f"{item.get('target_variable')}={item.get('literal')}"
            for item in assignments
            if str(item.get("target_variable", "")).startswith("TWCOB-")
        )
        lines.append(
            f"- `{action.get('key')}` -> `{xctl}` ({action.get('evidence')}); "
            f"{'performs RESET-TWA' if reset else 'no RESET-TWA edge found'}, "
            f"{'then goes to XCTL-MAIN' if main else 'no XCTL-MAIN edge found'}"
            f"{'; ' + assigned if assigned else ''}."
        )
    call = _call_by_target(root, program, "PDPRED")
    if call:
        lines.append(
            f"- `XCTL-MAIN` transfers control with `{call.get('call_type')}` to `{call.get('target')}` "
            f"at line {call.get('line_start')}."
        )
    _append_evidence(
        lines,
        [
            _cite("ui.cics.navigation/ui.cics.navigation.json", detail="PF key actions"),
            _cite("controlflow.cfg/controlflow.cfg.json", detail="XCTL reset edges"),
            _cite("architecture.call_parameters/architecture.call_parameters.json", line=(call or {}).get("line_start"), detail="PDPRED"),
        ],
    )
    return "\n".join(lines)


def _answer_key_flow(root: Path, program: str, question: str) -> str | None:
    keys = _key_tokens_in_question(question)
    if not keys:
        return None

    nav = _read_json(root / "ui.cics.navigation" / "ui.cics.navigation.json")
    actions = []
    if isinstance(nav, dict) and nav.get("program") == program:
        actions = [
            action for action in nav.get("content", {}).get("actions", [])
            if action.get("key") in keys
        ]

    edges = [
        edge for edge in _cfg_edges(root, program)
        if any(key in str(edge.get("condition", "")) for key in keys)
    ]
    if not actions and not edges:
        return None

    lines = [f"{program} key/navigation flow for {', '.join(keys)}:"]
    for action in actions:
        target = str(action.get("target", ""))
        lines.append(
            f"- UI action: `{action.get('context')}` key `{action.get('key')}` -> "
            f"`{target}` ({action.get('edge_type')}; {action.get('evidence')})."
        )
        for edge in [
            edge for edge in _cfg_edges(root, program)
            if str(edge.get("from", "")).upper() == target.upper()
        ][:6]:
            condition = edge.get("condition") or "unconditional"
            lines.append(
                f"  - then `{target}` -> `{edge.get('to')}` when "
                f"`{condition}` ({edge.get('evidence', '')})."
            )
    for edge in edges[:12]:
        condition = edge.get("condition") or "unconditional"
        lines.append(
            f"- CFG edge: `{edge.get('from')}` -> `{edge.get('to')}` when "
            f"`{condition}` ({edge.get('evidence', '')})."
        )
    _append_evidence(
        lines,
        [
            _cite("ui.cics.navigation/ui.cics.navigation.json", detail="key actions"),
            _cite("controlflow.cfg/controlflow.cfg.json", detail="key-conditioned edges"),
        ],
    )
    return "\n".join(lines)


def _key_tokens_in_question(question: str) -> list[str]:
    q = question.lower()
    mapping = {
        "enter": "DFHENTER",
        "pf1": "DFHPF1",
        "pf2": "DFHPF2",
        "pf3": "DFHPF3",
        "pf4": "DFHPF4",
        "pf5": "DFHPF5",
        "pf6": "DFHPF6",
        "pf7": "DFHPF7",
        "pf8": "DFHPF8",
        "pf9": "DFHPF9",
        "pf10": "DFHPF10",
        "pf11": "DFHPF11",
        "pf12": "DFHPF12",
    }
    keys = [target for token, target in mapping.items() if re.search(rf"\b{re.escape(token)}\b", q)]
    keys.extend(re.findall(r"\bDFH(?:ENTER|PF\d+)\b", question.upper()))
    return list(dict.fromkeys(keys))


def _answer_paragraph_behavior(root: Path, program: str, paragraph: str) -> str | None:
    paragraph = paragraph.upper()
    edges = _cfg_edges(root, program)
    outgoing = [edge for edge in edges if str(edge.get("from", "")).upper() == paragraph]
    incoming = [edge for edge in edges if str(edge.get("to", "")).upper() == paragraph]
    literals = _literal_items(root, program, paragraph=paragraph)
    calls = [call for call in _calls(root, program) if str(call.get("paragraph", "")).upper() == paragraph]
    variable_sites = _paragraph_variable_sites(root, program, paragraph)

    if not outgoing and not incoming and not literals and not calls and not variable_sites:
        return None

    lines = [f"{program} paragraph `{paragraph}` behavior from `final_scripts`:"]
    if incoming:
        lines.append("- incoming control-flow:")
        for edge in incoming[:8]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{edge.get('from')}` -> `{paragraph}` when `{condition}` ({edge.get('evidence', '')})")
    if outgoing:
        lines.append("- outgoing control-flow:")
        for edge in outgoing[:12]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{paragraph}` -> `{edge.get('to')}` when `{condition}` ({edge.get('evidence', '')})")
    if calls:
        lines.append("- external/service calls:")
        for call in calls:
            params = ", ".join(call.get("parameters", [])) or "no explicit parameter"
            lines.append(
                f"  - line {call.get('line_start')}: `{call.get('call_type')}` -> "
                f"`{call.get('target')}` with {params}"
            )
    if literals:
        lines.append("- literal assignments:")
        for item in literals[:12]:
            lines.append(
                f"  - line {item.get('line')}: `{item.get('target_variable')}` = "
                f"{item.get('literal')}"
            )
    if variable_sites:
        lines.append("- variable evidence in this paragraph:")
        for site in variable_sites[:12]:
            lines.append(
                f"  - {site['kind']} `{site['variable']}` line {site['line']}: {site['statement']}"
            )
    _append_evidence(
        lines,
        [
            _cite("controlflow.cfg/controlflow.cfg.json", detail=f"{paragraph} edges"),
            _cite("dataflow.literal_assignments/dataflow.literal_assignments.json", detail=f"{paragraph} literals"),
            _cite("architecture.call_parameters/architecture.call_parameters.json", detail=f"{paragraph} calls"),
            _cite("dataflow.variable/*.json", detail=f"{paragraph} variable sites"),
        ],
    )
    return "\n".join(lines)


def _paragraph_variable_sites(root: Path, program: str, paragraph: str) -> list[dict[str, str]]:
    sites: list[dict[str, str]] = []
    dataflow_dir = root / "dataflow.variable"
    if not dataflow_dir.exists():
        return sites
    for path in sorted(dataflow_dir.glob("dataflow.variable.*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("program") != program:
            continue
        variable = str(payload.get("content", {}).get("variable", "")).upper()
        evidence = payload.get("content", {}).get("evidence", {})
        for kind, key in (("write", "write_sites"), ("read", "read_sites"), ("control", "control_sites")):
            for site in evidence.get(key, []):
                if str(site.get("paragraph", "")).upper() != paragraph:
                    continue
                sites.append(
                    {
                        "kind": kind,
                        "variable": variable,
                        "line": str(site.get("line_start", "?")),
                        "statement": str(site.get("statement", "")).strip(),
                    }
                )
                break
    return sites


def _answer_error_paths(root: Path, program: str) -> str | None:
    edges = _cfg_edges(root, program)
    abend_edges = [edge for edge in edges if edge.get("to") == "ABEND00"]
    message_edges = [
        edge for edge in edges
        if edge.get("to") in {"BROWSE-FASE2-TASTOER", "BROWSE-FASE2-NOTFND"}
    ]
    restriction_edges = [
        edge for edge in edges
        if edge.get("to") == "XCTL-LIV4"
        and any(term in str(edge.get("condition", "")) for term in ("PXCSEMAF-STATUS", "PD1VOCI-TABVOX-NUMERO"))
    ]
    lines = [f"{program} error/message and abnormal paths from structured artifacts:"]
    if abend_edges:
        lines.append("- abnormal termination / ABEND00:")
        for edge in abend_edges[:12]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{edge.get('from')}` -> `ABEND00` when `{condition}` ({edge.get('evidence', '')})")
    if message_edges:
        lines.append("- user-facing error/message paths:")
        for edge in message_edges[:8]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{edge.get('from')}` -> `{edge.get('to')}` when `{condition}` ({edge.get('evidence', '')})")
    if restriction_edges:
        lines.append("- restriction/early-transfer paths:")
        for edge in restriction_edges[:6]:
            condition = edge.get("condition") or "unconditional"
            lines.append(f"  - `{edge.get('from')}` -> `{edge.get('to')}` when `{condition}` ({edge.get('evidence', '')})")
    for name in ("M1MSGO", "M1MSGL", "SCELTAL"):
        variable = _variable_payload(root, program, name)
        if variable:
            lines.append(f"- `{name}` message/map evidence:")
            lines.extend(_site_lines(variable, "write_sites", limit=5, indent="  "))
    sqlerror = _variable_payload(root, program, "SQLERROR")
    if sqlerror:
        lines.append("- SQL handling evidence:")
        lines.extend(_site_lines(sqlerror, "read_sites", limit=3, indent="  "))
    _append_evidence(
        lines,
        [
            _cite("controlflow.cfg/controlflow.cfg.json", detail="ABEND/message/restriction edges"),
            _cite("dataflow.variable/*.json", detail="message and SQL variable sites"),
        ],
    )
    return "\n".join(lines)


def _answer_variables_from_question(root: Path, program: str, question: str) -> str | None:
    variables = [_variable_payload(root, program, name) for name in _variables_in_question(question)]
    variables = [variable for variable in variables if variable]
    if not variables:
        return None
    lines = [f"{program} variable evidence:"]
    for variable in variables[:4]:
        lines.append(_format_variable_lineage(program, variable))
    return "\n\n".join(lines)


def _answer_copybooks(root: Path, program: str, q: str) -> str | None:
    payload = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    content = payload.get("content", {})
    all_copybooks = content.get("all", [])
    classified = content.get("classified", {})
    if "unused" in q:
        artifact = load_or_build_unused_copybooks(root, program)
        artifact_content = artifact.get("content", {})
        referenced = artifact_content.get("referenced_copybooks", [])
        needs_review = artifact_content.get("needs_review_copybooks", [])
        status_items = artifact_content.get("copybook_status", [])
        lines = [
            f"{program} COPY usage heuristic from `architecture.unused_copybooks`:",
            "This is not a full unused-copybook proof; it compares COPY members against available dataflow/call artifacts.",
            f"- Proof level: {artifact_content.get('proof_level', 'available-artifact review')}.",
            f"- COPY members listed: {', '.join(artifact_content.get('all_copybooks', all_copybooks))}",
            f"- COPY members with reference evidence: {', '.join(referenced) or 'none'}",
            (
                f"- Need review / possibly unused by this heuristic ({len(needs_review)}): "
                f"{', '.join(needs_review) or 'none'}"
            ),
            "- Proven unused COPY members: none from the available artifacts.",
        ]
        if status_items:
            lines.append("- per-copybook status:")
            for item in status_items:
                evidence = item.get("evidence", [])
                citation = ""
                if evidence:
                    citation = f" [{evidence[0].get('citation') or evidence[0].get('source')}]"
                lines.append(f"  - {item.get('copybook')}: {item.get('status')}{citation}")
        _append_evidence(
            lines,
            [
                _cite("architecture.unused_copybooks/architecture.unused_copybooks.json", detail="derived artifact"),
                _cite("architecture.copybooks/architecture.copybooks.json", detail="content.all"),
                _cite("dataflow.used_variables/dataflow.used_variables.json", detail="copybook-origin evidence"),
                _cite("dataflow.variable/*.json", detail="copybook-origin evidence"),
                _cite("architecture.call_parameters/architecture.call_parameters.json", detail="parameter evidence"),
            ],
        )
        return "\n".join(lines)
    lines = [f"{program} COPY members ({len(all_copybooks)}): {', '.join(all_copybooks)}."]
    for category, names in classified.items():
        lines.append(f"- {category}: {', '.join(names)}")
    _append_evidence(lines, [_cite("architecture.copybooks/architecture.copybooks.json", detail="content.classified")])
    return "\n".join(lines)


def _answer_screen_field_lineage(root: Path, program: str, question: str) -> str | None:
    artifact = load_or_build_screen_field_lineage(root, program)
    content = artifact.get("content", {})
    fields = content.get("fields", [])
    if not isinstance(fields, list) or not fields:
        return _answer_screen_field_lineage_fallback(root, program, question, content)

    by_name = {str(item.get("field", "")).upper(): item for item in fields if isinstance(item, dict)}
    tokens = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", question.upper())
        if token not in {"PDCBVC", "SCREEN", "FIELD", "MAP", "DATA", "ORIGIN", "COMPUTATION", "VARIABLE"}
    ]
    exact = next((token for token in tokens if token in by_name), None)
    if exact:
        return _format_screen_field_lineage(program, by_name[exact])

    related = []
    for token in tokens:
        related.extend(
            name
            for name, field in by_name.items()
            if name.startswith(token)
            or token.startswith(name)
            or str(field.get("family", "")).startswith(token)
            or token.startswith(str(field.get("family", "")))
        )
    related = sorted(set(related))
    if related:
        lines = [f"I found several {program} screen/map fields matching the reference:"]
        for name in related[:10]:
            field = by_name[name]
            lines.append(
                f"- {name}: origin {field.get('origin', '?')}; "
                f"family {field.get('family', '?')}; "
                f"modified in {_join_or_none(field.get('modified_in', []))}; "
                f"used in {_join_or_none(field.get('used_in', []))}; "
                f"controls flow: {'yes' if field.get('controls_flow') else 'no'}"
            )
        lines.append("Ask again with one exact field name to get the full origin/computation trace.")
        _append_evidence(lines, [_cite("screen_field_lineage/screen_field_lineage.json", detail="fields")])
        return "\n".join(lines)

    candidate_names = sorted(by_name)[:20]
    lines = [
        "I need the concrete map field name to trace data origin/computation safely.",
        (
            f"For {program}, `screen_field_lineage` can trace {len(by_name)} screen/map variables "
            f"from copybook origin(s) {', '.join(content.get('copybook_origins', [])) or 'unknown'}."
        ),
        f"Examples: {', '.join(candidate_names)}.",
        "Ask for one field, for example: `What feeds SCELTAI on the screen?`",
    ]
    _append_evidence(lines, [_cite("screen_field_lineage/screen_field_lineage.json", detail="fields")])
    return "\n".join(lines)


def _answer_screen_field_lineage_fallback(
    root: Path,
    program: str,
    question: str,
    lineage_content: dict[str, Any],
) -> str | None:
    variables = _known_variables(root, program)
    tokens = [
        token
        for token in re.findall(r"\b[A-Z][A-Z0-9-]{2,}\b", question.upper())
        if token in variables
    ]
    if not tokens:
        return None
    variable_name = tokens[0]
    payload = _variable_payload(root, program, variable_name)
    if not payload:
        return None

    family = _map_field_family(variable_name)
    family_members = sorted(name for name in variables if _map_field_family(name) == family)
    lines = [
        f"{program} screen/map field `{variable_name}` evidence:",
        (
            "`screen_field_lineage` is present but has no field records for this run, "
            "so this answer falls back to `dataflow.variable` evidence."
        ),
        _format_variable_lineage(program, payload),
    ]
    related = [name for name in family_members if name != variable_name]
    if related:
        lines.append("")
        lines.append(f"Same BMS-style field family `{family}` from dataflow variables: {', '.join(family_members)}.")
        lines.append("Related family members can represent input/output/length/attribute variants, but origin is only proven when the dataflow artifact says so.")
        for name in related[:8]:
            related_payload = _variable_payload(root, program, name)
            if not related_payload:
                continue
            content = related_payload.get("content", {})
            lines.append(
                f"- `{name}`: origin {content.get('origin', '?')}; "
                f"modified in {_join_or_none(content.get('modified_in', []))}; "
                f"used in {_join_or_none(content.get('used_in', []))}"
            )
    if lineage_content.get("limitations"):
        lines.append("")
        lines.append("Lineage limitation: " + str(lineage_content.get("limitations", [""])[0]))
    _append_evidence(
        lines,
        [
            _cite("screen_field_lineage/screen_field_lineage.json", detail="fields_count=0"),
            _cite(f"dataflow.variable/dataflow.variable.{variable_name}.json", detail="fallback variable evidence"),
        ],
    )
    return "\n".join(lines)


def _map_field_family(variable: str) -> str:
    variable = variable.upper()
    if len(variable) > 1 and variable[-1] in {"I", "O", "L", "A", "F"}:
        return variable[:-1]
    return variable


def _screen_variables(root: Path, program: str) -> list[dict[str, Any]]:
    variables: list[dict[str, Any]] = []
    for path in sorted((root / "dataflow.variable").glob("dataflow.variable.*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("program") != program:
            continue
        content = payload.get("content", {})
        origin = str(content.get("origin", "")).upper()
        if origin == "COPY:PDCBVCM":
            variables.append(payload)
    return variables


def _format_variable_lineage(program: str, payload: dict[str, Any]) -> str:
    content = payload.get("content", {})
    variable = content.get("variable", "?")
    lines = [
        f"{program} variable `{variable}`:",
        f"- origin: {content.get('origin', '?')}",
        f"- defined in: {_join_or_none(content.get('defined_in', []))}",
        f"- modified in: {_join_or_none(content.get('modified_in', []))}",
        f"- used in: {_join_or_none(content.get('used_in', []))}",
        f"- controls flow: {'yes' if content.get('controls_flow') else 'no'}",
    ]
    evidence = content.get("evidence", {})
    for label, key in (("writes", "write_sites"), ("reads", "read_sites"), ("controls", "control_sites")):
        sites = evidence.get(key, [])
        if not sites:
            continue
        lines.append(f"- {label}:")
        for site in sites[:6]:
            line = site.get("line_start", "?")
            paragraph = site.get("paragraph", "?")
            statement = str(site.get("statement", "")).strip()
            lines.append(f"  - line {line} {paragraph}: {statement} [{_site_citation(payload, line)}]")
    _append_evidence(lines, [_site_citation(payload, _line_from_site_list(payload, "read_sites") or _line_from_site_list(payload, "write_sites"))])
    return "\n".join(lines)


def _format_screen_field_lineage(program: str, field: dict[str, Any]) -> str:
    name = str(field.get("field", "?"))
    lines = [
        f"{program} screen/map field `{name}` lineage from `screen_field_lineage`:",
        f"- origin: {field.get('origin', '?')}",
        f"- BMS-style family: {field.get('family', '?')} ({', '.join(field.get('family_members', [])) or name})",
        f"- defined in: {_join_or_none(field.get('defined_in', []))}",
        f"- modified in: {_join_or_none(field.get('modified_in', []))}",
        f"- used in: {_join_or_none(field.get('used_in', []))}",
        f"- controls flow: {'yes' if field.get('controls_flow') else 'no'}",
    ]
    if field.get("read_sites"):
        lines.append("- read/input/control use:")
        for site in field.get("read_sites", [])[:8]:
            lines.append(
                f"  - line {site.get('line_start')} `{site.get('paragraph')}`: "
                f"{site.get('statement')} [{site.get('citation')}]"
            )
    if field.get("write_sites"):
        lines.append("- direct writes:")
        for site in field.get("write_sites", [])[:8]:
            lines.append(
                f"  - line {site.get('line_start')} `{site.get('paragraph')}`: "
                f"{site.get('statement')} [{site.get('citation')}]"
            )
    if field.get("control_sites"):
        lines.append("- control-flow use:")
        for site in field.get("control_sites", [])[:8]:
            lines.append(
                f"  - line {site.get('line_start')} `{site.get('paragraph')}`: "
                f"{site.get('statement')} [{site.get('citation')}]"
            )
    if field.get("literal_assignments"):
        lines.append("- related screen literal/attribute assignments in the same field family:")
        for item in field.get("literal_assignments", [])[:8]:
            lines.append(
                f"  - line {item.get('line')} `{item.get('paragraph')}`: "
                f"{item.get('target_variable')} = {item.get('literal')} [{item.get('citation')}]"
            )
    if field.get("related_variables"):
        lines.append("- connected variables seen in the same statements:")
        for item in field.get("related_variables", [])[:8]:
            lines.append(
                f"  - `{item.get('variable')}` at line {item.get('line')} "
                f"({item.get('relationship')}) [{item.get('citation')}]"
            )
    _append_evidence(
        lines,
        [
            _cite("screen_field_lineage/screen_field_lineage.json", detail=name),
            str(field.get("source_artifact", "")),
        ],
    )
    return "\n".join(lines)


def _join_or_none(values: list[Any]) -> str:
    return ", ".join(str(value) for value in values) if values else "none"


def _copybook_origins_from_dataflow(root: Path, program: str) -> set[str]:
    origins: set[str] = set()
    copybooks_payload = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    known_copybooks = set()
    if isinstance(copybooks_payload, dict) and copybooks_payload.get("program") == program:
        known_copybooks = set(copybooks_payload.get("content", {}).get("all", []))

    def mark_by_prefix(value: str) -> None:
        for copybook in known_copybooks:
            if value == copybook or value.startswith(f"{copybook}-"):
                origins.add(copybook)

    used = _read_json(root / "dataflow.used_variables" / "dataflow.used_variables.json")
    if isinstance(used, dict) and used.get("program") == program:
        for variable in used.get("variables", []):
            mark_by_prefix(str(variable.get("variable", "")))
            origin = str(variable.get("origin", ""))
            if origin.startswith("COPY:"):
                origins.add(origin.split(":", 1)[1])
    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("program") != program:
            continue
        origin = str(payload.get("content", {}).get("origin", ""))
        if origin.startswith("COPY:"):
            origins.add(origin.split(":", 1)[1])
        mark_by_prefix(str(payload.get("content", {}).get("variable", "")))

    literals = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if isinstance(literals, dict) and literals.get("program") == program:
        for item in literals.get("assignments", []):
            mark_by_prefix(str(item.get("target_variable", "")))

    calls = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if isinstance(calls, dict) and calls.get("program") == program:
        for call in calls.get("calls", []):
            for parameter in call.get("parameters", []):
                mark_by_prefix(str(parameter))
            for detail in call.get("parameter_details", []):
                mark_by_prefix(str(detail.get("field_prefix", "")))
                for variable in detail.get("variables", []):
                    mark_by_prefix(str(variable.get("variable", "")))

    if "DFHAID" in known_copybooks and _uses_cics_aid_constants(root, program):
        origins.add("DFHAID")
    return origins


def _uses_cics_aid_constants(root: Path, program: str) -> bool:
    used = _read_json(root / "dataflow.used_variables" / "dataflow.used_variables.json")
    if not isinstance(used, dict) or used.get("program") != program:
        return False
    for variable in used.get("variables", []):
        name = str(variable.get("variable", ""))
        origin = str(variable.get("origin", ""))
        if origin == "CICS_CONST" and (name.startswith("DFHPF") or name == "DFHENTER"):
            return True
    return False


def _answer_db2_sql(root: Path, program: str) -> str | None:
    db2_files = sorted((root / "architecture.db2_table").glob("architecture.db2_table.*.json"))
    sql_files = sorted((root / "architecture.sqlinclude").glob("architecture.sqlinclude.*.json"))
    db2 = [_read_json(path) for path in db2_files]
    sql = [_read_json(path) for path in sql_files]
    db2 = [item for item in db2 if isinstance(item, dict) and item.get("program") == program]
    sql = [item for item in sql if isinstance(item, dict) and item.get("program") == program]
    if not db2 and not sql:
        return None
    lines = [f"{program} DB2/SQL evidence:"]
    for item in db2:
        content = item.get("content", {})
        table = content.get("table") or item.get("title", "").replace(f"{program} DB2 table ", "")
        statement_type = content.get("statement_type") or content.get("verb") or "unknown statement"
        lines.append(f"- DB2 table {table}: {statement_type}")
    if sql:
        includes = [str(item.get("content", {}).get("include") or item.get("title", "")) for item in sql]
        lines.append(f"- SQL includes: {', '.join(includes)}")
    _append_evidence(
        lines,
        [
            _cite("architecture.db2_table/*.json", detail="DB2 table artifacts"),
            _cite("architecture.sqlinclude/*.json", detail="SQL include artifacts"),
        ],
    )
    return "\n".join(lines)


def _answer_datasets(root: Path, program: str) -> str:
    artifact = load_or_build_jcl_file_io(root, program)
    content = artifact.get("content", {})
    if content.get("has_jcl_linkage"):
        lines = [f"{program} JCL file-I/O evidence from `jcl.file_io`:"]
        evidence: list[str] = [_cite("jcl.file_io/" + f"jcl.file_io.{program.upper()}.json", detail="derived artifact")]
        jobs = content.get("matching_jobs", [])
        if jobs:
            lines.append("- matching job(s): " + ", ".join(str(job.get("job")) for job in jobs))
        reads = content.get("reads", [])
        writes = content.get("writes", [])
        sysout = content.get("sysout", [])
        if reads:
            lines.append(f"- reads ({len(reads)}):")
            for item in reads[:10]:
                lines.append(
                    f"  - {item.get('job')}/{item.get('step')} {item.get('ddname')}: "
                    f"{item.get('dsn')} [{item.get('citation')}]"
                )
                evidence.append(str(item.get("citation", "")))
        if writes:
            lines.append(f"- writes/produces ({len(writes)}):")
            for item in writes[:10]:
                lines.append(
                    f"  - {item.get('job')}/{item.get('step')} {item.get('ddname')}: "
                    f"{item.get('dsn')} [{item.get('citation')}]"
                )
                evidence.append(str(item.get("citation", "")))
        if sysout:
            lines.append(f"- SYSOUT outputs ({len(sysout)}):")
            for item in sysout[:8]:
                lines.append(
                    f"  - {item.get('job')}/{item.get('step')} {item.get('ddname')}: "
                    f"SYSOUT={item.get('sysout')} OUTPUT={item.get('output')} [{item.get('citation')}]"
                )
                evidence.append(str(item.get("citation", "")))
        _append_evidence(lines, evidence)
        return "\n".join(lines)

    known_jobs = content.get("known_jobs", [])
    known_programs = content.get("known_programs_sample", [])
    lines = [
        f"`jcl.file_io` found no JCL dataset/file-I/O linkage for {program}. "
        "Produced datasets are not evidenced for this program in the available final_scripts artifacts. "
        f"Known JCL job(s) in the artifact set: {', '.join(known_jobs) or 'none'}. "
        f"Known batch program sample: {', '.join(known_programs[:10]) or 'none'}."
    ]
    _append_evidence(
        lines,
        [
            _cite("jcl.file_io/" + f"jcl.file_io.{program.upper()}.json", detail="derived artifact"),
            _cite("jcl/**/*.json", detail="scanned JCL summaries and steps"),
        ],
    )
    return "\n".join(lines)


def _answer_ui_navigation(root: Path, program: str) -> str | None:
    payload = _read_json(root / "ui.cics.navigation" / "ui.cics.navigation.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    actions = payload.get("content", {}).get("actions", [])
    lines = [f"{program} CICS UI/navigation actions: {len(actions)} item(s)."]
    for action in actions[:20]:
        lines.append(
            f"- {action.get('context')}: key {action.get('key')} -> {action.get('target')} "
            f"({action.get('edge_type')})"
        )
    _append_evidence(lines, [_cite("ui.cics.navigation/ui.cics.navigation.json", detail="content.actions/maps")])
    return "\n".join(lines)


def _answer_business_rules(root: Path, program: str) -> str | None:
    rules = []
    for path in sorted((root / "business_rule").glob("business_rule*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and payload.get("program") == program:
            rules.append(payload)
    if not rules:
        return None
    lines = [f"{program} business rules: {len(rules)} rule artifact(s)."]
    for rule in rules[:20]:
        content = rule.get("content", {})
        rule_id = content.get("id") or rule.get("id") or "rule"
        condition = content.get("condition") or content.get("if") or rule.get("embedding_text", "")
        target = content.get("target") or content.get("then") or ""
        lines.append(f"- {rule_id}: {condition} {target}".strip())
    _append_evidence(lines, [_cite("business_rule/business_rule*.json", detail="rule artifacts")])
    return "\n".join(lines)
