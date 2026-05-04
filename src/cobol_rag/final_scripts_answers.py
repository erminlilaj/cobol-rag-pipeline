from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def answer_from_final_scripts(question: str) -> str | None:
    root = find_final_scripts_root()
    program = program_from_question(question)
    if root is None or program is None:
        return None

    q = question.lower()

    if _asks_about_lines_or_counts(q):
        answer = _answer_counts(root, program, q)
        if answer:
            return answer

    if _asks_about_dead_or_commented_code(q):
        answer = _answer_commented_code(root, program, q)
        if answer:
            return answer

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

    return None


def find_final_scripts_root() -> Path | None:
    cwd = Path.cwd().resolve()
    for base in (cwd, *cwd.parents):
        candidates = [
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


def _asks_about_lines_or_counts(q: str) -> bool:
    return any(term in q for term in ("how many", "number of", "count", "loc", "lines"))


def _asks_about_dead_or_commented_code(q: str) -> bool:
    return any(term in q for term in ("unused code", "dead code", "commented-out", "commented out", "commented code"))


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


def _asks_about_business_rules(q: str) -> bool:
    return any(term in q for term in ("business rule", "business rules", "rules", "condition", "conditions"))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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
        return f"{program} has {len(all_copybooks)} COPY members listed: {', '.join(all_copybooks)}."

    if ("call" in q or "external" in q or "outside" in q) and isinstance(calls, dict):
        call_items = calls.get("calls", [])
        return f"{program} has {len(call_items)} outgoing calls in `architecture.call_parameters.json`."

    if ("literal" in q or "forced" in q or "hardcoded" in q) and isinstance(literals, dict):
        items = literals.get("assignments", [])
        return f"{program} has {len(items)} literal assignments in `dataflow.literal_assignments.json`."

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
            return " ".join(parts)

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
    comments = _comments_payload(root, program)
    if not comments:
        return None
    commented = [
        comment for comment in comments.get("comments", [])
        if comment.get("classification") == "commented_out_code"
    ]
    lines = [f"Commented-out code/data found in {program}: {len(commented)} item(s)."]
    for comment in commented[:20]:
        lines.append(f"- line {comment.get('line')}: {str(comment.get('text_raw') or comment.get('text', '')).strip()}")
    if "copy" in q:
        copy_answer = _answer_copybooks(root, program, q)
        if copy_answer:
            lines.append("")
            lines.append(copy_answer)
    return "\n".join(lines)


def _answer_calls(root: Path, program: str) -> str | None:
    payload = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    calls = payload.get("calls", [])
    lines = [f"{program} outgoing calls with parameters:"]
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
    return "\n".join(lines)


def _answer_copybooks(root: Path, program: str, q: str) -> str | None:
    payload = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    content = payload.get("content", {})
    all_copybooks = content.get("all", [])
    classified = content.get("classified", {})
    if "unused" in q:
        used_origins = _copybook_origins_from_dataflow(root, program)
        heuristic_unused = [name for name in all_copybooks if name not in used_origins]
        lines = [
            f"{program} COPY usage heuristic:",
            "This is not a full unused-copybook proof; it compares COPY members against dataflow variable origins.",
            f"- COPY members listed: {', '.join(all_copybooks)}",
            f"- COPY members with variables referenced in dataflow: {', '.join(sorted(used_origins)) or 'none'}",
            f"- Need review / possibly unused by this heuristic: {', '.join(heuristic_unused) or 'none'}",
        ]
        return "\n".join(lines)
    lines = [f"{program} COPY members ({len(all_copybooks)}): {', '.join(all_copybooks)}."]
    for category, names in classified.items():
        lines.append(f"- {category}: {', '.join(names)}")
    return "\n".join(lines)


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
    return "\n".join(lines)


def _answer_datasets(root: Path, program: str) -> str:
    matched_jobs: list[str] = []
    for summary in (root / "jcl").glob("**/jcl.summary.json"):
        payload = _read_json(summary)
        if not isinstance(payload, dict):
            continue
        programs = {str(item).upper() for item in payload.get("programs", [])}
        if program.upper() in programs:
            matched_jobs.append(str(payload.get("job", summary.parent.name)))
    if matched_jobs:
        return f"{program} appears in JCL job(s): {', '.join(sorted(set(matched_jobs)))}. Check the job dataset artifacts for inputs/outputs."
    return (
        f"I found no JCL dataset/file-I/O artifact connecting {program} to produced datasets in `final_scripts`. "
        "For this PDCBVC index, dataset production is not evidenced; it looks like a CICS/DB2 program rather than a batch dataset producer."
    )


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
    return "\n".join(lines)
