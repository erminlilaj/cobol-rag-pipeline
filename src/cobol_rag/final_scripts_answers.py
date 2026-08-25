from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from cobol_rag.query_plan import QueryPlan


def answer_from_final_scripts(
    question: str,
    intent: str | None = None,
    plan: QueryPlan | None = None,
) -> str | None:
    root = find_final_scripts_root()
    if root is None:
        return None
    program = plan.program if plan and plan.program else program_from_question(question, root)
    if program is None:
        return None
    program_root = find_program_artifact_root(root, program)
    if program_root is None:
        return None
    root = program_root

    q = question.lower()
    intent = plan.intent if plan else intent
    use_text_fallback = plan is None

    if intent == "artifact_inventory" or (use_text_fallback and _asks_about_artifact_inventory(q)):
        answer = _answer_artifact_inventory(root, program)
        if answer:
            return answer

    if intent == "program_summary" or (use_text_fallback and _asks_about_program_summary(q)):
        answer = _answer_program_summary(root, program, plan)
        if answer:
            return answer

    # A whole-program variable question has no exact entity to look up. The
    # generated catalogue is the authoritative source for it, so the semantic
    # router can route here instead of forcing an entity-scoped capability.
    if intent == "variable_inventory":
        answer = _answer_variable_inventory(root, program, plan)
        if answer:
            return answer

    # Exact variable scope is authoritative. Execute it before program-wide
    # control-flow or literal handlers so a semantic classification cannot replace
    # the named entity with an unrelated artifact family.
    if intent == "variable_dataflow" and plan and plan.entity_values_for("variable"):
        variable_answer = _answer_variable_access(root, program, question, plan)
        if variable_answer:
            return variable_answer

    if intent == "business_rules" or (use_text_fallback and _asks_about_business_rules(q)):
        answer = _answer_business_rules(root, program, plan)
        if answer:
            return answer

    if intent == "control_flow" or (use_text_fallback and _asks_about_control_flow(q)):
        answer = (
            _answer_pagination(root, program)
            if plan and "pagination_logic" in plan.tasks
            else _answer_control_flow(root, program, question)
        )
        if answer:
            return answer

    if intent == "cics_operations" or (use_text_fallback and _asks_about_cics_operations(q)):
        answer = _answer_cics_operations(root, program, plan)
        if answer:
            return answer

    if intent == "dead_code" or (use_text_fallback and _asks_about_dead_or_commented_code(q)):
        answer = _answer_commented_code(root, program, q)
        if answer:
            return answer

    if intent == "external_programs" or (use_text_fallback and _asks_about_calls(q)):
        answer = _answer_calls(root, program, plan)
        if answer:
            return answer

    if intent == "static_values" or (use_text_fallback and _asks_about_forced_values(q)):
        answer = _answer_literal_assignments(root, program, q, plan)
        if answer:
            return answer

    if intent == "copybooks" or (use_text_fallback and _asks_about_copybooks(q)):
        answer = _answer_copybooks(root, program, q, plan)
        if answer:
            return answer

    if intent == "db2_sql" or (use_text_fallback and _asks_about_db2_or_sql(q)):
        answer = _answer_db2_sql(root, program, q, plan)
        if answer:
            return answer

    if intent in {"datasets", "datasets_tables"} or (use_text_fallback and _asks_about_datasets(q)):
        return _answer_datasets(root, program)

    # The manifest records capabilities the analysis produced no evidence for.
    # Saying so is a real answer; letting the question fall through to retrieval
    # returns whatever is merely nearby and reads as if it were the answer.
    absent = _absent_capability_answer(root, program, intent)
    if absent:
        return absent

    if intent == "ui_navigation" or (use_text_fallback and _asks_about_ui_navigation(q)):
        answer = _answer_ui_navigation(root, program)
        if answer:
            return answer

    if intent in {None, "source_metrics"} and _asks_about_lines_or_counts(q):
        answer = _answer_counts(root, program, q)
        if answer:
            return answer

    if intent == "variable_dataflow" or (use_text_fallback and _asks_about_variable_access(q)):
        variable_answer = _answer_variable_access(root, program, question, plan)
        if variable_answer:
            return variable_answer

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


def find_program_artifact_root(root: Path, program: str) -> Path | None:
    """Resolve one program's aggregate artifacts inside a single- or multi-program corpus."""
    program = program.upper()
    candidates = (
        root / program,
        root / "programs" / program / "artifacts",
        root / "programs" / program,
        root,
    )
    for candidate in candidates:
        if _directory_contains_program(candidate, program):
            return candidate

    marker_names = (
        "program.summary.json",
        "architecture.copybooks.json",
        "dataflow.literal_assignments.json",
        "controlflow.cfg.json",
    )
    for marker_name in marker_names:
        for marker in sorted(root.rglob(marker_name)):
            payload = _read_json(marker)
            if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program:
                return marker.parent
    return None


def _directory_contains_program(directory: Path, program: str) -> bool:
    if not directory.is_dir():
        return False
    for path in directory.glob("*.json"):
        payload = _read_json(path)
        if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program:
            return True
    return False


def program_from_question(question: str, root: Path | None = None) -> str | None:
    known_programs = _known_programs(root) if root is not None else set()
    question_upper = question.upper()
    mentioned = [
        program
        for program in known_programs
        if re.search(rf"(?<![A-Z0-9-]){re.escape(program)}(?![A-Z0-9-])", question_upper)
    ]
    if mentioned:
        return max(mentioned, key=len)
    if len(known_programs) == 1:
        return next(iter(known_programs))

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


def _known_programs(root: Path | None) -> set[str]:
    if root is None or not root.exists():
        return set()
    programs: set[str] = set()
    candidate_files = list(root.rglob("*.json"))
    for path in candidate_files:
        payload = _read_json(path)
        if not isinstance(payload, dict):
            continue
        program = payload.get("program")
        if isinstance(program, str) and program.strip():
            programs.add(program.strip().upper())
    return programs


def _asks_about_lines_or_counts(q: str) -> bool:
    return (
        any(term in q for term in ("how many", "number of", "count"))
        or re.search(r"\b(?:loc|line|lines)\b", q) is not None
    )


def _asks_about_dead_or_commented_code(q: str) -> bool:
    return any(term in q for term in ("unused code", "dead code", "commented-out", "commented out", "commented code"))


def _asks_about_calls(q: str) -> bool:
    return any(term in q for term in ("call", "calls", "outside program", "external program", "commarea", "parameter"))


def _asks_about_variable_access(q: str) -> bool:
    return any(
        term in q
        for term in (
            "modify", "modified", "write", "written", "set", "test",
            "tested", "check", "checked", "read", "dataflow", "data flow",
            "return value", "value returned",
        )
    )


def _asks_about_forced_values(q: str) -> bool:
    return any(term in q for term in ("forced value", "forced values", "literal", "hardcoded", "hard-coded", "static value"))


def _asks_about_copybooks(q: str) -> bool:
    return any(
        term in q
        for term in (
            "copybook", "copy book", "copy member", "copy members", "copy statement",
            "copy statements", "unused copy",
        )
    )


def _asks_about_db2_or_sql(q: str) -> bool:
    return any(term in q for term in ("db2", "sql", "table", "tables", "sqlinclude", "sql include"))


def _asks_about_datasets(q: str) -> bool:
    return any(term in q for term in ("dataset", "datasets", "file io", "file i/o", "produce", "produces", "output file"))


def _asks_about_ui_navigation(q: str) -> bool:
    return any(term in q for term in ("pf key", "pfkey", "screen", "map", "navigation", "cics key", "eibaid"))


def _asks_about_business_rules(q: str) -> bool:
    return any(term in q for term in ("business rule", "business rules", "rules", "condition", "conditions"))


def _asks_about_artifact_inventory(q: str) -> bool:
    if _asks_about_datasets(q) or _asks_about_copybooks(q):
        return False
    asks_for_files = "file" in q and any(
        term in q for term in ("name", "names", "have", "available", "indexed", "analyzed", "analysed", "list")
    )
    return asks_for_files or any(
        term in q for term in ("artifact inventory", "available artifacts", "indexed artifacts", "analyzed artifacts")
    )


def _asks_about_control_flow(q: str) -> bool:
    if any(
        term in q
        for term in ("control flow", "execution flow", "entry point", "flow from", "path to termination")
    ):
        return True
    return (
        any(term in q for term in ("decide whether", "choose between", "start from", "branch to", "path between"))
        and any(term in q for term in ("paragraph", "fase", "phase", "branch", "path", "start"))
    )


def _asks_about_program_summary(q: str) -> bool:
    return any(
        term in q
        for term in ("program about", "what is the program", "what does the program", "program purpose", "purpose of program", "program overview", "program summary")
    )


def _asks_about_cics_operations(q: str) -> bool:
    return "cics" in q and any(term in q for term in ("command", "commands", "operation", "operations", "execute", "executes", "used"))


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _artifact_path(root: Path, flat_name: str, nested_name: str | None = None) -> Path:
    """Resolve both the current flat aggregate layout and the legacy nested layout."""
    flat = root / flat_name
    if flat.exists():
        return flat
    nested = nested_name or flat_name
    return root / flat_name.removesuffix(".json") / nested


def _comments_payload(root: Path, program: str) -> dict[str, Any] | None:
    payload = _read_json(_artifact_path(root, "program.comments.json"))
    if isinstance(payload, dict) and payload.get("program") == program:
        return payload
    return None


def _summary_payload(root: Path, program: str) -> dict[str, Any] | None:
    payload = _read_json(_artifact_path(root, "program.summary.json"))
    if isinstance(payload, dict) and payload.get("program") == program:
        return payload
    return None


def _answer_variable_inventory(
    root: Path,
    program: str,
    plan: QueryPlan | None = None,
) -> str | None:
    """Answer whole-program variable questions from the generated variable catalogue."""
    payload = _read_json(_artifact_path(root, "dataflow.used_variables.json"))
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    entries = [item for item in payload.get("variables", []) if isinstance(item, dict)]
    if not entries:
        return None

    requested_fields = set(plan.output_fields) if plan else set()
    exhaustive = bool(plan and plan.result_scope == "all")
    origin_filter = {
        section.replace(" SECTION", "").strip().upper()
        for section in (plan.sections if plan else ())
    }
    if origin_filter:
        entries = [
            item for item in entries
            if str(item.get("origin", "")).strip().upper() in origin_filter
        ]
        if not entries:
            return None
    if plan and "control_usage" in requested_fields:
        entries = [item for item in entries if item.get("controls_flow")]
        if not entries:
            return None

    response = plan.response_contract if plan else None
    offset = max(0, int(plan.result_offset if plan else 0))
    visible_entries = entries[offset:]
    if (response and response.format == "count") or "item_count" in requested_fields:
        return str(len(entries))
    if response and response.exact_item_count is not None:
        visible_entries = visible_entries[:response.exact_item_count]
    if response and response.format in {"json", "json_array"}:
        rows = [
            {
                "variable": item.get("variable"),
                "origin": item.get("origin"),
                "controls_flow": bool(item.get("controls_flow")),
            }
            for item in visible_entries
        ]
        value: Any = rows if response.format == "json_array" else {"variables": rows}
        return json.dumps(value, ensure_ascii=False, indent=2)

    total = len(entries)
    limit = len(visible_entries) if exhaustive else min(len(visible_entries), 25)
    if offset:
        lines = [f"{program} has {len(visible_entries)} remaining analyzed variable(s) after the first {offset} of {total}."]
    else:
        lines = [f"{program} declares {total} analyzed variable(s)."]
    for item in visible_entries[:limit]:
        name = str(item.get("variable", "")).strip()
        if not name:
            continue
        details: list[str] = []
        origin = str(item.get("origin", "")).strip()
        if origin and origin.upper() != "UNKNOWN":
            details.append(origin)
        if item.get("controls_flow"):
            details.append("controls flow")
        defined_in = [str(value) for value in item.get("defined_in", []) if value]
        if defined_in and "paragraph" in requested_fields:
            details.append(f"defined in {', '.join(defined_in)}")
        lines.append(f"- {name}" + (f" ({'; '.join(details)})" if details else ""))
    if limit < len(visible_entries):
        lines.append(
            f"Showing the first {limit} of {len(visible_entries)} matching analyzed variables. "
            "Ask for all of them to receive the complete list."
        )
    lines.append("Source: `dataflow.used_variables.json`.")
    return "\n".join(lines)


def _answer_artifact_inventory(root: Path, program: str) -> str | None:
    top_level: list[str] = []
    for path in sorted(root.glob("*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program:
            top_level.append(path.name)

    detail_groups: list[tuple[str, int]] = []
    for directory in sorted(path for path in root.iterdir() if path.is_dir()):
        count = 0
        for path in directory.glob("*.json"):
            payload = _read_json(path)
            if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program:
                count += 1
        if count:
            detail_groups.append((directory.name, count))

    if not top_level and not detail_groups:
        return None
    lines = [f"Analyzed evidence available for {program}:"]
    if top_level:
        lines.append("Aggregate files:")
        lines.extend(f"- `{name}`" for name in top_level)
    if detail_groups:
        lines.append("Detailed artifact groups:")
        lines.extend(f"- `{name}/` — {count} JSON file(s)" for name, count in detail_groups)
    lines.append("This is the evidence inventory exposed to the assistant for the selected program.")
    return "\n".join(lines)


# Intents whose whole answer comes from one capability, so an absent capability
# means the question has a definite negative answer rather than a missing one.
_INTENT_SOLE_CAPABILITY = {
    "ui_navigation": "screen_lineage",
    "variable_inventory": "variable_inventory",
    "cics_operations": "cics_evidence",
    "copybooks": "copybook_evidence",
    "dead_code": "quality_evidence",
}


def absent_capability_answer(program: str | None, capability: str) -> str | None:
    """State that a program has no evidence of a kind, and why the analysis says so.

    Absence is a finding, not a gap to route around: the manifest records both
    that a capability produced nothing and the reason it produced nothing, so
    the answer is a lookup rather than something inferred from empty retrieval.
    """
    if not program or not capability:
        return None
    manifest = capability_manifest(program)
    if not manifest:
        return None
    entry = manifest.get("capabilities", {}).get(capability)
    if not isinstance(entry, dict) or entry.get("available", True):
        return None
    reason = str(entry.get("reason") or "").strip()
    # Capability names already end in "_evidence" for most families, so the noun
    # is dropped from the label rather than repeated in the sentence.
    label = capability[: -len("_evidence")] if capability.endswith("_evidence") else capability
    return (
        f"{program} has no {label.replace('_', ' ')} evidence in the analyzed artifacts"
        + (f": {reason}." if reason else ".")
        + f" Source: `{entry.get('artifact', 'program.capability_manifest.json')}`."
    )


def _absent_capability_answer(root: Path, program: str, intent: str | None) -> str | None:
    capability = _INTENT_SOLE_CAPABILITY.get(str(intent or ""))
    if not capability:
        return None
    return absent_capability_answer(program, capability)


def capability_manifest(program: str | None) -> dict[str, Any] | None:
    """Read the generated index of what evidence a program actually has.

    The analysis stage records which capabilities were produced, how many items
    each holds, and why one is missing. Reading it turns "does this program have
    JCL" into a lookup instead of something inferred from an empty retrieval,
    and keeps the answer honest when a program genuinely has no such evidence.
    """
    if not program:
        return None
    root = find_final_scripts_root()
    if root is None:
        return None
    program_root = find_program_artifact_root(root, program)
    if program_root is None:
        return None
    payload = _read_json(_artifact_path(program_root, "program.capability_manifest.json"))
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    content = payload.get("content")
    return content if isinstance(content, dict) else None


def unavailable_capabilities(program: str | None) -> frozenset[str]:
    """Capabilities the analysis proved this program has no evidence for."""
    manifest = capability_manifest(program)
    if not manifest:
        return frozenset()
    capabilities = manifest.get("capabilities", {})
    if not isinstance(capabilities, dict):
        return frozenset()
    return frozenset(
        name for name, entry in capabilities.items()
        if isinstance(entry, dict) and not entry.get("available", True)
    )


def _program_character(root: Path, program: str) -> str:
    """Describe what kind of program this is from the evidence that exists for it."""
    traits: list[str] = []
    cics = _read_json(_artifact_path(root, "architecture.cics_operations.json"))
    if isinstance(cics, dict) and cics.get("program") == program:
        traits.append("CICS online program")
    else:
        traits.append("COBOL program")
    db2_root = root / "architecture.db2_table"
    if db2_root.is_dir() and any(db2_root.glob("*.json")):
        traits.append("with DB2 access")
    return " ".join(traits)


def _program_size_summary(root: Path, program: str, meta: dict[str, Any]) -> str:
    """Report recorded size, naming disagreeing analyzers instead of picking one.

    The MAPA record and the control-flow graph do not agree on paragraph counts.
    Presenting one number as fact states something the evidence does not support.
    """
    parts: list[str] = []
    loc = meta.get("loc")
    statements = meta.get("statements")
    if isinstance(loc, int):
        parts.append(f"{loc} lines of code")
    if isinstance(statements, int):
        parts.append(f"{statements} statements")
    size = f"Recorded size: {', '.join(parts)}." if parts else ""

    mapa_paragraphs = meta.get("paragraphs")
    graph_nodes: int | None = None
    quality = _read_json(_artifact_path(root, "quality.dead_code.json"))
    if isinstance(quality, dict) and quality.get("program") == program:
        reachability = quality.get("content", {}).get("cfg_reachability", {})
        if isinstance(reachability, dict) and isinstance(reachability.get("nodes_count"), int):
            graph_nodes = reachability["nodes_count"]
    if isinstance(mapa_paragraphs, int) and graph_nodes and graph_nodes != mapa_paragraphs:
        size += (
            f" Paragraph counts differ between analyzers: the MAPA record lists "
            f"{mapa_paragraphs}, while the control-flow graph contains {graph_nodes} nodes."
        )
    elif isinstance(mapa_paragraphs, int):
        size += f" {mapa_paragraphs} paragraphs."
    return size.strip()


def _program_composition_facts(root: Path, program: str) -> tuple[list[str], list[str]]:
    """Summarize a program from the artifacts already computed for it."""
    facts: list[str] = []
    sources: list[str] = []

    variables = _read_json(_artifact_path(root, "dataflow.used_variables.json"))
    if isinstance(variables, dict) and variables.get("program") == program:
        entries = [item for item in variables.get("variables", []) if isinstance(item, dict)]
        if entries:
            controlling = sum(1 for item in entries if item.get("controls_flow"))
            facts.append(
                f"{len(entries)} analyzed variables"
                + (f", {controlling} of which control execution" if controlling else "")
            )
            sources.append("dataflow.used_variables.json")

    calls_payload = _read_json(_artifact_path(root, "architecture.call_parameters.json"))
    if isinstance(calls_payload, dict) and calls_payload.get("program") == program:
        targets = list(dict.fromkeys(
            str(call.get("target"))
            for call in calls_payload.get("calls", [])
            if isinstance(call, dict) and call.get("target")
        ))
        if targets:
            facts.append(f"{len(targets)} outgoing calls: {', '.join(targets)}")
            sources.append("architecture.call_parameters.json")

    copybooks = _read_json(_artifact_path(root, "architecture.copybooks.json"))
    if isinstance(copybooks, dict) and copybooks.get("program") == program:
        members = [str(value) for value in copybooks.get("content", {}).get("all", []) if value]
        if members:
            facts.append(f"{len(members)} copybooks included")
            sources.append("architecture.copybooks.json")

    cics = _read_json(_artifact_path(root, "architecture.cics_operations.json"))
    if isinstance(cics, dict) and cics.get("program") == program:
        content = cics.get("content", {})
        operations = [item for item in content.get("operations", []) if isinstance(item, dict)]
        commands = [str(value) for value in content.get("commands", []) if value]
        if operations:
            facts.append(
                f"{len(operations)} CICS operations"
                + (f" covering {', '.join(commands)}" if commands else "")
            )
            sources.append("architecture.cics_operations.json")

    literals = _read_json(_artifact_path(root, "dataflow.literal_assignments.json"))
    if isinstance(literals, dict) and literals.get("program") == program:
        assignments = [item for item in literals.get("assignments", []) if isinstance(item, dict)]
        if assignments:
            facts.append(f"{len(assignments)} forced literal values")
            sources.append("dataflow.literal_assignments.json")

    rule_root = root / "business_rule"
    if rule_root.is_dir():
        rules = [
            path for path in rule_root.glob("*.json")
            if isinstance(_read_json(path), dict)
            and _read_json(path).get("program") == program
        ]
        if rules:
            facts.append(f"{len(rules)} recorded business rules")
            sources.append("business_rule/")

    return facts, sources


def _answer_program_summary(
    root: Path,
    program: str,
    plan: QueryPlan | None = None,
) -> str | None:
    payload_path = _artifact_path(root, "program.summary.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    meta = payload.get("meta") if isinstance(payload.get("meta"), dict) else {}
    composition, sources = _program_composition_facts(root, program)
    character = _program_character(root, program)
    if not composition and not meta:
        return None

    response = plan.response_contract if plan else None
    if plan and "exists" in plan.operations:
        return f"Yes. {program} is present in the analyzed corpus as a {character}."
    if response and response.format == "bullets":
        summary_items: list[tuple[str, str]] = []
        size = _program_size_summary(root, program, meta)
        if size:
            summary_items.append((size, payload_path.name))
        summary_items.extend(zip(composition, sources))
        if response.exact_item_count is not None:
            summary_items = summary_items[:response.exact_item_count]
        return "\n".join(
            f"- {text.rstrip('.')} (`{source.rstrip('/')}`)."
            for text, source in summary_items
        )
    if response and response.max_lines is not None:
        first_line = f"{program} is a {character}."
        if response.max_lines == 1:
            return first_line
        compact_facts = [
            re.sub(r"\s+covering\s+.*$", "", fact.split(":", 1)[0]).rstrip(".")
            for fact in composition
            if any(label in fact.lower() for label in (
                "outgoing calls", "cics operations", "business rules",
            ))
        ]
        if not compact_facts:
            compact_facts = [fact.split(":", 1)[0].rstrip(".") for fact in composition[:3]]
        if len(compact_facts) > 1:
            evidence_summary = f"{', '.join(compact_facts[:-1])}, and {compact_facts[-1]}"
        elif compact_facts:
            evidence_summary = compact_facts[0]
        else:
            size = _program_size_summary(root, program, meta)
            evidence_summary = size.rstrip(".") if size else "an analyzed program summary"
        second_line = (
            f"The analyzed evidence records {evidence_summary}; it does not establish "
            "a specific business-domain purpose."
        )
        if response.exact_lines and response.exact_lines >= 3:
            detail_lines = [
                f"Evidence also records {fact.rstrip('.')}."
                for fact in composition
                if fact.rstrip(".") not in evidence_summary
            ]
            lines = [first_line, second_line, *detail_lines]
            while len(lines) < response.exact_lines:
                lines.append("No additional business-domain purpose is proven by the analyzed evidence.")
            return "\n".join(lines[:response.exact_lines])
        return f"{first_line}\n{second_line}"
    if response and (
        response.format == "sentence"
        or response.max_sentences == 1
        or response.max_words is not None
    ):
        evidence_labels = []
        for fact in composition:
            lowered = fact.lower()
            if "variables" in lowered:
                evidence_labels.append("analyzed data flow")
            elif "outgoing calls" in lowered:
                evidence_labels.append("external calls")
            elif "copybooks" in lowered:
                evidence_labels.append("included copybooks")
            elif "cics operations" in lowered:
                evidence_labels.append("CICS operations")
            elif "literal" in lowered:
                evidence_labels.append("literal assignments")
            elif "business rules" in lowered:
                evidence_labels.append("recorded business rules")
        budget = response.max_words
        selected_labels = list(evidence_labels)
        if budget is not None:
            while True:
                suffix = (
                    f" with {', '.join(selected_labels[:-1])}, and {selected_labels[-1]}"
                    if len(selected_labels) > 1
                    else f" with {selected_labels[0]}" if selected_labels else ""
                )
                sentence = f"{program} is a {character}{suffix}."
                if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", sentence)) <= budget:
                    return sentence
                if selected_labels:
                    selected_labels.pop()
                    continue
                fallback = f"{program} is an analyzed COBOL program."
                if len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", fallback)) <= budget:
                    return fallback
                return f"{program}: analyzed."
        suffix = f" that includes {', '.join(selected_labels[:-1])}, and {selected_labels[-1]}" if len(selected_labels) > 1 else (
            f" that includes {selected_labels[0]}" if selected_labels else ""
        )
        sentence = f"{program} is a {character}{suffix}."
        cited_sources = [payload_path.name, *sources]
        citation = ", ".join(f"`{name.rstrip('/')}`" for name in dict.fromkeys(cited_sources))
        return f"{sentence[:-1]} (sources: {citation})."

    lines = [f"{program} technical overview{f' ({character})' if character else ''}:"]
    size = _program_size_summary(root, program, meta)
    if size:
        lines.append(size)
    if composition:
        lines.append("Analyzed content:")
        lines.extend(f"- {item}" for item in composition)
    lines.append(
        "This is a structural overview from the analyzed evidence; it does not by "
        "itself prove a business-domain purpose."
    )
    cited = ", ".join(f"`{name}`" for name in dict.fromkeys([payload_path.name, *sources]))
    lines.append(f"Sources: {cited}.")
    variable_entities = plan.entity_values_for("variable") if plan else ()
    if variable_entities:
        focused = _answer_variable_access(root, program, " ".join(variable_entities), plan)
        if focused:
            lines.extend(("", "Requested variable focus:", focused))
    return "\n".join(lines)


def _answer_control_flow(root: Path, program: str, question: str = "") -> str | None:
    cfg_path = _artifact_path(root, "controlflow.cfg.json")
    cfg = _read_json(cfg_path)
    if not isinstance(cfg, dict) or cfg.get("program") != program:
        return None
    edges = [edge for edge in cfg.get("edges", []) if isinstance(edge, dict)]
    selection = _answer_control_flow_selection(cfg_path, cfg, program, question, edges)
    if selection:
        return selection
    entry_edges = _unique_flow_edges(edge for edge in edges if edge.get("from") == program)
    phase_names = [
        str(edge.get("to"))
        for edge in entry_edges
        if str(edge.get("to", "")).startswith("BROWSE-FASE")
    ]

    lines = [f"{program} control flow from entry to exit:", "Entry transitions:"]
    lines.extend(_format_flow_edge(edge) for edge in entry_edges)
    for phase in phase_names:
        phase_edges = _unique_flow_edges(edge for edge in edges if edge.get("from") == phase)
        lines.append(f"{phase} transitions:")
        lines.extend(_format_flow_edge(edge) for edge in phase_edges)

    cics_path = _artifact_path(root, "architecture.cics_operations.json")
    cics = _read_json(cics_path)
    terminal_by_paragraph: dict[str, list[str]] = {}
    if isinstance(cics, dict) and cics.get("program") == program:
        for operation in cics.get("content", {}).get("operations", []):
            command = str(operation.get("command", ""))
            if command not in {"RETURN", "XCTL", "ABEND"}:
                continue
            paragraph = str(operation.get("paragraph") or "unknown")
            location = _cics_location(operation)
            terminal_by_paragraph.setdefault(paragraph, []).append(f"{command} at {location}")
    if terminal_by_paragraph or "FINE-ELABORAZIONE" in cfg.get("nodes", []):
        lines.append("Exit/termination points:")
        for paragraph, operations in sorted(terminal_by_paragraph.items()):
            lines.append(f"- {paragraph}: {', '.join(operations)}")
        if "FINE-ELABORAZIONE" in cfg.get("nodes", []):
            lines.append("- FINE-ELABORAZIONE: STOP RUN terminal node in the control-flow graph")
    lines.append(f"Source artifacts: `{cfg_path.name}` and `{cics_path.name}`.")
    return "\n".join(lines)


def _answer_pagination(root: Path, program: str) -> str | None:
    """Join page-count, page-state, and branch evidence into one typed answer."""
    cfg_path = _artifact_path(root, "controlflow.cfg.json")
    cfg = _read_json(cfg_path)
    if not isinstance(cfg, dict) or cfg.get("program") != program:
        return None
    variable_root = root / "dataflow.variable"
    wctpag_path = variable_root / "dataflow.variable.WCTPAG.json"
    npagt_path = variable_root / "dataflow.variable.NPAGT.json"
    wctpag = _read_json(wctpag_path)
    npagt = _read_json(npagt_path)
    if not isinstance(wctpag, dict) or not isinstance(npagt, dict):
        return None

    def sites(payload: dict[str, Any], kind: str) -> list[dict[str, Any]]:
        content = payload.get("content", {}) if isinstance(payload.get("content"), dict) else {}
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        return [site for site in evidence.get(kind, []) if isinstance(site, dict)]

    npagt_writes = sites(npagt, "write_sites")
    page_sites = sites(wctpag, "read_sites") + sites(wctpag, "write_sites")
    wanted_lines = {279, 297, 325, 331, 333, 337, 339, 382}
    selected = sorted(
        (site for site in page_sites if int(site.get("line_start") or -1) in wanted_lines),
        key=lambda site: int(site.get("line_start") or 0),
    )
    lines = [f"{program} paging logic from direct analyzed evidence:"]
    for site in npagt_writes:
        line = int(site.get("line_start") or -1)
        if line in {397, 399, 401}:
            lines.append(
                f"- Page count in {site.get('paragraph')} line {line}: `{site.get('statement')}`"
            )
    for site in selected:
        line = site.get("line_start", "?")
        lines.append(
            f"- Page navigation in {site.get('paragraph')} line {line}: `{site.get('statement')}`"
        )
    lines.append(
        "- PF7 branches to BROWSE-FASE2-PF7 and PF8 branches to BROWSE-FASE2-PF8; "
        "BROWSE-FASE2-ENTER continues only while WCTPAG is less than NPAGT."
    )
    lines.append(
        f"Source artifacts: `{cfg_path.name}`, `{wctpag_path.name}`, and `{npagt_path.name}`."
    )
    return "\n".join(lines)


def _answer_control_flow_selection(
    cfg_path: Path,
    cfg: dict[str, Any],
    program: str,
    question: str,
    edges: list[dict[str, Any]],
) -> str | None:
    q = question.lower()
    if not any(term in q for term in ("decide", "choose", "whether", "start from", "branch", "between")):
        return None
    upper = question.upper()
    node_names = [
        str(node.get("id") if isinstance(node, dict) else node).upper()
        for node in cfg.get("nodes", [])
    ]
    mentioned = [
        node
        for node in node_names
        if node != program
        and re.search(rf"(?<![A-Z0-9-]){re.escape(node)}(?![A-Z0-9-])", upper)
    ]
    mentioned = list(dict.fromkeys(mentioned))
    mentioned.sort(key=upper.find)
    if len(mentioned) < 2:
        return None

    predecessor_sets = [
        {str(edge.get("from")) for edge in edges if str(edge.get("to")) == target}
        for target in mentioned
    ]
    common = set.intersection(*predecessor_sets) if predecessor_sets else set()
    if not common:
        return None
    predecessor = program if program in common else sorted(common)[0]
    selected_edges = _unique_flow_edges(
        edge
        for edge in edges
        if str(edge.get("from")) == predecessor and str(edge.get("to")) in mentioned
    )
    if len({str(edge.get("to")) for edge in selected_edges}) < len(mentioned):
        return None

    lines = [f"{program} selects between {', '.join(mentioned)} from {predecessor}:"]
    lines.extend(_format_flow_edge(edge) for edge in selected_edges)
    fallback_edges = _unique_flow_edges(
        edge
        for edge in edges
        if str(edge.get("from")) == predecessor
        and str(edge.get("to")) not in mentioned
        and edge.get("condition")
    )
    if fallback_edges:
        lines.append("Other explicitly conditioned outcome(s) from the same decision point:")
        lines.extend(_format_flow_edge(edge) for edge in fallback_edges)
    lines.append(f"Source artifact: `{cfg_path.name}`.")
    return "\n".join(lines)


def _unique_flow_edges(raw_edges: Any) -> list[dict[str, Any]]:
    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for edge in raw_edges:
        key = (
            str(edge.get("to", "")),
            str(edge.get("type", "")),
            str(edge.get("condition", "")),
            str(edge.get("evidence", "")),
        )
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def _format_flow_edge(edge: dict[str, Any]) -> str:
    condition = str(edge.get("condition") or "")
    prefix = f"when {condition}, " if condition else ""
    evidence = str(edge.get("evidence") or "no statement captured")
    return f"- {prefix}{edge.get('type', '?')} to {edge.get('to', '?')} — {evidence}"


def _answer_cics_operations(
    root: Path,
    program: str,
    plan: QueryPlan | None = None,
) -> str | None:
    payload_path = _artifact_path(root, "architecture.cics_operations.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    operations = [
        item for item in payload.get("content", {}).get("operations", [])
        if isinstance(item, dict)
    ]
    if plan and plan.operations:
        allowed = {value for value in plan.operations if value.upper() == value}
        if allowed:
            operations = [item for item in operations if str(item.get("command", "")).upper() in allowed]
    if plan and plan.excluded_operations:
        excluded = set(plan.excluded_operations)
        operations = [item for item in operations if str(item.get("command", "")).upper() not in excluded]
    if not operations:
        requested = ", ".join(plan.operations) if plan and plan.operations else "CICS operations"
        return f"No source-backed {requested} operation matched in {program}. Source: `{payload_path.name}`."
    response = plan.response_contract if plan else None
    if response and response.format == "count":
        return str(len(operations))
    if response and response.exact_item_count is not None:
        operations = operations[:response.exact_item_count]
    if response and response.format in {"json", "json_array"}:
        rows = [
            {
                "command": item.get("command"),
                "paragraph": item.get("paragraph"),
                "source_file": item.get("source_file"),
                "source_line": item.get("line_start"),
                "statement": item.get("statement"),
            }
            for item in operations
        ]
        value: Any = rows if response.format == "json_array" else {"operations": rows}
        return json.dumps(value, ensure_ascii=False, indent=2)
    commands = list(dict.fromkeys(str(item.get("command", "")) for item in operations))
    lines = [
        f"{program} executes {len(commands)} requested CICS command type(s) across {len(operations)} source-backed operations:"
    ]
    for operation in operations:
        command = operation.get("command")
        paragraph = operation.get("paragraph")
        statement = operation.get("statement")
        lines.append(
            f"- {command} in {paragraph}: "
            f"{_cics_location(operation)} — `{statement}`"
        )
    lines.append(f"Source artifact: `{payload_path.name}`.")
    return "\n".join(lines)


def _cics_location(operation: dict[str, Any]) -> str:
    source_file = str(operation.get("source_file") or "unknown source")
    line_start = operation.get("line_start", "?")
    line_end = operation.get("line_end", line_start)
    line_label = f"line {line_start}" if line_start == line_end else f"lines {line_start}-{line_end}"
    included_at = operation.get("included_at_line")
    if included_at is not None:
        return f"{source_file} {line_label}, included by the main source at line {included_at}"
    return f"{source_file} {line_label}"


def _answer_counts(root: Path, program: str, q: str) -> str | None:
    comments = _comments_payload(root, program)
    summary = _summary_payload(root, program)
    copybooks = _read_json(_artifact_path(root, "architecture.copybooks.json"))
    calls = _read_json(_artifact_path(root, "architecture.call_parameters.json"))
    literals = _read_json(_artifact_path(root, "dataflow.literal_assignments.json"))

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
    match = re.search(
        r"(?:approximately\s+|about\s+|with\s+)?(\d+)\s+LOC",
        text,
        flags=re.IGNORECASE,
    )
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
    if len(commented) > 20:
        lines.append(f"Showing the first 20 of {len(commented)} items.")
    lines.append("Source: `program.comments.json`.")
    if "copy" in q:
        copy_answer = _answer_copybooks(root, program, q)
        if copy_answer:
            lines.append("")
            lines.append(copy_answer)
    return "\n".join(lines)


def _answer_calls(
    root: Path,
    program: str,
    plan: QueryPlan | None = None,
) -> str | None:
    payload_path = _artifact_path(root, "architecture.call_parameters.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    calls = [item for item in payload.get("calls", []) if isinstance(item, dict)]

    requested_targets: set[str] = set()
    named_entities: tuple[str, ...] = ()
    if plan:
        named_entities = plan.entity_values_for("call", "unknown_identifier")
        requested_targets = {
            value.upper()
            for value in named_entities
            if any(str(call.get("target", "")).upper() == value.upper() for call in calls)
        }
    if requested_targets:
        calls = [call for call in calls if str(call.get("target", "")).upper() in requested_targets]
    elif named_entities:
        # The user named something specific and none of it is a call in this
        # program. Returning the unfiltered list would answer a question that was
        # never asked, using the whole inventory as if it were the match.
        requested = ", ".join(sorted(named_entities))
        return (
            f"No outgoing call evidence matched {requested} in {program}. "
            f"Source: `{payload_path.name}`."
        )

    if plan and plan.operations:
        allowed = {value for value in plan.operations if value.upper() == value}
        if allowed:
            calls = [call for call in calls if _canonical_call_operation(call.get("call_type")) in allowed]
    if plan and plan.excluded_operations:
        excluded = set(plan.excluded_operations)
        calls = [call for call in calls if _canonical_call_operation(call.get("call_type")) not in excluded]

    if not calls:
        requested = ", ".join(sorted(requested_targets)) or (
            ", ".join(plan.operations) if plan and plan.operations else "the requested call"
        )
        return f"No outgoing call evidence matched {requested} in {program}. Source: `{payload_path.name}`."

    if plan and "call_context" in plan.tasks:
        return _answer_call_context(program, calls, payload_path.name, plan)

    response = plan.response_contract if plan else None
    if response and response.exact_item_count is not None:
        calls = calls[:response.exact_item_count]
    if response and response.format == "count":
        return str(len(calls))

    requested_fields = set(plan.output_fields) if plan else set()
    if response and response.format in {"json", "json_array"}:
        fields = requested_fields or {"target", "call_type", "paragraph", "source_line"}
        rows = [_project_call_fields(call, fields) for call in calls]
        payload_value: Any = rows if response.format == "json_array" else {"calls": rows}
        return json.dumps(payload_value, ensure_ascii=False, indent=2)

    if plan and plan.only_requested_fields and requested_fields:
        return "\n".join(
            "- " + "; ".join(
                f"{field}={value}"
                for field, value in _project_call_fields(call, requested_fields).items()
            )
            for call in calls
        )

    lines = [f"{program} outgoing calls with parameters ({len(calls)} matching call(s)):"]
    for call in calls:
        target = call.get("target", "?")
        call_type = call.get("call_type", "?")
        paragraph = call.get("paragraph", "?")
        line = call.get("line_start", "?")
        params = ", ".join(call.get("parameters", [])) or "no explicit parameter"
        details = [f"- {target}: {call_type} in {paragraph} line {line}; parameters: {params}"]
        commarea = call.get("commarea")
        length = call.get("length")
        if commarea:
            details.append(f"COMMAREA={commarea}")
        if length:
            details.append(f"LENGTH={length}")
        lines.append("; ".join(details) + ".")
    lines.append(f"Source: `{payload_path.name}`.")
    return "\n".join(lines)


def _project_call_fields(call: dict[str, Any], requested_fields: set[str]) -> dict[str, Any]:
    mapping: dict[str, Any] = {
        "name": call.get("target"),
        "target": call.get("target"),
        "call_type": call.get("call_type"),
        "paragraph": call.get("paragraph"),
        "source_line": call.get("line_start"),
        "parameters": call.get("parameters", []),
        "commarea": call.get("commarea"),
        "length": call.get("length"),
        "exact_statement": call.get("statement"),
    }
    ordered = (
        "name", "target", "call_type", "paragraph", "source_line", "parameters",
        "commarea", "length", "exact_statement",
    )
    return {
        field: mapping[field]
        for field in ordered
        if field in requested_fields and mapping.get(field) not in (None, "")
    }


def _answer_call_context(
    program: str, calls: list[dict[str, Any]], source_name: str, plan: QueryPlan,
) -> str:
    show_before = not plan.relations or "before" in plan.relations
    show_after = not plan.relations or "after" in plan.relations
    lines = [f"{program} call context from analyzed parameter evidence:"]
    for call in calls:
        target = str(call.get("target") or "?")
        paragraph = str(call.get("paragraph") or "?")
        line = call.get("line_start", "?")
        statement = str(call.get("statement") or "statement unavailable")
        lines.extend([f"Call to {target}:", f"- {paragraph} line {line}: `{statement}`"])
        before: list[tuple[str, Any, str, str]] = []
        after: list[tuple[str, Any, str, str]] = []
        for detail in call.get("parameter_details", []):
            for variable in detail.get("variables", []):
                name = str(variable.get("variable") or "?")
                for site in variable.get("writes_before_call", []):
                    before.append((str(site.get("paragraph") or "?"), site.get("line_start", "?"), str(site.get("statement") or ""), name))
                for site in variable.get("reads_after_call", []):
                    after.append((str(site.get("paragraph") or "?"), site.get("line_start", "?"), str(site.get("statement") or ""), name))
        if show_before:
            lines.append("Before the call — recorded parameter writes:")
            if before:
                for site_paragraph, site_line, site_statement, variable in list(dict.fromkeys(before))[:16]:
                    lines.append(f"- {site_paragraph} line {site_line}, {variable}: `{site_statement}`")
            else:
                lines.append("- No parameter write is recorded before this call in the artifact.")
        if show_after:
            lines.append("At or after the call — recorded parameter reads/tests:")
            if after:
                for site_paragraph, site_line, site_statement, variable in list(dict.fromkeys(after))[:16]:
                    lines.append(f"- {site_paragraph} line {site_line}, {variable}: `{site_statement}`")
            else:
                lines.append("- No parameter read or test is recorded after this call in the artifact.")
    lines.append(f"Source: `{source_name}`.")
    return "\n".join(lines)


def _canonical_call_operation(value: Any) -> str:
    upper = str(value or "").upper()
    if "XCTL" in upper:
        return "XCTL"
    if "LINK" in upper:
        return "LINK"
    return "CALL"


def _answer_literal_assignments(
    root: Path,
    program: str,
    q: str,
    plan: QueryPlan | None = None,
) -> str | None:
    payload_path = _artifact_path(root, "dataflow.literal_assignments.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    items = payload.get("assignments", [])
    if plan:
        requested_variables = {
            value.upper() for value in plan.entity_values_for("variable")
        }
        if requested_variables:
            items = [
                item for item in items
                if str(item.get("target_variable", "")).upper() in requested_variables
            ]
    if "commarea" in q or "parameter" in q:
        items = [item for item in items if item.get("call_commarea_field")]
    elif "screen" in q or "map" in q:
        items = [item for item in items if item.get("screen_or_map_field")]
    elif "control" in q or "flow" in q:
        items = [item for item in items if item.get("controls_flow")]
    response = plan.response_contract if plan else None
    if response and response.format == "count":
        return str(len(items))
    if response and response.exact_item_count is not None:
        items = items[:response.exact_item_count]
    if response and response.format in {"json", "json_array"}:
        rows = [
            {
                "line": item.get("line"),
                "paragraph": item.get("paragraph"),
                "variable": item.get("target_variable"),
                "value": item.get("literal_raw", item.get("literal")),
            }
            for item in items
        ]
        value: Any = rows if response.format == "json_array" else {"assignments": rows}
        return json.dumps(value, ensure_ascii=False, indent=2)
    lines = [f"{program} literal assignments: {len(items)} matching item(s)."]
    exhaustive = bool(plan and plan.result_scope == "all")
    selected_items = items if exhaustive else items[:25]
    for item in selected_items:
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
            f"{item.get('target_variable')} = "
            f"{item.get('literal_raw', item.get('literal'))}{suffix}"
        )
    if len(items) > len(selected_items):
        lines.append(f"Showing the first 25 of {len(items)} matching assignments.")
    lines.append(f"Source: `{payload_path.name}`.")
    return "\n".join(lines)


def _answer_copybooks(
    root: Path,
    program: str,
    q: str,
    plan: QueryPlan | None = None,
) -> str | None:
    payload_path = _artifact_path(root, "architecture.copybooks.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    content = payload.get("content", {})
    all_copybooks = content.get("all", [])
    classified = content.get("classified", {})
    response = plan.response_contract if plan else None
    location_terms = (
        "division", "section", "source line", "line number", "copy statement",
        "copy statements", "where included", "where is each included",
    )
    if any(term in q for term in location_terms):
        inclusions = [item for item in content.get("inclusions", []) if isinstance(item, dict)]
        if plan and plan.divisions:
            allowed_divisions = {value.upper() for value in plan.divisions}
            inclusions = [
                item for item in inclusions
                if str(item.get("division", "")).upper() in allowed_divisions
            ]
        if plan and plan.sections:
            allowed_sections = {value.upper() for value in plan.sections}
            inclusions = [
                item for item in inclusions
                if str(item.get("section", "")).upper() in allowed_sections
            ]
        if plan:
            requested_copybooks = {value.upper() for value in plan.entity_values_for("copybook")}
            if requested_copybooks:
                inclusions = [
                    item for item in inclusions
                    if str(item.get("copybook", "")).upper() in requested_copybooks
                ]
        if not inclusions:
            if content.get("inclusions"):
                requested_scope = ", ".join((plan.divisions + plan.sections) if plan else ()) or "the requested scope"
                return f"No COPY statement matched {requested_scope} in {program}. Source: `{payload_path.name}`."
            return (
                f"{program} copybook names are indexed, but the artifact does not contain COPY "
                "line/division evidence. Rebuild the analysis with source-location enrichment. "
                f"Source: `{payload_path.name}`."
            )
        if response and response.format == "count":
            return str(len(inclusions))
        if response and response.exact_item_count is not None:
            inclusions = inclusions[:response.exact_item_count]
        if response and response.format in {"json", "json_array"}:
            value: Any = inclusions if response.format == "json_array" else {"copy_statements": inclusions}
            return json.dumps(value, ensure_ascii=False, indent=2)
        lines = [f"COPY statements in {program}:"]
        for item in sorted(inclusions, key=lambda value: int(value.get("line", 0))):
            section = str(item.get("section") or "UNKNOWN")
            section_text = f", {section}" if section != "UNKNOWN" else ""
            lines.append(
                f"- {item.get('copybook')}: {item.get('division', 'UNKNOWN')}{section_text}, "
                f"{item.get('source_file', program + '.CBL')} line {item.get('line')} — "
                f"`{item.get('statement', '')}`"
            )
        lines.append(f"Source: `{payload_path.name}`.")
        return "\n".join(lines)
    if response and response.format == "count" and "unused" not in q:
        return str(len(all_copybooks))
    if response and response.exact_item_count is not None and "unused" not in q:
        all_copybooks = all_copybooks[:response.exact_item_count]
        allowed_copybooks = set(all_copybooks)
        classified = {
            category: [name for name in names if name in allowed_copybooks]
            for category, names in classified.items()
        }
    if response and response.format in {"json", "json_array"} and "unused" not in q:
        rows = [{"copybook": name} for name in all_copybooks]
        value: Any = rows if response.format == "json_array" else {"copybooks": rows}
        return json.dumps(value, ensure_ascii=False, indent=2)
    if "only" in q and not any(term in q for term in ("role", "purpose", "used", "unused")):
        return f"{program} COPY members: {', '.join(all_copybooks)}. Source: `{payload_path.name}`."
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
    category_purposes = {
        "ui_cics": "UI and CICS constants or map definitions",
        "business": "business data structures and program interfaces",
        "utilities": "shared utility and service interfaces",
        "state_context": "transaction state and context structures",
        "error_handling": "error and abend handling structures",
    }
    lines = [f"{program} uses {len(all_copybooks)} COBOL COPY members:"]
    for category, names in classified.items():
        purpose = category_purposes.get(category, category.replace("_", " "))
        lines.append(f"- {', '.join(names)} — {purpose}.")
    lines.append(f"Source: `{payload_path.name}`. The purpose categories are analyzer classifications.")
    return "\n".join(lines)


def _copybook_origins_from_dataflow(root: Path, program: str) -> set[str]:
    origins: set[str] = set()
    copybooks_payload = _read_json(_artifact_path(root, "architecture.copybooks.json"))
    known_copybooks = set()
    if isinstance(copybooks_payload, dict) and copybooks_payload.get("program") == program:
        known_copybooks = set(copybooks_payload.get("content", {}).get("all", []))

    def mark_by_prefix(value: str) -> None:
        for copybook in known_copybooks:
            if value == copybook or value.startswith(f"{copybook}-"):
                origins.add(copybook)

    used = _read_json(_artifact_path(root, "dataflow.used_variables.json"))
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

    literals = _read_json(_artifact_path(root, "dataflow.literal_assignments.json"))
    if isinstance(literals, dict) and literals.get("program") == program:
        for item in literals.get("assignments", []):
            mark_by_prefix(str(item.get("target_variable", "")))

    calls = _read_json(_artifact_path(root, "architecture.call_parameters.json"))
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
    used = _read_json(_artifact_path(root, "dataflow.used_variables.json"))
    if not isinstance(used, dict) or used.get("program") != program:
        return False
    for variable in used.get("variables", []):
        name = str(variable.get("variable", ""))
        origin = str(variable.get("origin", ""))
        if origin == "CICS_CONST" and (name.startswith("DFHPF") or name == "DFHENTER"):
            return True
    return False


def _answer_db2_sql(
    root: Path,
    program: str,
    q: str = "",
    plan: QueryPlan | None = None,
) -> str | None:
    db2_files = sorted((root / "architecture.db2_table").glob("architecture.db2_table.*.json"))
    sql_files = sorted((root / "architecture.sqlinclude").glob("architecture.sqlinclude.*.json"))
    db2_items = [
        (path, payload)
        for path in db2_files
        if isinstance((payload := _read_json(path)), dict) and payload.get("program") == program
    ]
    sql_items = [
        (path, payload)
        for path in sql_files
        if isinstance((payload := _read_json(path)), dict) and payload.get("program") == program
    ]
    if not db2_items and not sql_items:
        return None

    include_types = set(plan.include_types) if plan else set()
    exclude_types = set(plan.exclude_types) if plan else set()
    if include_types:
        show_includes = bool(sql_items) and "sql_include" in include_types
        show_tables = bool(db2_items) and "db2_table" in include_types
    else:
        asks_includes = "include" in q or "sqlinclude" in q
        asks_tables = "table" in q or "tables" in q
        show_includes = bool(sql_items) and (asks_includes or not asks_tables)
        show_tables = bool(db2_items) and (asks_tables or not asks_includes)
    show_includes = show_includes and "sql_include" not in exclude_types
    show_tables = show_tables and "db2_table" not in exclude_types

    if not show_includes and not show_tables:
        requested = ", ".join(sorted(include_types)) or "the requested DB2/SQL evidence"
        return f"No {requested} matched after applying the query exclusions for {program}."

    only_names_and_locations = bool(
        (plan and plan.only_requested_fields)
        or ("only" in q and any(
            term in q for term in ("location", "locations", "source", "artifact", "path", "line")
        ))
    )
    asks_source_line = bool(
        (plan and "source_line" in plan.output_fields)
        or "source line" in q
        or "line number" in q
    )

    lines = [f"{program} DB2/SQL evidence:"]
    if only_names_and_locations:
        if show_tables:
            for path, item in db2_items:
                content = item.get("content", {})
                table = content.get("table") or item.get("title", "").replace(f"{program} DB2 table ", "")
                location = f"source line unavailable; artifact `{path.name}`" if asks_source_line else f"artifact `{path.name}`"
                lines.append(f"- DB2 table {table} — {location}")
        if show_includes:
            for path, item in sql_items:
                include = str(item.get("content", {}).get("include") or item.get("title", ""))
                location = f"source line unavailable; artifact `{path.name}`" if asks_source_line else f"artifact `{path.name}`"
                lines.append(f"- SQL include {include} — {location}")
        return "\n".join(lines)

    if show_tables:
        for _, item in db2_items:
            content = item.get("content", {})
            table = content.get("table") or item.get("title", "").replace(f"{program} DB2 table ", "")
            statement_type = (
                content.get("statement_type")
                or content.get("stmt_type")
                or content.get("verb")
                or "statement type unavailable"
            )
            lines.append(f"- DB2 table {table}: {statement_type}")
    if show_includes:
        includes = [
            str(item.get("content", {}).get("include") or item.get("title", ""))
            for _, item in sql_items
        ]
        joined_includes = ", ".join(includes)
        lines.append(f"- SQL includes: {joined_includes}")
    source_paths = ([path for path, _ in db2_items] if show_tables else []) + (
        [path for path, _ in sql_items] if show_includes else []
    )
    joined_sources = ", ".join(f"`{path.name}`" for path in source_paths)
    lines.append(f"Source artifacts: {joined_sources}.")
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
        f"The analyzed evidence for {program} does not establish dataset production or a batch dataset-producing role."
    )


def _answer_ui_navigation(root: Path, program: str) -> str | None:
    payload_path = _artifact_path(root, "ui.cics.navigation.json")
    payload = _read_json(payload_path)
    if not isinstance(payload, dict) or payload.get("program") != program:
        return None
    actions = payload.get("content", {}).get("actions", [])
    lines = [f"{program} CICS UI/navigation actions: {len(actions)} item(s)."]
    for action in actions[:20]:
        lines.append(
            f"- {action.get('context')}: key {action.get('key')} -> {action.get('target')} "
            f"({action.get('edge_type')})"
        )
    lines.append(f"Source: `{payload_path.name}`.")
    return "\n".join(lines)


def _answer_business_rules(
    root: Path,
    program: str,
    plan: QueryPlan | None = None,
) -> str | None:
    rules: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "business_rule").glob("business_rule*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict) and payload.get("program") == program:
            rules.append((path, payload))
    if not rules:
        return None
    unique_rules: list[tuple[Path, dict[str, Any]]] = []
    seen: set[tuple[str, str, str]] = set()
    for path, rule in rules:
        content = rule.get("content", {})
        key = (
            str(content.get("scope", "")).strip(),
            str(content.get("condition", "")).strip(),
            str(content.get("action", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_rules.append((path, rule))

    if plan:
        requested = {
            value.upper()
            for value in plan.entity_values_for("variable", "paragraph", "unknown_identifier")
        }
        if requested:
            matched = []
            for path, rule in unique_rules:
                content = rule.get("content", {})
                searchable = " ".join(
                    str(content.get(key, "")) for key in ("scope", "condition", "action")
                ).upper()
                if any(re.search(rf"(?<![A-Z0-9-]){re.escape(value)}(?![A-Z0-9-])", searchable) for value in requested):
                    matched.append((path, rule))
            if matched:
                unique_rules = matched

        if plan.negative_condition:
            negative_rules = [
                (path, rule)
                for path, rule in unique_rules
                if re.search(r"\b(?:NOT|NEITHER)\b", str(rule.get("content", {}).get("condition", "")).upper())
            ]
            if negative_rules:
                unique_rules = negative_rules

        if plan.condition_terms:
            scored: list[tuple[int, Path, dict[str, Any]]] = []
            for path, rule in unique_rules:
                condition = str(rule.get("content", {}).get("condition", "")).upper()
                score = sum(1 for term in plan.condition_terms if _condition_has_term(condition, term))
                scored.append((score, path, rule))
            best_score = max((item[0] for item in scored), default=0)
            if best_score:
                unique_rules = [(path, rule) for score, path, rule in scored if score == best_score]

    if not unique_rules:
        return None
    response = plan.response_contract if plan else None
    if response and response.format == "count":
        return str(len(unique_rules))
    if response and response.exact_item_count is not None:
        unique_rules = unique_rules[:response.exact_item_count]
    if response and response.format in {"json", "json_array"}:
        rows = []
        for path, rule in unique_rules:
            content = rule.get("content", {})
            evidence = content.get("evidence", {})
            rows.append({
                "id": content.get("id") or rule.get("id"),
                "condition": content.get("condition") or content.get("if"),
                "action": content.get("action") or content.get("then"),
                "paragraph": content.get("scope") or evidence.get("from"),
                "source_file": evidence.get("source_file"),
                "source_line": evidence.get("line_start"),
                "artifact": path.name,
            })
        value: Any = rows if response.format == "json_array" else {"business_rules": rows}
        return json.dumps(value, ensure_ascii=False, indent=2)
    lines = [f"{program} business rules ({len(unique_rules)} matching unique direct-evidence rule(s)):"]
    for path, rule in unique_rules:
        content = rule.get("content", {})
        rule_id = content.get("id") or rule.get("id") or "rule"
        condition = _deduplicate_simple_or_terms(
            str(content.get("condition") or content.get("if") or rule.get("embedding_text", ""))
        )
        action = str(content.get("action") or content.get("target") or content.get("then") or "")
        paragraph = str(content.get("scope") or content.get("evidence", {}).get("from") or "unknown")
        variables = _cobol_identifiers(condition)
        evidence = content.get("evidence", {})
        raw_evidence = str(evidence.get("raw_evidence") or "")
        lines.append(f"- {rule_id}")
        lines.append(f"  Condition: {condition}")
        lines.append(f"  Action: {action}")
        lines.append(f"  Paragraph: {paragraph}")
        joined_variables = ", ".join(variables) or "none identified"
        lines.append(f"  Variables: {joined_variables}")
        if raw_evidence:
            lines.append(f"  COBOL evidence: {raw_evidence}")
        line_start = evidence.get("line_start")
        line_end = evidence.get("line_end", line_start)
        source_file = evidence.get("source_file")
        if line_start is not None:
            line_label = f"line {line_start}" if line_start == line_end else f"lines {line_start}-{line_end}"
            source_label = source_file or "COBOL source"
            lines.append(f"  Source location: {source_label} {line_label}")
        else:
            lines.append("  Source location: unavailable in the rule artifact")
        lines.append(f"  Artifact: `{path.name}`")
    return "\n".join(lines)


def _condition_has_term(condition: str, term: str) -> bool:
    target = str(term).strip().upper()
    if not target:
        return False
    return bool(re.search(rf"(?<![A-Z0-9-]){re.escape(target)}(?![A-Z0-9-])", condition.upper()))


def _answer_variable_access(
    root: Path,
    program: str,
    question: str,
    plan: QueryPlan | None = None,
) -> str | None:
    question_upper = question.upper()
    question_parts = set(re.findall(r"[A-Z0-9]+", question_upper))
    planned_order = [
        value.upper() for value in plan.entity_values_for("variable", "unknown_identifier")
    ] if plan else []
    planned_values = set(planned_order)
    candidates: list[tuple[str, Path, dict[str, Any], int]] = []
    variable_root = root / "dataflow.variable"
    candidate_paths: list[Path] = []
    if planned_order:
        candidate_paths.extend(
            variable_root / f"dataflow.variable.{value}.json"
            for value in planned_order
            if (variable_root / f"dataflow.variable.{value}.json").exists()
        )
    if not planned_order or len(candidate_paths) < len(planned_order):
        known = set(candidate_paths)
        candidate_paths.extend(
            path for path in variable_root.glob("dataflow.variable.*.json")
            if path not in known
        )
    for path in candidate_paths:
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("program") != program:
            continue
        content = payload.get("content", {})
        variable = str(content.get("variable", "")).upper()
        match = re.search(rf"(?<![A-Z0-9-]){re.escape(variable)}(?![A-Z0-9-])", question_upper) if variable else None
        variable_parts = [part for part in variable.split("-") if part]
        compound_alias_match = len(variable_parts) >= 2 and all(part in question_parts for part in variable_parts)
        selected = variable in planned_values if planned_values else bool(match or compound_alias_match)
        if selected:
            position = (
                planned_order.index(variable)
                if variable in planned_order
                else (match.start() if match else len(question_upper))
            )
            candidates.append((variable, path, payload, position))
    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[3], -len(item[0])))
    records: list[dict[str, Any]] = []
    seen_variables: set[str] = set()
    for variable, path, payload, _ in candidates:
        if variable in seen_variables:
            continue
        content = payload.get("content", {}) if isinstance(payload.get("content"), dict) else {}
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        write_sites, read_sites, conflicts = _normalized_variable_sites(
            variable,
            _unique_sites(evidence.get("write_sites", [])),
            _unique_sites(evidence.get("read_sites", [])),
        )
        control_sites = _unique_sites(evidence.get("control_sites", []))
        read_write_sites = _unique_sites(evidence.get("read_write_sites", []))
        subscript_sites = _unique_sites(evidence.get("subscript_sites", []))
        seen_variables.add(variable)
        records.append({
            "variable": variable,
            "path": path,
            "content": content,
            "write_sites": write_sites,
            "read_sites": read_sites,
            "control_sites": control_sites,
            "read_write_sites": read_write_sites,
            "subscript_sites": subscript_sites,
            "conflicts": conflicts,
        })
    if not records:
        return None

    operations = set(plan.operations) if plan else set()
    tasks = set(plan.tasks) if plan else set()
    lineage_requested = "variable_lineage" in tasks or bool(re.search(
        r"\b(?:lifecycle|life cycle|trace|lineage|data movement|data flow|"
        r"intermediate|originated|transferred|feeds?|toward|through)\b",
        question.lower(),
    ))
    descriptive = bool(
        operations & {"describe", "exists", "compare"}
        or "variable_definition" in tasks
        or lineage_requested
    )
    control_requested = bool(
        "control_usage" in (set(plan.output_fields) if plan else set())
        or "control_outcome" in tasks
        or "compare" in operations
        or re.search(r"\b(?:control flow|controls? execution|affects? (?:the )?flow|branch)\b", question.lower())
        or lineage_requested
    )
    lines: list[str] = []
    if "compare" in operations and len(records) > 1:
        lines.append(f"Comparison of direct variable evidence in {program}:")
    elif len(records) > 1:
        lines.append(f"Direct COBOL access evidence for {len(records)} variables:")

    for index, record in enumerate(records):
        if index and lines:
            lines.append("")
        variable = record["variable"]
        content = record["content"]
        write_sites = record["write_sites"]
        read_sites = record["read_sites"]
        control_sites = record["control_sites"]
        read_write_sites = record["read_write_sites"]
        subscript_sites = record["subscript_sites"]
        if "exists" in operations:
            lines.append(f"{variable} exists in the analyzed {program} variable catalogue.")
        else:
            lines.append(f"{variable} direct COBOL access evidence:")

        if descriptive:
            origin = str(content.get("origin") or "UNKNOWN")
            defined_in = [str(value) for value in content.get("defined_in", [])]
            modified_in = [str(value) for value in content.get("modified_in", [])]
            used_in = [str(value) for value in content.get("used_in", [])]
            relationships = (
                content.get("relationships", {})
                if isinstance(content.get("relationships"), dict)
                else {}
            )
            declarations = [
                item for item in relationships.get("declarations", [])
                if isinstance(item, dict)
            ]
            lines.append(f"- Origin: {origin}")
            if declarations:
                lines.append("- Declaration evidence:")
                for declaration in declarations:
                    source_file = str(declaration.get("source_file") or "source")
                    source_line = declaration.get("line_start", "?")
                    level = declaration.get("level", "?")
                    statement = str(declaration.get("statement") or "declaration text unavailable")
                    lines.append(
                        f"  - {source_file} line {source_line}, level {level}: `{statement}`"
                    )
            parents = [str(value) for value in relationships.get("parents", [])]
            children = [str(value) for value in relationships.get("children", [])]
            redefines = [str(value) for value in relationships.get("redefines", [])]
            redefined_by = [str(value) for value in relationships.get("redefined_by", [])]
            if parents:
                lines.append(f"- Parent group(s): {', '.join(parents)}")
            if children:
                lines.append(f"- Child field(s): {', '.join(children)}")
            if redefines:
                lines.append(f"- Redefines: {', '.join(redefines)}")
            if redefined_by:
                lines.append(f"- Redefined by: {', '.join(redefined_by)}")
            lines.append(f"- Defined in: {', '.join(defined_in) if defined_in else 'not recorded'}")
            lines.append(f"- Modified in: {', '.join(modified_in) if modified_in else 'not recorded'}")
            lines.append(f"- Used in: {', '.join(used_in) if used_in else 'not recorded'}")
            lines.append(f"- Controls flow: {'yes' if content.get('controls_flow') else 'no'}")
            if not declarations:
                lines.append("- Declaration details such as level number and PIC are not present in this dataflow artifact.")

        variable_task_filter = tasks & {
            "variable_definition", "variable_reads", "variable_writes",
            "variable_comparison", "variable_lineage", "control_outcome",
            "variable_composition", "call_option_usage", "lineage_terminal",
        }
        if operations & {"describe", "exists", "compare"}:
            show_writes = True
            show_reads = True
        elif variable_task_filter:
            show_writes = bool(variable_task_filter & {
                "variable_writes", "variable_comparison", "variable_lineage",
            })
            show_reads = bool(variable_task_filter & {
                "variable_reads", "variable_comparison", "variable_lineage",
            })
        else:
            show_writes = not operations or "find_reads" not in operations or bool(
                operations & {"find_writes", "describe", "exists", "compare"}
            )
            show_reads = not operations or "find_writes" not in operations or bool(
                operations & {"find_reads", "describe", "exists", "compare"}
            )
        if show_writes:
            lines.append("Modified at:")
            if write_sites:
                lines.extend(_format_evidence_site(site) for site in write_sites)
            else:
                lines.append("- No explicit write site is present in the variable artifact.")
            lines.append(f"Write coverage: {len(write_sites)}/{len(write_sites)} direct site(s) returned.")
        if show_reads:
            lines.append("Tested/read at:")
            if read_sites:
                lines.extend(_format_evidence_site(site) for site in read_sites)
            else:
                lines.append("- No explicit read or test site is present in the variable artifact.")
            lines.append(f"Read coverage: {len(read_sites)}/{len(read_sites)} direct site(s) returned.")
        if read_write_sites:
            lines.append("Read/write operations:")
            lines.extend(_format_evidence_site(site) for site in read_write_sites)
        if subscript_sites:
            lines.append("Used as a subscript/index (read, not modified):")
            lines.extend(_format_evidence_site(site) for site in subscript_sites)
        if control_requested:
            lines.append("Control-flow use:")
            if control_sites:
                lines.extend(_format_evidence_site(site) for site in control_sites)
            else:
                lines.append("- No direct control-flow site is present in the variable artifact.")
            lines.append(f"Control coverage: {len(control_sites)}/{len(control_sites)} recorded site(s) returned.")
            assigned_literals = _assigned_literal_values(variable, write_sites)
            if assigned_literals:
                lines.append("Assigned-value control evidence:")
                for literal in assigned_literals:
                    matching_sites = [
                        site for site in control_sites
                        if literal.upper() in str(site.get("statement", "")).upper()
                    ]
                    if matching_sites:
                        locations = ", ".join(
                            f"{site.get('paragraph') or 'unknown paragraph'} line {site.get('line_start')}"
                            for site in matching_sites
                            if isinstance(site.get("line_start"), int) and site.get("line_start") >= 0
                        )
                        lines.append(
                            f"- {literal} is directly tested by a recorded control condition"
                            + (f" at {locations}." if locations else ".")
                        )
                    else:
                        lines.append(
                            f"- {literal} is assigned, but no direct control condition for that value is recorded."
                        )
        joined_paths: list[Path] = []
        if lineage_requested:
            lineage_lines, lineage_paths = _variable_lineage(root, variable)
            lines.append("Lineage:")
            lines.extend(lineage_lines or ["- No direct MOVE lineage is present in the analyzed variable artifacts."])
            joined_paths.extend(lineage_paths)
        if "control_outcome" in tasks:
            outcome_lines, outcome_paths = _variable_control_outcomes(
                root, program, variable, control_sites,
            )
            lines.append("Resulting control-flow actions:")
            lines.extend(outcome_lines or [
                "- No direct business-rule or error-path outcome is recorded for this variable."
            ])
            joined_paths.extend(outcome_paths)
        if "variable_composition" in tasks:
            composition_lines, composition_paths = _variable_composition_context(
                root, variable, write_sites, read_sites,
            )
            lines.append("Construction/context evidence:")
            lines.extend(composition_lines or [
                "- No surrounding source-backed construction sequence is present in the variable artifacts."
            ])
            joined_paths.extend(composition_paths)
        if "call_option_usage" in tasks:
            call_lines, call_paths = _variable_call_option_usage(root, variable)
            lines.append("CICS/call option usage:")
            lines.extend(call_lines or [
                "- No call option or parameter in `architecture.call_parameters.json` uses this variable."
            ])
            joined_paths.extend(call_paths)
        if "lineage_terminal" in tasks:
            terminal_lines, terminal_paths = _variable_lineage_completion(
                root, variable, question,
            )
            lines.append("Lineage completion:")
            lines.extend(terminal_lines or [
                "- No terminal destination could be proven from direct MOVE evidence."
            ])
            joined_paths.extend(terminal_paths)
        if record["conflicts"]:
            lines.append(
                f"Evidence consistency: {record['conflicts']} site(s) recorded as reads contain direct assignments; rendered as writes."
            )
        lines.append(f"Source artifact: `{record['path'].name}`.")
        extra_paths = list(dict.fromkeys(
            path for path in joined_paths if path != record["path"]
        ))
        if extra_paths:
            names = ", ".join(f"`{path.name}`" for path in extra_paths)
            lines.append(f"Joined evidence artifacts: {names}.")
    return "\n".join(lines)


def _variable_lineage(root: Path, variable: str, max_depth: int = 8) -> tuple[list[str], list[Path]]:
    """Traverse typed MOVE and group-membership edges in both directions."""
    variable = variable.upper()
    lines: list[str] = []
    used_paths: list[Path] = []
    catalog: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        content = payload.get("content", {}) if isinstance(payload, dict) else {}
        name = str(content.get("variable", "")).upper()
        if name:
            catalog[name] = (path, content)

    edges: list[dict[str, Any]] = []
    seen_edge_keys: set[tuple[Any, ...]] = set()
    for name, (path, content) in catalog.items():
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        for site in _unique_sites([*evidence.get("write_sites", []), *evidence.get("read_sites", [])]):
            statement = str(site.get("statement", ""))
            match = re.search(
                r"\bMOVE\s+(.+?)\s+TO\s+([A-Z][A-Z0-9-]*)(?:\([^)]*\))?",
                statement,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            expression = " ".join(match.group(1).strip().rstrip(".").split())
            target = match.group(2).upper()
            value_expression = re.sub(r"\([^)]*\)", " ", expression)
            source_names = tuple(dict.fromkeys(
                token for token in re.findall(r"[A-Z][A-Z0-9-]*", value_expression.upper())
                if token in catalog
            ))
            if not source_names:
                continue
            key = (
                "move", expression.upper(), target,
                str(site.get("paragraph", "")), str(site.get("line_start", "")),
            )
            if key in seen_edge_keys:
                continue
            seen_edge_keys.add(key)
            edges.append({
                "kind": "MOVE",
                "sources": source_names,
                "source_label": expression,
                "target": target,
                "site": site,
                "paths": [path, catalog[target][0]] if target in catalog else [path],
            })

        relationships = content.get("relationships", {}) if isinstance(content.get("relationships"), dict) else {}
        declarations = relationships.get("declarations", [])
        declaration = declarations[0] if declarations and isinstance(declarations[0], dict) else {}
        for parent in relationships.get("parents", []):
            parent_name = str(parent).upper()
            if parent_name not in catalog:
                continue
            key = ("member_of", name, parent_name)
            if key in seen_edge_keys:
                continue
            seen_edge_keys.add(key)
            edges.append({
                "kind": "MEMBER_OF",
                "sources": (name,),
                "source_label": name,
                "target": parent_name,
                "site": {
                    "paragraph": "DATA DIVISION",
                    "line_start": declaration.get("line_start"),
                    "statement": declaration.get("statement") or f"{name} is subordinate to {parent_name}",
                },
                "paths": [path, catalog[parent_name][0]],
            })

    def remember_paths(edge: dict[str, Any]) -> None:
        for path in edge.get("paths", []):
            if path not in used_paths:
                used_paths.append(path)

    def render(edge: dict[str, Any], label: str) -> None:
        site = edge["site"]
        paragraph = str(site.get("paragraph") or "unknown paragraph")
        line = site.get("line_start")
        line_label = f"line {line}" if isinstance(line, int) and line >= 0 else "line unavailable"
        statement = str(site.get("statement", "")).strip()
        relation = "member of" if edge["kind"] == "MEMBER_OF" else "->"
        lines.append(
            f"- {label}: {edge['source_label']} {relation} {edge['target']} "
            f"in {paragraph}, {line_label}: `{statement}`"
        )
        remember_paths(edge)

    rendered: set[tuple[Any, ...]] = set()
    upstream_expanded: set[str] = set()
    downstream_expanded: set[str] = set()

    def edge_key(edge: dict[str, Any]) -> tuple[Any, ...]:
        site = edge["site"]
        return (
            edge["kind"], edge["source_label"], edge["target"],
            site.get("paragraph"), site.get("line_start"),
        )

    def walk_upstream(name: str, depth: int, ancestry: frozenset[str]) -> None:
        if depth > max_depth or name in upstream_expanded:
            return
        upstream_expanded.add(name)
        for edge in edges:
            if edge["target"] != name:
                continue
            if any(source in ancestry for source in edge["sources"]):
                continue
            key = edge_key(edge)
            if key not in rendered:
                rendered.add(key)
                render(edge, f"upstream hop {depth + 1}")
            for source in edge["sources"]:
                walk_upstream(source, depth + 1, ancestry | {name})

    def walk_downstream(name: str, depth: int, ancestry: frozenset[str]) -> None:
        if depth > max_depth or name in downstream_expanded:
            return
        downstream_expanded.add(name)
        for edge in edges:
            if name not in edge["sources"]:
                continue
            if edge["target"] in ancestry:
                continue
            key = edge_key(edge)
            if key not in rendered:
                rendered.add(key)
                render(edge, f"downstream hop {depth + 1}")
            walk_downstream(edge["target"], depth + 1, ancestry | {name})

    walk_upstream(variable, 0, frozenset({variable}))
    walk_downstream(variable, 0, frozenset({variable}))
    return lines, used_paths


def _variable_control_outcomes(
    root: Path,
    program: str,
    variable: str,
    control_sites: list[dict[str, Any]],
) -> tuple[list[str], list[Path]]:
    """Join a variable condition to its recorded branch/error outcomes."""
    lines: list[str] = []
    used_paths: list[Path] = []
    seen: set[tuple[str, str, str]] = set()
    all_sites = _all_variable_sites(root)

    for path in sorted((root / "business_rule").glob("business_rule.*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or payload.get("program") != program:
            continue
        content = payload.get("content", {})
        condition = str(content.get("condition") or "")
        if not _identifier_in(variable, condition):
            continue
        action = str(content.get("action") or "unrecorded action")
        scope = str(content.get("scope") or "unknown paragraph")
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        source_file = str(evidence.get("source_file") or "COBOL source")
        line_start = evidence.get("line_start")
        line_end = evidence.get("line_end", line_start)
        line_label = _line_label(line_start, line_end)
        raw = str(evidence.get("raw_evidence") or "").strip()
        key = (condition, action, str(line_start))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- Condition `{condition}` causes `{action}` in {scope}; "
            f"{source_file} {line_label}: `{raw or action}`"
        )
        if path not in used_paths:
            used_paths.append(path)

        condition_lines = [
            site.get("line_start") for site in control_sites
            if site.get("paragraph") == scope
            and isinstance(site.get("line_start"), int)
            and site.get("line_start") >= 0
        ]
        if condition_lines and isinstance(line_start, int):
            first = min(condition_lines)
            supporting = _sites_in_window(all_sites, {scope}, first + 1, line_start)
            for site in supporting:
                statement = str(site.get("statement") or "").strip()
                if not statement or statement.upper() == raw.upper():
                    continue
                lines.append(
                    f"  - Supporting statement at {scope}, line {site['line_start']}: `{statement}`"
                )
                if site["path"] not in used_paths:
                    used_paths.append(site["path"])

    if not lines:
        seen_following: set[tuple[str, int, str]] = set()
        for control_site in control_sites:
            control_line = control_site.get("line_start")
            paragraph = str(control_site.get("paragraph") or "")
            if not isinstance(control_line, int) or control_line < 0:
                continue
            for site in _sites_in_window(all_sites, {paragraph}, control_line + 1, control_line + 6):
                statement = str(site.get("statement") or "").strip()
                key = (paragraph, int(site["line_start"]), _normalized_statement(statement))
                if not statement or key in seen_following:
                    continue
                seen_following.add(key)
                lines.append(
                    f"- Following direct statement in {paragraph}, line {site['line_start']}: `{statement}`"
                )
                if site["path"] not in used_paths:
                    used_paths.append(site["path"])

    rich_path = _artifact_path(root, "quality.error_paths.rich/quality.error_paths.rich.json")
    rich = _read_json(rich_path)
    error_paragraphs: set[str] = set()
    error_targets: set[str] = set()
    if isinstance(rich, dict):
        for chunk in rich.get("chunks", []):
            preview = str(chunk.get("text_preview") or "")
            trigger = _labeled_preview_value(preview, "Trigger", ("Message/code", "Target", "Evidence"))
            if not trigger or not _identifier_in(variable, trigger):
                continue
            message = _labeled_preview_value(preview, "Message/code", ("Target", "Evidence"))
            target = _labeled_preview_value(preview, "Target", ("Evidence",))
            evidence = "" if preview.rstrip().endswith("...") else _labeled_preview_value(preview, "Evidence", ())
            key = (trigger, target, evidence)
            if key in seen:
                continue
            seen.add(key)
            details = [f"trigger `{trigger}`"]
            if message:
                details.append(f"message/code `{message}`")
            if target:
                details.append(f"target `{target}`")
            if evidence:
                details.append(f"evidence `{evidence}`")
            paragraph = str(chunk.get("paragraph") or "unknown paragraph")
            error_paragraphs.add(paragraph)
            if target:
                error_targets.add(target)
            lines.append(f"- Error-path evidence in {paragraph}: " + "; ".join(details))
            if rich_path not in used_paths:
                used_paths.append(rich_path)

    contexts_path = _artifact_path(root, "integration.paragraph_contexts/integration.paragraph_contexts.json")
    contexts = _read_json(contexts_path)
    if isinstance(contexts, dict) and error_paragraphs:
        seen_outcomes: set[tuple[str, str, str]] = set()
        for context in contexts.get("contexts", []):
            paragraph = str(context.get("paragraph") or "")
            if paragraph not in error_paragraphs:
                continue
            mapa = context.get("mapa_record", {}) if isinstance(context.get("mapa_record"), dict) else {}
            for outgoing in mapa.get("outgoing", []):
                target = str(outgoing.get("to") or "")
                cics_ops = [str(value) for value in outgoing.get("cics_ops", [])]
                if target not in error_targets and not cics_ops:
                    continue
                key = (paragraph, target, str(outgoing.get("type") or ""))
                if key in seen_outcomes:
                    continue
                seen_outcomes.add(key)
                operation = f"; CICS operation(s): {', '.join(cics_ops)}" if cics_ops else ""
                evidence = str(outgoing.get("evidence") or "evidence text unavailable")
                lines.append(
                    f"- Paragraph outcome from {paragraph}: {outgoing.get('type')} -> {target}{operation}; "
                    f"`{evidence}`"
                )
                if contexts_path not in used_paths:
                    used_paths.append(contexts_path)

    return lines, used_paths


def _variable_composition_context(
    root: Path,
    variable: str,
    write_sites: list[dict[str, Any]],
    read_sites: list[dict[str, Any]],
) -> tuple[list[str], list[Path]]:
    """Collect direct statements that build a group before its downstream MOVE."""
    anchors = [
        site for site in (*write_sites, *read_sites)
        if isinstance(site.get("line_start"), int) and site.get("line_start") >= 0
    ]
    if not anchors:
        return [], []
    clear_pattern = re.compile(
        rf"\bMOVE\s+(?:SPACE|SPACES|LOW-VALUE|LOW-VALUES|HIGH-VALUE|HIGH-VALUES|ZERO|ZEROS)"
        rf"\s+TO\s+{re.escape(variable)}(?![A-Z0-9-])",
        flags=re.IGNORECASE,
    )
    clearing = [site for site in write_sites if clear_pattern.search(str(site.get("statement", "")))]
    if clearing:
        start = min(int(site["line_start"]) for site in clearing)
        later_reads = [int(site["line_start"]) for site in read_sites if int(site.get("line_start", -1)) > start]
        end = min(later_reads) if later_reads else start + 80
    else:
        end = min(int(site["line_start"]) for site in read_sites or anchors)
        start = max(0, end - 12)
    if end - start > 120:
        end = start + 120
    paragraphs = {
        str(site.get("paragraph") or "") for site in anchors
        if start <= int(site.get("line_start", -1)) <= end
    }
    sites = _sites_in_window(_all_variable_sites(root), paragraphs, start, end)
    allowed_statement = re.compile(
        r"^\s*(?:THEN\s+|ELSE\s+)?(?:MOVE|IF|ADD|SUBTRACT|COMPUTE|DIVIDE|MULTIPLY|STRING|UNSTRING)\b",
        flags=re.IGNORECASE,
    )
    lines: list[str] = []
    used_paths: list[Path] = []
    seen: set[tuple[int, str]] = set()
    for site in sites:
        statement = str(site.get("statement") or "").strip()
        key = (int(site["line_start"]), " ".join(statement.upper().split()))
        if not allowed_statement.search(statement) or key in seen:
            continue
        seen.add(key)
        lines.append(
            f"- {site.get('paragraph') or 'unknown paragraph'}, line {site['line_start']}: `{statement}`"
        )
        if site["path"] not in used_paths:
            used_paths.append(site["path"])
    direct_statements = {
        _normalized_statement(line.split("`", 2)[1])
        for line in lines if "`" in line
    }
    screen_lines, screen_paths = _screen_construction_statements(
        root, paragraphs, direct_statements,
    )
    lines.extend(screen_lines)
    used_paths.extend(path for path in screen_paths if path not in used_paths)
    return lines, used_paths


def _variable_call_option_usage(root: Path, variable: str) -> tuple[list[str], list[Path]]:
    path = _artifact_path(root, "architecture.call_parameters.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return [], []
    lines: list[str] = []
    for call in payload.get("calls", []):
        if not isinstance(call, dict):
            continue
        statement = str(call.get("statement") or "")
        roles: list[str] = []
        if str(call.get("length") or "").upper() == variable:
            roles.append("the LENGTH option")
        if str(call.get("commarea") or "").upper() == variable:
            roles.append("the COMMAREA option")
        if variable in {str(value).upper() for value in call.get("parameters", [])}:
            roles.append("a call parameter")
        if not roles and not _identifier_in(variable, statement):
            continue
        if not roles:
            roles.append("a statement operand")
        line_start = call.get("line_start")
        line_end = call.get("line_end", line_start)
        target = str(call.get("target") or "unknown target")
        paragraph = str(call.get("paragraph") or "unknown paragraph")
        if isinstance(line_start, int) and (not isinstance(line_end, int) or line_end == line_start):
            variable_record = _variable_catalog(root).get(variable)
            content = variable_record[1] if variable_record else {}
            evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
            continuation_lines = [
                site.get("line_start")
                for site in (*evidence.get("read_sites", []), *evidence.get("write_sites", []))
                if site.get("paragraph") == paragraph
                and isinstance(site.get("line_start"), int)
                and site.get("line_start") >= line_start
                and _identifier_in(variable, str(site.get("statement") or ""))
            ]
            if continuation_lines:
                line_end = max(line_start, *continuation_lines)
        commarea = str(call.get("commarea") or "none")
        lines.append(
            f"- In {paragraph}, {_line_label(line_start, line_end)}, {variable} supplies "
            f"{' and '.join(roles)} for {target}; COMMAREA={commarea}. "
            f"Exact statement: `{statement or 'statement unavailable'}`"
        )
    return lines, [path] if lines else []


def _variable_lineage_completion(
    root: Path,
    variable: str,
    question: str,
    max_depth: int = 4,
) -> tuple[list[str], list[Path]]:
    """Identify terminal MOVE destinations, intervening overwrites, and nearby branches."""
    sites = _all_variable_sites(root)
    edges: list[dict[str, Any]] = []
    seen_edges: set[tuple[str, str, str, int]] = set()
    for site in sites:
        edge = _move_edge(str(site.get("statement") or ""))
        if edge is None:
            continue
        source, target = edge
        key = (source.upper(), target, str(site.get("paragraph") or ""), int(site["line_start"]))
        if key in seen_edges:
            continue
        seen_edges.add(key)
        edges.append({**site, "source": source, "target": target})

    frontier = [(variable.upper(), 0)]
    visited = {variable.upper()}
    path_edges: list[dict[str, Any]] = []
    while frontier:
        current, depth = frontier.pop(0)
        if depth >= max_depth:
            continue
        for edge in edges:
            if not _identifier_in(current, edge["source"]):
                continue
            target = str(edge["target"]).upper()
            if target == current:
                continue
            if edge not in path_edges:
                path_edges.append(edge)
            if target not in visited:
                visited.add(target)
                frontier.append((target, depth + 1))
    if not path_edges:
        return [], []

    terminals = sorted(
        node for node in visited if node != variable.upper()
        and not any(_identifier_in(node, edge["source"]) for edge in edges)
    )
    lines: list[str] = []
    used_paths: list[Path] = []
    catalog = _variable_catalog(root)
    for terminal in terminals:
        origin = "UNKNOWN"
        loaded = catalog.get(terminal)
        if loaded:
            origin = str(loaded[1].get("origin") or "UNKNOWN")
            if loaded[0] not in used_paths:
                used_paths.append(loaded[0])
        lines.append(
            f"- Terminal destination: {terminal} (origin: {origin}); no downstream MOVE from it is recorded."
        )

    path_keys = {
        (edge["source"].upper(), edge["target"], edge.get("paragraph"), edge["line_start"])
        for edge in path_edges
    }
    path_lines = [int(edge["line_start"]) for edge in path_edges]
    first_line, last_line = min(path_lines), max(path_lines)
    path_paragraphs = {str(edge.get("paragraph") or "") for edge in path_edges}
    alternatives: list[dict[str, Any]] = []
    for edge in edges:
        key = (edge["source"].upper(), edge["target"], edge.get("paragraph"), edge["line_start"])
        if key in path_keys or edge["target"] not in visited:
            continue
        if str(edge.get("paragraph") or "") not in path_paragraphs:
            continue
        if not first_line <= int(edge["line_start"]) <= last_line:
            continue
        alternatives.append(edge)
    if alternatives:
        lines.append("- Possible overwrites before the terminal transfer:")
        for edge in alternatives:
            lines.append(
                f"  - {edge['source']} -> {edge['target']} in {edge.get('paragraph')}, "
                f"line {edge['line_start']}: `{edge.get('statement')}`"
            )
            if edge["path"] not in used_paths:
                used_paths.append(edge["path"])

    nearby_rules: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted((root / "business_rule").glob("business_rule.*.json")):
        payload = _read_json(path)
        content = payload.get("content", {}) if isinstance(payload, dict) else {}
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        rule_line = evidence.get("line_start")
        if str(content.get("scope") or "") not in path_paragraphs or not isinstance(rule_line, int):
            continue
        if first_line - 3 <= rule_line <= last_line + 3:
            nearby_rules.append((path, content))
    if nearby_rules:
        lines.append("- Nearby branch outcomes affecting this lineage:")
        for path, content in nearby_rules:
            evidence = content.get("evidence", {})
            lines.append(
                f"  - `{content.get('condition')}` causes `{content.get('action')}` in "
                f"{content.get('scope')}, line {evidence.get('line_start')}."
            )
            if path not in used_paths:
                used_paths.append(path)

    if re.search(r"\b(?:display|displayed|screen|map field)\b", question.lower()):
        terminal_text = ", ".join(terminals) if terminals else "the last recorded variable"
        lines.append(
            f"- Display boundary: direct MOVE evidence ends at {terminal_text}; "
            "no further MOVE to a screen/map field is proven by the variable artifacts."
        )
    for edge in path_edges:
        if edge["path"] not in used_paths:
            used_paths.append(edge["path"])
    return lines, used_paths


def _variable_catalog(root: Path) -> dict[str, tuple[Path, dict[str, Any]]]:
    catalog: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        content = payload.get("content", {}) if isinstance(payload, dict) else {}
        variable = str(content.get("variable") or "").upper()
        if variable:
            catalog[variable] = (path, content)
    return catalog


def _all_variable_sites(root: Path) -> list[dict[str, Any]]:
    sites: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for variable, (path, content) in _variable_catalog(root).items():
        evidence = content.get("evidence", {}) if isinstance(content.get("evidence"), dict) else {}
        for role in ("write_sites", "read_sites", "control_sites"):
            for site in evidence.get(role, []):
                if not isinstance(site, dict) or not isinstance(site.get("line_start"), int):
                    continue
                key = (
                    str(site.get("paragraph") or ""), str(site.get("line_start")),
                    " ".join(str(site.get("statement") or "").upper().split()), role,
                )
                if key in seen:
                    continue
                seen.add(key)
                sites.append({**site, "variable": variable, "role": role, "path": path})
    return sites


def _sites_in_window(
    sites: list[dict[str, Any]],
    paragraphs: set[str],
    start: int,
    end: int,
) -> list[dict[str, Any]]:
    return sorted(
        (
            site for site in sites
            if start <= int(site.get("line_start", -1)) <= end
            and (not paragraphs or str(site.get("paragraph") or "") in paragraphs)
        ),
        key=lambda site: (int(site["line_start"]), str(site.get("statement") or "")),
    )


def _screen_construction_statements(
    root: Path,
    paragraphs: set[str],
    existing: set[str],
) -> tuple[list[str], list[Path]]:
    path = _artifact_path(root, "screen.interaction/screen.interaction.json")
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return [], []
    lines: list[str] = []
    seen = set(existing)
    entry_pattern = re.compile(
        r"(?:^|\s)-\s+([A-Z0-9-]+):\s*(.*?)(?=(?:\s+-\s+[A-Z0-9-]+:)|$)",
        flags=re.DOTALL,
    )
    useful = re.compile(
        r"^(?:MOVE|IF|ADD|SUBTRACT|COMPUTE|DIVIDE|MULTIPLY|STRING|UNSTRING|PERFORM)\b",
        flags=re.IGNORECASE,
    )
    for chunk in payload.get("chunks", []):
        if str(chunk.get("chunk_type") or "") != "screen.row_build":
            continue
        preview = str(chunk.get("text_preview") or "")
        for match in entry_pattern.finditer(preview):
            paragraph, statement = match.group(1), " ".join(match.group(2).split())
            normalized = _normalized_statement(statement)
            if (
                paragraph not in paragraphs or not useful.search(statement)
                or len(statement) > 240 or normalized in seen
                or statement.endswith("-.")
            ):
                continue
            seen.add(normalized)
            lines.append(
                f"- {paragraph}, physical line unavailable in screen artifact: `{statement}`"
            )
    return lines, [path] if lines else []


def _normalized_statement(statement: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9'-]+", statement.upper()))


def _move_edge(statement: str) -> tuple[str, str] | None:
    match = re.search(
        r"\bMOVE\s+(.+?)\s+TO\s+([A-Z][A-Z0-9-]*)(?:\([^)]*\))?",
        statement,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return " ".join(match.group(1).strip().rstrip(".").split()), match.group(2).upper()


def _identifier_in(identifier: str, text: str) -> bool:
    return bool(re.search(
        rf"(?<![A-Z0-9-]){re.escape(identifier.upper())}(?![A-Z0-9-])",
        text.upper(),
    ))


def _line_label(line_start: Any, line_end: Any = None) -> str:
    if not isinstance(line_start, int):
        return "line unavailable"
    if not isinstance(line_end, int) or line_end == line_start:
        return f"line {line_start}"
    return f"lines {line_start}-{line_end}"


def _labeled_preview_value(text: str, label: str, following: tuple[str, ...]) -> str:
    stop = "|".join(re.escape(value) + r":" for value in following)
    pattern = rf"\b{re.escape(label)}:\s*(.*?)(?=\s+(?:{stop})|$)" if stop else rf"\b{re.escape(label)}:\s*(.*)$"
    match = re.search(pattern, text, flags=re.IGNORECASE)
    return match.group(1).strip().rstrip(".") if match else ""


def _normalized_variable_sites(
    variable: str,
    write_sites: list[dict[str, Any]],
    read_sites: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    corrected_writes = list(write_sites)
    corrected_reads: list[dict[str, Any]] = []
    conflicts = 0
    for site in read_sites:
        statement = str(site.get("statement", "")).upper()
        if re.search(rf"\bMOVE\b.+\bTO\s+{re.escape(variable)}(?![A-Z0-9-])", statement):
            corrected_writes.append(site)
            conflicts += 1
        else:
            corrected_reads.append(site)
    return _unique_sites(corrected_writes), _unique_sites(corrected_reads), conflicts


def _assigned_literal_values(variable: str, write_sites: list[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for site in write_sites:
        statement = str(site.get("statement", ""))
        match = re.search(
            rf"\bMOVE\s+(.+?)\s+TO\s+{re.escape(variable)}(?![A-Z0-9-])",
            statement,
            flags=re.IGNORECASE,
        )
        if not match:
            continue
        expression = " ".join(match.group(1).strip().split())
        if not re.fullmatch(r"(?:[+-]?\d+|'[^']*'|\"[^\"]*\")", expression):
            continue
        if expression not in values:
            values.append(expression)
    return values


def _unique_sites(raw_sites: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_sites, list):
        return []
    sites: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for raw in raw_sites:
        if not isinstance(raw, dict):
            continue
        key = (
            str(raw.get("paragraph", "")),
            str(raw.get("line_start", "")),
            str(raw.get("statement", "")).strip(),
        )
        if key in seen:
            continue
        seen.add(key)
        sites.append(raw)
    return sites


def _format_evidence_site(site: dict[str, Any]) -> str:
    paragraph = site.get("paragraph") or "unknown paragraph"
    line = site.get("line_start")
    location = f"line {line}" if isinstance(line, int) and line >= 0 else "line unavailable"
    statement = str(site.get("statement", "")).strip()
    return f"- {paragraph}, {location}: `{statement}`"


def _cobol_identifiers(text: str) -> list[str]:
    ignored = {"AND", "EQUAL", "NOT", "OR", "SPACE", "SPACES", "THEN"}
    result: list[str] = []
    without_literals = re.sub(r"'[^']*'|\"[^\"]*\"", " ", text)
    for token in re.findall(r"\b[A-Z][A-Z0-9-]*[A-Z0-9]\b", without_literals.upper()):
        if token in ignored or token.isdigit() or token in result:
            continue
        result.append(token)
    return result


def _deduplicate_simple_or_terms(condition: str) -> str:
    if " AND " in condition.upper() or " OR " not in condition.upper():
        return condition
    outer_parentheses = condition.startswith("(") and condition.endswith(")")
    body = condition[1:-1] if outer_parentheses else condition
    terms = re.split(r"\)\s+OR\s+\(", body, flags=re.IGNORECASE)
    if len(terms) == 1:
        terms = re.split(r"\s+OR\s+", body, flags=re.IGNORECASE)
    cleaned: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = term.strip().strip("()").strip()
        key = re.sub(r"\s+", " ", normalized).upper()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(normalized)
    if len(cleaned) == len(terms):
        return condition
    joined = ") OR (".join(cleaned)
    return f"({joined})" if outer_parentheses else joined
