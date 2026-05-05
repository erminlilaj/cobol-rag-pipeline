from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_quality_dead_code_artifact(root: Path, program: str) -> dict[str, Any]:
    program = program.upper()
    comments = _read_json(root / "program.comments" / "program.comments.json")
    cfg = _read_json(root / "controlflow.cfg" / "controlflow.cfg.json")

    commented_out: list[dict[str, Any]] = []
    if isinstance(comments, dict) and str(comments.get("program", "")).upper() == program:
        for comment in comments.get("comments", []):
            if not isinstance(comment, dict) or comment.get("classification") != "commented_out_code":
                continue
            commented_out.append(
                {
                    "line": comment.get("line"),
                    "section": comment.get("section"),
                    "paragraph": comment.get("paragraph"),
                    "kind": comment.get("kind"),
                    "text": str(comment.get("text_raw") or comment.get("text") or "").strip(),
                    "classification_reason": comment.get("classification_reason"),
                    "evidence": "program.comments.json",
                    "citation": _citation("program.comments/program.comments.json", line=comment.get("line")),
                }
            )

    reachability = _cfg_reachability(cfg, program)
    return {
        "schema_version": 1,
        "type": "quality.dead_code",
        "program": program,
        "title": f"{program} dead-code and commented-code evidence",
        "content": {
            "commented_out_code_count": len(commented_out),
            "commented_out_code": commented_out,
            "cfg_reachability": reachability,
            "unreachable_paragraphs": reachability.get("unreachable_nodes", []),
            "limitations": [
                "commented_out_code is direct evidence from program.comments.json.",
                "cfg_reachability is static and limited to nodes present in controlflow.cfg.json.",
                "An empty unreachable_paragraphs list is not a runtime proof that every statement executes.",
            ],
        },
        "evidence": {
            "comments_artifact": "program.comments/program.comments.json",
            "cfg_artifact": "controlflow.cfg/controlflow.cfg.json",
        },
    }


def build_unused_copybooks_artifact(root: Path, program: str) -> dict[str, Any]:
    program = program.upper()
    copybooks_payload = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
    if not isinstance(copybooks_payload, dict) or str(copybooks_payload.get("program", "")).upper() != program:
        all_copybooks: list[str] = []
        classified: dict[str, list[str]] = {}
    else:
        content = copybooks_payload.get("content", {})
        all_copybooks = [str(item).upper() for item in content.get("all", [])]
        classified = {
            str(name): [str(item).upper() for item in values]
            for name, values in content.get("classified", {}).items()
            if isinstance(values, list)
        }

    evidence = _copybook_evidence(root, program, all_copybooks)
    referenced = sorted(name for name, items in evidence.items() if items)
    needs_review = [name for name in all_copybooks if name not in referenced]

    copybook_status = []
    for name in all_copybooks:
        items = evidence.get(name, [])
        copybook_status.append(
            {
                "copybook": name,
                "status": "referenced_by_available_artifacts" if items else "needs_review_no_reference_in_available_artifacts",
                "evidence": items,
            }
        )

    return {
        "schema_version": 1,
        "type": "architecture.unused_copybooks",
        "program": program,
        "title": f"{program} COPY usage review",
        "content": {
            "copybooks_total": len(all_copybooks),
            "all_copybooks": all_copybooks,
            "classified": classified,
            "referenced_copybooks": referenced,
            "needs_review_count": len(needs_review),
            "needs_review_copybooks": needs_review,
            "unused_copybooks_proven": [],
            "copybook_status": copybook_status,
            "proof_level": (
                "available-artifact-reference review; not compiler-expanded source proof"
            ),
            "limitations": [
                "This artifact does not prove COPY members are unused.",
                "needs_review_copybooks are members with no reference evidence in the available final_scripts artifacts.",
                "A compiler-expanded source or copybook field-level parser is needed for stronger proof.",
            ],
        },
        "evidence": {
            "copybooks_artifact": "architecture.copybooks/architecture.copybooks.json",
            "dataflow_artifacts": [
                "dataflow.used_variables/dataflow.used_variables.json",
                "dataflow.variable/*.json",
                "dataflow.literal_assignments/dataflow.literal_assignments.json",
                "architecture.call_parameters/architecture.call_parameters.json",
            ],
        },
    }


def build_jcl_file_io_artifact(root: Path, program: str) -> dict[str, Any]:
    program = program.upper()
    summaries = _jcl_summaries(root)
    steps = _jcl_steps(root)
    matching_jobs = []
    matching_steps = []
    reads: list[dict[str, Any]] = []
    writes: list[dict[str, Any]] = []
    sysout: list[dict[str, Any]] = []

    for summary in summaries:
        programs = {str(item).upper() for item in summary.get("programs", [])}
        if program in programs:
            matching_jobs.append(_summary_item(summary))

    for step in steps:
        step_program = str(step.get("program") or step.get("target") or "").upper()
        if step_program != program:
            continue
        matching_steps.append(_step_item(step))
        reads.extend(_dd_items(step, wanted_access={"read"}))
        writes.extend(_dd_items(step, wanted_access={"write", "delete"}))
        sysout.extend(_dd_items(step, wanted_access={"sysout"}))

    matching_jobs = _unique_dicts(matching_jobs, ("job",))
    matching_steps = _unique_dicts(matching_steps, ("job", "step", "program"))
    reads = _unique_dicts(reads, ("job", "step", "ddname", "dsn", "access_type"))
    writes = _unique_dicts(writes, ("job", "step", "ddname", "dsn", "access_type"))
    sysout = _unique_dicts(sysout, ("job", "step", "ddname", "sysout", "output", "access_type"))

    known_jobs = sorted({str(summary.get("job", "")) for summary in summaries if summary.get("job")})
    known_programs = sorted(
        {
            str(item).upper()
            for summary in summaries
            for item in summary.get("programs", [])
            if str(item).strip()
        }
    )
    return {
        "schema_version": 1,
        "type": "jcl.file_io",
        "program": program,
        "title": f"{program} JCL file I/O evidence",
        "content": {
            "matching_jobs_count": len(matching_jobs),
            "matching_jobs": matching_jobs,
            "matching_steps_count": len(matching_steps),
            "matching_steps": matching_steps,
            "reads": reads,
            "writes": writes,
            "sysout": sysout,
            "has_jcl_linkage": bool(matching_jobs or matching_steps),
            "known_jobs": known_jobs,
            "known_programs_sample": known_programs[:30],
            "limitations": [
                "This artifact maps file I/O through JCL step summaries only.",
                "CICS online programs often have no JCL dataset production evidence.",
                "Program-to-JCL linkage depends on parsed EXEC PGM/procedure summaries.",
            ],
        },
        "evidence": {
            "jcl_root": "jcl/**/*.json",
        },
    }


def build_screen_field_lineage_artifact(root: Path, program: str) -> dict[str, Any]:
    program = program.upper()
    variables = _screen_variable_payloads(root, program)
    by_name = {str(payload.get("content", {}).get("variable", "")).upper(): payload for payload in variables}
    literals_by_target = _literal_assignments_by_target(root, program)

    fields: list[dict[str, Any]] = []
    for name, payload in sorted(by_name.items()):
        content = payload.get("content", {})
        family = _screen_family(name)
        family_members = sorted(candidate for candidate in by_name if _screen_family(candidate) == family)
        related_variables = _related_variables_for_screen_field(name, payload, by_name.keys())
        literal_assignments = []
        for member in family_members:
            literal_assignments.extend(literals_by_target.get(member, []))
        fields.append(
            {
                "field": name,
                "family": family,
                "family_members": family_members,
                "origin": content.get("origin"),
                "defined_in": content.get("defined_in", []),
                "modified_in": content.get("modified_in", []),
                "used_in": content.get("used_in", []),
                "controls_flow": bool(content.get("controls_flow")),
                "fanout_nodes": content.get("fanout_nodes", []),
                "write_sites": _site_items(name, content, "write_sites"),
                "read_sites": _site_items(name, content, "read_sites"),
                "control_sites": _site_items(name, content, "control_sites"),
                "literal_assignments": literal_assignments,
                "related_variables": related_variables,
                "source_artifact": f"dataflow.variable/dataflow.variable.{name}.json",
            }
        )

    copybook_origins = sorted(
        {
            str(field.get("origin", "")).split(":", 1)[1]
            for field in fields
            if str(field.get("origin", "")).startswith("COPY:")
        }
    )
    return {
        "schema_version": 1,
        "type": "screen_field_lineage",
        "program": program,
        "title": f"{program} screen/map field lineage",
        "content": {
            "fields_count": len(fields),
            "fields": fields,
            "copybook_origins": copybook_origins,
            "limitations": [
                "Lineage is derived from dataflow.variable artifacts and literal assignments.",
                "Input-origin values entered by terminal users have read/control evidence but may not have write evidence in COBOL.",
                "Family grouping uses common BMS suffixes such as I, O, L, A, and F.",
            ],
        },
        "evidence": {
            "variable_artifacts": "dataflow.variable/dataflow.variable.*.json",
            "literal_artifact": "dataflow.literal_assignments/dataflow.literal_assignments.json",
        },
    }


def build_all_missing_artifacts(root: Path, program: str) -> dict[str, dict[str, Any]]:
    return {
        "quality.dead_code": build_quality_dead_code_artifact(root, program),
        "architecture.unused_copybooks": build_unused_copybooks_artifact(root, program),
        "jcl.file_io": build_jcl_file_io_artifact(root, program),
        "screen_field_lineage": build_screen_field_lineage_artifact(root, program),
    }


def write_missing_artifacts(root: Path, program: str) -> list[Path]:
    artifacts = build_all_missing_artifacts(root, program)
    targets = {
        "quality.dead_code": root / "quality.dead_code" / "quality.dead_code.json",
        "architecture.unused_copybooks": root / "architecture.unused_copybooks" / "architecture.unused_copybooks.json",
        "jcl.file_io": root / "jcl.file_io" / f"jcl.file_io.{program.upper()}.json",
        "screen_field_lineage": root / "screen_field_lineage" / "screen_field_lineage.json",
    }
    written: list[Path] = []
    for artifact_type, payload in artifacts.items():
        target = targets[artifact_type]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(target)
    return written


def load_or_build_quality_dead_code(root: Path, program: str) -> dict[str, Any]:
    path = root / "quality.dead_code" / "quality.dead_code.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program.upper():
        return payload
    return build_quality_dead_code_artifact(root, program)


def load_or_build_unused_copybooks(root: Path, program: str) -> dict[str, Any]:
    path = root / "architecture.unused_copybooks" / "architecture.unused_copybooks.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program.upper():
        return payload
    return build_unused_copybooks_artifact(root, program)


def load_or_build_jcl_file_io(root: Path, program: str) -> dict[str, Any]:
    path = root / "jcl.file_io" / f"jcl.file_io.{program.upper()}.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program.upper():
        return payload
    return build_jcl_file_io_artifact(root, program)


def load_or_build_screen_field_lineage(root: Path, program: str) -> dict[str, Any]:
    path = root / "screen_field_lineage" / "screen_field_lineage.json"
    payload = _read_json(path)
    if isinstance(payload, dict) and str(payload.get("program", "")).upper() == program.upper():
        return payload
    return build_screen_field_lineage_artifact(root, program)


def program_has_jcl_evidence(root: Path, program: str) -> bool:
    program = program.upper()
    for summary in _jcl_summaries(root):
        if program in {str(item).upper() for item in summary.get("programs", [])}:
            return True
    for step in _jcl_steps(root):
        if str(step.get("program") or step.get("target") or "").upper() == program:
            return True
    return False


def _cfg_reachability(cfg: Any, program: str) -> dict[str, Any]:
    if not isinstance(cfg, dict) or str(cfg.get("program", "")).upper() != program.upper():
        return {
            "status": "not_available",
            "entry": program.upper(),
            "nodes_count": 0,
            "reachable_nodes_count": 0,
            "unreachable_nodes_count": 0,
            "unreachable_nodes": [],
        }

    edges = [edge for edge in cfg.get("edges", []) if isinstance(edge, dict)]
    nodes = {program.upper()}
    adjacency: dict[str, set[str]] = {}
    for edge in edges:
        source = str(edge.get("from", "")).upper()
        target = str(edge.get("to", "")).upper()
        if not source or not target:
            continue
        nodes.add(source)
        nodes.add(target)
        adjacency.setdefault(source, set()).add(target)

    seen = {program.upper()}
    stack = [program.upper()]
    while stack:
        node = stack.pop()
        for target in adjacency.get(node, set()):
            if target in seen:
                continue
            seen.add(target)
            stack.append(target)

    unreachable = sorted(nodes - seen)
    return {
        "status": "computed_from_controlflow_cfg",
        "entry": program.upper(),
        "nodes_count": len(nodes),
        "reachable_nodes_count": len(seen),
        "unreachable_nodes_count": len(unreachable),
        "unreachable_nodes": unreachable,
    }


def _copybook_evidence(root: Path, program: str, known_copybooks: list[str]) -> dict[str, list[dict[str, Any]]]:
    evidence: dict[str, list[dict[str, Any]]] = {name: [] for name in known_copybooks}

    def mark(copybook: str, source: str, detail: str, *, line: Any | None = None, citation: str | None = None) -> None:
        copybook = copybook.upper()
        if copybook not in evidence:
            return
        item = {"source": source, "detail": detail}
        if line is not None:
            item["line"] = line
        if citation:
            item["citation"] = citation
        if item not in evidence[copybook]:
            evidence[copybook].append(item)

    def mark_by_value(value: Any, source: str, detail: str, *, line: Any | None = None, citation: str | None = None) -> None:
        text = str(value).upper()
        if not text:
            return
        if text.startswith("COPY:"):
            mark(text.split(":", 1)[1], source, detail, line=line, citation=citation)
        for copybook in known_copybooks:
            if text == copybook or text.startswith(f"{copybook}-"):
                mark(copybook, source, detail, line=line, citation=citation)

    used = _read_json(root / "dataflow.used_variables" / "dataflow.used_variables.json")
    if isinstance(used, dict) and str(used.get("program", "")).upper() == program.upper():
        for variable in used.get("variables", []):
            if not isinstance(variable, dict):
                continue
            name = str(variable.get("variable", ""))
            origin = str(variable.get("origin", ""))
            line = _first_site_line(variable)
            citation = _citation("dataflow.used_variables/dataflow.used_variables.json", line=line, detail=name)
            mark_by_value(name, "dataflow.used_variables", f"variable {name}", line=line, citation=citation)
            mark_by_value(origin, "dataflow.used_variables", f"origin {origin}", line=line, citation=citation)
            if origin == "CICS_CONST" and (name.startswith("DFHPF") or name == "DFHENTER"):
                mark("DFHAID", "dataflow.used_variables", f"CICS AID constant {name}", line=line, citation=citation)

    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != program.upper():
            continue
        content = payload.get("content", {})
        variable = str(content.get("variable", ""))
        origin = str(content.get("origin", ""))
        line = _first_site_line(content)
        citation = _citation(f"dataflow.variable/{path.name}", line=line, detail=variable)
        mark_by_value(variable, "dataflow.variable", f"variable {variable}", line=line, citation=citation)
        mark_by_value(origin, "dataflow.variable", f"origin {origin}", line=line, citation=citation)

    literals = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if isinstance(literals, dict) and str(literals.get("program", "")).upper() == program.upper():
        for item in literals.get("assignments", []):
            if not isinstance(item, dict):
                continue
            target = str(item.get("target_variable", ""))
            citation = _citation(
                "dataflow.literal_assignments/dataflow.literal_assignments.json",
                line=item.get("line"),
                detail=target,
            )
            mark_by_value(
                target,
                "dataflow.literal_assignments",
                f"literal assignment target {target}",
                line=item.get("line"),
                citation=citation,
            )

    calls = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if isinstance(calls, dict) and str(calls.get("program", "")).upper() == program.upper():
        for call in calls.get("calls", []):
            if not isinstance(call, dict):
                continue
            for parameter in call.get("parameters", []):
                citation = _citation(
                    "architecture.call_parameters/architecture.call_parameters.json",
                    line=call.get("line_start"),
                    detail=str(call.get("target", "")),
                )
                mark_by_value(
                    parameter,
                    "architecture.call_parameters",
                    f"call parameter {parameter}",
                    line=call.get("line_start"),
                    citation=citation,
                )
            for detail in call.get("parameter_details", []):
                if not isinstance(detail, dict):
                    continue
                citation = _citation(
                    "architecture.call_parameters/architecture.call_parameters.json",
                    line=call.get("line_start"),
                    detail=str(detail.get("field_prefix", "")),
                )
                mark_by_value(
                    detail.get("field_prefix", ""),
                    "architecture.call_parameters",
                    "parameter field prefix",
                    line=call.get("line_start"),
                    citation=citation,
                )
                for variable in detail.get("variables", []):
                    if isinstance(variable, dict):
                        variable_line = _first_call_parameter_line(variable) or call.get("line_start")
                        variable_citation = _citation(
                            "architecture.call_parameters/architecture.call_parameters.json",
                            line=variable_line,
                            detail=str(variable.get("variable", "")),
                        )
                        mark_by_value(
                            variable.get("variable", ""),
                            "architecture.call_parameters",
                            "parameter variable",
                            line=variable_line,
                            citation=variable_citation,
                        )

    return evidence


def _jcl_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((root / "jcl").glob("**/jcl.summary.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            payload["__artifact_path"] = _relative_artifact_path(root, path)
            summaries.append(payload)
    return summaries


def _jcl_steps(root: Path) -> list[dict[str, Any]]:
    steps = []
    for path in sorted((root / "jcl").glob("**/jcl.steps.*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            payload["__artifact_path"] = _relative_artifact_path(root, path)
            steps.append(payload)
    return steps


def _summary_item(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "job": summary.get("job"),
        "purpose": summary.get("purpose"),
        "source": summary.get("source"),
        "source_artifact": summary.get("__artifact_path"),
        "programs": summary.get("programs", []),
        "steps_count": summary.get("steps_count"),
        "datasets_count": summary.get("datasets_count"),
        "external_inputs_count": summary.get("external_inputs_count"),
    }


def _step_item(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "job": step.get("job"),
        "step": step.get("step"),
        "program": step.get("program") or step.get("target"),
        "scope": step.get("scope"),
        "source": step.get("source"),
        "source_artifact": step.get("__artifact_path"),
        "reads_count": len(step.get("reads", [])),
        "writes_count": len(step.get("writes", [])),
        "deletes_count": len(step.get("deletes", [])),
        "dds_count": len(step.get("dds", [])),
    }


def _dd_items(step: dict[str, Any], wanted_access: set[str]) -> list[dict[str, Any]]:
    items = []
    for dd in step.get("dds", []):
        if not isinstance(dd, dict):
            continue
        access_type = str(dd.get("access_type", "")).lower()
        if access_type not in wanted_access:
            continue
        items.append(
            {
                "job": step.get("job"),
                "step": step.get("step"),
                "program": step.get("program") or step.get("target"),
                "ddname": dd.get("ddname"),
                "dsn": dd.get("dsn"),
                "disp": dd.get("disp"),
                "sysout": dd.get("sysout"),
                "output": dd.get("output"),
                "dataset_kind": dd.get("dataset_kind"),
                "access_type": dd.get("access_type"),
                "access_reason": dd.get("access_reason"),
                "source_lines": dd.get("source_lines"),
                "citation": _citation(
                    str(step.get("__artifact_path") or f"jcl/**/{step.get('job')}/jcl.steps.{step.get('step')}.json"),
                    line=(dd.get("source_lines") or {}).get("start"),
                    detail=str(dd.get("ddname", "")),
                ),
            }
        )
    return items


def _screen_variable_payloads(root: Path, program: str) -> list[dict[str, Any]]:
    variables: list[dict[str, Any]] = []
    screen_copybooks = _screen_copybooks(root, program)
    for path in sorted((root / "dataflow.variable").glob("dataflow.variable.*.json")):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != program.upper():
            continue
        origin = str(payload.get("content", {}).get("origin", "")).upper()
        if origin.startswith("COPY:") and origin.split(":", 1)[1] in screen_copybooks:
            variables.append(payload)
    return variables


def _screen_copybooks(root: Path, program: str) -> set[str]:
    result: set[str] = set()
    nav = _read_json(root / "ui.cics.navigation" / "ui.cics.navigation.json")
    if isinstance(nav, dict) and str(nav.get("program", "")).upper() == program.upper():
        for item in nav.get("content", {}).get("maps", []):
            mapset = str(item.get("mapset", "")).upper()
            if mapset:
                result.add(mapset)
    if not result:
        copybooks = _read_json(root / "architecture.copybooks" / "architecture.copybooks.json")
        if isinstance(copybooks, dict) and str(copybooks.get("program", "")).upper() == program.upper():
            for name in copybooks.get("content", {}).get("classified", {}).get("ui_cics", []):
                text = str(name).upper()
                if text.endswith("M"):
                    result.add(text)
    return result


def _literal_assignments_by_target(root: Path, program: str) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    payload = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != program.upper():
        return result
    for item in payload.get("assignments", []):
        if not isinstance(item, dict) or not item.get("screen_or_map_field"):
            continue
        target = str(item.get("target_variable", "")).upper()
        result.setdefault(target, []).append(
            {
                "target_variable": target,
                "literal": item.get("literal"),
                "paragraph": item.get("paragraph"),
                "line": item.get("line"),
                "statement": item.get("statement"),
                "citation": _citation(
                    "dataflow.literal_assignments/dataflow.literal_assignments.json",
                    line=item.get("line"),
                    detail=target,
                ),
            }
        )
    return result


def _site_items(variable: str, content: dict[str, Any], key: str) -> list[dict[str, Any]]:
    sites = []
    for site in content.get("evidence", {}).get(key, []):
        if not isinstance(site, dict):
            continue
        line = site.get("line_start")
        sites.append(
            {
                "paragraph": site.get("paragraph"),
                "line_start": line,
                "statement": site.get("statement"),
                "citation": _citation(f"dataflow.variable/dataflow.variable.{variable}.json", line=line),
            }
        )
    return sites


def _related_variables_for_screen_field(
    variable: str,
    payload: dict[str, Any],
    screen_names: Any,
) -> list[dict[str, Any]]:
    names = {str(name).upper() for name in screen_names}
    related: dict[str, dict[str, Any]] = {}
    content = payload.get("content", {})
    for key in ("write_sites", "read_sites", "control_sites"):
        for site in content.get("evidence", {}).get(key, []):
            statement = str(site.get("statement", ""))
            for token in _tokens_from_statement(statement):
                if token == variable.upper() or token in names or _is_cobol_keyword(token):
                    continue
                item = {
                    "variable": token,
                    "paragraph": site.get("paragraph"),
                    "line": site.get("line_start"),
                    "statement": statement,
                    "relationship": f"appears with {variable.upper()} in {key}",
                    "citation": _citation(
                        f"dataflow.variable/dataflow.variable.{variable.upper()}.json",
                        line=site.get("line_start"),
                        detail=token,
                    ),
                }
                existing = related.get(token)
                if existing and _positive_line(existing.get("line")) and not _positive_line(item.get("line")):
                    continue
                related[token] = item
    return sorted(related.values(), key=lambda item: (str(item.get("variable")), int(item.get("line") or 0)))


def _tokens_from_statement(statement: str) -> set[str]:
    return {token for token in re_find_tokens(statement) if "-" in token or token.startswith(("W", "PD", "TWCOB", "PX", "DFH", "SQL"))}


def re_find_tokens(text: str) -> list[str]:
    import re

    return re.findall(r"\b[A-Z][A-Z0-9-]{1,}\b", text.upper())


def _screen_family(variable: str) -> str:
    variable = variable.upper()
    if len(variable) > 1 and variable[-1] in {"A", "F", "I", "L", "O"}:
        return variable[:-1]
    return variable


def _is_cobol_keyword(token: str) -> bool:
    return token in {
        "AND",
        "END-IF",
        "EQUAL",
        "GREATER",
        "HIGH-VALUE",
        "IF",
        "MOVE",
        "NOT",
        "OR",
        "SPACES",
        "THEN",
        "TO",
    }


def _positive_line(value: Any) -> bool:
    return isinstance(value, int) and value > 0


def _first_site_line(payload: dict[str, Any]) -> Any | None:
    evidence = payload.get("evidence", {})
    if not evidence and isinstance(payload.get("content"), dict):
        evidence = payload.get("content", {}).get("evidence", {})
    for key in ("write_sites", "read_sites", "control_sites"):
        for site in evidence.get(key, []):
            line = site.get("line_start")
            if isinstance(line, int) and line > 0:
                return line
    return None


def _first_call_parameter_line(variable: dict[str, Any]) -> Any | None:
    for key in ("writes_before_call", "reads_before_call", "reads_after_call"):
        for site in variable.get(key, []):
            line = site.get("line_start")
            if isinstance(line, int) and line > 0:
                return line
    return None


def _citation(path: str, *, line: Any | None = None, detail: str | None = None) -> str:
    parts = [path]
    if line not in (None, "", -1):
        parts.append(f"line {line}")
    if detail:
        parts.append(str(detail))
    return " | ".join(parts)


def _read_json(path: Path) -> Any | None:
    candidates = [path]
    if len(path.parents) >= 2:
        # Support both layouts:
        #   root/quality.dead_code/quality.dead_code.json
        #   root/quality.dead_code.json
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


def _relative_artifact_path(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _unique_dicts(items: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for item in items:
        key = tuple(str(item.get(name, "")) for name in keys)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result
