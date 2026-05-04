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


def build_all_missing_artifacts(root: Path, program: str) -> dict[str, dict[str, Any]]:
    return {
        "quality.dead_code": build_quality_dead_code_artifact(root, program),
        "architecture.unused_copybooks": build_unused_copybooks_artifact(root, program),
        "jcl.file_io": build_jcl_file_io_artifact(root, program),
    }


def write_missing_artifacts(root: Path, program: str) -> list[Path]:
    artifacts = build_all_missing_artifacts(root, program)
    targets = {
        "quality.dead_code": root / "quality.dead_code" / "quality.dead_code.json",
        "architecture.unused_copybooks": root / "architecture.unused_copybooks" / "architecture.unused_copybooks.json",
        "jcl.file_io": root / "jcl.file_io" / f"jcl.file_io.{program.upper()}.json",
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


def _copybook_evidence(root: Path, program: str, known_copybooks: list[str]) -> dict[str, list[dict[str, str]]]:
    evidence: dict[str, list[dict[str, str]]] = {name: [] for name in known_copybooks}

    def mark(copybook: str, source: str, detail: str) -> None:
        copybook = copybook.upper()
        if copybook not in evidence:
            return
        item = {"source": source, "detail": detail}
        if item not in evidence[copybook]:
            evidence[copybook].append(item)

    def mark_by_value(value: Any, source: str, detail: str) -> None:
        text = str(value).upper()
        if not text:
            return
        if text.startswith("COPY:"):
            mark(text.split(":", 1)[1], source, detail)
        for copybook in known_copybooks:
            if text == copybook or text.startswith(f"{copybook}-"):
                mark(copybook, source, detail)

    used = _read_json(root / "dataflow.used_variables" / "dataflow.used_variables.json")
    if isinstance(used, dict) and str(used.get("program", "")).upper() == program.upper():
        for variable in used.get("variables", []):
            if not isinstance(variable, dict):
                continue
            name = str(variable.get("variable", ""))
            origin = str(variable.get("origin", ""))
            mark_by_value(name, "dataflow.used_variables", f"variable {name}")
            mark_by_value(origin, "dataflow.used_variables", f"origin {origin}")
            if origin == "CICS_CONST" and (name.startswith("DFHPF") or name == "DFHENTER"):
                mark("DFHAID", "dataflow.used_variables", f"CICS AID constant {name}")

    for path in (root / "dataflow.variable").glob("dataflow.variable.*.json"):
        payload = _read_json(path)
        if not isinstance(payload, dict) or str(payload.get("program", "")).upper() != program.upper():
            continue
        content = payload.get("content", {})
        variable = str(content.get("variable", ""))
        origin = str(content.get("origin", ""))
        mark_by_value(variable, "dataflow.variable", f"variable {variable}")
        mark_by_value(origin, "dataflow.variable", f"origin {origin}")

    literals = _read_json(root / "dataflow.literal_assignments" / "dataflow.literal_assignments.json")
    if isinstance(literals, dict) and str(literals.get("program", "")).upper() == program.upper():
        for item in literals.get("assignments", []):
            if not isinstance(item, dict):
                continue
            target = str(item.get("target_variable", ""))
            mark_by_value(target, "dataflow.literal_assignments", f"literal assignment target {target}")

    calls = _read_json(root / "architecture.call_parameters" / "architecture.call_parameters.json")
    if isinstance(calls, dict) and str(calls.get("program", "")).upper() == program.upper():
        for call in calls.get("calls", []):
            if not isinstance(call, dict):
                continue
            for parameter in call.get("parameters", []):
                mark_by_value(parameter, "architecture.call_parameters", f"call parameter {parameter}")
            for detail in call.get("parameter_details", []):
                if not isinstance(detail, dict):
                    continue
                mark_by_value(detail.get("field_prefix", ""), "architecture.call_parameters", "parameter field prefix")
                for variable in detail.get("variables", []):
                    if isinstance(variable, dict):
                        mark_by_value(variable.get("variable", ""), "architecture.call_parameters", "parameter variable")

    return evidence


def _jcl_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for path in sorted((root / "jcl").glob("**/jcl.summary.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            summaries.append(payload)
    return summaries


def _jcl_steps(root: Path) -> list[dict[str, Any]]:
    steps = []
    for path in sorted((root / "jcl").glob("**/jcl.steps.*.json")):
        payload = _read_json(path)
        if isinstance(payload, dict):
            steps.append(payload)
    return steps


def _summary_item(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "job": summary.get("job"),
        "purpose": summary.get("purpose"),
        "source": summary.get("source"),
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
            }
        )
    return items


def _read_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


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
