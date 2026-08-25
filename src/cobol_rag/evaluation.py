from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig, load_config
from cobol_rag.query import QueryError, answer_query
from cobol_rag.query_plan import validate_plan_answer
from cobol_rag.scope import SessionState


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    category: str
    passed: bool
    checks: dict[str, bool]
    actual: dict[str, Any]
    error: str = ""


def load_gold_cases(path: Path, *, allow_holdout: bool = False) -> list[dict[str, Any]]:
    if "holdout" in {part.lower() for part in path.parts} and not allow_holdout:
        raise ValueError(
            "Holdout suites are sealed from the normal regression runner. "
            "Use `python -m cobol_rag.holdout` for a deliberate holdout evaluation."
        )
    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            has_question = bool(isinstance(payload, dict) and payload.get("question"))
            has_turns = bool(isinstance(payload, dict) and payload.get("turns"))
            if not isinstance(payload, dict) or not payload.get("id") or not (has_question or has_turns):
                raise ValueError(f"Invalid gold case on line {line_number}")
            cases.append(payload)
    return cases


def evaluate_cases(cases: list[dict[str, Any]], config: AppConfig) -> dict[str, Any]:
    results = [_evaluate_case(case, config) for case in cases]
    passed = sum(1 for result in results if result.passed)
    check_totals: dict[str, list[bool]] = {}
    for result in results:
        for name, value in result.checks.items():
            check_totals.setdefault(name, []).append(value)
    metrics = {
        "cases": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "pass_rate": passed / len(results) if results else 0.0,
        **{
            f"{name}_accuracy": sum(values) / len(values)
            for name, values in sorted(check_totals.items())
            if values
        },
    }
    category_metrics: dict[str, dict[str, Any]] = {}
    for category in sorted({result.category for result in results}):
        category_results = [result for result in results if result.category == category]
        category_passed = sum(result.passed for result in category_results)
        category_metrics[category] = {
            "cases": len(category_results),
            "passed": category_passed,
            "failed": len(category_results) - category_passed,
            "pass_rate": category_passed / len(category_results) if category_results else 0.0,
        }
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "categories": category_metrics,
        "results": [asdict(result) for result in results],
    }


def save_evaluation(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = output_dir / f"gold-eval-{stamp}.json"
    markdown_path = output_dir / f"gold-eval-{stamp}.md"
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    lines = [
        "# COBOL RAG Gold Evaluation",
        "",
        f"- Cases: {metrics['cases']}",
        f"- Passed: {metrics['passed']}",
        f"- Failed: {metrics['failed']}",
        f"- Pass rate: {metrics['pass_rate']:.1%}",
        "",
        "| Case | Category | Result | Failed checks |",
        "|---|---|---|---|",
    ]
    for result in report["results"]:
        failed_checks = ", ".join(
            name for name, value in result["checks"].items() if not value
        )
        lines.append(
            f"| {result['case_id']} | {result['category']} | "
            f"{'PASS' if result['passed'] else 'FAIL'} | {failed_checks} |"
        )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def _evaluate_case(case: dict[str, Any], config: AppConfig) -> CaseResult:
    turns = case.get("turns")
    if isinstance(turns, list) and turns:
        return _evaluate_sequence_case(case, turns, config)
    try:
        answer = answer_query(
            str(case["question"]),
            config,
            session_state=SessionState(),
        )
    except (QueryError, RuntimeError, ValueError) as error:
        return CaseResult(
            case_id=str(case["id"]),
            category=str(case.get("category", "general")),
            passed=False,
            checks={"execution": False},
            actual={},
            error=str(error),
        )

    checks, actual = _checks_for_answer(case, answer)
    return CaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "general")),
        passed=all(checks.values()),
        checks=checks,
        actual=actual,
    )


def _evaluate_sequence_case(
    case: dict[str, Any],
    turns: list[dict[str, Any]],
    config: AppConfig,
) -> CaseResult:
    state = SessionState()
    history: list[str] = []
    checks: dict[str, bool] = {}
    actual_turns: list[dict[str, Any]] = []
    try:
        for index, turn in enumerate(turns, start=1):
            question = str(turn["question"])
            answer = answer_query(
                question,
                config,
                conversation_history="\n".join(history) or None,
                session_state=state,
            )
            turn_checks, actual = _checks_for_answer(turn, answer)
            checks.update({f"turn_{index}_{name}": value for name, value in turn_checks.items()})
            actual_turns.append(actual)
            if answer.plan and answer.plan.response_language:
                state.response_language = answer.plan.response_language
            if answer.route == "technical":
                state.update(
                    answer.scope,
                    [str(source.metadata.get("source_id", "")) for source in answer.sources],
                    plan=answer.plan.as_dict() if answer.plan else None,
                )
                history.append(f"Previous user question: {question}")
                history = history[-3:]
    except (KeyError, QueryError, RuntimeError, ValueError) as error:
        checks[f"turn_{len(actual_turns) + 1}_execution"] = False
        return CaseResult(
            case_id=str(case["id"]),
            category=str(case.get("category", "conversation")),
            passed=False,
            checks=checks,
            actual={"turns": actual_turns},
            error=str(error),
        )
    return CaseResult(
        case_id=str(case["id"]),
        category=str(case.get("category", "conversation")),
        passed=all(checks.values()),
        checks=checks,
        actual={"turns": actual_turns, "final_state": state.as_dict()},
    )


def _checks_for_answer(
    expected: dict[str, Any],
    answer: Any,
) -> tuple[dict[str, bool], dict[str, Any]]:
    source_files = [
        str(source.metadata.get("source_file") or source.metadata.get("evidence_path") or "")
        for source in answer.sources
    ]
    source_programs = {
        str(source.metadata.get("program", "")).upper()
        for source in answer.sources
        if str(source.metadata.get("program", "")).strip()
    }
    checks: dict[str, bool] = {"execution": True}
    expected_execution_mode = expected.get("expected_execution_mode")
    if expected_execution_mode is not None:
        checks["execution_mode"] = answer.execution_mode == expected_execution_mode
    expected_route = expected.get("expected_route")
    if expected_route is not None:
        checks["route"] = answer.route == expected_route
    expected_intent = expected.get("expected_intent")
    if expected_intent is not None:
        checks["intent"] = answer.scope.intent == expected_intent
    expected_program = expected.get("expected_program")
    if expected_program is not None:
        checks["program"] = answer.scope.program == expected_program
    resolved_program = str(answer.scope.program or "").upper()
    if resolved_program and source_programs:
        checks["program_isolation"] = source_programs == {resolved_program}
    expected_entity = expected.get("expected_entity")
    if expected_entity is not None:
        checks["entity"] = answer.scope.entity_value == expected_entity
    expected_entities = expected.get("expected_entities") or []
    if expected_entities:
        checks["entities"] = set(answer.scope.entity_values) == {str(value) for value in expected_entities}
    forbidden_entity = expected.get("forbidden_entity")
    if forbidden_entity is not None:
        checks["forbidden_entity"] = forbidden_entity not in answer.scope.entity_values
    expected_plan = expected.get("expected_plan") or {}
    if expected_plan:
        actual_plan = answer.plan.as_dict() if answer.plan else {}
        for field, value in expected_plan.items():
            actual_value = actual_plan.get(field)
            if isinstance(value, list):
                checks[f"plan_{field}"] = set(actual_value or []) == {str(item) for item in value}
            else:
                checks[f"plan_{field}"] = actual_value == value
    expected_sources = expected.get("expected_source_files") or []
    if expected_sources:
        checks["source_recall"] = all(
            any(expected_source in actual for actual in source_files)
            for expected_source in expected_sources
        )
    contains = expected.get("answer_contains") or []
    answer_lower = answer.answer.lower()
    if contains:
        checks["answer_content"] = all(str(value).lower() in answer_lower for value in contains)
    not_contains = expected.get("answer_not_contains") or []
    if not_contains:
        checks["answer_exclusions"] = all(str(value).lower() not in answer_lower for value in not_contains)
    if "should_abstain" in expected:
        abstained = any(
            marker in answer_lower
            for marker in (
                "no direct indexed evidence",
                "cannot answer safely",
                "could not find relevant indexed evidence",
                "will not return an unsupported answer",
            )
        )
        checks["abstention"] = abstained is bool(expected["should_abstain"])
    if (
        answer.plan
        and answer.route == "technical"
        and answer.guard_status != "insufficient"
        and not answer.execution_mode.endswith(("_rejected", "_partial"))
    ):
        checks["response_contract"] = validate_plan_answer(
            answer.plan, answer.answer,
        ).passed

    actual = {
        "question": answer.question,
        "route": answer.route,
        "intent": answer.scope.intent,
        "program": answer.scope.program,
        "entity": answer.scope.entity_value,
        "entities": list(answer.scope.entity_values),
        "guard_status": answer.guard_status,
        "execution_mode": answer.execution_mode,
        "plan": answer.plan.as_dict() if answer.plan else {},
        "source_files": source_files,
        "source_programs": sorted(source_programs),
        "trace_id": answer.trace_id,
        "answer": answer.answer,
    }
    return checks, actual


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the COBOL RAG gold-question suite.")
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--gold", type=Path, default=Path("evals/pdcbvc_gold.jsonl"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--final-scripts-dir",
        type=Path,
        help="Program final_scripts directory used by deterministic evidence handlers.",
    )
    args = parser.parse_args()
    if args.final_scripts_dir:
        os.environ["COBOL_RAG_FINAL_SCRIPTS_DIR"] = str(args.final_scripts_dir.resolve())
    config = load_config(args.config)
    report = evaluate_cases(load_gold_cases(args.gold), config)
    json_path, markdown_path = save_evaluation(
        report,
        args.output_dir or config.paths.eval_dir,
    )
    print(json.dumps(report["metrics"], indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if report["metrics"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
