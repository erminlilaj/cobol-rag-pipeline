from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cobol_rag.config import AppConfig, load_config  # noqa: E402
from cobol_rag.final_scripts_answers import find_final_scripts_root  # noqa: E402
from cobol_rag.query import answer_query  # noqa: E402


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str


@dataclass
class CaseResult:
    case_id: str
    category: str
    status: str
    question: str
    checks: list[CheckResult] = field(default_factory=list)
    answer: str = ""
    sources: int = 0
    elapsed_ms: int = 0
    skip_reason: str = ""
    expected: str = "pass"

    @property
    def passed_checks(self) -> int:
        return sum(1 for check in self.checks if check.passed)

    @property
    def total_checks(self) -> int:
        return len(self.checks)


def main() -> int:
    args = parse_args()
    root = args.repo_root.resolve()
    config = load_config(args.config)
    gold = load_gold(args.gold)
    environment = detect_environment(config)
    cases = select_cases(gold.get("cases", []), args.case, args.category)

    if args.list_cases:
        list_cases(cases)
        return 0

    defaults = gold.get("defaults", {})
    results: list[CaseResult] = []
    for case in cases:
        result = run_case(case, defaults, config, environment)
        results.append(result)
        print_case_result(result, include_answer=args.show_answers)
        if args.fail_fast and result.status == "fail":
            break

    report = build_report(gold, results, environment)
    print_summary(report)

    if args.json_output:
        write_json_report(args.json_output, report)
    if args.markdown_output:
        write_markdown_report(args.markdown_output, report)

    failures = report["summary"]["failed"]
    strict_skip_failed = args.strict_skips and report["summary"]["skipped"]
    strict_xpass_failed = args.strict_xpass and report["summary"]["xpassed"]
    return 1 if failures or strict_skip_failed or strict_xpass_failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run gold evaluation cases for the COBOL RAG assistant.")
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("eval/questions/pdcbvc_gold.json"),
        help="Gold JSON file to run.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/default.yaml"),
        help="COBOL RAG config file.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=ROOT,
        help="Repository root. Defaults to the parent of eval/.",
    )
    parser.add_argument("--case", action="append", default=[], help="Run one case id. Repeatable.")
    parser.add_argument("--category", action="append", default=[], help="Run one category. Repeatable.")
    parser.add_argument("--list-cases", action="store_true", help="List selected cases and exit.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first failing active case.")
    parser.add_argument("--strict-skips", action="store_true", help="Return non-zero if any case is skipped.")
    parser.add_argument("--strict-xpass", action="store_true", help="Return non-zero if a known-gap case unexpectedly passes.")
    parser.add_argument("--show-answers", action="store_true", help="Print answer text for every case.")
    parser.add_argument("--json-output", type=Path, help="Write a JSON evaluation report.")
    parser.add_argument("--markdown-output", type=Path, help="Write a Markdown evaluation report.")
    return parser.parse_args()


def load_gold(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(f"Gold file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(f"Invalid gold JSON in {path}: {error}") from error
    if not isinstance(data, dict):
        raise SystemExit(f"Gold file must contain a JSON object: {path}")
    if not isinstance(data.get("cases"), list):
        raise SystemExit(f"Gold file must contain a cases list: {path}")
    return data


def detect_environment(config: AppConfig) -> dict[str, Any]:
    final_scripts_root = find_final_scripts_root()
    manifest_path = config.paths.manifest_dir / f"{config.index.collection}.json"
    return {
        "final_scripts": final_scripts_root is not None,
        "final_scripts_root": str(final_scripts_root) if final_scripts_root else "",
        "rag_index": manifest_path.exists() or config.paths.chroma_dir.exists(),
        "manifest_path": str(manifest_path),
        "ollama": ollama_available(config),
        "llm_model": config.llm.model,
        "embedding_model": config.embedding.model,
    }


def ollama_available(config: AppConfig) -> bool:
    url = config.llm.base_url.rstrip("/") + "/api/tags"
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=2) as response:
            return 200 <= response.status < 300
    except (OSError, urllib.error.URLError):
        return False


def select_cases(
    cases: list[dict[str, Any]],
    case_ids: list[str],
    categories: list[str],
) -> list[dict[str, Any]]:
    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.get("id") in wanted]
    if categories:
        wanted_categories = set(categories)
        selected = [case for case in selected if case.get("category") in wanted_categories]
    return selected


def list_cases(cases: list[dict[str, Any]]) -> None:
    for case in cases:
        requirements = ", ".join(case.get("requirements", []))
        print(f"{case.get('id')} [{case.get('category')}] requirements={requirements}")


def run_case(
    case: dict[str, Any],
    defaults: dict[str, Any],
    config: AppConfig,
    environment: dict[str, Any],
) -> CaseResult:
    case_id = str(case.get("id", "unnamed"))
    category = str(case.get("category", "uncategorized"))
    question = str(case.get("question", ""))
    expected = str(case.get("expected", "pass"))
    requirements = case.get("requirements", defaults.get("requirements", []))
    missing = [req for req in requirements if not environment.get(req)]
    if missing:
        return CaseResult(
            case_id=case_id,
            category=category,
            status="skip",
            question=question,
            skip_reason=f"missing requirement(s): {', '.join(missing)}",
            expected=expected,
        )

    top_k = case.get("top_k", defaults.get("top_k"))
    chunk_types = case.get("chunk_types")
    started = time.perf_counter()
    answer = answer_query(
        question,
        config=config,
        top_k=int(top_k) if top_k is not None else None,
        chunk_types=chunk_types,
    )
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    checks = evaluate_assertions(
        answer=answer.answer,
        sources=answer.sources,
        assertions=case.get("assertions", {}),
        case_sensitive=bool(case.get("case_sensitive", defaults.get("case_sensitive", False))),
    )
    passed = all(check.passed for check in checks)
    status = status_for_expected(expected=expected, passed=passed)
    return CaseResult(
        case_id=case_id,
        category=category,
        status=status,
        question=question,
        checks=checks,
        answer=answer.answer,
        sources=len(answer.sources),
        elapsed_ms=elapsed_ms,
        expected=expected,
    )


def status_for_expected(expected: str, passed: bool) -> str:
    if expected == "known_gap":
        return "xpass" if passed else "xfail"
    return "pass" if passed else "fail"


def evaluate_assertions(
    *,
    answer: str,
    sources: list[Any],
    assertions: dict[str, Any],
    case_sensitive: bool,
) -> list[CheckResult]:
    checks: list[CheckResult] = []
    haystack = answer if case_sensitive else answer.lower()

    for fragment in assertions.get("contains_all", []):
        needle = str(fragment) if case_sensitive else str(fragment).lower()
        checks.append(CheckResult(
            name="contains_all",
            passed=needle in haystack,
            detail=str(fragment),
        ))

    for group in assertions.get("contains_any", []):
        fragments = group if isinstance(group, list) else [group]
        normalized = [str(fragment) if case_sensitive else str(fragment).lower() for fragment in fragments]
        passed = any(fragment in haystack for fragment in normalized)
        checks.append(CheckResult(
            name="contains_any",
            passed=passed,
            detail=" | ".join(str(fragment) for fragment in fragments),
        ))

    regex_flags = 0 if case_sensitive else re.IGNORECASE
    for pattern in assertions.get("regex_all", []):
        passed = re.search(str(pattern), answer, flags=regex_flags | re.MULTILINE) is not None
        checks.append(CheckResult(name="regex_all", passed=passed, detail=str(pattern)))

    for fragment in assertions.get("forbidden", []):
        needle = str(fragment) if case_sensitive else str(fragment).lower()
        checks.append(CheckResult(
            name="forbidden",
            passed=needle not in haystack,
            detail=str(fragment),
        ))

    if "min_sources" in assertions:
        expected = int(assertions["min_sources"])
        checks.append(CheckResult(
            name="min_sources",
            passed=len(sources) >= expected,
            detail=f"expected >= {expected}, got {len(sources)}",
        ))

    source_text = "\n".join(
        [str(getattr(source, "text", "")) + "\n" + json.dumps(getattr(source, "metadata", {}), sort_keys=True)
         for source in sources]
    )
    source_haystack = source_text if case_sensitive else source_text.lower()
    for fragment in assertions.get("source_contains_all", []):
        needle = str(fragment) if case_sensitive else str(fragment).lower()
        checks.append(CheckResult(
            name="source_contains_all",
            passed=needle in source_haystack,
            detail=str(fragment),
        ))

    if not checks:
        checks.append(CheckResult(name="has_answer", passed=bool(answer.strip()), detail="answer is non-empty"))

    return checks


def print_case_result(result: CaseResult, *, include_answer: bool) -> None:
    status_label = result.status.upper()
    timing = f"{result.elapsed_ms}ms" if result.elapsed_ms else "-"
    print(f"{status_label:5} {result.case_id} [{result.category}] checks={result.passed_checks}/{result.total_checks} time={timing}")
    if result.skip_reason:
        print(f"      {result.skip_reason}")
    failed_checks = [check for check in result.checks if not check.passed]
    for check in failed_checks[:8]:
        print(f"      missing/failed {check.name}: {check.detail}")
    if include_answer and result.answer:
        print(indent_block(result.answer, prefix="      "))


def indent_block(text: str, prefix: str) -> str:
    return "\n".join(prefix + line for line in text.splitlines())


def build_report(
    gold: dict[str, Any],
    results: list[CaseResult],
    environment: dict[str, Any],
) -> dict[str, Any]:
    by_category: dict[str, dict[str, int]] = {}
    for result in results:
        bucket = by_category.setdefault(
            result.category,
            {"pass": 0, "fail": 0, "skip": 0, "xfail": 0, "xpass": 0, "total": 0},
        )
        bucket[result.status] += 1
        bucket["total"] += 1

    summary = {
        "total": len(results),
        "passed": sum(1 for result in results if result.status == "pass"),
        "failed": sum(1 for result in results if result.status == "fail"),
        "skipped": sum(1 for result in results if result.status == "skip"),
        "xfailed": sum(1 for result in results if result.status == "xfail"),
        "xpassed": sum(1 for result in results if result.status == "xpass"),
    }
    active = summary["passed"] + summary["failed"]
    summary["active_pass_rate"] = round(summary["passed"] / active, 4) if active else None

    return {
        "suite": gold.get("suite"),
        "program": gold.get("program"),
        "schema_version": gold.get("schema_version"),
        "environment": environment,
        "summary": summary,
        "by_category": by_category,
        "results": [
            {
                "id": result.case_id,
                "category": result.category,
                "status": result.status,
                "expected": result.expected,
                "question": result.question,
                "checks": [
                    {"name": check.name, "passed": check.passed, "detail": check.detail}
                    for check in result.checks
                ],
                "sources": result.sources,
                "elapsed_ms": result.elapsed_ms,
                "skip_reason": result.skip_reason,
                "answer": result.answer,
            }
            for result in results
        ],
    }


def print_summary(report: dict[str, Any]) -> None:
    summary = report["summary"]
    print()
    print(
        "SUMMARY "
        f"pass={summary['passed']} fail={summary['failed']} skip={summary['skipped']} "
        f"xfail={summary['xfailed']} xpass={summary['xpassed']} total={summary['total']}"
    )
    if summary["active_pass_rate"] is not None:
        print(f"ACTIVE_PASS_RATE {summary['active_pass_rate']:.1%}")
    print("BY_CATEGORY")
    for category, bucket in sorted(report["by_category"].items()):
        print(
            f"  {category}: pass={bucket['pass']} fail={bucket['fail']} "
            f"skip={bucket['skip']} xfail={bucket['xfail']} xpass={bucket['xpass']} total={bucket['total']}"
        )


def write_json_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = report["summary"]
    lines = [
        f"# {report['suite']} Evaluation",
        "",
        f"- Program: `{report['program']}`",
        f"- Passed: {summary['passed']}",
        f"- Failed: {summary['failed']}",
        f"- Skipped: {summary['skipped']}",
        f"- XFailed: {summary['xfailed']}",
        f"- XPassed: {summary['xpassed']}",
        f"- Active pass rate: {summary['active_pass_rate'] if summary['active_pass_rate'] is not None else 'n/a'}",
        "",
        "## Results",
        "",
        "| Status | Case | Category | Checks |",
        "| --- | --- | --- | --- |",
    ]
    for result in report["results"]:
        passed = sum(1 for check in result["checks"] if check["passed"])
        total = len(result["checks"])
        lines.append(f"| {result['status']} | `{result['id']}` | {result['category']} | {passed}/{total} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
