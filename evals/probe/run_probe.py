"""Run the diagnostic probe suite against a live RAG API and print a scorecard.

This is a development diagnostic, not a sealed evaluation. It reports what the
system actually did (route, intent, execution mode, latency) next to loose
content expectations, so paraphrase brittleness and slow paths are visible per
cluster rather than hidden behind a single pass rate.

Usage:
    python evals/probe/run_probe.py --suite evals/probe/pdcbvc_probe_v1.jsonl \
        --api http://localhost:8000 --program PDCBVC --out report.json
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def post(api: str, path: str, payload: dict[str, Any] | None, timeout: float) -> dict[str, Any]:
    data = json.dumps(payload or {}).encode("utf-8")
    request = urllib.request.Request(
        f"{api.rstrip('/')}{path}", data=data,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def check_text(answer: str, must_include: list[str], must_not_include: list[str]) -> list[str]:
    lowered = answer.lower()
    problems = []
    for term in must_include:
        if str(term).lower() not in lowered:
            problems.append(f"missing:{term}")
    for term in must_not_include:
        if str(term).lower() in lowered:
            problems.append(f"forbidden:{term}")
    return problems


def ask(api: str, question: str, program: str | None, timeout: float) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        body: dict[str, Any] = {"message": question}
        if program:
            body["program"] = program
        result = post(api, "/api/chat", body, timeout)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return {"answer": "", "route": "ERROR", "error": str(error)}, time.perf_counter() - started
    return result, time.perf_counter() - started


def reset_session(api: str) -> str | None:
    """Isolate a case from prior session state, reporting rather than aborting."""
    try:
        post(api, "/api/chat/reset", {}, 60)
    except (urllib.error.URLError, TimeoutError, OSError) as error:
        return f"reset_failed:{type(error).__name__}"
    return None


def run_case(case: dict[str, Any], api: str, program: str | None, timeout: float) -> dict[str, Any]:
    reset_problem = reset_session(api)
    turns = case.get("turns") or [case]
    records: list[dict[str, Any]] = []
    problems: list[str] = [reset_problem] if reset_problem else []
    total_latency = 0.0
    for index, turn in enumerate(turns, start=1):
        result, latency = ask(api, turn["question"], program, timeout)
        total_latency += latency
        answer = str(result.get("answer") or "")
        plan = result.get("plan") or {}
        turn_problems = check_text(
            answer,
            list(turn.get("must_include", [])),
            list(turn.get("must_not_include", [])),
        )
        if result.get("route") == "ERROR":
            turn_problems.append(f"request_failed:{result.get('error', '')[:80]}")
        expected_route = case.get("expected_route") if len(turns) == 1 else None
        if expected_route and result.get("route") != expected_route:
            turn_problems.append(f"route:{result.get('route')}!={expected_route}")
        expected_mode = case.get("expected_mode") if len(turns) == 1 else None
        if expected_mode and result.get("execution_mode") != expected_mode:
            turn_problems.append(f"mode:{result.get('execution_mode')}!={expected_mode}")
        problems.extend(f"t{index}:{item}" for item in turn_problems)
        records.append({
            "turn": index,
            "question": turn["question"],
            "route": result.get("route"),
            "intent": plan.get("intent"),
            "tasks": plan.get("tasks"),
            "execution_mode": result.get("execution_mode"),
            "capabilities": [s.get("capability") for s in plan.get("subtasks", [])],
            "latency_s": round(latency, 1),
            "answer": answer,
        })
    return {
        "id": case["id"],
        "cluster": case.get("cluster", "uncategorized"),
        "passed": not problems,
        "problems": problems,
        "latency_s": round(total_latency, 1),
        "notes": case.get("notes", ""),
        "expect": case.get("expect", ""),
        "turns": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", required=True)
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--program", default="PDCBVC")
    parser.add_argument("--timeout", type=float, default=420.0)
    parser.add_argument("--out", default="")
    parser.add_argument("--only", default="", help="comma-separated case ids or clusters")
    args = parser.parse_args()

    suite_path = Path(args.suite)
    if "holdout" in suite_path.parts:
        raise SystemExit("Refusing to run a sealed holdout suite through the probe runner.")
    cases = [
        json.loads(line) for line in suite_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selector = {value.strip() for value in args.only.split(",") if value.strip()}
    if selector:
        cases = [
            case for case in cases
            if case["id"] in selector or case.get("cluster") in selector
        ]

    results = []
    for position, case in enumerate(cases, start=1):
        print(f"[{position}/{len(cases)}] {case['id']} ...", flush=True)
        try:
            result = run_case(case, args.api, args.program, args.timeout)
        except Exception as error:  # never lose a whole run to one bad case
            print(f"    ERROR {type(error).__name__}: {error}", flush=True)
            results.append({
                "id": case["id"], "cluster": case.get("cluster", "uncategorized"),
                "passed": False, "problems": [f"runner_error:{type(error).__name__}"],
                "latency_s": 0.0, "notes": case.get("notes", ""),
                "expect": case.get("expect", ""),
                "turns": [{
                    "turn": 1, "question": case.get("question", ""), "route": "ERROR",
                    "intent": None, "tasks": None, "execution_mode": "error",
                    "capabilities": [], "latency_s": 0.0, "answer": "",
                }],
            })
            continue
        state = "PASS" if result["passed"] else "FAIL"
        print(
            f"    {state} {result['latency_s']}s "
            f"route={result['turns'][0]['route']} "
            f"intent={result['turns'][0]['intent']} "
            f"mode={result['turns'][0]['execution_mode']}",
            flush=True,
        )
        if result["problems"]:
            print(f"    problems: {', '.join(result['problems'])}", flush=True)
        results.append(result)

    print("\n=== SCORECARD BY CLUSTER ===")
    clusters: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        clusters.setdefault(result["cluster"], []).append(result)
    for cluster, items in sorted(clusters.items()):
        passed = sum(1 for item in items if item["passed"])
        latencies = [item["latency_s"] for item in items]
        median = sorted(latencies)[len(latencies) // 2] if latencies else 0.0
        print(f"{cluster:22s} {passed}/{len(items):<3d} median {median:6.1f}s")
    total_passed = sum(1 for item in results if item["passed"])
    all_latencies = sorted(item["latency_s"] for item in results)
    print(f"\nTOTAL {total_passed}/{len(results)} passed")
    if all_latencies:
        print(
            f"latency  median {all_latencies[len(all_latencies) // 2]:.1f}s  "
            f"max {all_latencies[-1]:.1f}s  total {sum(all_latencies) / 60:.1f} min"
        )

    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nDetailed report: {args.out}")


if __name__ == "__main__":
    main()
