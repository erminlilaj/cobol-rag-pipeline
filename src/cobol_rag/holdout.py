from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobol_rag.config import load_config
from cobol_rag.evaluation import evaluate_cases, load_gold_cases


def verify_holdout_manifest(suite_path: Path, manifest_path: Path) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = str(manifest.get("sha256", "")).strip().lower()
    actual = hashlib.sha256(suite_path.read_bytes()).hexdigest()
    if not expected or actual != expected:
        raise ValueError(
            "The sealed holdout checksum does not match its manifest. "
            "Create a new versioned suite instead of editing the existing holdout in place."
        )
    return manifest


def save_holdout_report(
    report: dict[str, Any],
    output_dir: Path,
    manifest: dict[str, Any],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suite_id = str(manifest.get("suite_id", "holdout"))
    payload = {**report, "holdout_manifest": manifest}
    json_path = output_dir / f"holdout-eval-{suite_id}-{stamp}.json"
    markdown_path = output_dir / f"holdout-eval-{suite_id}-{stamp}.md"
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    lines = [
        "# Sealed COBOL RAG Holdout Evaluation",
        "",
        f"- Suite: {suite_id}",
        f"- Cases: {metrics['cases']}",
        f"- Passed: {metrics['passed']}",
        f"- Failed: {metrics['failed']}",
        f"- Pass rate: {metrics['pass_rate']:.1%}",
        "",
        "The suite checksum was verified before execution. Results must not be used "
        "to patch this suite version; create a new development case instead.",
    ]
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a sealed holdout suite exactly once for generalization measurement."
    )
    parser.add_argument("--config", type=Path, default=Path("config/default.yaml"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--final-scripts-dir", type=Path)
    parser.add_argument(
        "--acknowledge-sealed-suite",
        action="store_true",
        help="Confirm that results will not be used to tune this holdout version.",
    )
    args = parser.parse_args()
    if not args.acknowledge_sealed_suite:
        parser.error("--acknowledge-sealed-suite is required")
    manifest = verify_holdout_manifest(args.suite, args.manifest)
    if args.final_scripts_dir:
        os.environ["COBOL_RAG_FINAL_SCRIPTS_DIR"] = str(args.final_scripts_dir.resolve())
    cases = load_gold_cases(args.suite, allow_holdout=True)
    report = evaluate_cases(cases, load_config(args.config))
    json_path, markdown_path = save_holdout_report(report, args.output_dir, manifest)
    print(json.dumps(report["metrics"], indent=2))
    print(f"JSON: {json_path}")
    print(f"Markdown: {markdown_path}")
    return 0 if report["metrics"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
