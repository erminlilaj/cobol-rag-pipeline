from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig


ALLOWED_FEEDBACK_LABELS = {
    "wrong_program",
    "wrong_entity",
    "missing_identifier",
    "bad_chunking",
    "weak_rerank",
    "missing_parent",
    "hallucination",
    "failed_abstention",
    "missing_evidence",
    "other",
}


def write_answer_trace(config: AppConfig, payload: dict[str, Any]) -> str | None:
    if not config.observability.enabled:
        return None
    trace_id = uuid.uuid4().hex
    record = {
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }
    _write_json(config.paths.trace_dir / f"{trace_id}.json", record)
    return trace_id


def load_trace(config: AppConfig, trace_id: str) -> dict[str, Any] | None:
    safe_id = _safe_id(trace_id)
    path = config.paths.trace_dir / f"{safe_id}.json"
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def list_traces(config: AppConfig, limit: int = 20) -> list[dict[str, Any]]:
    if not config.paths.trace_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(
        config.paths.trace_dir.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[: max(1, min(limit, 200))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def write_feedback(
    config: AppConfig,
    *,
    trace_id: str,
    rating: str,
    labels: list[str],
    note: str = "",
) -> dict[str, Any]:
    if rating not in {"correct", "partial", "incorrect"}:
        raise ValueError("rating must be correct, partial, or incorrect")
    invalid = sorted(set(labels) - ALLOWED_FEEDBACK_LABELS)
    if invalid:
        raise ValueError("unsupported feedback labels: " + ", ".join(invalid))
    trace = load_trace(config, trace_id)
    if trace is None:
        raise ValueError("trace_id does not exist")
    feedback_id = uuid.uuid4().hex
    record = {
        "feedback_id": feedback_id,
        "trace_id": trace_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rating": rating,
        "labels": sorted(set(labels)),
        "note": note.strip()[:2000],
        "question": trace.get("question", ""),
        "route": trace.get("route", ""),
        "scope": trace.get("scope", {}),
        "source_ids": trace.get("answer", {}).get("source_ids", []),
    }
    _write_json(config.paths.feedback_dir / f"{feedback_id}.json", record)
    return record


def list_feedback(config: AppConfig, limit: int = 50) -> list[dict[str, Any]]:
    if not config.paths.feedback_dir.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(
        config.paths.feedback_dir.glob("*.json"),
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )[: max(1, min(limit, 500))]:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            records.append(payload)
    return records


def _safe_id(value: str) -> str:
    if not value or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError("invalid id")
    return value.lower()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
