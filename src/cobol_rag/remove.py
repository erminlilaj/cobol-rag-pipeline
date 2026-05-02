from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.index import delete_source, open_index
from cobol_rag.sync import ManifestEntry, get_manifest_path, read_manifest


@dataclass(frozen=True)
class RemovePlan:
    collection: str
    manifest_path: Path
    dry_run: bool
    entries: list[ManifestEntry] = field(default_factory=list)

    @property
    def total_documents(self) -> int:
        return len(self.entries)


def build_remove_plan(
    config: AppConfig,
    *,
    source_id: str | None = None,
    source_path: str | None = None,
    dry_run: bool = True,
) -> RemovePlan:
    if not source_id and not source_path:
        raise ValueError("Provide --source-id or --source-path")

    manifest_path = get_manifest_path(config)
    manifest = read_manifest(manifest_path)
    entries = [
        entry
        for entry in manifest.values()
        if _matches(entry, source_id=source_id, source_path=source_path)
    ]

    return RemovePlan(
        collection=config.index.collection,
        manifest_path=manifest_path,
        dry_run=dry_run,
        entries=sorted(entries, key=lambda entry: entry.source_id),
    )


def apply_remove_plan(config: AppConfig, plan: RemovePlan) -> None:
    resources = open_index(config)
    for entry in plan.entries:
        delete_source(resources, entry.source_id)
    _write_manifest_after_removal(plan.manifest_path, plan)


def _matches(
    entry: ManifestEntry,
    *,
    source_id: str | None,
    source_path: str | None,
) -> bool:
    if source_id and entry.source_id == source_id:
        return True
    if source_path and entry.source_path == source_path:
        return True
    return False


def _write_manifest_after_removal(path: Path, plan: RemovePlan) -> None:
    manifest = read_manifest(path)
    removed_ids = {entry.source_id for entry in plan.entries}
    remaining = [
        entry
        for entry in manifest.values()
        if entry.source_id not in removed_ids
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    sources = {
        entry.source_id: {
            "source_id": entry.source_id,
            "source_path": entry.source_path,
            "source_format": entry.source_format,
            "content_hash": entry.content_hash,
        }
        for entry in sorted(remaining, key=lambda entry: entry.source_id)
    }

    payload = {
        "collection": plan.collection,
        "sources": sources,
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")
