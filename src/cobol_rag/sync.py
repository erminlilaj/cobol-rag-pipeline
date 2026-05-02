from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.loaders import LoadedDocument, load_path


@dataclass(frozen=True)
class ManifestEntry:
    source_id: str
    source_path: str
    source_format: str
    content_hash: str


@dataclass(frozen=True)
class SyncItem:
    action: str
    source_id: str
    source_path: str
    source_format: str
    content_hash: str


@dataclass(frozen=True)
class SyncPlan:
    collection: str
    inbox_dir: Path
    manifest_path: Path
    dry_run: bool
    items: list[SyncItem] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)

    @property
    def total_documents(self) -> int:
        return len(self.items)

    def count(self, action: str) -> int:
        return sum(1 for item in self.items if item.action == action)


def build_sync_plan(config: AppConfig, dry_run: bool = True) -> SyncPlan:
    manifest_path = get_manifest_path(config)
    manifest = read_manifest(manifest_path)
    loaded = load_path(config.paths.inbox_dir, config=config)
    items = [
        _plan_item(document=doc, manifest=manifest)
        for doc in loaded
    ]

    return SyncPlan(
        collection=config.index.collection,
        inbox_dir=config.paths.inbox_dir,
        manifest_path=manifest_path,
        dry_run=dry_run,
        items=items,
    )


def get_manifest_path(config: AppConfig) -> Path:
    return config.paths.manifest_dir / f"{config.index.collection}.json"


def read_manifest(path: Path) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    sources = raw.get("sources", {})
    manifest: dict[str, ManifestEntry] = {}
    for source_id, entry in sources.items():
        manifest[source_id] = ManifestEntry(
            source_id=source_id,
            source_path=entry["source_path"],
            source_format=entry["source_format"],
            content_hash=entry["content_hash"],
        )
    return manifest


def _plan_item(
    *,
    document: LoadedDocument,
    manifest: dict[str, ManifestEntry],
) -> SyncItem:
    metadata = document.document.metadata
    source_id = str(metadata["source_id"])
    source_path = str(metadata["source_path"])
    source_format = str(metadata["source_format"])
    content_hash = str(metadata["content_hash"])
    previous = manifest.get(source_id)

    if previous is None:
        action = "add"
    elif previous.content_hash != content_hash:
        action = "update"
    else:
        action = "skip"

    return SyncItem(
        action=action,
        source_id=source_id,
        source_path=source_path,
        source_format=source_format,
        content_hash=content_hash,
    )
