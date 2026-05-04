from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from cobol_rag.bundle import find_bm25_index, list_bundle_chunks, resolve_index_path
from cobol_rag.config import AppConfig
from cobol_rag.index import open_index, upsert_document
from cobol_rag.loaders import LoadedDocument, LoaderError, load_path


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
    loaded_document: LoadedDocument | None = field(default=None, repr=False)


@dataclass(frozen=True)
class SyncPlan:
    collection: str
    inbox_dir: Path
    manifest_path: Path
    dry_run: bool
    items: list[SyncItem] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)
    bm25_index_path: Path | None = None

    @property
    def total_documents(self) -> int:
        return len(self.items)

    def count(self, action: str) -> int:
        return sum(1 for item in self.items if item.action == action)


def build_sync_plan(
    config: AppConfig,
    dry_run: bool = True,
    target: Path | None = None,
) -> SyncPlan:
    """Build a sync plan from config inbox or an explicit target path.

    When target is a knowledge-base_rag bundle (contains manifest.json), the
    recommended_index_path is resolved and only the files listed in
    chunks_manifest.json are loaded.
    """
    manifest_path = get_manifest_path(config)
    manifest = read_manifest(manifest_path)
    failures: list[str] = []
    bm25_path: Path | None = None

    sync_root = target or config.paths.inbox_dir
    index_path = resolve_index_path(sync_root)

    if target is not None:
        chunk_files = list_bundle_chunks(index_path)
        if chunk_files is not None:
            loaded: list[LoadedDocument] = []
            for file_path in chunk_files:
                try:
                    loaded.extend(load_path(file_path, config=config))
                except LoaderError as err:
                    failures.append(str(err))
        else:
            loaded = load_path(index_path, config=config)
        bm25_path = find_bm25_index(index_path)
    else:
        loaded = load_path(config.paths.inbox_dir, config=config)

    current_items = [_plan_item(document=doc, manifest=manifest) for doc in loaded]
    current_source_ids = {item.source_id for item in current_items}
    remove_items = _obsolete_items(
        manifest=manifest,
        sync_root=index_path,
        current_source_ids=current_source_ids,
    )
    items = current_items + remove_items

    return SyncPlan(
        collection=config.index.collection,
        inbox_dir=sync_root,
        manifest_path=manifest_path,
        dry_run=dry_run,
        items=items,
        failed=failures,
        bm25_index_path=bm25_path,
    )


def apply_sync_plan(config: AppConfig, plan: SyncPlan) -> None:
    resources = open_index(config)
    for item in plan.items:
        if item.action in {"add", "update"}:
            if item.loaded_document is None:
                continue
            upsert_document(resources, item.loaded_document.document)
        elif item.action == "remove":
            resources.chroma_collection.delete(where={"source_id": item.source_id})
    existing_raw = _read_manifest_raw(plan.manifest_path)
    write_manifest(plan.manifest_path, plan, existing_raw=existing_raw)


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


def write_manifest(
    path: Path,
    plan: SyncPlan,
    existing_raw: dict | None = None,
) -> None:
    """Write manifest, merging plan items into any existing entries.

    Merging ensures that syncing a bundle does not wipe entries from a previous
    inbox sync, and vice-versa.  bm25_index_path is preserved from the existing
    manifest when the current plan does not set a new one.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_raw = existing_raw or {}

    # Start from existing sources so multi-source collections survive
    sources: dict[str, dict] = dict(existing_raw.get("sources", {}))

    for item in sorted(plan.items, key=lambda i: i.source_id):
        if item.action == "remove":
            sources.pop(item.source_id, None)
            continue
        sources[item.source_id] = {
            "source_id": item.source_id,
            "source_path": item.source_path,
            "source_format": item.source_format,
            "content_hash": item.content_hash,
        }

    payload: dict = {"collection": plan.collection, "sources": sources}

    bm25_str = (
        str(plan.bm25_index_path)
        if plan.bm25_index_path is not None
        else existing_raw.get("bm25_index_path")
    )
    if bm25_str:
        payload["bm25_index_path"] = bm25_str

    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def _read_manifest_raw(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


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
        loaded_document=document,
    )


def _obsolete_items(
    *,
    manifest: dict[str, ManifestEntry],
    sync_root: Path,
    current_source_ids: set[str],
) -> list[SyncItem]:
    items: list[SyncItem] = []
    for entry in manifest.values():
        if entry.source_id in current_source_ids:
            continue
        if not _is_under_path(Path(entry.source_path), sync_root):
            continue
        items.append(
            SyncItem(
                action="remove",
                source_id=entry.source_id,
                source_path=entry.source_path,
                source_format=entry.source_format,
                content_hash=entry.content_hash,
                loaded_document=None,
            )
        )
    return items


def _is_under_path(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError:
        return False
    return True
