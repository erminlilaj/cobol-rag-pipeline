from __future__ import annotations

import hashlib
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

from cobol_rag.bundle import find_bm25_index, list_bundle_chunks, resolve_index_path
from cobol_rag.config import AppConfig
from cobol_rag.index import delete_source, open_index, upsert_documents
from cobol_rag.manifest_store import atomic_write_manifest, bm25_reference, source_records
from cobol_rag.loaders import LoadedDocument, LoaderError, load_path
from cobol_rag.sync_progress import SyncProgress


@dataclass(frozen=True, slots=True)
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
    *,
    progress: SyncProgress | None = None,
) -> SyncPlan:
    """Build a sync plan from config inbox or an explicit target path.

    When target is a knowledge-base_rag bundle (contains manifest.json), the
    recommended_index_path is resolved and only the files listed in
    chunks_manifest.json are loaded.
    """
    manifest_path = get_manifest_path(config)
    failures: list[str] = []
    bm25_path: Path | None = None

    sync_root = target or config.paths.inbox_dir
    with progress.phase("load incoming documents") if progress else nullcontext():
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

    with progress.phase("read existing manifest") if progress else nullcontext():
        # Only incoming identities and records under this synchronization root
        # can affect this plan. Do not retain the entire shared corpus in RAM.
        wanted = {str(doc.document.metadata['source_id']) for doc in loaded}
        manifest = read_manifest(manifest_path, source_ids=wanted, sync_root=index_path)
        pending = read_manifest(_journal_path(manifest_path, sync_root))
        # A previous vector write may not match the committed manifest. Keep
        # orphan additions visible for removal and never skip a touched source.
        for source_id, entry in pending.items():
            manifest.setdefault(source_id, entry)
    with progress.phase("compare incoming documents with manifest") if progress else nullcontext():
        current_items = [_plan_item(document=doc, manifest=manifest) for doc in loaded]
        current_items = [replace(item, action='update') if item.action == 'skip'
                         and item.source_id in pending else item for item in current_items]
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


def apply_sync_plan(config: AppConfig, plan: SyncPlan, *, progress: SyncProgress | None = None) -> None:
    journal = _journal_path(plan.manifest_path, plan.inbox_dir)
    touched = [item for item in plan.items if item.action != 'skip']
    same_bm25 = plan.bm25_index_path is None or str(plan.bm25_index_path) == bm25_reference(plan.manifest_path)
    if not touched and not journal.exists() and plan.manifest_path.exists() and same_bm25:
        with progress.phase('apply changes', total=0) if progress else nullcontext():
            pass
        if progress:
            progress.note('Unchanged sources: no vectors or manifest rewritten.')
        return
    if touched:
        atomic_write_manifest(journal, plan.collection,
            ((item.source_id, _manifest_record(item)) for item in touched))
    with progress.phase("open Chroma") if progress else nullcontext():
        resources = open_index(config, measure=progress.measure) if progress else open_index(config)
    total = None if progress is None else sum(
        item.action == "remove" or (item.action in {"add", "update"} and item.loaded_document is not None)
        for item in plan.items
    )
    with progress.phase("apply changes", total=total) if progress else nullcontext():
        pending = []
        def flush():
            if pending:
                upsert_documents(resources, [item.loaded_document.document for item in pending],
                                 chunk_mode=config.index.chunk_mode)
                if progress:
                    for _ in pending:
                        progress.advance()
                pending.clear()
        for item in plan.items:
            if item.action in {"add", "update"} and item.loaded_document is not None:
                pending.append(item)
                if len(pending) >= 32:
                    flush()
            elif item.action == "remove":
                flush()
                delete_source(resources, item.source_id)
                if progress:
                    progress.advance()
            else:
                continue
        flush()
    with progress.phase("save manifest") if progress else nullcontext():
        write_manifest(plan.manifest_path, plan)
        journal.unlink(missing_ok=True)


def get_manifest_path(config: AppConfig) -> Path:
    return config.paths.manifest_dir / f"{config.index.collection}.json"


def _journal_path(manifest: Path, scope: Path) -> Path:
    key = hashlib.sha256(str(scope.resolve()).encode('utf-8')).hexdigest()[:20]
    return manifest.with_name(f'{manifest.name}.{key}.pending.json')


def _manifest_record(item):
    return dict(source_id=item.source_id, source_path=item.source_path,
                source_format=item.source_format, content_hash=item.content_hash)


def read_manifest(path: Path, *, source_ids: set[str] | None = None,
                  sync_root: Path | None = None) -> dict[str, ManifestEntry]:
    if not path.exists():
        return {}
    under_root = lru_cache(maxsize=4096)(lambda name: _is_under_path(Path(name), sync_root))
    manifest: dict[str, ManifestEntry] = {}
    for source_id, entry in source_records(path):
        if source_ids is not None and source_id not in source_ids and not (
                sync_root is not None and under_root(entry['source_path'])):
            continue
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
    updates = {item.source_id: item for item in plan.items}
    def merged_records():
        existing = (existing_raw.get('sources', {}).items() if existing_raw is not None
                    else source_records(path))
        for source_id, entry in existing:
            item = updates.pop(source_id, None)
            if item is None:
                yield source_id, entry
            elif item.action != 'remove':
                yield source_id, _manifest_record(item)
        for source_id, item in updates.items():
            if item.action != 'remove':
                yield source_id, _manifest_record(item)
    bm25_str = (
        str(plan.bm25_index_path)
        if plan.bm25_index_path is not None
        else (existing_raw.get("bm25_index_path") if existing_raw is not None else bm25_reference(path))
    )
    atomic_write_manifest(path, plan.collection, merged_records(), bm25_str)


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
    elif (previous.content_hash != content_hash or previous.source_path != source_path
          or previous.source_format != source_format):
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
    under_root = lru_cache(maxsize=4096)(lambda name: _is_under_path(Path(name), sync_root))
    for entry in manifest.values():
        if entry.source_id in current_source_ids:
            continue
        if not under_root(entry.source_path):
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
