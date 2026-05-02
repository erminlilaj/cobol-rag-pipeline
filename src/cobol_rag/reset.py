from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from cobol_rag.config import AppConfig
from cobol_rag.index import collection_count, open_index
from cobol_rag.sync import get_manifest_path


@dataclass(frozen=True)
class ResetPlan:
    collection: str
    chroma_dir: Path
    manifest_path: Path
    dry_run: bool
    document_count: int
    manifest_exists: bool


def build_reset_plan(config: AppConfig, dry_run: bool = True) -> ResetPlan:
    resources = open_index(config)
    manifest_path = get_manifest_path(config)
    return ResetPlan(
        collection=config.index.collection,
        chroma_dir=config.paths.chroma_dir,
        manifest_path=manifest_path,
        dry_run=dry_run,
        document_count=collection_count(resources),
        manifest_exists=manifest_path.exists(),
    )


def apply_reset_plan(config: AppConfig, plan: ResetPlan) -> None:
    client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    existing = {collection.name for collection in client.list_collections()}
    if config.index.collection in existing:
        client.delete_collection(config.index.collection)
    client.get_or_create_collection(config.index.collection)

    if plan.manifest_path.exists():
        plan.manifest_path.unlink()
