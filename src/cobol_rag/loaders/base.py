from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from llama_index.core import Document


METADATA_VERSION = "1"


class LoaderError(Exception):
    """Raised when a loader cannot safely parse a source."""


@dataclass(frozen=True)
class LoadedDocument:
    document: Document
    loader_name: str
    source_path: Path


class LoaderAdapter(Protocol):
    name: str

    def can_load(self, path: Path) -> bool:
        ...

    def load(self, path: Path) -> list[LoadedDocument]:
        ...


def content_hash(value: str | bytes) -> str:
    if isinstance(value, str):
        value = value.encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def make_document(
    *,
    text: str,
    source_path: Path,
    source_format: str,
    source_id: str,
    extra_metadata: dict[str, Any] | None = None,
) -> Document:
    metadata = {
        "source_id": source_id,
        "source_path": str(source_path),
        "source_format": source_format,
        "source_name": source_path.name,
        "content_hash": content_hash(text),
        "metadata_version": METADATA_VERSION,
    }
    if extra_metadata:
        metadata.update(_clean_metadata(extra_metadata))
    return Document(
        text=text,
        metadata=metadata,
        id_=source_id,
        excluded_embed_metadata_keys=list(metadata.keys()),
    )


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            clean[key] = value
        else:
            clean[key] = stable_json(value)
    return clean
