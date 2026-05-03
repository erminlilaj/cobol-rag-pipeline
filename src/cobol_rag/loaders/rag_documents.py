from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.loaders.base import (
    LoadedDocument,
    LoaderError,
    content_hash,
    make_document,
    stable_json,
)


SCALAR_METADATA_TYPES = (str, int, float, bool)

PIPELINE_METADATA_FIELDS = {
    "source_id",
    "source_path",
    "source_format",
    "source_name",
    "metadata_version",
}

ALLOWED_METADATA_FIELDS = {
    "program",
    "chunk_type",
    "chunk_id",
    "title",
    "source_file",
    "source_kind",
    "json_index",
    "chunk_index",
    "chunk_count",
    "indexable",
    "schema_version",
    "pipeline_version",
}


class RagDocumentsLoader:
    """Load control_flow RAG factory documents.

    The factory emits records shaped like:
    {"id", "program", "type", "title", "text", "metadata"}.

    This loader keeps the prebuilt text unchanged and promotes `type` to
    `chunk_type`, which the retriever uses for filtering and intent reranking.
    """

    name = "rag_documents"

    def __init__(self, config: AppConfig) -> None:
        self.include_non_indexable = config.index.include_non_indexable

    def can_load(self, path: Path) -> bool:
        if not path.is_file():
            return False
        name = path.name.lower()
        return name in {"rag_documents.json", "rag_documents.jsonl"} or path.suffix.lower() == ".jsonl"

    def load(self, path: Path) -> list[LoadedDocument]:
        records = self._read_records(path)
        loaded: list[LoadedDocument] = []
        seen_source_ids: set[str] = set()

        for index, record in enumerate(records):
            metadata = self._extract_metadata(record)
            if not self.include_non_indexable and metadata.get("indexable") is False:
                continue

            text = record.get("text")
            if not isinstance(text, str) or not text.strip():
                raise LoaderError(f"{path} record {index} is missing non-empty text")

            source_id = self._source_id(record, index, seen_source_ids)
            record_id = self._record_id(record)
            if record_id:
                metadata.setdefault("chunk_id", record_id)
                metadata.setdefault("factory_record_id", record_id)

            document = make_document(
                text=text,
                source_path=path,
                source_format=self.name,
                source_id=source_id,
                extra_metadata=metadata,
            )
            loaded.append(
                LoadedDocument(
                    document=document,
                    loader_name=self.name,
                    source_path=path,
                )
            )
        return loaded

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        if path.suffix.lower() == ".jsonl":
            return self._read_jsonl(path)
        return self._read_json(path)

    def _read_jsonl(self, path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise LoaderError(f"Invalid JSONL in {path} line {line_number}: {error}") from error
                    if not isinstance(record, dict):
                        raise LoaderError(f"{path} line {line_number} must contain a JSON object")
                    records.append(record)
        except OSError as error:
            raise LoaderError(f"Could not read {path}: {error}") from error
        return records

    def _read_json(self, path: Path) -> list[dict[str, Any]]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise LoaderError(f"Invalid JSON in {path}: {error}") from error
        except OSError as error:
            raise LoaderError(f"Could not read {path}: {error}") from error

        if isinstance(data, list):
            records = data
        elif isinstance(data, dict):
            records = [data]
        else:
            raise LoaderError(f"{path} must contain a JSON object or list of objects")

        bad_index = next((index for index, item in enumerate(records) if not isinstance(item, dict)), None)
        if bad_index is not None:
            raise LoaderError(f"{path} record {bad_index} must contain a JSON object")
        return records

    def _extract_metadata(self, record: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        nested = record.get("metadata")
        nested = nested if isinstance(nested, dict) else {}

        program = self._scalar(record.get("program")) or self._scalar(nested.get("program"))
        if program:
            result["program"] = program.upper()

        doc_type = (
            self._scalar(record.get("chunk_type"))
            or self._scalar(record.get("type"))
            or self._scalar(nested.get("chunk_type"))
            or self._scalar(nested.get("type"))
        )
        if doc_type:
            result["chunk_type"] = doc_type

        title = self._scalar(record.get("title")) or self._scalar(nested.get("title"))
        if title:
            result["title"] = title

        for key, value in nested.items():
            if key == "source_id":
                if self._scalar(value):
                    result["factory_source_id"] = value
                continue
            if key == "content_hash":
                if self._scalar(value):
                    result["factory_content_hash"] = value
                continue
            if key in PIPELINE_METADATA_FIELDS:
                continue
            if key == "type":
                if self._scalar(value):
                    result.setdefault("chunk_type", value)
                continue
            if key in ALLOWED_METADATA_FIELDS and isinstance(value, SCALAR_METADATA_TYPES):
                result[key] = value

        return result

    def _source_id(self, record: dict[str, Any], index: int, seen: set[str]) -> str:
        record_id = self._record_id(record)
        if not record_id:
            record_id = content_hash(stable_json(record))[:24]
        base = f"{self.name}:{record_id}"
        source_id = base
        if source_id in seen:
            source_id = f"{base}:{index}"
        seen.add(source_id)
        return source_id

    def _record_id(self, record: dict[str, Any]) -> str:
        value = self._scalar(record.get("id"))
        if value:
            return value
        nested = record.get("metadata")
        if isinstance(nested, dict):
            return self._scalar(nested.get("chunk_id")) or self._scalar(nested.get("source_id"))
        return ""

    def _scalar(self, value: Any) -> str:
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, int | float | bool):
            return str(value)
        return ""
