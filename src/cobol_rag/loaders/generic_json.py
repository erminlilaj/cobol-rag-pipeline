from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.loaders.base import (
    LoadedDocument,
    LoaderError,
    make_document,
    stable_json,
)


class GenericJsonLoader:
    name = "generic_json"

    def __init__(self, config: AppConfig) -> None:
        loader_config = config.raw.get("loaders", {}).get("generic_json", {})
        self.text_fields = tuple(
            loader_config.get(
                "text_fields",
                ["text", "content", "summary", "description"],
            )
        )
        self.metadata_fields = tuple(
            loader_config.get(
                "metadata_fields",
                ["title", "name", "kind", "section"],
            )
        )

    def can_load(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in {".json", ".jsonl"}

    def load(self, path: Path) -> list[LoadedDocument]:
        records = self._read_records(path)
        loaded = []
        for index, record in enumerate(records):
            text = self._extract_text(record)
            metadata = self._extract_metadata(record)
            source_id = f"{self.name}:{path}:{index}"
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

    def _read_records(self, path: Path) -> list[Any]:
        if path.suffix.lower() == ".jsonl":
            return self._read_jsonl(path)
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise LoaderError(f"Invalid JSON in {path}: {error}") from error

        return data if isinstance(data, list) else [data]

    def _read_jsonl(self, path: Path) -> list[Any]:
        records = []
        try:
            with path.open("r", encoding="utf-8") as file:
                for line_number, line in enumerate(file, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as error:
                        raise LoaderError(f"Invalid JSONL in {path} line {line_number}: {error}") from error
        except OSError as error:
            raise LoaderError(f"Could not read {path}: {error}") from error
        return records

    def _extract_text(self, record: Any) -> str:
        if isinstance(record, dict):
            for field in self.text_fields:
                value = record.get(field)
                if isinstance(value, str) and value.strip():
                    return value
            return stable_json(record)
        if isinstance(record, str):
            return record
        return stable_json(record)

    def _extract_metadata(self, record: Any) -> dict[str, Any]:
        if not isinstance(record, dict):
            return {}
        result = {
            field: record.get(field)
            for field in self.metadata_fields
            if field in record
        }
        nested = record.get("metadata")
        nested = nested if isinstance(nested, dict) else {}
        if "type" in record and "chunk_type" not in result:
            result["chunk_type"] = record["type"]
        if "id" in record and "chunk_id" not in result:
            result["chunk_id"] = record["id"]
        for key in (
            "source_system",
            "source_chunk_type",
            "coverage_dimension",
            "entity_type",
            "entity_key",
            "target",
            "call_type",
            "variable",
            "paragraph",
            "line",
            "title",
            "source_kind",
        ):
            if key in record and key not in result:
                result[key] = record[key]
            if key in nested and key not in result:
                result[key] = nested[key]
        return {
            key: value
            for key, value in result.items()
            if isinstance(value, str | int | float | bool)
        }
