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


ALLOWED_METADATA_FIELDS = {
    "source_id",
    "source_path",
    "source_format",
    "source_name",
    "content_hash",
    "metadata_version",
    "chunk_type",
    "chunk_id",
    "program",
    "paragraph",
    "section",
    "indexable",
    "thin_chunk",
    "parse_quality",
    "schema_version",
    "pipeline_version",
}

PIPELINE_METADATA_FIELDS = {
    "source_id",
    "source_path",
    "source_format",
    "source_name",
    "content_hash",
    "metadata_version",
}

SCALAR_METADATA_TYPES = (str, int, float, bool)

TEXT_ENRICHMENT_FIELDS = {
    "copybooks_used",
    "resolved_copybooks",
    "stubbed_copybook_count",
    "stubbed_copybooks",
    "total_copybooks",
}


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
        self.include_non_indexable: bool = config.index.include_non_indexable

    def can_load(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() == ".json"

    def load(self, path: Path) -> list[LoadedDocument]:
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except json.JSONDecodeError as error:
            raise LoaderError(f"Invalid JSON in {path}: {error}") from error

        records = data if isinstance(data, list) else [data]
        loaded = []
        for index, record in enumerate(records):
            metadata = self._extract_metadata(record)
            if not self.include_non_indexable and metadata.get("indexable") is False:
                continue
            text = self._extract_text(record)
            text = self._append_structured_text(text, record)
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

        result: dict[str, Any] = {}

        # Only small scalar fields should become LlamaIndex metadata.
        # Large COBOL analysis structures must remain in document text, not metadata.
        for field in self.metadata_fields:
            if field in record and self._is_allowed_metadata(field, record[field]):
                result[field] = record[field]

        # Merge only whitelisted scalar fields from nested 'metadata' dict.
        # Nested keys take precedence over same-named top-level fields.
        nested = record.get("metadata")
        if isinstance(nested, dict):
            for field, value in nested.items():
                if self._is_allowed_metadata(field, value):
                    result[field] = value

        return result

    def _is_allowed_metadata(self, field: str, value: Any) -> bool:
        if field in PIPELINE_METADATA_FIELDS:
            return False
        return field in ALLOWED_METADATA_FIELDS and isinstance(
            value,
            SCALAR_METADATA_TYPES,
        )

    def _append_structured_text(self, text: str, record: Any) -> str:
        if not isinstance(record, dict):
            return text

        nested = record.get("metadata")
        if not isinstance(nested, dict):
            return text

        lines = []
        for field in sorted(TEXT_ENRICHMENT_FIELDS):
            value = nested.get(field)
            rendered = self._render_structured_fact(value)
            if rendered:
                lines.append(f"- {field}: {rendered}")

        if not lines:
            return text

        return "\n\nStructured facts from source JSON:\n" + "\n".join(lines) if not text else (
            text.rstrip() + "\n\nStructured facts from source JSON:\n" + "\n".join(lines)
        )

    def _render_structured_fact(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, SCALAR_METADATA_TYPES):
            return str(value)
        if isinstance(value, list):
            rendered_items = [self._render_structured_fact(item) for item in value]
            rendered_items = [item for item in rendered_items if item]
            return ", ".join(rendered_items)
        return ""
