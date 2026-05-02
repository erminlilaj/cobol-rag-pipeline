from __future__ import annotations

from pathlib import Path

from cobol_rag.loaders.base import LoadedDocument, LoaderError, make_document


class PlainTextLoader:
    name = "plain_text"
    suffixes = {
        ".cbl",
        ".cob",
        ".cpy",
        ".jcl",
        ".md",
        ".markdown",
        ".text",
        ".txt",
    }

    def can_load(self, path: Path) -> bool:
        return path.is_file() and path.suffix.lower() in self.suffixes

    def load(self, path: Path) -> list[LoadedDocument]:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            raise LoaderError(f"Could not read {path} as UTF-8 text") from error

        source_id = f"{self.name}:{path}"
        document = make_document(
            text=text,
            source_path=path,
            source_format=self.name,
            source_id=source_id,
            extra_metadata={"file_extension": path.suffix.lower()},
        )
        return [
            LoadedDocument(
                document=document,
                loader_name=self.name,
                source_path=path,
            )
        ]
