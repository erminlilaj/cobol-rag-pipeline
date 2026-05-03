from __future__ import annotations

from pathlib import Path

from cobol_rag.config import AppConfig
from cobol_rag.loaders.base import LoadedDocument, LoaderAdapter, LoaderError
from cobol_rag.loaders.generic_json import GenericJsonLoader
from cobol_rag.loaders.plain_text import PlainTextLoader
from cobol_rag.loaders.rag_documents import RagDocumentsLoader


def list_loaders(config: AppConfig) -> list[LoaderAdapter]:
    return [
        RagDocumentsLoader(config=config),
        GenericJsonLoader(config=config),
        PlainTextLoader(),
    ]


def get_loader(
    path: Path,
    config: AppConfig,
    loader_name: str | None = None,
) -> LoaderAdapter:
    loaders = list_loaders(config)
    if loader_name:
        for loader in loaders:
            if loader.name == loader_name:
                return loader
        names = ", ".join(loader.name for loader in loaders)
        raise LoaderError(f"Unknown loader '{loader_name}'. Available loaders: {names}")

    for loader in loaders:
        if loader.can_load(path):
            return loader

    names = ", ".join(loader.name for loader in loaders)
    raise LoaderError(f"No loader found for {path}. Available loaders: {names}")


def load_path(
    path: Path,
    config: AppConfig,
    loader_name: str | None = None,
) -> list[LoadedDocument]:
    if path.is_dir():
        loaded: list[LoadedDocument] = []
        for child in sorted(child for child in path.rglob("*") if child.is_file()):
            try:
                loader = get_loader(child, config=config, loader_name=loader_name)
            except LoaderError:
                continue
            loaded.extend(loader.load(child))
        return loaded

    loader = get_loader(path, config=config, loader_name=loader_name)
    return loader.load(path)
