from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PathConfig:
    chroma_dir: Path = Path(".chroma")
    inbox_dir: Path = Path("data/inbox")
    archive_dir: Path = Path("data/archive")
    manifest_dir: Path = Path("data/manifests")

    def __post_init__(self) -> None:
        object.__setattr__(self, "chroma_dir", Path(self.chroma_dir))
        object.__setattr__(self, "inbox_dir", Path(self.inbox_dir))
        object.__setattr__(self, "archive_dir", Path(self.archive_dir))
        object.__setattr__(self, "manifest_dir", Path(self.manifest_dir))


@dataclass(frozen=True)
class LlmConfig:
    provider: str = "ollama"
    model: str = "granite-code:8b-instruct"
    base_url: str = "http://localhost:11434"
    context_window: int = 4096
    request_timeout: int = 300
    temperature: float = 0.1


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "ollama"
    model: str = "mxbai-embed-large:latest"
    base_url: str = "http://localhost:11434"


@dataclass(frozen=True)
class IndexConfig:
    collection: str = "cobol-dev"
    chunk_mode: str = "pre_chunked"
    batch_size: int = 64


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 6
    filters: dict[str, Any] = field(default_factory=dict)
    similarity_cutoff: float | None = None


@dataclass(frozen=True)
class AnswerConfig:
    require_citations: bool = True
    show_sources: bool = True


@dataclass(frozen=True)
class AppConfig:
    paths: PathConfig = field(default_factory=PathConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    answers: AnswerConfig = field(default_factory=AnswerConfig)
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path = Path("config/default.yaml")) -> AppConfig:
    data = _read_yaml(path)
    data = _apply_env_overrides(data)
    return AppConfig(
        paths=PathConfig(**data.get("paths", {})),
        llm=LlmConfig(**data.get("llm", {})),
        embedding=EmbeddingConfig(**data.get("embedding", {})),
        index=IndexConfig(**data.get("index", {})),
        retrieval=RetrievalConfig(**data.get("retrieval", {})),
        answers=AnswerConfig(**data.get("answers", {})),
        raw=data,
    )


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config file must contain a YAML object: {path}")
    return loaded


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    result = dict(data)
    _set_nested(result, ("paths", "chroma_dir"), os.getenv("COBOL_RAG_CHROMA_DIR"))
    _set_nested(result, ("paths", "inbox_dir"), os.getenv("COBOL_RAG_INBOX_DIR"))
    _set_nested(result, ("index", "collection"), os.getenv("COBOL_RAG_COLLECTION"))
    _set_nested(result, ("llm", "model"), os.getenv("COBOL_RAG_LLM_MODEL"))
    _set_nested(result, ("llm", "base_url"), os.getenv("COBOL_RAG_LLM_BASE_URL"))
    _set_nested(result, ("embedding", "model"), os.getenv("COBOL_RAG_EMBEDDING_MODEL"))
    _set_nested(result, ("embedding", "base_url"), os.getenv("COBOL_RAG_EMBEDDING_BASE_URL"))

    if top_k := os.getenv("COBOL_RAG_TOP_K"):
        _set_nested(result, ("retrieval", "top_k"), int(top_k))
    if context_window := os.getenv("COBOL_RAG_LLM_CONTEXT_WINDOW"):
        _set_nested(result, ("llm", "context_window"), int(context_window))
    return result


def _set_nested(data: dict[str, Any], keys: tuple[str, str], value: Any | None) -> None:
    if value is None:
        return
    section, name = keys
    data.setdefault(section, {})
    data[section][name] = value
