from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.index import open_index


@dataclass(frozen=True)
class RetrievalResult:
    score: float | None
    text: str
    metadata: dict[str, Any]


def retrieve(query: str, config: AppConfig, top_k: int | None = None) -> list[RetrievalResult]:
    resources = open_index(config)
    retriever = resources.index.as_retriever(
        similarity_top_k=top_k or config.retrieval.top_k
    )
    nodes = retriever.retrieve(query)
    return [
        RetrievalResult(
            score=node.score,
            text=node.node.get_content(),
            metadata=dict(node.node.metadata),
        )
        for node in nodes
    ]
