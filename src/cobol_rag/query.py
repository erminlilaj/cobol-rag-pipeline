from __future__ import annotations

from dataclasses import dataclass

from cobol_rag.config import AppConfig
from cobol_rag.index import open_index
from cobol_rag.retrieve import RetrievalResult, retrieve


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    answer: str
    sources: list[RetrievalResult]


class QueryError(Exception):
    """Raised when answer generation fails after retrieval succeeds."""


def answer_query(
    question: str,
    config: AppConfig,
    top_k: int | None = None,
) -> QueryAnswer:
    sources = retrieve(question, config=config, top_k=top_k)
    if not sources:
        return QueryAnswer(
            question=question,
            answer="I could not find relevant indexed sources for this question.",
            sources=[],
        )

    resources = open_index(config)
    prompt = _build_prompt(question=question, sources=sources)
    try:
        response = resources.runtime.llm.complete(prompt)
    except Exception as error:
        raise QueryError(
            "Answer generation failed after retrieval succeeded. "
            "Check that the configured LLM is available in Ollama and fits in memory."
        ) from error
    return QueryAnswer(
        question=question,
        answer=str(response.text).strip(),
        sources=sources,
    )


def _build_prompt(question: str, sources: list[RetrievalResult]) -> str:
    context_blocks = []
    for index, source in enumerate(sources, start=1):
        source_id = source.metadata.get("source_id", f"source-{index}")
        source_path = source.metadata.get("source_path", "")
        context_blocks.append(
            "\n".join(
                [
                    f"[Source {index}]",
                    f"source_id: {source_id}",
                    f"source_path: {source_path}",
                    "text:",
                    source.text,
                ]
            )
        )

    context = "\n\n".join(context_blocks)
    return f"""You are answering a question using only the retrieved sources below.

Rules:
- Answer only from the retrieved sources.
- If the sources do not contain the answer, say that the indexed sources do not contain enough information.
- Keep the answer concise.
- Mention source ids inline when useful, but do not invent source ids.

Question:
{question}

Retrieved sources:
{context}

Answer:
"""
