from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms import ChatMessage
from llama_index.core.schema import Document, TextNode
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from pydantic import PrivateAttr

from cobol_rag.config import AppConfig

Measure = Callable[[str], AbstractContextManager[None]]


class _TimedEmbedding(OllamaEmbedding):
    _measure: Measure = PrivateAttr()

    def get_text_embedding_batch(self, texts, show_progress=False, **kwargs):
        with self._measure("embed via Ollama"):
            return super().get_text_embedding_batch(texts, show_progress=show_progress, **kwargs)


class _TimedChromaStore(ChromaVectorStore):
    _measure: Measure = PrivateAttr()

    def add(self, nodes, **kwargs):
        with self._measure("write vectors to Chroma"):
            return super().add(nodes, **kwargs)


@dataclass(frozen=True)
class LlamaIndexRuntime:
    llm: Ollama
    embed_model: OllamaEmbedding


@dataclass(frozen=True)
class IndexResources:
    runtime: LlamaIndexRuntime
    chroma_client: ClientAPI
    chroma_collection: Collection
    vector_store: ChromaVectorStore
    index: VectorStoreIndex
    measure: Measure | None = None


def build_llm(
    config: AppConfig,
    *,
    json_mode: bool = False,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> Ollama:
    """Create the configured Ollama LLM without opening the vector store."""
    if config.llm.provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    return Ollama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        context_window=config.llm.context_window,
        request_timeout=config.llm.request_timeout,
        temperature=config.llm.temperature if temperature is None else temperature,
        json_mode=json_mode,
        # Granite 4.2 thinks by default. Router calls often allow only 40-260
        # output tokens, so an unbounded reasoning trace can consume the entire
        # budget and leave an empty content response. Keep RAG calls concise and
        # machine-parseable unless a deployment explicitly opts into thinking.
        thinking=config.llm.thinking,
        additional_kwargs={
            "num_predict": max_output_tokens or config.llm.max_output_tokens
        },
    )


def compose_prose(
    config: AppConfig,
    *,
    system: str,
    user: str,
    max_output_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate natural-language prose through the model's own chat template.

    Ollama's completion endpoint feeds a prompt to the model raw, with no
    instruct template around it, so an instruct model continues the text rather
    than answering it. Asked in Italian whether it speaks Italian, granite
    completes with "Bene, tu?" -- at every temperature, and identically under a
    long instruction preamble, a chat-shaped prompt, and the bare question. The
    same question through the chat endpoint answers correctly and reproducibly.

    Structured extraction is unharmed by completion mode because a JSON schema
    re-anchors the model, which is why evidence routing works while small talk
    does not. Prose therefore goes through chat.
    """
    llm = build_llm(config, max_output_tokens=max_output_tokens, temperature=temperature)
    response = llm.chat([
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ])
    return str(response.message.content or "").strip()


def build_embedder(config: AppConfig) -> OllamaEmbedding:
    """Create the configured embedding model without opening the vector store."""
    if config.embedding.provider != "ollama":
        raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")
    return OllamaEmbedding(
        model_name=config.embedding.model,
        base_url=config.embedding.base_url,
        embed_batch_size=config.index.batch_size,
    )


def configure_llamaindex(config: AppConfig, *, measure: Measure | None = None) -> LlamaIndexRuntime:
    """Configure LlamaIndex global settings from the project config."""
    if config.embedding.provider != "ollama":
        raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")

    llm = build_llm(config)
    embedder_class = _TimedEmbedding if measure is not None else OllamaEmbedding
    embed_model = embedder_class(
        model_name=config.embedding.model,
        base_url=config.embedding.base_url,
        embed_batch_size=config.index.batch_size,
    )
    if isinstance(embed_model, _TimedEmbedding) and measure is not None:
        embed_model._measure = measure

    Settings.llm = llm
    Settings.embed_model = embed_model
    return LlamaIndexRuntime(llm=llm, embed_model=embed_model)


def open_index(config: AppConfig, *, measure: Measure | None = None) -> IndexResources:
    """Open the configured Chroma collection and wrap it with LlamaIndex."""
    runtime = configure_llamaindex(config, measure=measure)
    config.paths.chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.index.collection
    )
    if measure is None:
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    else:
        vector_store = _TimedChromaStore(chroma_collection=chroma_collection)
        vector_store._measure = measure
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=runtime.embed_model,
    )

    return IndexResources(
        runtime=runtime,
        chroma_client=chroma_client,
        chroma_collection=chroma_collection,
        vector_store=vector_store,
        index=index,
        measure=measure,
    )


def collection_count(resources: IndexResources) -> int:
    return resources.chroma_collection.count()


def delete_source(resources: IndexResources, source_id: str) -> None:
    """Delete all Chroma records for a normalized source id if present."""
    measure = getattr(resources, "measure", None)
    with measure("delete old vectors from Chroma") if measure else nullcontext():
        resources.chroma_collection.delete(where={"source_id": source_id})


def upsert_document(
    resources: IndexResources,
    document: Document,
    chunk_mode: str = "pre_chunked",
) -> None:
    """Refresh one normalized document in the vector index."""
    source_id = str(document.metadata["source_id"])
    delete_source(resources, source_id)
    if chunk_mode == "pre_chunked":
        node = TextNode(
            id_=document.id_,
            text=document.text,
            metadata=dict(document.metadata),
            excluded_embed_metadata_keys=list(document.excluded_embed_metadata_keys),
            excluded_llm_metadata_keys=list(document.excluded_llm_metadata_keys),
        )
        resources.index.insert_nodes([node])
        return
    resources.index.insert(document)
