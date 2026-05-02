from __future__ import annotations

from dataclasses import dataclass

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.models.Collection import Collection
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.schema import Document
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore

from cobol_rag.config import AppConfig


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


def configure_llamaindex(config: AppConfig) -> LlamaIndexRuntime:
    """Configure LlamaIndex global settings from the project config."""
    if config.llm.provider != "ollama":
        raise ValueError(f"Unsupported LLM provider: {config.llm.provider}")
    if config.embedding.provider != "ollama":
        raise ValueError(f"Unsupported embedding provider: {config.embedding.provider}")

    llm = Ollama(
        model=config.llm.model,
        base_url=config.llm.base_url,
        request_timeout=config.llm.request_timeout,
        temperature=config.llm.temperature,
    )
    embed_model = OllamaEmbedding(
        model_name=config.embedding.model,
        base_url=config.embedding.base_url,
        embed_batch_size=config.index.batch_size,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    return LlamaIndexRuntime(llm=llm, embed_model=embed_model)


def open_index(config: AppConfig) -> IndexResources:
    """Open the configured Chroma collection and wrap it with LlamaIndex."""
    runtime = configure_llamaindex(config)
    config.paths.chroma_dir.mkdir(parents=True, exist_ok=True)

    chroma_client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    chroma_collection = chroma_client.get_or_create_collection(
        name=config.index.collection
    )
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
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
    )


def collection_count(resources: IndexResources) -> int:
    return resources.chroma_collection.count()


def delete_source(resources: IndexResources, source_id: str) -> None:
    """Delete all Chroma records for a normalized source id if present."""
    resources.chroma_collection.delete(where={"source_id": source_id})


def upsert_document(resources: IndexResources, document: Document) -> None:
    """Refresh one normalized document in the vector index."""
    source_id = str(document.metadata["source_id"])
    delete_source(resources, source_id)
    resources.index.insert(document)
