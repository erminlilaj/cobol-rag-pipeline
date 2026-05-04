from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.index import open_index


@dataclass(frozen=True)
class RetrievalResult:
    score: float | None
    text: str
    metadata: dict[str, Any]


def retrieve(
    query: str,
    config: AppConfig,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
) -> list[RetrievalResult]:
    effective_k = top_k or config.retrieval.top_k
    mode = config.retrieval.mode
    intent = _detect_intent(query)
    retrieval_query = _expanded_query_for_intent(query, intent)

    if mode == "bm25":
        results = _bm25_only(retrieval_query, config, max(effective_k * 2, config.retrieval.bm25_top_k))
        return _intent_rerank(query, results, effective_k)

    if mode == "hybrid":
        bm25_path = _find_bm25_path(config)
        if bm25_path is not None:
            results = _hybrid(retrieval_query, config, effective_k, chunk_types, bm25_path)
            return _intent_rerank(query, results, effective_k)

    # vector-only (or hybrid with no bm25 index found)
    results = _vector(retrieval_query, config, max(effective_k * 2, config.retrieval.bm25_top_k), chunk_types)
    return _intent_rerank(query, results, effective_k)


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

def _vector(
    query: str,
    config: AppConfig,
    top_k: int,
    chunk_types: list[str] | None,
) -> list[RetrievalResult]:
    resources = open_index(config)
    filters = _make_filters(chunk_types)
    retriever = resources.index.as_retriever(
        similarity_top_k=top_k,
        filters=filters,
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


def _make_filters(chunk_types: list[str] | None):
    if not chunk_types:
        return None
    try:
        from llama_index.core.vector_stores.types import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
        return MetadataFilters(
            filters=[
                MetadataFilter(key="chunk_type", value=ct, operator=FilterOperator.EQ)
                for ct in chunk_types
            ],
            condition=FilterCondition.OR,
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# BM25-only retrieval
# ---------------------------------------------------------------------------

def _bm25_only(
    query: str,
    config: AppConfig,
    top_k: int,
) -> list[RetrievalResult]:
    bm25_path = _find_bm25_path(config)
    if bm25_path is None:
        return []
    from cobol_rag.bm25 import bm25_retrieve, load_bm25_index
    index = load_bm25_index(bm25_path)
    chunks_dir = bm25_path.parent
    hits = bm25_retrieve(query, index, chunks_dir, top_k)
    return _load_bm25_hits(hits)


# ---------------------------------------------------------------------------
# Hybrid: vector + BM25 fused by reciprocal rank fusion
# ---------------------------------------------------------------------------

def _hybrid(
    query: str,
    config: AppConfig,
    top_k: int,
    chunk_types: list[str] | None,
    bm25_path: Path,
) -> list[RetrievalResult]:
    from cobol_rag.bm25 import bm25_retrieve, load_bm25_index

    # Over-retrieve on the vector side so RRF has good candidates
    vector_k = max(top_k * 2, config.retrieval.bm25_top_k)
    vector_results = _vector(query, config, vector_k, chunk_types)

    bm25_index = load_bm25_index(bm25_path)
    chunks_dir = bm25_path.parent
    bm25_hits = bm25_retrieve(query, bm25_index, chunks_dir, config.retrieval.bm25_top_k)

    return _rrf_combine(vector_results, bm25_hits, max(top_k * 2, config.retrieval.bm25_top_k))


def _rrf_combine(
    vector_results: list[RetrievalResult],
    bm25_hits: list[tuple[str, float, Path]],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
    rrf: dict[str, float] = {}
    by_chunk_id: dict[str, RetrievalResult] = {}
    bm25_files: dict[str, Path] = {}

    for rank, result in enumerate(vector_results, start=1):
        chunk_id = result.metadata.get("chunk_id") or result.metadata.get("source_id", "")
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank)
        by_chunk_id[chunk_id] = result

    for rank, (chunk_id, _score, file_path) in enumerate(bm25_hits, start=1):
        rrf[chunk_id] = rrf.get(chunk_id, 0.0) + 1.0 / (k + rank)
        bm25_files[chunk_id] = file_path

    ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]

    results: list[RetrievalResult] = []
    for chunk_id, rrf_score in ranked:
        if chunk_id in by_chunk_id:
            r = by_chunk_id[chunk_id]
            results.append(RetrievalResult(score=rrf_score, text=r.text, metadata=r.metadata))
        elif chunk_id in bm25_files:
            file_path = bm25_files[chunk_id]
            if file_path.exists():
                try:
                    with file_path.open() as f:
                        doc = json.load(f)
                    results.append(
                        RetrievalResult(
                            score=rrf_score,
                            text=doc.get("text", ""),
                            metadata=doc.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, OSError):
                    pass
    return results


# ---------------------------------------------------------------------------
# Intent-aware reranking
# ---------------------------------------------------------------------------

def _intent_rerank(
    query: str,
    results: list[RetrievalResult],
    top_k: int,
) -> list[RetrievalResult]:
    intent = _detect_intent(query)
    if intent == "general" or not results:
        return results[:top_k]

    scored = [
        (_intent_score(intent, result) + _normalized_base_score(result), index, result)
        for index, result in enumerate(results)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    ranked = [result for _score, _index, result in scored]
    canonical_types = {
        "copybooks": {
            "architecture.copybooks",
            "global.copybook_usage",
            "global.copybook_usage.summary",
            "cobol_analysis_health",
            "program_summary",
            "dependencies",
            "copybook_resolution",
            "copybook_usage",
        },
        "external_programs": {
            "architecture.call_parameters",
            "architecture.calls",
            "architecture.call",
            "global.call_graph.summary",
            "global.call_target",
            "global.program_dependencies",
            "external_program_calls",
            "cics_operations",
            "dependencies",
        },
        "datasets_tables": {
            "architecture.db2_table",
            "architecture.sqlinclude",
            "global.db2_table_usage",
            "global.db2_table_usage.summary",
            "datasets_tables_resources",
            "dependencies",
            "cics_operations",
        },
        "dead_code": {
            "dead_code",
            "unused_copybooks",
            "commented_out_code",
            "program.comments",
            "program.comment",
            "cobol_analysis_health",
        },
        "comments": {
            "program.comments",
            "program.comment",
            "commented_out_code",
            "program_summary",
            "program.summary",
            "paragraph_logic",
        },
    }.get(intent)
    if canonical_types:
        canonical = [
            result
            for result in ranked
            if result.metadata.get("chunk_type") in canonical_types
        ]
        if canonical:
            return canonical[:top_k]

    return ranked[:top_k]


def _detect_intent(query: str) -> str:
    q = query.lower()
    if any(term in q for term in ("unused", "dead code", "inactive", "commented-out", "commented out", "unreachable")):
        return "dead_code"
    if "copybook" in q or "copy book" in q:
        return "copybooks"
    if any(term in q for term in ("hardcoded", "hard-coded", "static value", "static values", "forced value")):
        return "static_values"
    if any(term in q for term in ("outside program", "outside programs", "external program", "external programs", "external call", "external calls", "called program", "called programs", "with parameters", "commarea", "link", "xctl")):
        return "external_programs"
    if "comment" in q:
        return "comments"
    if any(term in q for term in ("dataset", "datasets", "table", "tables", "file", "files", "mapset", "mapsets", "queue", "queues", "transaction id")):
        return "datasets_tables"
    if any(term in q for term in ("resource", "resources", "dependency", "dependencies", "cics")):
        return "dependencies"
    if any(term in q for term in ("program about", "what is the program", "what does", "purpose", "overview", "summary")):
        return "program_summary"
    return "general"


def _expanded_query_for_intent(query: str, intent: str) -> str:
    expansions = {
        "copybooks": "copybooks_used total_copybooks resolved_copybooks stubbed_copybook_count stubbed_copybooks copybook resolution found missing",
        "static_values": "static values forced values hardcoded literals assignments constants",
        "external_programs": "external program calls LINK XCTL COMMAREA LENGTH called programs program transfers",
        "datasets_tables": "datasets tables resources DB2 SQL CICS files queues maps mapsets transaction ids",
        "dead_code": "dead code unused copybooks commented-out inactive unreachable negative evidence",
        "comments": "comments commented-out inactive code source comments",
    }
    extra = expansions.get(intent)
    if not extra:
        return query
    return f"{query}\n{extra}"


def _intent_score(intent: str, result: RetrievalResult) -> float:
    chunk_type = str(result.metadata.get("chunk_type", ""))
    text = result.text.lower()

    if intent == "copybooks":
        score = _chunk_boost(chunk_type, {
            "architecture.copybooks": 0.24,
            "global.copybook_usage.summary": 0.18,
            "global.copybook_usage": 0.16,
            "cobol_analysis_health": 0.16,
            "program_summary": 0.14,
            "dependencies": 0.12,
        })
        if "copybook" in text:
            score += 0.10
        if "copybooks_used" in text or "stubbed_copybooks" in text:
            score += 0.08
        if chunk_type in {"static_values", "paragraph_logic", "workflow", "cics_operations"}:
            score -= 0.06
        return score

    if intent == "static_values":
        score = _chunk_boost(chunk_type, {
            "dataflow.literal_assignments": 0.24,
            "static_values": 0.18,
        })
        if any(term in text for term in ("static values", "hardcoded", "literal", "forced value", "gets")):
            score += 0.05
        return score

    if intent == "external_programs":
        score = _chunk_boost(chunk_type, {
            "architecture.call_parameters": 0.30,
            "architecture.calls": 0.26,
            "architecture.call": 0.22,
            "global.program_dependencies": 0.18,
            "global.call_target": 0.16,
            "global.call_graph.summary": 0.14,
            "external_program_calls": 0.22,
            "cics_operations": 0.10,
            "dependencies": 0.08,
            "paragraph_logic": 0.02,
        })
        if any(term in text for term in ("external program calls", "outgoing call parameters", "commarea", "length", "program transfers")):
            score += 0.06
        if chunk_type in {"static_values", "dataflow.literal_assignments", "datasets_tables_resources"}:
            score -= 0.04
        return score

    if intent == "datasets_tables":
        score = _chunk_boost(chunk_type, {
            "architecture.db2_table": 0.24,
            "architecture.sqlinclude": 0.16,
            "global.db2_table_usage": 0.18,
            "global.db2_table_usage.summary": 0.16,
            "datasets_tables_resources": 0.22,
            "dependencies": 0.12,
            "cics_operations": 0.06,
        })
        if any(term in text for term in ("db2 tables", "datasets", "resources", "mapsets", "transaction ids")):
            score += 0.06
        if chunk_type in {"static_values", "external_program_calls"}:
            score -= 0.04
        return score

    if intent == "dead_code":
        score = _chunk_boost(chunk_type, {
            "dead_code": 0.22,
            "unused_copybooks": 0.18,
            "commented_out_code": 0.16,
            "cobol_analysis_health": 0.04,
        })
        if any(term in text for term in ("dead-code", "unused", "unreachable", "commented-out", "inactive")):
            score += 0.08
        if chunk_type in {"static_values", "dependencies", "cics_operations", "paragraph_logic"}:
            score -= 0.05
        return score

    if intent == "comments":
        score = _chunk_boost(chunk_type, {
            "program.comments": 0.20,
            "program.comment": 0.18,
            "commented_out_code": 0.20,
            "program_summary": 0.06,
            "program.summary": 0.06,
            "paragraph_logic": 0.04,
        })
        if any(term in text for term in ("comment", "commented-out", "inactive")):
            score += 0.08
        if chunk_type in {"static_values", "dependencies", "cics_operations"}:
            score -= 0.04
        return score

    if intent == "dependencies":
        score = _chunk_boost(chunk_type, {
            "global.program_dependencies": 0.18,
            "architecture.calls": 0.16,
            "architecture.copybooks": 0.14,
            "architecture.db2_table": 0.12,
            "architecture.sqlinclude": 0.10,
            "dependencies": 0.16,
            "cics_operations": 0.14,
            "datasets_tables_resources": 0.10,
            "external_program_calls": 0.08,
            "program_summary": 0.04,
        })
        if any(term in text for term in ("cics", "resources", "dependencies", "program transfers")):
            score += 0.05
        if chunk_type == "static_values":
            score -= 0.04
        return score

    if intent == "program_summary":
        score = _chunk_boost(chunk_type, {
            "program.summary": 0.20,
            "program_summary": 0.18,
            "cobol_analysis_health": 0.06,
            "dependencies": 0.04,
        })
        if "program " in text or "complexity" in text:
            score += 0.04
        if chunk_type == "static_values":
            score -= 0.06
        return score

    return 0.0


def _chunk_boost(chunk_type: str, boosts: dict[str, float]) -> float:
    return boosts.get(chunk_type, 0.0)


def _normalized_base_score(result: RetrievalResult) -> float:
    if result.score is None:
        return 0.0
    return min(max(float(result.score), 0.0), 1.0) * 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_bm25_path(config: AppConfig) -> Path | None:
    """Look up bm25_index_path stored in the collection manifest."""
    manifest_path = config.paths.manifest_dir / f"{config.index.collection}.json"
    if not manifest_path.exists():
        return None
    try:
        with manifest_path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    raw = data.get("bm25_index_path")
    if not raw:
        return None
    p = Path(raw)
    return p if p.exists() else None


def _load_bm25_hits(hits: list[tuple[str, float, Path]]) -> list[RetrievalResult]:
    results: list[RetrievalResult] = []
    for _chunk_id, score, file_path in hits:
        if not file_path.exists():
            continue
        try:
            with file_path.open() as f:
                doc = json.load(f)
            results.append(
                RetrievalResult(
                    score=score,
                    text=doc.get("text", ""),
                    metadata=doc.get("metadata", {}),
                )
            )
        except (json.JSONDecodeError, OSError):
            pass
    return results
