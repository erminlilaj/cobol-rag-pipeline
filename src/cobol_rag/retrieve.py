from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
import threading
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.index import open_index


_COLLECTION_CACHE_TTL_SECONDS = 30.0
_COLLECTION_CACHE_MAX_ENTRIES = 32
_collection_cache: "OrderedDict[tuple[str, str, str], tuple[float, dict[str, Any]]]" = OrderedDict()
_collection_cache_lock = threading.RLock()
_lexical_partition_cache: "OrderedDict[tuple[Any, ...], tuple[float, LexicalPartition]]" = OrderedDict()


def clear_retrieval_cache() -> None:
    """Invalidate cached lexical/parent snapshots after collection mutations."""
    with _collection_cache_lock:
        _collection_cache.clear()
        _lexical_partition_cache.clear()


def _collection_snapshot(
    config: AppConfig,
    metadata_filters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Read a program partition once, rather than scanning the whole corpus per query."""
    filters = metadata_filters or {}
    program = str(filters.get("program", "")).upper()
    key = (str(config.paths.chroma_dir), config.index.collection, program)
    now = time.monotonic()
    with _collection_cache_lock:
        cached = _collection_cache.pop(key, None)
        if cached is not None and now - cached[0] <= _COLLECTION_CACHE_TTL_SECONDS:
            _collection_cache[key] = cached
            return cached[1]

    resources = open_index(config)
    kwargs: dict[str, Any] = {"include": ["documents", "metadatas"]}
    if program:
        kwargs["where"] = {"program": program}
    raw = resources.chroma_collection.get(**kwargs)
    with _collection_cache_lock:
        _collection_cache[key] = (now, raw)
        while len(_collection_cache) > _COLLECTION_CACHE_MAX_ENTRIES:
            _collection_cache.popitem(last=False)
    return raw


@dataclass(frozen=True)
class RetrievalResult:
    score: float | None
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class EvidenceGuard:
    status: str
    reasons: tuple[str, ...] = ()
    exact_entity_hits: int = 0
    injection_signals: tuple[str, ...] = ()


@dataclass(frozen=True)
class RetrievalOutcome:
    results: list[RetrievalResult]
    vector_results: list[RetrievalResult]
    lexical_results: list[RetrievalResult]
    filters: dict[str, Any]
    intent: str
    correction_applied: bool
    expanded_count: int
    guard: EvidenceGuard


@dataclass(frozen=True)
class LexicalDocument:
    text: str
    metadata: dict[str, Any]
    terms: tuple[str, ...]
    frequencies: Counter[str]


@dataclass(frozen=True)
class LexicalPartition:
    documents: tuple[LexicalDocument, ...]
    document_frequency: Counter[str]
    average_length: float


def _compiled_lexical_partition(
    raw: dict[str, Any],
    *,
    chunk_types: list[str] | None,
    metadata_filters: dict[str, Any],
) -> LexicalPartition:
    """Compile one program partition once instead of tokenizing it per query."""
    allowed_types = tuple(sorted(set(chunk_types or [])))
    filter_key = tuple(sorted((str(key), str(value)) for key, value in metadata_filters.items()))
    cache_key = (id(raw), allowed_types, filter_key)
    now = time.monotonic()
    with _collection_cache_lock:
        cached = _lexical_partition_cache.pop(cache_key, None)
        if cached is not None and now - cached[0] <= _COLLECTION_CACHE_TTL_SECONDS:
            _lexical_partition_cache[cache_key] = cached
            return cached[1]

    allowed = set(allowed_types)
    documents: list[LexicalDocument] = []
    document_frequency: Counter[str] = Counter()
    for text, metadata in zip(raw.get("documents") or [], raw.get("metadatas") or []):
        if not isinstance(text, str) or not isinstance(metadata, dict):
            continue
        if allowed and metadata.get("chunk_type") not in allowed:
            continue
        if not _metadata_matches(metadata, metadata_filters):
            continue
        terms = tuple(_lexical_terms(text))
        frequencies = Counter(terms)
        document_frequency.update(frequencies.keys())
        documents.append(
            LexicalDocument(
                text=text,
                metadata=metadata,
                terms=terms,
                frequencies=frequencies,
            )
        )
    average_length = max(
        (sum(len(document.terms) for document in documents) / len(documents))
        if documents else 0.0,
        1.0,
    )
    partition = LexicalPartition(
        documents=tuple(documents),
        document_frequency=document_frequency,
        average_length=average_length,
    )
    with _collection_cache_lock:
        _lexical_partition_cache[cache_key] = (now, partition)
        while len(_lexical_partition_cache) > _COLLECTION_CACHE_MAX_ENTRIES * 2:
            _lexical_partition_cache.popitem(last=False)
    return partition


def retrieve(
    query: str,
    config: AppConfig,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    *,
    program: str | None = None,
    entity_key: str | None = None,
    entity_value: str | None = None,
    entity_keys: list[str] | tuple[str, ...] | None = None,
    entity_values: list[str] | tuple[str, ...] | None = None,
    intent: str | None = None,
    relations: list[str] | tuple[str, ...] | None = None,
) -> list[RetrievalResult]:
    return retrieve_with_trace(
        query,
        config,
        top_k=top_k,
        chunk_types=chunk_types,
        program=program,
        entity_key=entity_key,
        entity_value=entity_value,
        entity_keys=entity_keys,
        entity_values=entity_values,
        intent=intent,
        relations=relations,
    ).results


def retrieve_with_trace(
    query: str,
    config: AppConfig,
    top_k: int | None = None,
    chunk_types: list[str] | None = None,
    *,
    program: str | None = None,
    entity_key: str | None = None,
    entity_value: str | None = None,
    entity_keys: list[str] | tuple[str, ...] | None = None,
    entity_values: list[str] | tuple[str, ...] | None = None,
    intent: str | None = None,
    relations: list[str] | tuple[str, ...] | None = None,
) -> RetrievalOutcome:
    effective_k = top_k or config.retrieval.top_k
    entity_pairs = _normalize_entity_pairs(
        entity_key=entity_key,
        entity_value=entity_value,
        entity_keys=entity_keys,
        entity_values=entity_values,
    )
    mode = config.retrieval.mode
    effective_intent = intent or _detect_intent(query)
    retrieval_query = _expanded_query_for_intent(query, effective_intent)
    metadata_filters = dict(config.retrieval.filters)
    if program:
        metadata_filters["program"] = program.upper()
    candidate_k = max(effective_k * 3, config.retrieval.bm25_top_k)
    vector_results: list[RetrievalResult] = []
    lexical_results: list[RetrievalResult] = []

    if mode == "bm25":
        lexical_results = _filter_results(
            _bm25_only(retrieval_query, config, candidate_k), metadata_filters, chunk_types
        )
        combined = lexical_results
    elif mode == "hybrid":
        vector_results = _vector(
            retrieval_query, config, candidate_k, chunk_types, metadata_filters
        )
        lexical_results = _lexical_from_collection(
            retrieval_query, config, candidate_k, chunk_types, metadata_filters
        )
        combined = _rrf_result_lists(vector_results, lexical_results, candidate_k)
    else:
        vector_results = _vector(
            retrieval_query, config, candidate_k, chunk_types, metadata_filters
        )
        combined = vector_results

    ranked = _intent_rerank(
        query, combined, candidate_k, intent_override=effective_intent
    )
    correction_applied = False
    for requested_key, requested_value in entity_pairs:
        if _has_exact_entity(ranked, requested_key, requested_value):
            continue
        correction_applied = True
        correction_filters = {"program": program.upper()} if program else {}
        corrective = _lexical_from_collection(
            requested_value,
            config,
            candidate_k,
            chunk_types,
            correction_filters,
        )
        lexical_results = _deduplicate_results(corrective + lexical_results)
        ranked = _intent_rerank(
            query,
            _deduplicate_results(corrective + ranked),
            candidate_k,
            intent_override=effective_intent,
        )

    primary = _prioritize_entity(
        ranked,
        entity_key,
        entity_value,
        entity_pairs=entity_pairs,
    )[:effective_k]
    expanded = _expand_context(
        primary,
        config,
        program=program,
        entity_key=entity_key,
        entity_value=entity_value,
        entity_pairs=entity_pairs,
        intent=effective_intent,
        relations=tuple(relations or ()),
        allowed_chunk_types=tuple(chunk_types or ()),
    )
    results = _deduplicate_results(primary + expanded)
    guard = _validate_evidence(
        results,
        program=program,
        entity_key=entity_key,
        entity_value=entity_value,
        entity_pairs=entity_pairs,
        correction_applied=correction_applied,
    )
    return RetrievalOutcome(
        results=results,
        vector_results=vector_results,
        lexical_results=lexical_results,
        filters=metadata_filters,
        intent=effective_intent,
        correction_applied=correction_applied,
        expanded_count=len(expanded),
        guard=guard,
    )


# ---------------------------------------------------------------------------
# Vector retrieval
# ---------------------------------------------------------------------------

def _vector(
    query: str,
    config: AppConfig,
    top_k: int,
    chunk_types: list[str] | None,
    metadata_filters: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    resources = open_index(config)
    filters = _make_filters(chunk_types, metadata_filters)
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


def _make_filters(
    chunk_types: list[str] | None,
    metadata_filters: dict[str, Any] | None = None,
):
    if not chunk_types and not metadata_filters:
        return None
    try:
        from llama_index.core.vector_stores.types import (
            FilterCondition,
            FilterOperator,
            MetadataFilter,
            MetadataFilters,
        )
        filters: list[Any] = [
            MetadataFilter(key=key, value=value, operator=FilterOperator.EQ)
            for key, value in (metadata_filters or {}).items()
        ]
        if chunk_types:
            type_filters = MetadataFilters(
                filters=[
                MetadataFilter(key="chunk_type", value=ct, operator=FilterOperator.EQ)
                for ct in chunk_types
                ],
                condition=FilterCondition.OR,
            )
            filters.append(type_filters)
        return MetadataFilters(filters=filters, condition=FilterCondition.AND)
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


def _hybrid_from_collection(
    query: str,
    config: AppConfig,
    top_k: int,
    chunk_types: list[str] | None,
) -> list[RetrievalResult]:
    """Fuse vector results with lexical results from the live Chroma collection.

    Combined JSONL indexes do not necessarily ship a Cobol-REKT BM25 bundle.
    Scanning the modest local collection gives hybrid retrieval instead of
    silently degrading to vectors, and strongly preserves exact COBOL names.
    """
    candidate_k = max(top_k * 2, config.retrieval.bm25_top_k)
    vector_results = _vector(query, config, candidate_k, chunk_types)
    lexical_results = _lexical_from_collection(query, config, candidate_k, chunk_types)
    return _rrf_result_lists(vector_results, lexical_results, candidate_k)


def _lexical_from_collection(
    query: str,
    config: AppConfig,
    top_k: int,
    chunk_types: list[str] | None,
    metadata_filters: dict[str, Any] | None = None,
) -> list[RetrievalResult]:
    raw = _collection_snapshot(config, metadata_filters)
    partition = _compiled_lexical_partition(
        raw,
        chunk_types=chunk_types,
        metadata_filters=metadata_filters or {},
    )
    if not partition.documents:
        return []

    query_terms = list(dict.fromkeys(_lexical_terms(query)))
    if not query_terms:
        return []
    query_identifiers = set(_query_identifiers(query))
    document_frequency = partition.document_frequency
    average_length = partition.average_length
    total = len(partition.documents)
    scored: list[RetrievalResult] = []

    for document in partition.documents:
        text = document.text
        metadata = document.metadata
        frequencies = document.frequencies
        length = max(len(document.terms), 1)
        score = 0.0
        for term in query_terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            frequency_docs = document_frequency.get(term, 0)
            inverse_frequency = math.log((total - frequency_docs + 0.5) / (frequency_docs + 0.5) + 1)
            normalized_frequency = frequency * 2.5 / (
                frequency + 1.5 * (0.25 + 0.75 * length / average_length)
            )
            weight = 4.0 if term in query_identifiers else 1.0
            score += inverse_frequency * normalized_frequency * weight
        variable = str(metadata.get("variable", "")).upper()
        paragraph = str(metadata.get("paragraph", "")).upper()
        if variable in query_identifiers:
            score += 20.0
        if paragraph in query_identifiers:
            score += 12.0
        if score > 0:
            scored.append(RetrievalResult(score=score, text=text, metadata=metadata))
    scored.sort(key=lambda result: float(result.score or 0.0), reverse=True)
    return _deduplicate_results(scored)[:top_k]


def _rrf_result_lists(
    first: list[RetrievalResult],
    second: list[RetrievalResult],
    top_k: int,
    k: int = 60,
) -> list[RetrievalResult]:
    scores: dict[str, float] = {}
    results: dict[str, RetrievalResult] = {}
    for result_list in (first, second):
        for rank, result in enumerate(_deduplicate_results(result_list), start=1):
            key = _result_key(result)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            results.setdefault(key, result)
    ranked_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]
    return [
        RetrievalResult(score=scores[key], text=results[key].text, metadata=results[key].metadata)
        for key in ranked_keys
    ]


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
    intent_override: str | None = None,
) -> list[RetrievalResult]:
    results = _deduplicate_results(results)
    intent = intent_override or _detect_intent(query)
    if intent == "general" or not results:
        return results[:top_k]

    scored = [
        (
            _intent_score(intent, result)
            + _exact_entity_score(query, result)
            + _normalized_base_score(result),
            index,
            result,
        )
        for index, result in enumerate(results)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    ranked = [result for _score, _index, result in scored]
    canonical_types = {
        "copybooks": {
            "architecture.copybooks",
            "architecture.unused_copybooks",
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
            "jcl.file_io",
            "global.db2_table_usage",
            "global.db2_table_usage.summary",
            "datasets_tables_resources",
            "dependencies",
            "cics_operations",
        },
        "dead_code": {
            "dead_code",
            "unused_copybooks",
            "architecture.unused_copybooks",
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
        "business_rules": {
            "business_rule",
            "cobol_rekt.business_rule",
            "integration.business_rules",
        },
        "variable_dataflow": {
            "dataflow.variable",
            "dataflow.variable.rich",
            "integration.variable_context",
            "dataflow.used_variables",
            "business_rule",
            "cobol_rekt.business_rule",
        },
        "control_flow": {
            "controlflow.cfg",
            "paragraph_logic",
            "workflow",
            "integration.control_flow",
            "architecture.cics_operations",
            "cics_operations",
        },
        "cics_operations": {
            "architecture.cics_operations",
            "cics_operations",
            "cobol_rekt.cics_operation",
            "cobol_rekt.cics_resource",
            "controlflow.cfg",
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


# Ordinary English words that also name a COBOL concept in passing. They are a
# topic hint, never a reading of the question: "name me cobol files", "does it
# write to any queue" and "summary of the results" all land on an intent chosen
# by one common noun. This is a property of English, not of any one question, so
# the list does not grow per bug.
_TOPICAL_INTENT_TERMS: tuple[str, ...] = (
    "tables", "table", "files", "file", "queues", "queue",
    "resources", "resource", "dependencies", "dependency",
    "comments", "comment", "what does", "purpose", "overview", "summary",
)

INTENT_BASIS_EXPLICIT = "explicit"
INTENT_BASIS_TOPICAL = "topical"


def detect_intent_with_basis(
    query: str, ignore_identifiers: Sequence[str] = (),
) -> tuple[str, str]:
    """Classify the question, and report whether the classification is load-bearing.

    Detection is keyword-driven, and a keyword match cannot tell "I recognised
    EXEC CICS" from "the question contained the word file". Both used to arrive
    as certainty, and certainty outranks the semantic planner -- so "name me
    cobol files" and "does PDCBVC write to any queue" were answered from JCL
    dataset evidence, over a planner that had correctly asked for an artifact
    inventory and for CICS operations.

    The basis is decided by counterfactual: strip the generic vocabulary and
    classify again. An intent that survives rested on something specific and
    keeps its authority. An intent that disappears rested on a common noun, and
    is reported as topical so the planner can overrule it.
    """
    # A keyword that occurs only inside an identifier the user named says
    # nothing about the question. "What triggers XCTL-LIV4?" is a question about
    # a paragraph, but the letters "xctl" inside its name routed it to outgoing
    # calls and answered with the program's five call sites.
    for identifier in ignore_identifiers:
        if identifier:
            query = re.sub(re.escape(identifier), " ", query, flags=re.IGNORECASE)
    intent = _detect_intent(query)
    if intent == "general":
        return intent, INTENT_BASIS_EXPLICIT
    stripped = query
    for term in _TOPICAL_INTENT_TERMS:
        stripped = re.sub(rf"(?<![A-Za-z0-9_-]){re.escape(term)}(?![A-Za-z0-9_-])",
                          " ", stripped, flags=re.IGNORECASE)
    if _detect_intent(stripped) != intent:
        return intent, INTENT_BASIS_TOPICAL
    return intent, INTENT_BASIS_EXPLICIT


def _detect_intent(query: str) -> str:
    q = query.lower()
    # "Do you have a file called X?" asks whether an analyzed program exists;
    # it is not a JCL dataset/file-I/O question.  Exact program resolution in
    # the typed planner will confirm which program was named.
    if re.search(
        r"\b(?:(?:do\s+)?you\s+have|have\s+you\s+got|is\s+there)\b.{0,45}"
        r"\b(?:file|program|source)\b.{0,30}\b(?:called|named)\b",
        q,
    ):
        return "program_summary"
    if any(
        term in q
        for term in (
            "analyzed evidence files",
            "available artifacts",
            "evidence artifacts",
            "name of the files",
            "files you have",
        )
    ):
        return "artifact_inventory"
    if any(term in q for term in ("unused", "dead code", "inactive", "commented-out", "commented out", "unreachable")):
        return "dead_code"
    if (
        "copybook" in q
        or "copy book" in q
        or "copy member" in q
        or "copy statement" in q
        or "copy statements" in q
    ):
        return "copybooks"
    if any(term in q for term in ("business rule", "business rules", "rules implemented", "direct cobol evidence")):
        return "business_rules"
    if any(term in q for term in ("control flow", "execution flow", "entry point", "path to termination", "execution path")):
        return "control_flow"
    if (
        re.search(r"\bwalk\s+(?:me\s+)?through\b", q)
        and re.search(r"\b(?:start|beginning|entry)\b.{0,80}\b(?:finish|end|termination|exit)\b", q)
    ) or re.search(r"\bfrom\s+(?:start|beginning|entry)\s+to\s+(?:finish|end|termination|exit)\b", q):
        return "control_flow"
    if (
        re.search(r"\b(?:pagination|page navigation|paging)\b", q)
        or re.search(r"\b(?:page|move|go)\s+(?:through|between)\s+(?:the\s+)?results?\b", q)
        or re.search(r"\b(?:move|go|step|navigate)s?\s+(?:through|between)\s+(?:the\s+)?result\s+pages?\b", q)
        or re.search(r"\b(?:next|previous|prior)\s+(?:result\s+)?page\b", q)
    ):
        return "control_flow"
    if (
        any(term in q for term in ("decide whether", "choose between", "start from", "branch to", "path between"))
        and any(term in q for term in ("paragraph", "fase", "phase", "branch", "path", "start"))
    ):
        return "control_flow"
    if "cics" in q and any(term in q for term in ("command", "operation", "execute", "where")):
        return "cics_operations"
    if _query_identifiers(query) and any(
        term in q
        for term in (
            "modify",
            "modified",
            "write",
            "written",
            "set",
            "test",
            "tested",
            "check",
            "checked",
            "read",
            "used",
            "dataflow",
            "data flow",
        )
    ):
        return "variable_dataflow"
    if (
        re.search(r"\b(?:variables?|data items?|fields?)\b", q)
        and re.search(
            r"\b(?:name|list|show|sample|declare[ds]?|which|what|how many|count|"
            r"tell me about|control(?:s| the)? flow|decide(?:s)? (?:the )?flow)\b",
            q,
        )
    ):
        return "variable_inventory"
    if any(term in q for term in (
        "hardcoded", "hard-coded", "static value", "static values", "forced value",
        "literal assignment", "literal assignments",
    )):
        return "static_values"
    if any(term in q for term in ("outside program", "outside programs", "external program", "external programs", "external call", "external calls", "called program", "called programs", "with parameters", "commarea", "link", "xctl")):
        return "external_programs"
    if "comment" in q:
        return "comments"
    if any(term in q for term in ("db2", "sql include", "sqlinclude")):
        return "datasets_tables"
    # A BMS mapset is a screen resource, and the only evidence of it is the CICS
    # SEND/RECEIVE that names it -- not JCL datasets or DB2 tables. It sat in the
    # datasets_tables branch below from before map and mapset were resolvable
    # entities, which sent "which mapset contains PDCBVC1?" to file-I/O evidence.
    if any(term in q for term in ("mapset", "mapsets")):
        return "cics_operations"
    if any(term in q for term in ("dataset", "datasets", "table", "tables", "file", "files", "queue", "queues", "transaction id")):
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
        "business_rules": "business rules conditions actions paragraphs direct COBOL evidence",
        "variable_dataflow": "variable dataflow definitions writes reads tests control flow paragraphs",
        "control_flow": "control flow entry point branches transitions terminal paragraphs RETURN XCTL ABEND STOP RUN",
        "cics_operations": "CICS commands operations paragraphs physical source lines SEND RECEIVE LINK XCTL RETURN ABEND",
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

    if intent == "business_rules":
        score = _chunk_boost(chunk_type, {
            "business_rule": 0.30,
            "cobol_rekt.business_rule": 0.30,
            "integration.business_rules": 0.24,
        })
        if "condition" in text and ("action" in text or "control flows" in text):
            score += 0.08
        return score

    if intent == "variable_dataflow":
        score = _chunk_boost(chunk_type, {
            "dataflow.variable": 0.34,
            "dataflow.variable.rich": 0.30,
            "integration.variable_context": 0.28,
            "dataflow.used_variables": 0.18,
            "business_rule": 0.12,
            "cobol_rekt.business_rule": 0.12,
        })
        if any(term in text for term in ("write_sites", "read_sites", "modified in", "used in")):
            score += 0.08
        if chunk_type == "architecture.unused_copybooks":
            score -= 0.20
        return score

    if intent == "control_flow":
        score = _chunk_boost(chunk_type, {
            "controlflow.cfg": 0.34,
            "paragraph_logic": 0.24,
            "workflow": 0.22,
            "integration.control_flow": 0.20,
            "architecture.cics_operations": 0.18,
            "cics_operations": 0.14,
        })
        if any(term in text for term in ("entry", "transition", "control flow", "return", "abend")):
            score += 0.08
        return score

    if intent == "cics_operations":
        score = _chunk_boost(chunk_type, {
            "architecture.cics_operations": 0.36,
            "cics_operations": 0.24,
            "cobol_rekt.cics_operation": 0.20,
            "cobol_rekt.cics_resource": 0.18,
            "controlflow.cfg": 0.12,
        })
        if "exec cics" in text or "source_file" in text and "line_start" in text:
            score += 0.08
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


def _exact_entity_score(query: str, result: RetrievalResult) -> float:
    identifiers = set(_query_identifiers(query))
    if not identifiers:
        return 0.0
    score = 0.0
    for key in ("variable", "paragraph", "target", "copybook"):
        if str(result.metadata.get(key, "")).upper() in identifiers:
            score += 0.50
    text_upper = result.text.upper()
    if any(identifier in text_upper for identifier in identifiers):
        score += 0.12
    return score


def _query_identifiers(value: str) -> list[str]:
    return list(dict.fromkeys(re.findall(r"\b[A-Z][A-Z0-9]*(?:-[A-Z0-9]+)+\b", value.upper())))


def _lexical_terms(value: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+(?:-[A-Z0-9]+)*", value.upper())


def _result_key(result: RetrievalResult) -> str:
    metadata = result.metadata
    source_id = str(metadata.get("source_id", "")).strip()
    if source_id:
        return f"source:{source_id}"
    chunk_id = str(metadata.get("chunk_id", "")).strip()
    if chunk_id:
        return f"chunk:{chunk_id}"
    return f"text:{hash(result.text)}"


def _deduplicate_results(results: list[RetrievalResult]) -> list[RetrievalResult]:
    deduplicated: list[RetrievalResult] = []
    seen: set[str] = set()
    for result in results:
        key = _result_key(result)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(result)
    return deduplicated


def _metadata_matches(metadata: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(metadata.get(key) == value for key, value in filters.items())


def _filter_results(
    results: list[RetrievalResult],
    metadata_filters: dict[str, Any],
    chunk_types: list[str] | None,
) -> list[RetrievalResult]:
    allowed_types = set(chunk_types or [])
    return [
        result
        for result in results
        if _metadata_matches(result.metadata, metadata_filters)
        and (not allowed_types or result.metadata.get("chunk_type") in allowed_types)
    ]


def _normalize_entity_pairs(
    *,
    entity_key: str | None,
    entity_value: str | None,
    entity_keys: list[str] | tuple[str, ...] | None,
    entity_values: list[str] | tuple[str, ...] | None,
) -> tuple[tuple[str | None, str], ...]:
    values = list(entity_values or ())
    if not values and entity_value:
        values = [entity_value]
    keys: list[str | None] = list(entity_keys or ())
    if not keys and entity_key:
        keys = [entity_key]
    if len(keys) < len(values):
        keys.extend([None] * (len(values) - len(keys)))
    return tuple((keys[index], value) for index, value in enumerate(values) if value)


def _has_exact_entity(
    results: list[RetrievalResult],
    entity_key: str | None,
    entity_value: str | None,
) -> bool:
    return any(_result_matches_entity(result, entity_key, entity_value) for result in results)


def _result_matches_entity(
    result: RetrievalResult,
    entity_key: str | None,
    entity_value: str | None,
) -> bool:
    metadata = result.metadata
    if entity_key and str(metadata.get("entity_key", "")).upper() == entity_key.upper():
        return True
    if not entity_value:
        return False
    target = entity_value.upper()
    if any(
        str(metadata.get(key, "")).upper() == target
        for key in ("variable", "paragraph", "target", "copybook", "db2_table", "sql_include")
    ):
        return True
    return bool(
        re.search(
            rf"(?<![A-Z0-9-]){re.escape(target)}(?![A-Z0-9-])",
            result.text.upper(),
        )
    )


def _prioritize_entity(
    results: list[RetrievalResult],
    entity_key: str | None,
    entity_value: str | None,
    *,
    entity_pairs: tuple[tuple[str | None, str], ...] = (),
) -> list[RetrievalResult]:
    exact: list[RetrievalResult] = []
    supporting: list[RetrievalResult] = []
    requested = entity_pairs or ((entity_key, entity_value),) if entity_value else ()
    for result in results:
        matches = any(
            _result_matches_entity(result, requested_key, requested_value)
            for requested_key, requested_value in requested
        )
        role = "exact_child" if matches else "retrieved_child"
        enriched = _with_context_role(result, role)
        (exact if role == "exact_child" else supporting).append(enriched)
    return exact + supporting


def _expand_context(
    primary: list[RetrievalResult],
    config: AppConfig,
    *,
    program: str | None,
    entity_key: str | None,
    entity_value: str | None,
    entity_pairs: tuple[tuple[str | None, str], ...] = (),
    intent: str,
    relations: tuple[str, ...] = (),
    allowed_chunk_types: tuple[str, ...] = (),
) -> list[RetrievalResult]:
    if not primary or not program:
        return []
    raw = _collection_snapshot(config, {"program": program.upper()})
    documents = raw.get("documents") or []
    metadatas = raw.get("metadatas") or []
    seen = {_result_key(result) for result in primary}
    parent_candidates: list[RetrievalResult] = []
    sibling_candidates: list[RetrievalResult] = []
    primary_parent_ids = {
        str(value)
        for result in primary
        for value in (
            result.metadata.get("parent_id"),
            result.metadata.get("program_parent_id"),
            result.metadata.get("domain_parent_id"),
        )
        if value
    }
    primary_entity_ids = {
        str(result.metadata.get("entity_parent_id", ""))
        for result in primary
        if result.metadata.get("entity_parent_id")
    }

    for text, metadata in zip(documents, metadatas):
        if not isinstance(text, str) or not isinstance(metadata, dict):
            continue
        if metadata.get("program") != program.upper():
            continue
        candidate = RetrievalResult(score=0.0, text=text, metadata=metadata)
        if _result_key(candidate) in seen:
            continue
        node_id = str(metadata.get("node_id", ""))
        if node_id and node_id in primary_parent_ids:
            parent_candidates.append(_with_context_role(candidate, "parent_context"))
            continue
        if allowed_chunk_types and str(metadata.get("chunk_type", "")) not in allowed_chunk_types:
            continue
        same_intent = metadata.get("intent_domain") == intent
        level = str(metadata.get("hierarchy_level", ""))
        if same_intent and level in {"program", "domain"}:
            parent_candidates.append(_with_context_role(candidate, "parent_context"))
            continue
        requested = entity_pairs or ((entity_key, entity_value),) if entity_value else ()
        if requested and any(
            _result_matches_entity(candidate, requested_key, requested_value)
            for requested_key, requested_value in requested
        ):
            sibling_candidates.append(_with_context_role(candidate, "entity_sibling"))
            continue
        parent_id = str(metadata.get("parent_id", ""))
        if same_intent and parent_id and parent_id in primary_parent_ids:
            sibling_candidates.append(_with_context_role(candidate, "domain_sibling"))
            continue
        entity_parent_id = str(metadata.get("entity_parent_id", ""))
        if entity_parent_id and entity_parent_id in primary_entity_ids:
            sibling_candidates.append(_with_context_role(candidate, "entity_sibling"))

    relation_rich = bool(relations)
    # Keep both the immediate domain and the program summary when available.
    # One parent was insufficient for a real entity -> domain -> program walk.
    parent_limit = 3 if relation_rich else 2
    sibling_limit = max(6, len(entity_pairs) * 3) if relation_rich else 2
    # Canonical intent reranking intentionally drops unrelated chunk types, but
    # a program parent is cross-domain context by definition. Preserve those
    # explicit hierarchy matches after ordering the intent-specific parent.
    parents = _intent_rerank(
        "", parent_candidates, len(parent_candidates), intent_override=intent,
    )
    parents = _deduplicate_results(
        parents + [candidate for candidate in parent_candidates if candidate not in parents]
    )[:parent_limit]
    siblings = _intent_rerank("", sibling_candidates, sibling_limit, intent_override=intent)
    return _deduplicate_results(parents + siblings)


def _with_context_role(result: RetrievalResult, role: str) -> RetrievalResult:
    metadata = dict(result.metadata)
    metadata["context_role"] = role
    return RetrievalResult(score=result.score, text=result.text, metadata=metadata)


def _validate_evidence(
    results: list[RetrievalResult],
    *,
    program: str | None,
    entity_key: str | None,
    entity_value: str | None,
    entity_pairs: tuple[tuple[str | None, str], ...] = (),
    correction_applied: bool,
) -> EvidenceGuard:
    reasons: list[str] = []
    if not results:
        reasons.append("no_retrieved_evidence")
    if program and any(result.metadata.get("program") != program.upper() for result in results):
        reasons.append("wrong_program_evidence")
    requested = entity_pairs or ((entity_key, entity_value),) if entity_value else ()
    exact_hits = sum(
        1
        for result in results
        if any(
            _result_matches_entity(result, requested_key, requested_value)
            for requested_key, requested_value in requested
        )
    )
    missing_entities = [
        requested_value
        for requested_key, requested_value in requested
        if not any(
            _result_matches_entity(result, requested_key, requested_value)
            for result in results
        )
    ]
    if missing_entities:
        reasons.append("missing_exact_entity_evidence")
        reasons.extend(f"missing_exact_entity_evidence:{value}" for value in missing_entities)
    injection_signals = tuple(
        sorted(
            {
                signal
                for result in results
                for signal in _prompt_injection_signals(result.text)
            }
        )
    )
    if reasons:
        status = "insufficient"
    elif correction_applied:
        status = "corrected"
    else:
        status = "pass"
    return EvidenceGuard(
        status=status,
        reasons=tuple(reasons),
        exact_entity_hits=exact_hits,
        injection_signals=injection_signals,
    )


def _prompt_injection_signals(text: str) -> list[str]:
    lowered = text.lower()
    patterns = {
        "ignore_previous_instructions": r"ignore\s+(all\s+)?previous\s+instructions",
        "system_prompt_reference": r"system\s+prompt",
        "instruction_override": r"(?:instead|must)\s+(?:answer|respond|follow)",
        "secret_exfiltration": r"(?:reveal|print|exfiltrate).{0,30}(?:secret|token|password|key)",
    }
    return [name for name, pattern in patterns.items() if re.search(pattern, lowered)]


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
