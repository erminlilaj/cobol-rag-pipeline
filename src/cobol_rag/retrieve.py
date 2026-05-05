from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cobol_rag.config import AppConfig
from cobol_rag.index import open_index

# Matches lowercase COBOL-style identifiers: two or more hyphen-separated segments
# e.g. "twcob-varcont-numfunz", "pd1voci-funzione", "ws-status"
_COBOL_IDENT_RE = re.compile(r'[a-z][a-z0-9]*(?:-[a-z0-9]+){1,}')


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
        bm25_paths = _find_bm25_paths(config)
        if bm25_paths:
            results = _hybrid(retrieval_query, config, effective_k, chunk_types, bm25_paths)
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
    bm25_paths: list[Path],
) -> list[RetrievalResult]:
    from cobol_rag.bm25 import bm25_retrieve, load_bm25_index

    # Over-retrieve on the vector side so RRF has good candidates
    vector_k = max(top_k * 2, config.retrieval.bm25_top_k)
    try:
        vector_results = _vector(query, config, vector_k, chunk_types)
    except Exception:
        # Hybrid retrieval should still be useful when the local embedding
        # server is unavailable. BM25 is deterministic and does not need Ollama.
        vector_results = []

    bm25_hits = []
    search_paths = _filter_bm25_paths_by_program_mention(query, bm25_paths) or bm25_paths
    for bm25_path in search_paths:
        bm25_index = load_bm25_index(bm25_path)
        chunks_dir = bm25_path.parent
        bm25_hits.extend(
            bm25_retrieve(query, bm25_index, chunks_dir, config.retrieval.bm25_top_k)
        )
        canonical_types = _canonical_chunk_types(_detect_intent(query))
        if canonical_types:
            bm25_hits.extend(
                bm25_retrieve(
                    query,
                    bm25_index,
                    chunks_dir,
                    config.retrieval.bm25_top_k,
                    chunk_types=canonical_types,
                )
            )
            for chunk_type in sorted(canonical_types):
                bm25_hits.extend(
                    bm25_retrieve(
                        query,
                        bm25_index,
                        chunks_dir,
                        min(3, config.retrieval.bm25_top_k),
                        chunk_types={chunk_type},
                    )
                )

    # Supplemental: for pagination questions add a direct BM25 query for
    # calculation-formula paragraphs (e.g. CALCOLA-NPAG) that contain DIVIDE /
    # MAX-RIGHE.  These never appear in normal results because their text does
    # not share terms with "pf7", "pf8", or "pagination".
    q_lc = query.lower()
    if any(t in q_lc for t in ("pagination", "pf7", "pf8")) and any(
        t in q_lc for t in ("calculated", "maintained", "how is", "how are")
    ):
        for bm25_path in search_paths:
            bm25_index = load_bm25_index(bm25_path)
            chunks_dir = bm25_path.parent
            bm25_hits.extend(
                bm25_retrieve(
                    "CALCOLA-NPAG DIVIDE MAX-RIGHE NPAGT pages calculation",
                    bm25_index,
                    chunks_dir,
                    3,
                    chunk_types={"paragraph_logic"},
                )
            )

    if _detect_intent(query) in {"field_mapping", "pd1voci_parameter_preparation"} and "pd1voci" in q_lc:
        for bm25_path in search_paths:
            bm25_index = load_bm25_index(bm25_path)
            chunks_dir = bm25_path.parent
            bm25_hits.extend(
                bm25_retrieve(
                    "INIZ-PARAM PD1VOCI-COD-VOCE PD1VOCI-CODDIP-MATR PD1VOCI-CODDIP-PAD PD1VOCI-TIPO-VARIAZ TWCOB-FUNZIONE TWCOB-SP-MATR",
                    bm25_index,
                    chunks_dir,
                    4,
                    chunk_types={"paragraph_logic"},
                )
            )

    candidate_limit = max(
        top_k * 4,
        config.retrieval.bm25_top_k,
        len(_canonical_chunk_types(_detect_intent(query)) or set()) * 3,
    )
    return _rrf_combine(vector_results, bm25_hits, candidate_limit)


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
    filtered = _filter_by_program_mention(query, results)
    if filtered:
        results = filtered
    if intent == "general" or not results:
        return results[:top_k]

    scored = [
        (
            _intent_score(intent, result)
            + _exact_identifier_score(query, result)
            + _query_type_score(query, result)
            + _normalized_base_score(result),
            index,
            result,
        )
        for index, result in enumerate(results)
    ]
    scored.sort(key=lambda item: (item[0], -item[1]), reverse=True)

    ranked = [result for _score, _index, result in scored]
    canonical_types = _canonical_chunk_types(intent)
    if canonical_types:
        canonical = [
            result
            for result in ranked
            if _base_chunk_type(result) in canonical_types
        ]
        if canonical:
            return canonical[:top_k]

    return ranked[:top_k]


def _canonical_chunk_types(intent: str) -> set[str] | None:
    canonical = {
        "copybooks": {"copybook_mentions", "copybook_fields", "cobol_analysis_health", "analysis_health", "program_summary", "dependencies"},
        "copybook_roles": {"copybook_mentions", "copybook_fields", "cobol_analysis_health", "analysis_health", "program_summary", "dependencies"},
        "external_programs": {"external_program_calls", "call_contract", "cics_operations", "cics.program_transfer", "cics.operation", "dependencies"},
        "datasets_tables": {"datasets_tables_resources", "cics.resource", "dependencies", "cics_operations"},
        "dead_code": {"dead_code", "unused_copybooks", "commented_out_code", "cobol_analysis_health"},
        "comments": {"commented_out_code", "program_summary", "paragraph_logic"},
        "sections": {"program_summary", "paragraph_logic", "workflow"},
        "static_values": {"static_values"},
        "dependencies": {
            "dependencies", "cics_operations", "cics.resource", "cics.operation",
            "cics.program_transfer", "datasets_tables_resources", "external_program_calls",
        },
        "data_flow": {"dataflow.variable", "variable_group", "call_contract", "paragraph_logic"},
        "variable_usage": {"dataflow.variable", "variable_group", "paragraph_logic", "call_contract"},
        "field_mapping": {"paragraph_logic", "dataflow.variable", "call_contract", "variable_group"},
        "pd1voci_parameter_preparation": {"paragraph_logic", "dataflow.variable", "call_contract", "variable_group", "static_values"},
        "pagination": {"screen.pagination", "paragraph_logic", "dataflow.variable", "controlflow.cfg"},
        "row_build": {"screen.row_build", "paragraph_logic", "dataflow.variable", "variable_group"},
        "row_selection": {"screen.selection", "screen.row_build", "paragraph_logic", "dataflow.variable", "controlflow.cfg"},
        "condition_path": {"paragraph_logic", "controlflow.cfg", "dataflow.variable", "screen.pagination", "error_path"},
        "control_flow": {
            "paragraph_logic",
            "workflow",
            "controlflow.cfg",
            "call_contract",
            "variable_group",
            "screen.pagination",
            "screen.selection",
            "screen.row_build",
            "screen.key_dispatch",
        },
        "error_path": {"error_path", "paragraph_logic", "workflow", "cics.error_handler", "cics_operations", "controlflow.cfg", "dataflow.variable"},
        "error_paths": {"error_path", "paragraph_logic", "workflow", "cics.error_handler", "cics_operations", "controlflow.cfg", "dataflow.variable"},
    }.get(intent)
    return _expanded_chunk_type_aliases(canonical) if canonical else None


def _expanded_chunk_type_aliases(chunk_types: set[str]) -> set[str]:
    """Accept native cobol-rekt names plus namespaced combined-RAG names."""
    expanded: set[str] = set()
    for chunk_type in chunk_types:
        expanded.add(chunk_type)
        expanded.add(f"cobol_rekt.{chunk_type}")
        expanded.add(chunk_type.replace(".", "_"))
        expanded.add(f"cobol_rekt.{chunk_type.replace('.', '_')}")
    return expanded


def _base_chunk_type(result: RetrievalResult) -> str:
    raw = (
        result.metadata.get("chunk_type")
        or result.metadata.get("source_chunk_type")
        or result.metadata.get("original_chunk_type")
        or result.metadata.get("type")
        or ""
    )
    chunk_type = str(raw)
    if "." in chunk_type and chunk_type.startswith(("cobol_rekt.", "mapa.")):
        chunk_type = chunk_type.split(".", 1)[1]
    return chunk_type.replace("_", ".") if chunk_type.startswith("screen_") else chunk_type


def _filter_by_program_mention(
    query: str,
    results: list[RetrievalResult],
) -> list[RetrievalResult]:
    """Keep matching programs when the user explicitly names an indexed program."""
    query_upper = query.upper()
    wanted: set[str] = set()
    for result in results:
        program = str(result.metadata.get("program", "")).upper()
        if not program:
            continue
        stem = re.sub(r"\.(CBL|COB)$", "", program)
        if program in query_upper or stem in query_upper:
            wanted.add(program)
    if not wanted:
        return []
    filtered = [
        result
        for result in results
        if str(result.metadata.get("program", "")).upper() in wanted
    ]
    return filtered


def _detect_intent(query: str) -> str:
    q = query.lower()
    if any(term in q for term in ("unused", "dead code", "inactive", "commented-out", "commented out", "unreachable")):
        return "dead_code"
    if _is_field_mapping_question(q):
        return "field_mapping"
    if _is_pd1voci_parameter_question(q):
        return "pd1voci_parameter_preparation"
    if any(term in q for term in ("pagination", "page count", "number of pages", "total number of pages", "pf7", "pf8")):
        return "pagination"
    if any(term in q for term in ("prep-riga", "row build", "build each", "build the display", "display row", "displayed row")):
        return "row_build"
    if any(term in q for term in ("selected row", "row selection", "invalid selection", "sceltai", "progressivo")):
        return "row_selection"
    if ("copybook" in q or "copy book" in q) and any(term in q for term in ("role", "play", "used for", "purpose", "what does each")):
        return "copybook_roles"
    if "copybook" in q or "copy book" in q:
        return "copybooks"
    if any(term in q for term in ("hardcoded", "hard-coded", "static value", "static values", "forced value", "who reads", "set or declared")):
        return "static_values"
    if _is_condition_path_question(q):
        return "condition_path"
    # "influence" + COBOL-style hyphenated identifiers → need dataflow.variable sites
    if "influence" in q and _COBOL_IDENT_RE.search(q):
        return "data_flow"
    if any(term in q for term in (
        "error message", "abnormal termination", "abend", "error path",
        "invalid key", "invalid function key", "invalid selection",
        "all paths", "lead to error", "failed service", "sql error",
    )):
        return "error_paths"
    if any(term in q for term in ("dataflow", "data flow")):
        return "data_flow"
    if "variable" in q or re.search(r"\bwhat does\s+[a-z0-9-]+\s+do\b", q):
        return "variable_usage"
    if any(term in q for term in (
        "how does", "how is", "sequence", "decide", "decision",
        "when does", "full sequence", "operations performed", "steps",
        "what happens", "branch", "condition", "phase", "fase", "browse",
        "validate", "build each", "prepared", "calculated", "maintained",
        "pagination", "function key", "semaphore",
        "pf1", "pf2", "pf3", "pf4", "pf7", "pf8", "pf9",
        "select", "selected row", "before the map", "paragraph",
    )):
        return "control_flow"
    if any(term in q for term in ("outside program", "outside programs", "external program", "external programs", "external call", "external calls", "called program", "called programs", "with parameters", "commarea", "link", "xctl")):
        return "external_programs"
    if "comment" in q:
        return "comments"
    if any(term in q for term in ("dataset", "datasets", "table", "tables", "file", "files", "mapset", "mapsets", "queue", "queues", "transaction id")):
        return "datasets_tables"
    if any(term in q for term in ("resource", "resources", "dependency", "dependencies", "cics")):
        return "dependencies"
    if any(term in q for term in ("section", "sections")):
        return "sections"
    if any(term in q for term in ("program about", "what is the program", "what does", "purpose", "overview", "summary")):
        return "program_summary"
    return "general"


def _expanded_query_for_intent(query: str, intent: str) -> str:
    expansions = {
        "copybooks": "copybooks_used total_copybooks resolved_copybooks stubbed_copybook_count stubbed_copybooks copybook fields found missing",
        "static_values": "static values forced values hardcoded literals assignments constants MOVE VALUE variable name assignment category",
        "external_programs": "external program calls LINK XCTL COMMAREA LENGTH called programs program transfers cics.program_transfer",
        "datasets_tables": "datasets tables resources DB2 SQL CICS files queues maps mapsets transaction ids cics.resource TRANSID RETURN",
        "dead_code": "dead code unused copybooks commented-out inactive unreachable negative evidence",
        "comments": "comments commented-out inactive code source comments",
        "sections": "program summary procedure sections paragraphs workflow paragraph_logic",
        "control_flow": "workflow paragraph_logic sequence procedure steps phase decision branch PERFORM GOTO COBOL paragraph call_contract screen pagination selection key dispatch",
        "pagination": "screen pagination CALCOLA-NPAG MAX-RIGHE NPAGT RESTO WCTPAG TWCOB-VARCONT-NPAGINA PF7 PF8 ENTER BROWSE-FASE2-PF7 BROWSE-FASE2-PF8",
        "row_build": "screen row build PREP-RIGA VOCE FUNZ WDESCVO WPROGREC PDRUTI01-F05-VALORE IMPORTO-RATA DATA-IMPIANTO DATA-CESSAZIONE",
        "row_selection": "screen selection selected row SCELTAI WPROGR WPROGREC WCTRIG BROWSE-FASE2-SEL BROWSE-FASE2-NOSEL BROWSE-FASE2-NOTFND progressivo",
        "pd1voci_parameter_preparation": "INIZ-PARAM PD1VOCI parameter preparation TWCOB-VARCONT-NUMFUNZ TWCOB-FUNZIONE PD1VOCI-FUNZIONE PD1VOCI-TIPO-ESTRA PD1VOCI-TIPO-VOCE call_contract",
        "condition_path": "condition path what happens when READ-TAB-SEMAF PXCSEMAF-STATUS XCTL-LIV4 TWCOB-FUNZIONE TWCOB-ID-SISTEMA branch",
        "field_mapping": "field mapping copied moved MOVE source target PDRTWA2 PD1VOCI INIZ-PARAM PD1VOCI-COD-VOCE PD1VOCI-CODDIP-MATR PD1VOCI-CODDIP-PAD PD1VOCI-TIPO-VARIAZ TWCOB-FUNZIONE TWCOB-SP-MATR dataflow variable call_contract",
        "variable_usage": "variable dataflow usage reads writes assigned modified control-flow decisions field paragraph",
        "copybook_roles": "copybooks roles purpose used for copybook_mentions copybook_fields system copybooks CICS BMS COMMAREA SQL TWA",
        "error_path": "error_path error abend termination invalid paragraph_logic workflow TASTOER NOTFND NOSEL ABEND-CODE error message cics error_handler",
        "error_paths": "error_path error abend termination invalid paragraph_logic workflow TASTOER NOTFND NOSEL ABEND-CODE error message cics error_handler PXCSEMAF-STATUS ERRORE-SQL",
        "dependencies": "dependencies resources CICS program transfers LINK XCTL transaction TRANSID datasets tables files queues cics.resource cics.operation",
        "data_flow": "variable field dataflow variable_group usage assigned MOVE COMPUTE PERFORM working-storage linkage semaphore area value COMMAREA parameter call_contract influence condition branch",
    }
    extra = expansions.get(intent)
    if not extra:
        return query
    q_lc = query.lower()
    # For pagination questions, add formula terms so BM25 surfaces the page-count
    # calculation paragraph (CALCOLA-NPAG) which uses DIVIDE MAX-RIGHE INTO NPAGT.
    # That paragraph has low vector similarity to "pf7/pf8/user interactions" but
    # is the authoritative source for how page count is computed.
    if intent == "pagination" or (intent == "control_flow" and any(t in q_lc for t in ("pagination", "pf7", "pf8"))):
        extra += " CALCOLA-NPAG MAX-RIGHE NPAGT DIVIDE pages calculation"
    return f"{query}\n{extra}"


def _intent_score(intent: str, result: RetrievalResult) -> float:
    chunk_type = _base_chunk_type(result)
    text = result.text.lower()

    if intent == "copybooks":
        score = _chunk_boost(chunk_type, {
            "copybook_mentions": 0.20,
            "copybook_fields": 0.16,
            "cobol_analysis_health": 0.14,
            "analysis_health": 0.14,
            "program_summary": 0.12,
            "dependencies": 0.10,
        })
        if "copybook" in text:
            score += 0.10
        if "copybooks_used" in text or "stubbed_copybooks" in text:
            score += 0.08
        if chunk_type in {"static_values", "paragraph_logic", "workflow", "cics_operations"}:
            score -= 0.06
        return score

    if intent == "copybook_roles":
        score = _chunk_boost(chunk_type, {
            "copybook_mentions": 0.22,
            "program_summary": 0.18,
            "cobol_analysis_health": 0.16,
            "analysis_health": 0.16,
            "copybook_fields": 0.08,
            "dependencies": 0.08,
        })
        if any(term in text for term in ("copybooks_used", "known_system_copybooks", "stubbed", "purpose", "copybook")):
            score += 0.08
        if chunk_type in {"paragraph_logic", "static_values", "dataflow.variable", "controlflow.cfg"}:
            score -= 0.08
        return score

    if intent == "static_values":
        score = _chunk_boost(chunk_type, {"static_values": 0.18})
        if "static values" in text or "hardcoded" in text:
            score += 0.05
        return score

    if intent == "external_programs":
        score = _chunk_boost(chunk_type, {
            "external_program_calls": 0.22,
            "call_contract": 0.18,
            "cics.program_transfer": 0.18,
            "cics_operations": 0.10,
            "cics.operation": 0.08,
            "dependencies": 0.08,
            "dataflow.variable": 0.06,
            "paragraph_logic": 0.04,
        })
        if any(term in text for term in ("external program calls", "commarea", "length", "program transfers", "xctl", "link")):
            score += 0.06
        if chunk_type in {"static_values", "datasets_tables_resources"}:
            score -= 0.04
        return score

    if intent == "datasets_tables":
        score = _chunk_boost(chunk_type, {
            "datasets_tables_resources": 0.22,
            "cics.resource": 0.16,
            "dependencies": 0.12,
            "cics_operations": 0.06,
        })
        if any(term in text for term in ("db2 tables", "datasets", "resources", "mapsets", "transaction ids", "transid")):
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
            "commented_out_code": 0.20,
            "program_summary": 0.06,
            "paragraph_logic": 0.04,
        })
        if any(term in text for term in ("comment", "commented-out", "inactive")):
            score += 0.08
        if chunk_type in {"static_values", "dependencies", "cics_operations"}:
            score -= 0.04
        return score

    if intent == "dependencies":
        score = _chunk_boost(chunk_type, {
            "dependencies": 0.16,
            "cics_operations": 0.14,
            "cics.resource": 0.14,
            "cics.operation": 0.12,
            "cics.program_transfer": 0.12,
            "datasets_tables_resources": 0.10,
            "external_program_calls": 0.08,
            "program_summary": 0.04,
        })
        if any(term in text for term in ("cics", "resources", "dependencies", "program transfers", "transid")):
            score += 0.05
        if chunk_type == "static_values":
            score -= 0.04
        return score

    if intent == "program_summary":
        score = _chunk_boost(chunk_type, {
            "program_summary": 0.18,
            "cobol_analysis_health": 0.06,
            "dependencies": 0.04,
        })
        if "program " in text or "complexity" in text:
            score += 0.04
        if chunk_type == "static_values":
            score -= 0.06
        return score

    if intent == "pagination":
        score = _chunk_boost(chunk_type, {
            "screen.pagination": 0.34,
            "paragraph_logic": 0.22,
            "dataflow.variable": 0.10,
            "controlflow.cfg": 0.08,
        })
        if any(term in text for term in ("calcola-npag", "max-righe", "npagt", "resto", "wctpag", "pf7", "pf8")):
            score += 0.12
        if chunk_type in {"program_summary", "copybook_fields", "static_values", "dead_code"}:
            score -= 0.12
        return score

    if intent == "row_build":
        score = _chunk_boost(chunk_type, {
            "screen.row_build": 0.30,
            "paragraph_logic": 0.24,
            "dataflow.variable": 0.12,
            "variable_group": 0.06,
        })
        if any(term in text for term in ("prep-riga", "voce", "funz", "wdescvo", "wprogrec", "importo-rata")):
            score += 0.12
        if chunk_type in {"program_summary", "copybook_fields", "screen.selection", "static_values"}:
            score -= 0.12
        return score

    if intent == "row_selection":
        score = _chunk_boost(chunk_type, {
            "screen.selection": 0.30,
            "screen.row_build": 0.14,
            "paragraph_logic": 0.20,
            "dataflow.variable": 0.10,
            "controlflow.cfg": 0.06,
        })
        if any(term in text for term in ("sceltai", "wprogr", "wprogrec", "wctrig", "browse-fase2-sel")):
            score += 0.12
        if chunk_type in {"program_summary", "copybook_fields", "static_values"}:
            score -= 0.12
        return score

    if intent == "pd1voci_parameter_preparation":
        score = _chunk_boost(chunk_type, {
            "paragraph_logic": 0.30,
            "dataflow.variable": 0.18,
            "call_contract": 0.16,
            "variable_group": 0.08,
            "static_values": 0.04,
        })
        if "iniz-param" in text:
            score += 0.18
        if any(term in text for term in ("pd1voci-funzione", "pd1voci-tipo-estra", "pd1voci-tipo-voce", "twcob-varcont-numfunz")):
            score += 0.12
        if chunk_type in {"external_program_calls", "program_summary", "copybook_fields", "screen.row_build"}:
            score -= 0.14
        return score

    if intent == "field_mapping":
        score = _chunk_boost(chunk_type, {
            "paragraph_logic": 0.28,
            "dataflow.variable": 0.22,
            "call_contract": 0.14,
            "variable_group": 0.08,
        })
        if "iniz-param" in text:
            score += 0.18
        if any(term in text for term in (" move ", "move ", "pd1voci-", "twcob-", "iniz-param", "prep-riga")):
            score += 0.10
        if chunk_type in {"copybook_fields", "program_summary", "copybook_mentions"}:
            score -= 0.18
        return score

    if intent == "variable_usage":
        score = _chunk_boost(chunk_type, {
            "dataflow.variable": 0.32,
            "variable_group": 0.18,
            "paragraph_logic": 0.12,
            "call_contract": 0.08,
        })
        if any(term in text for term in ("variable dataflow", "reads:", "writes:", "used in control-flow", "move", "assigned")):
            score += 0.10
        if chunk_type in {"program_summary", "copybook_fields", "dependencies"}:
            score -= 0.12
        return score

    if intent == "condition_path":
        score = _chunk_boost(chunk_type, {
            "paragraph_logic": 0.26,
            "controlflow.cfg": 0.22,
            "dataflow.variable": 0.14,
            "error_path": 0.12,
            "screen.pagination": 0.04,
        })
        if any(term in text for term in ("twcob-funzione", "twcob-id-sistema", "read-tab-semaf", "pxcsemaf-status", "xctl-liv4")):
            score += 0.14
        if chunk_type in {"program_summary", "copybook_fields", "static_values"}:
            score -= 0.12
        return score

    if intent == "control_flow":
        score = _chunk_boost(chunk_type, {
            "paragraph_logic": 0.26,
            "workflow": 0.22,
            "screen.pagination": 0.20,
            "screen.selection": 0.20,
            "screen.row_build": 0.20,
            "screen.key_dispatch": 0.20,
            "call_contract": 0.14,
            "variable_group": 0.10,
            "dataflow.variable": 0.08,
            "controlflow.cfg": 0.04,
            "cobol_analysis_health": 0.02,
        })
        if any(term in text for term in ("workflow", "sequence", "paragraph", "perform", "goto", "branch", "phase", "fase")):
            score += 0.06
        if chunk_type in {"program_summary", "static_values", "datasets_tables_resources", "dead_code", "unused_copybooks"}:
            score -= 0.06
        return score

    if intent in {"error_path", "error_paths"}:
        score = _chunk_boost(chunk_type, {
            "error_path": 0.30,
            "paragraph_logic": 0.22,
            "workflow": 0.16,
            "cics.error_handler": 0.14,
            "cics_operations": 0.08,
            "variable_group": 0.06,
            "controlflow.cfg": 0.04,
            "dataflow.variable": 0.04,
        })
        if any(term in text for term in ("error", "abend", "invalid", "notfnd", "nosel", "tastoer", "termination", "abnormal")):
            score += 0.08
        if chunk_type in {"program_summary", "static_values", "datasets_tables_resources", "dead_code", "copybook_mentions"}:
            score -= 0.06
        return score

    if intent == "data_flow":
        score = _chunk_boost(chunk_type, {
            "dataflow.variable": 0.28,
            "variable_group": 0.22,
            "paragraph_logic": 0.14,
            "call_contract": 0.12,
            "static_values": 0.06,
        })
        if any(term in text for term in ("variable", "field", "assigned", "move", "compute", "usage", "semaphore", "working-storage", "linkage")):
            score += 0.06
        if chunk_type in {"program_summary", "datasets_tables_resources", "dead_code", "copybook_mentions", "controlflow.cfg"}:
            score -= 0.06
        return score

    return 0.0


def _chunk_boost(chunk_type: str, boosts: dict[str, float]) -> float:
    return boosts.get(chunk_type, 0.0)


def _is_parameter_preparation_question(q: str) -> bool:
    return any(
        term in q
        for term in (
            "parameter prepared",
            "parameters prepared",
            "parameter preparation",
            "parameters for",
            "how are the parameters",
            "how are parameters",
            "prepared",
            "prepare",
            "set up",
            "setup",
            "before the call",
            "before the link",
            "set before link",
            "passed fields",
            "fields copied",
            "influence",
            "affect",
        )
    )


def _is_pd1voci_parameter_question(q: str) -> bool:
    return "pd1voci" in q and _is_parameter_preparation_question(q)


def _is_field_mapping_question(q: str) -> bool:
    if not any(term in q for term in ("field", "fields", "copied", "moved", "move ", "mapping", "map into")):
        return False
    return any(term in q for term in ("pd1voci", "pdrtwa2", "twcob-", "copybook", "commarea", "before the link"))


def _is_condition_path_question(q: str) -> bool:
    if not any(term in q for term in ("what happens", "when", "under what condition", "condition", "if ")):
        return False
    return any(term in q for term in ("twcob-funzione", "twcob-id-sistema", "pxcsemaf-status", "read-tab-semaf", "xctl-liv4"))


def _exact_identifier_score(query: str, result: RetrievalResult) -> float:
    identifiers = _query_identifiers(query)
    if not identifiers:
        return 0.0

    haystack_parts = [result.text]
    for key in (
        "chunk_id", "source_id", "source_path", "program", "paragraph", "section",
        "variable", "target", "title", "chunk_type", "source_chunk_type", "original_chunk_type",
        "type", "interaction_kind", "comment_english",
    ):
        value = result.metadata.get(key)
        if value:
            haystack_parts.append(str(value))
    haystack = "\n".join(haystack_parts).upper()

    score = 0.0
    for identifier in identifiers:
        if identifier in haystack:
            score += 0.08
    return min(score, 0.48)


def _query_type_score(query: str, result: RetrievalResult) -> float:
    q = query.lower()
    chunk_type = _base_chunk_type(result)
    score = 0.0
    if chunk_type == "screen.pagination" and any(term in q for term in ("pagination", "pf7", "pf8", "enter")):
        score += 0.28
    if chunk_type == "screen.row_build" and any(term in q for term in ("prep-riga", "displayed row", "build each", "voice code", "progressivo")):
        score += 0.28
    if chunk_type == "screen.selection" and any(term in q for term in ("select", "selected row", "progressivo", "selection")):
        score += 0.28
    if chunk_type == "screen.key_dispatch" and any(term in q for term in ("function key", "pf1", "pf2", "pf3", "pf4", "pf9")):
        score += 0.28
    if chunk_type == "error_path" and any(term in q for term in ("error", "abend", "invalid", "abnormal", "failed service", "sql")):
        score += 0.16
    # Boost paragraph_logic chunks that contain the actual calculation formula when
    # the question is about pagination — these hold the DIVIDE MAX-RIGHE logic that
    # screen.pagination chunks reference but do not inline.
    if chunk_type == "paragraph_logic" and any(term in q for term in ("pagination", "calculated", "maintained", "pf7", "pf8")):
        text_lc = result.text.lower()
        if any(term in text_lc for term in ("max-righe", "divide", "giving npagt", "calcola-npag")):
            score += 0.25
    if chunk_type == "program_summary" and _is_detailed_procedural_question(q):
        score -= 0.25
    return score


def _is_detailed_procedural_question(q: str) -> bool:
    return any(
        term in q
        for term in (
            "how does", "how are", "what happens", "when", "which fields",
            "path", "paths", "row", "pagination", "page", "prepared",
            "before the", "under what", "condition", "pf7", "pf8",
        )
    )


def _query_identifiers(query: str) -> set[str]:
    raw = re.findall(r"\b[A-Za-z][A-Za-z0-9-]{2,}(?:\.[A-Za-z0-9]+)?\b", query.upper())
    stop = {
        "AND", "ARE", "BEFORE", "BETWEEN", "BROWSE", "CALL", "CICS", "COBOL",
        "COMPARE", "CONTROL", "DOES", "EACH", "EXEC", "EXPLAIN", "EXTERNAL",
        "FLOW", "FROM", "FUNCTION", "HAPPENS", "HOW", "IDENTIFY", "INCLUDE",
        "INCLUDING", "INVALID", "MAP", "PROGRAM", "READ", "ROUTINES", "ROW",
        "SELECTED", "STATUS", "TERMINAL", "THE", "UNDER", "UNEXPECTED",
        "VALUE", "WHAT", "WHEN", "WHETHER", "WHICH", "WITH",
    }
    return {token for token in raw if token not in stop and len(token) >= 3}


def _normalized_base_score(result: RetrievalResult) -> float:
    if result.score is None:
        return 0.0
    return min(max(float(result.score), 0.0), 1.0) * 0.1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_bm25_path(config: AppConfig) -> Path | None:
    paths = _find_bm25_paths(config)
    return paths[0] if paths else None


def _find_bm25_paths(config: AppConfig) -> list[Path]:
    """Look up bm25_index_path stored in the collection manifest."""
    manifest_path = config.paths.manifest_dir / f"{config.index.collection}.json"
    if not manifest_path.exists():
        return []
    try:
        with manifest_path.open() as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    raw_paths = data.get("bm25_index_paths") or []
    if data.get("bm25_index_path"):
        raw_paths.append(data["bm25_index_path"])
    paths = []
    for raw in raw_paths:
        p = Path(raw)
        if p.exists():
            paths.append(p)
    return list(dict.fromkeys(paths))


def _filter_bm25_paths_by_program_mention(query: str, bm25_paths: list[Path]) -> list[Path]:
    """Search the named program's bundle first when the query names one."""
    query_upper = query.upper()
    matched: list[Path] = []
    for path in bm25_paths:
        path_upper = str(path).upper()
        bundle_name = path.parent.parent.name.upper()
        candidates = {bundle_name, bundle_name.replace(".KNOWLEDGE-BASE_RAG", "")}
        candidates.update(part.upper() for part in path.parts if part.upper().endswith((".CBL", ".COB")))
        for candidate in candidates:
            stem = re.sub(r"\.(CBL|COB)(\.KNOWLEDGE-BASE_RAG)?$", "", candidate)
            if candidate and (candidate in query_upper or stem in query_upper or candidate in path_upper and stem in query_upper):
                matched.append(path)
                break
    return list(dict.fromkeys(matched))


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
