from __future__ import annotations

import json
import math
from pathlib import Path


def load_bm25_index(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def bm25_retrieve(
    query: str,
    index: dict,
    chunks_dir: Path,
    top_k: int,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[tuple[str, float, Path]]:
    """Score each indexed chunk against query terms using BM25.

    Returns list of (chunk_id, score, file_path) sorted by score descending.
    The bm25_index.json stores terms uppercased, so we uppercase the query too.
    """
    query_terms = {t for t in query.upper().split() if len(t) > 1}
    if not query_terms:
        return []

    entries: list[dict] = index.get("entries", [])
    N = max(index.get("total_chunks", len(entries)), 1)
    avgdl = max(index.get("avg_doc_length", 1), 1)

    # Document frequency: how many entries contain each term
    df: dict[str, int] = {}
    for entry in entries:
        for term in entry.get("term_freq", {}):
            df[term] = df.get(term, 0) + 1

    scored: list[tuple[str, float, Path]] = []
    for entry in entries:
        score = 0.0
        dl = max(entry.get("total_tokens", 1), 1)
        tf_map: dict[str, int] = entry.get("term_freq", {})
        structured: set[str] = set(entry.get("structured_terms", []))

        for term in query_terms:
            tf = tf_map.get(term, 0)
            # structured_terms are high-value identifiers; treat a match as tf=1
            if tf == 0 and term in structured:
                tf = 1
            if tf == 0:
                continue
            df_t = max(df.get(term, 0), 0)
            idf = math.log((N - df_t + 0.5) / (df_t + 0.5) + 1)
            tf_norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
            score += idf * tf_norm

        if score > 0:
            scored.append((entry["chunk_id"], score, chunks_dir / entry["file"]))

    return sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]
