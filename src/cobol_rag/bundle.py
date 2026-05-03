from __future__ import annotations

import json
from pathlib import Path


def resolve_index_path(path: Path) -> Path:
    """If path is a knowledge-base_rag bundle (has manifest.json), return recommended_index_path.

    Leaves non-bundle paths unchanged so callers can use this unconditionally.
    """
    manifest = path / "manifest.json"
    if not manifest.exists():
        return path
    try:
        with manifest.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return path
    index_subpath = data.get("recommended_index_path", "chunks")
    return path / index_subpath


def list_bundle_chunks(chunks_dir: Path) -> list[Path] | None:
    """Return the authoritative list of chunk files from chunks_manifest.json.

    Returns None when chunks_manifest.json is absent, signalling the caller to
    fall back to a plain directory load.
    """
    manifest = chunks_dir / "chunks_manifest.json"
    if not manifest.exists():
        return None
    try:
        with manifest.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    return [chunks_dir / c["file"] for c in data.get("chunks", [])]


def find_bm25_index(chunks_dir: Path) -> Path | None:
    """Return the path to bm25_index.json if present alongside the chunks."""
    candidate = chunks_dir / "bm25_index.json"
    return candidate if candidate.exists() else None
