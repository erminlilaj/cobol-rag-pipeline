"""Focused unit tests for loader metadata flattening, indexable filtering, and bundle resolution."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from cobol_rag.bundle import find_bm25_index, list_bundle_chunks, resolve_index_path
from cobol_rag.config import AppConfig, IndexConfig


def _make_config(include_non_indexable: bool = False) -> AppConfig:
    return AppConfig(index=IndexConfig(include_non_indexable=include_non_indexable))


# ---------------------------------------------------------------------------
# GenericJsonLoader — metadata flattening
# ---------------------------------------------------------------------------

class TestMetadataFlattening:
    def _loader(self, include_non_indexable: bool = False):
        from cobol_rag.loaders.generic_json import GenericJsonLoader
        return GenericJsonLoader(config=_make_config(include_non_indexable))

    def test_flat_record_no_nested_metadata(self, tmp_path: Path):
        data = {"text": "hello", "kind": "test"}
        f = tmp_path / "doc.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        assert len(docs) == 1
        assert docs[0].document.text == "hello"

    def test_nested_metadata_merged(self, tmp_path: Path):
        data = {
            "text": "PDB305 calls TE0CDUMP",
            "metadata": {
                "chunk_type": "dependencies",
                "chunk_id": "PDB305.CBL:dependencies",
                "program": "PDB305.CBL",
                "indexable": True,
                "parse_quality": "degraded",
            },
        }
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        assert len(docs) == 1
        meta = docs[0].document.metadata
        assert meta["chunk_type"] == "dependencies"
        assert meta["chunk_id"] == "PDB305.CBL:dependencies"
        assert meta["program"] == "PDB305.CBL"
        assert meta["parse_quality"] == "degraded"

    def test_nested_metadata_preserves_source_fields(self, tmp_path: Path):
        data = {
            "text": "x",
            "metadata": {
                "chunk_type": "workflow",
                "content_hash": "source-provided-hash",
                "source_id": "source-provided-id",
            },
        }
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        meta = docs[0].document.metadata
        # pipeline-assigned fields must survive
        assert "source_id" in meta
        assert "source_path" in meta
        assert "source_format" in meta
        assert "content_hash" in meta
        assert meta["content_hash"] != "source-provided-hash"
        assert meta["source_id"] != "source-provided-id"

    def test_list_of_records(self, tmp_path: Path):
        data = [
            {"text": "a", "metadata": {"chunk_type": "workflow"}},
            {"text": "b", "metadata": {"chunk_type": "paragraph_logic"}},
        ]
        f = tmp_path / "chunks.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        assert len(docs) == 2
        assert docs[0].document.metadata["chunk_type"] == "workflow"
        assert docs[1].document.metadata["chunk_type"] == "paragraph_logic"

    def test_complex_metadata_values_dropped(self, tmp_path: Path):
        data = {
            "text": "x",
            "metadata": {
                "chunk_type": "dependencies",
                "cics_resources": [{"target": "PDB3051", "target_kind": "MAP"}],
                "static_values": [{"name": "WS-X", "value": "ABC"}],
                "confidence": {"score": 0.7},
                "calls": ["PD0UTI01"],
            },
        }
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        meta = docs[0].document.metadata
        assert meta["chunk_type"] == "dependencies"
        assert "cics_resources" not in meta
        assert "static_values" not in meta
        assert "confidence" not in meta
        assert "calls" not in meta

    def test_copybook_facts_appended_to_text_not_metadata(self, tmp_path: Path):
        data = {
            "text": "Analysis health for PDB305.CBL.",
            "metadata": {
                "chunk_type": "cobol_analysis_health",
                "copybooks_used": ["PDRTELR", "PDRAL01"],
                "total_copybooks": 15,
                "resolved_copybooks": 9,
                "stubbed_copybook_count": 6,
                "stubbed_copybooks": [
                    "DFHAID [CICS]: PF key checks will reference undefined variables",
                ],
            },
        }
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        text = docs[0].document.text
        meta = docs[0].document.metadata

        assert "Structured facts from source JSON" in text
        assert "copybooks_used: PDRTELR, PDRAL01" in text
        assert "total_copybooks: 15" in text
        assert "stubbed_copybooks: DFHAID [CICS]" in text
        assert "copybooks_used" not in meta
        assert "stubbed_copybooks" not in meta

    def test_non_whitelisted_scalar_metadata_dropped(self, tmp_path: Path):
        data = {
            "text": "x",
            "metadata": {
                "chunk_type": "dependencies",
                "title": "Large title is not a filter field",
                "kind": "legacy",
            },
        }
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        loader = self._loader()
        docs = loader.load(f)
        meta = docs[0].document.metadata
        assert meta["chunk_type"] == "dependencies"
        assert "title" not in meta
        assert "kind" not in meta


# ---------------------------------------------------------------------------
# GenericJsonLoader — indexable filtering
# ---------------------------------------------------------------------------

class TestIndexableFiltering:
    def _loader(self, include_non_indexable: bool = False):
        from cobol_rag.loaders.generic_json import GenericJsonLoader
        return GenericJsonLoader(config=_make_config(include_non_indexable))

    def test_indexable_false_skipped_by_default(self, tmp_path: Path):
        data = {"text": "skip me", "metadata": {"chunk_type": "x", "indexable": False}}
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        docs = self._loader().load(f)
        assert docs == []

    def test_indexable_true_included(self, tmp_path: Path):
        data = {"text": "include me", "metadata": {"chunk_type": "x", "indexable": True}}
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        docs = self._loader().load(f)
        assert len(docs) == 1

    def test_indexable_absent_included(self, tmp_path: Path):
        # Records without an indexable flag (non-cobol-rekt JSON) should pass through
        data = {"text": "plain record"}
        f = tmp_path / "doc.json"
        f.write_text(json.dumps(data))
        docs = self._loader().load(f)
        assert len(docs) == 1

    def test_indexable_false_included_when_flag_set(self, tmp_path: Path):
        data = {"text": "non-indexable", "metadata": {"indexable": False}}
        f = tmp_path / "chunk.json"
        f.write_text(json.dumps(data))
        docs = self._loader(include_non_indexable=True).load(f)
        assert len(docs) == 1

    def test_list_mixed_indexable(self, tmp_path: Path):
        data = [
            {"text": "keep", "metadata": {"indexable": True}},
            {"text": "drop", "metadata": {"indexable": False}},
            {"text": "keep too"},  # no indexable flag → keep
        ]
        f = tmp_path / "mixed.json"
        f.write_text(json.dumps(data))
        docs = self._loader().load(f)
        assert len(docs) == 2
        texts = {d.document.text for d in docs}
        assert "keep" in texts
        assert "keep too" in texts
        assert "drop" not in texts


# ---------------------------------------------------------------------------
# Bundle resolution
# ---------------------------------------------------------------------------

class TestBundleResolution:
    def _make_bundle(self, tmp_path: Path, recommended: str = "chunks") -> Path:
        bundle = tmp_path / "knowledge-base_rag"
        bundle.mkdir()
        manifest = {"recommended_index_path": recommended, "program": "TEST.CBL"}
        (bundle / "manifest.json").write_text(json.dumps(manifest))
        chunks = bundle / recommended
        chunks.mkdir(parents=True, exist_ok=True)
        return bundle

    def test_non_bundle_path_unchanged(self, tmp_path: Path):
        assert resolve_index_path(tmp_path) == tmp_path

    def test_bundle_resolves_to_recommended_path(self, tmp_path: Path):
        bundle = self._make_bundle(tmp_path, "chunks")
        resolved = resolve_index_path(bundle)
        assert resolved == bundle / "chunks"

    def test_bundle_custom_index_path(self, tmp_path: Path):
        bundle = self._make_bundle(tmp_path, "index_files")
        assert resolve_index_path(bundle) == bundle / "index_files"

    def test_list_bundle_chunks_from_manifest(self, tmp_path: Path):
        bundle = self._make_bundle(tmp_path)
        chunks_dir = bundle / "chunks"
        chunk_files = ["a.json", "b.json"]
        for name in chunk_files:
            (chunks_dir / name).write_text("{}")
        chunks_manifest = {
            "chunks": [{"file": name} for name in chunk_files]
        }
        (chunks_dir / "chunks_manifest.json").write_text(json.dumps(chunks_manifest))

        result = list_bundle_chunks(chunks_dir)
        assert result is not None
        assert len(result) == 2
        names = {p.name for p in result}
        assert names == set(chunk_files)

    def test_list_bundle_chunks_no_manifest_returns_none(self, tmp_path: Path):
        assert list_bundle_chunks(tmp_path) is None

    def test_find_bm25_index_present(self, tmp_path: Path):
        (tmp_path / "bm25_index.json").write_text("{}")
        assert find_bm25_index(tmp_path) == tmp_path / "bm25_index.json"

    def test_find_bm25_index_absent(self, tmp_path: Path):
        assert find_bm25_index(tmp_path) is None


# ---------------------------------------------------------------------------
# Sync pruning
# ---------------------------------------------------------------------------

class TestSyncPruning:
    def test_obsolete_manifest_entries_under_sync_root_are_removed(self, tmp_path: Path):
        from cobol_rag.sync import ManifestEntry, _obsolete_items

        root = tmp_path / "bundle" / "chunks"
        root.mkdir(parents=True)
        current = root / "current.json"
        stale = root / "stale.json"
        other = tmp_path / "other" / "outside.json"

        manifest = {
            "current": ManifestEntry(
                source_id="current",
                source_path=str(current),
                source_format="generic_json",
                content_hash="a",
            ),
            "stale": ManifestEntry(
                source_id="stale",
                source_path=str(stale),
                source_format="generic_json",
                content_hash="b",
            ),
            "outside": ManifestEntry(
                source_id="outside",
                source_path=str(other),
                source_format="generic_json",
                content_hash="c",
            ),
        }

        obsolete = _obsolete_items(
            manifest=manifest,
            sync_root=root,
            current_source_ids={"current"},
        )

        assert [item.action for item in obsolete] == ["remove"]
        assert [item.source_id for item in obsolete] == ["stale"]

    def test_write_manifest_drops_removed_items(self, tmp_path: Path):
        from cobol_rag.sync import SyncItem, SyncPlan, write_manifest

        manifest_path = tmp_path / "manifest.json"
        existing = {
            "sources": {
                "stale": {
                    "source_id": "stale",
                    "source_path": str(tmp_path / "stale.json"),
                    "source_format": "generic_json",
                    "content_hash": "old",
                }
            },
            "bm25_index_path": str(tmp_path / "bm25_index.json"),
        }
        plan = SyncPlan(
            collection="test",
            inbox_dir=tmp_path,
            manifest_path=manifest_path,
            dry_run=False,
            items=[
                SyncItem(
                    action="remove",
                    source_id="stale",
                    source_path=str(tmp_path / "stale.json"),
                    source_format="generic_json",
                    content_hash="old",
                )
            ],
        )

        write_manifest(manifest_path, plan, existing_raw=existing)

        written = json.loads(manifest_path.read_text())
        assert written["sources"] == {}
        assert written["bm25_index_path"] == str(tmp_path / "bm25_index.json")


# ---------------------------------------------------------------------------
# Retrieval reranking
# ---------------------------------------------------------------------------

class TestIntentReranking:
    def test_copybook_query_expansion_targets_structured_count_fields(self):
        from cobol_rag.retrieve import _detect_intent, _expanded_query_for_intent

        query = "how many copybooks are inside?"
        expanded = _expanded_query_for_intent(query, _detect_intent(query))

        assert "total_copybooks" in expanded
        assert "resolved_copybooks" in expanded
        assert "stubbed_copybooks" in expanded

    def test_copybook_question_prefers_copybook_evidence(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.4,
                text="Static values for PDB305.",
                metadata={"chunk_type": "static_values"},
            ),
            RetrievalResult(
                score=0.1,
                text="Structured facts from source JSON:\n- copybooks_used: PDRTELR, PDRAL01",
                metadata={"chunk_type": "program_summary"},
            ),
        ]

        ranked = _intent_rerank("which copybooks are inside the program?", results, 2)
        assert ranked[0].metadata["chunk_type"] == "program_summary"

    def test_copybook_question_drops_paragraphs_when_canonical_chunks_exist(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.5,
                text="Paragraph text. Structured facts from source JSON: - copybooks_used: A, B",
                metadata={"chunk_type": "paragraph_logic"},
            ),
            RetrievalResult(
                score=0.1,
                text="Structured facts from source JSON: - total_copybooks: 2",
                metadata={"chunk_type": "cobol_analysis_health"},
            ),
        ]

        ranked = _intent_rerank("how many copybooks are inside?", results, 2)
        assert [r.metadata["chunk_type"] for r in ranked] == ["cobol_analysis_health"]

    def test_static_value_question_prefers_static_values(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.4,
                text="Program summary.",
                metadata={"chunk_type": "program_summary"},
            ),
            RetrievalResult(
                score=0.1,
                text="Static values for PDB305.",
                metadata={"chunk_type": "static_values"},
            ),
        ]

        ranked = _intent_rerank("what hardcoded values are in PDB305?", results, 2)
        assert ranked[0].metadata["chunk_type"] == "static_values"

    def test_resource_question_prefers_dependencies(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.4,
                text="Paragraph logic.",
                metadata={"chunk_type": "paragraph_logic"},
            ),
            RetrievalResult(
                score=0.1,
                text="External dependencies for program PDB305.CBL: CICS commands.",
                metadata={"chunk_type": "dependencies"},
            ),
        ]

        ranked = _intent_rerank("what resources does PDB305 use?", results, 2)
        assert ranked[0].metadata["chunk_type"] == "dependencies"

    def test_external_program_question_prefers_external_call_chunk(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.6,
                text="CICS operations: SEND MAP PDB3051, RETURN TRANSID PRED.",
                metadata={"chunk_type": "cics_operations"},
            ),
            RetrievalResult(
                score=0.1,
                text="External program calls:\n- LINK PD0GCODA in LINK-CODA: COMMAREA WPDRGCODA, LENGTH PDRGCODA-LUNGH.",
                metadata={"chunk_type": "external_program_calls"},
            ),
            RetrievalResult(
                score=0.5,
                text="Static values for PDB305.",
                metadata={"chunk_type": "static_values"},
            ),
        ]

        ranked = _intent_rerank(
            "Which outside programs and with which parameters are used?",
            results,
            3,
        )
        assert ranked[0].metadata["chunk_type"] == "external_program_calls"
        assert all(r.metadata["chunk_type"] != "static_values" for r in ranked)

    def test_external_calls_phrase_detects_external_program_intent(self):
        from cobol_rag.retrieve import _detect_intent

        assert _detect_intent("tell me about external calls") == "external_programs"

    def test_parameter_preparation_question_is_control_flow_not_external_listing(self):
        from cobol_rag.retrieve import _detect_intent

        question = (
            "In PDCBVC.CBL, how are the parameters for the external program PD1VOCI "
            "prepared, and how does TWCOB-FUNZIONE influence PD1VOCI-FUNZIONE?"
        )

        assert _detect_intent(question) == "pd1voci_parameter_preparation"

    def test_specific_pdcbvc_intents_are_detected(self):
        from cobol_rag.retrieve import _detect_intent

        assert _detect_intent("How does PDCBVC calculate the total number of pages?") == "pagination"
        assert _detect_intent("How does PREP-RIGA build each displayed row?") == "row_build"
        assert _detect_intent("Which fields from PDRTWA2 are copied into PD1VOCI before LINK?") == "field_mapping"
        assert _detect_intent("What does variable FUNZ do?") == "variable_usage"
        assert _detect_intent("What happens when TWCOB-FUNZIONE is I and TWCOB-ID-SISTEMA = 'IP'?") == "condition_path"

    def test_control_flow_rerank_boosts_exact_identifiers_and_cfg(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.9,
                text="Program PDCBVC.CBL has 922 CFG nodes and high complexity.",
                metadata={"chunk_type": "program_summary", "program": "PDCBVC.CBL"},
            ),
            RetrievalResult(
                score=0.1,
                text="IF TWCOB-FASE = '1' THEN GO TO BROWSE-FASE1. IF TWCOB-FASE = '2' THEN GO TO BROWSE-FASE2.",
                metadata={
                    "chunk_type": "controlflow.cfg",
                    "chunk_id": "PDCBVC.CBL:controlflow:entry",
                    "program": "PDCBVC.CBL",
                },
            ),
        ]

        ranked = _intent_rerank(
            "How does TWCOB-FASE decide BROWSE-FASE1 or BROWSE-FASE2 in PDCBVC.CBL?",
            results,
            2,
        )

        assert ranked[0].metadata["chunk_type"] == "controlflow.cfg"
        assert all(r.metadata["chunk_type"] != "program_summary" for r in ranked)

    def test_hybrid_retrieval_falls_back_to_bm25_when_vector_fails(self, tmp_path, monkeypatch):
        import json

        from cobol_rag.config import AppConfig, RetrievalConfig
        import cobol_rag.retrieve as retrieve_module

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (chunks_dir / "entry.json").write_text(
            json.dumps(
                {
                    "text": "IF TWCOB-FASE = '1' THEN GO TO BROWSE-FASE1.",
                    "metadata": {
                        "chunk_id": "PDCBVC.CBL:entry",
                        "chunk_type": "controlflow.cfg",
                        "program": "PDCBVC.CBL",
                    },
                }
            )
        )
        index_path = chunks_dir / "bm25_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "total_chunks": 1,
                    "avg_doc_length": 5,
                    "entries": [
                        {
                            "chunk_id": "PDCBVC.CBL:entry",
                            "file": "entry.json",
                            "total_tokens": 5,
                            "term_freq": {"TWCOB-FASE": 1},
                            "structured_terms": ["TWCOB-FASE"],
                        }
                    ],
                }
            )
        )

        def fail_vector(*_args, **_kwargs):
            raise RuntimeError("embedding server unavailable")

        monkeypatch.setattr(retrieve_module, "_vector", fail_vector)
        config = AppConfig(retrieval=RetrievalConfig(bm25_top_k=2))

        results = retrieve_module._hybrid("TWCOB-FASE", config, 1, None, [index_path])

        assert len(results) == 1
        assert results[0].metadata["chunk_type"] == "controlflow.cfg"

    def test_bm25_tokenizer_matches_punctuated_identifiers(self, tmp_path):
        import json

        from cobol_rag.bm25 import bm25_retrieve

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (chunks_dir / "pagination.json").write_text(
            json.dumps(
                {
                    "text": "Pagination uses PF7, PF8, WCTPAG, and TWCOB-VARCONT-NPAGINA.",
                    "metadata": {
                        "chunk_id": "PDCBVC.CBL:screen.pagination:1",
                        "chunk_type": "screen.pagination",
                        "program": "PDCBVC.CBL",
                    },
                }
            )
        )
        index = {
            "total_chunks": 1,
            "avg_doc_length": 4,
            "entries": [
                {
                    "chunk_id": "PDCBVC.CBL:screen.pagination:1",
                    "file": "pagination.json",
                    "chunk_type": "screen.pagination",
                    "total_tokens": 4,
                    "term_freq": {"PF7": 1, "PF8": 1, "WCTPAG": 1},
                    "structured_terms": ["TWCOB-VARCONT-NPAGINA"],
                }
            ],
        }

        hits = bm25_retrieve("PF7, PF8, and TWCOB-VARCONT-NPAGINA?", index, chunks_dir, 3)

        assert [hit[0] for hit in hits] == ["PDCBVC.CBL:screen.pagination:1"]

    def test_hybrid_retrieval_samples_canonical_chunks_for_intent(self, tmp_path, monkeypatch):
        import json

        from cobol_rag.config import AppConfig, RetrievalConfig
        import cobol_rag.retrieve as retrieve_module

        chunks_dir = tmp_path / "chunks"
        chunks_dir.mkdir()
        (chunks_dir / "summary.json").write_text(
            json.dumps(
                {
                    "text": "Browse summary mentions PF7 PF8 and many generic words.",
                    "metadata": {
                        "chunk_id": "PDCBVC.CBL:paragraph:BROWSE-FASE2",
                        "chunk_type": "paragraph_logic",
                        "program": "PDCBVC.CBL",
                    },
                }
            )
        )
        (chunks_dir / "pagination.json").write_text(
            json.dumps(
                {
                    "text": "Screen pagination facts for PDCBVC.CBL: PF7 subtracts WCTPAG, PF8 adds WCTPAG.",
                    "metadata": {
                        "chunk_id": "PDCBVC.CBL:screen.pagination:1",
                        "chunk_type": "screen.pagination",
                        "program": "PDCBVC.CBL",
                    },
                }
            )
        )
        index_path = chunks_dir / "bm25_index.json"
        index_path.write_text(
            json.dumps(
                {
                    "total_chunks": 2,
                    "avg_doc_length": 6,
                    "entries": [
                        {
                            "chunk_id": "PDCBVC.CBL:paragraph:BROWSE-FASE2",
                            "file": "summary.json",
                            "chunk_type": "paragraph_logic",
                            "total_tokens": 6,
                            "term_freq": {"BROWSE": 3, "PF7": 1, "PF8": 1},
                            "structured_terms": [],
                        },
                        {
                            "chunk_id": "PDCBVC.CBL:screen.pagination:1",
                            "file": "pagination.json",
                            "chunk_type": "screen.pagination",
                            "total_tokens": 6,
                            "term_freq": {"PAGINATION": 1, "PF7": 1, "PF8": 1, "WCTPAG": 2},
                            "structured_terms": [],
                        },
                    ],
                }
            )
        )

        def fail_vector(*_args, **_kwargs):
            raise RuntimeError("embedding server unavailable")

        monkeypatch.setattr(retrieve_module, "_vector", fail_vector)
        config = AppConfig(retrieval=RetrievalConfig(bm25_top_k=1))

        results = retrieve_module._hybrid(
            "How is pagination maintained when PF7 or PF8 is pressed?",
            config,
            2,
            None,
            [index_path],
        )

        assert {result.metadata["chunk_type"] for result in results} == {
            "paragraph_logic",
            "screen.pagination",
        }

    def test_bm25_paths_are_filtered_when_program_is_named(self, tmp_path):
        from cobol_rag.retrieve import _filter_bm25_paths_by_program_mention

        pdc = tmp_path / "PDCBVC.CBL.knowledge-base_rag" / "chunks" / "bm25_index.json"
        pdb = tmp_path / "PDB305.CBL.knowledge-base_rag" / "chunks" / "bm25_index.json"

        paths = _filter_bm25_paths_by_program_mention(
            "In PDCBVC.CBL, explain pagination.",
            [pdb, pdc],
        )

        assert paths == [pdc]

    def test_control_flow_rerank_prefers_targeted_screen_chunk(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.9,
                text="BROWSE-FASE2 generic paragraph mentions PF7 and PF8.",
                metadata={"chunk_type": "paragraph_logic", "program": "PDCBVC.CBL"},
            ),
            RetrievalResult(
                score=0.1,
                text="Screen pagination facts: PF7 subtracts WCTPAG, PF8 adds WCTPAG, ENTER keeps page state.",
                metadata={"chunk_type": "screen.pagination", "program": "PDCBVC.CBL"},
            ),
        ]

        ranked = _intent_rerank(
            "In PDCBVC.CBL, how is pagination maintained when ENTER, PF7, or PF8 are pressed?",
            results,
            2,
        )

        assert ranked[0].metadata["chunk_type"] == "screen.pagination"

    def test_pagination_intent_penalizes_program_summary(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.9,
                text="Program PDCBVC.CBL has 922 CFG nodes and many copybooks.",
                metadata={"chunk_type": "program_summary", "program": "PDCBVC.CBL"},
            ),
            RetrievalResult(
                score=0.1,
                text="CALCOLA-NPAG divides MAX-RIGHE into PD1VOCI-TABVOX-NUMERO giving NPAGT remainder RESTO.",
                metadata={"chunk_type": "paragraph_logic", "paragraph": "CALCOLA-NPAG", "program": "PDCBVC.CBL"},
            ),
        ]

        ranked = _intent_rerank("How does PDCBVC calculate the total number of pages?", results, 2)

        assert ranked[0].metadata["paragraph"] == "CALCOLA-NPAG"
        assert all(r.metadata["chunk_type"] != "program_summary" for r in ranked)

    def test_dataset_question_prefers_dedicated_resource_chunk(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.5,
                text="External dependencies for PDB305.CBL.",
                metadata={"chunk_type": "dependencies"},
            ),
            RetrievalResult(
                score=0.1,
                text="Datasets, tables, and resources:\nDB2 tables read: DUAL.\nCICS queues: TWCOB-TS-CODA.",
                metadata={"chunk_type": "datasets_tables_resources"},
            ),
            RetrievalResult(
                score=0.4,
                text="External program calls:\n- LINK PD0GCODA.",
                metadata={"chunk_type": "external_program_calls"},
            ),
        ]

        ranked = _intent_rerank("Which dataset/Tables are used by this program?", results, 3)
        assert ranked[0].metadata["chunk_type"] == "datasets_tables_resources"
        assert all(r.metadata["chunk_type"] != "external_program_calls" for r in ranked)

    def test_dead_code_question_keeps_only_explicit_evidence_when_available(self):
        from cobol_rag.retrieve import RetrievalResult, _intent_rerank

        results = [
            RetrievalResult(
                score=0.9,
                text="Paragraph SEND-PDB3051 sends a map.",
                metadata={"chunk_type": "paragraph_logic"},
            ),
            RetrievalResult(
                score=0.1,
                text="Dead-code analysis: no explicit unused paragraphs were detected.",
                metadata={"chunk_type": "dead_code"},
            ),
            RetrievalResult(
                score=0.8,
                text="Static values for PDB305.",
                metadata={"chunk_type": "static_values"},
            ),
        ]

        ranked = _intent_rerank("Is there any unused code/copy in this program?", results, 3)
        assert [r.metadata["chunk_type"] for r in ranked] == ["dead_code"]


# ---------------------------------------------------------------------------
# Structured answer helpers
# ---------------------------------------------------------------------------

class TestStructuredAnswers:
    def test_copybook_answer_uses_structured_facts(self):
        from cobol_rag.query import _try_copybook_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "Analysis health for PDB305.CBL. Notes: degraded parse quality.\n\n"
                    "Structured facts from source JSON:\n"
                    "- resolved_copybooks: 9\n"
                    "- stubbed_copybook_count: 6\n"
                    "- stubbed_copybooks: DFHAID [CICS]: PF key checks, "
                    "DFHBMSCA [CICS]: Screen attributes, PDPSQLER: Unknown impact, "
                    "PDRTELR: missing variables, PDWSQLER: Unknown impact, SQLCA: Unknown impact\n"
                    "- total_copybooks: 15"
                ),
                metadata={"chunk_type": "cobol_analysis_health"},
            ),
            RetrievalResult(
                score=1.0,
                text=(
                    "Structured facts from source JSON:\n"
                    "- copybooks_used: PDRTELR, PDRAL01, PDRGCODA, PDRUTI01"
                ),
                metadata={"chunk_type": "program_summary"},
            ),
        ]

        answer = _try_copybook_answer("which copybooks are found and stubbed?", sources)

        assert answer is not None
        assert "15 total, 9 resolved/found, 6 stubbed" in answer
        assert "Copybooks listed as used: PDRTELR, PDRAL01, PDRGCODA, PDRUTI01." in answer
        assert "Stubbed copybooks: DFHAID [CICS], DFHBMSCA [CICS], PDPSQLER, PDRTELR, PDWSQLER, SQLCA." in answer

    def test_static_values_answer_uses_static_chunk_lines(self):
        from cobol_rag.query import _try_static_values_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "Static and forced values for PDB305.CBL:\n"
                    "- PD1AC-FUNZIONE = '01' (category: parameter setup)\n"
                    "- WABEND-CODE = 'BR00' (category: abend code)"
                ),
                metadata={"chunk_type": "static_values"},
            ),
            RetrievalResult(
                score=0.9,
                text="CICS operations for PDB305.CBL.",
                metadata={"chunk_type": "cics_operations"},
            ),
        ]

        answer = _try_static_values_answer("Is there any Forced value, and for who?", sources)

        assert answer is not None
        assert "Forced/static values found:" in answer
        assert "- PD1AC-FUNZIONE = '01'" in answer
        assert "- WABEND-CODE = 'BR00'" in answer

    def test_static_values_answer_deduplicates_curated_and_appended_facts(self):
        from cobol_rag.query import _try_static_values_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "Static and forced values for PDB305.CBL:\n"
                    "- PD1AC-FUNZIONE: '01'. Category: parameter setup\n\n"
                    "Structured facts from source JSON:\n"
                    "- PD1AC-FUNZIONE: '01'"
                ),
                metadata={"chunk_type": "static_values"},
            )
        ]

        answer = _try_static_values_answer("Is there any Forced value?", sources)

        assert answer is not None
        assert answer.count("PD1AC-FUNZIONE") == 1
        assert "Category: parameter setup" in answer

    def test_external_program_answer_uses_dedicated_call_chunk(self):
        from cobol_rag.query import _try_external_programs_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "External program calls for PDB305.CBL:\n"
                    "- LINK PD1AC in LINK-PD1AC: COMMAREA WPD1AC, LENGTH PD1AC-LUNGH.\n"
                    "- XCTL PDPRED in XCTL-MAIN."
                ),
                metadata={"chunk_type": "external_program_calls"},
            )
        ]

        answer = _try_external_programs_answer(
            "Which outside programs and with which parameters are used?",
            sources,
        )

        assert answer is not None
        assert "External program calls:" in answer
        assert "LINK PD1AC" in answer
        assert "COMMAREA WPD1AC" in answer
        assert "XCTL PDPRED" in answer

    def test_external_calls_short_phrase_uses_dedicated_call_chunk(self):
        from cobol_rag.query import _try_external_programs_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="External program calls:\n- LINK PD0GCODA in LINK-CODA: COMMAREA WPDRGCODA, LENGTH PDRGCODA-LUNGH.",
                metadata={"chunk_type": "external_program_calls"},
            )
        ]

        answer = _try_external_programs_answer("tell me about external calls", sources)

        assert answer is not None
        assert "LINK PD0GCODA" in answer

    def test_external_commarea_question_filters_to_commarea_calls(self):
        from cobol_rag.query import _try_external_programs_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "External program calls:\n"
                    "- LINK PD1AC in LINK-PD1AC: COMMAREA WPD1AC, LENGTH PD1AC-LUNGH, target_source literal.\n"
                    "- XCTL PDPRED in XCTL-MAIN: target_source literal."
                ),
                metadata={"chunk_type": "external_program_calls"},
            )
        ]

        answer = _try_external_programs_answer("which of those use COMMAREA?", sources)

        assert answer is not None
        assert "LINK PD1AC" in answer
        assert "COMMAREA WPD1AC" in answer
        assert "target_source literal" not in answer
        assert "XCTL PDPRED" not in answer

    def test_external_program_answer_does_not_hijack_parameter_preparation(self):
        from cobol_rag.query import _try_external_programs_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="External program calls:\n- LINK PD1VOCI in LINK-PD1VOCI: COMMAREA WPD1VOCI, LENGTH 32000.",
                metadata={"chunk_type": "external_program_calls"},
            )
        ]

        answer = _try_external_programs_answer(
            "How are the parameters for external program PD1VOCI prepared?",
            sources,
        )

        assert answer is None

    def test_external_program_answer_drops_unknown_duplicate_calls(self):
        from cobol_rag.query import _try_external_programs_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "External program calls:\n"
                    "- CALL PXRSEMAF: USING PXCSEMAF-AREA, target_source dynamic.\n"
                    "- CALL UNKNOWN: USING PXCSEMAF-AREA, target_source dynamic."
                ),
                metadata={"chunk_type": "external_program_calls"},
            )
        ]

        answer = _try_external_programs_answer("tell me about external calls", sources)

        assert answer is not None
        assert "CALL PXRSEMAF" in answer
        assert "CALL UNKNOWN" not in answer
        assert "target_source dynamic" not in answer

    def test_program_summary_answer_uses_summary_chunk(self):
        from cobol_rag.query import _try_program_summary_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "This is a very-high-complexity CICS online program with DB2 database access.\n"
                    "Program PDB305.CBL has 636 CFG nodes, 935 edges, and 1 variables defined.\n"
                    "Analysis confidence: medium (0.66).\n"
                    "Structured facts from source JSON:\n"
                    "- copybooks_used: PDRTELR"
                ),
                metadata={"chunk_type": "program_summary"},
            )
        ]

        answer = _try_program_summary_answer("what this program is about", sources)

        assert answer is not None
        assert "Program summary:" in answer
        assert "CICS online program with DB2 database access" in answer
        assert "Structured facts from source JSON" not in answer

    def test_datasets_tables_answer_uses_dedicated_resource_chunk(self):
        from cobol_rag.query import _try_datasets_tables_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "Datasets, tables, and resources for PDB305.CBL:\n"
                    "DB2 tables read: DUAL.\n"
                    "SQL operations: SELECT.\n"
                    "CICS queues: TWCOB-TS-CODA.\n"
                    "CICS mapsets: PDB305M."
                ),
                metadata={"chunk_type": "datasets_tables_resources"},
            )
        ]

        answer = _try_datasets_tables_answer("Which dataset/Tables are used by this program?", sources)

        assert answer is not None
        assert "Datasets, tables, and resources:" in answer
        assert "DB2 tables read: DUAL." in answer
        assert "CICS queues: TWCOB-TS-CODA." in answer

    def test_dead_code_answer_refuses_to_infer_no_unused_code(self):
        from cobol_rag.query import _try_dead_code_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="Dependencies for PDB305.CBL: CICS SEND and RETURN.",
                metadata={"chunk_type": "dependencies"},
            )
        ]

        answer = _try_dead_code_answer("Is there any unused code/copy in this program?", sources)

        assert answer is not None
        assert "do not contain enough explicit dead-code or unused-copy evidence" in answer
        assert "will not infer" in answer

    def test_comments_answer_refuses_without_comment_chunk(self):
        from cobol_rag.query import _try_comments_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="This is a CICS online program with DB2 database access.",
                metadata={"chunk_type": "program_summary"},
            )
        ]

        answer = _try_comments_answer("what comments does this program have?", sources)

        assert answer is not None
        assert "do not contain a dedicated comments or commented-out-code chunk" in answer

    def test_copybook_line_question_does_not_return_status(self):
        from cobol_rag.query import _try_copybook_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="Structured facts from source JSON:\n- total_copybooks: 15",
                metadata={"chunk_type": "cobol_analysis_health"},
            )
        ]

        answer = _try_copybook_answer("in which lines are copybooks mentioned?", sources)

        assert answer is not None
        assert "do not contain source line numbers" in answer
        assert "15 total" not in answer

    def test_copybook_parameter_question_does_not_return_status(self):
        from cobol_rag.query import _try_copybook_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="Structured facts from source JSON:\n- total_copybooks: 15",
                metadata={"chunk_type": "cobol_analysis_health"},
            )
        ]

        answer = _try_copybook_answer("what parameters do you get from copybooks?", sources)

        assert answer is not None
        assert "do not contain copybook field/parameter extraction" in answer
        assert "15 total" not in answer

    def test_pd1voci_parameter_answer_uses_iniz_param_not_external_calls(self):
        from cobol_rag.query import _try_pd1voci_parameter_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "INIZ-PARAM\n"
                    "- MOVE SPACE TO PD1VOCI-DATI\n"
                    "- MOVE TWCOB-FUNZIONE TO PD1VOCI-TIPO-VARIAZ\n"
                    "- MOVE '11' TO PD1VOCI-FUNZIONE\n"
                    "- MOVE '12' TO PD1VOCI-FUNZIONE\n"
                    "- MOVE '02' TO PD1VOCI-FUNZIONE\n"
                    "- MOVE 'A' TO PD1VOCI-TIPO-ESTRA\n"
                    "INIZ-PARAM-010 IF TWCOB-VARCONT-NUMFUNZ = '2' THEN MOVE '1' TO PD1VOCI-TIPO-VOCE"
                ),
                metadata={"chunk_type": "paragraph_logic", "paragraph": "INIZ-PARAM"},
            )
        ]

        answer = _try_pd1voci_parameter_answer(
            "How are the parameters for external program PD1VOCI prepared and how does TWCOB-FUNZIONE influence it?",
            sources,
        )

        assert answer is not None
        assert "INIZ-PARAM" in answer
        assert "PD1VOCI-FUNZIONE" in answer
        assert "TWCOB-VARCONT-NUMFUNZ" in answer
        assert "External program calls:" not in answer

    def test_pagination_answer_uses_formula_and_navigation_evidence(self):
        from cobol_rag.query import _try_pagination_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="CALCOLA-NPAG DIVIDE MAX-RIGHE INTO PD1VOCI-TABVOX-NUMERO GIVING NPAGT REMAINDER RESTO. ADD +1 TO NPAGT.",
                metadata={"chunk_type": "paragraph_logic", "paragraph": "CALCOLA-NPAG"},
            ),
            RetrievalResult(
                score=1.0,
                text="Screen pagination facts: BROWSE-FASE2-PF7 SUBTRACT 2 FROM WCTPAG. BROWSE-FASE2-PF8. BROWSE-FASE2-ENTER MOVE WCTPAG TO TWCOB-VARCONT-NPAGINA.",
                metadata={"chunk_type": "screen.pagination"},
            ),
        ]

        answer = _try_pagination_answer("How does PDCBVC calculate the total number of pages?", sources)

        assert answer is not None
        assert "CALCOLA-NPAG" in answer
        assert "MAX-RIGHE" in answer
        assert "NPAGT" in answer
        assert "RESTO" in answer
        assert "Program summary" not in answer

    def test_condition_path_answer_blocks_function_code_hallucination(self):
        from cobol_rag.query import _try_condition_path_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "IF (TWCOB-FUNZIONE = 'I' OR 'A' OR 'C' OR 'D' OR 'P') "
                    "AND TWCOB-ID-SISTEMA = 'IP' THEN PERFORM READ-TAB-SEMAF "
                    "IF PXCSEMAF-STATUS = 1 THEN MOVE 'INSERIMENTO/AGGIORNAMENTO NON PERMESSI' "
                    "TO TWCOB-AREA-MSG GO TO XCTL-LIV4"
                ),
                metadata={"chunk_type": "controlflow.cfg"},
            )
        ]

        answer = _try_condition_path_answer(
            "What happens when TWCOB-FUNZIONE is one of I, A, C, D, or P and TWCOB-ID-SISTEMA = 'IP'?",
            sources,
        )

        assert answer is not None
        assert "READ-TAB-SEMAF" in answer
        assert "PXCSEMAF-STATUS = 1" in answer
        assert "XCTL-LIV4" in answer
        assert "password" not in answer.lower()
        assert "account" not in answer.lower()

    def test_row_build_answer_uses_prep_riga_paragraph(self):
        from cobol_rag.query import _try_row_build_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "PREP-RIGA MOVE PD1VOCI-TABVOX-CODVOX(PD1VOCI-IND) TO VOCE "
                    "IF TWCOB-FUNZIONE = 'I' THEN MOVE SPACES TO FUNZ ELSE MOVE PD1VOCI-TABVOX-TIPVAR(PD1VOCI-IND) TO FUNZ "
                    "MOVE PD1VOCI-TABVOX-DESCRIZ(PD1VOCI-IND) TO WDESCVO "
                    "MOVE PD1VOCI-TABVOX-IRATA(PD1VOCI-IND) TO PDRUTI01-F05-VALORE "
                    "MOVE PD1VOCI-TABVOX-PROGVOX(PD1VOCI-IND) TO WPROGREC"
                ),
                metadata={"chunk_type": "paragraph_logic", "paragraph": "PREP-RIGA"},
            )
        ]

        answer = _try_row_build_answer("How does PREP-RIGA build each displayed row?", sources)

        assert answer is not None
        assert "PD1VOCI-TABVOX-CODVOX" in answer
        assert "WDESCVO" in answer
        assert "PDRUTI01-F05-VALORE" in answer
        assert "WPROGREC" in answer

    def test_field_mapping_answer_does_not_dump_copybook_fields(self):
        from cobol_rag.query import _try_field_mapping_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "INIZ-PARAM MOVE TWCOB-VARCONT-VOCE-LIV4 TO PD1VOCI-COD-VOCE "
                    "MOVE TWCOB-SP-MATR TO PD1VOCI-CODDIP-MATR "
                    "MOVE TWCOB-FUNZIONE TO PD1VOCI-TIPO-VARIAZ"
                ),
                metadata={"chunk_type": "paragraph_logic", "paragraph": "INIZ-PARAM"},
            ),
            RetrievalResult(
                score=0.5,
                text="PDRTWA2 copybook fields: TWCOB-PARTE-PRIMA, TWCOB-LLTS, TWCOB-INIZIO",
                metadata={"chunk_type": "copybook_fields"},
            ),
        ]

        answer = _try_field_mapping_answer("Which fields from PDRTWA2 are copied into PD1VOCI before the CICS LINK?", sources)

        assert answer is not None
        assert "does not show a whole-copybook move" in answer
        assert "TWCOB-VARCONT-VOCE-LIV4 -> PD1VOCI-COD-VOCE" in answer
        assert "TWCOB-LLTS" not in answer

    def test_variable_usage_answer_for_funz_uses_dataflow(self):
        from cobol_rag.query import _try_variable_usage_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "Variable dataflow for FUNZ in PDCBVC.CBL. "
                    "paragraph PREP-RIGA, IF_BRANCH: IF TWCOB-FUNZIONE = 'I' "
                    "THEN MOVE SPACES TO FUNZ ELSE MOVE PD1VOCI-TABVOX-TIPVAR(PD1VOCI-IND) TO FUNZ."
                ),
                metadata={"chunk_type": "dataflow.variable", "variable": "FUNZ"},
            )
        ]

        answer = _try_variable_usage_answer("What does variable FUNZ do?", sources)

        assert answer is not None
        assert "PREP-RIGA" in answer
        assert "PD1VOCI-TABVOX-TIPVAR" in answer
        assert "record being processed" not in answer

    def test_copybook_roles_answer_describes_roles(self):
        from cobol_rag.query import _try_copybook_roles_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text="Analysis confidence: medium. Notes: 8 copybook(s) stubbed.",
                metadata={
                    "chunk_type": "program_summary",
                    "copybooks_used": ["DFHAID", "PDCBVCM", "PD1VOCI", "PDRTWA2", "PXCSEMAF"],
                    "mentions": [{"copybook": "DFHAID", "stubbed": True}],
                },
            )
        ]

        answer = _try_copybook_roles_answer("Which copybooks are included and what role does each one play?", sources)

        assert answer is not None
        assert "DFHAID" in answer
        assert "PF key" in answer
        assert "PD1VOCI" in answer
        assert "COMMAREA" in answer

    def test_error_path_answer_keeps_paths_separate_from_message_fields(self):
        from cobol_rag.query import _try_error_path_answer
        from cobol_rag.retrieve import RetrievalResult

        sources = [
            RetrievalResult(
                score=1.0,
                text=(
                    "ABEND00 WABEND-CODE GET1 PXCSEMAF-STATUS = 1 XCTL-LIV4 "
                    "BROWSE-FASE2-NOSEL BROWSE-FASE2-NOTFND M1MSGO M1MSGL SCELTAL"
                ),
                metadata={"chunk_type": "error_path"},
            )
        ]

        answer = _try_error_path_answer(
            "Identify all paths that lead to an error message or abnormal termination, including invalid function keys, missing records, invalid selection, failed service calls, SQL errors, and semaphore restrictions.",
            sources,
        )

        assert answer is not None
        assert "Invalid function key" in answer
        assert "SQL errors" in answer
        assert "M1MSGO" in answer
        assert "not separate control-flow paths" in answer


# ---------------------------------------------------------------------------
# Chat history isolation
# ---------------------------------------------------------------------------

class TestChatHistory:
    def test_chat_passes_history_without_polluting_current_question(self, monkeypatch):
        from cobol_rag import chat as chat_module
        from cobol_rag.chat import ChatSession
        from cobol_rag.query import QueryAnswer

        calls = []

        def fake_answer_query(**kwargs):
            calls.append(kwargs)
            return QueryAnswer(
                question=kwargs["question"],
                answer="ok",
                sources=[],
            )

        monkeypatch.setattr(chat_module, "answer_query", fake_answer_query)
        session = ChatSession(config=None)

        session.ask("Is there any unused code/copy in this program?")
        session.ask("Which dataset/Tables are used by this program?")

        assert calls[0]["question"] == "Is there any unused code/copy in this program?"
        assert calls[0]["conversation_history"] is None
        assert calls[1]["question"] == "Which dataset/Tables are used by this program?"
        assert "unused code/copy" not in calls[1]["question"]
        assert "unused code/copy" in calls[1]["conversation_history"]
