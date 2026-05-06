from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from cobol_rag.config import AppConfig
from cobol_rag.config import load_config
from cobol_rag.loaders.generic_json import GenericJsonLoader
from cobol_rag.loaders.rag_documents import RagDocumentsLoader
from cobol_rag.query import (
    answer_query,
    _try_business_rules_answer,
    _try_conflict_provenance_answer,
    _try_sql_includes_answer,
    _try_ui_navigation_answer,
    _try_variable_dataflow_answer,
)
from cobol_rag.retrieve import (
    RetrievalResult,
    _base_chunk_type,
    _exact_identifier_score,
    _expand_entity_companions,
    _record_identity,
)


def _source(text: str, **metadata) -> RetrievalResult:
    return RetrievalResult(score=1.0, text=text, metadata=metadata)


def test_default_config_loads_chunk_type_boost_path():
    config = load_config()

    assert config.retrieval.chunk_type_boosts_path == "config/chunk_type_boosts.yaml"


def test_rag_documents_loader_preserves_source_aware_metadata(tmp_path: Path):
    path = tmp_path / "rag_documents.jsonl"
    record = {
        "id": "doc-1",
        "program": "PDCBVC",
        "type": "architecture.call",
        "title": "PDCBVC call PD1VOCI",
        "text": "PDCBVC calls PD1VOCI.",
        "metadata": {
            "source_system": "mapa_hamza",
            "source_chunk_type": "architecture.call",
            "coverage_dimension": "static_inventory",
            "entity_type": "call",
            "entity_key": "PDCBVC|PD1VOCI|LINK",
            "target": "PD1VOCI",
            "call_type": "LINK",
            "source_id": "source-provided",
            "content_hash": "source-hash",
            "source_bundle_path": "cobol-rekt/knowledge-base_rag/PDCBVC",
            "original_chunk_id": "rekt-42",
            "sha256": "abc123",
            "left_source_system": "mapa_hamza",
            "right_source_system": "cobol_rekt",
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    docs = RagDocumentsLoader(AppConfig()).load(path)

    assert len(docs) == 1
    meta = docs[0].document.metadata
    assert meta["chunk_type"] == "architecture.call"
    assert meta["chunk_id"] == "doc-1"
    assert meta["source_system"] == "mapa_hamza"
    assert meta["source_chunk_type"] == "architecture.call"
    assert meta["coverage_dimension"] == "static_inventory"
    assert meta["entity_key"] == "PDCBVC|PD1VOCI|LINK"
    assert meta["target"] == "PD1VOCI"
    assert meta["call_type"] == "LINK"
    assert meta["source_bundle_path"] == "cobol-rekt/knowledge-base_rag/PDCBVC"
    assert meta["original_chunk_id"] == "rekt-42"
    assert meta["sha256"] == "abc123"
    assert meta["left_source_system"] == "mapa_hamza"
    assert meta["right_source_system"] == "cobol_rekt"
    assert meta["factory_source_id"] == "source-provided"
    assert meta["content_hash"] != "source-hash"


def test_generic_json_loader_accepts_jsonl_and_promotes_type_id(tmp_path: Path):
    path = tmp_path / "generic.jsonl"
    path.write_text(
        json.dumps({"id": "x1", "type": "business_rule", "text": "BR text", "program": "PDCBVC"}) + "\n",
        encoding="utf-8",
    )

    docs = GenericJsonLoader(AppConfig()).load(path)

    assert len(docs) == 1
    assert docs[0].document.metadata["chunk_type"] == "business_rule"
    assert docs[0].document.metadata["chunk_id"] == "x1"


def test_namespaced_chunk_type_routes_by_source_chunk_type():
    result = _source(
        "call contract",
        chunk_type="cobol_rekt.call_contract",
        source_chunk_type="call_contract",
    )

    assert _base_chunk_type(result) == "call_contract"


def test_exact_identifier_score_checks_entity_metadata():
    result = _source(
        "inventory",
        chunk_type="architecture.call",
        entity_key="PDCBVC|PD1VOCI|LINK",
        target="PD1VOCI",
    )

    assert _exact_identifier_score("Which COMMAREA is passed to PD1VOCI?", result) > 0


def test_entity_expansion_fetches_companion_chunks(monkeypatch):
    base = _source(
        "MAPA call inventory",
        source_id="mapa-1",
        source_system="mapa_hamza",
        chunk_type="architecture.call",
        entity_key="PDCBVC|PD1VOCI|LINK",
    )

    class FakeCollection:
        def get(self, **_kwargs):
            return {
                "documents": ["cobol-rekt call contract"],
                "metadatas": [
                    {
                        "source_id": "rekt-1",
                        "source_system": "cobol_rekt",
                        "chunk_type": "cobol_rekt.call_contract",
                        "source_chunk_type": "call_contract",
                        "coverage_dimension": "deep_logic",
                        "entity_key": "PDCBVC|PD1VOCI|LINK",
                    }
                ],
            }

    monkeypatch.setattr(
        "cobol_rag.retrieve.open_index",
        lambda _config: SimpleNamespace(chroma_collection=FakeCollection()),
    )

    expanded = _expand_entity_companions([base], AppConfig())

    assert [item.metadata["source_id"] for item in expanded] == ["mapa-1", "rekt-1"]


def test_entity_expansion_keeps_companions_without_source_id(monkeypatch):
    base = _source(
        "MAPA call inventory",
        source_system="mapa_hamza",
        chunk_type="architecture.call",
        entity_key="PDCBVC|PD1VOCI|LINK",
    )

    class FakeCollection:
        def get(self, **_kwargs):
            return {
                "documents": ["cobol-rekt call contract"],
                "metadatas": [
                    {
                        "source_system": "cobol_rekt",
                        "chunk_type": "cobol_rekt.call_contract",
                        "source_chunk_type": "call_contract",
                        "coverage_dimension": "deep_logic",
                        "entity_key": "PDCBVC|PD1VOCI|LINK",
                    }
                ],
            }

    monkeypatch.setattr(
        "cobol_rag.retrieve.open_index",
        lambda _config: SimpleNamespace(chroma_collection=FakeCollection()),
    )

    expanded = _expand_entity_companions([base], AppConfig())

    assert len(expanded) == 2
    assert expanded[1].metadata["source_system"] == "cobol_rekt"
    assert _record_identity(expanded[0]) != _record_identity(expanded[1])


def test_direct_answer_handlers_emit_provenance():
    business = _try_business_rules_answer(
        "What business rules apply?",
        [_source("content.condition: X\ncontent.action: JUMP", chunk_type="business_rule", source_system="mapa_hamza")],
    )
    ui = _try_ui_navigation_answer(
        "Which PF keys are handled?",
        [_source("PF7 -> BROWSE-FASE2-PF7", chunk_type="ui.cics.navigation", source_system="mapa_hamza")],
    )
    dataflow = _try_variable_dataflow_answer(
        "Where is TWCOB-FASE read or written?",
        [_source("variable TWCOB-FASE read_sites line 100", chunk_type="dataflow.variable", source_system="mapa_hamza")],
    )
    sql = _try_sql_includes_answer(
        "Which SQL includes are used?",
        [_source("SQLCA\nPDPSQLER", chunk_type="architecture.sqlinclude", source_system="mapa_hamza")],
    )
    conflict = _try_conflict_provenance_answer(
        "Can I trust this count?",
        [_source("do not collapse different counting methods", chunk_type="integration.conflicts", source_system="integration", coverage_dimension="conflict_report")],
    )

    for answer in (business, ui, dataflow, sql, conflict):
        assert answer is not None
        assert "Sources used:" in answer


def test_answer_query_uses_rag_before_final_scripts(monkeypatch):
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: "structured fallback")
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)
    monkeypatch.setattr(
        "cobol_rag.query.retrieve",
        lambda *_args, **_kwargs: [
            _source(
                "content.condition: X\ncontent.action: JUMP",
                chunk_type="business_rule",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ],
    )

    answer = answer_query("What business rules apply to PDCBVC?", AppConfig())

    assert "Business-rule evidence found" in answer.answer
    assert "structured fallback" not in answer.answer
    assert answer.sources


def test_answer_query_does_not_shortcut_final_scripts_when_rag_is_grounded(monkeypatch):
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: "structured shortcut")
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)
    monkeypatch.setattr(
        "cobol_rag.query.retrieve",
        lambda *_args, **_kwargs: [
            _source(
                "Program: PDCBVC\nType: program.summary\nPDCBVC overview from indexed RAG evidence.",
                chunk_type="program.summary",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ],
    )

    class FakeLlm:
        def complete(self, prompt):
            assert "structured shortcut" not in prompt
            return SimpleNamespace(text="RAG-generated answer from retrieved evidence")

    monkeypatch.setattr(
        "cobol_rag.query.open_index",
        lambda _config: SimpleNamespace(runtime=SimpleNamespace(llm=FakeLlm())),
    )

    answer = answer_query("What is PDCBVC?", AppConfig())

    assert answer.answer == "RAG-generated answer from retrieved evidence"
    assert "structured shortcut" not in answer.answer
    assert answer.sources


def test_answer_query_rejects_ungrounded_named_target(monkeypatch):
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)
    monkeypatch.setattr(
        "cobol_rag.query.retrieve",
        lambda *_args, **_kwargs: [
            _source(
                "DFHPF1 -> XCTL-LIV1\nDFHPF2 -> XCTL-LIV2",
                chunk_type="ui.cics.navigation",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ],
    )

    answer = answer_query("Which paths can lead to rome?", AppConfig())

    assert "do not have indexed evidence for `ROME`" in answer.answer
    assert "similar-looking control-flow" in answer.answer


def test_answer_query_retrieves_with_current_question_only(monkeypatch):
    seen: dict[str, str] = {}
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)

    def fake_retrieve(question, *_args, **_kwargs):
        seen["question"] = question
        return [
            _source(
                "content.condition: X\ncontent.action: JUMP",
                chunk_type="business_rule",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ]

    monkeypatch.setattr("cobol_rag.query.retrieve", fake_retrieve)

    answer = answer_query(
        "Use this conversation history only to resolve follow-up references.\n"
        "Conversation history:\n"
        "User: Which paths can lead to ABEND00?\n"
        "Assistant: ABEND00 answer\n"
        "Current question:\n"
        "What business rules apply to PDCBVC?",
        AppConfig(),
    )

    assert seen["question"] == "What business rules apply to PDCBVC?"
    assert "ABEND00" not in seen["question"]
    assert "Business-rule evidence found" in answer.answer


def test_answer_query_falls_back_when_rag_unavailable(monkeypatch):
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: "structured fallback")
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)

    def raise_retrieval(*_args, **_kwargs):
        raise RuntimeError("embedding unavailable")

    monkeypatch.setattr("cobol_rag.query.retrieve", raise_retrieval)

    answer = answer_query("Which copybooks are used by PDCBVC?", AppConfig())

    assert answer.answer == "structured fallback"
    assert answer.sources == []
