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
    _entity_present_in_text,
    _extract_grounding_facts,
    _grounded_fallback_answer,
    _looks_off_evidence_answer,
    _question_entities,
    _try_business_rules_answer,
    _try_conflict_provenance_answer,
    _try_sql_includes_answer,
    _try_ui_navigation_answer,
    _try_variable_dataflow_answer,
)
from cobol_rag.retrieve import (
    RetrievalResult,
    _base_chunk_type,
    _detect_intent,
    _exact_identifier_score,
    _expand_entity_companions,
    _expanded_query_for_intent,
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


def test_retrieval_intent_detects_program_overview_variants():
    assert _detect_intent("what pdcbvc?") == "program_summary"
    assert _detect_intent("what the hell is PDCBVC?") == "program_summary"
    assert _detect_intent("what is file PDCBVC about?") == "program_summary"
    assert _detect_intent("what is the logic inside the file PDCBVC?") == "control_flow"


def test_retrieval_intent_detects_pf_number_questions():
    assert _detect_intent("Compare PF1, PF2, PF3, PF4, PF7, PF8, and PF9 behavior.") == "ui_navigation"
    expanded = _expanded_query_for_intent("Compare PF1, PF2, PF3, PF4, PF7, PF8, and PF9 behavior.", "ui_navigation")

    assert "DFHPF1" in expanded
    assert "screen.key_dispatch" in expanded


def test_question_entities_ignore_action_words():
    assert "PRESSES" not in _question_entities("What does BROWSE-FASE2 do when the user presses ENTER?")
    assert "BROWSE-FASE2" in _question_entities("What does BROWSE-FASE2 do when the user presses ENTER?")


def test_entity_grounding_accepts_paragraph_prefix_evidence():
    haystack = "DFHENTER routes to BROWSE-FASE2-ENTER.".upper()

    assert _entity_present_in_text("BROWSE-FASE2", haystack)


def test_fact_extraction_promotes_privileged_evidence():
    facts = _extract_grounding_facts(
        [
            _source(
                "Privileged structured evidence from final_scripts.\n"
                "PDCBVC calls PD1FS00 in PREP-LINK-PD1FS00.\n"
                "PD1FS00-SESS-FLAG '1' means Aperta.",
                chunk_type="privileged.final_scripts",
                source_system="mapa_hamza",
                program="PDCBVC",
                source_id="final_scripts:test",
            )
        ]
    )

    assert "Privileged structured evidence" in facts
    assert "PD1FS00-SESS-FLAG '1' means Aperta" in facts


def test_off_evidence_validator_rejects_external_generic_answer():
    sources = [
        _source(
            "Privileged structured evidence from final_scripts.\nPDCBVC has 15 commented-out code/data items.",
            chunk_type="privileged.final_scripts",
            source_system="mapa_hamza",
            program="PDCBVC",
        )
    ]

    assert _looks_off_evidence_answer(
        "give me some examples of dead code",
        "See [Source 1](https://github.com/oss-specs/specs/blob/master/pdf/PDCBVC.pdf).",
        sources,
    )
    assert _looks_off_evidence_answer(
        "is there unused copybooks in PDCBVC?",
        "Run git checkout -b new-branch-name to create a branch.",
        sources,
    )
    assert _looks_off_evidence_answer(
        "is there unused copybooks in PDCBVC?",
        "PDCBVC is a program that is used to perform a specific task. It is not clear what this program does.",
        sources,
    )
    assert _looks_off_evidence_answer(
        "Which paths can lead to ABEND00?",
        "Recommendations: Centralize Error Handling and Review Page Logic to enhance reliability.",
        sources,
    )
    assert _looks_off_evidence_answer(
        "When does PDCBVC call PD1FS00?",
        "Step-by-step reasoning: Add after the CICS LINK an evaluation of PD1FS00-RETURN.",
        sources,
    )


def test_grounded_fallback_uses_evidence_not_generic_text():
    sources = [
        _source(
            "Privileged structured evidence from final_scripts.\n"
            "PDCBVC dead-code evidence: commented-out code/data: 15 item(s).\n"
            "line 75: 03 LUNG PIC S9(4) COMP VALUE +32000.",
            chunk_type="privileged.final_scripts",
            source_system="mapa_hamza",
            program="PDCBVC",
            source_id="final_scripts:dead",
        )
    ]

    answer = _grounded_fallback_answer("give me some examples of dead code", sources)

    assert "commented-out code/data: 15" in answer
    assert "line 75" in answer
    assert "github.com" not in answer


def test_control_flow_expansion_targets_pagination_and_selection():
    pagination = _expanded_query_for_intent(
        "How does PDCBVC calculate the total number of pages for the browse result?",
        "control_flow",
    )
    selection = _expanded_query_for_intent(
        "When the user selects a row, how does PDCBVC validate the selected progressivo?",
        "control_flow",
    )

    assert "CALCOLA-NPAG" in pagination
    assert "PD1VOCI-TABVOX-NUMERO" in pagination
    assert "BROWSE-FASE2-SEL" in selection
    assert "SCELTAI" in selection


def test_answer_query_abstains_when_required_pagination_facts_are_missing(monkeypatch):
    monkeypatch.setattr("cobol_rag.query.answer_from_final_scripts", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query.preflight_entity_answer", lambda _question: None)
    monkeypatch.setattr("cobol_rag.query._rag_runtime_available", lambda _base_url: True)
    monkeypatch.setattr(
        "cobol_rag.query.retrieve",
        lambda *_args, **_kwargs: [
            _source(
                "Program: PDCBVC\nBROWSE-FASE2-PF8 handles a key.",
                chunk_type="screen.key_dispatch",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ],
    )

    answer = answer_query("How does PDCBVC calculate the total number of pages for the browse result?", AppConfig())

    assert "do not have enough indexed evidence" in answer.answer
    assert "CALCOLA-NPAG" in answer.answer


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
                "Program: PDCBVC\ncontent.condition: X\ncontent.action: JUMP",
                chunk_type="business_rule",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ],
    )
    monkeypatch.setattr(
        "cobol_rag.query.open_index",
        lambda _config: SimpleNamespace(
            runtime=SimpleNamespace(llm=SimpleNamespace(complete=lambda _prompt: SimpleNamespace(text="RAG business-rule answer")))
        ),
    )

    answer = answer_query("What business rules apply to PDCBVC?", AppConfig())

    assert answer.answer == "RAG business-rule answer"
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
            assert "structured shortcut" in prompt
            assert "Use this as evidence, not as a prewritten answer." in prompt
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
                "Program: PDCBVC\ncontent.condition: X\ncontent.action: JUMP",
                chunk_type="business_rule",
                source_system="mapa_hamza",
                program="PDCBVC",
            )
        ]

    monkeypatch.setattr("cobol_rag.query.retrieve", fake_retrieve)
    monkeypatch.setattr(
        "cobol_rag.query.open_index",
        lambda _config: SimpleNamespace(
            runtime=SimpleNamespace(llm=SimpleNamespace(complete=lambda _prompt: SimpleNamespace(text="RAG business-rule answer")))
        ),
    )

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
    assert answer.answer == "RAG business-rule answer"


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
