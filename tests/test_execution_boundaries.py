import json
from unittest.mock import patch

import pytest

from cobol_rag.query_ir import EntityMembership, compile_query
from cobol_rag.final_scripts_answers import answer_entity_membership
from cobol_rag.query import _claim_supported_by_sources, _typed_query_answer, RetrievalResult
from types import SimpleNamespace


@pytest.mark.parametrize("question", [
    "Which of AAA and BBB calls SHARED?",
    "Does BBB invoke SHARED, or does AAA?",
    "Find callers of SHARED among AAA and BBB.",
])
def test_call_role_wins_over_same_named_copybook(question):
    query = compile_query(question, program="AAA", programs=("AAA", "BBB"),
                          calls=("SHARED",), copybooks=("SHARED",),
                          capability="call_evidence")
    assert isinstance(query, EntityMembership)
    assert query.entity_type == "call"
    assert query.entity == "SHARED"


def test_copybook_membership_stays_copybook():
    query = compile_query("Which of AAA and BBB includes SHARED?", program="AAA",
                          programs=("AAA", "BBB"), calls=("SHARED",),
                          copybooks=("SHARED",), capability="copybook_evidence")
    assert query.entity_type == "copybook"


def test_unresolved_operand_cannot_become_inventory():
    assert compile_query("Does MISSING-FIELD exist in AAA or BBB?", program="AAA",
                         programs=("AAA", "BBB"), entity_type="variable",
                         unresolved_entities=("MISSING-FIELD",)) is None


@pytest.mark.parametrize("payload, expected", [
    ({"calls": []}, "not present"),
    ({}, "membership unknown"),
    ({"calls": None}, "membership unknown"),
    (None, "membership unknown"),
    ({"calls": [{"target": "SHARED", "paragraph": "MAIN", "line_start": 12,
                 "call_type": "CALL"}]}, "present; CALL in MAIN line 12"),
])
def test_call_membership_uses_calls_not_copybooks(tmp_path, payload, expected):
    (tmp_path / "architecture.copybooks.json").write_text(json.dumps({"content": {"all": ["SHARED"]}}))
    if payload is not None:
        (tmp_path / "architecture.call_parameters.json").write_text(json.dumps(payload))
    with patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=tmp_path), \
         patch("cobol_rag.final_scripts_answers.find_program_artifact_root", return_value=tmp_path):
        result = answer_entity_membership(("AAA",), "SHARED", "call")
    assert "AAA:" in result
    assert expected in result
    assert "Source: `architecture.call_parameters.json`" in result


def test_legacy_variable_list_inventory(tmp_path):
    (tmp_path / "dataflow.used_variables.json").write_text(json.dumps([{"variable": "FIELD-A"}]))
    with patch("cobol_rag.final_scripts_answers.find_final_scripts_root", return_value=tmp_path), \
         patch("cobol_rag.final_scripts_answers.find_program_artifact_root", return_value=tmp_path):
        result = answer_entity_membership(("AAA",), "FIELD-A", "variable")
    assert "AAA: present" in result


def test_coordinated_claim_requires_each_member_to_have_evidence():
    sources = [RetrievalResult(1.0, f"Program DEMO. Paragraph {name}. EXEC CICS SYNCPOINT END-EXEC.", {})
               for name in ("FIRST-PARA", "SECOND-PARA")]
    claim = "The DEMO paragraphs that contain a CICS SYNCPOINT are FIRST-PARA and SECOND-PARA."
    assert _claim_supported_by_sources(claim, sources, exact_code=False)
    assert not _claim_supported_by_sources(claim, sources[:1], exact_code=False)
    assert not _claim_supported_by_sources(claim.replace("SECOND-PARA", "OTHER-PARA"), sources, exact_code=False)


def test_list_validation_does_not_join_unrelated_names():
    sources = [RetrievalResult(1.0, "Program DEMO executes CICS SYNCPOINT.", {}),
               RetrievalResult(1.0, "Paragraph FIRST-PARA. MOVE A TO B.", {}),
               RetrievalResult(1.0, "Paragraph SECOND-PARA. MOVE B TO C.", {})]
    assert not _claim_supported_by_sources(
        "The DEMO paragraphs that contain a CICS SYNCPOINT are FIRST-PARA and SECOND-PARA.",
        sources, exact_code=False)


def test_typed_execution_preserves_planned_capability_without_reranking():
    plan = SimpleNamespace(subtasks=[SimpleNamespace(capability="call_evidence", required=True)])
    context = dict(program="AAA", programs=("AAA", "BBB"), calls=("SHARED",), copybooks=("SHARED",))
    with patch("cobol_rag.query._typed_query_context", return_value=context), \
         patch("cobol_rag.query._routed_capability") as rank, \
         patch("cobol_rag.query._execute_typed_query", return_value="answer") as execute:
        assert _typed_query_answer(plan, "Which of AAA and BBB calls SHARED?") == "answer"
    rank.assert_not_called()
    assert execute.call_args.args[1].entity_type == "call"
