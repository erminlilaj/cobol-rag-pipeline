import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

from cobol_rag.config import AppConfig
from cobol_rag.query import (
    QueryRoutingDecision, _recover_conversational_route,
    _ensure_executable_query_spec,
)


def test_minimal_recovery_retains_model_reply():
    llm = Mock()
    llm.complete.return_value = SimpleNamespace(text='{"route":"conversational","reply":"Hello!"}')
    with patch('cobol_rag.query.build_llm', return_value=llm), patch('cobol_rag.query._conversational_route_is_blocked', return_value=False):
        result = _recover_conversational_route('Good morning', AppConfig(), None, None, None)
    assert result.reply == 'Hello!'
    assert 'query_spec' not in llm.complete.call_args.args[0]


def test_recovery_cannot_bypass_technical_guard():
    llm = Mock()
    llm.complete.return_value = SimpleNamespace(text='{"route":"conversational","reply":"Hello!"}')
    with patch('cobol_rag.query.build_llm', return_value=llm), patch('cobol_rag.query._conversational_route_is_blocked', return_value=True):
        assert _recover_conversational_route('Explain these calls', AppConfig(), None, None, None) is None


def test_invalid_call_type_is_repaired_not_executed():
    spec = dict(operator='list', capability='call_evidence', entity_types=['program'], entity_values=[], fields=['target'], filters=[dict(field='call_type', operator='eq', values=['BROWSING'])])
    fixed = dict(spec, filters=[])
    llm = Mock()
    llm.complete.return_value = SimpleNamespace(text=json.dumps(fixed))
    with patch('cobol_rag.query.build_llm', return_value=llm):
        result = _ensure_executable_query_spec('List programs used for browsing', AppConfig(), QueryRoutingDecision('technical', '', 'external_programs', query_spec=spec), preliminary_plan=None, preliminary_scope=None)
    assert result.query_spec['filters'] == []
    assert 'invalid_call_type_filter' in llm.complete.call_args.args[0]


def test_valid_call_type_does_not_require_repair():
    spec = dict(operator='list', capability='call_evidence', entity_types=['program'], entity_values=[], fields=['target'], filters=[dict(field='call_type', operator='eq', values=['LINK'])])
    with patch('cobol_rag.query.build_llm') as build:
        result = _ensure_executable_query_spec('List LINK calls', AppConfig(), QueryRoutingDecision('technical', '', 'external_programs', query_spec=spec), preliminary_plan=None, preliminary_scope=None)
    assert result.query_spec['filters'][0]['values'] == ['LINK']
    build.assert_not_called()


def test_numeric_control_literals_are_not_substrings():
    from cobol_rag.final_scripts_answers import _condition_contains_literal
    assert _condition_contains_literal('IF WCTPAG = 1', '+1')
    assert _condition_contains_literal('IF VALUE = -1', '-01')
    assert not _condition_contains_literal('IF WCTPAG = 10', '1')
    assert not _condition_contains_literal("IF VALUE = '1'", '+1')
    assert not _condition_contains_literal('IF FIELD-1 = 2', '1')


def test_followup_quantity_is_preserved():
    from cobol_rag.query_plan import parse_response_contract
    for message, count in [('List the first five of them only.', 5), ('Show the first 3 variables', 3), ('Give the first seven of those', 7)]:
        assert parse_response_contract(message).exact_item_count == count


def test_inventory_slices_before_rendering(tmp_path):
    from cobol_rag.final_scripts_answers import answer_inventory
    (tmp_path / 'dataflow.used_variables.json').write_text(json.dumps({'variables':[{'variable':f'V{i}'} for i in range(9)]}))
    with patch('cobol_rag.final_scripts_answers.find_final_scripts_root', return_value=tmp_path), patch('cobol_rag.final_scripts_answers.find_program_artifact_root', return_value=tmp_path):
        answer = answer_inventory('TEST', 'variable', limit=3, offset=2)
    assert [line for line in answer.splitlines() if line.startswith('- ')] == ['- V2','- V3','- V4']
    assert '3 of 9' in answer
