from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import pytest

from cobol_rag.activity_context import MODEL_PURPOSES, model_context, tool_context, file_purpose, source_preview
from cobol_rag import progress


@pytest.mark.parametrize('phase', list(MODEL_PURPOSES))
def test_public_model_purpose_is_stage_specific_not_model_reasoning(phase):
    payload = {'previous_candidate': 'PRIVATE CANDIDATE', 'original_question': 'PRIVATE QUESTION',
        'feedback': ['PRIVATE REVIEW TEXT'], 'observations': [
            {'call': {'args': {'program': 'OWNER'}}, 'result': {'rows': [
                {'program': 'OWNER', 'source_file': 'OWNER.CBL', 'evidence_id': 'E1', 'text': 'PRIVATE SOURCE'}]}}]}
    before = deepcopy(payload)
    result = model_context(phase, payload)
    assert result['purpose'] == MODEL_PURPOSES[phase]
    assert result['context_files'] == ['OWNER: OWNER.CBL']
    assert result['context_rows'] == 1
    assert result['programs'] == ['OWNER']
    assert 'PRIVATE' not in str(result)
    assert payload == before


def test_model_context_is_bounded_and_covers_retained_and_review_evidence():
    rows = [{'program': f'P{i}', 'source_file': f'P{i}.CBL'} for i in range(50)]
    result = model_context('review', {'retained_source_evidence': {'rows': rows},
                                      'evidence': {'rows': rows}})
    assert len(result['programs']) == 12 and len(result['context_files']) == 12
    assert result['context_rows'] == 100


def test_zero_and_truncated_results_explain_scope_without_absence_claims():
    args = {'program': 'OWNER', 'terms': ['ITEM']}
    zero = tool_context('source_outline', args, {'returned': 0})
    assert 'does not prove' in zero['outcome']
    found = tool_context('source_outline', args, {'returned': 1, 'rows': [
        {'name': 'RULE', 'first_line': 20, 'last_line': 40}]})
    assert found['matches'] == [{'paragraph': 'RULE', 'start': 20, 'end': 40}]
    assert 'navigation only' in found['next_hint']
    incomplete = tool_context('source', {'program': 'OWNER', 'paragraph': 'RULE'},
                              {'returned': 240, 'requested_complete': False})
    assert 'not returned completely' in incomplete['next_hint']
    assert 'not proof' in tool_context('source', {'program': 'OWNER'}, {'returned': 0})['outcome']


def test_source_previews_prefer_real_conditions_and_assignments_over_blank_headers():
    rows = [{'source_file': 'A.CBL', 'line': i, 'text': text, 'is_comment': comment}
            for i, text, comment in [(1, 'RULE.', False), (2, '', False),
                (3, '* MOVE OLD', True), (4, "IF CODE = '1234'", False),
                (5, 'MOVE 100 TO AMOUNT', False), (6, 'ELSE', False),
                (7, 'MOVE 200 TO AMOUNT', False)]]
    assert [row['line'] for row in source_preview(rows)] == [4, 5, 7]
    assert 'indexed' in file_purpose(Path('program.source_lines.jsonl'), 'source')
    assert 'control-flow context' in file_purpose(Path('controlflow.cfg.json'), 'graph')


def test_real_budget_emits_safe_stage_context_and_draft_not_verified_answer(monkeypatch):
    from cobol_rag.investigation import Budget
    from cobol_rag.investigation_protocol import decision_schema
    from cobol_rag.config import AppConfig
    response = dict(action='final', mode='technical', status='partial', answer='PRIVATE DRAFT', evidence_ids=['E1'])
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=lambda *a, **kw:
        SimpleNamespace(chat=lambda *a, **kw: SimpleNamespace(message=SimpleNamespace(content=json.dumps(response))))))
    with progress.track('activity-context', 'budget-test'):
        result = Budget(AppConfig()).call('PRIVATE SYSTEM', {'previous_candidate': 'PRIVATE PREVIOUS'},
                                          decision_schema(), phase='answer_repair')
    events = progress.snapshot('activity-context', 'budget-test')['events']
    start = next(event['details'] for event in events if event['kind'] == 'model_start')
    assert 'Revise an answer' in start['purpose']
    assert start['context_chars'] > 0 and start['max_context_tokens'] > 0
    assert start['remaining_calls'] >= 0
    assert result['answer'] == 'PRIVATE DRAFT'
    assert any(event['kind'] == 'draft_ready' and 'still needs' in event['message'] for event in events)
    assert 'PRIVATE' not in str(events)


@pytest.mark.parametrize('reply', [{'passed': False, 'issues': 7},
    {'action': 'tools', 'calls': None}, {'action': 'final', 'evidence_ids': 3}])
def test_activity_does_not_crash_before_malformed_review_can_be_repaired(monkeypatch, reply):
    from cobol_rag.investigation import Budget
    from cobol_rag.config import AppConfig
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=lambda *a, **kw:
        SimpleNamespace(chat=lambda *a, **kw: SimpleNamespace(message=SimpleNamespace(content=json.dumps(reply))))))
    assert Budget(AppConfig()).call('Review', {}, phase='review') == reply
