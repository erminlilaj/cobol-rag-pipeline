"""Regression tests for navigation/protocol/recovery, independent of business codes."""
from copy import deepcopy
from types import SimpleNamespace
import json
import sys

import pytest
from jsonschema import validate, ValidationError

from test_investigation import evidence, FakeBudget
from cobol_rag.config import AppConfig
from cobol_rag.investigation import (Budget, normalize_decision, investigate, answer_checks,
                                     synthesize_partial)
from cobol_rag.investigation_protocol import (REQUEST, TOOL_SCHEMAS, normalize_optional_nulls,
                                              decision_schema, tool_help)
from cobol_rag.investigation_recovery import (evidence_progress, recovery_source_evidence,
                                             outline_followups)
from cobol_rag.investigation_tools import EvidenceTools, digest


@pytest.mark.parametrize('question', ['Explain currency conversion in A',
    'Quali controlli esegue A?', 'Which files does A write?', 'Trace the amount through A'])
def test_irrelevant_empty_relation_does_not_break_resolution(question):
    request = dict(resolved_question=question, output='other', limit=None, call_relations=[],
                   jcl_scope=None, analysis_checks=[])
    fixed = normalize_optional_nulls(request, REQUEST)
    validate(fixed, REQUEST)
    assert 'call_relations' not in fixed and 'jcl_scope' not in fixed
    assert fixed['analysis_checks'] == [] and request['call_relations'] == []
    with pytest.raises(ValidationError):
        validate(normalize_optional_nulls({**request, 'call_relations': [{}]}, REQUEST), REQUEST)


@pytest.mark.parametrize('raw,expected', [('ITEM-A, ITEM-B', ['ITEM-A', 'ITEM-B']),
    ('9876, 5432', ['9876', '5432']), ('payment adjustment', ['payment adjustment']),
    ("'A,B'", ["'A,B'"]), ('MOVE A, B TO C', ['MOVE A, B TO C'])])
def test_outline_term_representation_repair(raw, expected):
    original = dict(action='tools', calls=[dict(tool='source_outline', args=dict(program='A', terms=raw))])
    fixed = normalize_decision(original, None)
    assert fixed['calls'][0]['args']['terms'] == expected
    validate(fixed, decision_schema())
    assert original['calls'][0]['args']['terms'] == raw


def test_help_exposes_types_and_depth_enum_without_guessing_numeric_mapping():
    help_text = tool_help()
    assert 'terms?:string[]' in help_text and 'depth?:brief|standard|detailed' in help_text
    bad = normalize_decision(dict(action='tools', calls=[dict(tool='describe',
        args=dict(identifier='A', depth=2))]), None)
    with pytest.raises(ValidationError):
        validate(bad, decision_schema())


@pytest.mark.parametrize('citation,expected', [('[E1-E3]', ['E1', 'E2', 'E3']),
    ('[E1, E3]', ['E1', 'E3']), ('[E1-E9]', ['E1', 'E2', 'E3'])])
def test_known_citation_groups_reconciled_without_approving_unknown_ids(evidence, citation, expected):
    evidence.execute('query', dict(programs=['A'], table='variables'))
    fixed = normalize_decision(dict(action='final', mode='technical', status='partial',
        answer='Recorded evidence ' + citation, evidence_ids=[]), evidence)
    assert fixed['evidence_ids'] == expected
    checks = answer_checks('Explain the evidence', fixed, evidence, [])
    assert ('unlisted_citation' in checks) == ('E9' in citation)


def rows(program, table):
    if table != 'source':
        return []
    return [dict(program=program, source_file=program + '.CBL', line=index,
                 paragraph='RULE', division='PROCEDURE DIVISION', text=text, normalized=text,
                 _artifact='program.source_lines.jsonl')
            for index, text in enumerate(['RULE.', "IF ITEM-CODE = '9876'",
                "AND YEAR-NO = '2020'", 'MOVE 12345 TO AMOUNT', 'ELSE',
                'MOVE 67890 TO AMOUNT', 'END-IF.'], 10)]


def test_recovery_reads_actual_body_and_keeps_limits(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    evidence.execute('source_outline', dict(program='A', terms=['9876']))
    evidence.execute('source_outline', dict(program='B', terms=['9876']))
    calls = outline_followups(evidence, set())
    assert {call['args']['program'] for call in calls} == {'A', 'B'}
    for call in calls:
        assert call['args']['end'] - call['args']['start'] < 120
        evidence.execute(call['tool'], call['args'])
    assert not outline_followups(evidence, set())
    snapshot = recovery_source_evidence(evidence)
    assert {row['program'] for row in snapshot['rows']} == {'A', 'B'}
    assert any('MOVE 12345' in row['text'] for row in snapshot['rows'])
    assert any('MOVE 67890' in row['text'] for row in snapshot['rows'])
    assert all(row['evidence_id'] in evidence.evidence for row in snapshot['rows'])


def test_evidence_progress_does_not_reward_changed_outline_terms_or_page_handles(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    evidence.execute('source_outline', dict(program='A', terms=['9876']))
    before = evidence_progress(evidence)
    evidence.execute('source_outline', dict(program='A', terms=['AMOUNT'], limit=99))
    assert evidence_progress(evidence) == before
    evidence.execute('source_range', dict(program='A', start=10, end=16))
    assert evidence_progress(evidence) != before


def test_collection_pages_count_only_new_facts(evidence):
    evidence.execute('query', dict(programs=['A'], table='variables', limit=1))
    first = evidence_progress(evidence)
    evidence.execute('query', dict(programs=['A'], table='variables', limit=2))
    second = evidence_progress(evidence)
    assert first != second  # V2 is a new fact.
    evidence.execute('query', dict(programs=['A'], table='variables', limit=100))
    assert evidence_progress(evidence) == second


@pytest.mark.parametrize('depth,output,blocked', [('detailed', 'other', True),
    ('brief', 'list', False), ('standard', 'count', False)])
def test_navigation_is_not_a_detailed_behavior_answer(evidence, monkeypatch, depth, output, blocked):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    result = evidence.execute('source_outline', dict(program='A', terms=['9876']))
    candidate = dict(mode='technical', status='partial', answer='RULE contains the item reference.',
                     evidence_ids=[result['rows'][0]['evidence_id']])
    errors = answer_checks('Explain the behavior', candidate, evidence, [],
                          dict(depth=depth, output=output, limit=None))
    assert any(error.startswith('Source outlines are navigation only') for error in errors) == blocked


def test_stalled_outline_repetition_opens_body_before_recovery_synthesis(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    call = dict(tool='source_outline', args=dict(program='A', terms=['9876']))
    class ReadBudget(FakeBudget):
        maximum = 12
        def call(self, system, payload, *args, **kwargs):
            if self.calls == 3:
                source = payload['observations'][-1]['result']['rows']
                assert any('MOVE 12345' in row['text'] for row in source)
                self.calls += 1
                return dict(action='final', mode='technical', status='partial',
                    answer='For item 9876, 2020 selects 12345; otherwise 67890. Caller conditions remain unverified.',
                    evidence_ids=[row['evidence_id'] for row in source])
            return super().call(system, payload, *args, **kwargs)
    budget = ReadBudget([dict(action='tools', calls=[deepcopy(call)]) for _ in range(3)] +
                        [dict(passed=True, issues=[])])
    result = investigate('Explain the item rule in A', AppConfig(), budget=budget,
        resolved_request=dict(resolved_question='Explain the rule', output='other', depth='detailed', limit=None))
    assert result['status'] == 'partial', result['trace']
    assert any(step.get('recovery') == 'unread_outline' for step in result['trace'])


def test_partial_synthesis_retains_source_after_empty_observations_and_checks_review(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    source = evidence.execute('source_range', dict(program='A', start=10, end=16))
    class PartialBudget(FakeBudget):
        config = AppConfig()
        def call(self, system, payload, *args, **kwargs):
            if self.calls == 0:
                assert 'EACH program' in system
                assert any('MOVE 12345' in row['text'] for row in payload['retained_source_evidence']['rows'])
            return super().call(system, payload, *args, **kwargs)
    answer = dict(action='final', mode='technical', status='partial',
        answer='A assigns 12345 for 2020 and 67890 otherwise; B remains unverified. [E2-E7]',
        evidence_ids=[source['rows'][0]['evidence_id']])
    budget = PartialBudget([answer, dict(passed=True, issues=[])])
    result = synthesize_partial('Explain the rules in A and B', dict(output='other'), [],
        evidence, [{'result': {'rows': []}}] * 3, [], budget, [])
    assert result and result['status'] == 'partial'
    budget = PartialBudget([answer, dict(passed=False, issues=['Contradicts the enclosing condition'])])
    assert synthesize_partial('Explain A and B', dict(output='other'), [], evidence, [], [], budget, []) is None


def test_actual_budget_repairs_terms_and_empty_optional_metadata(monkeypatch):
    values = iter([dict(resolved_question='Explain A', output='other', limit=None, call_relations=[]),
                   dict(action='tools', calls=[dict(tool='source_outline', args=dict(program='A', terms='9876, 5432'))])])
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    def build(*args, **kwargs):
        def chat(*args, **kwargs):
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(values))))
        return SimpleNamespace(chat=chat)
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=build))
    budget = Budget(AppConfig())
    assert 'call_relations' not in budget.call('Resolve', {}, REQUEST)
    decision = budget.call('Investigate', {}, decision_schema())
    assert decision['calls'][0]['args']['terms'] == ['9876', '5432']
    assert budget.calls == 2


def test_real_protocol_boundary_recovers_bad_depth_and_delivers_supported_program(evidence, monkeypatch):
    """A new business code, two owners, a bad tool argument, real executor + review."""
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t) if p == 'A' else [])
    original_execute = EvidenceTools.execute
    source_ids = []
    def execute(self, tool, args):
        result = original_execute(self, tool, args)
        if tool == 'source' and args.get('program') == 'A':
            source_ids[:] = [row['evidence_id'] for row in result['rows']]
        return result
    monkeypatch.setattr(EvidenceTools, 'execute', execute)
    count = 0
    prompts = []
    def build(*args, **kwargs):
        def chat(messages, **kwargs):
            nonlocal count
            count += 1
            prompts.append(messages)
            response = {
                1: dict(action='tools', calls=[dict(tool='source_outline', args=dict(program='A', terms='9876, 5432'))]),
                2: dict(action='tools', calls=[dict(tool='describe', args=dict(identifier='B', programs=['B'], depth=2))]),
                3: dict(action='tools', calls=[dict(tool='source_range', args=dict(program='A', start=10, end=16))]),
                4: dict(action='final', mode='technical', status='partial',
                    answer='A selects 12345 for item 9876 in 2020 and 67890 otherwise. B remains unverified.',
                    evidence_ids=list(source_ids)),
                5: dict(passed=True, issues=[]),
            }[count]
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(response)))
        return SimpleNamespace(chat=chat)
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=build))
    result = investigate('Explain the item rule in A and B', AppConfig(), budget=Budget(AppConfig()),
        resolved_request=dict(resolved_question='Explain the item rule in A and B', output='other',
                              depth='detailed', limit=None))
    assert result['status'] == 'partial', result['trace']
    assert count == 5
    assert 'Repair the failed read-only tool request' in prompts[2][0]['content']
    assert any('depth' in step.get('error', '') for step in result['trace'])


def test_followup_budget_and_empty_outlines_are_not_synthetic_source(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: rows(p, t))
    empty = evidence.execute('source_outline', dict(program='A', terms=['NOT-FOUND']))
    assert empty['returned'] == 0 and empty['inspection_state'] == 'navigation_only'
    assert not outline_followups(evidence, set())
    evidence.execute('source_outline', dict(program='A', terms=['9876']))
    calls = outline_followups(evidence, set(), maximum=1)
    assert len(calls) == 1
    assert not outline_followups(evidence, {digest(calls[0])}, maximum=1)
