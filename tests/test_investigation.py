"""Contract checks for investigation, using small synthetic evidence and no LLM."""
from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from cobol_rag.config import AppConfig
from cobol_rag.investigation import investigate, tool_view, normalize_decision
from cobol_rag.investigation_tools import EvidenceTools, ToolError


@pytest.fixture
def evidence(monkeypatch):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A', 'B'))
    data = {
        ('A', 'files'): [dict(name='A.CBL', source_file='A.CBL')],
        ('B', 'files'): [dict(name='B.CBL', source_file='B.CBL')],
        ('A', 'variables'): [dict(name='V1', controls_flow=True), dict(name='V2', controls_flow=False)],
        ('B', 'variables'): [dict(name='V1', controls_flow=False)],
        ('A', 'calls'): [dict(name='B', caller='A', target='B', commarea='AREA')],
        ('B', 'calls'): [],
    }
    def rows(self, program, table):
        return [dict(r, program=program, _artifact=table, _row_id=f'{program}:{table}:{i}')
                for i, r in enumerate(deepcopy(data[(program, table)]))]
    monkeypatch.setattr(EvidenceTools, 'rows', rows)
    return EvidenceTools(AppConfig())


def test_count_followup_preserves_collection(evidence):
    count = evidence.execute('query', dict(programs=['A'], table='variables', limit=0))
    assert count['total_matches'] == 2 and count['returned'] == 0
    followup = EvidenceTools(AppConfig(), evidence.memory)
    filtered = followup.execute('select', dict(result_id=count['result_id'], basis='collection',
        where=[dict(field='controls_flow', op='eq', value=True)]))
    assert [r['name'] for r in filtered['rows']] == ['V1']


def test_displayed_and_whole_collection_are_distinct(evidence):
    page = evidence.execute('query', dict(programs=['A'], table='variables', limit=1))
    shown = evidence.execute('select', dict(result_id=page['result_id'], basis='displayed'))
    whole = evidence.execute('select', dict(result_id=page['result_id'], basis='collection'))
    assert shown['total_matches'] == 1 and whole['total_matches'] == 2


def test_incoming_query_joins_parameter_to_caller(evidence):
    found = evidence.execute('query', dict(programs=['A', 'B'], table='calls',
        where=[dict(field='target', op='eq', value='B')]))
    assert [(r['caller'], r['commarea']) for r in found['rows']] == [('A', 'AREA')]


def test_callers_primitive_keeps_direction_and_parameters(evidence):
    found = evidence.execute('callers', dict(target='B'))
    assert [(r['caller'], r['target'], r['commarea']) for r in found['rows']] == [('A', 'B', 'AREA')]
    assert evidence.execute('callers', dict(target='A'))['total_matches'] == 0


def test_empty_collection_supports_scoped_absence(evidence):
    from cobol_rag.investigation import answer_checks
    found = evidence.execute('callers', dict(target='A'))
    candidate = dict(answer='No callers of A were found in this analyzed inventory.', mode='technical',
        status='complete', evidence_ids=[found['collection_evidence_id']])
    assert answer_checks('Who calls A?', candidate, evidence, []) == []


def test_large_member_preview_preserves_last_name():
    result = dict(total_matches=33, rows=[dict(evidence_id=f'E{i}', name=f'LONG-VARIABLE-{i}',
        program='A', statement='x' * 500, controls_flow=True) for i in range(33)])
    view = tool_view(result)
    assert [r['name'] for r in view['rows']] == [r['name'] for r in result['rows']]


def test_source_span_representation_repair_does_not_infer_addresses(evidence):
    decision = normalize_decision(dict(action='source', args=dict(program='A', spans=['227,227', '489,489'])), evidence)
    assert decision['calls'][0]['args']['spans'] == [[227, 227], [489, 489]]


def test_old_citations_are_not_reused_as_current_evidence(evidence):
    evidence.memory['last_exchange'] = dict(question='List variables.', answer='V1 and V2 [E1].')
    assert '[E1]' not in evidence.context()['last_exchange']['answer']
    assert '[E1]' in evidence.memory['last_exchange']['answer']


def test_unknown_filter_is_not_silently_ignored(evidence):
    with pytest.raises(ToolError, match='unavailable'):
        evidence.execute('query', dict(programs=['A'], table='variables',
            where=[dict(field='invented', op='eq', value=True)]))


def test_files_tool_alias(evidence):
    result = normalize_decision(dict(action='tools', calls=[dict(tool='files', args=dict(programs=['A']))]), evidence)
    assert result['calls'][0] == dict(tool='files', args=dict(programs=['A']))
    assert evidence.execute('files', {})['total_matches'] == 2


def test_source_range_is_inclusive_and_bounded(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(program=p, source_file='A.CBL', line=n, text=str(n)) for n in range(207, 217)])
    result = evidence.execute('source_range', dict(program='A', start=207, end=215))
    assert [r['line'] for r in result['rows']] == list(range(207, 216))
    with pytest.raises(ToolError):
        evidence.execute('source_range', dict(program='A', start=215, end=207))


def test_copybooks_uses_inclusions_not_filename_extensions(evidence, monkeypatch):
    def rows(p, t):
        assert t == 'copybooks'
        return [dict(name='NO_EXTENSION', program=p, _artifact='architecture.copybooks.json', _row_id='1')]
    monkeypatch.setattr(evidence, 'rows', rows)
    result = evidence.execute('copybooks', dict(programs=['A']))
    assert result['rows'][0]['name'] == 'NO_EXTENSION'


def test_describe_retains_distinct_entity_roles(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(name='X', program=p, _artifact=t)] if t in ('calls', 'copybooks') else [])
    result = evidence.execute('describe', dict(identifier='X', programs=['A']))
    assert {r['entity_type'] for r in result['rows']} == {'calls', 'copybooks'}


def test_source_member_stem_resolves_only_when_unique(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(program=p, source_file='A.CBL', line=50, text='STOP RUN.')])
    assert evidence.execute('source', dict(program='A', source_file='a', spans=[[50, 50]]))['rows'][0]['text'] == 'STOP RUN.'
    with pytest.raises(ToolError, match='Exact available members'):
        evidence.execute('source', dict(program='A', source_file='OTHER', spans=[[50, 50]]))
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(source_file='A.CBL'), dict(source_file='A.CPY')])
    with pytest.raises(ToolError, match='ambiguous'):
        evidence.execute('source', dict(program='A', source_file='A', spans=[[50, 50]]))


def test_quality_tool_preserves_findings_and_proof_limits(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(program=p, _artifact=t,
        commented_out_code_count=15, unreachable_paragraphs=['HANDLER'],
        needs_review_copybooks=['COPY1'], unused_copybooks_proven=[], limitations=['Not compiler proof'])])
    result = evidence.execute('quality', dict(programs=['A']))
    assert len(result['rows']) == 2
    assert result['rows'][0]['commented_out_code_count'] == 15
    assert result['rows'][1]['unused_copybooks_proven'] == []
    assert result['rows'][1]['limitations'] == ['Not compiler proof']


def test_natural_count_is_not_an_explicit_bare_integer_contract():
    from cobol_rag.investigation import response_contract
    assert response_contract('How many variables does A have?').format == 'default'
    assert response_contract('How many variables does A have? Only the number.').format == 'count'


def test_action_schema_requires_final_evidence_and_mode():
    from cobol_rag.investigation import schema
    from jsonschema import validate, ValidationError
    with pytest.raises(ValidationError):
        validate({'action': 'final', 'answer': 'hello'}, schema())


def test_count_followup_keeps_unit(evidence):
    result = evidence.execute('query', dict(programs=['A'], table='variables', limit=0))
    selected = evidence.execute('select', dict(result_id=result['result_id'], basis='collection', limit=0))
    assert selected['unit'] == 'recorded variables'


def test_json_container_recovery_does_not_invent_content():
    import json
    from cobol_rag.investigation import parse_model_json
    assert parse_model_json('{"a": ["x"\n\n') == {'a': ['x']}
    for invalid in ['{"a":"unfinished', '{"a":', '{"a":true,', '{"a":tru']:
        with pytest.raises(json.JSONDecodeError):
            parse_model_json(invalid)


def test_program_scope_cannot_be_an_unresolved_variable_filter(evidence):
    with pytest.raises(ToolError, match='Entity type mismatch'):
        evidence.execute('query', dict(programs=['A'], table='variables', where=[dict(field='name', op='eq', value='A')]))


def test_general_label_does_not_allow_unsupported_program_claim(evidence):
    from cobol_rag.investigation import answer_checks
    candidate = dict(answer='A has no files.', mode='general', status='complete', evidence_ids=[])
    assert 'corpus_entity_answer_requires_evidence' in answer_checks('What files exist?', candidate, evidence, [])


def test_exact_paragraph_search_includes_definition(evidence, monkeypatch):
    source = [dict(source_file='A.CBL', line=10, paragraph='MAIN', text='GO TO HANDLER.', division='PROCEDURE DIVISION'),
              dict(source_file='A.CBL', line=20, paragraph='HANDLER', text='HANDLER.', division='PROCEDURE DIVISION'),
              dict(source_file='A.CBL', line=21, paragraph='HANDLER', text='EXEC CICS ABEND END-EXEC.', division='PROCEDURE DIVISION')]
    monkeypatch.setattr(EvidenceTools, 'rows', lambda *args: source)
    found = evidence.execute('search', dict(programs=['A'], text='HANDLER', limit=1))
    assert found['returned'] == 1
    assert 'EXEC CICS ABEND' in found['exact_paragraph_definitions'][0]['definition']


def test_conversational_label_does_not_skip_review(evidence):
    budget = FakeBudget([
        dict(action='final', answer='Hello!', mode='conversational', status='complete', evidence_ids=[]),
        dict(passed=False, issues=['Does not answer the code question']),
        RuntimeError('offline'), RuntimeError('offline'), RuntimeError('offline'),
    ])
    result = investigate('Explain program A', AppConfig(), budget=budget)
    assert result['status'] == 'failed'


def test_copy_is_not_a_call_category(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda *args: [dict(name='X', call_type='CICSLINK', _row_id='x')])
    with pytest.raises(ToolError, match='Invalid call_type'):
        evidence.execute('query', dict(programs=['A'], table='calls', where=[dict(field='call_type', op='eq', value='COPY')]))


def test_count_unit_and_verified_exchange(evidence):
    budget = FakeBudget([
        dict(action='query', args=dict(programs=['A'], table='variables', limit=0)),
        dict(action='final', mode='technical', status='complete', answer='2', evidence_ids=['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('How many variables in A?', AppConfig(), budget=budget)
    assert result['tools'].evidence['E1']['unit'] == 'recorded variables'
    assert result['tools'].context()['last_exchange']['answer'] == '2'


def test_shared_entities_keep_both_programs(evidence):
    a = evidence.execute('query', dict(programs=['A'], table='variables'))
    b = evidence.execute('query', dict(programs=['B'], table='variables'))
    common = evidence.execute('compare', dict(left_id=a['result_id'], right_id=b['result_id'], operation='intersection'))
    assert [r['name'] for r in common['rows']] == ['V1']
    assert {r['program'] for r in common['rows'][0]['members']} == {'A', 'B'}


class FakeBudget:
    maximum = 6
    output_reserve = 0
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.deadline = time.monotonic() + 30
    def call(self, *args, **kwargs):
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


def final(answer='Two programs: A and B. [E2] [E3]'):
    return dict(action='final', mode='technical', status='complete', answer=answer,
                evidence_ids=['E2', 'E3'], coverage=[dict(requirement='List programs', status='answered')])


def test_review_failure_can_be_repaired(evidence):
    budget = FakeBudget([
        dict(action='tools', requirements=['List programs'], calls=[dict(tool='inventory', args={})]),
        final(), dict(passed=False, issues=['Please clarify scope']),
        final('The analyzed programs are A and B. [E2] [E3]'), dict(passed=True, issues=[]),
    ])
    result = investigate('List the programs.', AppConfig(), budget=budget)
    assert result['status'] == 'complete' and budget.calls == 5


def test_review_exception_cannot_accept_unreviewed_answer(evidence):
    budget = FakeBudget([
        dict(action='tools', requirements=['List programs'], calls=[dict(tool='inventory', args={})]),
        final(), RuntimeError('review unavailable'), RuntimeError('offline'), RuntimeError('offline'),
    ])
    result = investigate('List the programs.', AppConfig(), budget=budget)
    assert result['status'] == 'failed'


def test_disabled_by_default():
    assert not AppConfig().investigation.enabled


def test_large_rows_preserve_count_and_collection_evidence():
    result = dict(result_id='R1', total_matches=105, collection_evidence_id='E1',
                  rows=[dict(evidence_id='E2', name='V', details='x' * 12000)])
    view = tool_view(result)
    assert view['total_matches'] == 105
    assert view['collection_evidence_id'] == 'E1'
    assert view['result_id'] == 'R1'


def test_flat_tool_envelope_preserves_filters(evidence):
    args = dict(programs=['A'], table='calls', where=[dict(field='target', op='eq', value='B')])
    result = normalize_decision(dict(action='query', args=args), evidence)
    assert result['calls'] == [dict(tool='query', args=args)]


def test_single_program_representation_is_normalized_without_scope_change(evidence):
    result = normalize_decision(dict(action='query', args=dict(programs='A', table='variables')), evidence)
    assert result['calls'][0]['args']['programs'] == ['A']
    result = normalize_decision(dict(action='tools', calls=dict(tool='source', args=dict(program='A', spans=[[1,2]]))), evidence)
    assert result['calls'][0]['args']['program'] == 'A'


def test_active_subject_is_not_the_corpus_roster(evidence):
    evidence.execute('query', dict(programs=['A'], table='variables', limit=0))
    context = evidence.context()
    assert context['active_subject']['programs'] == ['A']
    assert context['active_subject']['entity_collection'] == 'variables'
    assert 'programs' not in context


def test_filter_shorthand_preserves_meaning(evidence):
    result = normalize_decision(dict(action='query', args=dict(programs='A', table='variables', where={'controls_flow': True})), evidence)
    assert result['calls'][0]['args']['where'] == [dict(field='controls_flow', op='eq', value=True)]


def test_result_bundle_supports_returned_members_but_count_only_does_not(evidence):
    from cobol_rag.investigation import answer_checks
    result = evidence.execute('query', dict(programs=['A'], table='variables'))
    bundle = evidence.evidence[result['collection_evidence_id']]
    assert [r['name'] for r in bundle['member_rows']] == ['V1', 'V2']
    assert bundle['programs'] == ['A']
    candidate = dict(answer='V1 and V2', mode='technical', status='complete', evidence_ids=[result['collection_evidence_id']])
    assert answer_checks('List the variables.', candidate, evidence, []) == []
    count = evidence.execute('query', dict(programs=['A'], table='variables', limit=0))
    candidate['evidence_ids'] = [count['collection_evidence_id']]
    assert any('counts only' in r for r in answer_checks('List the variables.', candidate, evidence, []))


def test_list_preview_does_not_silently_stop_at_five_rows():
    result = dict(rows=[dict(evidence_id=f'E{i}', name=f'V{i}') for i in range(33)])
    assert len(tool_view(result)['rows']) == 33


def test_declared_list_must_cover_returned_members(evidence):
    from cobol_rag.investigation import answer_checks
    result = evidence.execute('query', dict(programs=['A'], table='variables'))
    candidate = dict(answer='V1', mode='technical', status='complete', collection_output='list',
                     evidence_ids=[result['collection_evidence_id']])
    assert any('V2' in reason for reason in answer_checks('List variables.', candidate, evidence, []))
    candidate['answer'] = 'V1, V2.'
    assert answer_checks('List variables.', candidate, evidence, []) == []


def test_wrong_entity_type_reports_known_alternative(evidence, monkeypatch):
    original = evidence.rows
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(name='HANDLER')] if t == 'paragraphs' else original(p, t))
    with pytest.raises(ToolError, match='paragraphs'):
        evidence.execute('query', dict(programs=['A'], table='variables',
            where=[dict(field='name', op='eq', value='HANDLER')]))


def test_table_tool_alias_preserves_scope(evidence):
    result = normalize_decision(dict(action='tools', calls=[dict(tool='variables', args=dict(programs=['B']))]), evidence)
    assert result['calls'] == [dict(tool='query', args=dict(programs=['B'], table='variables'))]


def test_cited_count_retains_result_handle_without_model_repeating_it(evidence):
    budget = FakeBudget([
        dict(action='query', args=dict(programs=['A'], table='variables', limit=0)),
        dict(action='final', mode='technical', status='complete', answer='2', evidence_ids=['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('How many variables are recorded in A?', AppConfig(), budget=budget)
    assert result['status'] == 'complete'
    rid = result['result_id']
    assert result['tools'].memory['last_result_id'] == rid
    assert result['tools'].memory['collections'][rid]['displayed_ids'] == []
