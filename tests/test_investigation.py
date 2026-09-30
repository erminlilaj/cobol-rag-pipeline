"""Contract checks for investigation, using small synthetic evidence and no LLM."""
from copy import deepcopy
from types import SimpleNamespace
import time

import pytest

from cobol_rag.config import AppConfig
from cobol_rag.investigation import investigate, tool_view, normalize_decision
from cobol_rag.investigation_tools import EvidenceTools, ToolError


def test_source_semantics_preserves_text_but_excludes_boundary_heading():
    from cobol_rag.investigation_tools import source_semantics
    rows = [dict(source_file='A.CBL', paragraph='FAIL', text='COPY ERR.', line=1),
            dict(source_file='A.CBL', paragraph='FAIL', text='SKIP1', line=2),
            dict(source_file='A.CBL', paragraph='FAIL', text='* SAVE STATE', is_comment=True, line=3),
            dict(source_file='A.CBL', paragraph='FAIL', text='* *****', is_comment=True, line=4),
            dict(source_file='A.CBL', paragraph='SAVE', text='SAVE.', line=5)]
    before = deepcopy(rows)
    fixed = source_semantics(rows)
    assert rows == before
    assert [r['text'] for r in fixed] == [r['text'] for r in rows]
    assert fixed[0]['source_role'].startswith('inclusion_directive')
    assert fixed[1]['source_role'].startswith('listing_directive')
    assert fixed[2]['paragraph'] is None and fixed[3]['paragraph'] is None
    assert fixed[4]['paragraph'] == 'SAVE'


def test_source_semantics_keeps_internal_comments():
    from cobol_rag.investigation_tools import source_semantics
    rows = [dict(source_file='B.CBL', paragraph='WORK', text='* INTERNAL', is_comment=True),
            dict(source_file='B.CBL', paragraph='WORK', text='MOVE A TO B.')]
    assert source_semantics(rows)[0]['paragraph'] == 'WORK'


def test_nested_review_repair_preserves_rejection():
    from cobol_rag.investigation import review_answer
    response = dict(passed=False, issues=['Missing evidence'], requested_call_relation=None,
                    repair={'repair_type': 'evidence'})
    budget = FakeBudget([response])
    result = review_answer(budget, 'Review', {'evidence_contracts': []}, [])
    assert result['passed'] is False and result['repair'] == 'evidence'
    assert result['issues'] == ['Missing evidence'] and budget.calls == 1


def test_inline_citations_reconcile_only_existing_evidence():
    tools = SimpleNamespace(evidence={'E1': {'text': 'fact'}})
    candidate = dict(action='final', answer='Fact [E1], unknown [E99]', evidence_ids=[])
    normalized = normalize_decision(candidate, tools)
    assert normalized['evidence_ids'] == ['E1']
    assert '[E99]' in normalized['answer']
    assert candidate['evidence_ids'] == []


def test_nested_evidence_repair_retains_issue():
    from cobol_rag.investigation import review_answer
    response = dict(passed=False, issues=[], repair={'evidence': 'Need source context'})
    result = review_answer(FakeBudget([response]), 'Review', {'evidence_contracts': []}, [])
    assert result['repair'] == 'evidence' and result['passed'] is False
    assert result['issues'] == ['Need source context']


def test_quoted_protocol_predicate_and_flow_direction():
    call = dict(action='tools', calls=[dict(tool='query', args=dict(
        programs=['A'], table='cics', where="command eq 'SYNCPOINT'"))])
    assert normalize_decision(call, None)['calls'][0]['args']['where'] == [
        dict(field='command', op='eq', value='SYNCPOINT')]
    call['calls'][0] = dict(tool='flow_edges', args=dict(programs=['A'], paragraph='FAIL', direction='to'))
    assert normalize_decision(call, None)['calls'][0]['args']['direction'] == 'incoming'


def test_cics_explicit_command_is_lossless_predicate():
    call = dict(action='tools', calls=[dict(tool='query', args=dict(
        programs=['A'], table='cics', command='SYNCPOINT'))])
    args = normalize_decision(call, None)['calls'][0]['args']
    assert args['where'] == [dict(field='command', op='eq', value='SYNCPOINT')]
    assert 'command' not in args


def test_source_window_preserves_condition_continuations():
    rows = [dict(evidence_id=f'E{i}', line=i, source_file='A.CBL', text=f'line {i}') for i in range(1, 11)]
    result = dict(rows=[], source_contexts=[dict(variable='RC', access_line=1, window=dict(rows=rows))])
    before = deepcopy(result)
    view = tool_view(result)
    assert '[E10] A.CBL:10 line 10' in view['source_context']
    assert result == before


def test_nested_operations_are_not_sampled_when_full_result_fits():
    result = dict(rows=[], paragraph_operations=[dict(evidence_id=f'E{i}', statement=f'OPERATION {i}') for i in range(9)])
    assert tool_view(result)['paragraph_operations'] == result['paragraph_operations']


def test_requested_page_must_match_result(evidence, monkeypatch):
    from cobol_rag.investigation import answer_checks
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [
        dict(name=f'V{i:02}', program=p, _row_id=str(i), _artifact='variables') for i in range(1, 19)])
    request = dict(output='list', offset=5, limit=5, order_by='name')
    result = evidence.execute('query', dict(programs=['A'], table='variables', order_by='name', offset=5, limit=5))
    candidate = dict(answer='V06, V07, V08, V09, V10', mode='technical', status='complete',
        collection_output='list', result_id=result['result_id'], evidence_ids=[result['collection_evidence_id']])
    assert not answer_checks('Show the sixth through tenth variables alphabetically.', candidate, evidence, [], request)
    result = evidence.execute('query', dict(programs=['A'], table='variables', order_by='name', limit=10))
    candidate.update(result_id=result['result_id'], evidence_ids=[result['collection_evidence_id']])
    assert any('Requested page differs' in e for e in answer_checks('Show sixth through tenth.', candidate, evidence, [], request))


def test_comparison_members_do_not_duplicate_large_payloads(evidence, monkeypatch):
    from cobol_rag.investigation import result_directory, call_review_errors
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(name='SVC', target='SVC', caller=p,
        program=p, _artifact='calls', _row_id=p, parameter_details=['x' * 10000])])
    a = evidence.execute('callees', dict(programs=['A']))
    b = evidence.execute('callees', dict(programs=['B']))
    result = evidence.execute('compare', dict(left_id=a['result_id'], right_id=b['result_id'], operation='intersection'))
    assert [r['name'] for r in result['rows']] == ['SVC']
    assert all('parameter_details' not in r for r in result['rows'][0]['members'])
    handle = result_directory(evidence)[-1]
    assert handle['result_id'] == result['result_id'] and handle['names'] == ['SVC']
    assert handle['evidence_id'] == result['collection_evidence_id']
    contracts = [dict(evidence_id=result['collection_evidence_id'], relation=result['relation'])]
    assert not call_review_errors(dict(requested_call_relation=dict(direction='outgoing', callers=['A', 'B'])), contracts)
    assert call_review_errors(dict(requested_call_relation=dict(direction='incoming', targets=['A'])), contracts)


def test_group_comparison_preserves_parent_filters(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [
        dict(name=n, program=p, caller=p, target=n, _artifact='calls', _row_id=p+n)
        for n in ('COMMON', p+'ONLY')])
    parent = evidence.execute('callees', dict(programs=['A', 'B']))
    shared = evidence.execute('compare_groups', dict(result_id=parent['result_id'], group_by='program',
        left_value='A', right_value='B', field='name', operation='intersection'))
    assert [r['name'] for r in shared['rows']] == ['COMMON']
    assert {m['program'] for m in shared['rows'][0]['members']} == {'A', 'B'}
    exclusive = evidence.execute('compare_groups', dict(result_id=parent['result_id'], group_by='program',
        left_value='A', right_value='B', operation='difference'))
    assert [r['name'] for r in exclusive['rows']] == ['AONLY']
    assert shared['relation']['operation'] == 'intersection'


def test_static_dependencies_trace_intermediates_without_crossing_sites(evidence, monkeypatch):
    sites = []
    for line, source, target in [(10, 'X', 'TEMP'), (11, 'TEMP', 'Y'), (12, 'Y', 'X')]:
        for variable, kind in [(source, 'read'), (target, 'write')]:
            sites.append(dict(program='A', source_file='A.CBL', paragraph='WORK', line_start=line,
                statement=f'MOVE {source} TO {target}.', variable=variable, access_kind=kind))
    sites.append(dict(program='A', source_file='OTHER.CPY', paragraph='WORK', line_start=10,
                      statement='MOVE X TO TEMP.', variable='UNRELATED', access_kind='write'))
    definitions = [dict(name='TEMP', origin='WORKING-STORAGE',
                        relationships={'declarations': [dict(statement='01 TEMP PIC ZZ9.')]})]
    monkeypatch.setattr(evidence, 'rows', lambda p, t: definitions if t == 'variables' else sites)
    result = evidence.execute('data_dependencies', dict(programs=['A'], source='X', target='Y'))
    assert result['returned'] == 1
    assert result['rows'][0]['variables'][1]['origin'] == 'WORKING-STORAGE'
    assert [(e['from'], e['to']) for e in result['rows'][0]['steps']] == [('X', 'TEMP'), ('TEMP', 'Y')]
    assert evidence.execute('data_dependencies', dict(programs=['A'], source='X', target='Y', max_depth=1))['returned'] == 0
    assert evidence.execute('data_dependencies', dict(programs=['A'], source='X', target='UNRELATED'))['returned'] == 0
    assert 'not proof' in result['limitation']


def test_empty_access_evidence_does_not_claim_declaration_absence(evidence, monkeypatch):
    from cobol_rag.investigation import collection_contracts
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [])
    result = evidence.execute('variable_access', dict(programs=['A'], variables=['MISSING']))
    contract = collection_contracts(evidence, [result['collection_evidence_id']])[0]
    assert 'does not prove' in contract['evidence_scope']


def test_empty_call_result_states_parameter_scope(evidence):
    from cobol_rag.investigation import collection_contracts
    result = evidence.execute('callers', dict(target='A'))
    contract = collection_contracts(evidence, [result['collection_evidence_id']])[0]
    assert contract['total_matches'] == 0
    assert 'no matching call parameters' in contract['evidence_scope']


def test_optional_review_repair_null_does_not_reject_valid_answer():
    from cobol_rag.investigation import review_answer
    budget = FakeBudget([dict(passed=True, issues=[], repair=None, corrected_request=None)])
    review = review_answer(budget, 'Review', dict(evidence_contracts=[]), [])
    assert review['passed'] is True and budget.calls == 1


def test_incidental_call_evidence_does_not_invent_direction_requirement():
    from cobol_rag.investigation import call_review_errors
    contracts = [dict(evidence_id='E1', relation=dict(caller_scope=['A'], target_filters=[]))]
    assert not call_review_errors(dict(requested_call_relation=dict(direction='not_requested', reason='Program purpose overview')), contracts)
    # An explicit incoming question still cannot use unfiltered outgoing calls.
    assert call_review_errors(dict(requested_call_relation=dict(direction='incoming', targets=['A'])), contracts)


def test_incoming_request_target_alias_preserves_direction(evidence):
    from cobol_rag.investigation import call_review_errors
    found = evidence.execute('callers', dict(target='A'))
    contracts = [dict(evidence_id=found['collection_evidence_id'], relation=found['relation'])]
    assert not call_review_errors(dict(requested_call_relation=dict(direction='incoming', target='A')), contracts)
    assert call_review_errors(dict(requested_call_relation=dict(direction='outgoing', callers=['A'])), contracts)


def test_empty_relation_review_rechecks_projection_not_auto_acceptance():
    from cobol_rag.investigation import review_answer
    class ProjectionBudget(FakeBudget):
        def call(self, instructions, payload, **kwargs):
            assert 'empty_relation_proof' in payload
            if self.calls == 1:
                assert 'previous_review' in payload
            return super().call(instructions, payload, **kwargs)
    budget = ProjectionBudget([dict(passed=False, issues=['Parameters missing']), dict(passed=True, issues=[])])
    review = review_answer(budget, 'Review', dict(evidence_contracts=[], verified_collections=[
        dict(total_matches=0, relation=dict(direction='incoming_to_target'))]), [])
    assert review['passed'] is True and budget.calls == 2


def test_copybook_review_join_preserves_unknown_and_proof(monkeypatch):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A',))
    tools = EvidenceTools(AppConfig())
    monkeypatch.setattr(tools, 'root', lambda p: None)
    books = dict(content=dict(inclusions=[dict(copybook='STATE'), dict(copybook='OLD')]))
    review = dict(content=dict(needs_review_copybooks=['OLD'], unused_copybooks_proven=[], proof_level='static'))
    monkeypatch.setattr(tools, 'load', lambda p, a: deepcopy(books if a == 'architecture.copybooks.json' else review))
    result = tools.execute('query', dict(programs=['A'], table='copybooks', where=[dict(field='needs_review', op='eq', value=True)]))
    assert [r['name'] for r in result['rows']] == ['OLD']
    assert result['rows'][0]['proven_unused'] is False
    def missing_review(p, a):
        if a != 'architecture.copybooks.json':
            raise ToolError('Analysis unavailable')
        return deepcopy(books)
    monkeypatch.setattr(tools, 'load', missing_review)
    assert all('needs_review' not in r for r in tools.rows('A', 'copybooks'))


def test_cfg_call_is_not_cobol_call():
    from cobol_rag.investigation import call_kind_errors
    rows = [dict(source_operation='PERFORM', **{'from': 'CHECK', 'to': 'FAIL', 'type': 'CALL'})]
    assert call_kind_errors('CHECK reaches FAIL via CALL.', rows)
    assert not call_kind_errors('CHECK reaches FAIL via PERFORM.', rows)


def test_source_declaration_hints_include_commented_group(monkeypatch):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A',))
    tools = EvidenceTools(AppConfig())
    monkeypatch.setattr(tools, 'root', lambda p: None)
    monkeypatch.setattr(tools, 'path', lambda *a: None)
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts._read_source_lines', lambda *a: [
        dict(line=7, source_file='A.CBL', normalized='01 GROUP-NAME.', is_comment=True, division='DATA DIVISION'),
        dict(line=8, source_file='A.CBL', normalized='* 03 CHILD PIC X.', is_comment=True, division='DATA DIVISION')])
    rows = tools.rows('A', 'source')
    assert [r['declaration']['name'] for r in rows] == ['GROUP-NAME', 'CHILD']
    assert all(r['declaration']['active'] is False for r in rows)


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


def test_review_contract_retains_filter_and_count_beyond_preview(evidence):
    from cobol_rag.investigation import collection_contracts
    result = evidence.execute('query', dict(programs=['A'], table='variables',
        where=[dict(field='controls_flow', op='eq', value=True)]))
    contracts = collection_contracts(evidence, [result['rows'][0]['evidence_id']])
    assert contracts[0]['recipe']['args']['where'][0]['value'] is True
    assert contracts[0]['total_matches'] == 1
    assert contracts[0]['complete'] is True
    empty = evidence.execute('callers', dict(target='A'))
    assert collection_contracts(evidence, [empty['collection_evidence_id']])[0]['total_matches'] == 0


def test_single_predicate_protocol_normalization():
    decision = dict(action='tools', calls=[dict(tool='query', args=dict(
        programs=['A'], table='variables', where='controls_flow eq true'))])
    normalized = normalize_decision(decision, None)
    assert normalized['calls'][0]['args']['where'] == [dict(field='controls_flow', op='eq', value=True)]
    decision['calls'][0]['args']['where'] = 'controls_flow eq true OR origin eq COPY'
    assert isinstance(normalize_decision(decision, None)['calls'][0]['args']['where'], str)
    for expression, op in [('target IS NOT NULL', 'neq'), ('target IS NULL', 'eq')]:
        decision['calls'][0]['args']['where'] = expression
        assert normalize_decision(decision, None)['calls'][0]['args']['where'] == [dict(field='target', op=op, value=None)]


def test_graph_list_checks_endpoints_not_table_label(evidence, monkeypatch):
    from cobol_rag.investigation import answer_checks
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [
        dict(name='edges', program=p, _artifact='controlflow.cfg.json', _row_id='edge1',
             **{'from': 'PREP', 'to': 'FAIL', 'condition': 'RC = 1'})])
    result = evidence.execute('flow_edges', dict(programs=['A'], paragraph='FAIL', direction='incoming'))
    candidate = dict(answer='PREP transfers to FAIL when RC = 1.', mode='technical', status='complete',
        collection_output='list', result_id=result['result_id'], evidence_ids=[result['collection_evidence_id']])
    assert not answer_checks('Which paragraphs enter FAIL?', candidate, evidence, [], dict(output='list'))
    candidate['answer'] = 'edges'
    assert any('PREP' in e for e in answer_checks('Which paragraphs enter FAIL?', candidate, evidence, [], dict(output='list')))


def test_social_claim_failure_can_rewrite_without_tools(evidence):
    class CaptureBudget(FakeBudget):
        def call(self, *args, **kwargs):
            if self.calls == 1:
                assert args[2]['properties']['action'] != {'const': 'tools'}
            return super().call(*args, **kwargs)
    budget = CaptureBudget([
        dict(action='final', mode='conversational', status='complete', evidence_ids=[], answer='Good morning! Ask about A'),
        dict(action='final', mode='conversational', status='complete', evidence_ids=[], answer='Good morning!'),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Good morning!', AppConfig(), budget=budget,
        resolved_request=dict(resolved_question='Good morning!', scope='conversational', output='other', limit=None))
    assert result['status'] == 'complete', result.get('trace')
    assert result['tool_calls'] == 0


def test_multiple_input_collections_require_selected_answer_set(evidence):
    from cobol_rag.investigation import answer_checks
    a = evidence.execute('query', dict(programs=['A'], table='variables'))
    b = evidence.execute('query', dict(programs=['B'], table='variables'))
    candidate = dict(answer='V1', mode='technical', status='complete', collection_output='list',
        evidence_ids=[a['collection_evidence_id'], b['collection_evidence_id']])
    errors = answer_checks('Which variables are shared?', candidate, evidence, [], dict(output='list'))
    assert any('Select or compare' in e for e in errors)
    assert not any('omits returned members' in e for e in errors)
    shared = evidence.execute('compare', dict(left_id=a['result_id'], right_id=b['result_id'], operation='intersection'))
    candidate['result_id'] = shared['result_id']
    candidate['evidence_ids'].append(shared['collection_evidence_id'])
    assert not answer_checks('Which variables are shared?', candidate, evidence, [], dict(output='list'))


def test_selected_subset_does_not_require_context_members(evidence):
    from cobol_rag.investigation import answer_checks
    context = evidence.execute('query', dict(programs=['A'], table='variables'))
    selected = evidence.execute('select', dict(result_id=context['result_id'], basis='collection',
        where=[dict(field='controls_flow', op='eq', value=True)]))
    candidate = dict(answer='V1 controls flow.', mode='technical', status='complete',
        collection_output='list', result_id=selected['result_id'],
        evidence_ids=[context['collection_evidence_id'], selected['collection_evidence_id']])
    assert not answer_checks('Which variables control flow?', candidate, evidence, [], dict(output='list'))
    candidate['answer'] = 'None.'
    assert any('omits returned members' in e for e in answer_checks(
        'Which variables control flow?', candidate, evidence, [], dict(output='list')))


def test_flow_edges_preserves_exact_destination_and_direction(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [
        dict(name='edges', program=p, _artifact='controlflow.cfg.json', _row_id='1',
             **{'from': 'ENTRY', 'to': 'BROWSE', 'condition': 'PHASE = 2', 'line': 20}),
        dict(name='edges', program=p, _artifact='controlflow.cfg.json', _row_id='2',
             **{'from': 'BROWSE', 'to': 'BROWSE-ENTER', 'condition': 'ENTER KEY', 'line': 30})])
    incoming = evidence.execute('flow_edges', dict(programs=['A'], paragraph='BROWSE', direction='incoming'))
    outgoing = evidence.execute('flow_edges', dict(programs=['A'], paragraph='BROWSE', direction='outgoing'))
    assert incoming['rows'][0]['condition'] == 'PHASE = 2'
    assert outgoing['rows'][0]['to'] == 'BROWSE-ENTER'


def test_entry_condition_context_keeps_static_scope_and_exact_target(evidence, monkeypatch):
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [
        dict(program=p, _artifact='controlflow.cfg.json', _row_id='entry',
             **{'from': 'ENTRY', 'to': 'BROWSE', 'condition': 'PHASE = 2', 'line': 20}),
        dict(program=p, _artifact='controlflow.cfg.json', _row_id='child',
             **{'from': 'BROWSE', 'to': 'BROWSE-ENTER', 'condition': 'ENTER KEY', 'line': 30})])
    context = evidence.paragraph_flow_context('A', 'BROWSE')
    assert context['incoming']['complete'] is True
    assert context['incoming']['total'] == 1
    assert context['incoming']['rows'][0]['condition'] == 'PHASE = 2'
    assert context['outgoing']['rows'][0]['to'] == 'BROWSE-ENTER'
    assert 'not runtime reachability' in context['evidence_scope']


@pytest.mark.parametrize('passed', [True, False])
def test_entry_condition_review_scope_does_not_override_verdict(passed):
    from cobol_rag.investigation import review_answer
    verdict = dict(passed=passed, issues=[] if passed else ['Wrong destination'], repair='answer')
    class EntryBudget(FakeBudget):
        def call(self, instructions, payload, **kwargs):
            assert 'unless the original question explicitly requests those guarantees' in instructions
            assert 'Still reject unsupported claims of guaranteed execution' in instructions
            assert 'graph citation is not mandatory' in instructions
            assert 'not the GO TO line in isolation' in instructions
            return super().call(instructions, payload, **kwargs)
    result = review_answer(EntryBudget([verdict]), 'Review', {
        'question': 'What condition enters BROWSE?',
        'evidence_contracts': [],
        'paragraph_flow': [{'paragraph': 'BROWSE', 'incoming': {'total': 1}}]}, [])
    assert result['passed'] is passed
    assert result['issues'] == verdict['issues']


def test_group_context_answer_names_source_field_not_only_destination(evidence):
    from cobol_rag.investigation import answer_checks
    evidence.group_contexts.append(dict(program='A', group='SCREEN-ROW',
        source_prefix='SERVICE', candidate_source_fields=['SERVICE-VALUE']))
    source_id = evidence.register(dict(program='A', _artifact='program.source_lines.jsonl',
        source_file='A.CBL', line=10, text='MOVE SERVICE-VALUE TO SCREEN-VALUE.'))
    candidate = dict(mode='technical', status='complete', evidence_ids=[source_id],
                     answer='SCREEN-VALUE is the field from SERVICE.')
    assert any('source_field_names_missing' in error for error in answer_checks(
        'Which fields of SERVICE appear in SCREEN-ROW?', candidate, evidence, []))
    candidate['answer'] = 'SERVICE-VALUE is copied to SCREEN-VALUE.'
    assert not any('source_field_names_missing' in error for error in answer_checks(
        'Which fields of SERVICE appear in SCREEN-ROW?', candidate, evidence, []))


def test_parameter_preparation_and_copybook_categories_are_queryable(monkeypatch, tmp_path):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A',))
    tools = EvidenceTools(AppConfig())
    monkeypatch.setattr(tools, 'root', lambda p: tmp_path)
    payloads = {
        'architecture.copybooks.json': {'content': {'inclusions': [{'copybook': 'STATE'}],
            'classified': {'state_context': ['STATE']}, 'classification_note': 'Heuristic'}},
        'architecture.call_parameters.json': {'calls': [{'target': 'SERVICE', 'line_start': 12,
            'parameter_details': [{'parameter': 'AREA', 'variables': [{'variable': 'FLAG',
                'writes_before_call': [{'paragraph': 'PREP', 'line_start': 10, 'statement': 'MOVE 1 TO FLAG'}]}]}]}]},
    }
    def load(p, name):
        if name not in payloads:
            raise ToolError('Analysis unavailable')
        return deepcopy(payloads[name])
    monkeypatch.setattr(tools, 'load', load)
    writes = tools.execute('query', dict(programs=['A'], table='parameter_writes',
        where=[dict(field='parameter', op='eq', value='AREA')]))
    assert writes['rows'][0]['line_start'] == 10
    assert writes['rows'][0]['target'] == 'SERVICE'
    books = tools.execute('query', dict(programs=['A'], table='copybooks',
        where=[dict(field='categories', op='contains', value='state_context')]))
    assert books['rows'][0]['name'] == 'STATE'
    assert books['rows'][0]['classification_note'] == 'Heuristic'


def test_optional_null_options_do_not_break_tool_batch():
    decision = normalize_decision(dict(action='tools', calls=[dict(tool='callees', args=dict(programs=['A'], target=None))]), None)
    assert decision['calls'][0]['args'] == {'programs': ['A']}
    required = normalize_decision(dict(action='tools', calls=[dict(tool='callers', args=dict(target=None))]), None)
    assert required['calls'][0]['args']['target'] is None


def test_grouped_call_kind_claim_must_match_retrieved_calls():
    from cobol_rag.investigation import call_kind_errors
    rows = [dict(program='A', target='B', call_type='CICSLINK'), dict(program='A', target='C', call_type='CICSXCTL')]
    assert call_kind_errors('A makes CICS LINK calls to B and C.', rows)
    assert not call_kind_errors('A makes CICS LINK calls to B.', rows)
    assert not call_kind_errors('A uses CICS LINK to B and CICS XCTL to C.', rows)
    rows[0]['call_inventory_counts'] = {'CICSLINK': 1, 'CICSXCTL': 1}
    assert call_kind_errors('A makes two CICS LINK calls and one CICS XCTL call.', rows)
    assert not call_kind_errors('A makes one CICS LINK call and one CICS XCTL call.', rows)


def test_detailed_explanation_cannot_claim_complete_with_only_overview(evidence):
    from cobol_rag.investigation import answer_checks
    evidence.evidence['E1'] = dict(_artifact='program.summary.json', program='A', identity='A program')
    candidate = dict(answer='A is a program [E1].', mode='technical', status='complete', evidence_ids=['E1'])
    request = dict(depth='detailed', output='summary')
    assert any('only overview evidence' in e for e in answer_checks('Explain A.', candidate, evidence, [], request))
    assert not any('only overview evidence' in e for e in answer_checks(
        'Explain A.', candidate, evidence, [], dict(depth='brief', output='summary')))
    candidate['status'] = 'partial'
    assert not any('only overview evidence' in e for e in answer_checks('Explain A.', candidate, evidence, [], request))


def test_detailed_description_returns_bounded_behavior_not_written_answer(evidence, monkeypatch):
    original = evidence.rows
    def rows(program, table):
        if table == 'summary':
            return [dict(name=program, program=program, identity='Example', _artifact='program.summary.json', _row_id='summary')]
        if table in {'edges', 'cics'}:
            return [dict(program=program, _artifact=table, _row_id=str(i), paragraph='HANDLER',
                         statement='EXEC CICS RETURN END-EXEC.', condition='FLAG = 1') for i in range(12)]
        return original(program, table)
    monkeypatch.setattr(evidence, 'rows', rows)
    result = evidence.execute('describe', dict(identifier='A', depth='detailed'))
    assert result['detail_coverage']['edges'] == dict(total=12, returned=8)
    assert result['detail_coverage']['cics'] == dict(total=12, returned=8)
    assert 'answer' not in result
    assert all(r['evidence_id'] in evidence.evidence for r in result['rows'])
    assert next(r for r in result['rows'] if r.get('target') == 'B')['relation']['caller_scope'] == ['A']
    preview = tool_view(result)
    assert any(r.get('statement') for r in preview['rows'])
    assert any(r.get('identity') == 'Example' for r in preview['rows'])


def test_wildcard_identifier_is_not_false_absence(evidence):
    with pytest.raises(ToolError, match='literal variable identifiers'):
        evidence.execute('variable_access', dict(programs=['A'], variables=['*'], access_kind='control'))


def test_owner_program_cannot_be_misused_as_declaration_origin(evidence):
    with pytest.raises(ToolError, match='declaration origin'):
        evidence.execute('query', dict(programs=['A'], table='variables',
                         where=[dict(field='origin', op='eq', value='A')], limit=0))


def test_boolean_filter_cannot_silently_match_nothing(evidence):
    with pytest.raises(ToolError, match='boolean'):
        evidence.execute('query', dict(programs=['A'], table='variables',
            where=[dict(field='controls_flow', op='eq', value='eq')]))


def test_unknown_filter_checked_even_after_no_match(evidence):
    with pytest.raises(ToolError, match='unavailable'):
        evidence.execute('query', dict(programs=['A'], table='variables', where=[
            dict(field='name', op='eq', value='MISSING'),
            dict(field='invented_field', op='eq', value=True)]))


def test_corpus_scope_cannot_bypass_evidence_as_general(evidence):
    from cobol_rag.investigation import answer_checks
    candidate = dict(answer='I can list them if you ask.', mode='general', evidence_ids=[])
    assert 'corpus_entity_answer_requires_evidence' in answer_checks(
        'What can I inspect?', candidate, evidence, [], dict(scope='corpus', output='list', limit=None))


def test_reference_resolution_receives_verified_query(evidence):
    from cobol_rag.investigation import resolve_request
    evidence.execute('callees', dict(programs=['A'], target='B'))
    captured = []
    class ResolverBudget:
        def call(self, instructions, payload, *args, **kwargs):
            captured.append(payload)
            return dict(resolved_question='Does C call B?', output='other', limit=None, scope='corpus')
    resolve_request('Does C call it?', '', evidence.memory, ResolverBudget())
    assert captured[0]['previous_verified_query']['args']['table'] == 'calls'
    assert captured[0]['CURRENT QUESTION'] == 'Does C call it?'


def test_exact_member_count_does_not_require_bullet_format(evidence):
    from cobol_rag.investigation import answer_checks
    result = evidence.execute('query', dict(programs=['A'], table='variables', limit=2))
    candidate = dict(answer='The variables are V1 and V2.', mode='technical', status='complete',
                     collection_output='list', evidence_ids=[result['collection_evidence_id']])
    assert not answer_checks('List the first two variables.', candidate, evidence, [], dict(output='list', limit=2))


def test_call_target_set_keeps_caller_scope(evidence):
    result = evidence.execute('callers', dict(programs=['A'], target=['B', 'C']))
    assert [r['target'] for r in result['rows']] == ['B']
    assert result['relation']['caller_scope'] == ['A']
    assert result['relation']['target_filters'] == [dict(field='target', op='in', value=['B', 'C'])]


def test_outgoing_target_array_and_string_have_same_relation_meaning(evidence):
    from cobol_rag.investigation import call_review_errors
    result = evidence.execute('callees', dict(programs=['A'], target='B'))
    contracts = [dict(evidence_id='E1', relation=result['relation'])]
    for restriction in (dict(target='B'), dict(targets=['B'])):
        assert not call_review_errors(dict(requested_call_relation=dict(direction='outgoing', callers=['A'], **restriction)), contracts)
    assert call_review_errors(dict(requested_call_relation=dict(direction='outgoing', callers=['A'], targets=['C'])), contracts)


def test_unpaged_target_lookups_are_batched_without_changing_scope(evidence):
    decision = normalize_decision(dict(action='tools', calls=[dict(tool='callers', id='client-label', args=dict(programs=['A'], target=t))
                                                             for t in ['B', 'C', 'D', 'E', 'F']]), evidence)
    assert decision['calls'] == [dict(tool='callers', args=dict(programs=['A'], target=['B','C','D','E','F']))]


def test_equivalent_infix_filter_is_normalized(evidence):
    decision = normalize_decision(dict(action='query', args=dict(programs=['A'], table='variables',
        where=[{'controls_flow': 'eq', 'value': True}])), evidence)
    assert decision['calls'][0]['args']['where'] == [dict(field='controls_flow', op='eq', value=True)]


def test_budget_keeps_json_transport_without_duplicate_decision_prompt(monkeypatch):
    import sys, json
    from cobol_rag.investigation import Budget, schema
    captured = []
    def chat(messages, **kwargs):
        captured.append((messages, kwargs))
        return SimpleNamespace(message=SimpleNamespace(content=json.dumps(dict(action='tools', calls=[dict(tool='inventory', args={})]))))
    monkeypatch.setitem(sys.modules, 'llama_index.core.llms', SimpleNamespace(ChatMessage=lambda **kw: kw))
    monkeypatch.setitem(sys.modules, 'cobol_rag.index', SimpleNamespace(build_llm=lambda *a, **kw: SimpleNamespace(chat=chat)))
    contract = schema()
    Budget(AppConfig()).call('Choose a tool.', {'question': 'List programs'}, contract)
    assert captured[0][1]['format'] == 'json'
    assert 'response_schema' not in json.loads(captured[0][0][1]['content'])


def test_row_citation_can_prove_complete_single_member_list(evidence):
    from cobol_rag.investigation import answer_checks
    result = evidence.execute('callees', dict(programs=['A'], target='B'))
    candidate = dict(answer='A calls B with AREA.', mode='technical', status='complete',
                     collection_output='list', evidence_ids=[result['rows'][0]['evidence_id']])
    assert not answer_checks('What parameters are passed to B?', candidate, evidence, [], dict(output='list', limit=None))


def test_list_execution_does_not_inherit_count_only_transport(evidence):
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='query', args=dict(programs=['A'], table='variables', limit=0))]),
        dict(action='final', mode='technical', status='complete', answer='V1, V2.', evidence_ids=['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Which variables?', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Which variables in A?', output='list', limit=None))
    assert result['status'] == 'complete'
    assert next(s for s in result['trace'] if s.get('tool') == 'query')['returned'] == 2


def test_tool_failure_gets_isolated_argument_repair(evidence):
    repaired_payloads = []
    class RepairBudget(FakeBudget):
        def call(self, instructions, payload, *args, **kwargs):
            if 'failed_request' in payload:
                repaired_payloads.append(deepcopy(payload))
            return super().call(instructions, payload, *args, **kwargs)
    budget = RepairBudget([
        dict(action='query', args=dict(programs=['A'], table='variables', where=[dict(field='origin', op='eq', value='A')], limit=0)),
        dict(action='query', args=dict(programs=['A'], table='variables', limit=0)),
        dict(action='final', mode='technical', status='complete', answer='2', evidence_ids=['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('How many variables in A?', AppConfig(), budget=budget)
    assert result['status'] == 'complete'
    assert len(repaired_payloads) == 1
    assert 'declaration origin' in repaired_payloads[0]['failed_request']['error']
    assert repaired_payloads[0]['failed_request']['proposed_repair'] == {
        'tool': 'query', 'args': {'programs': ['A'], 'table': 'variables', 'where': [], 'limit': 0}}
    assert 'observations' not in repaired_payloads[0]


@pytest.mark.parametrize('direction,expected', [('asc', 'V1'), ('desc', 'V2')])
def test_sort_representation_preserves_direction_and_pages_after_sort(evidence, direction, expected):
    decision = normalize_decision(dict(action='query', args=dict(programs=['A'], table='variables',
        order_by=[dict(field='name', op=direction)], limit=1)), evidence)
    result = evidence.execute('query', decision['calls'][0]['args'])
    assert result['rows'][0]['name'] == expected


def test_full_list_cannot_pass_as_partial_page_or_omit_output_kind(evidence):
    from cobol_rag.investigation import answer_checks
    page = evidence.execute('query', dict(programs=['A'], table='variables', limit=1))
    candidate = dict(answer='V1', mode='technical', status='complete', evidence_ids=[page['collection_evidence_id']])
    request = dict(output='list', limit=None)
    errors = answer_checks('Which variables?', candidate, evidence, [], request)
    assert any('collection_output' in e for e in errors)
    assert any('incomplete' in e for e in errors)
    candidate['collection_output'] = 'list'
    assert not answer_checks('Which variable?', candidate, evidence, [], dict(output='list', limit=1))


def test_request_is_frozen_before_tools(evidence):
    request = dict(resolved_question='List programs', output='list', limit=None)
    captured = []
    class Capture(FakeBudget):
        def call(self, instructions, payload, *args, **kwargs):
            captured.append(deepcopy(payload.get('resolved_request')))
            return super().call(instructions, payload, *args, **kwargs)
    answer = final()
    answer['collection_output'] = 'list'
    answer['evidence_ids'] = ['E1', 'E2', 'E3']
    answer['request_contract'] = dict(resolved_question='Different question', output='summary', limit=None)
    budget = Capture([dict(action='tools', request_contract=request, calls=[dict(tool='inventory', args={})]),
                      answer, dict(passed=True, issues=[])])
    result = investigate('List programs', AppConfig(), budget=budget)
    assert result['status'] == 'complete'
    assert captured[0] is None
    assert all(c == request for c in captured[1:])


def test_valid_tool_request_without_optional_contract_passes_actual_schema(evidence):
    from jsonschema import validate
    class SchemaCheckingBudget(FakeBudget):
        def call(self, instructions, payload, output_schema=None, **kwargs):
            response = super().call(instructions, payload, output_schema, **kwargs)
            if output_schema:
                validate(response, output_schema)
            return response
    budget = SchemaCheckingBudget([
        dict(action='tools', calls=[dict(tool='query', args=dict(programs=['A'], table='variables', limit=0))]),
        dict(action='final', mode='technical', status='complete', answer='2', evidence_ids=['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('How many variables does A have?', AppConfig(), budget=budget)
    assert result['status'] == 'complete'
    assert result['tool_calls'] == 1


def test_reviewer_cannot_substitute_a_retrieved_callee(evidence):
    request = dict(resolved_question='Does A call C?', output='other', limit=None,
                   call_relation=dict(direction='outgoing', callers=['A'], target='C'))
    budget = FakeBudget([
        dict(action='tools', request_contract=request, calls=[dict(tool='callees', args=dict(programs=['A']))]),
        dict(action='final', mode='technical', status='complete', answer='A calls B.', evidence_ids=['E2']),
        dict(passed=True, issues=[], requested_call_relation=dict(direction='outgoing', callers=['A'], target='B')),
        RuntimeError('stop'), RuntimeError('stop'),
    ])
    result = investigate('Does A call it too?', AppConfig(), budget=budget)
    assert result['status'] == 'failed'
    assert any('Requested callee changed' in issue for step in result['trace'] for issue in step.get('issues', []))


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


def test_call_relations_preserve_direction_even_when_empty(evidence):
    outgoing = evidence.execute('callees', dict(programs=['A']))
    incoming = evidence.execute('callers', dict(target='A'))
    assert outgoing['total_matches'] == 1 and incoming['total_matches'] == 0
    assert outgoing['relation']['direction'] == 'outgoing_from_callers'
    assert incoming['relation']['direction'] == 'incoming_to_target'
    assert incoming['unit'] == 'program-call records'
    assert incoming['relation']['target_filters'] == [dict(field='target', op='eq', value='A')]
    assert evidence.evidence[incoming['collection_evidence_id']]['relation'] == incoming['relation']
    selected = evidence.execute('select', dict(result_id=incoming['result_id'], basis='collection'))
    assert selected['relation']['direction'] == 'incoming_to_target'


def test_review_cannot_approve_reversed_call_evidence(evidence):
    from cobol_rag.investigation import call_review_errors
    incoming = evidence.execute('callers', dict(target='A'))
    contracts = [dict(evidence_id=incoming['collection_evidence_id'], relation=incoming['relation'])]
    review = dict(passed=True, requested_call_relation=dict(subject='A', direction='outgoing'))
    assert call_review_errors(review, contracts)[0].startswith('call_direction_mismatch')
    review['requested_call_relation']['direction'] = 'incoming'
    assert call_review_errors(review, contracts) == []
    assert call_review_errors(dict(passed=True), contracts)


def test_outgoing_target_filter_is_not_automatically_incoming_question(evidence):
    from cobol_rag.investigation import call_review_errors
    found = evidence.execute('query', dict(programs=['A'], table='calls',
        where=[dict(field='target', op='eq', value='B')]))
    contracts = [dict(evidence_id='E1', relation=found['relation'])]
    # "Does A call B?" and "Does B have A as a caller?" use the same edge.
    for subject, direction in [('A', 'outgoing'), ('B', 'incoming')]:
        assert call_review_errors(dict(requested_call_relation=dict(subject=subject, direction=direction, target='B')), contracts) == []


@pytest.mark.parametrize('text', ['392 MAPA paragraphs', '14 source procedure paragraphs', 'Physical lines: 392', '392 paragraphs'])
def test_metric_labels_cannot_be_swapped(text):
    from cobol_rag.investigation import metric_label_errors
    facts = [dict(metric_facts=[dict(field='mapa_paragraphs', value=14),
        dict(field='source_procedure_paragraphs', value=69), dict(field='physical_source_lines', value=912)])]
    assert metric_label_errors(text, facts)


def test_metric_labels_accept_correct_units():
    from cobol_rag.investigation import metric_label_errors
    facts = [dict(metric_facts=[dict(field='mapa_loc', value=392), dict(field='mapa_paragraphs', value=14),
        dict(field='source_procedure_paragraphs', value=69), dict(field='physical_source_lines', value=912)])]
    assert metric_label_errors('392 MAPA LOC, 14 MAPA paragraphs, 69 source paragraphs and 912 physical lines.', facts) == []


def test_known_external_role_cannot_be_denied_without_lookup(evidence, monkeypatch):
    from cobol_rag.investigation import answer_checks
    from cobol_rag.scope import EntityReference
    monkeypatch.setattr('cobol_rag.scope._catalogue', lambda _: (('A',),
        (EntityReference('A', 'call', 'EXTERNAL-X', 'A|EXTERNAL-X'),)))
    candidate = dict(answer='I have no information about EXTERNAL-X.', mode='general', status='complete', evidence_ids=[])
    assert 'corpus_entity_answer_requires_evidence' in answer_checks('Explain external-x', candidate, evidence, [])
    assert evidence.mentioned_entities('What is external-x?')[0]['entity_type'] == 'call'
    assert evidence.mentioned_entities('good morning') == []


def test_filtered_cics_keeps_host_paragraph_context(evidence, monkeypatch):
    operations = [dict(program='A', name=command, command=command, paragraph='ERROR-HANDLER',
        statement=command, _artifact='architecture.cics_operations.json', _row_id=str(i))
        for i, command in enumerate(['SYNCPOINT', 'LINK', 'ABEND'])]
    monkeypatch.setattr(evidence, 'rows', lambda p, t: deepcopy(operations))
    found = evidence.execute('query', dict(programs=['A'], table='cics',
        where=[dict(field='command', op='eq', value='ABEND')]))
    assert found['total_matches'] == 1  # Context must NOT change the requested selection.
    context = found['paragraph_contexts'][0]
    assert context['entity_type'] == 'paragraph'
    assert [r['command'] for r in context['operations']] == ['SYNCPOINT', 'LINK', 'ABEND']


def test_describe_lookup_records_analysis_gaps(evidence, monkeypatch):
    def missing(p, t):
        if t == 'calls':
            raise ToolError('Analysis gap')
        return []
    monkeypatch.setattr(evidence, 'rows', missing)
    found = evidence.execute('describe', dict(identifier='MISSING-X', programs=['A']))
    lookup = evidence.evidence[found['lookup_evidence_id']]
    assert lookup['matched_roles'] == 0 and lookup['analysis_gaps']


def test_summary_metric_provenance_uses_actual_source_fields(monkeypatch):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A',))
    tool = EvidenceTools(AppConfig())
    monkeypatch.setattr(tool, 'root', lambda p: None)
    data = {'program.summary.json': dict(content='COBOL program', meta=dict(loc=120, paragraphs=8)),
            'program.comments.json': dict(metrics=dict(total_lines=300, total_procedure_paragraphs=21), comments=[]),
            'controlflow.cfg.json': dict(nodes=list(range(22))), 'architecture.call_parameters.json': dict(calls=[])}
    monkeypatch.setattr(tool, 'load', lambda p, a: deepcopy(data[a]))
    row = tool.rows('A', 'summary')[0]
    facts = {f['field']: f for f in row['metric_facts']}
    assert facts['mapa_loc']['value'] == 120 and facts['mapa_loc']['unit'] == 'MAPA LOC'
    assert facts['source_procedure_paragraphs']['value'] == 21
    assert facts['source_procedure_paragraphs']['source_artifact'] == 'program.comments.json'
    assert set(row['source_artifacts']) == set(data)


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
    assert view['returned_member_names'] == [r['name'] for r in result['rows']]
    assert view['preview_incomplete']
    assert view['rows'] and all(r['statement'] == 'x' * 500 for r in view['rows'])


def test_context_packing_keeps_source_facts_and_does_not_mutate_evidence():
    from cobol_rag.investigation import pack_observations, SYSTEM
    rows = [dict(evidence_id='E1', source_file='ERROR.CPY', line=3,
                 statement='EXEC CICS LINK PROGRAM(UTILITY) COMMAREA(ERROR-CODE) LENGTH(4) END-EXEC')]
    newest = dict(result=dict(rows=rows, result_id='new'))
    payload = dict(observations=[dict(result=dict(result_id='old', rows=['x' * 20000])), deepcopy(newest)])
    packed = pack_observations(payload, SYSTEM, 8192)
    assert packed['observations'][-1] == newest
    assert packed['omitted_observations']['result_ids'] == ['old']


def test_alphabetical_alias_matches_verified_order(evidence):
    from cobol_rag.investigation import answer_checks
    result = evidence.execute('query', dict(programs=['A'], table='variables', order_by='name'))
    candidate = dict(answer='V1, V2', mode='technical', status='complete', collection_output='list',
                     result_id=result['result_id'], evidence_ids=[result['collection_evidence_id']])
    assert answer_checks('List variables alphabetically.', candidate, evidence, [],
                         dict(output='list', order_by='alphabetical', limit=None)) == []
    assert any('ordering' in e for e in answer_checks('List in reverse order.', candidate, evidence, [],
                         dict(output='list', order_by='-name', limit=None)))


def test_source_body_alias_and_copybook_predicates(evidence):
    from cobol_rag.investigation_protocol import TOOL_SCHEMAS
    from jsonschema import validate
    for spans in ('body', 'all', ['body'], ['all'], ['start', 'end']):
        decision = normalize_decision(dict(action='source', args=dict(program='A', paragraph='ERROR', spans=spans)), evidence)
        assert decision['calls'][0]['args'] == dict(program='A', paragraph='ERROR')
    args = dict(programs=['A'], where=[dict(field='needs_review', op='eq', value=True)])
    validate(args, TOOL_SCHEMAS['copybooks'])


def test_detailed_read_batch_remains_bounded():
    from cobol_rag.investigation_protocol import decision_schema
    from jsonschema import validate, ValidationError
    call = dict(tool='source', args=dict(program='A', paragraph='WORK'))
    validate(dict(action='tools', calls=[call] * 4), decision_schema())
    with pytest.raises(ValidationError):
        validate(dict(action='tools', calls=[call] * 7), decision_schema())


def test_tool_calls_alias_preserves_operations_and_rejects_conflicts():
    calls = [dict(tool='source_range', args=dict(program='A', start=1, end=9))]
    assert normalize_decision(dict(tool_calls=calls), None) == dict(action='tools', calls=calls)
    with pytest.raises(ToolError, match='Conflicting'):
        normalize_decision(dict(tool_calls=calls, calls=[]), None)


def test_source_line_citations_without_text_are_not_an_answer(evidence):
    from cobol_rag.investigation import answer_checks
    eid = evidence.register(dict(_artifact='program.source_lines.jsonl', program='A',
                                 line=207, source_file='A.CBL', text='MOVE 1 TO X.'))
    candidate = dict(answer=f'Line 207: [{eid}]', mode='technical', status='complete', evidence_ids=[eid])
    assert any('source_entries_missing_text' in e for e in answer_checks('Show A line 207.', candidate, evidence, []))
    candidate['answer'] = f'Line 207: MOVE 1 TO X. [{eid}]'
    assert not answer_checks('Show A line 207.', candidate, evidence, [])
    blank = evidence.register(dict(_artifact='program.source_lines.jsonl', program='A', line=208, text='   '))
    candidate.update(answer=f'Line 208: [{blank}]', evidence_ids=[blank])
    assert not answer_checks('Show A line 208.', candidate, evidence, [])


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


def test_describe_preserves_variable_check_sites(evidence, monkeypatch):
    sites = {'control_sites': [{'paragraph': 'CHECK', 'line_start': 49, 'statement': "IF FLAG = 'E'"}]}
    monkeypatch.setattr(evidence, 'rows', lambda p, t: [dict(name='FLAG', program=p,
        _artifact=t, evidence=sites, controls_flow=True)] if t == 'variables' else [])
    result = evidence.execute('describe', {'identifier': 'FLAG', 'programs': ['A']})
    assert tool_view(result)['rows'][0]['evidence'] == sites


def test_variable_access_exposes_late_sites_as_queryable_rows(monkeypatch, tmp_path):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('A',))
    tool = EvidenceTools(AppConfig())
    monkeypatch.setattr(tool, 'root', lambda p: tmp_path)
    monkeypatch.setattr(tool, 'load', lambda p, a: {'variables': [dict(variable='VALUE', evidence={
        'read_sites': [dict(paragraph='P', line_start=n, statement=f'MOVE VALUE TO V{n}') for n in range(1, 21)],
        'control_sites': [dict(paragraph='CHECK', line_start=-1, statement='VALUE > ZERO')],
    })]})
    result = tool.execute('query', {'programs': ['A'], 'table': 'variable_access',
        'where': [{'field': 'variable', 'op': 'eq', 'value': 'VALUE'},
                  {'field': 'line_start', 'op': 'eq', 'value': 19}]})
    assert result['total_matches'] == 1
    assert tool_view(result)['rows'][0]['statement'] == 'MOVE VALUE TO V19'
    assert result['rows'][0]['access_kind'] == 'read'
    assert result['rows'][0]['_artifact'] == 'dataflow.used_variables.json'
    control = tool.execute('query', {'programs': ['A'], 'table': 'variable_access',
        'where': [{'field': 'access_kind', 'op': 'eq', 'value': 'unlocated_control'}]})
    assert control['returned'] == 1
    assert control['rows'][0]['line_start'] is None
    assert control['rows'][0]['recorded_line_start'] == -1
    assert tool.execute('variable_access', {'programs': ['A'], 'variables': ['VALUE'], 'access_kind': 'control'})['returned'] == 0
    page = tool.execute('variable_access', {'programs': ['A'], 'variables': ['VALUE', 'OTHER'],
                                          'access_kind': 'read', 'offset': 19, 'limit': 1})
    assert page['total_matches'] == 20 and page['returned'] == 1


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


def test_reviewer_cannot_replace_original_request(evidence):
    request = dict(resolved_question='List variables in A.', scope='corpus', output='list', limit=None)
    answer = dict(action='final', answer='V1, V2', mode='technical', status='complete',
                  evidence_ids=['E1'], collection_output='list')
    budget = FakeBudget([
        dict(action='query', args=dict(programs=['A'], table='variables')),
        deepcopy(answer),
        dict(passed=False, issues=['Reconsider request'], repair='request',
             corrected_request=dict(resolved_question='Count variables in A.', output='count', limit=None)),
        dict(action='query', args=dict(programs=['A'], table='variables', order_by='name', limit=100)),
        {**answer, 'evidence_ids': ['E4']}, dict(passed=True, issues=[]),
    ])
    budget.maximum = 8
    result = investigate('List variables in A.', AppConfig(), budget=budget, resolved_request=request)
    assert result['status'] == 'complete'
    assert any(step.get('request_preserved') for step in result['trace'])
    assert not any('corrected_request' in step for step in result['trace'])
    assert request['output'] == 'list'


def final(answer='Two programs: A and B. [E2] [E3]'):
    return dict(action='final', mode='technical', status='complete', answer=answer,
                evidence_ids=['E2', 'E3'], coverage=[dict(requirement='List programs', status='answered')])


def test_review_failure_can_be_repaired(evidence):
    budget = FakeBudget([
        dict(action='tools', requirements=['List programs'], calls=[dict(tool='inventory', args={})]),
        final(), dict(passed=False, issues=['Please clarify scope']),
        dict(action='tools', calls=[dict(tool='inventory', args={'limit': 100})]),
        final('The analyzed programs are A and B. [E2] [E3]'), dict(passed=True, issues=[]),
    ])
    result = investigate('List the programs.', AppConfig(), budget=budget)
    assert result['status'] == 'complete' and budget.calls == 6


def test_followup_context_is_frozen_during_new_tool_queries(evidence):
    previous = evidence.execute('query', {'programs': ['A'], 'table': 'variables'})
    evidence.memory['last_exchange'] = {'question': 'List variables', 'answer': 'V1, V2', 'result_id': previous['result_id']}
    contexts = []
    class CapturingBudget(FakeBudget):
        def call(self, instructions, payload, *args, **kwargs):
            contexts.append(deepcopy(payload['conversation']))
            return super().call(instructions, payload, *args, **kwargs)
    budget = CapturingBudget([
        dict(action='tools', requirements=['List programs'], calls=[dict(tool='inventory', args={})]),
        final(), dict(passed=True, issues=[]),
    ])
    result = investigate('List programs', AppConfig(), state=SimpleNamespace(investigation_memory=evidence.memory), budget=budget)
    assert result['status'] == 'complete'
    assert all(c['last_result_id'] == previous['result_id'] for c in contexts)
    assert contexts[0] == contexts[-1]


def test_memory_keeps_refined_filters_but_removes_old_citations(evidence):
    first = evidence.execute('query', {'programs': ['A'], 'table': 'variables'})
    predicate = dict(field='controls_flow', op='eq', value=True)
    evidence.execute('select', dict(result_id=first['result_id'], basis='collection', where=[predicate]))
    evidence.memory['last_exchange'] = {'answer': 'V1 [E1, E2-E19] [Source 1]'}
    context = evidence.context()
    assert context['active_subject']['filters'] == [predicate]
    assert context['active_subject']['unit'] == 'recorded variables'
    assert context['last_exchange']['answer'].strip() == 'V1'


def test_new_turn_does_not_automatically_promote_memory_to_evidence(evidence):
    evidence.execute('query', {'programs': ['A'], 'table': 'variables'})
    class ContextOnlyBudget(FakeBudget):
        def call(self, instructions, payload, *args, **kwargs):
            if self.calls == 0:
                assert payload['observations'] == []
                assert payload['conversation']['last_result_id']
            return super().call(instructions, payload, *args, **kwargs)
    budget = ContextOnlyBudget([
        dict(action='tools', calls=[dict(tool='inventory', args={})]),
        final(), dict(passed=True, issues=[]),
    ])
    result = investigate('List programs', AppConfig(), state=SimpleNamespace(investigation_memory=evidence.memory), budget=budget)
    assert result['status'] == 'complete'
    assert not any(t.get('tool') == 'memory_recall' for t in result['trace'])


def test_unresolved_request_labels_but_preserves_previous_context(evidence):
    evidence.execute('query', {'programs': ['A'], 'table': 'variables'})
    evidence.memory['unresolved_turn'] = {'question': 'Show lines 10 to 20 in A'}
    before = deepcopy(evidence.memory)
    context = evidence.context()
    assert context['unresolved_turn']['question'] == 'Show lines 10 to 20 in A'
    assert context['last_result_id']
    assert 'unresolved' in context['focus_status']
    assert context['active_subject']['entity_collection'] == 'variables'
    assert evidence.memory == before


def test_memory_recall_revalidates_without_mutating_collection(evidence):
    first = evidence.execute('query', {'programs': ['A'], 'table': 'variables', 'limit': 0})
    before = deepcopy(evidence.memory)
    recalled = evidence.recall_active()
    assert recalled['total_matches'] == 2 and recalled['returned'] == 2
    assert recalled['collection_evidence_id'] in evidence.evidence
    assert evidence.memory == before
    count_only = evidence.recall_active(limit=1)
    assert count_only['total_matches'] == 2 and count_only['rows'] == []
    assert not count_only['complete']
    evidence.memory['collections'][first['result_id']]['fingerprint'] = 'stale'
    with pytest.raises(ToolError, match='Corpus changed'):
        evidence.recall_active()


def test_review_protocol_repair_does_not_rewrite_answer(evidence):
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='callees', args={'programs': ['A']})]),
        dict(action='final', mode='technical', status='complete',
             answer='A calls B. [E2]', evidence_ids=['E2']),
        dict(passed=True, issues=[]),
        dict(passed=True, issues=[], requested_call_relation={'callers': ['A'], 'direction': 'outgoing'}),
    ])
    result = investigate('Which programs does A call?', AppConfig(), budget=budget)
    assert result['status'] == 'complete' and budget.calls == 4, result['trace']
    assert [step['action'] for step in result['trace'] if 'action' in step] == ['tools', 'final']
    assert any(step.get('review_protocol_errors') for step in result['trace'])


@pytest.mark.parametrize('bad', [dict(passed=True, issues=[]), {'passed': 'true', 'issues': []}, []])
def test_invalid_review_stops_after_one_protocol_repair(bad):
    from cobol_rag.investigation import review_answer, ReviewProtocolError
    budget = FakeBudget([bad, bad])
    with pytest.raises(ReviewProtocolError, match='review_protocol_invalid'):
        review_answer(budget, 'Review', {'evidence_contracts': [{'relation': {'caller_scope': ['A']}}]}, [])
    assert budget.calls == 2


def test_null_relationship_is_valid_only_without_call_evidence():
    from cobol_rag.investigation import review_answer, ReviewProtocolError
    response = dict(passed=True, issues=[], requested_call_relation=None)
    budget = FakeBudget([response])
    assert review_answer(budget, 'Review', {'evidence_contracts': []}, [])['passed']
    assert budget.calls == 1
    with pytest.raises(ReviewProtocolError):
        review_answer(FakeBudget([response, response]), 'Review',
                      {'evidence_contracts': [{'relation': {'caller_scope': ['A']}}]}, [])


def test_repaired_review_still_rejects_reversed_direction(evidence):
    from cobol_rag.investigation import review_answer, call_review_errors
    found = evidence.execute('callers', {'target': 'A'})
    contracts = [{'evidence_id': found['collection_evidence_id'], 'relation': found['relation']}]
    budget = FakeBudget([dict(passed=True, issues=[]), dict(passed=True, issues=[],
        requested_call_relation={'callers': ['A'], 'direction': 'outgoing'})])
    review = review_answer(budget, 'Review', {'evidence_contracts': contracts}, [])
    assert call_review_errors(review, contracts)[0].startswith('call_direction_mismatch')


def test_review_requires_evidence_for_each_call_subject(evidence):
    from cobol_rag.investigation import call_review_errors
    contracts = [{'evidence_id': 'E1', 'relation': evidence.execute('callees', {'programs': ['A']})['relation']}]
    review = dict(requested_call_relation={'subjects': ['A', 'B'], 'direction': 'outgoing'})
    assert call_review_errors(review, contracts)
    contracts.append({'evidence_id': 'E2', 'relation': evidence.execute('callees', {'programs': ['B']})['relation']})
    assert call_review_errors(review, contracts) == []


def test_incoming_review_repairs_conflicting_target_field(evidence):
    from cobol_rag.investigation import review_answer, call_review_errors
    contracts = [{'evidence_id': 'E1', 'relation': evidence.execute('callers', {'target': 'A'})['relation']}]
    budget = FakeBudget([
        dict(passed=True, issues=[], requested_call_relation={'subjects': ['B'], 'direction': 'incoming', 'target': 'A'}),
        dict(passed=True, issues=[], requested_call_relation={'targets': ['A'], 'direction': 'incoming'}),
    ])
    review = review_answer(budget, 'Review', {'evidence_contracts': contracts}, [])
    assert budget.calls == 2 and call_review_errors(review, contracts) == []


def test_supplemental_review_roles_do_not_create_false_rejection(evidence):
    from cobol_rag.investigation import review_answer, call_review_errors
    review = dict(passed=True, issues=[], requested_call_relation={
        'direction': 'outgoing', 'callers': ['A'], 'targets': ['B']})
    contracts = [{'evidence_id': 'E1', 'relation': evidence.execute('callees', {'programs': ['A']})['relation']}]
    for scope in [contracts, []]:
        budget = FakeBudget([review])
        checked = review_answer(budget, 'Review', {'evidence_contracts': scope}, [])
        assert budget.calls == 1 and call_review_errors(checked, scope) == []


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
    result['collection_evidence_id'] = 'Ecollection'
    assert tool_view(result)['returned_member_names'] == [f'V{i}' for i in range(33)]


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


def test_repeated_successful_count_reuses_evidence(evidence):
    call = dict(action='query', args=dict(programs=['A'], table='variables', limit=0))
    budget = FakeBudget([deepcopy(call), deepcopy(call),
        dict(action='final', mode='technical', status='complete', answer='2', evidence_ids=['E1']),
        dict(passed=True, issues=[])])
    result = investigate('How many variables are recorded in A?', AppConfig(), budget=budget)
    assert result['status'] == 'complete'
    count = evidence.execute('query', dict(programs=['A'], table='variables', limit=0))
    assert 'total_matches is the answer count' in tool_view(count)['count_instruction']
