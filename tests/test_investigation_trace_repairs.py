"""Generalized reproductions of the 17-Nov failure classes; no live LLM."""
from copy import deepcopy

import pytest

from test_investigation import evidence, FakeBudget
from cobol_rag.config import AppConfig
from cobol_rag.investigation import (investigate, usable_request, call_review_errors,
                                     answer_checks, review_answer, collection_contracts)
from cobol_rag.investigation_tools import EvidenceTools, match, ToolError
from cobol_rag.citations import evidence_ids, render_citations
from cobol_rag.source_review import review_context


def final(answer, ids, status='complete'):
    return dict(action='final', mode='technical', status=status, answer=answer, evidence_ids=ids)


def query(program):
    return dict(tool='query', args=dict(programs=[program], table='variables'))


def test_cached_read_does_not_forbid_a_new_relevant_lookup(evidence):
    budget = FakeBudget([
        dict(action='tools', calls=[query('A')]),
        dict(action='tools', calls=[query('A')]),
        dict(action='tools', calls=[query('B')]),
        final('A has 2 variables; B has 1.', ['E1', 'E4']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Explain the variable inventories of A and B.', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Explain A and B.', output='other', limit=None))
    assert result['status'] == 'complete', result['trace']
    assert any(step.get('status') == 'cached' for step in result['trace'])
    assert any(step.get('tool') == 'query' and step.get('args', {}).get('programs') == ['B'] for step in result['trace'])


def test_stagnant_cached_rounds_still_stop(evidence):
    budget = FakeBudget([dict(action='tools', calls=[query('A')]) for _ in range(5)])
    budget.maximum = 10
    result = investigate('Explain A.', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Explain A.', output='other', limit=None))
    assert any(step.get('status') == 'stopped_no_progress' for step in result['trace'])
    assert budget.calls == 4


def test_plural_roles_and_internal_paragraphs_are_not_frozen_as_external_calls():
    request = dict(output='other', limit=None, call_relations=[
        dict(direction='outgoing', targets=['SERVICE']),
        dict(direction='incoming', targets=['LOCAL-ROUTINE']),
        dict(direction='outgoing', callers=['OWNER'], targets=['SERVICE'])])
    entities = [dict(name='OWNER', entity_type='program'),
                dict(name='LOCAL-ROUTINE', entity_type='paragraph', program='OWNER')]
    result = usable_request(request, entities)
    assert result['call_relations'] == [request['call_relations'][2]]
    assert len(request['call_relations']) == 3


def test_internal_perform_callers_can_answer_without_an_external_program_contract(evidence, monkeypatch):
    edges = [dict(program='A', name='edges', **{'from': name, 'to': 'LOCAL'},
                  source_operation='PERFORM', type='CALL_RANGE', evidence='PERFORM LOCAL THRU LOCAL-EXIT.',
                  _artifact='controlflow.cfg.json', _row_id=name)
             for name in ('FIRST', 'SECOND', 'THIRD', 'FOURTH')]
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: edges if t == 'edges' else [])
    monkeypatch.setattr(EvidenceTools, 'mentioned_entities', lambda self, text: [
        dict(name='A', program='A', entity_type='program'),
        dict(name='LOCAL', program='A', entity_type='paragraph')])
    monkeypatch.setattr(EvidenceTools, 'paragraph_operations', lambda *a: [])
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='flow_edges', args=dict(programs=['A'], paragraph='LOCAL', direction='incoming'))]),
        final('FIRST, SECOND, THIRD and FOURTH invoke LOCAL through PERFORM THRU.', ['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Who performs LOCAL inside A?', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Who performs LOCAL inside A?', output='list',
                                               limit=None, call_relation=dict(direction='incoming', targets=['LOCAL'])))
    assert result['status'] == 'complete', result['trace']
    candidate = final('FIRST, SECOND and THIRD invoke LOCAL.', ['E1'])
    candidate['collection_output'] = 'list'
    assert any('FOURTH' in error for error in answer_checks('List paragraph callers.', candidate,
        result['tools'], [], dict(output='list')))


def test_unambiguous_local_entity_description_does_not_scan_the_corpus(evidence, monkeypatch):
    programs_read = []
    def rows(self, program, table):
        programs_read.append(program)
        return [dict(name='LOCAL-VALUE', program=program, _artifact='variables', _row_id=program)] if table == 'variables' else []
    monkeypatch.setattr(EvidenceTools, 'rows', rows)
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='describe', args=dict(identifier='LOCAL-VALUE'))]),
        final('LOCAL-VALUE is recorded in A.', ['E1']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Explain LOCAL-VALUE in A.', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Explain LOCAL-VALUE in A.', output='other', limit=None))
    assert result['status'] == 'complete', result['trace']
    assert set(programs_read) == {'A'}


def test_incoming_caller_search_is_not_silently_narrowed_to_its_target(evidence):
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='callers', args=dict(target='A'))]),
        final('No analyzed callers of A were found.', ['E1']),
        dict(passed=True, issues=[], requested_call_relation=dict(direction='incoming', targets=['A'])),
    ])
    result = investigate('Which programs call A?', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Which programs call A?', output='other', limit=None,
                                               call_relation=dict(direction='incoming', targets=['A'])))
    assert result['status'] == 'complete', result['trace']
    call = next(step for step in result['trace'] if step.get('tool') == 'callers')
    assert 'programs' not in call['args']


def combined_contracts():
    return [dict(program='OWNER', caller='OWNER', target=target, call_type='CALL',
                 relation=dict(caller_scope=['OWNER'], target_filters=[
                     dict(field='target', op='in', value=['SERVICE1', 'SERVICE2'])]))
            for target in ('SERVICE1', 'SERVICE2')]


def test_combined_call_result_supports_per_target_dimensions_but_not_missing_or_wrong_roles():
    review = dict(requested_call_relations=[dict(direction='outgoing', callers=['OWNER'], targets=[target])
                                          for target in ('SERVICE1', 'SERVICE2')])
    contracts = combined_contracts()
    assert not call_review_errors(review, contracts)
    assert call_review_errors(review, contracts[:1])
    wrong = deepcopy(review)
    wrong['requested_call_relations'][0]['callers'] = ['OTHER']
    assert call_review_errors(wrong, contracts)
    wrong['requested_call_relations'][0] = dict(direction='incoming', targets=['OWNER'])
    assert call_review_errors(wrong, contracts)
    one_dimension = dict(requested_call_relation=dict(direction='outgoing', callers=['OWNER'],
                                                       targets=['SERVICE1', 'SERVICE2']))
    # The exact parent query scope establishes direction even if the prose
    # later needs additional rows for member details. Membership is checked
    # separately; direction validation must not fabricate a reversal.
    assert not call_review_errors(one_dimension, contracts[:1])


def test_narrative_support_collections_are_not_an_implicit_set_operation(evidence):
    a = evidence.execute('query', dict(programs=['A'], table='variables'))
    b = evidence.execute('query', dict(programs=['B'], table='variables'))
    candidate = dict(mode='technical', status='complete', collection_output='list',
                     answer='V1 is recorded in both programs; V2 in A.',
                     evidence_ids=[a['collection_evidence_id'], b['collection_evidence_id']])
    assert not answer_checks('Explain the evidence in A and B.', candidate, evidence, [],
                             dict(output='list', collection_semantics='narrative'))
    assert any('Select or compare' in error for error in answer_checks(
        'Which members are shared?', candidate, evidence, [], dict(output='list', collection_semantics='set_comparison')))


def test_ambiguous_collection_contract_requires_semantic_review_not_an_arbitrary_intersection(evidence):
    a = evidence.execute('query', dict(programs=['A'], table='variables'))
    b = evidence.execute('query', dict(programs=['B'], table='variables'))
    payload = dict(resolved_request=dict(output='list', limit=None), evidence_contracts=[],
                   verified_collections=collection_contracts(evidence, [a['collection_evidence_id'], b['collection_evidence_id']]))
    budget = FakeBudget([dict(passed=True, issues=[], requested_collection_semantics='narrative')])
    result = review_answer(budget, 'Review original meaning.', payload, [])
    assert result['requested_collection_semantics'] == 'narrative'


def test_interface_review_accepts_typed_plural_dimensions_without_invalid_frozen_metadata():
    review = dict(passed=True, issues=[], requested_call_relations=[
        dict(direction='outgoing', callers=['OWNER'], targets=[target]) for target in ('SERVICE1', 'SERVICE2')])
    result = review_answer(FakeBudget([review]), 'Review interfaces.',
                          dict(evidence_contracts=combined_contracts(), resolved_request=dict(output='other')), [])
    assert result['passed'] and not call_review_errors(result, combined_contracts())


def source_rows(program, table):
    if table != 'source':
        return []
    rows = [
        (1, 'DATA DIVISION', None, '01 DATA-AREA PIC X.'),
        (2, 'PROCEDURE DIVISION', 'START', 'START.'),
        (3, 'PROCEDURE DIVISION', 'START', 'OPEN INPUT INPUT-FILE.'),
        (4, 'PROCEDURE DIVISION', 'CALCULATE', 'CALCULATE.'),
        (5, 'PROCEDURE DIVISION', 'CALCULATE', 'COMPUTE IMPONIB = IMPORTO * PERC / 100.'),
        (6, 'PROCEDURE DIVISION', 'CHECK', 'CHECK.'),
        (7, 'PROCEDURE DIVISION', 'CHECK', "IF FLAG = 'S'"),
        (8, 'PROCEDURE DIVISION', 'CHECK', 'THEN MOVE ZERO TO AMOUNT'),
        (9, 'PROCEDURE DIVISION', 'CHECK', 'GO TO FINISH.'),
        (10, 'PROCEDURE DIVISION', 'CHECK', "MOVE '2' TO FUNCTION-CODE."),
    ]
    return [dict(program=program, source_file=f'{program}.CBL', line=line, division=division,
                 paragraph=paragraph, text=text, normalized=text, _artifact='program.source_lines.jsonl',
                 _row_id=f'{program}:{line}') for line, division, paragraph, text in rows]


def test_source_outline_finds_arithmetic_and_validation_without_english_business_name_matching(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: source_rows(p, t))
    result = evidence.execute('source_outline', dict(program='A', verbs=['COMPUTE']))
    assert [(row['name'], row['first_line']) for row in result['rows']] == [('CALCULATE', 4)]
    assert result['rows'][0]['matches'][0]['line'] == 5
    result = evidence.execute('source_outline', dict(program='A', terms=['FLAG']))
    assert [row['name'] for row in result['rows']] == ['CHECK']
    empty = evidence.execute('source', dict(program='A', paragraph='PROCEDURE-DIVISION'))
    assert empty['returned'] == 0 and empty['paragraph_lookup'] == 'unavailable'
    assert 'CHECK' in empty['available_paragraphs']


def test_empty_source_can_be_followed_by_a_real_paragraph(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: source_rows(p, t))
    budget = FakeBudget([
        dict(action='tools', calls=[dict(tool='source', args=dict(program='A', paragraph='NONEXISTENT'))]),
        dict(action='tools', calls=[dict(tool='source', args=dict(program='A', paragraph='CHECK'))]),
        final("When FLAG = 'S', processing branches to FINISH.", ['E2', 'E4']),
        dict(passed=True, issues=[]),
    ])
    result = investigate('Explain the check in A.', AppConfig(), budget=budget,
                         resolved_request=dict(resolved_question='Explain the check in A.', output='other', limit=None))
    assert result['status'] == 'complete', result['trace']


def test_source_review_retains_true_branch_and_exact_storage_query_scope(evidence, monkeypatch):
    monkeypatch.setattr(EvidenceTools, 'rows', lambda self, p, t: source_rows(p, t))
    evidence.execute('source', dict(program='A', paragraph='CHECK'))
    context = review_context(evidence, [dict(name='FLAG', program='A', entity_type='variable')])
    branch = context['condition_windows'][0]
    assert branch['contains_focal_name']
    assert branch['recorded_through_period']
    assert branch['source'][-1]['text'] == 'GO TO FINISH.'


def test_empty_group_access_query_cannot_be_presented_as_complete_storage_proof(evidence):
    result = evidence.execute('query', dict(programs=['A'], table='variables',
                                          where=[dict(field='name', op='eq', value='MISSING')]))
    row = evidence.evidence[result['collection_evidence_id']]
    row['recipe']['args']['table'] = 'variable_access'
    candidate = dict(mode='technical', status='complete', answer='No modification occurs.',
                     evidence_ids=[result['collection_evidence_id']])
    request = dict(output='other', analysis_checks=['storage_effects'])
    assert any('Storage-effect coverage is partial' in error for error in answer_checks(
        'Does the area change?', candidate, evidence, [], request))
    candidate.update(status='partial', answer='No direct accesses are recorded; child fields, aliases and callee effects remain unverified.')
    assert not answer_checks('Does the area change?', candidate, evidence, [], request)


@pytest.mark.parametrize('kind', ['condition_scope', 'storage_effects', 'flow_destinations', 'execution_environment'])
def test_reviewer_cannot_approve_a_claim_it_marks_unsupported(kind):
    budget = FakeBudget([dict(passed=True, issues=[], source_claim_checks=[
        dict(check=kind, claim='A categorical claim.', supported=False, reason='The recorded scope does not support it.')])])
    review = review_answer(budget, 'Review', dict(evidence_contracts=[],
        resolved_request=dict(output='other', analysis_checks=[kind])), [])
    assert review['passed'] is False and review['repair'] == 'answer'


def test_optional_dynamic_target_field_does_not_break_literal_call_filtering():
    predicate = dict(field='target_variable', op='eq', value='DISPATCH')
    assert not match(dict(target='LITERAL', call_type='CALL'), predicate)
    assert match(dict(target='SERVICE', target_variable='DISPATCH'), predicate)
    with pytest.raises(ToolError):
        match(dict(target='SERVICE'), dict(field='misspelled_field', op='eq', value='X'))


def test_citation_groups_expand_only_registered_selected_members(evidence):
    assert evidence_ids('Facts [E1, E3-E5].') == ['E1', 'E3', 'E4', 'E5']
    assert render_citations('Facts [E1, E3-E5].', ['E1', 'E3', 'E4', 'E5']) == 'Facts [Source 1] [Source 2] [Source 3] [Source 4].'
    with pytest.raises(ValueError):
        render_citations('Facts [E1-E3].', ['E1', 'E3'])
    with pytest.raises(ValueError):
        evidence_ids('Facts [E8-E2].')
    with pytest.raises(ValueError):
        evidence_ids('Facts [E1-E999999].')
    with pytest.raises(ValueError):
        evidence_ids('Facts [E1 / E3].')
