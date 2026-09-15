from dataclasses import asdict
from types import SimpleNamespace
import json

import pytest

from cobol_rag.query import QueryRoutingDecision, QueryError, _prepare_execution_decision, _route_query
from cobol_rag.query_plan import QueryPlan, QuerySpecification, merge_semantic_plan
from cobol_rag.query_ir import compile_query
from cobol_rag.scope import QueryScope, EntityReference, SessionState
from cobol_rag.config import AppConfig
from cobol_rag.final_scripts_answers import answer_semantic_projection


def scope(*names, kind='variable'):
    return QueryScope(program='DEMO', programs=('DEMO',), entity_source='question',
                      entities=tuple(EntityReference('DEMO', kind, n, 'DEMO|' + n) for n in names))


def prepare(question, spec, resolved=None, state=None):
    decision = QueryRoutingDecision('technical', '', 'general', query_spec=spec)
    return _prepare_execution_decision(decision, resolved or scope(), state, question)[0]


@pytest.mark.parametrize('question', ['What is the type of ITEM-A?', 'Describe ITEM-A.', 'Show the declaration for ITEM-A.'])
def test_missing_explicit_target_is_bound_and_survives_merge(question):
    resolved = scope('ITEM-A')
    decision = prepare(question, {'operator': 'describe', 'capability': 'variable_access', 'fields': ['type']}, resolved)
    plan = merge_semantic_plan(QueryPlan(program='DEMO', programs=('DEMO',), entities=resolved.entities),
                               {'query_spec': decision.query_spec})
    assert plan.query_spec.entity_values == ('ITEM-A',)
    assert plan.entities == resolved.entities


def test_partial_explicit_targets_cannot_disappear_during_scope_refinement():
    decision = prepare('Compare ITEM-A and ITEM-B', {'operator': 'compare', 'capability': 'variable_access',
                       'entity_values': ['ITEM-A'], 'fields': ['type']}, scope('ITEM-A', 'ITEM-B'))
    assert set(decision.query_spec['entity_values']) == {'ITEM-A', 'ITEM-B'}


@pytest.mark.parametrize('question,resolved', [
    ('What is ASKTIME?', scope()),
    ('Explain ABEND00.', scope('ABEND00', kind='paragraph')),
    ('What is ASKTIME?', scope('OLD-FIELD')),
])
def test_wrong_targetless_variable_capability_requires_repair(question, resolved):
    with pytest.raises(QueryError):
        prepare(question, {'operator': 'describe', 'capability': 'variable_access'}, resolved)


def test_explicit_target_overrides_ordinal_memory():
    with pytest.raises(QueryError, match='explicitly named'):
        prepare('Describe ITEM-A', {'operator': 'describe', 'capability': 'variable_access', 'reference_index': 1},
                scope('ITEM-A'), SessionState(last_result_entities=list(scope('OLD-FIELD').entities)))


def test_detail_executor_cannot_broaden_missing_target(tmp_path, monkeypatch):
    root = tmp_path / 'DEMO'
    root.mkdir()
    (root / 'dataflow.used_variables.json').write_text(json.dumps({'variables': [{'variable': 'SECRETLY-ALL'}]}))
    monkeypatch.setenv('COBOL_RAG_FINAL_SCRIPTS_DIR', str(tmp_path))
    spec = QuerySpecification('describe', 'variable_access', fields=('type',))
    query = compile_query('any wording', program='DEMO', programs=('DEMO',), query_spec=spec)
    assert answer_semantic_projection(query) is None


@pytest.mark.parametrize('question,expected', [
    ('Read lines 486 and 217 of DEMO', [('486', '486'), ('217', '217')]),
    ('Show DEMO lines 10 through 12 and 50', [('10', '12'), ('50', '50')]),
])
def test_all_requested_addresses_survive_partial_model_plan(question, expected):
    decision = prepare(question, {'operator': 'lookup', 'capability': 'source_line_lookup',
                       'filters': [{'field': 'line_start', 'operator': 'eq', 'values': ['486']}]})
    filters = {f['field']: f['values'] for f in decision.query_spec['filters']}
    assert list(zip(filters['line_start'], filters['line_end'])) == expected


@pytest.mark.parametrize('quote', [None, 'first one'])
def test_previous_limit_is_not_current_authority(quote):
    with pytest.raises(QueryError, match='THIS request'):
        prepare('Which copybooks does DEMO use?', {'operator': 'list', 'capability': 'copybook_evidence',
                'limit': 1, 'selection_evidence': quote})


def test_current_selection_instruction_is_accepted():
    spec = {'operator': 'list', 'capability': 'variable_inventory', 'limit': 3, 'selection_evidence': 'first three'}
    assert prepare('Show the first three variables', spec).query_spec['limit'] == 3


def test_inventory_can_filter_by_copybook_origin_without_becoming_detail():
    spec = {'operator': 'list', 'capability': 'variable_inventory',
            'filters': [{'field': 'origin', 'operator': 'eq', 'values': ['COPY:MYCOPY']}]}
    decision = prepare('List variables from MYCOPY', spec, scope('MYCOPY', kind='copybook'))
    assert not decision.query_spec.get('entity_values')
    assert decision.query_spec['capability'] == 'variable_inventory'


def test_invalid_model_plan_gets_one_repair_not_an_inventory(monkeypatch):
    import cobol_rag.query as module
    specs = iter([
        {'operator': 'describe', 'capability': 'variable_access'},
        {'operator': 'describe', 'capability': 'paragraph_evidence', 'entity_values': ['ABEND00'], 'fields': ['body']},
    ])
    calls = []
    class Model:
        def chat(self, messages, **kwargs):
            calls.append(messages)
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps({'route': 'technical', 'query_spec': next(specs)})))
    monkeypatch.setattr(module, 'build_llm', lambda *a, **k: Model())
    monkeypatch.setattr(module, 'analyzed_programs', lambda: ('DEMO',))
    result = _route_query('Explain ABEND00.', AppConfig(), preliminary_scope=scope('ABEND00', kind='paragraph'))
    assert result.query_spec['capability'] == 'paragraph_evidence'
    assert len(calls) == 2
    assert 'explicit entity types' in str(calls[1][0].content)


def test_count_acknowledgement_does_not_escape_as_unclear(monkeypatch):
    import cobol_rag.query as module
    replies = iter([
        {'route': 'unclear', 'reply': 'Counting variables in DEMO.'},
        {'route': 'technical', 'query_spec': {'operator': 'aggregate', 'capability': 'variable_inventory', 'fields': ['count']}},
    ])
    class Model:
        def chat(self, *a, **k):
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps(next(replies))))
    monkeypatch.setattr(module, 'build_llm', lambda *a, **k: Model())
    plan = QueryPlan(program='DEMO', intent='variable_inventory', tasks=('variable_inventory',))
    result = _route_query('How many variables in DEMO?', AppConfig(), preliminary_scope=scope(), preliminary_plan=plan)
    assert result.query_spec['operator'] == 'aggregate'
