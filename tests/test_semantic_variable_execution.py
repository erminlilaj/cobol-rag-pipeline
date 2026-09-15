from dataclasses import asdict, replace
import json

import pytest

from cobol_rag.query_plan import QueryPlan, QuerySpecification, QueryFilter, merge_semantic_plan
from cobol_rag.query_ir import compile_query, entity_type_named
from cobol_rag.final_scripts_answers import answer_semantic_projection
from cobol_rag.scope import EntityReference, QueryScope, SessionState
from cobol_rag.query import _scope_with_result_reference, QueryRoutingDecision


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    root = tmp_path / 'DEMO'
    root.mkdir()
    rows = [
        {'variable': 'GROUP-A', 'origin': 'WORKING-STORAGE', 'controls_flow': False,
         'relationships': {'children': ['ITEM-A'], 'declarations': [
             {'line_start': 10, 'source_file': 'DEMO.CBL', 'statement': '01 GROUP-A.'}]}},
        {'variable': 'ITEM-A', 'origin': 'WORKING-STORAGE', 'controls_flow': True,
         'relationships': {'parents': ['GROUP-A'], 'declarations': [
             {'line_start': 11, 'source_file': 'DEMO.CBL', 'statement': '05 ITEM-A PIC 9(4).'}]}}
    ]
    (root / 'dataflow.used_variables.json').write_text(json.dumps({'program': 'DEMO', 'variables': rows}))
    monkeypatch.setenv('COBOL_RAG_FINAL_SCRIPTS_DIR', str(tmp_path))
    return rows


def execute(spec):
    query = compile_query('irrelevant wording', program='DEMO', programs=('DEMO',), query_spec=spec)
    return answer_semantic_projection(query)


def test_count_and_filtered_count(corpus):
    spec = QuerySpecification('aggregate', 'variable_inventory', fields=('count',))
    assert execute(spec) == '2'
    assert execute(replace(spec, filters=(QueryFilter('controls_flow', values=('true',)),))) == '1'


def test_order_and_limit(corpus):
    spec = QuerySpecification('list', 'variable_inventory', limit=1, order_by='source_line')
    result = execute(spec)
    assert '- GROUP-A' in result
    assert '- ITEM-A' not in result


@pytest.mark.parametrize('name,expected', [('GROUP-A', 'Group item'), ('ITEM-A', 'PIC 9(4)')])
def test_declaration_type(corpus, name, expected):
    result = execute(QuerySpecification('describe', 'variable_access', entity_values=(name,), fields=('type',)))
    assert expected in result


def test_semantic_spec_overrides_keyword_inventory():
    plan = QueryPlan(program='DEMO', programs=('DEMO',), intent='source_metrics', tasks=('source_metrics',), confidence=.95)
    merged = merge_semantic_plan(plan, {'query_spec': asdict(QuerySpecification('aggregate', 'variable_inventory', fields=('count',)))})
    assert merged.query_spec.capability == 'variable_inventory'
    assert 'source_metrics' not in merged.tasks
    assert merged.response_contract.format == 'count'


def test_ordinal_uses_displayed_result_not_input_scope():
    first = EntityReference('DEMO', 'variable', 'GROUP-A', 'DEMO|GROUP-A')
    second = EntityReference('DEMO', 'variable', 'ITEM-A', 'DEMO|ITEM-A')
    state = SessionState()
    scope = QueryScope(program='DEMO', programs=('DEMO',))
    state.update(scope, [], QueryPlan(result_entities=(second, first)).as_dict())
    decision = QueryRoutingDecision('technical', '', 'variable_dataflow', query_spec={
        'capability': 'variable_access', 'reference_index': 1})
    selected = _scope_with_result_reference(scope, decision, state)
    assert selected.entity_value == 'ITEM-A'


def test_identifier_components_are_not_inventory_types():
    assert entity_type_named('Explain PREPARA-MAP-010') is None


def test_call_direction_is_not_reinterpreted():
    spec = QuerySpecification('list', 'call_evidence', direction='outgoing')
    for question in ['who invokes DEMO', 'programs called in DEMO browsing file']:
        query = compile_query(question, program='DEMO', query_spec=spec)
        assert query.direction == 'outgoing'


def test_missing_call_direction_is_not_executed(corpus):
    assert execute(QuerySpecification('list', 'call_evidence')) is None


def test_named_parameter_lookup_does_not_become_variable_inventory(corpus, tmp_path):
    (tmp_path / 'DEMO' / 'architecture.call_parameters.json').write_text(json.dumps({
        'program': 'DEMO', 'calls': [{'target': 'SUBPGM', 'call_type': 'CICSLINK',
                                    'parameters': ['WS-AREA'], 'commarea': 'WS-AREA', 'line_start': 40}]}))
    spec = QuerySpecification('project', 'call_evidence', entity_values=('SUBPGM',),
                              direction='outgoing', target_entity='SUBPGM', fields=('commarea', 'parameters'))
    result = execute(spec)
    assert 'WS-AREA' in result and 'GROUP-A' not in result


def test_followup_context_is_bounded(monkeypatch):
    from cobol_rag.query import _execution_routing_prompt
    monkeypatch.setattr('cobol_rag.final_scripts_answers.analyzed_programs', lambda: ('DEMO',))
    entities = tuple(EntityReference('DEMO', 'variable', f'ITEM-{i}', f'DEMO|VARIABLE|ITEM-{i}') for i in range(500))
    scope = QueryScope(program='DEMO', programs=('DEMO',), entities=entities)
    state = SessionState(last_result_entities=list(entities))
    prompt = _execution_routing_prompt('List the first three of them', scope, state)
    assert len(prompt) < 9000
    assert 'ITEM-499' not in prompt


def test_filtered_inventory_does_not_require_every_previous_entity():
    entities = tuple(EntityReference('DEMO', 'variable', n, n) for n in ('GROUP-A', 'ITEM-A'))
    plan = QueryPlan(program='DEMO', programs=('DEMO',), entities=entities)
    spec = QuerySpecification('list', 'variable_inventory', fields=('name',), limit=1)
    merged = merge_semantic_plan(plan, {'query_spec': asdict(spec)})
    assert merged.entities == ()


def test_end_to_end_count_and_ordinal(corpus, monkeypatch):
    from types import SimpleNamespace
    from cobol_rag.config import AppConfig
    import cobol_rag.query as query_module
    specs = iter([
        {'operator': 'aggregate', 'capability': 'variable_inventory', 'fields': ['count']},
        {'operator': 'list', 'capability': 'variable_inventory', 'fields': ['name'], 'limit': 1, 'selection_evidence': 'first one'},
        {'operator': 'describe', 'capability': 'variable_access', 'fields': ['type'], 'reference_index': 1},
    ])
    class Model:
        def chat(self, messages, **kwargs):
            assert messages[0].role.value == 'system'
            assert kwargs['format']['oneOf'][0]['properties']['query_spec']['required'] == ['operator', 'capability']
            return SimpleNamespace(message=SimpleNamespace(content=json.dumps({'route': 'technical', 'query_spec': next(specs)})))
    monkeypatch.setattr(query_module, 'build_llm', lambda *args, **kwargs: Model())
    monkeypatch.setattr(query_module, 'write_answer_trace', lambda *args, **kwargs: None)
    state = SessionState()
    answers = []
    for question in ('How many variables in DEMO?', 'List the first one.', 'What type is the first one?'):
        result = query_module.answer_query(question, AppConfig(), session_state=state, target_program='DEMO')
        answers.append(result.answer)
        state.update(result.scope, [], result.plan.as_dict())
    assert answers[0] == '2'
    assert '- GROUP-A' in answers[1] and '- ITEM-A' not in answers[1]
    assert 'Group item' in answers[2], (result.debug.get('validation'), result.plan.response_contract, result.plan.tasks)
