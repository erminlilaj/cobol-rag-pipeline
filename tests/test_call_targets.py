from copy import deepcopy
import json

import pytest

from cobol_rag.call_targets import resolve_targets
from cobol_rag.config import AppConfig
from cobol_rag.investigation_tools import EvidenceTools, ToolError


def example(prefix=None, statement='CALL WNOMEPGM USING WPD1DF.'):
    lines = ['CALL-PD2DF.', *(prefix or ["MOVE 'PD2DF' TO WNOMEPGM."]), statement,
             "IF PD1DF-RETURN NOT = '0'", 'GO TO ABEND00.']
    call_line = len(lines) - 2
    call = dict(caller='PDHVAL11', target='WNOMEPGM', call_type='CALL', paragraph='CALL-PD2DF',
                line_start=call_line, line_end=call_line, statement=statement, parameters=['WPD1DF'])
    rows = [dict(program='PDHVAL11', source_file='PDHVAL11.CBL', paragraph='CALL-PD2DF',
                 line=i, text=text) for i, text in enumerate(lines, 1)]
    return call, rows


def test_source_move_resolves_target_not_parameter_or_paragraph():
    call, rows = example()
    original = deepcopy(call)
    found = resolve_targets([call], rows)[0]
    assert call == original
    assert found['target'] == 'PD2DF'
    assert found['target_variable'] == 'WNOMEPGM'
    assert found['target_kind'] == 'resolved_variable'
    assert found['parameters'] == ['WPD1DF']
    assert found['target_resolution']['assignment_source'][0]['line'] == 2
    assert found['target_resolution']['call_source']['line'] == call['line_start']


@pytest.mark.parametrize('prefix', [
    ['MOVE "ARBITRARY-SVC" TO WNOMEPGM.'],
    ["move 'ARBITRARY-SVC'", 'to WNOMEPGM.'],
    ["MOVE 'OLD' TO WNOMEPGM.", "MOVE 'ARBITRARY-SVC' TO WNOMEPGM."],
    ["MOVE 'ARBITRARY-SVC' TO WNOMEPGM.", '* MOVE OTHER TO WNOMEPGM.', 'SKIP1'],
])
def test_representation_and_unseen_names(prefix):
    call, rows = example(prefix)
    assert resolve_targets([call], rows)[0]['target'] == 'ARBITRARY-SVC'


@pytest.mark.parametrize('prefix', [
    ["IF FLAG = 'Y'", "MOVE 'GUESS' TO WNOMEPGM."],
    ["IF FLAG = 'Y'", "MOVE 'GUESS' TO WNOMEPGM", 'END-IF.'],
    ["MOVE 'GUESS' TO WNOMEPGM.", 'MOVE OTHER TO WNOMEPGM.'],
    ["MOVE 'GUESS' TO WNOMEPGM.", 'INITIALIZE AREA.'],
    ["MOVE 'GUESS' TO WNOMEPGM.", 'CALL OTHER USING AREA.'],
    ["MOVE 'GUESS' TO WNOMEPGM.", 'PERFORM SET-TARGET.'],
    ["* MOVE 'GUESS' TO WNOMEPGM."],
    ["MOVE 'GUESS' TO DIFFERENT-VARIABLE."],
])
def test_ambiguous_or_unsafe_paths_stay_unresolved(prefix):
    call, rows = example(prefix)
    found = resolve_targets([call], rows)[0]
    assert found['target'] == 'WNOMEPGM'
    assert found['target_kind'] == 'unresolved_variable'


@pytest.mark.parametrize('damage', ['different_paragraph', 'different_file', 'missing_line', 'wrong_call'])
def test_proof_must_match_exact_source_call_site(damage):
    call, rows = example()
    if damage == 'different_paragraph':
        rows[1]['paragraph'] = 'OTHER'
    elif damage == 'different_file':
        rows[1]['source_file'] = 'COPYBOOK.CPY'
    elif damage == 'missing_line':
        rows = rows[1:]
    else:
        rows[call['line_start'] - 1]['text'] = 'CALL OTHER USING AREA.'
    assert resolve_targets([call], rows)[0]['target_kind'] == 'unresolved_variable'


def test_fixed_format_and_literal_call_are_preserved():
    call, rows = example()
    for row in rows:
        row['text'] = f"{row['line']:06} " + row['text']
    assert resolve_targets([call], rows)[0]['target'] == 'PD2DF'
    literal = dict(call, target='REAL', statement="CALL 'REAL' USING AREA.")
    found = resolve_targets([literal], [])[0]
    assert found['target'] == 'REAL' and found['target_kind'] == 'literal'


@pytest.fixture
def tools(monkeypatch, tmp_path):
    call, rows = example()
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('PDHVAL11', 'PD2DF'))
    result = EvidenceTools(AppConfig())
    monkeypatch.setattr(result, 'root', lambda p: tmp_path)
    monkeypatch.setattr(result, 'path', lambda p, a: tmp_path / a)
    monkeypatch.setattr(result, 'load', lambda p, a: dict(calls=[call]) if p == 'PDHVAL11' else dict(calls=[]))
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts._read_source_lines', lambda root, p: deepcopy(rows))
    return result


def test_real_tool_rows_resolve_before_target_filters(tools):
    outgoing = tools.execute('callees', dict(programs=['PDHVAL11'], target='PD2DF'))
    assert outgoing['total_matches'] == 1
    incoming = tools.execute('callers', dict(target='PD2DF'))
    assert incoming['total_matches'] == 1
    assert incoming['rows'][0]['caller'] == 'PDHVAL11'
    assert incoming['rows'][0]['parameters'] == ['WPD1DF']
    assert outgoing['rows'][0]['name'] == 'PD2DF'
    assert outgoing['rows'][0]['parameters'] == ['WPD1DF']
    incoming = tools.execute('callers', dict(programs=['PDHVAL11'], target='PD2DF'))
    assert incoming['total_matches'] == 1
    assert incoming['relation']['direction'] == 'incoming_to_target'


def test_missing_source_does_not_invent_target(tools, monkeypatch):
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts._read_source_lines', lambda *a: [])
    found = tools.execute('callees', dict(programs=['PDHVAL11']))
    assert found['rows'][0]['target'] == 'WNOMEPGM'
    assert found['rows'][0]['target_kind'] == 'unresolved_variable'


def test_corpus_connections_use_same_resolution_and_keep_published_origin(tools, monkeypatch, tmp_path):
    call, _ = example()
    graph = dict(schema_version=1, edges=[dict(id='old-id', kind='program_call', source='PDHVAL11',
        target='WNOMEPGM', program='PDHVAL11', source_artifact='architecture.call_parameters.json',
        evidence={key: call[key] for key in ('statement', 'call_type', 'line_start', 'paragraph')})])
    (tmp_path / 'corpus.relationships.json').write_text(json.dumps(graph))
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.find_final_scripts_root', lambda: tmp_path)
    result = tools.execute('connections', dict(where=[dict(field='kind', op='eq', value='program_call'),
                                                     dict(field='target', op='eq', value='PD2DF')]))
    assert result['total_matches'] == 1
    row = result['rows'][0]
    assert row['target'] == 'PD2DF' and row['recorded_target'] == 'WNOMEPGM'
    assert row['evidence']['target_analyzed']


def test_same_variable_can_resolve_to_different_programs_at_different_sites():
    first, rows = example()
    second, other = example(["MOVE 'OTHER-SERVICE' TO WNOMEPGM."])
    second.update(paragraph='SECOND-CALL', line_start=second['line_start'] + 10)
    for row in other:
        row['line'] += 10
        row['paragraph'] = 'SECOND-CALL'
        if row['text'] == 'CALL-PD2DF.':
            row['text'] = 'SECOND-CALL.'
    found = resolve_targets([first, second], rows + other)
    assert [row['target'] for row in found] == ['PD2DF', 'OTHER-SERVICE']


def test_cics_variable_program_uses_same_proof_without_changing_call_type():
    statement = 'EXEC CICS LINK PROGRAM(WNOMEPGM) COMMAREA(AREA) END-EXEC.'
    call, rows = example(statement=statement)
    call.update(call_type='CICSLINK', commarea='AREA')
    found = resolve_targets([call], rows)[0]
    assert found['target'] == 'PD2DF'
    assert found['call_type'] == 'CICSLINK' and found['commarea'] == 'AREA'
