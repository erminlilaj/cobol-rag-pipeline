from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace
import json

import pytest

from cobol_rag import progress


@pytest.fixture(autouse=True)
def clean_activity():
    with progress._lock:
        progress._runs.clear()
    yield
    with progress._lock:
        progress._runs.clear()


def test_activity_is_session_owned_incremental_bounded_and_context_local():
    with progress.track('owner', 'request'):
        for _ in range(250):
            progress.emit('tool_start', 'Reading evidence.', tool='source', scope={'program': 'DEMO'})
        snapshot = progress.snapshot('owner', 'request')
        assert len(snapshot['events']) == 200 and snapshot['truncated']
        assert progress.snapshot('other-owner', 'request') is None
        cursor = snapshot['last_sequence']
        assert progress.snapshot('owner', 'request', cursor)['events'] == []
    assert progress.snapshot('owner', 'request')['status'] == 'finished'
    after = progress.snapshot('owner', 'request', cursor)
    assert len(after['events']) == 1 and after['events'][0]['kind'] == 'finished'
    progress.emit('outside', 'Must not attach to old request.')
    assert progress.snapshot('owner', 'request')['last_sequence'] == after['last_sequence']
    with pytest.raises(progress.ProgressConflict):
        with progress.track('owner', 'request'):
            pass


def test_errors_close_the_feed_and_records_are_expired(monkeypatch):
    with pytest.raises(RuntimeError):
        with progress.track('owner', 'request'):
            raise RuntimeError('PRIVATE error text is not published')
    result = progress.snapshot('owner', 'request')
    assert result['status'] == 'error'
    assert 'PRIVATE' not in str(result)
    monkeypatch.setattr(progress, 'TTL', -1)
    assert progress.snapshot('owner', 'request') is None


def test_file_events_distinguish_actual_reads_cache_and_indexed_source_views(tmp_path, monkeypatch):
    from cobol_rag.investigation_tools import EvidenceTools
    from cobol_rag.config import AppConfig
    monkeypatch.setattr('cobol_rag.investigation_tools.artifacts.analyzed_programs', lambda: ('DEMO',))
    monkeypatch.setattr(EvidenceTools, 'root', lambda self, p: tmp_path)
    artifact = tmp_path / 'program.summary.json'
    artifact.write_text(json.dumps({'content': 'overview'}))
    source = tmp_path / 'program.source_lines.jsonl'
    source.write_text(json.dumps(dict(program='DEMO', source_file='DEMO.CBL', line=7,
                                      paragraph='MAIN', text="CALL 'SERVICE'.")) + '\n')
    with progress.track('owner', 'files'):
        tools = EvidenceTools(AppConfig())
        tools.load('DEMO', 'program.summary.json')
        tools.load('DEMO', 'program.summary.json')
        result = tools.execute('source', dict(program='DEMO', paragraph='MAIN'))
    events = progress.snapshot('owner', 'files')['events']
    assert result['rows'][0]['text'] == "CALL 'SERVICE'."
    assert sum(e['kind'] == 'file_open' and e['details']['path'] == str(artifact) for e in events) == 1
    assert any(e['kind'] == 'file_cached' and e['details']['path'] == str(artifact) for e in events)
    assert any(e['kind'] == 'file_open' and e['details']['path'] == str(source) for e in events)
    view = next(e for e in events if e['kind'] == 'source_selection')
    assert view['details']['source_file'] == 'DEMO.CBL' and view['details']['first_line'] == 7
    assert 'not a fresh read' in view['details']['storage']
    assert not any(e['kind'] == 'file_open' and e['details']['path'].endswith('DEMO.CBL') for e in events)


def test_real_api_exposes_activity_while_the_answer_is_still_running(monkeypatch):
    from fastapi.testclient import TestClient
    from cobol_rag import api
    from cobol_rag.query import QueryAnswer
    from cobol_rag.config import AppConfig
    started, finish = Event(), Event()
    class Session:
        config = AppConfig()
        state = SimpleNamespace(as_dict=lambda: {})
        def context_snapshot(self):
            return {'memory_enabled': False}
        def ask(self, message, target_program=None):
            progress.emit('tool_start', 'Reading evidence.', tool='source', scope={'program': 'DEMO', 'paragraph': 'MAIN'})
            started.set()
            assert finish.wait(5)
            return QueryAnswer(question=message, answer='Supported answer.', sources=[], debug={})
    monkeypatch.setattr(api, 'get_chat_session', lambda *a: Session())
    client = TestClient(api.app)
    with ThreadPoolExecutor(max_workers=1) as workers:
        future = workers.submit(client.post, '/api/chat', headers={'X-Session-ID': 'owner'},
                                json={'message': 'Explain DEMO', 'request_id': 'run-1'})
        try:
            assert started.wait(5)
            response = client.get('/api/chat/progress/run-1', headers={'X-Session-ID': 'owner'})
            assert response.status_code == 200
            assert response.json()['status'] == 'running'
            assert any(event['details'].get('tool') == 'source' for event in response.json()['events'])
            assert not future.done()
            assert client.get('/api/chat/progress/run-1', headers={'X-Session-ID': 'other'}).status_code == 404
        finally:
            finish.set()
        assert future.result(timeout=5).status_code == 200
    assert client.get('/api/chat/progress/run-1', headers={'X-Session-ID': 'owner'}).json()['status'] == 'finished'
    assert client.post('/api/chat', headers={'X-Session-ID': 'owner'},
                       json={'message': 'Again', 'request_id': 'run-1'}).status_code == 409
    assert client.post('/api/chat', json={'message': 'No owner', 'request_id': 'run-2'}).status_code == 400
