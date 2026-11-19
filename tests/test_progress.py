from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

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
