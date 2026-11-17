"""Exercise actual session/registry code without loading vector or model SDKs."""
import ast
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Event, RLock
from types import ModuleType, SimpleNamespace

import pytest

from cobol_rag.config import AppConfig
from cobol_rag.scope import SessionState, QueryScope

SOURCE = Path(__file__).resolve().parents[1] / 'src' / 'cobol_rag'


@pytest.fixture
def lifecycle(monkeypatch):
    # Keep production data classes and method bodies; substitute only the
    # expensive answer_query boundary, as concurrency must not call an LLM.
    module = ModuleType('_conversation_lifecycle_under_test')
    monkeypatch.setitem(__import__('sys').modules, module.__name__, module)
    module.__dict__.update(SessionState=SessionState, QueryScope=QueryScope)
    query = ast.parse((SOURCE / 'query.py').read_text())
    definitions = [node for node in query.body if getattr(node, 'name', '') == 'QueryAnswer' or
                   isinstance(node, ast.Assign) and any(getattr(t, 'id', '') == 'REJECTED_EXECUTION_MODES' for t in node.targets)]
    chat = ast.parse((SOURCE / 'chat.py').read_text())
    removed = {'cobol_rag.query', 'cobol_rag.retrieve'}
    chat.body = [node for node in chat.body if not (isinstance(node, ast.ImportFrom) and node.module in removed)]
    # Put QueryAnswer after the imports and before ChatSession.
    at = next(i for i, node in enumerate(chat.body) if isinstance(node, ast.ClassDef))
    chat.body[at:at] = definitions
    exec(compile(chat, str(SOURCE / 'chat.py'), 'exec'), module.__dict__)
    config = replace(AppConfig(), investigation=replace(AppConfig().investigation, enabled=True, memory_enabled=True))
    module.answer_query = lambda **kw: module.QueryAnswer(
        question=kw['question'], answer='Supported', sources=[], route='technical',
        scope=QueryScope(program='A'), debug={'answer_status': 'complete'})
    module.config = config
    return module


def test_default_and_disabled_memory_never_forward_or_store_context(lifecycle):
    m = lifecycle
    assert AppConfig().investigation.memory_enabled is False
    session = m.ChatSession(config=replace(m.config, investigation=replace(m.config.investigation, memory_enabled=False)))
    session.state.investigation_memory = {'last_exchange': {'answer': 'Old claim'}}
    session.state.current_program = 'OLD'
    seen = []
    def answer(**kw):
        seen.append((kw['conversation_history'], kw['session_state']))
        return m.QueryAnswer(question=kw['question'], answer='Supported', sources=[])
    m.answer_query = answer
    for question in ('First?', 'Second?'):
        session.ask(question)
    assert all(history is None and state.investigation_memory == {} and state.current_program is None
               for history, state in seen)
    assert session.state.investigation_memory == {}
    assert session._history_context() is None
    assert not session.context_snapshot()['memory_enabled']


def test_shipped_configuration_still_disables_memory(monkeypatch):
    from cobol_rag.config import load_config
    monkeypatch.delenv('COBOL_RAG_MEMORY_ENABLED', raising=False)
    config = load_config(SOURCE.parents[1] / 'config' / 'default.yaml')
    assert config.investigation.memory_enabled is False


def test_partial_and_failed_turns_are_labelled_without_assistant_prose(lifecycle):
    m = lifecycle
    session = m.ChatSession(config=m.config)
    m.answer_query = lambda **kw: m.QueryAnswer(question=kw['question'], answer='Private partial prose',
        sources=[], route='technical', debug={'answer_status': 'partial'})
    session.ask('Explain A')
    m.answer_query = lambda **kw: m.QueryAnswer(question=kw['question'], answer='Rejected prose',
        sources=[], route='technical', debug={'status': 'rejected', 'validation': {'passed': False}})
    session.ask('Explain B')
    history = __import__('json').loads(session._history_context())
    assert [item['status'] for item in history] == ['partial', 'unresolved']
    assert all('answer' not in item for item in history)
    assert session.state.current_program is None  # Neither answer established a program scope.
    assert session.state.investigation_memory['unresolved_turn']['question'] == 'Explain B'


def test_concurrent_questions_see_committed_previous_turn(lifecycle):
    m = lifecycle
    session = m.ChatSession(config=m.config)
    started, release, second_started = Event(), Event(), Event()
    histories = []
    def answer(**kw):
        histories.append(kw['conversation_history'])
        if kw['question'] == 'First':
            started.set()
            assert release.wait(3)
        else:
            second_started.set()
        return m.QueryAnswer(question=kw['question'], answer='Supported', sources=[])
    m.answer_query = answer
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(session.ask, 'First')
        assert started.wait(1)
        second = workers.submit(session.ask, 'Second')
        assert not second_started.wait(.03)
        release.set()
        assert first.result(2).execution_mode != 'cancelled'
        assert second.result(2).execution_mode != 'cancelled'
    assert [turn.user for turn in session.turns] == ['First', 'Second']
    assert 'First' in histories[1]
    assert session.revision == 2


@pytest.mark.parametrize('operation', ['cancel', 'reset'])
def test_cancel_or_reset_discards_inflight_and_old_queued_questions(lifecycle, operation):
    m = lifecycle
    session = m.ChatSession(config=m.config)
    started, release, queued = Event(), Event(), Event()
    calls = []
    def answer(**kw):
        calls.append(kw['question'])
        # Mutation must affect the snapshot, not live state, even on cancel.
        kw['session_state'].current_program = 'UNCOMMITTED'
        started.set()
        assert release.wait(3)
        return m.QueryAnswer(question=kw['question'], answer='Supported', sources=[])
    m.answer_query = answer
    # Signal when the second request enters the serialized path. Holding the
    # state lock lets the test establish its enrollment before cancellation.
    class EnrollLock:
        def __enter__(self):
            if calls:
                queued.set()
            self.lock.acquire()
        def __exit__(self, *args):
            self.lock.release()
        lock = __import__('threading').Lock()
    session._request_lock = EnrollLock()
    with ThreadPoolExecutor(max_workers=2) as workers:
        first = workers.submit(session.ask, 'First')
        assert started.wait(1)
        second = workers.submit(session.ask, 'Obsolete follow-up')
        assert queued.wait(1)
        getattr(session, operation)()
        release.set()
        assert first.result(2).execution_mode == 'cancelled'
        assert second.result(2).execution_mode == 'cancelled'
    assert calls == ['First'] and session.turns == []
    assert session.state.current_program is None
    assert not session.is_busy()


def test_reset_rotates_server_context_and_new_request_can_commit(lifecycle):
    m = lifecycle
    session = m.ChatSession(config=m.config)
    previous = session.ask('First').debug['session_context']
    session.reset()
    current = session.ask('New topic').debug['session_context']
    assert current['id'] != previous['id'] and current['revision'] == 1
    assert [turn.user for turn in session.turns] == ['New topic']


@pytest.fixture
def registry(lifecycle):
    m = lifecycle
    class HTTPError(Exception):
        def __init__(self, status_code, detail):
            self.status_code, self.detail = status_code, detail
    clock = SimpleNamespace(now=0)
    m.__dict__.update(HTTPException=HTTPError, CONFIG_PATH='unused',
        Request=object,
        _MAX_CHAT_SESSIONS=2, _CHAT_SESSION_TTL_SECONDS=60,
        _SESSION_ID_PATTERN=__import__('re').compile(r'^[A-Za-z0-9._:-]{1,128}$'),
        _chat_sessions=OrderedDict(), _chat_sessions_lock=RLock(),
        load_config=lambda path: m.config, time=SimpleNamespace(monotonic=lambda: clock.now))
    tree = ast.parse((SOURCE / 'api.py').read_text())
    functions = [node for node in tree.body if getattr(node, 'name', '') in
                 {'_normalize_session_id', '_request_session_id', 'get_chat_session'}]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'api.py', 'exec'), m.__dict__)
    return m, clock


def test_registry_isolates_missing_ids_and_rejects_invalid_ids(registry):
    m, clock = registry
    assert m.get_chat_session() is not m.get_chat_session()
    assert m.get_chat_session('A') is m.get_chat_session('A')
    assert m.get_chat_session('A') is not m.get_chat_session('B')
    with pytest.raises(m.HTTPException) as error:
        m.get_chat_session('not a valid id')
    assert error.value.status_code == 400


def test_registry_expiry_changes_context_but_does_not_expire_active_requests(registry):
    m, clock = registry
    old = m.get_chat_session('A')
    clock.now = 61
    fresh = m.get_chat_session('A')
    assert fresh.context_id != old.context_id
    fresh._pending = 1
    clock.now = 122
    assert m.get_chat_session('A') is fresh


def test_registry_never_evicts_busy_conversations(registry):
    m, clock = registry
    first, second = m.get_chat_session('A'), m.get_chat_session('B')
    first._pending = second._pending = 1
    with pytest.raises(m.HTTPException) as error:
        m.get_chat_session('C')
    assert error.value.status_code == 503
    assert m.get_chat_session('A') is first
    second._pending = 0
    m.get_chat_session('C')
    assert 'A' in m._chat_sessions and 'B' not in m._chat_sessions
