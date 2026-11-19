"""Bounded session-owned activity events, not model reasoning or draft answers."""
from collections import OrderedDict, deque
from contextlib import contextmanager
from contextvars import ContextVar
from threading import RLock
import time

_active = ContextVar('rag_activity', default=None)
_lock = RLock()
_runs = OrderedDict()
MAX_RUNS = 256
MAX_EVENTS = 200
TTL = 1800


class ProgressConflict(ValueError):
    pass


def _bounded(value, depth=0):
    if depth > 4:
        return '(details omitted)'
    if isinstance(value, str):
        return value[:300] + ('…' if len(value) > 300 else '')
    if isinstance(value, dict):
        return {str(key)[:60]: _bounded(item, depth + 1) for key, item in list(value.items())[:12]}
    if isinstance(value, (list, tuple)):
        preview = [_bounded(item, depth + 1) for item in value[:12]]
        return preview if len(value) <= 12 else {'preview': preview, 'total': len(value), 'truncated': True}
    return value if value is None or type(value) in {int, float, bool} else '(details omitted)'


def emit(kind, message, **details):
    run = _active.get()
    if run is None:
        return
    # Explicit callers supply only executed action metadata, never prompts,
    # raw model text, candidate answers or chain-of-thought.
    with _lock:
        run['sequence'] += 1
        run['events'].append(dict(sequence=run['sequence'], kind=kind, message=message[:300],
                                  elapsed_ms=round((time.monotonic() - run['started']) * 1000),
                                  details=_bounded(details)))
        run['updated'] = time.monotonic()


@contextmanager
def file_read(path, *, program=None, role='analysis artifact'):
    """Wrap an actual read only; cache reuse gets a different event."""
    if path is None or _active.get() is None:
        yield
        return
    started = time.monotonic()
    try:
        size = path.stat().st_size if path.is_file() else None
    except OSError:
        size = None
    emit('file_open', 'Opening file.', path=str(path), program=program, role=role,
         bytes=size)
    try:
        yield
    except Exception:
        emit('file_error', 'File could not be read.', path=str(path), program=program)
        raise
    else:
        emit('file_read', 'Finished reading file.', path=str(path), program=program, role=role,
             duration_ms=round((time.monotonic() - started) * 1000, 2))


@contextmanager
def track(owner, request_id):
    if not request_id:
        yield
        return
    with _lock:
        now = time.monotonic()
        for key, value in list(_runs.items()):
            if now - value['updated'] > TTL and value['status'] != 'running':
                del _runs[key]
        key = (owner, request_id)
        if key in _runs:
            raise ProgressConflict('This activity request ID has already been used.')
        while len(_runs) >= MAX_RUNS:
            victim = next((key for key, value in _runs.items() if value['status'] != 'running'), None)
            if victim is None:
                raise ProgressConflict('Too many active activity feeds; try again later.')
            del _runs[victim]
        run = dict(started=now, updated=now, status='running', sequence=0, events=deque(maxlen=MAX_EVENTS))
        _runs[key] = run
    token = _active.set(run)
    emit('queued', 'Request received; waiting for this conversation to be available.')
    try:
        yield
    except BaseException:
        emit('error', 'Request ended with an error.')
        with _lock:
            run['status'] = 'error'
        raise
    else:
        emit('finished', 'Request processing finished.')
        with _lock:
            run['status'] = 'finished'
    finally:
        _active.reset(token)


def snapshot(owner, request_id, after=0):
    with _lock:
        run = _runs.get((owner, request_id))
        if run is None or time.monotonic() - run['updated'] > TTL and run['status'] != 'running':
            return None
        return dict(status=run['status'], events=[dict(event) for event in run['events'] if event['sequence'] > after],
                    last_sequence=run['sequence'], truncated=bool(run['events'] and after < run['events'][0]['sequence'] - 1))
