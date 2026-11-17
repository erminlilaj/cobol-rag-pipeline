"""Bounded-memory access to existing JSON manifests; no format migration."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import ijson


def source_records(path: Path):
    if path.exists():
        with path.open('rb') as handle:
            # Validate the envelope without materializing sources. A malformed
            # existing manifest must never become a silently empty inventory.
            events = ijson.parse(handle)
            if next(events, None) != ('', 'start_map', None):
                raise ValueError('Manifest must be a JSON object.')
            for prefix, event, _ in events:
                if prefix == 'sources':
                    if event != 'start_map':
                        raise ValueError('Manifest sources must be a JSON object.')
                    break
            handle.seek(0)
            for source_id, entry in ijson.kvitems(handle, 'sources'):
                if not isinstance(entry, dict):
                    raise ValueError(f'Invalid manifest source: {source_id}')
                yield source_id, entry


def bm25_reference(path: Path):
    if not path.exists():
        return None
    with path.open('rb') as handle:
        return next(ijson.items(handle, 'bm25_index_path'), None)


def atomic_write_manifest(path: Path, collection: str, records, bm25=None):
    """Preserve the old manifest on failure; hold at most one entry in memory.

    Writers must remain serialized, as with the vector store itself. The temp
    file is on the same filesystem so replace does not expose truncated JSON.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            handle.write('{"collection":')
            json.dump(collection, handle)
            if bm25 is not None:
                handle.write(',"bm25_index_path":')
                json.dump(str(bm25), handle)
            handle.write(',"sources":{')
            first = True
            for source_id, entry in records:
                if not first:
                    handle.write(',')
                first = False
                json.dump(source_id, handle)
                handle.write(':')
                json.dump(entry, handle, ensure_ascii=True, separators=(',', ':'))
            handle.write('}}\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
