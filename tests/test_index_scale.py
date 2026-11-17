"""Scale/recovery tests: real local Chroma, deterministic offline embeddings."""
import json
import tracemalloc
import ijson
import pytest
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding

from cobol_rag.config import AppConfig, PathConfig
from cobol_rag.manifest_store import atomic_write_manifest, source_records
from cobol_rag.sync import (SyncItem, SyncPlan, apply_sync_plan, build_sync_plan,
                            read_manifest, write_manifest, _obsolete_items, ManifestEntry)


def entry(source_id, path):
    return dict(source_id=source_id, source_path=str(path), source_format='test', content_hash='old')


def plan_for(path, *items):
    return SyncPlan(collection='test', inbox_dir=path.parent, manifest_path=path,
                    dry_run=False, items=list(items))


def test_streamed_merge_keeps_other_programs_and_bm25(tmp_path):
    path = tmp_path / 'corpus.json'
    original = dict(collection='test', bm25_index_path='/indexes/lexical.json',
                    sources={key: entry(key, tmp_path / key) for key in ('keep', 'update', 'remove')})
    path.write_text(json.dumps(original))
    items = [SyncItem('update', 'update', '/new/path', 'test', 'new'),
             SyncItem('add', 'add', '/new/item', 'test', 'added'),
             SyncItem('remove', 'remove', '/old', 'test', 'old')]
    write_manifest(path, plan_for(path, *items))
    result = json.loads(path.read_text())
    assert result['sources']['keep'] == original['sources']['keep']
    assert set(result['sources']) == {'keep', 'update', 'add'}
    assert result['sources']['update']['content_hash'] == 'new'
    assert result['bm25_index_path'] == original['bm25_index_path']
    assert read_manifest(path)['add'].content_hash == 'added'


def test_manifest_failure_preserves_old_file_and_cleans_temp(tmp_path):
    path = tmp_path / 'corpus.json'
    old = '{"collection":"test","sources":{"old":{}}}'
    path.write_text(old)
    def broken():
        yield 'new', entry('new', '/new')
        raise OSError('Disk write failed')
    with pytest.raises(OSError, match='Disk write failed'):
        atomic_write_manifest(path, 'test', broken())
    assert path.read_text() == old
    assert list(tmp_path.glob('*.tmp')) == []


@pytest.mark.parametrize('contents', ['[]', '{"sources":[]}', '{"sources":{"x":null}}', '{"sources":{'])
def test_invalid_manifest_is_not_overwritten(tmp_path, contents):
    path = tmp_path / 'corpus.json'
    path.write_text(contents)
    with pytest.raises((ValueError, ijson.JSONError)):
        write_manifest(path, plan_for(path))
    assert path.read_text() == contents


def test_filtered_manifest_reads_and_rewrite_have_bounded_memory(tmp_path):
    path = tmp_path / 'corpus.json'
    atomic_write_manifest(path, 'test', ((str(i), entry(str(i), f'/other/{i // 100}/source.jsonl'))
                                       for i in range(20000)))
    tracemalloc.start()
    try:
        selected = read_manifest(path, source_ids={'19999'}, sync_root=tmp_path / 'selected')
        assert set(selected) == {'19999'}
        write_manifest(path, plan_for(path, SyncItem('add', 'new', '/new', 'test', 'new')))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 5 * 1024 * 1024, peak
    assert sum(1 for _ in source_records(path)) == 20001


def test_obsolete_path_checks_reuse_repeated_source_paths(tmp_path, monkeypatch):
    checks = []
    def under(path, root):
        checks.append(path)
        return path.name == 'selected.jsonl'
    monkeypatch.setattr('cobol_rag.sync._is_under_path', under)
    manifest = {str(i): ManifestEntry(str(i), '/root/selected.jsonl' if i % 2 else '/elsewhere/other.jsonl', 'test', 'old')
                for i in range(1000)}
    result = _obsolete_items(manifest=manifest, sync_root=tmp_path, current_source_ids=set())
    assert len(result) == 500 and len(checks) == 2


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, '_embed_model', Settings._embed_model)
    monkeypatch.setattr(Settings, '_llm', Settings._llm)
    monkeypatch.setattr(OllamaEmbedding, '_get_text_embeddings', lambda self, texts:
                        [[1.0, 0.5, 0.25] for _ in texts])
    result = AppConfig(paths=PathConfig(chroma_dir=tmp_path / 'chroma', inbox_dir=tmp_path / 'inbox',
                                      manifest_dir=tmp_path / 'manifests'))
    result.paths.inbox_dir.mkdir()
    return result


def test_real_chroma_retry_after_manifest_failure_replaces_partial_vectors(config, monkeypatch):
    from cobol_rag.index import open_index
    source = config.paths.inbox_dir / 'source.txt'
    source.write_text('OLD BODY')
    first = build_sync_plan(config, dry_run=False)
    original_write = write_manifest
    def fail(*args, **kwargs):
        raise OSError('Simulated interruption before manifest commit')
    monkeypatch.setattr('cobol_rag.sync.write_manifest', fail)
    with pytest.raises(OSError):
        apply_sync_plan(config, first)
    assert not first.manifest_path.exists()
    resources = open_index(config)
    assert resources.chroma_collection.count() == 1
    source.write_text('NEW BODY')
    monkeypatch.setattr('cobol_rag.sync.write_manifest', original_write)
    retry = build_sync_plan(config, dry_run=False)
    assert retry.count('update') == 1  # The recovery journal knows vectors may exist.
    apply_sync_plan(config, retry)
    assert resources.chroma_collection.count() == 1
    assert resources.chroma_collection.get(include=['documents'])['documents'] == ['NEW BODY']
    assert build_sync_plan(config).count('skip') == 1


def test_real_chroma_batch_has_one_delete_per_32_documents(config, monkeypatch):
    from chromadb.api.models.Collection import Collection
    from cobol_rag.index import open_index
    deletes = []
    original = Collection.delete
    def counted(self, *args, **kwargs):
        deletes.append(kwargs.get('where'))
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Collection, 'delete', counted)
    for i in range(65):
        (config.paths.inbox_dir / f'{i}.txt').write_text(f'Source {i}')
    plan = build_sync_plan(config, dry_run=False)
    apply_sync_plan(config, plan)
    assert len(deletes) == 3
    assert [len(d['source_id']['$in']) for d in deletes] == [32, 32, 1]
    assert open_index(config).chroma_collection.count() == 65
    again = build_sync_plan(config, dry_run=False)
    assert again.count('skip') == 65
    before = again.manifest_path.stat().st_mtime_ns
    apply_sync_plan(config, again)
    assert len(deletes) == 3
    assert again.manifest_path.stat().st_mtime_ns == before


def test_embedding_failure_does_not_delete_previous_vectors(config, monkeypatch):
    from cobol_rag.index import open_index
    source = config.paths.inbox_dir / 'source.txt'
    source.write_text('ORIGINAL BODY')
    apply_sync_plan(config, build_sync_plan(config, dry_run=False))
    manifest = (config.paths.manifest_dir / f'{config.index.collection}.json')
    old_manifest = manifest.read_bytes()
    source.write_text('CHANGED BODY')
    def fail(self, texts):
        raise RuntimeError('Embedding server unavailable')
    monkeypatch.setattr(OllamaEmbedding, '_get_text_embeddings', fail)
    with pytest.raises(RuntimeError, match='unavailable'):
        apply_sync_plan(config, build_sync_plan(config, dry_run=False))
    assert manifest.read_bytes() == old_manifest
    collection = open_index(config).chroma_collection
    assert collection.get(include=['documents'])['documents'] == ['ORIGINAL BODY']


def test_journal_forces_repair_when_input_reverts_to_old_manifest_hash(config, monkeypatch):
    from cobol_rag.index import open_index
    source = config.paths.inbox_dir / 'source.txt'
    source.write_text('ORIGINAL')
    apply_sync_plan(config, build_sync_plan(config, dry_run=False))
    source.write_text('INTERMEDIATE')
    original_write = write_manifest
    monkeypatch.setattr('cobol_rag.sync.write_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError):
        apply_sync_plan(config, build_sync_plan(config, dry_run=False))
    source.write_text('ORIGINAL')
    monkeypatch.setattr('cobol_rag.sync.write_manifest', original_write)
    retry = build_sync_plan(config, dry_run=False)
    assert retry.count('update') == 1  # Not skip: vectors contain INTERMEDIATE.
    apply_sync_plan(config, retry)
    assert open_index(config).chroma_collection.get(include=['documents'])['documents'] == ['ORIGINAL']
    assert build_sync_plan(config).count('skip') == 1


def test_journal_removes_orphan_addition_missing_from_committed_manifest(config, monkeypatch):
    from cobol_rag.index import open_index
    source = config.paths.inbox_dir / 'orphan.txt'
    source.write_text('ORPHAN')
    original_write = write_manifest
    monkeypatch.setattr('cobol_rag.sync.write_manifest', lambda *a, **k: (_ for _ in ()).throw(OSError('interrupted')))
    with pytest.raises(OSError):
        apply_sync_plan(config, build_sync_plan(config, dry_run=False))
    source.unlink()
    monkeypatch.setattr('cobol_rag.sync.write_manifest', original_write)
    retry = build_sync_plan(config, dry_run=False)
    assert retry.count('remove') == 1
    apply_sync_plan(config, retry)
    assert open_index(config).chroma_collection.count() == 0
