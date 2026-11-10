from threading import Event

import chromadb
import pytest
from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from typer.testing import CliRunner

from cobol_rag.cli import app
from cobol_rag.config import AppConfig, PathConfig
from cobol_rag.sync import apply_sync_plan, build_sync_plan
from cobol_rag.sync_progress import SyncProgress


def test_heartbeat_reports_blocked_operation_without_completed_changes():
    lines = []
    heartbeat = Event()

    def capture(line):
        lines.append(line)
        if "elapsed" in line and "embed via Ollama" in line:
            heartbeat.set()

    with SyncProgress(capture, interval=0.01) as progress:
        with progress.phase("apply changes", total=2):
            with progress.measure("embed via Ollama"):
                assert heartbeat.wait(2), lines
            progress.advance()
        progress.summary()
    assert any("0/2 changes" in line and "embed via Ollama" in line for line in lines)
    assert any("done in" in line and "1/2 changes" in line for line in lines)
    assert progress.seconds["embed via Ollama"] > 0


def test_phase_failure_does_not_report_success():
    lines = []
    with pytest.raises(RuntimeError, match="offline"):
        with SyncProgress(lines.append) as progress:
            with progress.phase("open Chroma"):
                raise RuntimeError("offline")
    assert any("open Chroma: failed" in line for line in lines)
    assert not any("open Chroma: done" in line for line in lines)


def test_apply_measures_embedding_chroma_and_unchanged_documents(tmp_path, monkeypatch):
    # Use a real disposable Chroma index; replace only the remote embedding
    # transport so the test can run without an Ollama server.
    monkeypatch.setattr(Settings, "_embed_model", Settings._embed_model)
    monkeypatch.setattr(Settings, "_llm", Settings._llm)
    embeddings = []

    def embed(_self, texts):
        embeddings.extend(texts)
        return [[1.0, 0.5, 0.25] for _ in texts]

    monkeypatch.setattr(OllamaEmbedding, "_get_text_embeddings", embed)
    config = AppConfig(paths=PathConfig(
        chroma_dir=tmp_path / "chroma", inbox_dir=tmp_path / "inbox",
        manifest_dir=tmp_path / "manifests",
    ))
    config.paths.inbox_dir.mkdir()
    (config.paths.inbox_dir / "sample.cbl").write_text("IDENTIFICATION DIVISION.", encoding="utf-8")
    lines = []
    with SyncProgress(lines.append) as progress:
        plan = build_sync_plan(config, dry_run=False, progress=progress)
        assert plan.count("add") == 1
        apply_sync_plan(config, plan, progress=progress)
    assert len(embeddings) == 1
    assert any("1/1 changes" in line for line in lines)
    assert all(progress.seconds[label] > 0 for label in (
        "read existing manifest", "load incoming documents", "open Chroma", "embed via Ollama",
        "write vectors to Chroma", "delete old vectors from Chroma", "save manifest",
    ))
    client = chromadb.PersistentClient(path=str(config.paths.chroma_dir))
    assert client.get_collection(config.index.collection).count() == 1
    assert plan.manifest_path.is_file()

    with SyncProgress(lines.append) as progress:
        plan = build_sync_plan(config, dry_run=False, progress=progress)
        assert plan.count("skip") == 1
        apply_sync_plan(config, plan, progress=progress)
    assert len(embeddings) == 1
    assert "embed via Ollama" not in progress.seconds
    assert any("0/0 changes" in line for line in lines)


def test_cli_dry_run_reports_timings_without_writes_and_can_be_quiet(tmp_path):
    target = tmp_path / "example.txt"
    target.write_text("Example source text.", encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"paths:\n  chroma_dir: {tmp_path / 'chroma'}\n  manifest_dir: {tmp_path / 'manifests'}\n",
        encoding="utf-8",
    )
    runner = CliRunner()
    args = ["sync", str(target), "--dry-run", "--config", str(config)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "[index sync] Plan: 1 add" in result.stderr
    assert "Timings" in result.stderr
    assert "open Chroma" not in result.stderr
    result = runner.invoke(app, [*args, "--no-progress"])
    assert result.exit_code == 0, result.output
    assert "[index sync]" not in result.output
    assert not (tmp_path / "chroma").exists()
    assert not (tmp_path / "manifests").exists()
