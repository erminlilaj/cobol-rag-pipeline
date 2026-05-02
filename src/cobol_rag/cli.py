from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cobol_rag.config import load_config
from cobol_rag.index import collection_count, open_index
from cobol_rag.loaders import LoaderError, load_path
from cobol_rag.sync import SyncPlan, apply_sync_plan, build_sync_plan

app = typer.Typer(
    help="Flexible local RAG pipeline for COBOL analysis artifacts.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """Run COBOL RAG commands."""


@app.command()
def config(
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Print the active configuration summary."""
    settings = load_config(path)
    table = Table(title="COBOL RAG Configuration")
    table.add_column("Area")
    table.add_column("Setting")
    table.add_column("Value")

    table.add_row("paths", "chroma_dir", str(settings.paths.chroma_dir))
    table.add_row("paths", "inbox_dir", str(settings.paths.inbox_dir))
    table.add_row("paths", "manifest_dir", str(settings.paths.manifest_dir))
    table.add_row("llm", "provider", settings.llm.provider)
    table.add_row("llm", "model", settings.llm.model)
    table.add_row("embedding", "provider", settings.embedding.provider)
    table.add_row("embedding", "model", settings.embedding.model)
    table.add_row("index", "collection", settings.index.collection)
    table.add_row("retrieval", "top_k", str(settings.retrieval.top_k))
    table.add_row("answers", "require_citations", str(settings.answers.require_citations))
    console.print(table)


@app.command("index-info")
def index_info(
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Open the configured LlamaIndex/Chroma index and print a summary."""
    settings = load_config(path)
    resources = open_index(settings)

    table = Table(title="COBOL RAG Index")
    table.add_column("Setting")
    table.add_column("Value")
    table.add_row("chroma_dir", str(settings.paths.chroma_dir))
    table.add_row("collection", settings.index.collection)
    table.add_row("documents", str(collection_count(resources)))
    table.add_row("llm", settings.llm.model)
    table.add_row("embedding", settings.embedding.model)
    console.print(table)


@app.command()
def inspect(
    target: Path = typer.Argument(..., help="File or directory to inspect."),
    loader: str | None = typer.Option(
        None,
        "--loader",
        "-l",
        help="Force a loader by name instead of auto-detecting.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
    preview_chars: int = typer.Option(
        120,
        "--preview-chars",
        min=0,
        help="Number of text preview characters to show.",
    ),
) -> None:
    """Inspect files through general loaders without indexing them."""
    settings = load_config(path)
    try:
        loaded = load_path(target, config=settings, loader_name=loader)
    except LoaderError as error:
        raise typer.BadParameter(str(error)) from error

    summary = Table(title="Inspection Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("target", str(target))
    summary.add_row("documents", str(len(loaded)))
    summary.add_row("indexing", "no")
    console.print(summary)

    detail = Table(title="Loaded Documents")
    detail.add_column("Loader")
    detail.add_column("Source ID")
    detail.add_column("Source Path")
    detail.add_column("Chars", justify="right")
    detail.add_column("Preview")

    for item in loaded:
        document = item.document
        preview = _preview(document.text, preview_chars)
        detail.add_row(
            item.loader_name,
            str(document.metadata.get("source_id", "")),
            str(document.metadata.get("source_path", item.source_path)),
            str(len(document.text)),
            preview,
        )
    console.print(detail)


@app.command()
def sync(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="Preview the sync plan or apply it. Only dry-run is implemented now.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Plan inbox synchronization using general loaders."""
    settings = load_config(path)
    plan = build_sync_plan(settings, dry_run=dry_run)
    if not dry_run:
        apply_sync_plan(settings, plan)
    _print_sync_plan(plan)


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


def _print_sync_plan(plan: SyncPlan) -> None:
    summary = Table(title="Sync Plan")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("collection", plan.collection)
    summary.add_row("inbox", str(plan.inbox_dir))
    summary.add_row("manifest", str(plan.manifest_path))
    summary.add_row("dry_run", str(plan.dry_run))
    summary.add_row("documents", str(plan.total_documents))
    summary.add_row("would_add", str(plan.count("add")))
    summary.add_row("would_update", str(plan.count("update")))
    summary.add_row("would_skip", str(plan.count("skip")))
    summary.add_row("indexing", "no" if plan.dry_run else "yes")
    summary.add_row("manifest_write", "no" if plan.dry_run else "yes")
    console.print(summary)

    detail = Table(title="Sync Items")
    detail.add_column("Action")
    detail.add_column("Source Format")
    detail.add_column("Source ID")
    detail.add_column("Source Path")
    for item in plan.items:
        detail.add_row(
            item.action,
            item.source_format,
            item.source_id,
            item.source_path,
        )
    console.print(detail)


if __name__ == "__main__":
    app()
