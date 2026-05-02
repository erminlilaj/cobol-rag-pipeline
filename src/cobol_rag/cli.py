from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cobol_rag.config import load_config
from cobol_rag.index import collection_count, open_index
from cobol_rag.loaders import LoaderError, load_path

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


def _preview(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return f"{compact[:limit].rstrip()}..."


if __name__ == "__main__":
    app()
