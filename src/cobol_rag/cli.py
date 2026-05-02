from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from cobol_rag.config import load_config

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


if __name__ == "__main__":
    app()
