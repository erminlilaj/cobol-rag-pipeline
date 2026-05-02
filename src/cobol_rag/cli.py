from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from cobol_rag.chat import ChatSession
from cobol_rag.config import load_config
from cobol_rag.index import collection_count, open_index
from cobol_rag.loaders import LoaderError, load_path
from cobol_rag.query import QueryAnswer, QueryError, answer_query
from cobol_rag.remove import RemovePlan, apply_remove_plan, build_remove_plan
from cobol_rag.reset import ResetPlan, apply_reset_plan, build_reset_plan
from cobol_rag.retrieve import RetrievalResult, retrieve as retrieve_documents
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
    table.add_row("llm", "context_window", str(settings.llm.context_window))
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
        help="Preview the sync plan or apply it.",
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


@app.command()
def remove(
    source_id: str | None = typer.Option(
        None,
        "--source-id",
        help="Remove one manifest/index entry by normalized source id.",
    ),
    source_path: str | None = typer.Option(
        None,
        "--source-path",
        help="Remove manifest/index entries by original source path.",
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="Preview removal or apply it.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Remove indexed documents by general metadata."""
    settings = load_config(path)
    try:
        plan = build_remove_plan(
            settings,
            source_id=source_id,
            source_path=source_path,
            dry_run=dry_run,
        )
    except ValueError as error:
        raise typer.BadParameter(str(error)) from error

    if not dry_run:
        apply_remove_plan(settings, plan)
    _print_remove_plan(plan)


@app.command()
def reset(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--apply",
        help="Preview reset or apply it.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Reset the configured Chroma collection and its manifest."""
    settings = load_config(path)
    plan = build_reset_plan(settings, dry_run=dry_run)
    if not dry_run:
        apply_reset_plan(settings, plan)
    _print_reset_plan(plan)


@app.command()
def retrieve(
    query: str = typer.Argument(..., help="Question or search text."),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Number of retrieval results to return.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
    preview_chars: int = typer.Option(
        180,
        "--preview-chars",
        min=0,
        help="Number of text preview characters to show.",
    ),
) -> None:
    """Retrieve matching documents without generating an answer."""
    settings = load_config(path)
    results = retrieve_documents(query=query, config=settings, top_k=top_k)
    _print_retrieval_results(query, results, preview_chars)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to answer from indexed sources."),
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Number of retrieved sources to use.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Answer one question using retrieved sources and citations."""
    settings = load_config(path)
    try:
        answer = answer_query(question=question, config=settings, top_k=top_k)
    except QueryError as error:
        raise typer.BadParameter(str(error)) from error
    _print_query_answer(answer)


@app.command()
def chat(
    top_k: int | None = typer.Option(
        None,
        "--top-k",
        min=1,
        help="Number of retrieved sources to use for each turn.",
    ),
    collection: str | None = typer.Option(
        None,
        "--collection",
        help="Override the configured Chroma collection.",
    ),
    once: str | None = typer.Option(
        None,
        "--once",
        help="Ask one chat message and exit. Useful for verification.",
    ),
    path: Path = typer.Option(
        Path("config/default.yaml"),
        "--config",
        "-c",
        help="Path to the YAML config file.",
    ),
) -> None:
    """Start a terminal chat over the indexed sources."""
    settings = load_config(path)
    if collection:
        settings = replace(
            settings,
            index=replace(settings.index, collection=collection),
        )

    session = ChatSession(config=settings, top_k=top_k)
    if once:
        try:
            answer = session.ask(once)
        except QueryError as error:
            raise typer.BadParameter(str(error)) from error
        _print_query_answer(answer)
        return

    console.print("COBOL RAG chat. Commands: /sources, /reset, /exit")
    while True:
        try:
            message = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break

        if not message:
            continue
        if message in {"/exit", "/quit"}:
            break
        if message == "/reset":
            session.reset()
            console.print("Chat memory cleared.")
            continue
        if message == "/sources":
            _print_sources(session.last_sources())
            continue
        if message == "/help":
            console.print("Commands: /sources, /reset, /exit")
            continue

        try:
            answer = session.ask(message)
        except QueryError as error:
            console.print(f"Error: {error}")
            continue
        _print_query_answer(answer)


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


def _print_remove_plan(plan: RemovePlan) -> None:
    summary = Table(title="Remove Plan")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("collection", plan.collection)
    summary.add_row("manifest", str(plan.manifest_path))
    summary.add_row("dry_run", str(plan.dry_run))
    summary.add_row("documents", str(plan.total_documents))
    summary.add_row("indexing_delete", "no" if plan.dry_run else "yes")
    summary.add_row("manifest_write", "no" if plan.dry_run else "yes")
    console.print(summary)

    detail = Table(title="Remove Items")
    detail.add_column("Source Format")
    detail.add_column("Source ID")
    detail.add_column("Source Path")
    detail.add_column("Content Hash")
    for entry in plan.entries:
        detail.add_row(
            entry.source_format,
            entry.source_id,
            entry.source_path,
            entry.content_hash,
        )
    console.print(detail)


def _print_reset_plan(plan: ResetPlan) -> None:
    summary = Table(title="Reset Plan")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("collection", plan.collection)
    summary.add_row("chroma_dir", str(plan.chroma_dir))
    summary.add_row("manifest", str(plan.manifest_path))
    summary.add_row("dry_run", str(plan.dry_run))
    summary.add_row("documents_before", str(plan.document_count))
    summary.add_row("manifest_exists", str(plan.manifest_exists))
    summary.add_row("collection_reset", "no" if plan.dry_run else "yes")
    summary.add_row("manifest_removed", "no" if plan.dry_run else "yes")
    console.print(summary)


def _print_retrieval_results(
    query: str,
    results: list[RetrievalResult],
    preview_chars: int,
) -> None:
    summary = Table(title="Retrieval Summary")
    summary.add_column("Metric")
    summary.add_column("Value")
    summary.add_row("query", query)
    summary.add_row("results", str(len(results)))
    summary.add_row("llm_answer", "no")
    console.print(summary)

    table = Table(title="Retrieved Sources")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source Format")
    table.add_column("Source ID")
    table.add_column("Source Path")
    table.add_column("Preview")
    for rank, result in enumerate(results, start=1):
        score = "" if result.score is None else f"{result.score:.4f}"
        table.add_row(
            str(rank),
            score,
            str(result.metadata.get("source_format", "")),
            str(result.metadata.get("source_id", "")),
            str(result.metadata.get("source_path", "")),
            _preview(result.text, preview_chars),
        )
    console.print(table)


def _print_query_answer(answer: QueryAnswer) -> None:
    console.print(Panel(answer.answer, title="Answer"))
    _print_sources(answer.sources)


def _print_sources(sources: list[RetrievalResult]) -> None:
    if not sources:
        console.print("No sources for the current chat turn.")
        return
    table = Table(title="Sources")
    table.add_column("Rank", justify="right")
    table.add_column("Score", justify="right")
    table.add_column("Source Format")
    table.add_column("Source ID")
    table.add_column("Source Path")
    for rank, source in enumerate(sources, start=1):
        score = "" if source.score is None else f"{source.score:.4f}"
        table.add_row(
            str(rank),
            score,
            str(source.metadata.get("source_format", "")),
            str(source.metadata.get("source_id", "")),
            str(source.metadata.get("source_path", "")),
        )
    console.print(table)


if __name__ == "__main__":
    app()
