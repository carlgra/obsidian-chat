"""CLI interface for obsidian-chat."""

import sys
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import config
from .llm import LLMClient
from .logging import get_logger
from .rag import ObsidianRAG
from .utils import __version__, SYSTEM_PROMPT, build_context_prompt

log = get_logger(__name__)

app = typer.Typer(
    name="obsidian-chat",
    help="Chat with a local LLM using RAG from your Obsidian vault.",
)
console = Console()


def version_callback(value: bool):
    """Print version and exit."""
    if value:
        console.print(f"obsidian-chat v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-V", callback=version_callback, is_eager=True,
        help="Show version and exit."
    ),
):
    """Chat with a local LLM using RAG from your Obsidian vault."""
    pass


@app.command()
def index(
    vault_path: Optional[str] = typer.Option(
        None, "--vault", "-v", help="Path to Obsidian vault (overrides env var)"
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Force reindex all files"
    ),
):
    """Index your Obsidian vault for RAG queries."""
    if vault_path:
        config.vault_path = vault_path

    errors = config.validate()
    if errors:
        for error in errors:
            console.print(f"[red]Error:[/red] {error}")
        raise typer.Exit(1)

    log.info("Indexing vault: {}", config.vault_path)
    console.print(f"[blue]Indexing vault:[/blue] {config.vault_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Indexing notes...", total=None)
        rag = ObsidianRAG(vault_path=config.vault_path)
        stats = rag.index_vault(force_reindex=force)

    console.print(f"[green]Done![/green]")
    console.print(f"  Files processed: {stats['files_processed']}")
    console.print(f"  Files updated: {stats['files_updated']}")
    console.print(f"  Files unchanged: {stats['files_skipped']}")
    console.print(f"  Files removed: {stats['files_removed']}")
    console.print(f"  Chunks added: {stats['chunks_added']}")

    if stats["errors"]:
        console.print(f"[yellow]Warnings ({len(stats['errors'])}):[/yellow]")
        for error in stats["errors"][:5]:
            console.print(f"  - {error}")
        if len(stats["errors"]) > 5:
            console.print(f"  ... and {len(stats['errors']) - 5} more")


@app.command()
def stats():
    """Show statistics about the indexed vault."""
    try:
        rag = ObsidianRAG()
        info = rag.get_stats()
        console.print(Panel(
            f"[bold]Collection:[/bold] {info['collection_name']}\n"
            f"[bold]Total chunks:[/bold] {info['total_chunks']}\n"
            f"[bold]Vault path:[/bold] {info['vault_path']}",
            title="Index Statistics",
        ))
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to search for"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Query the indexed notes without LLM (just RAG results)."""
    try:
        rag = ObsidianRAG()
        results = rag.query(question, top_k=top_k)

        if not results:
            console.print("[yellow]No relevant notes found.[/yellow]")
            return

        for i, result in enumerate(results, 1):
            console.print(Panel(
                result["content"][:500] + ("..." if len(result["content"]) > 500 else ""),
                title=f"[bold]{i}. {result['source']}[/bold] (score: {result['score']:.3f})",
            ))

    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)


@app.command()
def chat(
    vault_path: Optional[str] = typer.Option(
        None, "--vault", "-v", help="Path to Obsidian vault"
    ),
    no_rag: bool = typer.Option(
        False, "--no-rag", help="Disable RAG context"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of RAG results"),
):
    """Start an interactive chat session with RAG context."""
    if vault_path:
        config.vault_path = vault_path

    # Initialize components
    llm = LLMClient()
    rag = None if no_rag else ObsidianRAG()

    if rag:
        stats_info = rag.get_stats()
        if stats_info["total_chunks"] == 0:
            console.print(
                "[yellow]Warning:[/yellow] No indexed notes found. "
                "Run 'obsidian-chat index' first."
            )

    console.print(Panel(
        "Chat with your local LLM using Obsidian notes as context.\n"
        "Type [bold]exit[/bold] or [bold]quit[/bold] to end the session.\n"
        "Type [bold]/clear[/bold] to clear conversation history.",
        title="Obsidian Chat",
    ))

    messages: list[dict] = []

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            console.print("[dim]Goodbye![/dim]")
            break

        if user_input.lower() == "/clear":
            messages.clear()
            console.print("[dim]Conversation cleared.[/dim]")
            continue

        # Get RAG context
        context_prompt = ""
        if rag:
            with console.status("[dim]Searching notes...[/dim]"):
                contexts = rag.query(user_input, top_k=top_k)
                context_prompt = build_context_prompt(contexts)

            if contexts:
                sources = ", ".join(set(c["source"] for c in contexts))
                console.print(f"[dim]Found context from: {sources}[/dim]")

        # Build message with context
        if context_prompt:
            full_message = f"{context_prompt}\n\nUser question: {user_input}"
        else:
            full_message = user_input

        messages.append({"role": "user", "content": full_message})

        # Get LLM response
        console.print("[bold green]Assistant:[/bold green] ", end="")

        try:
            full_response = ""
            for chunk in llm.chat(messages, system_prompt=SYSTEM_PROMPT, stream=True):
                console.print(chunk, end="")
                full_response += chunk
            console.print()

            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            console.print(f"\n[red]Error communicating with LLM:[/red] {e}")
            messages.pop()  # Remove failed user message

    llm.close()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Question to ask"),
    no_rag: bool = typer.Option(False, "--no-rag", help="Disable RAG context"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of RAG results"),
):
    """Ask a single question (non-interactive)."""
    llm = LLMClient()
    rag = None if no_rag else ObsidianRAG()

    # Get RAG context
    context_prompt = ""
    if rag:
        with console.status("[dim]Searching notes...[/dim]"):
            contexts = rag.query(question, top_k=top_k)
            context_prompt = build_context_prompt(contexts)

        if contexts:
            sources = ", ".join(set(c["source"] for c in contexts))
            console.print(f"[dim]Using context from: {sources}[/dim]\n")

    # Build message
    if context_prompt:
        full_message = f"{context_prompt}\n\nUser question: {question}"
    else:
        full_message = question

    messages = [{"role": "user", "content": full_message}]

    try:
        for chunk in llm.chat(messages, system_prompt=SYSTEM_PROMPT, stream=True):
            console.print(chunk, end="")
        console.print()
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)
    finally:
        llm.close()


@app.command()
def mcp():
    """Start the MCP server (stdio transport) for Claude Desktop / Claude Code."""
    from .mcp_server import mcp as mcp_server

    mcp_server.run(transport="stdio")


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
):
    """Start the web server with API and UI."""
    import uvicorn
    from .server import app as fastapi_app

    console.print(Panel(
        f"Starting server at [bold]http://{host}:{port}[/bold]\n"
        f"API docs at [bold]http://{host}:{port}/docs[/bold]\n"
        f"Press [bold]Ctrl+C[/bold] to stop.",
        title="Obsidian Chat Server",
    ))

    uvicorn.run(fastapi_app, host=host, port=port)


if __name__ == "__main__":
    app()
