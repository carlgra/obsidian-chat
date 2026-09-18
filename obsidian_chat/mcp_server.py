"""MCP server for obsidian-chat — exposes vault search and research as tools."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from mcp.server.fastmcp import FastMCP, Context

from .config import config
from .logging import get_logger
from .rag import ObsidianRAG
from .research import scholar_search, is_research_available, ResearchError

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(server: FastMCP) -> AsyncIterator[dict]:
    """Initialize ObsidianRAG once on startup."""
    log.info("MCP server starting — initializing RAG with vault: {}", config.vault_path)
    rag = ObsidianRAG()
    log.info("RAG initialized with {} chunks", rag.collection.count())
    yield {"rag": rag}


mcp = FastMCP(
    "obsidian-chat",
    instructions="Search your Obsidian vault and academic papers",
    lifespan=lifespan,
)


@mcp.tool()
def search_notes(query: str, top_k: int = 5, ctx: Context = None) -> list[dict]:
    """Search your Obsidian vault for notes matching a query.

    Uses semantic search (RAG) to find the most relevant note chunks.

    Args:
        query: Natural language search query.
        top_k: Number of results to return (default 5).

    Returns:
        List of matching note chunks with content, source file, and relevance score.
    """
    rag: ObsidianRAG = ctx.request_context.lifespan_context["rag"]
    results = rag.query(query, top_k=top_k)
    log.debug("search_notes '{}' returned {} results", query[:50], len(results))
    return results


@mcp.tool()
def search_papers(query: str, limit: int = 5) -> list[dict] | str:
    """Search academic papers via Semantic Scholar.

    Requires SEMANTIC_SCHOLAR_API_KEY to be configured in .env.

    Args:
        query: Search query for academic papers.
        limit: Maximum number of papers to return (default 5).

    Returns:
        List of papers with title, year, abstract, and URL.
    """
    if not is_research_available():
        return "Research not available — set SEMANTIC_SCHOLAR_API_KEY in .env"

    try:
        papers = scholar_search(query, limit=limit)
        log.debug("search_papers '{}' returned {} papers", query[:50], len(papers))
        return papers
    except ResearchError as e:
        log.error("Research error: {}", e.message)
        return f"Research search failed: {e.message}"


@mcp.tool()
def index_vault(force: bool = False, ctx: Context = None) -> dict:
    """Index or reindex the Obsidian vault.

    Scans the vault for Markdown, text, and PDF files, then indexes them
    for semantic search.

    Args:
        force: If True, delete existing index and rebuild from scratch.

    Returns:
        Statistics: files_processed, chunks_added, errors.
    """
    rag: ObsidianRAG = ctx.request_context.lifespan_context["rag"]
    log.info("index_vault called (force={})", force)
    stats = rag.index_vault(force_reindex=force)
    log.info("Indexing complete: {} files, {} chunks", stats["files_processed"], stats["chunks_added"])
    return stats


@mcp.tool()
def vault_stats(ctx: Context = None) -> dict:
    """Get statistics about the indexed Obsidian vault.

    Returns:
        Total chunks indexed, collection name, and vault path.
    """
    rag: ObsidianRAG = ctx.request_context.lifespan_context["rag"]
    return rag.get_stats()
