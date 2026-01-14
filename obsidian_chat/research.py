"""Research module for fetching academic papers from Semantic Scholar."""

import httpx

from .config import config
from .logging import get_logger

log = get_logger(__name__)

SEMANTIC_SCHOLAR_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


class ResearchError(Exception):
    """Exception raised when research API call fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def scholar_search(query: str, limit: int = 5) -> list[dict]:
    """Search Semantic Scholar for academic papers.

    Args:
        query: Search query string.
        limit: Maximum number of results to return.

    Returns:
        List of dicts with keys: title, year, abstract, url.

    Raises:
        ResearchError: If API key is not configured or API call fails.
    """
    api_key = config.semantic_scholar_api_key
    if not api_key:
        raise ResearchError(
            "SEMANTIC_SCHOLAR_API_KEY is not configured. "
            "Add it to your .env file to enable research features."
        )

    log.debug("Searching Semantic Scholar: '{}' (limit={})", query[:50], limit)

    params = {
        "query": query,
        "limit": limit,
        "fields": "title,year,abstract,url",
    }

    headers = {
        "x-api-key": api_key,
    }

    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.get(
                SEMANTIC_SCHOLAR_API_URL,
                params=params,
                headers=headers,
            )

            if response.status_code != 200:
                log.error("Semantic Scholar API error: {}", response.status_code)
                raise ResearchError(
                    f"Semantic Scholar API error ({response.status_code}): {response.text[:200]}",
                    status_code=response.status_code,
                )

            data = response.json()
            results = []

            for paper in data.get("data", []):
                results.append({
                    "title": paper.get("title"),
                    "year": paper.get("year"),
                    "abstract": paper.get("abstract"),
                    "url": paper.get("url"),
                })

            log.debug("Found {} papers", len(results))
            return results

    except httpx.ConnectError:
        raise ResearchError("Cannot connect to Semantic Scholar API.")
    except httpx.TimeoutException:
        raise ResearchError("Semantic Scholar API request timed out.")


def build_research_context(papers: list[dict]) -> str:
    """Build a context prompt from research paper results.

    Args:
        papers: List of paper dicts from scholar_search.

    Returns:
        Formatted context string for the LLM prompt.
    """
    if not papers:
        return ""

    context_parts = ["Here are relevant academic papers:\n"]

    for i, paper in enumerate(papers, 1):
        title = paper.get("title", "Unknown")
        year = paper.get("year", "N/A")
        abstract = paper.get("abstract", "No abstract available.")
        url = paper.get("url", "")

        context_parts.append(f"--- Paper {i}: {title} ({year}) ---")
        if abstract:
            # Truncate very long abstracts
            if len(abstract) > 500:
                abstract = abstract[:500] + "..."
            context_parts.append(abstract)
        if url:
            context_parts.append(f"URL: {url}")
        context_parts.append("")

    return "\n".join(context_parts)


def is_research_available() -> bool:
    """Check if research functionality is available (API key configured)."""
    return bool(config.semantic_scholar_api_key)
