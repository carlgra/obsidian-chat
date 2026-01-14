"""Shared utilities for obsidian-chat."""

from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def get_version() -> str:
    """Get the package version from installed metadata or pyproject.toml."""
    try:
        return version("obsidian-chat")
    except PackageNotFoundError:
        # Fallback: read from pyproject.toml during development
        pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
        if pyproject_path.exists():
            import re
            content = pyproject_path.read_text()
            match = re.search(r'version\s*=\s*"([^"]+)"', content)
            if match:
                return match.group(1)
        return "0.0.0"


__version__ = get_version()


# Shared prompts
SYSTEM_PROMPT = """You are a helpful assistant with access to the user's personal notes from their Obsidian vault.
Use the provided context from their notes to answer questions accurately and helpfully.
When referencing information from the notes, mention which note it came from.
If the context doesn't contain relevant information, say so and answer based on your general knowledge."""

SUMMARIZE_PROMPT = """Summarize this conversation concisely in 2-3 sentences, capturing the key topics discussed and any important conclusions or information shared. Focus on what would be useful context for continuing the conversation."""


def build_context_prompt(contexts: list[dict]) -> str:
    """Build a context prompt from RAG results.

    Args:
        contexts: List of dicts with 'source' and 'content' keys.

    Returns:
        Formatted context string for the LLM prompt.
    """
    if not contexts:
        return ""

    context_parts = ["Here is relevant context from your Obsidian notes:\n"]
    for ctx in contexts:
        context_parts.append(f"--- From: {ctx['source']} ---")
        context_parts.append(ctx["content"])
        context_parts.append("")

    return "\n".join(context_parts)
