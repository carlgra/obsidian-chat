"""Pytest fixtures for obsidian-chat tests."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def temp_vault():
    """Create a temporary vault with sample markdown files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vault_path = Path(tmpdir)

        # Create sample notes
        (vault_path / "note1.md").write_text(
            "# Python Basics\n\nPython is a programming language.\n\n"
            "## Variables\n\nVariables store data."
        )
        (vault_path / "note2.md").write_text(
            "# JavaScript\n\nJavaScript is used for web development.\n\n"
            "## Functions\n\nFunctions are reusable code blocks."
        )
        (vault_path / "subfolder").mkdir()
        (vault_path / "subfolder" / "note3.md").write_text(
            "# Machine Learning\n\nML is a subset of AI.\n\n"
            "## Neural Networks\n\nNeural networks learn patterns."
        )

        yield vault_path


@pytest.fixture
def temp_chroma_dir():
    """Create a temporary directory for ChromaDB."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        "choices": [
            {
                "message": {
                    "content": "This is a test response from the LLM."
                }
            }
        ]
    }


@pytest.fixture
def mock_llm_stream_response():
    """Mock streaming LLM API response."""
    chunks = [
        'data: {"choices": [{"delta": {"content": "Hello"}}]}\n',
        'data: {"choices": [{"delta": {"content": " world"}}]}\n',
        'data: {"choices": [{"delta": {"content": "!"}}]}\n',
        'data: [DONE]\n',
    ]
    return chunks
