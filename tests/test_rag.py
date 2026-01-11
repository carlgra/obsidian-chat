"""Tests for the RAG module.

These tests are marked as 'slow' because they load the sentence-transformers
embedding model which takes several seconds.

Run with: pytest -m slow
Skip with: pytest -m "not slow"
"""

import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# Mark all tests in this module as slow
pytestmark = pytest.mark.slow


class TestObsidianRAG:
    """Tests for ObsidianRAG class."""

    def test_rag_initialization(self, temp_vault, temp_chroma_dir):
        """Test RAG initializes correctly."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_collection",
        )

        assert rag.vault_path == Path(temp_vault)
        assert rag.collection is not None

    def test_index_vault(self, temp_vault, temp_chroma_dir):
        """Test indexing a vault with multiple file types."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_index",
        )

        stats = rag.index_vault()

        assert stats["files_processed"] == 4  # note1, note2, subfolder/note3, readme.txt
        assert stats["chunks_added"] > 0
        assert isinstance(stats["errors"], list)
        assert stats["by_type"]["markdown"] == 3
        assert stats["by_type"]["text"] == 1

    def test_index_vault_force_reindex(self, temp_vault, temp_chroma_dir):
        """Test force reindexing clears existing data."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_reindex",
        )

        # First index
        stats1 = rag.index_vault()
        first_count = stats1["chunks_added"]

        # Force reindex
        stats2 = rag.index_vault(force_reindex=True)

        # Should have same number of chunks (not doubled)
        assert stats2["chunks_added"] == first_count

    def test_query_returns_results(self, temp_vault, temp_chroma_dir):
        """Test querying returns relevant results."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_query",
        )
        rag.index_vault()

        results = rag.query("Python programming", top_k=3)

        assert len(results) > 0
        assert all("content" in r for r in results)
        assert all("source" in r for r in results)
        assert all("score" in r for r in results)

    def test_query_empty_collection(self, temp_vault, temp_chroma_dir):
        """Test querying empty collection returns empty list."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_empty",
        )

        results = rag.query("anything", top_k=5)

        assert results == []

    def test_get_stats(self, temp_vault, temp_chroma_dir):
        """Test getting index statistics."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_stats",
        )
        rag.index_vault()

        stats = rag.get_stats()

        assert "total_chunks" in stats
        assert "collection_name" in stats
        assert "vault_path" in stats
        assert stats["total_chunks"] > 0

    def test_ignores_non_markdown_files(self, temp_chroma_dir):
        """Test that non-markdown files are ignored."""
        from obsidian_chat.rag import ObsidianRAG

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            (vault_path / "note.md").write_text("# Valid Note\n\nContent here.")
            (vault_path / "image.png").write_bytes(b"fake image data")
            (vault_path / "data.json").write_text('{"key": "value"}')

            rag = ObsidianRAG(
                vault_path=str(vault_path),
                persist_dir=str(temp_chroma_dir),
                collection_name="test_ignore",
            )
            stats = rag.index_vault()

            assert stats["files_processed"] == 1

    def test_handles_empty_files(self, temp_chroma_dir):
        """Test that empty markdown files don't cause errors."""
        from obsidian_chat.rag import ObsidianRAG

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)
            (vault_path / "empty.md").write_text("")
            (vault_path / "whitespace.md").write_text("   \n\n   ")
            (vault_path / "valid.md").write_text("# Valid\n\nContent.")

            rag = ObsidianRAG(
                vault_path=str(vault_path),
                persist_dir=str(temp_chroma_dir),
                collection_name="test_empty_files",
            )
            stats = rag.index_vault()

            # Should process files without error (empty ones may be skipped)
            assert stats["files_processed"] >= 1

    def test_query_respects_top_k(self, temp_vault, temp_chroma_dir):
        """Test that top_k limits results."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_topk",
        )
        rag.index_vault()

        results = rag.query("programming", top_k=1)

        assert len(results) <= 1

    def test_results_include_source_file(self, temp_vault, temp_chroma_dir):
        """Test that results include the source file path."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_source",
        )
        rag.index_vault()

        results = rag.query("Python", top_k=5)

        # Should find the Python note
        sources = [r["source"] for r in results]
        assert any("note1" in s for s in sources)

    def test_indexes_text_files(self, temp_vault, temp_chroma_dir):
        """Test that text files are indexed and searchable."""
        from obsidian_chat.rag import ObsidianRAG

        rag = ObsidianRAG(
            vault_path=str(temp_vault),
            persist_dir=str(temp_chroma_dir),
            collection_name="test_text",
        )
        rag.index_vault()

        results = rag.query("data science statistics", top_k=5)

        # Should find the text file
        sources = [r["source"] for r in results]
        assert any("readme.txt" in s for s in sources)

    def test_pdf_extraction(self, temp_chroma_dir):
        """Test PDF text extraction."""
        from obsidian_chat.rag import ObsidianRAG
        from pypdf import PdfWriter
        from io import BytesIO

        with tempfile.TemporaryDirectory() as tmpdir:
            vault_path = Path(tmpdir)

            # Create a simple PDF with text
            pdf_path = vault_path / "test.pdf"
            writer = PdfWriter()
            writer.add_blank_page(width=612, height=792)
            # Note: PdfWriter doesn't easily add text, so we test the extraction
            # mechanism works without errors on a blank PDF
            with open(pdf_path, "wb") as f:
                writer.write(f)

            rag = ObsidianRAG(
                vault_path=str(vault_path),
                persist_dir=str(temp_chroma_dir),
                collection_name="test_pdf",
            )

            # Should not raise an error
            stats = rag.index_vault()
            assert "pdf" in stats["by_type"]
