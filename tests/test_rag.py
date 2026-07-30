"""Tests for the RAG module.

These tests are marked as 'slow' because they load the sentence-transformers
embedding model which takes several seconds.

Run with: pytest -m slow
Skip with: pytest -m "not slow"
"""

import os
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


class TestIncrementalIndexing:
    """Tests for mtime-based incremental reindexing."""

    def _rag(self, vault, chroma, name):
        from obsidian_chat.rag import ObsidianRAG

        return ObsidianRAG(
            vault_path=str(vault),
            persist_dir=str(chroma),
            collection_name=name,
        )

    def test_unchanged_files_are_skipped(self, temp_vault, temp_chroma_dir):
        """A second run over an untouched vault re-embeds nothing."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_skip")

        first = rag.index_vault()
        second = rag.index_vault()

        assert first["files_processed"] == 4
        assert second["files_processed"] == 0
        assert second["chunks_added"] == 0
        assert second["files_skipped"] == 4

    def test_modified_file_is_reindexed(self, temp_vault, temp_chroma_dir):
        """Editing a note replaces its chunks rather than leaving stale text."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_modify")
        rag.index_vault()

        note = temp_vault / "note1.md"
        note.write_text("# Python Basics\n\nPython now covers async programming.")
        # Ensure the mtime actually differs from the indexed value
        os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 10))

        stats = rag.index_vault()

        assert stats["files_updated"] == 1
        assert stats["files_processed"] == 1
        assert stats["files_skipped"] == 3

        stored = rag.collection.get(where={"source": "note1.md"})
        combined = " ".join(stored["documents"])
        assert "async programming" in combined
        assert "Variables store data" not in combined

    def test_shrunk_file_drops_trailing_chunks(self, temp_vault, temp_chroma_dir):
        """A note that gets much shorter leaves no orphaned chunks behind."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_shrink")

        note = temp_vault / "long.md"
        note.write_text("word " * 2000)
        rag.index_vault()
        long_chunks = len(rag.collection.get(where={"source": "long.md"})["ids"])
        assert long_chunks > 1

        note.write_text("short")
        os.utime(note, (note.stat().st_atime, note.stat().st_mtime + 10))
        rag.index_vault()

        assert len(rag.collection.get(where={"source": "long.md"})["ids"]) == 1

    def test_deleted_file_is_purged(self, temp_vault, temp_chroma_dir):
        """Removing a note from the vault removes it from the index."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_delete")
        rag.index_vault()

        assert rag.collection.get(where={"source": "note2.md"})["ids"]
        (temp_vault / "note2.md").unlink()

        stats = rag.index_vault()

        assert stats["files_removed"] == 1
        assert rag.collection.get(where={"source": "note2.md"})["ids"] == []

    def test_legacy_index_without_mtime_is_upgraded(self, temp_vault, temp_chroma_dir):
        """Chunks indexed before mtime tracking get reindexed once, not duplicated."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_legacy")
        rag.index_vault()

        # Simulate a pre-upgrade index by rewriting every chunk without mtime.
        # Chroma's update() merges metadata, so the rows have to be replaced.
        existing = rag.collection.get(include=["metadatas", "documents", "embeddings"])
        stripped = [
            {k: v for k, v in m.items() if k != "mtime"} for m in existing["metadatas"]
        ]
        rag.collection.delete(ids=existing["ids"])
        rag.collection.add(
            ids=existing["ids"],
            embeddings=existing["embeddings"],
            documents=existing["documents"],
            metadatas=stripped,
        )
        before = rag.collection.count()
        assert all("mtime" not in m for m in rag.collection.get(include=["metadatas"])["metadatas"])

        stats = rag.index_vault()

        assert stats["files_updated"] == 4
        assert stats["files_skipped"] == 0
        assert rag.collection.count() == before  # replaced, not duplicated

    def test_file_larger_than_max_batch_is_indexed(self, temp_vault, temp_chroma_dir):
        """A note chunking past Chroma's per-call record cap still indexes fully."""
        rag = self._rag(temp_vault, temp_chroma_dir, "test_incr_batch")

        # Force a small cap so the test stays fast but exercises the batching.
        rag._max_batch_size = lambda: 5

        note = temp_vault / "huge.md"
        note.write_text("alpha beta gamma delta " * 1200)
        expected = len(rag._chunk_text(note.read_text()))
        assert expected > 5  # otherwise the batching path isn't hit

        stats = rag.index_vault()

        assert not [e for e in stats["errors"] if "huge.md" in e]
        stored = rag.collection.get(where={"source": "huge.md"})
        assert len(stored["ids"]) == expected
        # chunk_index must stay contiguous across batch boundaries
        assert sorted(m["chunk_index"] for m in stored["metadatas"]) == list(range(expected))
