"""RAG module for indexing and querying Obsidian notes with ChromaDB."""

from pathlib import Path
from typing import Callable, Iterator

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .config import config


# Directories to ignore when indexing
DEFAULT_IGNORE_DIRS = {
    ".venv",
    "venv",
    ".git",
    ".obsidian",
    "node_modules",
    "__pycache__",
    ".trash",
    ".DS_Store",
}

# Supported file extensions and their types
SUPPORTED_EXTENSIONS = {
    ".md": "markdown",
    ".txt": "text",
    ".pdf": "pdf",
}


class ObsidianRAG:
    """RAG system for Obsidian vault using ChromaDB."""

    def __init__(
        self,
        vault_path: str | None = None,
        persist_dir: str | None = None,
        collection_name: str | None = None,
        embedding_model: str | None = None,
        ignore_dirs: set[str] | None = None,
    ):
        self.vault_path = Path(vault_path or config.vault_path)
        self.persist_dir = persist_dir or config.chroma_persist_dir
        self.collection_name = collection_name or config.collection_name
        self.ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS

        # Ensure persist directory exists
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.chroma_client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

        # Initialize embedding model
        model_name = embedding_model or config.embedding_model
        self.embedder = SentenceTransformer(model_name)

        # Get or create collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _should_ignore(self, path: Path) -> bool:
        """Check if a path should be ignored based on ignore_dirs."""
        for part in path.parts:
            if part in self.ignore_dirs:
                return True
        return False

    def _iter_supported_files(self) -> Iterator[Path]:
        """Iterate over all supported files in the vault, excluding ignored dirs."""
        for ext in SUPPORTED_EXTENSIONS:
            for file_path in self.vault_path.rglob(f"*{ext}"):
                if not self._should_ignore(file_path.relative_to(self.vault_path)):
                    yield file_path

    def _extract_text(self, file_path: Path) -> str:
        """Extract text content from a file based on its type."""
        ext = file_path.suffix.lower()
        file_type = SUPPORTED_EXTENSIONS.get(ext)

        if file_type in ("markdown", "text"):
            return file_path.read_text(encoding="utf-8")

        elif file_type == "pdf":
            return self._extract_pdf_text(file_path)

        return ""

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            from pypdf import PdfReader

            reader = PdfReader(file_path)
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n\n".join(text_parts)
        except Exception as e:
            raise ValueError(f"Failed to extract PDF text: {e}")

    def _chunk_text(
        self, text: str, chunk_size: int = 1000, overlap: int = 200
    ) -> list[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk)
            start = end - overlap
        return chunks

    def _max_batch_size(self) -> int:
        """Largest number of records Chroma will accept in one call."""
        try:
            return max(1, self.chroma_client.get_max_batch_size())
        except Exception:
            # Older clients don't expose the limit; fall back to a safe floor.
            return 1000

    def _indexed_mtimes(self) -> dict[str, float | None]:
        """Map each indexed source path to the mtime it was indexed at.

        Sources indexed before mtime tracking existed map to None, which marks
        them as present-but-unknown so they get reindexed once.
        """
        indexed: dict[str, float | None] = {}
        existing = self.collection.get(include=["metadatas"])

        for metadata in existing["metadatas"] or []:
            source = metadata.get("source")
            if source is None:
                continue
            mtime = metadata.get("mtime")
            # A source is unchanged only if every one of its chunks agrees.
            if source in indexed and indexed[source] != mtime:
                indexed[source] = None
            else:
                indexed[source] = mtime

        return indexed

    def index_vault(
        self,
        force_reindex: bool = False,
        progress_callback: "Callable[[dict], None] | None" = None,
    ) -> dict:
        """Index all supported files in the Obsidian vault.

        Supports: Markdown (.md), Text (.txt), PDF (.pdf)

        Indexing is incremental: files whose mtime matches the indexed copy are
        skipped, modified files have their old chunks replaced, and files that
        have disappeared from the vault are purged from the collection.

        Args:
            force_reindex: If True, delete existing collection and reindex.
            progress_callback: Optional callback receiving progress dicts.

        Returns:
            Dict with indexing statistics.
        """
        if force_reindex:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        stats = {
            "files_processed": 0,
            "files_updated": 0,
            "files_skipped": 0,
            "files_removed": 0,
            "chunks_added": 0,
            "errors": [],
            "by_type": {"markdown": 0, "text": 0, "pdf": 0},
        }

        # Collect files upfront so we can report total count
        files = list(self._iter_supported_files())
        total_files = len(files)

        if progress_callback:
            progress_callback({"phase": "scanning", "total_files": total_files})

        indexed = {} if force_reindex else self._indexed_mtimes()
        seen_sources: set[str] = set()

        for file_index, file_path in enumerate(files):
            try:
                relative_path = file_path.relative_to(self.vault_path)
                source = str(relative_path)
                file_type = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown")
                seen_sources.add(source)

                # Chroma stores metadata as float64; round so it round-trips.
                mtime = round(file_path.stat().st_mtime, 3)
                was_indexed = source in indexed

                # Unchanged since the last run — nothing to do.
                if was_indexed and indexed[source] == mtime:
                    stats["files_skipped"] += 1
                    continue

                if progress_callback:
                    progress_callback({
                        "phase": "indexing",
                        "current_file": file_index + 1,
                        "total_files": total_files,
                        "file_name": file_path.name,
                    })

                # Extract text content
                content = self._extract_text(file_path)
                chunks = self._chunk_text(content) if content.strip() else []

                # Drop the previous chunks only once extraction has succeeded,
                # otherwise a failed read would empty a good file from the index.
                if was_indexed:
                    self.collection.delete(where={"source": source})

                # Emptied file: its chunks are gone and there is nothing to add.
                if not chunks:
                    continue

                embeddings = self.embedder.encode(chunks).tolist()

                # Chroma caps how many records a single call may carry, and a
                # long note can chunk well past it.
                batch_size = self._max_batch_size()
                for start in range(0, len(chunks), batch_size):
                    batch = chunks[start : start + batch_size]
                    self.collection.upsert(
                        ids=[
                            f"{source}::chunk_{start + i}" for i in range(len(batch))
                        ],
                        embeddings=embeddings[start : start + batch_size],
                        documents=batch,
                        metadatas=[
                            {
                                "source": source,
                                "chunk_index": start + i,
                                "title": file_path.stem,
                                "file_type": file_type,
                                "mtime": mtime,
                            }
                            for i in range(len(batch))
                        ],
                    )
                stats["chunks_added"] += len(chunks)

                stats["files_processed"] += 1
                if was_indexed:
                    stats["files_updated"] += 1
                stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1

            except Exception as e:
                stats["errors"].append(f"{file_path}: {e}")

        # Purge files that have been deleted or renamed out of the vault
        for source in set(indexed) - seen_sources:
            try:
                self.collection.delete(where={"source": source})
                stats["files_removed"] += 1
            except Exception as e:
                stats["errors"].append(f"{source}: failed to purge: {e}")

        return stats

    def query(self, query_text: str, top_k: int | None = None) -> list[dict]:
        """Query the indexed notes for relevant context.

        Args:
            query_text: The query to search for.
            top_k: Number of results to return.

        Returns:
            List of dicts with 'content', 'source', and 'score' keys.
        """
        top_k = top_k or config.top_k

        # Generate query embedding
        query_embedding = self.embedder.encode(query_text).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Format results
        formatted = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                formatted.append(
                    {
                        "content": doc,
                        "source": results["metadatas"][0][i]["source"],
                        "title": results["metadatas"][0][i]["title"],
                        "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    }
                )

        return formatted

    def get_stats(self) -> dict:
        """Get statistics about the indexed collection."""
        return {
            "total_chunks": self.collection.count(),
            "collection_name": self.collection_name,
            "vault_path": str(self.vault_path),
        }
