"""RAG module for indexing and querying Obsidian notes with ChromaDB."""

from pathlib import Path
from typing import Iterator

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

    def index_vault(self, force_reindex: bool = False) -> dict:
        """Index all supported files in the Obsidian vault.

        Supports: Markdown (.md), Text (.txt), PDF (.pdf)

        Args:
            force_reindex: If True, delete existing collection and reindex.

        Returns:
            Dict with indexing statistics.
        """
        if force_reindex:
            self.chroma_client.delete_collection(self.collection_name)
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

        stats = {
            "files_processed": 0,
            "chunks_added": 0,
            "errors": [],
            "by_type": {"markdown": 0, "text": 0, "pdf": 0},
        }

        for file_path in self._iter_supported_files():
            try:
                relative_path = file_path.relative_to(self.vault_path)
                file_type = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "unknown")

                # Extract text content
                content = self._extract_text(file_path)

                # Skip empty files
                if not content.strip():
                    continue

                # Chunk the content
                chunks = self._chunk_text(content)

                for i, chunk in enumerate(chunks):
                    doc_id = f"{relative_path}::chunk_{i}"

                    # Check if already indexed
                    existing = self.collection.get(ids=[doc_id])
                    if existing["ids"]:
                        continue

                    # Generate embedding
                    embedding = self.embedder.encode(chunk).tolist()

                    # Add to collection
                    self.collection.add(
                        ids=[doc_id],
                        embeddings=[embedding],
                        documents=[chunk],
                        metadatas=[
                            {
                                "source": str(relative_path),
                                "chunk_index": i,
                                "title": file_path.stem,
                                "file_type": file_type,
                            }
                        ],
                    )
                    stats["chunks_added"] += 1

                stats["files_processed"] += 1
                stats["by_type"][file_type] = stats["by_type"].get(file_type, 0) + 1

            except Exception as e:
                stats["errors"].append(f"{file_path}: {e}")

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
