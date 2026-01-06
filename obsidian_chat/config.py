"""Configuration management for obsidian-chat."""

import os
from pathlib import Path
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from the project root (parent of this file's directory)
_project_root = Path(__file__).parent.parent
_env_file = _project_root / ".env"
load_dotenv(_env_file, override=True)


@dataclass
class Config:
    """Application configuration."""

    # LLM API settings
    llm_base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:1234/v1")
    llm_model: str = os.getenv("LLM_MODEL", "local-model")
    llm_api_key: str = os.getenv("LLM_API_KEY", "not-needed")

    # Obsidian vault settings
    vault_path: str = os.getenv("OBSIDIAN_VAULT_PATH", "")

    # ChromaDB settings
    chroma_persist_dir: str = str(Path(os.getenv(
        "CHROMA_PERSIST_DIR",
        str(Path.home() / ".obsidian-chat" / "chroma"),
    )).expanduser())
    collection_name: str = os.getenv("CHROMA_COLLECTION", "obsidian_notes")

    # Embedding model
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # RAG settings
    top_k: int = int(os.getenv("RAG_TOP_K", "5"))

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors."""
        errors = []
        if not self.vault_path:
            errors.append("OBSIDIAN_VAULT_PATH is not set")
        elif not Path(self.vault_path).exists():
            errors.append(f"Vault path does not exist: {self.vault_path}")
        return errors


config = Config()
