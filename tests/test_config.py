"""Tests for the config module."""

import importlib
import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for Config class."""

    def test_config_loads_from_env(self, temp_vault, temp_chroma_dir):
        """Test that config loads values from environment variables."""
        env_vars = {
            "LLM_BASE_URL": "http://test-server:1234/v1",
            "LLM_MODEL": "test-model",
            "LLM_API_KEY": "test-key",
            "OBSIDIAN_VAULT_PATH": str(temp_vault),
            "CHROMA_PERSIST_DIR": str(temp_chroma_dir),
        }

        # Mock load_dotenv to prevent .env from overriding test vars
        with patch("dotenv.load_dotenv"):
            with patch.dict(os.environ, env_vars, clear=False):
                # Reload module to pick up new env vars
                import obsidian_chat.config
                importlib.reload(obsidian_chat.config)
                from obsidian_chat.config import Config

                config = Config()

                assert config.llm_base_url == "http://test-server:1234/v1"
                assert config.llm_model == "test-model"
                assert config.llm_api_key == "test-key"

    def test_config_has_defaults(self):
        """Test that config dataclass has default values defined."""
        from obsidian_chat.config import Config

        # Check that Config can be instantiated (has defaults)
        config = Config()

        # These should have some value (either from env or defaults)
        assert config.llm_base_url is not None
        assert config.llm_model is not None
        assert config.embedding_model is not None

    def test_config_validate_missing_vault(self):
        """Test validation fails when vault path doesn't exist."""
        from obsidian_chat.config import Config

        config = Config()
        config.vault_path = "/nonexistent/path/that/doesnt/exist"
        errors = config.validate()

        assert len(errors) > 0
        assert any("not exist" in e.lower() or "vault" in e.lower() for e in errors)

    def test_config_validate_empty_vault(self):
        """Test validation fails when vault path is empty."""
        from obsidian_chat.config import Config

        config = Config()
        config.vault_path = ""
        errors = config.validate()

        assert len(errors) > 0

    def test_config_validate_valid_vault(self, temp_vault):
        """Test validation passes with valid vault."""
        from obsidian_chat.config import Config

        config = Config()
        config.vault_path = str(temp_vault)
        errors = config.validate()

        assert len(errors) == 0

    def test_chroma_persist_dir_expansion(self):
        """Test that ~ is expanded in chroma persist dir."""
        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": "~/.obsidian-chat/chroma"}):
            import obsidian_chat.config
            importlib.reload(obsidian_chat.config)
            from obsidian_chat.config import Config

            config = Config()

            assert "~" not in config.chroma_persist_dir
            assert config.chroma_persist_dir.startswith("/")

    def test_config_rag_top_k_default(self):
        """Test RAG top_k has a sensible default."""
        from obsidian_chat.config import Config

        config = Config()

        assert config.top_k > 0
        assert isinstance(config.top_k, int)
