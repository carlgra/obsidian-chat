"""Tests for the config module."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


class TestConfig:
    """Tests for Config class."""

    def test_config_loads_from_env(self, mock_env):
        """Test that config loads values from environment variables."""
        # Import fresh to pick up mocked env
        from obsidian_chat.config import Config

        config = Config()

        assert config.llm_base_url == "http://localhost:1234/v1"
        assert config.llm_model == "test-model"
        assert config.llm_api_key == "test-key"
        assert config.vault_path == mock_env["OBSIDIAN_VAULT_PATH"]

    def test_config_default_values(self):
        """Test that config has sensible defaults."""
        with patch.dict(os.environ, {}, clear=True):
            from obsidian_chat.config import Config

            config = Config()

            assert config.llm_base_url == "http://localhost:1234/v1"
            assert config.llm_model == "local-model"
            assert config.embedding_model == "all-MiniLM-L6-v2"
            assert config.rag_top_k == 5

    def test_config_validate_missing_vault(self, mock_env):
        """Test validation fails when vault path doesn't exist."""
        with patch.dict(os.environ, {"OBSIDIAN_VAULT_PATH": "/nonexistent/path"}):
            from obsidian_chat.config import Config

            config = Config()
            errors = config.validate()

            assert len(errors) > 0
            assert any("vault" in e.lower() or "not exist" in e.lower() for e in errors)

    def test_config_validate_valid_vault(self, mock_env, temp_vault):
        """Test validation passes with valid vault."""
        from obsidian_chat.config import Config

        config = Config()
        config.vault_path = str(temp_vault)
        errors = config.validate()

        assert len(errors) == 0

    def test_chroma_persist_dir_expansion(self):
        """Test that ~ is expanded in chroma persist dir."""
        with patch.dict(os.environ, {"CHROMA_PERSIST_DIR": "~/.obsidian-chat/chroma"}):
            from obsidian_chat.config import Config

            config = Config()

            assert "~" not in config.chroma_persist_dir
            assert config.chroma_persist_dir.startswith("/")
