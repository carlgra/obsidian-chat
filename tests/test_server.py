"""Tests for the FastAPI server."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, test_client):
        """Test health endpoint returns OK status."""
        response = test_client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "llm_url" in data
        assert "vault_path" in data
        assert "indexed_chunks" in data


class TestStatsEndpoint:
    """Tests for the /stats endpoint."""

    def test_get_stats(self, test_client):
        """Test stats endpoint returns index statistics."""
        response = test_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "collection_name" in data
        assert "vault_path" in data


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_notes(self, test_client):
        """Test querying notes without LLM."""
        # First index the vault
        test_client.post("/index", json={"force": True})

        response = test_client.post(
            "/query",
            json={"query": "Python", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)

    def test_query_with_custom_top_k(self, test_client):
        """Test query respects top_k parameter."""
        test_client.post("/index", json={"force": True})

        response = test_client.post(
            "/query",
            json={"query": "programming", "top_k": 1},
        )

        assert response.status_code == 200
        data = response.json()
        assert len(data["results"]) <= 1


class TestIndexEndpoint:
    """Tests for the /index endpoint."""

    def test_index_vault(self, test_client):
        """Test indexing the vault."""
        response = test_client.post("/index", json={"force": False})

        assert response.status_code == 200
        data = response.json()
        assert "files_processed" in data
        assert "chunks_added" in data
        assert "errors" in data

    def test_force_reindex(self, test_client):
        """Test force reindexing."""
        # First index
        test_client.post("/index", json={"force": False})

        # Force reindex
        response = test_client.post("/index", json={"force": True})

        assert response.status_code == 200
        data = response.json()
        assert data["files_processed"] > 0


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_non_streaming(self, test_client):
        """Test non-streaming chat."""
        # Index first
        test_client.post("/index", json={"force": True})

        # Mock the LLM response
        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = "This is a test response."

            response = test_client.post(
                "/chat",
                json={
                    "message": "What is Python?",
                    "top_k": 5,
                    "use_rag": True,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data

    def test_chat_without_rag(self, test_client):
        """Test chat with RAG disabled."""
        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = "Response without RAG."

            response = test_client.post(
                "/chat",
                json={
                    "message": "Hello",
                    "use_rag": False,
                    "stream": False,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["sources"] == []

    def test_chat_with_history(self, test_client):
        """Test chat with conversation history."""
        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = "Response with history."

            response = test_client.post(
                "/chat",
                json={
                    "message": "Follow up question",
                    "use_rag": False,
                    "stream": False,
                    "history": [
                        {"role": "user", "content": "First question"},
                        {"role": "assistant", "content": "First answer"},
                    ],
                },
            )

        assert response.status_code == 200

    def test_chat_with_summary(self, test_client):
        """Test chat with conversation summary."""
        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = "Response with summary context."

            response = test_client.post(
                "/chat",
                json={
                    "message": "New question",
                    "use_rag": False,
                    "stream": False,
                    "summary": "Previously discussed Python basics.",
                },
            )

        assert response.status_code == 200


class TestChatStreamEndpoint:
    """Tests for the /chat/stream endpoint."""

    def test_chat_stream(self, test_client):
        """Test streaming chat endpoint."""
        test_client.post("/index", json={"force": True})

        def mock_stream(*args, **kwargs):
            yield "Hello"
            yield " world"

        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = mock_stream()

            response = test_client.post(
                "/chat/stream",
                json={
                    "message": "Hello",
                    "top_k": 5,
                    "use_rag": True,
                },
            )

        assert response.status_code == 200
        assert "Hello world" in response.text


class TestSummarizeEndpoint:
    """Tests for the /summarize endpoint."""

    def test_summarize_conversation(self, test_client):
        """Test summarizing a conversation."""
        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.return_value = "User asked about Python and got an explanation."

            response = test_client.post(
                "/summarize",
                json={
                    "messages": [
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a programming language."},
                    ],
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "summary" in data
        assert len(data["summary"]) > 0


class TestStaticFiles:
    """Tests for static file serving."""

    def test_serve_index_html(self, test_client):
        """Test that root serves the web UI."""
        response = test_client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Obsidian Chat" in response.text


class TestErrorHandling:
    """Tests for error handling."""

    def test_chat_llm_error(self, test_client):
        """Test that LLM errors are handled gracefully."""
        from obsidian_chat.llm import LLMError

        with patch("obsidian_chat.server.llm") as mock_llm:
            mock_llm.chat.side_effect = LLMError("Connection failed", status_code=502)

            response = test_client.post(
                "/chat",
                json={
                    "message": "Hello",
                    "use_rag": False,
                    "stream": False,
                },
            )

        assert response.status_code == 502
        assert "Connection failed" in response.json()["detail"]
