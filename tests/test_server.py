"""Tests for the FastAPI server."""

from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def mock_rag():
    """Mock the RAG component."""
    mock = MagicMock()
    mock.query.return_value = [
        {"content": "Test content", "source": "test.md", "title": "Test", "score": 0.9}
    ]
    mock.get_stats.return_value = {
        "total_chunks": 100,
        "collection_name": "test",
        "vault_path": "/test/vault",
    }
    mock.index_vault.return_value = {
        "files_processed": 5,
        "chunks_added": 50,
        "errors": [],
    }
    mock.collection.count.return_value = 100
    return mock


@pytest.fixture
def mock_llm():
    """Mock the LLM component."""
    mock = MagicMock()
    mock.chat.return_value = "This is a test response."
    return mock


@pytest.fixture
def test_client(mock_rag, mock_llm):
    """Create a test client with mocked dependencies."""
    with patch("obsidian_chat.server.rag", mock_rag), \
         patch("obsidian_chat.server.llm", mock_llm), \
         patch("obsidian_chat.server.ObsidianRAG", return_value=mock_rag), \
         patch("obsidian_chat.server.LLMClient", return_value=mock_llm):

        from obsidian_chat.server import app

        # Manually set the globals
        import obsidian_chat.server as server_module
        server_module.rag = mock_rag
        server_module.llm = mock_llm

        with TestClient(app) as client:
            yield client


class TestVersionEndpoint:
    """Tests for the /version endpoint."""

    def test_get_version(self, test_client):
        """Test version endpoint returns version string."""
        response = test_client.get("/version")

        assert response.status_code == 200
        data = response.json()
        assert "version" in data
        assert isinstance(data["version"], str)
        assert len(data["version"]) > 0


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, test_client, mock_rag):
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

    def test_get_stats(self, test_client, mock_rag):
        """Test stats endpoint returns index statistics."""
        response = test_client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert "collection_name" in data
        assert "vault_path" in data


class TestQueryEndpoint:
    """Tests for the /query endpoint."""

    def test_query_notes(self, test_client, mock_rag):
        """Test querying notes without LLM."""
        response = test_client.post(
            "/query",
            json={"query": "Python", "top_k": 5},
        )

        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        mock_rag.query.assert_called_once()

    def test_query_with_custom_top_k(self, test_client, mock_rag):
        """Test query respects top_k parameter."""
        response = test_client.post(
            "/query",
            json={"query": "programming", "top_k": 3},
        )

        assert response.status_code == 200
        mock_rag.query.assert_called_with("programming", top_k=3)


class TestIndexEndpoint:
    """Tests for the /index endpoint."""

    def test_index_vault(self, test_client, mock_rag):
        """Test indexing the vault."""
        response = test_client.post("/index", json={"force": False})

        assert response.status_code == 200
        data = response.json()
        assert "files_processed" in data
        assert "chunks_added" in data
        assert "errors" in data

    def test_force_reindex(self, test_client, mock_rag):
        """Test force reindexing."""
        response = test_client.post("/index", json={"force": True})

        assert response.status_code == 200
        mock_rag.index_vault.assert_called_with(force_reindex=True)


class TestChatEndpoint:
    """Tests for the /chat endpoint."""

    def test_chat_non_streaming(self, test_client, mock_llm, mock_rag):
        """Test non-streaming chat."""
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
        mock_llm.chat.assert_called_once()

    def test_chat_without_rag(self, test_client, mock_llm, mock_rag):
        """Test chat with RAG disabled."""
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
        mock_rag.query.assert_not_called()

    def test_chat_with_history(self, test_client, mock_llm):
        """Test chat with conversation history."""
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
        # Check that history was included in the call
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        assert len(messages) >= 3  # history + current

    def test_chat_with_summary(self, test_client, mock_llm):
        """Test chat with conversation summary."""
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
        # Check that summary context was included
        call_args = mock_llm.chat.call_args
        messages = call_args[0][0]
        assert any("Previous conversation summary" in str(m) for m in messages)


class TestChatStreamEndpoint:
    """Tests for the /chat/stream endpoint."""

    def test_chat_stream(self, test_client, mock_llm, mock_rag):
        """Test streaming chat endpoint."""

        def mock_stream(*args, **kwargs):
            yield "Hello"
            yield " world"

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

    def test_summarize_conversation(self, test_client, mock_llm):
        """Test summarizing a conversation."""
        mock_llm.chat.return_value = "User asked about Python."

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

    def test_chat_llm_error(self, test_client, mock_llm):
        """Test that LLM errors are handled gracefully."""
        from obsidian_chat.llm import LLMError

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
