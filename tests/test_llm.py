"""Tests for the LLM client module."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from obsidian_chat.llm import LLMClient, LLMError


class TestLLMClient:
    """Tests for LLMClient class."""

    def test_client_initialization(self):
        """Test client initializes with provided values."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        assert client.base_url == "http://test:8000/v1"
        assert client.model == "test-model"
        assert client.api_key == "test-key"

    def test_client_strips_trailing_slash(self):
        """Test that trailing slash is stripped from base URL."""
        client = LLMClient(base_url="http://test:8000/v1/")

        assert client.base_url == "http://test:8000/v1"

    def test_chat_non_streaming(self, mock_llm_response):
        """Test non-streaming chat request."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_llm_response

        with patch.object(client.client, "post", return_value=mock_response):
            result = client.chat(
                messages=[{"role": "user", "content": "Hello"}],
                stream=False,
            )

        assert result == "This is a test response from the LLM."

    def test_chat_with_system_prompt(self, mock_llm_response):
        """Test that system prompt is included in messages."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_llm_response

        with patch.object(client.client, "post", return_value=mock_response) as mock_post:
            client.chat(
                messages=[{"role": "user", "content": "Hello"}],
                system_prompt="You are helpful.",
                stream=False,
            )

            # Check that system message was included
            call_args = mock_post.call_args
            payload = call_args.kwargs["json"]
            assert payload["messages"][0]["role"] == "system"
            assert payload["messages"][0]["content"] == "You are helpful."

    def test_chat_streaming(self, mock_llm_stream_response):
        """Test streaming chat request."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = mock_llm_stream_response

        with patch.object(client.client, "stream") as mock_stream:
            mock_stream.return_value.__enter__.return_value = mock_response

            chunks = list(
                client.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=True,
                )
            )

        assert chunks == ["Hello", " world", "!"]

    def test_connection_error_handling(self):
        """Test that connection errors are handled gracefully."""
        client = LLMClient(
            base_url="http://nonexistent:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        with patch.object(
            client.client, "post", side_effect=httpx.ConnectError("Connection refused")
        ):
            with pytest.raises(LLMError) as exc_info:
                client.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "Cannot connect" in str(exc_info.value)

    def test_timeout_error_handling(self):
        """Test that timeout errors are handled gracefully."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        with patch.object(
            client.client, "post", side_effect=httpx.TimeoutException("Timeout")
        ):
            with pytest.raises(LLMError) as exc_info:
                client.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "timed out" in str(exc_info.value)

    def test_http_error_handling(self):
        """Test that HTTP errors return meaningful messages."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_response.json.side_effect = Exception("Not JSON")

        with patch.object(client.client, "post", return_value=mock_response):
            with pytest.raises(LLMError) as exc_info:
                client.chat(
                    messages=[{"role": "user", "content": "Hello"}],
                    stream=False,
                )

            assert "500" in str(exc_info.value)

    def test_client_close(self):
        """Test that client can be closed."""
        client = LLMClient(
            base_url="http://test:8000/v1",
            model="test-model",
            api_key="test-key",
        )

        with patch.object(client.client, "close") as mock_close:
            client.close()
            mock_close.assert_called_once()


class TestLLMError:
    """Tests for LLMError exception."""

    def test_error_with_message(self):
        """Test LLMError with just a message."""
        error = LLMError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.status_code is None

    def test_error_with_status_code(self):
        """Test LLMError with message and status code."""
        error = LLMError("Test error", status_code=500)

        assert error.message == "Test error"
        assert error.status_code == 500
