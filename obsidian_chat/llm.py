"""LLM client for OpenAI-compatible APIs."""

from typing import Generator

import httpx

from .config import config


class LLMError(Exception):
    """Exception raised when LLM API call fails."""

    def __init__(self, message: str, status_code: int | None = None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class LLMClient:
    """Client for interacting with OpenAI-compatible LLM APIs."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.base_url = (base_url or config.llm_base_url).rstrip("/")
        self.model = model or config.llm_model
        self.api_key = api_key or config.llm_api_key
        self.client = httpx.Client(timeout=120.0)

    def chat(
        self,
        messages: list[dict],
        system_prompt: str | None = None,
        stream: bool = True,
    ) -> Generator[str, None, None] | str:
        """Send a chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            system_prompt: Optional system prompt to prepend.
            stream: Whether to stream the response.

        Yields:
            Response text chunks if streaming, otherwise returns full response.
        """
        all_messages = []
        if system_prompt:
            all_messages.append({"role": "system", "content": system_prompt})
        all_messages.extend(messages)

        payload = {
            "model": self.model,
            "messages": all_messages,
            "stream": stream,
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if stream:
            return self._stream_response(payload, headers)
        else:
            return self._get_response(payload, headers)

    def _stream_response(
        self, payload: dict, headers: dict
    ) -> Generator[str, None, None]:
        """Stream response from the API."""
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            ) as response:
                if response.status_code != 200:
                    error_text = response.read().decode()
                    raise LLMError(
                        f"LLM API error ({response.status_code}): {error_text[:200]}",
                        status_code=response.status_code,
                    )
                for line in response.iter_lines():
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            import json

                            chunk = json.loads(data)
                            # Check for error in chunk
                            if "error" in chunk:
                                raise LLMError(chunk["error"].get("message", "Unknown error"))
                            if content := chunk.get("choices", [{}])[0].get(
                                "delta", {}
                            ).get("content"):
                                yield content
                        except json.JSONDecodeError:
                            continue
        except httpx.ConnectError:
            raise LLMError(f"Cannot connect to LLM at {self.base_url}. Is it running?")
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out. The model may be overloaded.")
        except httpx.HTTPStatusError as e:
            raise LLMError(f"LLM HTTP error: {e.response.status_code}", status_code=e.response.status_code)

    def _get_response(self, payload: dict, headers: dict) -> str:
        """Get non-streaming response from the API."""
        try:
            response = self.client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=headers,
            )
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", {}).get("message", response.text[:200])
                except Exception:
                    error_msg = response.text[:200]
                raise LLMError(
                    f"LLM API error ({response.status_code}): {error_msg}",
                    status_code=response.status_code,
                )
            data = response.json()
            return data["choices"][0]["message"]["content"]
        except httpx.ConnectError:
            raise LLMError(f"Cannot connect to LLM at {self.base_url}. Is it running?")
        except httpx.TimeoutException:
            raise LLMError("LLM request timed out. The model may be overloaded.")

    def close(self):
        """Close the HTTP client."""
        self.client.close()
