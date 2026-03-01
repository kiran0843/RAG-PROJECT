"""Ollama API client for local LLM inference."""

import json
from typing import Generator

import requests

DEFAULT_MODEL = "mistral"
DEFAULT_TEMPERATURE = 0.1
DEFAULT_MAX_TOKENS = 2048


class OllamaError(Exception):
    """Raised when the Ollama API returns an error or the request fails."""

    pass


class OllamaClient:
    """Sync client for the local Ollama API."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = DEFAULT_MODEL,
        timeout: float = 120.0,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._temperature = temperature
        self._max_tokens = max_tokens

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Single completion. Returns stripped text. Raises OllamaError on failure."""
        payload = {
            "model": self._model,
            "prompt": (prompt or "").strip(),
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens if max_tokens is not None else self._max_tokens,
            },
        }
        if system:
            payload["system"] = (system or "").strip()

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(f"Could not connect to Ollama at {self._base_url}. Is it running?") from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(f"Ollama request timed out after {self._timeout}s") from e
        except requests.exceptions.HTTPError as e:
            msg = f"Ollama API error: {e}"
            if resp is not None and hasattr(resp, "text") and resp.text:
                try:
                    err_body = resp.json()
                    if "error" in err_body:
                        msg = f"Ollama API error: {err_body['error']}"
                except Exception:
                    msg = f"Ollama API error: {resp.status_code} {resp.text[:200]}"
            raise OllamaError(msg) from e

        data = resp.json()
        if "error" in data:
            raise OllamaError(f"Ollama returned error: {data['error']}")
        return (data.get("response") or "").strip()

    def generate_stream(
        self,
        prompt: str,
        *,
        system: str | None = None,
        temperature: float | None = None,
    ) -> Generator[str, None, None]:
        """Stream completion. Yields text chunks."""
        payload = {
            "model": self._model,
            "prompt": (prompt or "").strip(),
            "stream": True,
            "options": {"temperature": temperature if temperature is not None else self._temperature},
        }
        if system:
            payload["system"] = (system or "").strip()

        try:
            resp = requests.post(
                f"{self._base_url}/api/generate",
                json=payload,
                stream=True,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(f"Could not connect to Ollama at {self._base_url}. Is it running?") from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(f"Ollama request timed out after {self._timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise OllamaError(f"Ollama API error: {e}") from e

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in data:
                raise OllamaError(f"Ollama returned error: {data['error']}")
            if data.get("response"):
                yield data["response"]
            if data.get("done"):
                break

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Chat completion. messages: list of {role, content} dicts. Returns stripped text."""
        payload = {
            "model": self._model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature if temperature is not None else self._temperature,
                "num_predict": max_tokens if max_tokens is not None else self._max_tokens,
            },
        }
        try:
            resp = requests.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=self._timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise OllamaError(f"Could not connect to Ollama at {self._base_url}. Is it running?") from e
        except requests.exceptions.Timeout as e:
            raise OllamaError(f"Ollama request timed out after {self._timeout}s") from e
        except requests.exceptions.HTTPError as e:
            raise OllamaError(f"Ollama API error: {e}") from e

        data = resp.json()
        if "error" in data:
            raise OllamaError(f"Ollama returned error: {data['error']}")
        msg = data.get("message", {})
        return (msg.get("response") or "").strip()
