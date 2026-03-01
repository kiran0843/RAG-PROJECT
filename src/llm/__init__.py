"""LLM interface via Ollama."""

from src.llm.ollama_client import OllamaClient, OllamaError

__all__ = ["OllamaClient", "OllamaError"]
