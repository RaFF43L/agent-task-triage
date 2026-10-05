"""Providers de modelos. Importar este pacote registra todos os providers."""

from . import bedrock, ollama, openai  # noqa: F401  (auto-registro)
from .base import ModelProvider, ProviderRegistry, resolve_transport

__all__ = ["ModelProvider", "ProviderRegistry", "resolve_transport"]
