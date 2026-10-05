"""Abstração de providers de modelos (LLM + embeddings).

`ModelProvider` é a interface agnóstica: o resto do código depende dela, não de
implementações concretas. `ProviderRegistry` mapeia nome -> classe; cada provider
se auto-registra com o decorator `@ProviderRegistry.register`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar
from urllib.parse import urlparse

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel


def resolve_transport(base_url: str) -> str:
    """Deriva o transporte a partir do base_url (strategy).

    - vazio -> `bedrock` (credenciais AWS, sem base_url)
    - `localhost`/`127.0.0.1:11434` -> `ollama`
    - qualquer outro -> `openai` (API compatível: api.openai.com, vLLM,
      LM Studio, OpenRouter...)

    No futuro, este mapeamento pode virar uma config explícita (+1); por ora
    vive na env: basta definir `LLM_BASE_URL` / `EMBED_BASE_URL`.
    """
    if not base_url:
        return "bedrock"
    parsed = urlparse(base_url)
    host = parsed.hostname or ""
    if host in ("localhost", "127.0.0.1") and parsed.port == 11434:
        return "ollama"
    return "openai"


class ModelProvider(ABC):
    """Interface de um provider de modelos (chat + embeddings).

    Implementações concretas (Bedrock, Ollama, OpenAI, ...) leem suas próprias
    configurações e constroem os modelos. O restante do código só conhece esta
    abstração.
    """

    name: str

    @abstractmethod
    def model_large(self, **kwargs) -> BaseChatModel:
        """Modelo principal (mais capaz) para respostas do agente."""

    @abstractmethod
    def model_fast(self, **kwargs) -> BaseChatModel:
        """Modelo rápido/barato para classificação/triagem."""

    @abstractmethod
    def model_judge(self, **kwargs) -> BaseChatModel:
        """Modelo avaliador (LLM-as-judge), configurável por provider."""

    @abstractmethod
    def embeddings(self, **kwargs) -> Embeddings:
        """Modelo de embeddings para RAG."""


class ProviderRegistry:
    """Registro de providers por nome. Novo provider = nova classe registrada."""

    _providers: ClassVar[dict[str, type[ModelProvider]]] = {}
    _instances: ClassVar[dict[str, ModelProvider]] = {}

    @classmethod
    def register(cls, provider_cls: type[ModelProvider]) -> type[ModelProvider]:
        """Decorator: registra a classe sob `provider_cls.name`."""
        cls._providers[provider_cls.name] = provider_cls
        return provider_cls

    @classmethod
    def get(cls, name: str) -> ModelProvider:
        """Retorna a instância (singleton) do provider pelo nome."""
        try:
            return cls._instances.setdefault(name, cls._providers[name]())
        except KeyError:
            raise ValueError(
                f"Provider desconhecido: {name!r}. Disponíveis: {sorted(cls._providers)}"
            ) from None

    @classmethod
    def available(cls) -> list[str]:
        return sorted(cls._providers)
