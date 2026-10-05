"""Facade agnóstica de modelos.

Delega para o provider registrado em `LLM_PROVIDER` / `EMBED_PROVIDER`
(via `ProviderRegistry`). Nenhuma lógica de provider vive aqui — adicionar um
provider novo é criar uma classe em `ai/providers/` e registrá-la.
"""

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

from ai.providers import ProviderRegistry, resolve_transport
from config import envs
from config.logging import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """Facade: mesma API de sempre, transporte derivado do base_url."""

    @staticmethod
    def _chat_provider():
        return ProviderRegistry.get(resolve_transport(envs.llm_base_url))

    @staticmethod
    def model_large(**kwargs) -> BaseChatModel:
        """Main model (most capable) for agent responses."""
        return LLMFactory._chat_provider().model_large(**kwargs)

    @staticmethod
    def model_fast(**kwargs) -> BaseChatModel:
        """Fast/cheap model, ideal for classification/triage tasks."""
        return LLMFactory._chat_provider().model_fast(**kwargs)

    @staticmethod
    def model_judge(**kwargs) -> BaseChatModel:
        """Model for LLM-as-judge evaluation (most capable, temperature 0).

        Used to evaluate agent response quality with multiple criteria.
        Uses the main model (most capable) with temperature 0 for consistency
        in evaluations.
        """
        return LLMFactory._chat_provider().model_judge(**kwargs)

    @staticmethod
    def embeddings(**kwargs) -> Embeddings:
        """Embeddings model for RAG, transporte derivado de `EMBED_BASE_URL`."""
        return ProviderRegistry.get(resolve_transport(envs.embed_base_url)).embeddings(**kwargs)


llm_factory = LLMFactory()
