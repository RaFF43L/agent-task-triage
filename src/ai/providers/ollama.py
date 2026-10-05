"""Provider Ollama: modelos open source locais (Llama, etc.).

Transporte derivado de `LLM_BASE_URL` / `EMBED_BASE_URL` apontando para
`localhost:11434` (ver `resolve_transport`). Modelos vêm das variáveis
unificadas `MODEL` / `MODEL_JUDGE` / `EMBED_MODEL`.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings

from config import envs

from .base import ModelProvider, ProviderRegistry


@ProviderRegistry.register
class OllamaProvider(ModelProvider):
    name = "ollama"

    def model_large(self, **kwargs):
        return ChatOllama(
            model=envs.model,
            base_url=envs.llm_base_url,
            **kwargs,
        )

    def model_fast(self, **kwargs):
        return ChatOllama(
            model=envs.model,
            base_url=envs.llm_base_url,
            **kwargs,
        )

    def model_judge(self, **kwargs):
        return ChatOllama(
            model=envs.model_judge or envs.model,
            base_url=envs.llm_base_url,
            **kwargs,
        )

    def embeddings(self, **kwargs):
        return OllamaEmbeddings(
            model=envs.embed_model,
            base_url=envs.embed_base_url,
            **kwargs,
        )
