"""Provider OpenAI-compatível: chat + embeddings via API.

Transporte derivado de `LLM_BASE_URL` / `EMBED_BASE_URL` (qualquer endpoint
OpenAI-compatível: api.openai.com, vLLM, LM Studio, OpenRouter...). A chave
vem de `LLM_API_KEY` ou, se vazia, de `OPENAI_API_KEY` do ambiente.
"""

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import envs

from .base import ModelProvider, ProviderRegistry


@ProviderRegistry.register
class OpenAIProvider(ModelProvider):
    name = "openai"

    def model_large(self, **kwargs):
        return ChatOpenAI(
            model=envs.model,
            base_url=envs.llm_base_url or None,
            api_key=envs.llm_api_key,
            **kwargs,
        )

    def model_fast(self, **kwargs):
        return ChatOpenAI(
            model=envs.model,
            base_url=envs.llm_base_url or None,
            api_key=envs.llm_api_key,
            **kwargs,
        )

    def model_judge(self, **kwargs):
        return ChatOpenAI(
            model=envs.model_judge or envs.model,
            base_url=envs.llm_base_url or None,
            api_key=envs.llm_api_key,
            **kwargs,
        )

    def embeddings(self, **kwargs):
        return OpenAIEmbeddings(
            model=envs.embed_model,
            base_url=envs.embed_base_url or None,
            api_key=envs.llm_api_key,
            dimensions=envs.embed_dimensions,
            **kwargs,
        )
