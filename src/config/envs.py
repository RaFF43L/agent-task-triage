from typing import Literal

from dotenv import find_dotenv, load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

dotenv_path = find_dotenv(usecwd=True)
if dotenv_path:
    load_dotenv(dotenv_path, override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        case_sensitive=False,
        extra="ignore",
    )

    # general
    app_name: str = "4dev-lang-graph"
    app_env: Literal["production", "stage", "dev"] = "dev"

    # logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_format: Literal["text", "json"] = "text"

    # llm — o transporte é DERIVADO de LLM_BASE_URL (strategy):
    #   vazio = Bedrock (credenciais AWS); localhost:11434 = Ollama;
    #   qualquer outro = API OpenAI-compatível (api.openai.com, vLLM, LM Studio...)
    aws_region: str = "us-east-1"
    llm_base_url: str = ""
    model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    model_fast: str = "us.anthropic.claude-haiku-4-5-20251001-v1:0"
    # Judge model (LLM-as-judge); default = model
    model_judge: str = ""
    # Chave para transportes OpenAI-compatíveis; default = OPENAI_API_KEY do ambiente
    llm_api_key: str | None = None
    # Bedrock native evaluator model ID (for finalize_node evaluation)
    bedrock_judge_model_id: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"

    # embeddings — transporte derivado de EMBED_BASE_URL (mesma regra do LLM)
    embed_base_url: str = ""
    embed_model: str = "amazon.titan-embed-text-v2:0"
    embed_dimensions: int = 1024

    # agent behavior
    max_tokens: int = 4096
    # Evaluation mode: 'simple' (fast, backward compatible) or 'detailed' (multi-dimensional)
    eval_mode: Literal["simple", "detailed"] = "simple"
    # Enable Bedrock native evaluation in finalize_node (before final response)
    enable_bedrock_judge: bool = True
    # Bedrock evaluation type: 'quality', 'accuracy', or 'helpfulness'
    bedrock_eval_type: str = "quality"
    # Output token limit for tools (financial/software). Short responses
    # drastically reduce perceived latency.
    tool_max_tokens: int = 700
    recursion_limit: int = 25

    # rag / pgvector
    # Connection URL in psycopg v3 format (langchain-postgres requires the driver).
    database_url: str = "postgresql+psycopg://postgres:postgres@localhost:5433/rag"
    rag_collection: str = "documents"
    # Postgres schema where pgvector tables (langchain_pg_*) live.
    # Changing schema = physically separate set of tables.
    rag_schema: str = "public"
    rag_chunk_size: int = 1000
    rag_chunk_overlap: int = 150
    rag_top_k: int = 4


envs = Settings()
