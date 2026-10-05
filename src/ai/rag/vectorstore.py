"""Vector store (Postgres + pgvector) for the RAG pipeline.

Encapsulates the `PGVector` instance from `langchain-postgres`, embedding with
the Titan v2 model (via `LLMFactory.embeddings`). The store is created on demand
(lazy singleton per collection+schema) to avoid opening a database connection on
module import.

About "tables": `langchain-postgres` uses TWO FIXED tables
(`langchain_pg_collection` and `langchain_pg_embedding`) whose names are
hardcoded in the lib (no `table_name` parameter). To isolate data there are two
levels:

- `collection`: a logical subset of rows within the fixed tables
  (one row in `langchain_pg_collection` + the `collection_id` in embeddings).
- `schema`: a Postgres NAMESPACE. Changing the schema makes the pair of tables
  `langchain_pg_*` be created PHYSICALLY separate (e.g.: `faq.langchain_pg_embedding`
  vs `public.langchain_pg_embedding`). We achieve this by fixing the `search_path`
  of the connection via `engine_args`.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_postgres import PGVector
from sqlalchemy import create_engine, text

from ai.llm import llm_factory
from config.envs import envs
from config.logging import get_logger

logger = get_logger(__name__)


def _ensure_schema(schema: str) -> None:
    """Creates the Postgres schema if it doesn't exist (idempotent)."""
    if schema == "public":
        return
    engine = create_engine(envs.database_url)
    try:
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{schema}"'))
    finally:
        engine.dispose()


@lru_cache(maxsize=16)
def get_vectorstore(
    collection: str | None = None,
    schema: str | None = None,
) -> PGVector:
    """Returns the PGVector instance for (collection, schema).

    Creates the schema and tables (`langchain_pg_*`) on demand, if necessary.
    Cached by (collection, schema) to reuse connections.
    """
    coll = collection or envs.rag_collection
    sch = schema or envs.rag_schema

    logger.info(
        f"Initializing PGVector (collection={coll}, schema={sch}, "
        f"embed={envs.embed_model})"
    )

    _ensure_schema(sch)

    # Fixes the connection's search_path to the target schema: the fixed tables from
    # langchain-postgres (langchain_pg_*) will be created/queried there.
    engine_args = {"connect_args": {"options": f"-csearch_path={sch}"}}

    return PGVector(
        embeddings=llm_factory.embeddings(),
        collection_name=coll,
        connection=envs.database_url,
        use_jsonb=True,
        engine_args=engine_args,
    )
