"""Schemas (Pydantic) and RAG business logic — transport-agnostic.

Mirrors the pattern of `actions/triage.py`: `api.py` only does HTTP transport and
delegates to these functions (`ingest_*` / `search_documents`).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from ai.rag import ingest_text, ingest_file, search
from config.logging import get_logger

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class IngestTextRequest(BaseModel):
    """Ingestion of plain text (without file)."""

    text: str = Field(..., min_length=1, description="Content to be indexed")
    source: str = Field(
        default="inline",
        description="Origin identifier (appears in metadata)",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Extra metadata per document"
    )
    collection: str | None = Field(
        default=None,
        description="Target collection (default: envs.rag_collection)",
    )
    schema_name: str | None = Field(
        default=None,
        alias="schema",
        description="Target Postgres schema — physically separate tables "
        "(default: envs.rag_schema)",
    )


class IngestResponse(BaseModel):
    """Result of an ingestion."""

    source: str
    chunks: int
    ids: list[str]


class SearchRequest(BaseModel):
    """Similarity search."""

    query: str = Field(..., min_length=1, description="User query")
    k: int | None = Field(default=None, ge=1, le=50, description="Number of chunks")
    filter: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Metadata filter (langchain-postgres). E.g.: "
            '{"category": "support"} or {"source": {"$in": ["a.md", "b.md"]}}'
        ),
    )
    collection: str | None = Field(
        default=None,
        description="Target collection (default: envs.rag_collection)",
    )
    schema_name: str | None = Field(
        default=None,
        alias="schema",
        description="Target Postgres schema (default: envs.rag_schema)",
    )


class SearchHit(BaseModel):
    content: str
    score: float
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    hits: list[SearchHit]


# --------------------------------------------------------------------------- #
# Business logic
# --------------------------------------------------------------------------- #
async def ingest_text_document(
    text: str,
    source: str,
    metadata: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> IngestResponse:
    """Indexes plain text."""
    result = ingest_text(
        text,
        source=source,
        extra_metadata=metadata,
        collection=collection,
        schema=schema,
    )
    return IngestResponse(**result)


async def ingest_uploaded_file(
    filename: str,
    data: bytes,
    metadata: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> IngestResponse:
    """Indexes an uploaded file (txt/md/pdf)."""
    result = ingest_file(
        filename,
        data,
        extra_metadata=metadata,
        collection=collection,
        schema=schema,
    )
    return IngestResponse(**result)


async def search_documents(
    query: str,
    k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> SearchResponse:
    """Searches for the most relevant chunks for the query (with optional filter)."""
    hits = search(
        query,
        k=k,
        metadata_filter=metadata_filter,
        collection=collection,
        schema=schema,
    )
    return SearchResponse(query=query, hits=[SearchHit(**h) for h in hits])
