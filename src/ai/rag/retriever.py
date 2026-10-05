"""Retrieval in RAG: similarity search in pgvector."""

from __future__ import annotations

from typing import Any

from ai.rag.vectorstore import get_vectorstore
from config.envs import envs
from config.logging import get_logger

logger = get_logger(__name__)


def search(
    query: str,
    k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> list[dict[str, Any]]:
    """Searches for the k chunks most similar to query.

    Returns a list of dicts with `content`, `score` (0-1, higher = better) and
    `metadata` (source, chunk_index, etc.).

    `metadata_filter` (optional) restricts the search by metadata before measuring
    similarity. Accepts langchain-postgres mini-language, e.g.:
    `{"category": "support"}` or `{"source": {"$in": ["a.md", "b.md"]}}`.

    `collection`/`schema` (optional) select the set of tables/collection
    to search in (see `get_vectorstore`).
    """
    top_k = k or envs.rag_top_k
    logger.info(
        f"RAG search (k={top_k}, filter={metadata_filter or '-'}): {query[:80]}"
    )

    results = get_vectorstore(collection, schema).similarity_search_with_relevance_scores(
        query, k=top_k, filter=metadata_filter or None
    )
    return [
        {
            "content": doc.page_content,
            "score": float(score),
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]


def build_context(query: str, k: int | None = None) -> str:
    """Builds a context block (text) with the retrieved chunks.

    Useful for injecting directly into an LLM prompt (RAG pattern).
    """
    hits = search(query, k=k)
    if not hits:
        return ""
    blocks = []
    for i, h in enumerate(hits, start=1):
        src = h["metadata"].get("source", "unknown")
        blocks.append(f"[Chunk {i} | source: {src}]\n{h['content']}")
    return "\n\n".join(blocks)
