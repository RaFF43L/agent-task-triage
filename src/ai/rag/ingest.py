"""Document ingestion in RAG: chunk -> embed -> store (pgvector).

Flow:
1. Receives text (already extracted by a parser) or file bytes.
2. Splits into chunks with RecursiveCharacterTextSplitter.
3. Embeds (Titan v2) and writes to PGVector, attaching metadata (source, etc.).
"""

from __future__ import annotations

import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ai.attachments import parse_bytes
from ai.rag.vectorstore import get_vectorstore
from config.envs import envs
from config.logging import get_logger

logger = get_logger(__name__)


def _splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=envs.rag_chunk_size,
        chunk_overlap=envs.rag_chunk_overlap,
        add_start_index=True,
    )


def _make_documents(
    text: str, source: str, extra_metadata: dict[str, Any] | None = None
) -> list[Document]:
    """Splits the text and builds Documents with metadata per chunk."""
    chunks = _splitter().split_text(text)
    base = {"source": source, **(extra_metadata or {})}
    return [
        Document(
            page_content=chunk,
            metadata={**base, "chunk_index": i},
        )
        for i, chunk in enumerate(chunks)
    ]


def ingest_text(
    text: str,
    source: str,
    extra_metadata: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Ingests plain text. Returns summary (source, chunks, ids).

    `collection`/`schema` (optional) allow writing to an isolated
    table/collection set (see `get_vectorstore`).
    """
    if not text.strip():
        raise ValueError("Empty text: nothing to ingest.")

    documents = _make_documents(text, source, extra_metadata)
    ids = [str(uuid.uuid4()) for _ in documents]

    logger.info(f"Ingesting '{source}': {len(documents)} chunk(s).")
    get_vectorstore(collection, schema).add_documents(documents, ids=ids)

    return {"source": source, "chunks": len(documents), "ids": ids}


def ingest_file(
    filename: str,
    data: bytes,
    extra_metadata: dict[str, Any] | None = None,
    collection: str | None = None,
    schema: str | None = None,
) -> dict[str, Any]:
    """Extracts text from a file (txt/md/pdf) and ingests it."""
    text = parse_bytes(filename, data)
    return ingest_text(
        text,
        source=filename,
        extra_metadata=extra_metadata,
        collection=collection,
        schema=schema,
    )
