"""create pgvector extension and langchain-postgres tables

Revision ID: 001
Revises: None
Create Date: 2026-09-23

Formalizes the RAG schema that was previously created at runtime by
`langchain-postgres` (PGVector). The tables follow exactly the layout that the
library expects, so runtime creation becomes a no-op (uses if_not_exists).
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, UUID
from pgvector.sqlalchemy import Vector

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # pgvector extension (idempotent).
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Collections (one per logical document set).
    op.create_table(
        "langchain_pg_collection",
        sa.Column("uuid", UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("cmetadata", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("uuid"),
        sa.UniqueConstraint("name"),
        if_not_exists=True,
    )

    # Embeddings (1024 dims -> Titan Text v2).
    op.create_table(
        "langchain_pg_embedding",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("collection_id", UUID(as_uuid=True), nullable=True),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("document", sa.String(), nullable=True),
        sa.Column("cmetadata", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["collection_id"],
            ["langchain_pg_collection.uuid"],
            ondelete="CASCADE",
        ),
        if_not_exists=True,
    )

    # GIN index for metadata filters (cmetadata->>'key').
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_cmetadata_gin "
        "ON langchain_pg_embedding USING gin (cmetadata jsonb_path_ops)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_cmetadata_gin")
    op.drop_table("langchain_pg_embedding")
    op.drop_table("langchain_pg_collection")
