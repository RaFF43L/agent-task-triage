"""Parsers de arquivo para ingestao no RAG.

Extraem texto puro de diferentes formatos. O texto retornado sera fatiado em
chunks e embedado na etapa de ingestao.
"""

from __future__ import annotations

import io
from pathlib import Path

from config.logging import get_logger

logger = get_logger(__name__)

# Extensoes suportadas -> descricao (usada em erros/validacao).
SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}


def parse_txt(data: bytes) -> str:
    """Decodifica texto puro (txt/markdown) tolerando encoding."""
    return data.decode("utf-8", errors="replace")


def parse_pdf(data: bytes) -> str:
    """Extrai texto de um PDF pagina a pagina usando pypdf."""
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(text)
        else:
            logger.warning(f"PDF: pagina {i + 1} sem texto extraivel (possivel scan).")
    return "\n\n".join(pages)


def parse_bytes(filename: str, data: bytes) -> str:
    """Roteia para o parser correto conforme a extensao do arquivo."""
    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Extensao '{ext}' nao suportada. Use: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    if ext == ".pdf":
        return parse_pdf(data)
    return parse_txt(data)


def parse_file(path: str | Path) -> str:
    """Le um arquivo do disco e extrai seu texto."""
    p = Path(path)
    return parse_bytes(p.name, p.read_bytes())
