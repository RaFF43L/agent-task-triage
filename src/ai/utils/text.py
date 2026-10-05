"""Text helpers for handling content returned by Bedrock.

`ChatBedrockConverse` can return `content` as a string or as a list
of blocks (e.g.: [{"type": "text", "text": "..."}]). This function normalizes
both formats to a simple string — useful both for accumulating tool responses
and for extracting text from streaming chunks.
"""

import re
from typing import Any


# Marker that some models insert before the final response ("... to customer:").
_CUSTOMER_MARKER = re.compile(
    r".*?(?:message\s+(?:final\s+)?to\s+customer|response\s+to\s+customer)\s*:?\s*",
    re.IGNORECASE | re.DOTALL,
)

# "Leaked" internal reasoning lines that should be removed from the beginning.
_INTERNAL_LINE = re.compile(
    r"^\s*(?:\**\s*)?(?:\[|\()?\s*"
    r"(?:simulation|internal\s+analysis|reasoning|"
    r"diagnosis|verification|backend\s+service|"
    r"identified\s+cause|solution|applied\s+action|"
    r"log[s]?)\b.*$",
    re.IGNORECASE,
)


def sanitize_customer_message(text: str) -> str:
    """Removes any 'internal reasoning' preamble leaked by the LLM.

    Safety net (prompts already ask for only the final message, but fast models
    sometimes print internal sections). Strategy:
    1. If there's an explicit marker like "MESSAGE TO CUSTOMER:", keep only
       what comes after it.
    2. Otherwise, discard initial lines that are clearly internal notes
       (Diagnosis, Simulation, logs, etc.) and separators '---'.
    """
    if not text:
        return text

    cleaned = text.strip()

    # (1) Explicit marker for start of message to customer.
    match = _CUSTOMER_MARKER.match(cleaned)
    if match:
        cleaned = cleaned[match.end():]
        # Remove residual markdown/punctuation from marker (e.g.: "**", ":") at top.
        cleaned = re.sub(r"^[\s*_:#>\-]+", "", cleaned).strip()

    # (2) Prune internal lines at the top, until the 1st "conversational" line.
    lines = cleaned.split("\n")
    start = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped in {"---", "***", "___"} or _INTERNAL_LINE.match(
            stripped
        ):
            start = i + 1
            continue
        break

    if start < len(lines):
        cleaned = "\n".join(lines[start:]).strip()

    return cleaned or text.strip()


def chunk_to_text(content: Any) -> str:
    """Extracts text from a `content` that can be str, list of blocks, or None."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(item.get("text") or item.get("content") or "")
        return "".join(parts)
    return str(content)
