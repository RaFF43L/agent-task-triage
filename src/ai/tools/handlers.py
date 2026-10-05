"""Executor tools by category (financial / software).

Applied latency optimizations:
- Use the FAST model (Haiku) instead of the large one (Sonnet).  [B]
- Limit output size via `max_tokens`.                            [A]
- Concise prompts asking for short responses.                    [A]
- Consume the model via `astream` to allow token-by-token
  streaming when executed within the graph.                      [D]

When there's a previous rejected attempt, the `rejection_reason` is injected
into the prompt for correction.
"""

from langchain_core.messages import HumanMessage

from ai.llm import llm_factory
from ai.prompts import (
    FINANCIAL_TOOL_PROMPT,
    SOFTWARE_TOOL_PROMPT,
    RETRY_FEEDBACK_TEMPLATE,
)
from ai.utils import chunk_to_text
from config.envs import envs
from config.logging import get_logger

logger = get_logger(__name__)


def _build_feedback(rejection_reason: str) -> str:
    if rejection_reason:
        return RETRY_FEEDBACK_TEMPLATE.format(rejection_reason=rejection_reason)
    return ""


async def _run_tool(prompt: str) -> str:
    """Executes the fast model in streaming mode and accumulates the final text.

    Using `astream` (instead of `invoke`) is what allows the graph to emit
    `on_chat_model_stream` events, enabling SSE in the endpoint.
    """
    model = llm_factory.model_fast(
        temperature=0.3,
        max_tokens=envs.tool_max_tokens,
    )
    parts: list[str] = []
    async for chunk in model.astream([HumanMessage(content=prompt)]):
        parts.append(chunk_to_text(chunk.content))
    return "".join(parts)


async def financial_tool(message: str, rejection_reason: str = "") -> str:
    """Resolves requests from the financial category (Financial Support Node)."""
    logger.info("Executing tool: financial (Haiku, streaming)")
    prompt = FINANCIAL_TOOL_PROMPT.format(
        message=message,
        feedback=_build_feedback(rejection_reason),
    )
    return await _run_tool(prompt)


async def software_tool(message: str, rejection_reason: str = "") -> str:
    """Resolves requests from the software category (Technical Support Node)."""
    logger.info("Executing tool: software (Haiku, streaming)")
    prompt = SOFTWARE_TOOL_PROMPT.format(
        message=message,
        feedback=_build_feedback(rejection_reason),
    )
    return await _run_tool(prompt)
