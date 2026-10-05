"""Triage business logic — transport framework agnostic.

Mirrors the role of `actions/chat.py` from the reference project: receives
validated input and orchestrates the agent (LangGraph graph). Can be called by
FastAPI, a worker, a CLI, etc.

Exposes two consumption forms:
- `process_triage`: blocking, returns the complete `TriageResponse`.
- `process_triage_stream`: asynchronous (generator) that emits streaming
  events (tool tokens + lifecycle) for SSE.
"""

import asyncio
import json
from typing import Any, AsyncGenerator

from actions.base import TriageResponse
from ai.agent import triage_agent
from config.envs import envs
from config.logging import get_logger

logger = get_logger(__name__)


async def process_triage(message: str) -> TriageResponse:
    """Executes the triage graph for a message and returns the result."""
    logger.info(f"Processing triage: {message[:80]}")

    result = await triage_agent.ainvoke(
        {"message": message},
        config={"recursion_limit": envs.recursion_limit},
    )

    return TriageResponse(
        category=result["category"],
        priority=result["priority"],
        reasoning=result["reasoning"],
        response=result["response"],
        work_result=result.get("work_result", ""),
        quality_approved=result.get("quality_approved", False),
        rejection_reason=result.get("rejection_reason", ""),
        attempts=result.get("attempts", 0),
        metadata=result.get("metadata", {}),
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    """Formats a line in the Server-Sent Events protocol."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Nodes whose tools produce the text transmitted to the user.
_TOOL_NODES = {"financial", "software"}


# Size of each chunk when "re-transmitting" the final approved response, creating
# the typing effect without exposing intermediate content (which may be rejected).
_REPLAY_CHUNK_SIZE = 6
_REPLAY_DELAY_S = 0.012


async def process_triage_stream(message: str) -> AsyncGenerator[str, None]:
    """Executes the graph and transmits ONLY the final approved response.

    Rationale: quality validation only occurs AFTER the tool finishes generating.
    Therefore, transmitting raw tool tokens live would show text that may be
    rejected and regenerated. Instead, during tool+quality we emit only status
    events (spinner), and only at the end — when we have the approved response
    sanitized by `finalize_node` — do we stream it, re-chunked to maintain the
    typing effect.

    Events emitted:
    - `triage`: {category, priority, reasoning} as soon as classification is done.
    - `attempt`: {attempts} when a new tool attempt begins.
    - `status`: {phase} signals the current phase (processing/reviewing/retrying).
    - `quality`: {approved, rejection_reason} after each review.
    - `token`: {text} chunks ONLY of the final approved response.
    - `done`: complete final payload (serialized TriageResponse).
    - `error`: {detail} in case of failure.
    """
    logger.info(f"Streaming triage: {message[:80]}")

    final_state: dict[str, Any] = {}
    current_attempt = 0

    try:
        async for event in triage_agent.astream_events(
            {"message": message},
            config={"recursion_limit": envs.recursion_limit},
            version="v2",
        ):
            kind = event["event"]
            name = event.get("name", "")

            # 1) Classification completed
            if kind == "on_chain_end" and name == "triage":
                out = event["data"].get("output") or {}
                yield _sse(
                    "triage",
                    {
                        "category": out.get("category"),
                        "priority": out.get("priority"),
                        "reasoning": out.get("reasoning"),
                    },
                )

            # 2) New tool attempt started — status only, no content.
            elif kind == "on_chain_start" and name in _TOOL_NODES:
                current_attempt += 1
                yield _sse("attempt", {"attempts": current_attempt})
                phase = "processing" if current_attempt == 1 else "retrying"
                yield _sse("status", {"phase": phase, "attempts": current_attempt})

            # 3) Quality review started — status.
            elif kind == "on_chain_start" and name == "quality":
                yield _sse("status", {"phase": "reviewing"})

            # 4) Quality review result
            elif kind == "on_chain_end" and name == "quality":
                out = event["data"].get("output") or {}
                yield _sse(
                    "quality",
                    {
                        "approved": out.get("quality_approved", False),
                        "rejection_reason": out.get("rejection_reason", ""),
                    },
                )

            # 5) Final graph state
            elif kind == "on_chain_end" and name == "LangGraph":
                final_state = event["data"].get("output") or {}

        # Only now do we have the final, approved, and sanitized response. We transmit
        # it re-chunked to give the typing effect (without risk of rejection).
        final_text = final_state.get("response", "")
        for i in range(0, len(final_text), _REPLAY_CHUNK_SIZE):
            yield _sse("token", {"text": final_text[i : i + _REPLAY_CHUNK_SIZE]})
            await asyncio.sleep(_REPLAY_DELAY_S)

        response = TriageResponse(
            category=final_state.get("category", "software"),
            priority=final_state.get("priority", "medium"),
            reasoning=final_state.get("reasoning", ""),
            response=final_text,
            work_result=final_state.get("work_result", ""),
            quality_approved=final_state.get("quality_approved", False),
            rejection_reason=final_state.get("rejection_reason", ""),
            attempts=final_state.get("attempts", 0),
            metadata=final_state.get("metadata", {}),
        )
        yield _sse("done", response.model_dump())

    except Exception as exc:  # noqa: BLE001
        logger.exception("Error in triage streaming")
        yield _sse("error", {"detail": str(exc)})
