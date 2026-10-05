from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END

from ai.llm import llm_factory
from ai.states import TriageState, TriageResult, QualityResult
from ai.tools import financial_tool, software_tool
from ai.utils import sanitize_customer_message
from ai.evaluation import evaluate_response
from ai.evaluation.bedrock_evaluator import bedrock_evaluator
from ai.prompts import (
    TRIAGE_SYSTEM_PROMPT,
    NOT_APPROVED_NOTE,
)
from config import envs
from config.logging import get_logger

logger = get_logger(__name__)

MAX_ATTEMPTS = 3


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
def triage_node(state: TriageState) -> dict:
    """Classifies the message as 'financial' or 'software' (structured output)."""
    logger.info("Executing message triage...")

    model = llm_factory.model_fast(temperature=0).with_structured_output(TriageResult)
    result: TriageResult = model.invoke(
        [
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=state["message"]),
        ]
    )

    logger.info(f"Triage: category={result.category} priority={result.priority}")
    return {
        "category": result.category,
        "priority": result.priority,
        "reasoning": result.reasoning,
        "attempts": 0,
        "rejection_reason": "",
        "metadata": {
            "triage_category": result.category,
            "triage_priority": result.priority,
        },
    }


async def financial_node(state: TriageState) -> dict:
    """Financial Support Node: simulates invoice/payment/refund verification.

    Updates metadata in the global state recording the processing.
    """
    attempts = state.get("attempts", 0) + 1
    work_result = await financial_tool(
        message=state["message"],
        rejection_reason=state.get("rejection_reason", ""),
    )
    return {
        "work_result": work_result,
        "attempts": attempts,
        "metadata": {
            "handled_by": "financial_support_node",
            "simulated_action": "invoice/payment/refund verification",
            "financial_attempts": attempts,
        },
    }


async def software_node(state: TriageState) -> dict:
    """Technical Support Node: simulates processing and finding resolutions for
    infrastructure or software problems.

    Updates metadata in the global state recording the processing.
    """
    attempts = state.get("attempts", 0) + 1
    work_result = await software_tool(
        message=state["message"],
        rejection_reason=state.get("rejection_reason", ""),
    )
    return {
        "work_result": work_result,
        "attempts": attempts,
        "metadata": {
            "handled_by": "technical_support_node",
            "simulated_action": "infrastructure/software diagnosis and resolution",
            "software_attempts": attempts,
        },
    }


def quality_node(state: TriageState) -> dict:
    """Validates tool result using LLM-as-judge. Approves or rejects with rejection_reason.
    
    Uses the new LLM-as-judge evaluation system that can operate in two modes:
    - simple: Fast evaluation compatible with the previous system
    - detailed: Multi-dimensional evaluation with detailed scores
    
    The mode can be configured via EVAL_MODE environment variable.
    """
    logger.info(f"Validating quality (attempt {state.get('attempts', 0)})...")

    # Use evaluate_response from the new LLM-as-judge evaluation system
    eval_mode = getattr(envs, "eval_mode", "simple")
    
    result = evaluate_response(
        message=state["message"],
        work_result=state["work_result"],
        category=state["category"],
        priority=state.get("priority", "medium"),
        attempts=state.get("attempts", 1),
        mode=eval_mode,
    )

    logger.info(f"Quality: approved={result.approved}")
    if result.overall_score > 0:
        logger.info(f"Overall score: {result.overall_score:.2f}/5.0")
    
    if result.detailed_feedback:
        logger.debug(f"Detailed feedback:\n{result.detailed_feedback}")

    return {
        "quality_approved": result.approved,
        "rejection_reason": result.rejection_reason if not result.approved else "",
        "metadata": {
            "quality_approved": result.approved,
            "quality_checked_at_attempt": state.get("attempts", 0),
            "quality_score": result.overall_score,
            "eval_mode": eval_mode,
        },
    }


def finalize_node(state: TriageState) -> dict:
    """Assembles final response to user with optional Bedrock evaluation.
    
    If ENABLE_BEDROCK_JUDGE=True, evaluates the response using Bedrock native
    evaluator before returning to user. This adds a final layer of quality
    validation using AWS evaluation models.
    
    The resolution has already been generated (and possibly transmitted via streaming)
    by the tool. Here we format and, optionally, evaluate with Bedrock before
    returning.
    """
    # Safety net: remove any internal preamble leaked by the LLM.
    work_result = sanitize_customer_message(state.get("work_result", ""))
    approved = state.get("quality_approved", False)
    
    # Prepare initial response
    if approved:
        response = work_result
    else:
        response = work_result + NOT_APPROVED_NOTE.format(
            attempts=state.get("attempts", 0)
        )
    
    # Apply Bedrock native evaluation if enabled (final judge before user response)
    bedrock_eval_result = {}
    if envs.enable_bedrock_judge and approved:
        logger.info("Applying Bedrock native evaluation in finalize_node...")
        try:
            bedrock_eval_result = bedrock_evaluator.evaluate_response(
                prompt=state["message"],
                response=work_result,
                category=state["category"],
                evaluation_type=envs.bedrock_eval_type,
            )
            
            # If Bedrock judge rejects, add warning to response
            if not bedrock_eval_result.get("approved", True):
                logger.warning(
                    f"Bedrock judge rejected response: {bedrock_eval_result.get('reasoning', 'No reason')}"
                )
                response = work_result + (
                    "\n\n---\n"
                    "⚠️ This response has been flagged by the final quality evaluation "
                    "and will be reviewed by our team. "
                    f"Reason: {bedrock_eval_result.get('reasoning', 'Quality below expected')}"
                )
        except Exception as e:
            logger.error(f"Bedrock evaluation failed in finalize_node: {e}")
            # Continue with original response if evaluation fails

    return {
        "response": response,
        "metadata": {
            "bedrock_evaluation": bedrock_eval_result if bedrock_eval_result else None,
        },
    }


# --------------------------------------------------------------------------- #
# Routers (conditional edges)
# --------------------------------------------------------------------------- #
def route_by_category(state: TriageState) -> Literal["financial", "software"]:
    """After triage, directs to the tool for the category."""
    return state["category"]


def route_after_quality(
    state: TriageState,
) -> Literal["finalize", "financial", "software"]:
    """Decides: approved -> finalize; rejected and < MAX -> back to tool; else finalize."""
    if state.get("quality_approved"):
        return "finalize"
    if state.get("attempts", 0) >= MAX_ATTEMPTS:
        logger.info("Attempt limit reached. Ending without approval.")
        return "finalize"
    logger.info("Rejected. Resending to tool with rejection_reason.")
    return state["category"]


# --------------------------------------------------------------------------- #
# Graph construction
# --------------------------------------------------------------------------- #
def build_triage_graph():
    """Graph: triage -> [financial|software] -> quality -> (loop|finalize) -> END."""
    graph = StateGraph(TriageState)

    graph.add_node("triage", triage_node)
    graph.add_node("financial", financial_node)
    graph.add_node("software", software_node)
    graph.add_node("quality", quality_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "triage")

    # Triage -> tool by category
    graph.add_conditional_edges(
        "triage",
        route_by_category,
        {"financial": "financial", "software": "software"},
    )

    # Each tool -> quality
    graph.add_edge("financial", "quality")
    graph.add_edge("software", "quality")

    # Quality -> finalize or back to tool (retry loop)
    graph.add_conditional_edges(
        "quality",
        route_after_quality,
        {
            "finalize": "finalize",
            "financial": "financial",
            "software": "software",
        },
    )

    graph.add_edge("finalize", END)

    return graph.compile()


# Compiled graph, ready for use.
triage_agent = build_triage_graph()
