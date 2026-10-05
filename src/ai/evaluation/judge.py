"""LLM-as-judge implementation for evaluating agent responses."""

from typing import Literal
from langchain_core.messages import HumanMessage, SystemMessage

from ai.llm import llm_factory
from ai.evaluation.models import (
    DetailedEvaluation,
    EvaluationResult,
)
from ai.evaluation.prompts import (
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_PROMPT,
    SIMPLE_JUDGE_SYSTEM_PROMPT,
    SIMPLE_JUDGE_USER_PROMPT,
)
from config.logging import get_logger

logger = get_logger(__name__)


def evaluate_response_detailed(
    message: str,
    work_result: str,
    category: Literal["financial", "software"],
    priority: str = "medium",
    attempts: int = 1,
) -> DetailedEvaluation:
    """Perform comprehensive multi-dimensional evaluation using LLM-as-judge.
    
    This function uses a judge LLM to evaluate the agent response across
    multiple dimensions (relevance, accuracy, completeness, clarity, tone,
    conciseness) and provides detailed scoring and feedback.
    
    Args:
        message: Original customer request
        work_result: Agent's response to be evaluated
        category: Category of the request (financial or software)
        priority: Priority level of the request
        attempts: Number of attempts made to generate this response
        
    Returns:
        DetailedEvaluation with scores, approval status, and detailed feedback
    """
    logger.info(f"Running detailed LLM-as-judge evaluation (attempt {attempts})...")
    
    # Use model_judge for evaluation (more capable, temperature=0 for consistency)
    model = llm_factory.model_judge(temperature=0).with_structured_output(
        DetailedEvaluation
    )
    
    evaluation: DetailedEvaluation = model.invoke(
        [
            SystemMessage(content=JUDGE_SYSTEM_PROMPT),
            HumanMessage(
                content=JUDGE_USER_PROMPT.format(
                    message=message,
                    category=category,
                    work_result=work_result,
                    priority=priority,
                    attempts=attempts,
                )
            ),
        ]
    )
    
    logger.info(
        f"Detailed evaluation complete: overall_score={evaluation.overall_score:.2f}, "
        f"approved={evaluation.approved}"
    )
    logger.debug(f"Evaluation summary: {evaluation.summary}")
    
    return evaluation


def evaluate_response(
    message: str,
    work_result: str,
    category: Literal["financial", "software"],
    priority: str = "medium",
    attempts: int = 1,
    mode: Literal["simple", "detailed"] = "simple",
) -> EvaluationResult:
    """Evaluate agent response using LLM-as-judge.
    
    This is the main evaluation function that can run in two modes:
    - simple: Fast evaluation with basic pass/fail and feedback
    - detailed: Comprehensive multi-dimensional evaluation
    
    Args:
        message: Original customer request
        work_result: Agent's response to be evaluated
        category: Category of the request (financial or software)
        priority: Priority level of the request
        attempts: Number of attempts made to generate this response
        mode: Evaluation mode (simple or detailed)
        
    Returns:
        EvaluationResult with approval status and feedback
    """
    if mode == "detailed":
        # Run detailed evaluation
        detailed = evaluate_response_detailed(
            message=message,
            work_result=work_result,
            category=category,
            priority=priority,
            attempts=attempts,
        )
        
        # Convert to EvaluationResult
        rejection_reason = ""
        if not detailed.approved:
            # Combine critical issues into rejection reason
            if detailed.critical_issues:
                rejection_reason = "Critical issues:\n" + "\n".join(
                    f"- {issue}" for issue in detailed.critical_issues
                )
            else:
                # Fallback: use improvement suggestions
                rejection_reason = "Necessary improvements:\n" + "\n".join(
                    f"- {sugg}" for sugg in detailed.improvement_suggestions[:3]
                )
        
        # Build detailed feedback for logging
        feedback_parts = [
            f"Overall Score: {detailed.overall_score:.2f}/5.0",
            f"- Relevance: {detailed.relevance.score}/5",
            f"- Accuracy: {detailed.accuracy.score}/5",
            f"- Completeness: {detailed.completeness.score}/5",
            f"- Clarity: {detailed.clarity.score}/5",
            f"- Tone: {detailed.tone.score}/5",
            f"- Conciseness: {detailed.conciseness.score}/5",
            f"\nSummary: {detailed.summary}",
        ]
        
        return EvaluationResult(
            approved=detailed.approved,
            rejection_reason=rejection_reason,
            overall_score=detailed.overall_score,
            detailed_feedback="\n".join(feedback_parts),
        )
    
    else:
        # Simple mode - fast evaluation compatible with existing system
        logger.info(f"Running simple LLM-as-judge evaluation (attempt {attempts})...")
        
        model = llm_factory.model_fast(temperature=0).with_structured_output(
            EvaluationResult
        )
        
        result: EvaluationResult = model.invoke(
            [
                SystemMessage(content=SIMPLE_JUDGE_SYSTEM_PROMPT),
                HumanMessage(
                    content=SIMPLE_JUDGE_USER_PROMPT.format(
                        message=message,
                        category=category,
                        work_result=work_result,
                    )
                ),
            ]
        )
        
        logger.info(f"Simple evaluation: approved={result.approved}")
        
        return result
