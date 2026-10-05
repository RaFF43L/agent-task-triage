"""Data models for LLM-as-judge evaluation system."""

from typing import Literal
from pydantic import BaseModel, Field


class EvaluationCriteria(BaseModel):
    """Individual criterion evaluation with score and justification."""

    score: int = Field(
        ge=1,
        le=5,
        description="Score from 1 (poor) to 5 (excellent) for this criterion",
    )
    justification: str = Field(
        description="Brief explanation of why this score was given"
    )


class DetailedEvaluation(BaseModel):
    """Comprehensive multi-dimensional evaluation of agent response.
    
    This follows the LLM-as-judge pattern with multiple evaluation dimensions,
    each scored independently with detailed justification.
    """

    # Multi-dimensional scoring (1-5 scale)
    relevance: EvaluationCriteria = Field(
        description="Does the response directly address the user's request? "
        "Is it on-topic and focused?"
    )
    accuracy: EvaluationCriteria = Field(
        description="Is the information provided correct and factual? "
        "Are there any errors or misleading statements?"
    )
    completeness: EvaluationCriteria = Field(
        description="Does the response provide all necessary information? "
        "Are there clear next steps or action items?"
    )
    clarity: EvaluationCriteria = Field(
        description="Is the response clear, well-structured, and easy to understand? "
        "Is the language appropriate for the customer?"
    )
    tone: EvaluationCriteria = Field(
        description="Is the tone professional, courteous, and empathetic? "
        "Does it match the expected customer service standards?"
    )
    conciseness: EvaluationCriteria = Field(
        description="Is the response appropriately concise without being too brief or too verbose? "
        "Does it respect the customer's time?"
    )

    # Overall assessment
    overall_score: float = Field(
        ge=1.0,
        le=5.0,
        description="Weighted average of all criteria scores",
    )
    approved: bool = Field(
        description="True if the response meets minimum quality standards (overall_score >= 3.5)"
    )
    critical_issues: list[str] = Field(
        default_factory=list,
        description="List of critical issues that must be addressed if not approved",
    )
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="Specific actionable suggestions for improving the response",
    )

    # Summary
    summary: str = Field(
        description="Brief overall assessment summarizing the evaluation"
    )


class EvaluationResult(BaseModel):
    """Simplified evaluation result compatible with existing quality node.
    
    This is a backward-compatible interface that can be used as a drop-in
    replacement for the existing QualityResult model.
    """

    approved: bool = Field(
        description="True if the response meets quality standards"
    )
    rejection_reason: str = Field(
        default="",
        description="If rejected, explains what needs to be corrected",
    )
    overall_score: float = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Overall quality score from 1-5",
    )
    detailed_feedback: str = Field(
        default="",
        description="Optional detailed feedback for logging/monitoring",
    )
