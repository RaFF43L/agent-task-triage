"""Actions for evaluation endpoints."""

from typing import Literal
from pydantic import BaseModel, Field

from ai.evaluation import evaluate_response, evaluate_response_detailed
from ai.evaluation.models import DetailedEvaluation, EvaluationResult


class EvaluationRequest(BaseModel):
    """Request to evaluate an agent response."""

    message: str = Field(description="Original customer request")
    work_result: str = Field(description="Agent response to evaluate")
    category: Literal["financial", "software"] = Field(
        description="Category of the request"
    )
    priority: str = Field(default="medium", description="Priority level")
    attempts: int = Field(default=1, ge=1, description="Attempt number")
    mode: Literal["simple", "detailed"] = Field(
        default="simple", description="Evaluation mode"
    )


__all__ = [
    "EvaluationRequest",
    "EvaluationResult",
    "DetailedEvaluation",
    "evaluate_response",
    "evaluate_response_detailed",
]
