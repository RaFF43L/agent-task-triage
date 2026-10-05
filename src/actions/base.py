from typing import Any

from pydantic import BaseModel, Field

from ai.states import TriageCategory, TriagePriority


class TriageRequest(BaseModel):
    """Request body for the triage endpoint."""

    message: str = Field(
        ...,
        min_length=1,
        description="User message to be triaged",
        examples=["My system is down and I can't work!"],
    )


class TriageResponse(BaseModel):
    """Response from the triage endpoint."""

    category: TriageCategory
    priority: TriagePriority
    reasoning: str
    response: str
    # Flow metadata (tool + quality + retry)
    work_result: str
    quality_approved: bool
    rejection_reason: str
    attempts: int
    # Global metadata accumulated by nodes
    metadata: dict[str, Any] = {}
