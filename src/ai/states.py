from typing import Literal, Annotated, Any
from typing_extensions import TypedDict
from pydantic import BaseModel, Field


def merge_metadata(
    left: dict[str, Any] | None, right: dict[str, Any] | None
) -> dict[str, Any]:
    """Reducer: merges metadata that each node adds to the global state."""
    return {**(left or {}), **(right or {})}


# The 2 possible triage classifications.
TriageCategory = Literal["financial", "software"]

TriagePriority = Literal["low", "medium", "high", "urgent"]


class TriageResult(BaseModel):
    """Structured output from Bedrock model classification."""

    category: TriageCategory = Field(
        description="Request category: 'financial' or 'software'"
    )
    priority: TriagePriority = Field(description="Service priority")
    reasoning: str = Field(description="Short justification for the classification")


class QualityResult(BaseModel):
    """Structured output from quality node."""

    approved: bool = Field(
        description="True if the tool result solves the request with quality"
    )
    rejection_reason: str = Field(
        default="",
        description="If rejected, objectively explains what needs to be fixed",
    )


class TriageState(TypedDict, total=False):
    """State that travels through the graph nodes."""

    # Input
    message: str

    # Produced by triage
    category: TriageCategory
    priority: TriagePriority
    reasoning: str

    # Produced by tools (financial/software)
    work_result: str

    # Produced/controlled by quality node and retry loop
    quality_approved: bool
    rejection_reason: str
    attempts: int

    # Global metadata: each node adds/updates keys here (merge via reducer)
    metadata: Annotated[dict[str, Any], merge_metadata]

    # Final response
    response: str
