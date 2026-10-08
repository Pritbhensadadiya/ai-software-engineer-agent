from pydantic import BaseModel, Field


class TaskResponse(BaseModel):
    """Structured analysis of a software development task."""

    summary: str = Field(description="A concise summary of the requested task.")

    implementation_plan: list[str] = Field(
        description="Ordered steps for implementing the task."
    )

    assumptions: list[str] = Field(
        description="Assumptions that may need confirmation."
    )

    risks: list[str] = Field(description="Potential technical or security risks.")
