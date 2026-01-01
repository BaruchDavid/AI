from pydantic import BaseModel, Field
from typing import Literal


class DiagnosticConfig(BaseModel):
    truncated_ratio_threshold: float = Field(ge=0.0, le=1.0)
    context_warning_ratio: float = Field(ge=0.0, le=1.0)
    hallucination_ratio_threshold: float = Field(gt=1.0)
    slow_latency_ms: int = Field(gt=0)
    min_completion_tokens: int = Field(ge=1)
    task_type: Literal["analysis", "explanation"]
