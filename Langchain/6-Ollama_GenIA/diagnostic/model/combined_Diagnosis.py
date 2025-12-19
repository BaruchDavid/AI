from pydantic import BaseModel
from .llm_Diagnosis import LlmDiagnosis


class CombinedDiagnosis(BaseModel):
    rule_based: LlmDiagnosis | None
    llm_based: LlmDiagnosis
    final: LlmDiagnosis
