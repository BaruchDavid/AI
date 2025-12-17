from pydantic import BaseModel
from Llm_Diagnosis import LlmDiagnosis


class CombinedDiagnosis(BaseModel):
    rule_based: LlmDiagnosis | None
    llm_based: LlmDiagnosis
    final: LlmDiagnosis
