from enum import Enum


class DiagnosisMode(Enum):
    RULES_ONLY = "rules_only"
    LLM_ONLY = "llm_only"
    RULES_THEN_LLM = "rules_then_llm"
    RULES_AND_LLM = "rules_and_llm"
