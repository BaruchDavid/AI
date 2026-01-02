from pydantic import BaseModel
from typing import Optional, Literal
from pydantic import Field

""" Das ist kein normaler Datencontainer, sondern:
✔️ Eingabe von einer LLM
✔️ externe, unsichere Daten
✔️ muss validiert werden
✔️ muss Fehler werfen können
👉 Genau dafür ist Pydantic gemacht. 

Kommt das Objekt von außen?
→ pydantic.BaseModel, was die Typsicherheit bietet

"""


class LlmDiagnosis(BaseModel):
    issue: Literal[
        "normal",
        "truncated_response",
        "context_loss",
        "slow_response",
        "hallucination_risk",
    ] = Field(
        description="Primary issue detected in the LLM response. "
        "Use 'normal' if no problem is detected."
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence level of the diagnosis. "
        "0.0 means very uncertain, 1.0 means very certain.",
    )
    reason: str = Field(description="Short explanation justifying the selected issue.")
