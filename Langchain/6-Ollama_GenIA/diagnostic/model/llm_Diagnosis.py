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
    ]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
