from dataclasses import dataclass
from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class LlmResult:
    text: str
    meta_daten: dict
    raw: AIMessage
