from dataclasses import dataclass
from langchain_core.messages import AIMessage


@dataclass(frozen=True)
class LlmResult:
    text: str
    meta_daten: dict
    raw: AIMessage

    """ verwende post_init, weil @dataclass schon selbst die init-methode implementiert """

    def __post_init__(self):
        normalized = self._map_ollama_metadata(self.meta_daten)
        object.__setattr__(self, "meta_daten", normalized)

    @staticmethod
    def _map_ollama_metadata(llm_meta_data: dict) -> dict:
        return {
            "model": llm_meta_data.get("model"),
            "prompt_tokens": llm_meta_data.get("prompt_eval_count", 0),
            "completion_tokens": llm_meta_data.get("eval_count", 0),
            "latency_ms": int(llm_meta_data.get("total_duration", 0) / 1_000_000),
            "load_latency_ms": int(llm_meta_data.get("load_duration", 0) / 1_000_000),
            "done_reason": llm_meta_data.get("done_reason"),
        }
