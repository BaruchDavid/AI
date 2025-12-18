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
    def _map_ollama_metadata(meta: dict) -> dict:
        return {
            "model": meta.get("model"),
            "prompt_tokens": meta.get("prompt_eval_count", 0),
            "completion_tokens": meta.get("eval_count", 0),
            "latency_ms": int(meta.get("total_duration", 0) / 1_000_000),
            "load_latency_ms": int(meta.get("load_duration", 0) / 1_000_000),
            "done_reason": meta.get("done_reason"),
        }
