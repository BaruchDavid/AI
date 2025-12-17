from typing import Any, Mapping, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from Llm_Diagnosis import LlmDiagnosis


class LlmDiagnosticUtil:

    def __init__(
        self,
        llm: ChatOllama,
        *,
        max_expected_completion_tokens: int = 50,
        max_prompt_tokens: int = 3000,
        slow_latency_ms: int = 3000,
    ):
        self.__llm = llm
        self.__max_expected_completion_tokens = max_expected_completion_tokens
        self.__max_prompt_tokens = max_prompt_tokens
        self.__slow_latency_ms = slow_latency_ms

    """ Harte Regel-Pruefungen """

    def _rule_based_check(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        task_type: str,
    ) -> Optional[LlmDiagnosis]:
        # 1️⃣ Antwort bricht ab
        if completion_tokens < 5 and task_type in {"explanation", "analysis"}:
            return LlmDiagnosis(
                issue="truncated_response",
                confidence=0.9,
                reason="Completation tokens are to less for 'explantion, analysis' task",
            )

        # 2️⃣ Kontextverlust
        if prompt_tokens > self.__max_prompt_tokens:
            return LlmDiagnosis(
                issue="context_loss",
                confidence=0.8,
                reason="max prompt token limit has been exceeded!!!",
            )

        # 3️⃣ Langsame Antwort
        if latency_ms > self.__slow_latency_ms:
            return LlmDiagnosis(
                issue="slow_response",
                confidence=0.85,
                reason="max slow latency has been exceeded",
            )

        return None

    def _llm_based_check(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        task_type: str,
    ) -> LlmDiagnosis:

        structered_llm = self.__llm.with_structured_output(LlmDiagnosis)

        messages = [
            SystemMessage(
                content=(
                    "You are an LLM diagnostics assistant. "
                    "Analyze the following metadata and identify the most likely issue."
                )
            ),
            HumanMessage(
                content=f"""
                            Task type: {task_type}
                            Prompt tokens: {prompt_tokens}
                            Completion tokens: {completion_tokens}
                            Latency (ms): {latency_ms}

                            Choose one issue:
                            - normal
                            - truncated_response
                            - context_loss
                            - slow_response
                            - hallucination_risk

                            Provide a short reason and confidence.
                            """
            ),
        ]

        return structered_llm.invoke(messages)

    def diagnose(self, *, prompt_tokens: int, completion_tokens: int, latency_ms: int, task_type: str) -> LlmDiagnosis:
        