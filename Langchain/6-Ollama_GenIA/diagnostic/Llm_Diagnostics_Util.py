from typing import Any, Mapping, Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from .diagnosis_Mode import DiagnosisMode
from .llm_Diagnosis import LlmDiagnosis
from .combined_Diagnosis import CombinedDiagnosis
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage


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

    """     
    ✔ ChatPromptTemplate.from_messages
    ✔ {format_instructions} im Human-Teil
    ✔ PydanticOutputParser → richtiges Tool
    ✔ prompt | self.__llm | parser → richtige Chain
    ✔ Typisierte Rückgabe LlmDiagnosis """

    def _llm_based_check(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        task_type: str,
    ) -> LlmDiagnosis:

        pydanticOutputParser = PydanticOutputParser(pydantic_object=LlmDiagnosis)
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are an LLM diagnostics assistant. "
                    "Analyze the following metadata and identify the most likely issue.",
                ),
                (
                    "human",
                    """
                Task type: {task_type}
                Prompt tokens: {prompt_tokens}
                Completion tokens: {completion_tokens}
                Latency (ms): {latency_ms}

                Choose ONE issue from the following list:
                - normal
                - truncated_response
                - context_loss
                - slow_response
                - hallucination_risk

                Return ONLY valid JSON.
                Do not add explanations or markdown.

                {format_instructions}
                """,
                ),
            ]
        )

        prompt = prompt.partial(
            format_instructions=pydanticOutputParser.get_format_instructions()
        )

        chain = prompt | self.__llm | pydanticOutputParser

        try:
            return chain.invoke(
                {
                    "task_type": task_type,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "latency_ms": latency_ms,
                }
            )
        except Exception as e:
            # optional: fallback
            return LlmDiagnosis(
                issue="hallucination_risk",
                reason="LLM output could not be parsed reliably.",
                confidence=0.3,
            )

    def diagnose(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        task_type: str,
        diagnose_mode: DiagnosisMode = DiagnosisMode.RULES_AND_LLM,
    ) -> CombinedDiagnosis:

        rule_result: LlmDiagnosis | None = None
        llm_result: LlmDiagnosis | None = None

        # --- RULES_ONLY ---
        if diagnose_mode == DiagnosisMode.RULES_ONLY:
            rule_result = self._rule_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            final = rule_result or LlmDiagnosis(
                issue="normal",
                confidence=1.0,
                reason="No rule-based issues detected.",
            )

            return CombinedDiagnosis(
                rule_based=rule_result,
                llm_based=final,
                final=final,
            )

        # --- LLM_ONLY ---
        if diagnose_mode == DiagnosisMode.LLM_ONLY:
            llm_result = self._llm_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            return CombinedDiagnosis(
                rule_based=None,
                llm_based=llm_result,
                final=llm_result,
            )

        # --- RULES_THEN_LLM ---
        if diagnose_mode == DiagnosisMode.RULES_THEN_LLM:
            rule_result = self._rule_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            if rule_result:
                return CombinedDiagnosis(
                    rule_based=rule_result,
                    llm_based=rule_result,
                    final=rule_result,
                )

            llm_result = self._llm_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            return CombinedDiagnosis(
                rule_based=None,
                llm_based=llm_result,
                final=llm_result,
            )

        # --- RULES_AND_LLM (default / dein Wunsch) ---
        if diagnose_mode == DiagnosisMode.RULES_AND_LLM:
            rule_result = self._rule_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            llm_result = self._llm_based_check(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                task_type=task_type,
            )

            # Entscheidungslogik: LLM gewinnt, aber Regeln beeinflussen Confidence
            final = llm_result

            if rule_result and rule_result.issue != llm_result.issue:
                final = LlmDiagnosis(
                    issue=llm_result.issue,
                    confidence=min(llm_result.confidence, rule_result.confidence),
                    reason=(
                        f"Rule-based check suggested '{rule_result.issue}'. "
                        f"LLM analysis suggested '{llm_result.issue}'. "
                        f"Final decision based on LLM."
                    ),
                )

            return CombinedDiagnosis(
                rule_based=rule_result,
                llm_based=llm_result,
                final=final,
            )

        # --- Sicherheitsnetz ---
        raise ValueError(f"Unsupported diagnosis mode: {diagnose_mode}")
