from typing import Optional
from langchain_community.chat_models import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from .diagnosis_Mode import DiagnosisMode
from .llm_Diagnosis import LlmDiagnosis
from .combined_Diagnosis import CombinedDiagnosis
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import SystemMessage, HumanMessage
from config.diagnostic_config import DiagnosticConfig


class LlmDiagnosticUtil:

    def __init__(
        self,
        *,
        llm: ChatOllama,
        max_expected_completion_tokens: int = 50,
        max_prompt_tokens: int = 3000,
        config: DiagnosticConfig,
    ):
        self.__llm = llm
        self.__max_expected_completion_tokens = max_expected_completion_tokens
        self.__max_prompt_tokens = max_prompt_tokens
        self.__config = config

    """ Harte Regel-Pruefungen """

    from typing import Optional

    def _rule_based_check(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: int,
        task_type: str,
    ) -> Optional[LlmDiagnosis]:
        """
        Rule-based diagnostics.

        Priority order:
        1. Truncated response (unexpectedly short completion)
        2. Context loss (prompt near/exceeds context window)
        3. Slow response (high latency)
        4. Hallucination risk (unexpectedly long completion)
        """

        # Avoid division by zero
        expected_tokens = max(self.__max_expected_completion_tokens, 1)
        completion_ratio = completion_tokens / expected_tokens

        ## 1️⃣ Truncated response (too short for the task)
        ## < 30 % der erwarteten Länge bei erklärenden Tasks fast immer unvollständig
        ## < 10 % dann würde man echte Abbrüche zu spät erkennen
        ## < 50 % könnte eine kurze pregnante Antwort sein

        if (
            completion_ratio < self.__config.hallucination_ratio_threshold
            and task_type in {"explanation", "analysis"}
        ):
            return LlmDiagnosis(
                issue="truncated_response",
                confidence=0.9,
                reason=(
                    f"Completion tokens ({completion_tokens}) are significantly below "
                    f"the expected amount ({expected_tokens}) for task type '{task_type}'."
                ),
            )

        ## 2️⃣ Context loss (near or exceeding prompt limit)
        ## Kontextverlust beginnt vor dem harten Limit
        ## viele Modelle degradieren ab ~90–95 %
        if prompt_tokens >= int(
            self.__max_prompt_tokens * self.__config.context_warning_ratio
        ):
            return LlmDiagnosis(
                issue="context_loss",
                confidence=0.8,
                reason=(
                    f"Prompt tokens ({prompt_tokens}) are near or exceed the maximum "
                    f"allowed limit ({self.__max_prompt_tokens})."
                ),
            )

        # 3️⃣ Slow response
        if latency_ms > self.__config.slow_latency_ms:
            return LlmDiagnosis(
                issue="slow_response",
                confidence=0.85,
                reason=(
                    f"Latency ({latency_ms} ms) exceeds the configured slow-response "
                    f"threshold ({self.__config.slow_latency_ms} ms)."
                ),
            )

        ## 4️⃣ Hallucination risk (unexpectedly long completion)
        ## Gedanke: Fast doppelte Länge → LLM schweift vom Thema ab
        ## Warum nicht 1.2? erklärende Antworten können länger sein
        ## Warum nicht 3.0? dann erkennt man Probleme zu spät
        ## 📌 1.7–2.0 ist ein guter Sweet Spot
        if completion_ratio > self.__config.hallucination_ratio_threshold:
            return LlmDiagnosis(
                issue="hallucination_risk",
                confidence=0.7,
                reason=(
                    f"Completion tokens ({completion_tokens}) significantly exceed "
                    f"the expected amount ({expected_tokens}), indicating potential "
                    f"hallucination or prompt misinterpretation."
                ),
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
