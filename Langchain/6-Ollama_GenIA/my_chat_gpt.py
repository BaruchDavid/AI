import logging
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from llm_Result import LlmResult
from history.session_history_store import SessionHistoryStore
from langchain_core.runnables.history import RunnableWithMessageHistory


class MyChatGpt:
    def __init__(self, *, llm_name: str, history_store: SessionHistoryStore):
        self.logger = logging.getLogger(__name__)
        self.__llm = ChatOllama(model=llm_name, streaming=True)
        self._history_store = history_store
        self._prompt = self.__build_prompt()
        base_chain = self._prompt | self.__llm
        self._chain = self._buildHistoryWrapper(base_chain)

    ## definiert wie History benutzt wird
    def _buildHistoryWrapper(self, base_chain) -> RunnableWithMessageHistory:
        return RunnableWithMessageHistory(
            base_chain,
            self._history_store.get_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    def get_llm(self) -> ChatOllama:
        return self.__llm

    """ Chain-basierter Ansatz """

    def execute_chain(self, *, message: str, session_id: str):
        full_text = ""
        try:
            for chunk in self._chain.stream(
                {"input": message},
                config={"configurable": {"session_id": session_id}},
            ):
                yield chunk
                self.__save_history(history=full_text, chunk=chunk)
                full_text += chunk

        except Exception:
            MAX_LOG_CHARS = 400
            self.logger.exception(
                "LLM streaming failed | session_id=%s | message=%r",
                session_id,
                message[:MAX_LOG_CHARS],
            )
        raise

        raw_result = AIMessage(content=full_text, response_metadata={"streamed": True})
        yield LlmResult(full_text, raw_result.response_metadata, raw_result)

    def __save_history(self, *, history: str, chunk: str) -> str:
        if hasattr(chunk, "content"):
            history += chunk.content
        return history

    def __build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are a helpful assistant."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

    """ Direkter Message-Ansatz """

    def execute_prompt(self, prompt) -> LlmResult:
        message_prompt = self.__build_message(prompt)
        raw_result = self.__llm.invoke(message_prompt)
        return LlmResult(raw_result.content, raw_result.response_metadata, raw_result)

    def __build_message(self, prompt) -> list:
        return [
            SystemMessage(
                content="You are a professional translator. "
                + "Translate the following English text to German.Return only the translation."
            ),
            HumanMessage(content=prompt),
        ]
