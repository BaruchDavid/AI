from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.chat_models import ChatOllama
from llm_Result import LlmResult


class MyChatGpt:
    def __init__(self, llm_name: str):
        self.__llm = ChatOllama(
            model=llm_name
        )  ## definiere und initialisiere instanz-variablen
        self.__output_parser = (
            StrOutputParser()
        )  ## definiere und initialisiere instanz-variablen

    def get_llm(self) -> ChatOllama:
        return self.__llm

    """ Chain-basierter Ansatz """

    def execute_chain(self, message: str) -> LlmResult:
        prompt = self.__build_prompt()
        chain = prompt | self.__llm
        raw_result = chain.invoke({"question": message})
        return LlmResult(raw_result.content, raw_result.response_metadata, raw_result)

    def __build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. Please respond to the question asked",
                ),
                ("user", "Question: {question}"),
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
