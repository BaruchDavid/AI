"""
Code Agent
Handles the agent logic and state management.
"""

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from prompts import PromptLoader


class AgentState(TypedDict):
    """State for our coding agent."""

    messages: Annotated[list[AnyMessage], add_messages]


class CodeAgent:
    """
    A coding agent that can write and read files with high code quality.

    Responsibilities:
    - Manage the LLM configuration
    - Enforce coding standards via system prompt
    - Bind tools to the LLM
    - Process agent state through the LLM
    """

    def __init__(self, model: str = "qwen/qwen3-32b", temperature: float = 0):

        # System Prompt für Code-Qualität
        self.system_prompt = PromptLoader.get_instance(None).get_prompt("code_agent")

        # LLM initialisieren
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.tools = []
        self.llm_with_tools = None

    def bind_tools(self, tools: list):

        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)

    def _ensure_system_prompt(self, messages: list[AnyMessage]) -> list[AnyMessage]:

        if not self._has_system_message(messages):
            return [SystemMessage(content=self.system_prompt)] + messages
        return messages

    def process(self, state: AgentState) -> dict:

        if not self.llm_with_tools:
            raise ValueError(
                "Tools must be bound before processing. Call bind_tools() first."
            )

        messages = state["messages"]

        # Ensure system prompt is present
        messages = self._ensure_system_prompt(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def _has_system_message(self, messages: list[AnyMessage]) -> bool:
        # any gibt True zurück, wenn mind. eine Instanze der 'msg' true ist
        return any(isinstance(msg, SystemMessage) for msg in messages)
