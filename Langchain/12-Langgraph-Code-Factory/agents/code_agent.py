"""
Code Agent
Handles the agent logic and state management.
"""

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq


class AgentState(TypedDict):
    """State for our coding agent."""

    messages: Annotated[list[AnyMessage], add_messages]


class CodeAgent:
    """
    A coding agent that can write and read files.

    Responsibilities:
    - Manage the LLM configuration
    - Bind tools to the LLM
    - Process agent state through the LLM
    """

    def __init__(self, model: str = "qwen/qwen3-32b", temperature: float = 0):
        """
        Initialize the code agent.

        Args:
            model: The LLM model to use
            temperature: Temperature for generation (0 = deterministic)
        """
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.tools = []
        self.llm_with_tools = None

    def bind_tools(self, tools: list):
        """
        Bind tools to the LLM.

        Args:
            tools: List of LangChain tools
        """
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)

    def process(self, state: AgentState) -> dict:
        """
        Process the agent state through the LLM.

        Args:
            state: Current agent state

        Returns:
            Updated state with LLM response
        """
        if not self.llm_with_tools:
            raise ValueError(
                "Tools must be bound before processing. Call bind_tools() first."
            )

        messages = state["messages"]
        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
