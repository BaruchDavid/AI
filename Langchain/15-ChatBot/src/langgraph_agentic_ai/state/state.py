from typing import Annotated  # labeling
from typing_extensions import TypedDict, list
from langgraph.graph.message import add_messages


class State(TypedDict):
    """Represents the state of the LangGraph AgenticAI application,
    including user input, selected LLM model, and other relevant information.
    """

    messages: Annotated[list, add_messages]  # List of messages exchanged in the application
