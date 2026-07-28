from typing_extensions import TypedDict  # for type hinting
from langgraph.graph.message import add_messages
from typing import Annotated


class State(TypedDict):
    """Represents the state of the LangGraph AgenticAI application,
    including user input, selected LLM model, and other relevant information.
    """

    messages: Annotated[list, add_messages]  # List of messages exchanged in the application
    news_data: list  # Fetched news articles from Tavily
    summary: str  # Summarized news content
    filename: str  # Path to the saved summary file
