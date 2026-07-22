
from src.langgraph_agentic_ai.state import State

class BasicChatbotNode:
    def __init__(self, model):
        self.model = model

    def process(self, state:State) -> dict:
        """
        Processes the input text using the provided LLM and returns the response.
        """
        return {"messages": self.llm.invoke(state["messages"])}