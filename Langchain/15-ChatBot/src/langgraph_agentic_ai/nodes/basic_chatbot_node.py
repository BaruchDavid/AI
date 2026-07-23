from src.langgraph_agentic_ai.state.state import State


class BasicChatbotNode:
    def __init__(self, model):
        self.llm = model

    def process(self, state: State) -> dict:
        """
        Processes the input text using the provided LLM and returns the response.
        """
        return {"messages": self.llm.invoke(state["messages"])}
