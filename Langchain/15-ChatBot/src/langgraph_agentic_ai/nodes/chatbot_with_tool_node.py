from src.langgraph_agentic_ai.state.state import State
from langgraph.graph.message import add_messages
from typing import Annotated


class ChatbotWithToolNode:
    def __init__(self, llm):
        self.llm = llm

    def process(self, input: Annotated[str, "User input"]):
        """
        Processes the user input and generates a response using the provided LLM.

        Args:
            input (str): The user input to be processed.

        Returns:
            str: The generated response.
        """
        response = self.llm.generate_response(input)
        add_messages(State, "user", input)
        add_messages(State, "bot", response)
        return response

    ## when we have not tools, just invoke llm
    def process(self, state: State) -> dict:
        """
        Processes the user input and generates a response using the provided LLM.

        Args:
            state (State): The current state of the conversation.

        """
        user_input = state["messages"][-1] if state["messages"] else ""
        llm_response = self.llm.invoke([{"role": "user", "content": user_input}])
        tool_response = f"Tool intergration for: '{user_input}'"
        return {"messages": [llm_response, tool_response]}

    ## binding llm with tools
    def crate_chatbot(self, tools):
        """
        Returns a ChatbotWithToolNode node function
        """
        llm_with_tools = self.llm.bind_tools(tools)

        def chatbot_node(state: State):
            """
            Chatbot logic for processing the input state and returning a response.
            """
            return {"messages": [llm_with_tools.invoke(state["messages"])]}

        return chatbot_node
