from langgraph.graph import StateGraph, START, END
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.nodes.basic_chatbot_node import BasicChatbotNode


class GraphBuilder:
    def __init__(self, model):
        self.graph_builder = StateGraph(State)
        self.llm = model

    def basic_chatbot_build_graph(self):
        """
        Builds a bsic chatbot graph using LangGraph.
        This method initializes a chatbot node using the 'BasicChatbotNode' class
        and intergrates it into the graph. The chatbot node is set as bot the
        entry and exit point of the graph.
        """

        self.basic_chatbot_node = BasicChatbotNode(self.llm)

        self.graph_builder.add_node("BasicChatbotNode", self.basic_chatbot_node.process)
        self.graph_builder.add_edge(START, "BasicChatbotNode")
        self.graph_builder.add_edge("BasicChatbotNode", END)

    def setup_graph(self, usecase):
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        return self.graph_builder.compile()
