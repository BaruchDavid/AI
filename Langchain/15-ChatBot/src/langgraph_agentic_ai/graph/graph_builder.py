from webbrowser import get

from langgraph.graph import StateGraph, START, END
from src.langgraph_agentic_ai.state.state import State
from src.langgraph_agentic_ai.nodes.basic_chatbot_node import BasicChatbotNode
from src.langgraph_agentic_ai.tools.search_tool import get_tavily_tools, create_tooLnode
from src.langgraph_agentic_ai.nodes.chatbot_with_tool_node import ChatbotWithToolNode
from langgraph.prebuilt import ToolNode, tools_condition


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

    def chatbot_with_tools_build_graph(self):
        """
        Builds an advanced chatbot graph with tool intergration.
        This method creates a chatbot graph that includes both a chbatbode node
        and tool node. It defines tools, initializes the chatbot with tool capabilities and sets up conditional and direct edged between nodes.
        The chatbot node is set as the entry point.
        """

        tools = get_tavily_tools()
        tools_node = create_tooLnode(tools)

        llm = self.llm

        obj_chatbot_with_nodes = ChatbotWithToolNode(llm)
        chatbot_node = obj_chatbot_with_nodes.crate_chatbot(tools)

        ## define nodes in the graph
        self.graph_builder.add_node("ChatbotWithToolsNode", chatbot_node)
        self.graph_builder.add_node("tools", tools_node)

        ## define edges in the graph
        self.graph_builder.add_edge(START, "ChatbotWithToolsNode")
        """
        start  -> ChatbotWithWeb -> tool-call -> zurück zum ChatbotWithWeb -> END
        und man muss entscheiden, ob man vom ChatbotWithWeb tool-call macht oder zur END geht. Das ist die Bedingung, 
        die man in der graph_builder.add_conditional_edges() Methode implementieren muss. 
        """
        self.graph_builder.add_conditional_edges("ChatbotWithToolsNode", tools_condition)

        self.graph_builder.add_edge("tools", "ChatbotWithToolsNode")  ## Kante führt zurück zum ChatbotWithWeb, um die Antwort zu erhalten.
        self.graph_builder.add_edge("ChatbotWithToolsNode", END)

    def setup_graph(self, usecase):
        if usecase == "Basic Chatbot":
            self.basic_chatbot_build_graph()
        if usecase == "Chatboot with Tools":
            self.chatbot_with_tools_build_graph()
        return self.graph_builder.compile()
