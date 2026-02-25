"""
Graph Builder
Responsible for constructing the LangGraph workflow.
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents import CodeAgent, AgentState


class GraphBuilder:
    """
    Builds the LangGraph workflow for the coding agent.

    Responsibilities:
    - Define nodes and edges
    - Configure conditional routing
    - Compile the graph
    """

    def __init__(self, agent: CodeAgent, tools: list):
        """
        Initialize the graph builder.

        Args:
            agent: The CodeAgent instance
            tools: List of tools to use in the tool node
        """
        self.agent = agent
        self.tools = tools
        self.graph = None

    def build(self):
        """
        Build and compile the graph.

        Returns:
            Compiled LangGraph
        """
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("agent", self.agent.process)
        builder.add_node("tools", ToolNode(self.tools))

        # Add edges
        builder.add_edge(START, "agent")

        # Conditional edge: agent decides whether to use tools or end
        builder.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools", "__end__": END}
        )

        # After tools execution, return to agent
        builder.add_edge("tools", "agent")

        # Compile
        self.graph = builder.compile()
        return self.graph

    def get_graph(self):
        """
        Get the compiled graph.

        Returns:
            Compiled graph or None if not built yet
        """
        if not self.graph:
            raise ValueError("Graph not built yet. Call build() first.")
        return self.graph
