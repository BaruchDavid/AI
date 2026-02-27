"""
Graph Builder
Responsible for constructing the LangGraph workflow.
"""

import logging
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from agents import CodeAgent, AgentState

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds the LangGraph workflow for the coding agent.

    Responsibilities:
    - Define nodes and edges
    - Configure conditional routing
    - Compile the graph
    """

    def __init__(self, agent: CodeAgent, tools: list):
        """Initialize the graph builder."""
        logger.info("🏗️  Initializing GraphBuilder")
        self.agent = agent
        self.tools = tools
        self.graph = None
        self.iteration_count = 0

    def build(self):
        """Build and compile the graph."""
        logger.info("🔨 Building graph structure...")
        
        builder = StateGraph(AgentState)

        # Add nodes with logging wrappers
        builder.add_node("agent", self._wrap_agent_node)
        builder.add_node("tools_list", self._wrap_tools_node)

        # Add edges
        builder.add_edge(START, "agent")
        builder.add_conditional_edges(
            "agent", tools_condition, {"tools": "tools_list", "__end__": END}
        )
        builder.add_edge("tools_list", "agent")

        # Compile
        self.graph = builder.compile()
        logger.info("✅ Graph compiled successfully")
        
        return self.graph
    
    def _wrap_agent_node(self, state: AgentState) -> dict:
        """Wrapper for agent node with iteration tracking."""
        self.iteration_count += 1
        logger.info(f"\n{'='*60}")
        logger.info(f"🔄 ITERATION #{self.iteration_count} - AGENT NODE")
        logger.info(f"{'='*60}")
        
        result = self.agent.process(state)
        return result
    
    def _wrap_tools_node(self, state: AgentState) -> dict:
        """Wrapper for tools node with execution logging."""
        logger.info(f"\n{'='*60}")
        logger.info(f"🔧 TOOLS NODE - Executing tools")
        logger.info(f"{'='*60}")
        
        # Get the last AI message to see which tools will be called
        messages = state["messages"]
        if messages:
            last_msg = messages[-1]
            if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                for tool_call in last_msg.tool_calls:
                    logger.info(f"⚙️  Executing: {tool_call['name']}({tool_call['args']})")
        
        # Execute tools
        tool_node = ToolNode(self.tools)
        result = tool_node.invoke(state)
        
        # Log results
        if isinstance(result, dict) and "messages" in result:
            for msg in result["messages"]:
                if hasattr(msg, 'content'):
                    result_preview = str(msg.content)[:150]
                    logger.info(f"✅ Tool result: {result_preview}...")
        
        logger.info(f"{'='*60}\n")
        
        return result

    def get_graph(self):
        """Get the compiled graph."""
        if not self.graph:
            raise ValueError("Graph not built yet. Call build() first.")
        return self.graph