from langchain_community.tools.tavily_search import TavilySearchResults
import os
from langgraph.prebuilt import ToolNode


def get_tavily_tools():
    """
    Returns a list of Tavily search tools wrapped as ToolNode instances.
    """
    tools = [TavilySearchResults(max_results=2, api_key=os.getenv("TAVILY_API_KEY"))]
    return tools


def create_tooLnode(tools):
    """
    Creates a ToolNode instance for each tool in the provided list.

    Args:
        tools (list): A list of tool instances.

    Returns:
        list: A list of ToolNode instances.
    """

    return ToolNode(tools=tools)
