"""
Ein Agent, der code schreibt, aber noch nicht testet
"""

from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import HumanMessage, AnyMessage
from langchain_groq import ChatGroq
from langchain_core.tools import tool


# ==== Tools als LanchChain Tools definieren ==== #


@tool
def write_file(filepath: str, content: str) -> str:
    """
    Write code or text to a file.

    Args:
        filepath: Path to the file (e.g. 'is_prime.py')
        content: The content to write

    Returns:
        Success message with filepath
    """
    try:
        with open(filepath, "w", encoding="utf-8") as file:
            file.write(content)
        return (
            f"✅ File '{filepath}' successfully created with {len(content)} characters"
        )
    except Exception as ex:

        return f"❌ Error writing file: {str(e)}"


@tool
def read_file(filepath: str) -> str:
    """
    Read the content of a file.

    Args:
        filepath: Path to the file

    Returns:
        File content or error message
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            content = f.read()
        return f"File content of '{filepath}':\n\n{content}"
    except FileNotFoundError:
        return f"❌ File '{filepath}' not found"
    except Exception as ex:
        return f"❌ Error reading file: {str(e)}"


# ==== State Definition ==== #
class AgentState(TypedDict):
    """Sate für unseren Agenten"""

    messages: Annotated[list[AnyMessage], add_messages]


# ==== LLM mit Tools verbinden ==== #

groq_0_llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)

tools_list = [write_file, read_file]

llm_with_tools = llm.bind_tools(tools_list)


# ==== Node-Funktion ==== #
def agent_node(state: AgentState) -> dict:
    """
    Der Agent-Node, der das LLM aufruft.
    """
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ==== Graph aufbauen ==== #
def create_agent():
    """Erstellt den LangGraph Agenten"""
    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools_list", ToolNode(tools_list))

    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent", tools_condition, {"tools": "tools_list", "__end__": END}
    )

    builder.add_edge("tools", "agent")

    graph = builder.compile()

    return graph


# ==== Test-Funktion ==== #
def run_agent(task: str):
    """
    Run the agent with a given task.

    Args:
        task: The task for the agent (e.g. "Write a function to check prime numbers")
    """
    graph = create_agent()
    print(f"🤖 Agent Task: {task}\n")
    print("=" * 60)

    initial_state = {"messages": [HumanMessage(content=task)]}
    result = graph.invoke(initial_state)

    print("\n📝 Agent Response:\n")
    for message in result["message"]:
        if hasattr(message, "content") and message.content:
            print(f"{message.__class__.__name__} : {message.content}\n")

    return result


if __name__ == "__main__":
    task = """
    Write a Python function called 'is_prime' that checks if a number is prime.
    Save it to a file called 'is_prime.py'.
    The function should:
    - Take an integer n as input
    - Return True if n is prime, False otherwise
    - Handle edge cases (n < 2)    
    """

    run_agent(task)
    print("\n" + "=" * 60)
    print("✅ Agent finished! Check 'is_prime.py' in your directory.")
