"""
Main Entry Point
Orchestrates the coding agent system.
"""

import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from tools import get_all_tools
from agents import CodeAgent
from graph import GraphBuilder


def setup_environment():
    """Load environment variables."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")


def run_agent(task: str):
    """
    Run the coding agent with a given task.

    Args:
        task: The task description for the agent

    Returns:
        Final agent state with all messages
    """
    
    setup_environment()

    
    tools = get_all_tools()

    agent = CodeAgent(model="qwen/qwen3-32b", temperature=0)
    agent.bind_tools(tools)


    builder = GraphBuilder(agent, tools)
    graph = builder.build()


    print(f"🤖 Agent Task: {task}\n")
    print("=" * 60)


    initial_state = {"messages": [HumanMessage(content=task)]}
    result = graph.invoke(initial_state)


    print("\n📝 Agent Response:\n")
    for message in result["messages"]:
        if hasattr(message, "content") and message.content:
            print(f"{message.__class__.__name__}: {message.content}\n")

    return result


def main():
    """Main entry point."""
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


if __name__ == "__main__":
    main()
