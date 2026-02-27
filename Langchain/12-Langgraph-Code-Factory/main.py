"""
Main Entry Point
Orchestrates the coding agent system.
"""

import os
import logging
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

from tools import get_all_tools
from agents import CodeAgent
from graph import GraphBuilder

# Configure root logger
logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def setup_environment():
    """Load environment variables."""
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise ValueError("GROQ_API_KEY not found in environment variables")


def run_agent(task: str):
    """Run the coding agent with a given task."""

    logger.info("\n" + "🚀 " * 30)
    logger.info("STARTING CODE AGENT")
    logger.info("🚀 " * 30 + "\n")

    setup_environment()

    # Get tools
    tools = get_all_tools()
    logger.info(f"📦 Loaded {len(tools)} tools")

    # Create agent
    agent = CodeAgent(model="qwen/qwen3-32b", temperature=0)
    agent.bind_tools(tools)

    # Build graph
    builder = GraphBuilder(agent, tools)
    graph = builder.build()

    # Display task
    logger.info(f"\n📋 TASK:")
    logger.info(f"{task}\n")
    logger.info("=" * 60 + "\n")

    # Run agent
    initial_state = {"messages": [HumanMessage(content=task)]}

    try:
        result = graph.invoke(initial_state)

        logger.info("\n" + "✅ " * 30)
        logger.info("AGENT COMPLETED SUCCESSFULLY")
        logger.info("✅ " * 30 + "\n")

        # Show summary
        logger.info(f"📊 Summary:")
        logger.info(f"   Total iterations: {builder.iteration_count}")
        logger.info(f"   Total messages: {len(result['messages'])}")

        # Show final response
        logger.info("\n💬 FINAL RESPONSE:")
        for message in result["messages"]:
            if hasattr(message, "content") and message.content:
                msg_type = message.__class__.__name__
                if msg_type == "AIMessage" and not getattr(message, "tool_calls", None):
                    logger.info(f"\n{message.content}\n")

        return result

    except Exception as e:
        logger.error(f"\n❌ ERROR: {e}")
        raise


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
    print("✅ Check 'is_prime.py' in your directory!")


if __name__ == "__main__":
    main()
