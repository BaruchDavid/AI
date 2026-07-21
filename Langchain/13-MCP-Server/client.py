from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_groq import ChatGroq, GroqClient
from dotenv import load_dotenv
import asyncio
import os

load_dotenv()


async def main():
    # Create a Groq client to query the vector database
    groq_client = GroqClient()

    # Create a MultiServerMCPClient to communicate with multiple MCP servers
    mcp_client = MultiServerMCPClient(
        {
            "math": {
                "command": "python",
                "args": ["mathserver.py"],
                "transport": "stdio",
            },
            "weather": {
                "command": "python",
                "args": ["weatherserver.py"],
                "transport": "streamable-http",
            },
        }
    )

    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

    mcp_tools = await client.get_tools()
    qroq_model = ChatGroq(groq_client="qwen-qwq-32b")
    agent = create_react_agent(model=qroq_model, tools=mcp_tools)
    math_response = await agent.ainvoke(
        {
            "messages": [
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "user", "content": "What is the weather in New York?"},
            ]
        }
    )

    print("Math response:", math_response["messages"][0]["content"])


asyncio.run(main())
