"""
Code Agent
Handles the agent logic and state management.
"""

import logging
from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from prompts import PromptLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class AgentState(TypedDict):
    """State for our coding agent."""

    messages: Annotated[list[AnyMessage], add_messages]


class CodeAgent:
    """
    A coding agent that can write and read files with high code quality.

    Responsibilities:
    - Manage the LLM configuration
    - Enforce coding standards via system prompt
    - Bind tools to the LLM
    - Process agent state through the LLM
    """

    def __init__(self, model: str = "qwen/qwen3-32b", temperature: float = 0):
        logger.info(f"🔧 Initializing CodeAgent with model: {model}")

        # System Prompt für Code-Qualität
        self.system_prompt = PromptLoader.get_instance(None).get_prompt("code_agent")
        logger.debug(f"📋 Loaded system prompt ({len(self.system_prompt)} chars)")

        # LLM initialisieren
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.tools = []
        self.llm_with_tools = None

        logger.info("✅ CodeAgent initialized successfully")

    def bind_tools(self, tools: list):
        """Bind tools to the LLM."""
        logger.info(f"🔗 Binding {len(tools)} tools to LLM")
        for tool in tools:
            logger.debug(f"  - {tool.name}: {tool.description}")

        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)

        logger.info("✅ Tools bound successfully")

    def _ensure_system_prompt(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """Ensure system prompt is present."""
        if not self._has_system_message(messages):
            logger.debug("➕ Adding system prompt to messages")
            return [SystemMessage(content=self.system_prompt)] + messages
        logger.debug("✓ System prompt already present")
        return messages

    def process(self, state: AgentState) -> dict:
        """Process the agent state through the LLM."""
        logger.info("\n" + "=" * 60)
        logger.info("🤖 AGENT PROCESSING")
        logger.info("=" * 60)

        if not self.llm_with_tools:
            raise ValueError(
                "Tools must be bound before processing. Call bind_tools() first."
            )

        messages = state["messages"]
        logger.info(f"📨 Current message count: {len(messages)}")

        # Log last message for context
        if messages:
            last_msg = messages[-1]
            msg_type = last_msg.__class__.__name__

            if isinstance(last_msg, ToolMessage):
                logger.info(
                    f"📬 Last message: {msg_type} - Tool '{last_msg.name}' returned result"
                )
                logger.debug(f"   Result preview: {str(last_msg.content)[:100]}...")
            elif isinstance(last_msg, AIMessage):
                if last_msg.tool_calls:
                    logger.info(
                        f"📬 Last message: {msg_type} with {len(last_msg.tool_calls)} tool call(s)"
                    )
                else:
                    logger.info(f"📬 Last message: {msg_type} (no tool calls)")
            else:
                logger.info(f"📬 Last message: {msg_type}")

        # Ensure system prompt is present
        messages = self._ensure_system_prompt(messages)

        # Invoke LLM
        logger.info("🧠 Invoking LLM...")
        response = self.llm_with_tools.invoke(messages)

        # Analyze response
        if isinstance(response, AIMessage):
            if response.tool_calls:
                logger.info(
                    f"🔧 LLM decided to use {len(response.tool_calls)} tool(s):"
                )
                for i, tool_call in enumerate(response.tool_calls, 1):
                    logger.info(f"   {i}. Tool: {tool_call['name']}")
                    logger.info(f"      Args: {tool_call['args']}")
                    logger.debug(f"      ID: {tool_call['id']}")
            else:
                logger.info("✅ LLM completed task (no more tools needed)")
                if response.content:
                    content_preview = response.content[:200]
                    logger.info(f"💬 Response preview: {content_preview}...")

        # Check for reasoning (if available)
        if (
            hasattr(response, "additional_kwargs")
            and "reasoning_content" in response.additional_kwargs
        ):
            reasoning = response.additional_kwargs["reasoning_content"]
            logger.info(f"🤔 LLM Reasoning ({len(reasoning)} chars):")
            # Show first 300 chars of reasoning
            logger.info(f"   {reasoning[:300]}...")

        logger.info("=" * 60 + "\n")

        return {"messages": [response]}

    def _has_system_message(self, messages: list[AnyMessage]) -> bool:
        """Check if system message is present."""
        return any(isinstance(msg, SystemMessage) for msg in messages)
