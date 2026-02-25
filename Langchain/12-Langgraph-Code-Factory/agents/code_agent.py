"""
Code Agent
Handles the agent logic and state management.
"""

from typing import TypedDict, Annotated
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq


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
        """
        Initialize the code agent.

        Args:
            model: The LLM model to use
            temperature: Temperature for generation (0 = deterministic)
        """
        # System Prompt für Code-Qualität
        self.system_prompt = """You are an expert Python developer who writes clean, professional code.

When writing code, ALWAYS follow these rules:
1. Add comprehensive docstrings (Google style) for all functions
2. Add inline comments for complex logic
3. Follow PEP 8 style guide
4. Include type hints for function arguments and return values
5. Write clean, readable, maintainable code
6. Handle edge cases properly

Example of GOOD code:
```python
def is_prime(n: int) -> bool:
    \"\"\"
    Check if a number is prime.
    
    A prime number is a natural number greater than 1 that has no positive
    divisors other than 1 and itself.
    
    Args:
        n: The integer to check for primality
        
    Returns:
        True if n is prime, False otherwise
        
    Examples:
        >>> is_prime(2)
        True
        >>> is_prime(4)
        False
        >>> is_prime(17)
        True
    \"\"\"
    # Handle edge cases: numbers less than 2 are not prime
    if n < 2:
        return False
    
    # 2 is the only even prime number
    if n == 2:
        return True
    
    # Even numbers greater than 2 are not prime
    if n % 2 == 0:
        return False
    
    # Check odd divisors up to sqrt(n)
    # We only need to check up to the square root because
    # if n has a factor greater than sqrt(n), it must also
    # have a corresponding factor less than sqrt(n)
    for i in range(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    
    return True
```

IMPORTANT: When saving code to a file, make sure it's complete, well-documented, and production-ready."""

        # LLM initialisieren
        self.llm = ChatGroq(model=model, temperature=temperature)
        self.tools = []
        self.llm_with_tools = None

    def bind_tools(self, tools: list):
        """
        Bind tools to the LLM.

        Args:
            tools: List of LangChain tools
        """
        self.tools = tools
        self.llm_with_tools = self.llm.bind_tools(tools)

    def _ensure_system_prompt(self, messages: list[AnyMessage]) -> list[AnyMessage]:
        """
        Ensure that the system prompt is present at the beginning of the message list.

        If no SystemMessage is found, prepends the system prompt to the messages.

        Args:
            messages: Original list of messages

        Returns:
            Updated list of messages with system prompt at the beginning
        """
        if not self._has_system_message(messages):
            return [SystemMessage(content=self.system_prompt)] + messages
        return messages

    def process(self, state: AgentState) -> dict:
        """
        Process the agent state through the LLM with system prompt.

        Args:
            state: Current agent state

        Returns:
            Updated state with LLM response
        """
        if not self.llm_with_tools:
            raise ValueError(
                "Tools must be bound before processing. Call bind_tools() first."
            )

        messages = state["messages"]

        # Ensure system prompt is present
        messages = self._ensure_system_prompt(messages)

        response = self.llm_with_tools.invoke(messages)
        return {"messages": [response]}
