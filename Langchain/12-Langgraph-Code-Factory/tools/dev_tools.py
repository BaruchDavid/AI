"""
Development Tools for AI Agent
Tools that allow an agent to write, read, and manipulate files.
"""

from langchain_core.tools import tool


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
        return (f"✅ File '{filepath}' successfully created with {len(content)} characters")
    except Exception as ex:
        return f"❌ Error writing file: {str(ex)}"


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
            content = file.read()
        return f"File content of '{filepath}':\n\n{content}"
    except FileNotFoundError:
        return f"❌ File '{filepath}' not found"
    except Exception as ex:
        return f"❌ Error reading file: {str(ex)}"


def get_all_tools():
    """
    Returns all available development tools.

    Returns:
        List of LangChain tools
    """
    return [write_file, read_file]
