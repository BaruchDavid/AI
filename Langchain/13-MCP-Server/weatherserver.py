from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")


@mcp.tool()
async def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    # In a real implementation, you would call a weather API here.
    # For demonstration purposes, we'll return a dummy weather report.
    return f"The current weather in {location} is sunny with a temperature of 25°C."


# Use streamable HTTP for communication with the MCP client
if __name__ == "__main__":
    mcp.run(transport="streamable-http")
