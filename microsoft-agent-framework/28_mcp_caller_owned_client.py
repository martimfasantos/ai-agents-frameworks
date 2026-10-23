import asyncio

from dotenv import load_dotenv
from mcp import Client
from mcp.server import MCPServer

from agent_framework import Agent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- An in-process MCP server built with the MCP SDK 2.x MCPServer
- A caller-owned mcp.Client passed to MCPStdioTool(mcp_client=...)
- Closing the tool wrapper without closing the shared MCP client

Normally the MCP tool opens and closes its own connection. With
mcp_client= you open the high-level MCP Client yourself and the
framework only borrows it: closing the tool wrapper leaves the client
open, so several wrappers (or your own code) can share one connection.
Despite the class name, MCPStdioTool spawns no subprocess here — the
Client connects to the MCPServer object in-process.

For more details, visit:
https://github.com/microsoft/agent-framework/blob/python-1.23.0/python/packages/core/agent_framework/_mcp.py#L3720
-------------------------------------------------------
"""

# --- 1. An in-process MCP server with one mock tool ---
# MCPServer sets up root logging at INFO by default, which floods the demo output.
server = MCPServer("weather-server", log_level="WARNING")


@server.tool()
def get_weather(city: str) -> str:
    """Returns the current weather for a city (simulated)."""
    # Printed to our own stdout, which proves the server runs in this process.
    print(f"  [mcp server] get_weather({city!r})")
    forecasts = {
        "lisbon": "Sunny, 24°C",
        "oslo": "Light snow, -3°C",
        "tokyo": "Cloudy, 18°C",
    }
    return forecasts.get(city.lower(), "No data")


async def main() -> None:
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 2. The caller opens (and will close) the MCP client ---
    async with Client(server) as mcp_client:
        for question in ("What's the weather in Lisbon?", "Is it snowing in Oslo?"):
            # --- 3. A fresh tool wrapper per question borrows the same client ---
            async with MCPStdioTool(name="weather", mcp_client=mcp_client) as weather_tool:
                agent = Agent(
                    client=client,
                    name="weather-agent",
                    instructions="Answer weather questions with get_weather in one sentence.",
                    tools=[weather_tool],
                )
                print(f"User: {question}")
                result = await agent.run(question)
                print(f"Agent: {result.text}")
            print(f"Wrapper connected after close: {weather_tool.is_connected}\n")

        # --- 4. Both wrappers are closed, but the caller's client still works ---
        print("=== Direct call on the caller-owned client ===")
        direct = await mcp_client.call_tool("get_weather", {"city": "Tokyo"})
        print(f"Tokyo: {direct.content[0].text}")
    # Only here, when the caller's own `async with` exits, is the connection closed.


if __name__ == "__main__":
    asyncio.run(main())
