import asyncio
from typing import Any

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    tool,
    create_sdk_mcp_server,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Defining custom tools with the @tool decorator
- Wrapping tools in an in-process MCP server via create_sdk_mcp_server()
- Passing MCP servers to query() and auto-approving tools
- Tool naming convention: mcp__{server_name}__{tool_name}

Custom tools extend the agent's capabilities with your own functions.
The @tool decorator defines the tool schema and handler, then
create_sdk_mcp_server() bundles tools into an in-process MCP server
that runs inside your application with zero IPC overhead.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/custom-tools
-------------------------------------------------------
"""


# --- 1. Define custom tools ---
@tool(
    "get_weather",
    "Get the current weather for a city",
    {"city": str},
)
async def get_weather(args: dict[str, Any]) -> dict[str, Any]:
    """Simulated weather lookup."""
    weather_data = {
        "lisbon": "Sunny, 25°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 20°C",
    }
    city = args["city"].lower()
    forecast = weather_data.get(city, f"No data available for {args['city']}")
    return {"content": [{"type": "text", "text": forecast}]}


@tool(
    "get_population",
    "Get the population of a city",
    {"city": str},
)
async def get_population(args: dict[str, Any]) -> dict[str, Any]:
    """Simulated population lookup."""
    populations = {
        "lisbon": "545,000",
        "london": "8,982,000",
        "tokyo": "13,960,000",
    }
    city = args["city"].lower()
    pop = populations.get(city, "Unknown")
    return {
        "content": [{"type": "text", "text": f"Population of {args['city']}: {pop}"}]
    }


# --- 2. Bundle tools into an in-process MCP server ---
city_server = create_sdk_mcp_server(
    name="city_info",
    version="1.0.0",
    tools=[get_weather, get_population],
)


# --- 3. Run a query with custom tools ---
async def main():
    options = ClaudeAgentOptions(
        mcp_servers={"city_info": city_server},
        # Auto-approve all tools on this server with wildcard
        allowed_tools=["mcp__city_info__*"],
    )

    async for message in query(
        prompt="What's the weather and population of Lisbon?",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(message.result)


if __name__ == "__main__":
    asyncio.run(main())
