import os
import sys
import tempfile

from mcp import StdioServerParameters

from smolagents import CodeAgent, OpenAIModel, ToolCollection

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents MCP tool integration:

- Creating tools from an MCP (Model Context Protocol) server
- Using ToolCollection.from_mcp() to load MCP tools
- Running an agent with dynamically loaded MCP tools
- Writing a minimal MCP server for demonstration

MCP is an open standard that lets agents discover and use
tools exposed by external servers. smolagents integrates
with MCP servers via ToolCollection.from_mcp(), making it
easy to use tools from any MCP-compatible server.

For more details, visit:
https://huggingface.co/docs/smolagents/tutorials/tools#tool-collection
-------------------------------------------------------
"""

# --- 1. Create a minimal MCP server script ---
# This creates a temporary Python MCP server that exposes a simple tool.
MCP_SERVER_CODE = '''
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")

@mcp.tool()
def get_country_info(country: str) -> str:
    """Get basic information about a country.

    Args:
        country: The name of the country.
    """
    data = {
        "france": "France: Capital Paris, population 67M, language French, currency Euro",
        "japan": "Japan: Capital Tokyo, population 125M, language Japanese, currency Yen",
        "brazil": "Brazil: Capital Brasilia, population 214M, language Portuguese, currency Real",
    }
    return data.get(country.lower(), f"No information available for {country}")

@mcp.tool()
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature value to convert.
        from_unit: Source unit ('celsius' or 'fahrenheit').
        to_unit: Target unit ('celsius' or 'fahrenheit').
    """
    if from_unit.lower() == "celsius" and to_unit.lower() == "fahrenheit":
        result = (value * 9/5) + 32
        return f"{value}°C = {result:.1f}°F"
    elif from_unit.lower() == "fahrenheit" and to_unit.lower() == "celsius":
        result = (value - 32) * 5/9
        return f"{value}°F = {result:.1f}°C"
    return f"Cannot convert from {from_unit} to {to_unit}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
'''

# --- 2. Write the MCP server to a temp file ---
server_file = os.path.join(tempfile.gettempdir(), "smolagents_mcp_demo_server.py")
with open(server_file, "w") as f:
    f.write(MCP_SERVER_CODE)

print("=== MCP Tools Demo ===\n")
print(f"MCP server script: {server_file}\n")

# --- 3. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

# --- 4. Load tools from MCP server and run agent ---
# ToolCollection.from_mcp is a context manager that manages the MCP server lifecycle
with ToolCollection.from_mcp(
    StdioServerParameters(command=sys.executable, args=[server_file]),
    trust_remote_code=True,
    structured_output=False,
) as tool_collection:
    print(f"Loaded {len(tool_collection.tools)} tools from MCP server:")
    for t in tool_collection.tools:
        print(f"  - {t.name}: {t.description[:80]}")

    # --- 5. Create agent with MCP tools ---
    agent = CodeAgent(
        tools=tool_collection.tools,
        model=model,
        max_steps=6,
    )

    # --- 6. Run a query using MCP tools ---
    print("\n--- Query: Country info via MCP ---")
    result = agent.run(
        "Get information about Japan and convert 30 celsius to fahrenheit. "
        "Summarize in 2-3 sentences."
    )
    print(f"Result: {result}")

# --- 7. Clean up ---
os.unlink(server_file)
