import asyncio
import os

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_mcp_adapters.client import MultiServerMCPClient

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Model Context Protocol (MCP) integration
- Creating a custom MCP server with FastMCP
- Loading MCP tools into an agent via MultiServerMCPClient

MCP is an open protocol that standardizes how applications provide
tools and context to LLMs. LangChain agents can use MCP tools via
the langchain-mcp-adapters library. This example creates a local
MCP math server and connects it to an agent as a tool source.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/mcp
-------------------------------------------------------
"""

# The MCP server tools are defined inline below because the server
# runs as a subprocess via stdio transport. FastMCP is imported only
# in the subprocess command string.

MCP_SERVER_CODE = """
from fastmcp import FastMCP

mcp = FastMCP("Math")

@mcp.tool()
def add(a: int, b: int) -> int:
    "Add two numbers"
    return a + b

@mcp.tool()
def multiply(a: int, b: int) -> int:
    "Multiply two numbers"
    return a * b

@mcp.tool()
def factorial(n: int) -> int:
    "Calculate factorial of n (up to 20)"
    if n < 0 or n > 20:
        return -1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

if __name__ == "__main__":
    mcp.run(transport="stdio")
"""


# --- 1. Main async function ---
async def main():
    # --- 2. Connect to the MCP server via stdio ---
    client = MultiServerMCPClient(
        {
            "math": {
                "transport": "stdio",
                "command": "python",
                "args": ["-c", MCP_SERVER_CODE],
            },
        }
    )

    # --- 3. Load tools from MCP server ---
    tools = await client.get_tools()
    print(f"Loaded {len(tools)} tools from MCP server: {[t.name for t in tools]}\n")

    # --- 4. Create agent with MCP tools ---
    agent = create_agent(
        model=init_chat_model(f"openai:{settings.OPENAI_MODEL_NAME}"),
        tools=tools,
        system_prompt="You are a math assistant. Use the available tools to solve math problems.",
    )

    # --- 5. Use the agent with MCP tools ---
    print("=== Math Query 1: Addition and Multiplication ===")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is (3 + 5) * 12?"}]}
    )
    print(f"{result['messages'][-1].content}\n")

    print("=== Math Query 2: Factorial ===")
    result = await agent.ainvoke(
        {"messages": [{"role": "user", "content": "What is the factorial of 7?"}]}
    )
    print(f"{result['messages'][-1].content}")


if __name__ == "__main__":
    asyncio.run(main())
