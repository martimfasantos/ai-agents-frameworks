import asyncio
import os
import tempfile
from pathlib import Path

from langchain.agents import create_agent
from langchain.mcp import MCPAdapter
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------
In this example, we explore LangChain with the following features:
- The first-party langchain.mcp namespace, new in LangChain 1.4.0
- MCPAdapter to turn any MCP server into LangChain tools
- Passing a local server as a Path, not a string
- MCPAdapter.list_tools() feeding create_agent directly

The Model Context Protocol lets an agent borrow tools from any MCP
server. Until 1.4.0 this meant the separate langchain-mcp-adapters
package; MCPAdapter brings it in-tree. It delegates transport
negotiation to FastMCP, converts discovered MCP tools into async
LangChain tools, and — unlike the third-party adapter — answers a server
that asks for input mid-call with a LangGraph interrupt(), so a human
can respond and the run resumes.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/mcp
-----------------------------------------------------------------------
"""

# The server runs as a subprocess over stdio, so its tools are defined in
# a standalone script written to a temp file.
MCP_SERVER_CODE = '''
from fastmcp import FastMCP

mcp = FastMCP("Math")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@mcp.tool()
def factorial(n: int) -> int:
    """Calculate factorial of n (up to 20)"""
    if n < 0 or n > 20:
        return -1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


if __name__ == "__main__":
    mcp.run(transport="stdio")
'''


async def main():
    # --- 1. Write the server script to disk ---
    with tempfile.TemporaryDirectory() as tmpdir:
        server_path = Path(tmpdir) / "math_server.py"
        server_path.write_text(MCP_SERVER_CODE)

        # --- 2. Adapt the server into LangChain tools ---
        # A Path is required for a local server. MCPAdapter rejects a plain
        # string that is not an http(s) URL, so a filename arriving from
        # config or from a model can never silently launch a subprocess.
        adapter = MCPAdapter(server_path)
        tools = await adapter.list_tools()

        print("=== MCPAdapter ===\n")
        print("Tools discovered on the MCP server:")
        for tool in tools:
            print(f"  {tool.name}: {tool.description}")

        # --- 3. Hand them straight to an agent ---
        agent = create_agent(
            model=ChatOpenAI(model=settings.OPENAI_MODEL_NAME),
            tools=tools,
            system_prompt="Use the tools to compute. Answer with just the number.",
        )

        for question in ("What is 17 plus 25?", "What is 6 factorial?"):
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            called = [
                call["name"]
                for message in result["messages"]
                for call in getattr(message, "tool_calls", None) or []
            ]
            print(f"\nQ: {question}")
            print(f"   tools used: {called}")
            print(f"   answer:     {result['messages'][-1].text}")


if __name__ == "__main__":
    asyncio.run(main())
