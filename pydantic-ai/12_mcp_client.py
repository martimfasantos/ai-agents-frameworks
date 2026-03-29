import asyncio
import sys

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.mcp import MCPServerStdio

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- Connecting to MCP (Model Context Protocol) servers
- MCPServerStdio for local stdio-based MCP servers
- Using MCP tools as agent toolsets
- Listing available tools from an MCP server

MCP is an open protocol that lets AI agents discover and use tools
exposed by external servers. Pydantic AI's MCP client supports stdio,
SSE, and StreamableHTTP transports. This example uses MCPServerStdio
to connect to a local filesystem MCP server that exposes file operations.

NOTE: This example requires the 'mcp' package and a local MCP server.
If the MCP server is not available, it will print available info and exit.

For more details, visit:
https://ai.pydantic.dev/mcp/client/
-----------------------------------------------------------------------
"""


async def main():
    print("=== MCP Client Example ===\n")

    # --- 1. Define the MCP server connection ---
    # Using npx to run the filesystem MCP server (reads the current directory)
    # This requires Node.js and npx to be installed.
    mcp_server = MCPServerStdio(
        "npx",
        args=["-y", "@anthropic-ai/mcp-filesystem", "."],
        tool_prefix="fs",  # Prefix all tools with "fs_" to avoid collisions
    )

    # --- 2. Create agent with MCP toolset ---
    agent = Agent(
        model=settings.OPENAI_MODEL_NAME,
        instructions=(
            "You are a file system assistant. Use the available tools "
            "to help with file operations. Be concise in your responses."
        ),
        toolsets=[mcp_server],
    )

    # --- 3. Run the agent with MCP tools ---
    try:
        async with agent:
            # List available tools from the MCP server
            print("Step 1: Discovering tools from MCP server...")
            print()

            # Ask the agent to list files
            result = await agent.run("List the Python files in the current directory.")
            print(f"Response: {result.output}")
            print()

            # Ask about a specific file
            print("Step 2: Reading a file...")
            result2 = await agent.run(
                "Read the settings.py file and tell me what model is configured."
            )
            print(f"Response: {result2.output}")

    except FileNotFoundError:
        print("NOTE: 'npx' not found. MCP stdio server requires Node.js.")
        print("Install Node.js to run this example: https://nodejs.org/")
        print()
        print("MCP server configuration shown for reference:")
        print(f"  Command: npx")
        print(f"  Args: -y @anthropic-ai/mcp-filesystem .")
        print(f"  Tool prefix: fs")
        sys.exit(0)
    except Exception as e:
        print(f"MCP connection error: {e}")
        print()
        print("This is expected if the MCP server binary is not installed.")
        print("The example demonstrates the MCP client configuration pattern.")
        sys.exit(0)


if __name__ == "__main__":
    asyncio.run(main())
