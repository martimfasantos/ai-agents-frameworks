import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    AssistantMessage,
    ToolUseBlock,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Connecting to external MCP servers via stdio transport
- Configuring McpStdioServerConfig with command and args
- Using tools exposed by external MCP servers

External MCP servers run as separate processes and communicate via
stdio, HTTP, or SSE. This is useful for connecting to third-party
tool servers like filesystem, database, or API gateways. The SDK
manages the server lifecycle automatically.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/mcp
-------------------------------------------------------
"""

# --- 1. Configure an external MCP server (stdio) ---
# This example uses npx to run the official filesystem MCP server.
# It provides tools like read_file, write_file, list_directory, etc.
# Make sure Node.js is installed for this example to work.

options = ClaudeAgentOptions(
    mcp_servers={
        "filesystem": {
            "type": "stdio",
            "command": "npx",
            "args": [
                "-y",
                "@modelcontextprotocol/server-filesystem",
                ".",  # Allow access to current directory only
            ],
        },
    },
    # Auto-approve read operations from the filesystem server
    allowed_tools=["mcp__filesystem__read_file", "mcp__filesystem__list_directory"],
    permission_mode="bypassPermissions",
)


# --- 2. Query using the external MCP server's tools ---
async def main():
    async for message in query(
        prompt="Use the filesystem MCP tools to list the files in the current directory, then read settings.py.",
        options=options,
    ):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"[MCP Tool] {block.name}: {block.input}")

        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n--- Result ---\n{message.result}")


if __name__ == "__main__":
    asyncio.run(main())
