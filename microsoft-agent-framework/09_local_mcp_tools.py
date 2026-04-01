import asyncio

from dotenv import load_dotenv

from agent_framework import Agent, MCPStdioTool
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Local MCP tools via MCPStdioTool (stdio transport)
- Using npx-based MCP servers as agent tools
- Async context manager pattern for MCP lifecycle

MCP (Model Context Protocol) tools connect agents to
external capabilities through a standardized protocol.
MCPStdioTool launches a local subprocess and communicates
via stdin/stdout — ideal for local tool servers.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/tools/local-mcp-tools?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 2. Create an MCP tool using a local stdio server ---
    # This example uses the official MCP filesystem server.
    # It requires Node.js / npx to be installed.
    # In production, you can use any MCP-compatible server.
    async with MCPStdioTool(
        name="filesystem",
        command="npx",
        args=["-y", "@modelcontextprotocol/server-filesystem", "."],
        load_prompts=False,
    ) as mcp_tool:
        # --- 3. Create the agent with the MCP tool ---
        agent = client.as_agent(
            name="file-explorer",
            instructions=(
                "You are a file exploration assistant. "
                "Use your tools to list and read files. Be concise."
            ),
            tools=[mcp_tool],
        )

        # --- 4. Run the agent — it will use MCP tools to explore files ---
        print("Asking agent to list files in the current directory...")
        result = await agent.run("List the files in the current directory.")

        # --- 5. Print the result ---
        print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
