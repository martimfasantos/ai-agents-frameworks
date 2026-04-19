import asyncio

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.mcp import MCPTools
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- MCP (Model Context Protocol) tool integration
- Connecting to an MCP server via stdio transport
- Using MCPTools as an async context manager
- Exposing external tool servers to an Agno agent

MCP is an open protocol that lets agents use tools exposed by
external servers. Agno's MCPTools connects to any MCP-compatible
server and makes its tools available to the agent. This example
uses the official MCP filesystem server to demonstrate reading
files from the local filesystem.

For more details, visit:
https://docs.agno.com/tools/mcp-tools
-------------------------------------------------------
"""


# --- 1. Run agent with MCP tools ---
async def main() -> None:
    # Connect to the MCP filesystem server via stdio.
    # This requires npx (Node.js) installed on the system.
    # The server exposes tools like read_file, list_directory, etc.
    async with MCPTools(
        command="npx -y @modelcontextprotocol/server-filesystem /tmp",
    ) as mcp_tools:
        agent = Agent(
            model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
            tools=[mcp_tools],
            instructions=[
                "You are a file system assistant.",
                "Use the available MCP tools to interact with the filesystem.",
                "Be helpful and concise. Keep responses to 3-5 lines max.",
            ],
            markdown=False,
        )

        # Ask the agent to list files in /tmp
        run_output = await agent.arun(
            "List the files and directories in /tmp. Just show the first 10 entries."
        )
        pprint_run_response(run_output)


if __name__ == "__main__":
    asyncio.run(main())
