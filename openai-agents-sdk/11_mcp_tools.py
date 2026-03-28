import asyncio
import os
from agents import Agent, Runner, HostedMCPTool
from openai.types.responses import ResponseTextDeltaEvent
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------------------------
In this example, we explore Model Context Protocol (MCP) tools —
a standardized way to connect AI agents to external data sources
and tools, like a "USB-C port for AI applications."

Features demonstrated:
- HostedMCPTool: OpenAI-hosted MCP, no local server needed
- The model lists and calls remote MCP tools automatically
- Streaming MCP results with Runner.run_streamed
- Tool approval flows for hosted MCP servers

HostedMCPTool is unique to the OpenAI Agents SDK — it delegates tool
execution to OpenAI's infrastructure, so you don't need to run any
local MCP servers. Just point at a URL and the model handles the rest.

Other MCP transports also available (not shown here):
- MCPServerStdio: local subprocess over stdin/stdout
- MCPServerSse: HTTP with Server-Sent Events (legacy)
- MCPServerStreamableHttp: Streamable HTTP (recommended for self-hosted)
-------------------------------------------------------------------------
"""


async def main():
    print("=== MCP Tools Example ===\n")

    # 1. Define a HostedMCPTool pointing at a public MCP server
    #    gitmcp.io provides MCP access to GitHub repositories
    #    No local server setup needed — OpenAI hosts the tool execution
    mcp_tool = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "gitmcp",
            "server_url": "https://gitmcp.io/openai/openai-agents-python",
            "require_approval": "never",
        }
    )

    # 2. Create an agent with the hosted MCP tool
    agent = Agent(
        name="Repo Explorer",
        instructions=(
            "You are a helpful assistant that can explore GitHub repositories "
            "using MCP tools. Answer questions about the repository structure, "
            "code, and documentation. Be concise."
        ),
        tools=[mcp_tool],
        model=settings.OPENAI_MODEL_NAME,
    )

    # 3. Ask a question — the model will automatically discover and use MCP tools
    print("--- Basic MCP query ---")
    print(
        "User: What programming language is the openai-agents-python repo written in?\n"
    )
    result = await Runner.run(
        agent, "What programming language is the openai-agents-python repo written in?"
    )
    print(f"Assistant: {result.final_output}\n")

    # 4. Streaming MCP results — consume output incrementally
    print("--- Streaming MCP query ---")
    print("User: What are the main features of the OpenAI Agents SDK?\n")
    print("Assistant: ", end="", flush=True)

    result = Runner.run_streamed(
        agent, "What are the main features of the OpenAI Agents SDK? Be brief."
    )
    async for event in result.stream_events():
        if event.type == "raw_response_event":
            if isinstance(event.data, ResponseTextDeltaEvent):
                print(event.data.delta, end="", flush=True)
    print(f"\n")

    # 5. Demonstrate MCP with approval required
    print("--- MCP with approval flow ---")
    mcp_tool_with_approval = HostedMCPTool(
        tool_config={
            "type": "mcp",
            "server_label": "gitmcp_approved",
            "server_url": "https://gitmcp.io/openai/openai-agents-python",
            "require_approval": "always",
        },
        # on_approval_request provides programmatic auto-approval
        # In production, you might prompt a human instead
        on_approval_request=lambda request: {"approve": True},
    )

    agent_with_approval = Agent(
        name="Approved Repo Explorer",
        instructions="Explore the repository using MCP tools. Be concise.",
        tools=[mcp_tool_with_approval],
        model=settings.OPENAI_MODEL_NAME,
    )

    print("User: How many examples are in the repository?\n")
    result = await Runner.run(
        agent_with_approval, "How many example files are in the repository?"
    )
    print(f"Assistant: {result.final_output}\n")

    print("=== MCP Tools Demo Complete ===")
    print("HostedMCPTool lets you connect to any MCP server without local setup!")


if __name__ == "__main__":
    asyncio.run(main())
