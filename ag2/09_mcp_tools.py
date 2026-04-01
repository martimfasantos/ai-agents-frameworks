import asyncio
import os
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from autogen import ConversableAgent, LLMConfig
from autogen.mcp import create_toolkit

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- MCP (Model Context Protocol) tool integration
- Using create_toolkit() to load tools from an MCP server
- Connecting to a local FastMCP server via stdio transport

AG2 can connect to MCP servers to dynamically discover and
use tools. The create_toolkit() function converts MCP tools
into AG2-compatible tools that agents can call. This example
uses a local calculator MCP server (mcp_server.py).

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/mcp/client/
-------------------------------------------------------
"""


async def main() -> None:
    """Run the MCP tools example."""
    # --- 1. Configure LLM ---
    llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

    # --- 2. Set up the MCP server connection ---
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )

    # --- 3. Connect to MCP server and create toolkit ---
    print("=== MCP Tools: Calculator Agent ===\n")
    print("Connecting to MCP server...")

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            toolkit = await create_toolkit(session=session, use_mcp_tools=True)
            print(f"Loaded {len(toolkit.tools)} tools from MCP server\n")

            # --- 4. Create the agent with MCP tools ---
            agent = ConversableAgent(
                name="calculator",
                system_message=(
                    "You are a calculator agent. Use the provided tools to "
                    "solve math problems. Show your work step by step. "
                    "Reply TERMINATE when the calculation is complete."
                ),
                llm_config=llm_config,
                human_input_mode="NEVER",
            )

            # --- 5. Create user proxy and register tools ---
            user = ConversableAgent(
                name="user",
                human_input_mode="NEVER",
                llm_config=False,
                is_termination_msg=lambda x: (
                    "TERMINATE" in (x.get("content", "") or "")
                ),
            )

            toolkit.register_for_llm(agent)
            toolkit.register_for_execution(user)

            # --- 6. Run the calculation (async for MCP compatibility) ---
            result = await user.a_initiate_chat(
                agent,
                message="Calculate (15 + 27) * 3 using the tools.",
                max_turns=5,
            )

            print("\n=== MCP Tools Demo Complete ===")
            print(f"Final answer: {result.summary}")


if __name__ == "__main__":
    asyncio.run(main())
