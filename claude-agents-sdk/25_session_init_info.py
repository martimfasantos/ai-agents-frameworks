import asyncio
from typing import Any, cast

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    SystemInitData,
    tool,
    create_sdk_mcp_server,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Reading the init SystemMessage (subtype "init") at the start of a session
- Narrowing its payload with cast(SystemInitData, msg.data) for typed access
- Inspecting model, CLI version, permission mode, tools, and MCP server status
- Feature detection through the capabilities list instead of version checks

Before the first reply, the CLI sends a system/init message describing the
session it started: which model and Claude Code version are running, which
tools and MCP servers are available, and which protocol capabilities the
CLI supports. The SDK types this payload as SystemInitData, so you can
check that an MCP server actually connected, or that the CLI supports
interrupt receipts, before relying on them.

For more details, visit:
https://code.claude.com/docs/en/agent-sdk/agent-loop#message-types
-------------------------------------------------------
"""


# --- 1. A small in-process MCP server so the init payload lists one ---
@tool("get_time", "Get the current time in a city", {"city": str})
async def get_time(args: dict[str, Any]) -> dict[str, Any]:
    """Simulated clock lookup."""
    return {"content": [{"type": "text", "text": f"It is 14:30 in {args['city']}."}]}


clock_server = create_sdk_mcp_server(name="clock", version="1.0.0", tools=[get_time])

options = ClaudeAgentOptions(
    model=settings.ANTHROPIC_MODEL_NAME,
    mcp_servers={"clock": clock_server},
    allowed_tools=["mcp__clock__*"],
    max_turns=1,
)


# --- 2. Print the typed fields of the init message ---
def show_init(init: SystemInitData) -> None:
    tools = init["tools"]
    mcp_tools = [name for name in tools if name.startswith("mcp__")]
    print(f"Model:             {init['model']}")
    print(f"Claude Code:       {init['claude_code_version']}")
    print(f"Permission mode:   {init['permissionMode']}")
    print(f"Tools:             {len(tools)} (first five: {', '.join(tools[:5])})")
    print(f"MCP tools:         {', '.join(mcp_tools)}")
    for server in init["mcp_servers"]:
        print(f"MCP server:        {server['name']} -> {server['status']}")

    # Optional keys: absent on older CLIs, so read them with .get()
    capabilities = init.get("capabilities", [])
    print(f"Capabilities:      {capabilities}")
    print(f"Fast mode state:   {init.get('fast_mode_state', 'not reported')}")

    # --- 3. Feature-detect instead of comparing version strings ---
    supports_receipts = "interrupt_receipt_v1" in capabilities
    print(f"\ninterrupt_receipt_v1 supported: {supports_receipts}")


# --- 4. Run one turn and read the init message before the reply ---
async def main():
    async for message in query(prompt="Reply with just 'ready'.", options=options):
        if isinstance(message, SystemMessage) and message.subtype == "init":
            print("=== Session init ===")
            show_init(cast(SystemInitData, message.data))

        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n=== Result ===\n{message.result}")


if __name__ == "__main__":
    asyncio.run(main())
