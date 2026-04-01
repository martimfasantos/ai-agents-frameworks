import asyncio

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    ToolUseBlock,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Using built-in tools (Read, Bash, Glob, Grep)
- Configuring allowed_tools to auto-approve specific tools
- Inspecting tool use blocks in assistant messages

The SDK ships with built-in tools like Read, Bash, Glob, and Grep
that let Claude interact with the local filesystem. The allowed_tools
list acts as a permission allowlist — tools listed there run without
prompting. Tools not listed still exist but require approval.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/overview#capabilities
-------------------------------------------------------
"""

# --- 1. Configure the agent with built-in tools ---
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Glob"],
    permission_mode="bypassPermissions",
)


# --- 2. Run a query that uses built-in tools ---
async def main():
    async for message in query(
        prompt="List all Python files in the current directory and show the first 5 lines of settings.py",
        options=options,
    ):
        # Inspect tool calls made by the assistant
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, ToolUseBlock):
                    print(f"[Tool Call] {block.name}: {block.input}")

        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n--- Result ---\n{message.result}")


if __name__ == "__main__":
    asyncio.run(main())
