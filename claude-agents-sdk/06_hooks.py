import asyncio
from typing import Any

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    HookMatcher,
    HookInput,
    HookJSONOutput,
    HookContext,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- PreToolUse hooks to intercept and log tool calls before execution
- PostToolUse hooks to inspect tool results after execution
- Hook matchers to target specific tools by name
- Returning hook decisions (allow, deny, additionalContext)

Hooks let you observe, modify, or block tool calls at runtime. A
PreToolUse hook fires before a tool runs — you can log, modify input,
or deny the call. A PostToolUse hook fires after, letting you inspect
results or inject additional context for the model.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/hooks
-------------------------------------------------------
"""


# --- 1. Define hook callbacks ---
async def log_pre_tool_use(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Log every tool call before execution."""
    tool_name = hook_input.get("tool_name", "unknown")
    tool_args = hook_input.get("tool_input", {})
    print(f"  [PreToolUse] About to call: {tool_name}")
    print(f"               Input: {tool_args}")

    # Allow the tool to proceed
    return {"continue_": True}


async def post_tool_add_context(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Add context after a tool runs."""
    tool_name = hook_input.get("tool_name", "unknown")
    print(f"  [PostToolUse] Finished: {tool_name}")

    return {
        "continue_": True,
        "hookSpecificOutput": {
            "hookEventName": "PostToolUse",
            "additionalContext": f"The {tool_name} tool completed successfully.",
        },
    }


async def block_bash(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Block Bash tool calls via a PreToolUse hook."""
    print("  [PreToolUse] BLOCKED: Bash tool is not allowed!")
    return {
        "continue_": True,
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "Bash is blocked by security policy.",
        },
    }


# --- 2. Configure hooks with matchers ---
options = ClaudeAgentOptions(
    allowed_tools=["Read", "Glob"],
    hooks={
        # Log all tool calls before execution
        "PreToolUse": [
            HookMatcher(matcher=None, hooks=[log_pre_tool_use]),
            # Block Bash specifically
            HookMatcher(matcher="Bash", hooks=[block_bash]),
        ],
        # Add context after all tool calls
        "PostToolUse": [
            HookMatcher(matcher=None, hooks=[post_tool_add_context]),
        ],
    },
)


# --- 3. Run the agent ---
async def main():
    async for message in query(
        prompt="List all .py files in the current directory.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"\n--- Result ---\n{message.result}")


if __name__ == "__main__":
    asyncio.run(main())
