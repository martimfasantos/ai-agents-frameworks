import asyncio
import json

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    HookEventMessage,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Hook event streaming with include_hook_events=True
- Receiving HookEventMessage in the message stream
- Observing PreToolUse and PostToolUse lifecycle events
- Registering command hooks inline through the settings option

When include_hook_events is enabled, hook lifecycle events are emitted
as HookEventMessage objects in the message stream: hook_started when a
hook begins and hook_response with its outcome and output. This gives your
application visibility into every hook invocation without logging inside
the hooks themselves. Only hooks the CLI runs itself (command, http,
mcp_tool) emit these events; in-process callback hooks passed through
hooks=... do not.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/hooks
-------------------------------------------------------
"""

# --- 1. Define command hooks as inline settings JSON ---
HOOK_SETTINGS = {
    "hooks": {
        "PreToolUse": [
            {"matcher": "Glob", "hooks": [{"type": "command", "command": "echo pre-glob-check"}]}
        ],
        "PostToolUse": [
            {"matcher": "Glob", "hooks": [{"type": "command", "command": "echo post-glob-audit"}]}
        ],
    }
}

# --- 2. Configure with hook event streaming enabled ---
options = ClaudeAgentOptions(
    model=settings.ANTHROPIC_MODEL_NAME,
    allowed_tools=["Read", "Glob"],
    permission_mode="bypassPermissions",
    include_hook_events=True,
    settings=json.dumps(HOOK_SETTINGS),
)


# --- 3. Run and observe hook events in the stream ---
async def main():
    print("=== Hook Event Streaming ===\n")
    hook_event_count = 0

    async for message in query(
        prompt="Use Glob once to find files matching '[0-9][0-9]_*.py', then reply with just the number of matches.",
        options=options,
    ):
        if isinstance(message, HookEventMessage):
            hook_event_count += 1
            data = message.data
            print(f"[HookEvent #{hook_event_count}]")
            print(f"  Event:   {message.hook_event_name}")
            print(f"  Subtype: {message.subtype}")
            print(f"  Hook:    {data.get('hook_name')}")
            if message.subtype == "hook_response":
                print(f"  Outcome: {data.get('outcome')} (stdout: {data.get('stdout', '').strip()})")
            print()

        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(f"--- Result ---")
            print(f"{message.result}")
            print(f"\nTotal hook events received: {hook_event_count}")


if __name__ == "__main__":
    asyncio.run(main())
