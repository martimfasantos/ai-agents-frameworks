import asyncio
from typing import cast

from dotenv import load_dotenv

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    ResultMessage,
    HookMatcher,
    HookInput,
    HookContext,
    HookJSONOutput,
    StopHookInput,
    StopHookSpecificOutput,
    PostToolUseHookInput,
)

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Typed hook inputs: StopHookInput and PostToolUseHookInput
- Reading last_assistant_message in a Stop hook to check the final answer
- Returning StopHookSpecificOutput.additionalContext to make the agent continue
- Using stop_hook_active to avoid an endless continuation loop
- Timing every tool call with PostToolUseHookInput.duration_ms

A Stop hook runs when the agent is about to end its turn. Here it acts as
an output-format guard: if the answer has no "TL;DR:" line, it returns
additionalContext and Claude Code keeps the turn going so the model can
fix it. Unlike decision="block", this is labelled as hook feedback rather
than a hook error. On the retry stop_hook_active is True, so the hook lets
the agent stop instead of looping.

For more details, visit:
https://code.claude.com/docs/en/hooks#stop-decision-control
-------------------------------------------------------
"""

REQUIRED_PREFIX = "TL;DR:"


# --- 1. PostToolUse hook: report how long each tool call took ---
async def time_tool_call(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Print the execution time the CLI measured for each tool call."""
    post_input = cast(PostToolUseHookInput, hook_input)
    duration = post_input.get("duration_ms")
    print(f"  [PostToolUse] {post_input['tool_name']:<5} took {duration} ms")
    return {}


# --- 2. Stop hook: require a TL;DR line before the agent may stop ---
async def require_tldr(
    hook_input: HookInput,
    tool_use_id: str | None,
    context: HookContext,
) -> HookJSONOutput:
    """Send the agent back to work when its last answer lacks a TL;DR line."""
    stop_input = cast(StopHookInput, hook_input)
    last_message = stop_input.get("last_assistant_message", "")
    has_tldr = any(line.strip().startswith(REQUIRED_PREFIX) for line in last_message.splitlines())

    print(f"\n  [Stop] stop_hook_active={stop_input['stop_hook_active']}, has TL;DR line={has_tldr}")
    print(f"  [Stop] last_assistant_message: {last_message!r}")

    # stop_hook_active is True when this stop follows a hook continuation
    if has_tldr or stop_input["stop_hook_active"]:
        print("  [Stop] -> allow the agent to stop")
        return {}

    print("  [Stop] -> missing TL;DR, returning additionalContext so the agent continues")
    feedback: StopHookSpecificOutput = {
        "hookEventName": "Stop",
        "additionalContext": (
            f"Your answer must end with a single line that starts with '{REQUIRED_PREFIX}' "
            "summarising it in under 15 words. Repeat your answer with that line added."
        ),
    }
    return {"hookSpecificOutput": feedback}


# --- 3. Register both hooks ---
options = ClaudeAgentOptions(
    model=settings.ANTHROPIC_MODEL_NAME,
    system_prompt="You are a concise assistant. Answer in at most two sentences.",
    allowed_tools=["Glob", "Read"],
    hooks={
        "PostToolUse": [HookMatcher(matcher=None, hooks=[time_tool_call])],
        "Stop": [HookMatcher(matcher=None, hooks=[require_tldr])],
    },
)


# --- 4. Run the agent and check the final answer ---
async def main():
    prompt = (
        "Use Glob to count the files matching '[0-9][0-9]_*.py', then Read settings.py "
        "and tell me how many examples there are and which default model they use."
    )
    print(f"Prompt: {prompt}\n")

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            final = message.result or ""
            print(f"\n--- Final answer ---\n{final}")
            print(f"\nEnds with a TL;DR line: {REQUIRED_PREFIX in final}")


if __name__ == "__main__":
    asyncio.run(main())
