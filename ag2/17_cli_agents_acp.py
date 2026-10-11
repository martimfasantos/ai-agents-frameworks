import asyncio
import os

from ag2 import Agent
from ag2.acp import ClaudeCodeConfig
from ag2.events import ModelMessageChunk, ModelReasoning
from ag2.events.tool_events import BuiltinToolCallEvent

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- ag2.acp configs driving an external CLI coding agent (Claude Code)
- An ACP-backed Agent using the same run() / reply API
- Streaming the CLI agent's thoughts and tool calls as AG2 events

The Agent Client Protocol lets AG2 drive Claude Code, Codex or
OpenCode as ordinary agents: the config IS the model config, so no
Agent API changes. Everything the CLI agent does is externalised
onto AG2's event stream, so it can be observed and gated like any
other agent. CodexConfig and OpenCodeConfig are drop-in
replacements for the ClaudeCodeConfig used here.

Requires Node.js (the Claude Code ACP adapter is launched through
npx) and an existing Claude Code login under ~/.claude.

For more details, visit:
https://docs.ag2.ai/docs/user-guide/acp/client/
-------------------------------------------------------
"""

WORKSPACE = "res/acp_workspace"


def observe(event: object) -> None:
    """Render the CLI agent's work as it streams in."""
    if isinstance(event, ModelReasoning):
        print(f"  [thinking] {event.content}")
    elif isinstance(event, BuiltinToolCallEvent):
        print(f"  [tool] {event.name}({event.arguments})")
    elif isinstance(event, ModelMessageChunk):
        print(event.content, end="", flush=True)


async def main() -> None:
    os.makedirs(WORKSPACE, exist_ok=True)

    # --- 1. Configure the Claude Code adapter as the agent's "model" ---
    # permission_policy="auto" approves the CLI agent's file edits.
    config = ClaudeCodeConfig(
        command=["npx", "-y", "@agentclientprotocol/claude-agent-acp"],
        cwd=os.path.abspath(WORKSPACE),
        permission_policy="auto",
        startup_timeout=120.0,
        turn_timeout=180.0,
    )

    # --- 2. Drive the CLI agent for one turn, watching its events live ---
    print("=== Driving Claude Code over ACP ===\n")
    async with config:
        agent = Agent("coder", config=config)
        async with agent.run("Create hello.txt containing the single word 'hello'.") as run:
            run.stream.subscribe(observe)
            reply = await run.result()

    print(f"\n\nClaude Code: {reply.body}")
    print(f"Workspace files: {sorted(os.listdir(WORKSPACE))}")


if __name__ == "__main__":
    asyncio.run(main())
