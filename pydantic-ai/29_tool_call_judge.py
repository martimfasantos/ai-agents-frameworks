import asyncio

from dotenv import load_dotenv

from pydantic_ai import Agent, ToolCallPart, ToolReturnPart
from pydantic_ai_harness.tool_call_judge import ToolCallJudge, ToolCallVerdict

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- ToolCallJudge (pydantic-ai-harness): a second model votes on tool calls
- tools=[...] to judge only the risky tools
- A custom denial_message that the agent receives instead of the result
- on_verdict to observe every decision from application code

Whether a tool call is safe often depends on its arguments rather than
the tool: deleting a cache file is fine, deleting the production database
is not. ToolCallJudge asks a second model one risk question about each
selected call after its arguments are validated and before the tool body
runs. `yes` blocks the call, `no` lets it run, and `unsure` follows
on_uncertain ('block' by default). A blocked call is not an error: the
agent reads the denial message and carries on.

NOTE: pydantic-ai-harness is Alpha (0.x), so its API may change between
minor releases. The judge is a filter, not a security boundary.

For more details, visit:
https://pydantic.dev/docs/ai/harness/tool-call-judge/
-----------------------------------------------------------------------
"""

# Simulated file system
FILES = {
    "tmp/build-cache.log": "stale build output",
    "config/production.db": "live customer records",
}

# --- 1. Record each verdict, keyed by the call it judged ---
verdicts: dict[str, ToolCallVerdict] = {}


def record_verdict(verdict: ToolCallVerdict) -> None:
    verdicts[verdict.tool_call_id] = verdict


# --- 2. Put a judge in front of delete_file ---
judge = ToolCallJudge(
    settings.OPENAI_MODEL_NAME,
    tools=["delete_file"],
    question="Would this delete production data, customer records or configuration?",
    denial_message=(
        "The safety judge blocked this `{tool_name}` call, so nothing was deleted. "
        "Do not retry; tell the user the file was kept."
    ),
    on_verdict=record_verdict,
)

agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Call delete_file once for each file the user names. Be concise.",
    capabilities=[judge],
)


@agent.tool_plain
def delete_file(path: str) -> str:
    """Delete a file by its path."""
    FILES.pop(path, None)
    print(f"  [tool] delete_file({path!r}) ran")
    return f"Deleted {path}."


async def main() -> None:
    # --- 3. One harmless deletion and one destructive one ---
    print("=== Running the agent ===")
    result = await agent.run("Delete tmp/build-cache.log and config/production.db.")

    # --- 4. Match each verdict to its arguments and to what the agent was told ---
    print("\n=== Judge verdicts ===")
    parts = [part for message in result.all_messages() for part in message.parts]
    returns = {p.tool_call_id: p.content for p in parts if isinstance(p, ToolReturnPart)}
    for call in parts:
        if isinstance(call, ToolCallPart) and call.tool_call_id in verdicts:
            verdict = verdicts[call.tool_call_id]
            print(f"{call.tool_name}({call.args_as_dict()})")
            print(f"  vote={verdict.answer} -> {verdict.verdict}")
            print(f"  agent received: {returns[call.tool_call_id]}")

    print(f"\nFiles left: {sorted(FILES)}")
    # The judge's own model requests are added to the run's usage
    print(f"Model requests (agent + judge): {result.usage.requests}")
    print(f"Response: {result.output}")


if __name__ == "__main__":
    asyncio.run(main())
