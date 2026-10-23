import asyncio
import os
from typing import Annotated

from ag2 import Agent, AgentReply, Plugin, Variable, tool
from ag2.config import OpenAIConfig
from ag2.events import ToolCallEvent, ToolResultEvent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Plugin bundling a prompt fragment, a tool, and a context variable
- Passing it to a single agent.ask() call via plugins=
- The same agent answering without any of it on the next call

Plugins normally attach when an agent is built. Passing plugins=
to ask(), run(), resume() or reply.ask() scopes them to that one
invocation: the prompt fragment, tools and variable defaults are
removed when the call ends. One long-lived agent can then serve
each caller with the capabilities and identity of that caller.

For more details, visit:
https://docs.ag2.ai/docs/user-guide/system_prompts/#plugins-for-one-invocation
-------------------------------------------------------
"""

QUESTION = "How many loyalty points do I have?"


# --- 1. A tool that only the plugin brings ---
# customer_id is injected from context variables, so the model never sees it.
@tool
def get_loyalty_points(customer_id: Annotated[str, Variable()]) -> str:
    """Look up the signed-in customer's loyalty points balance."""
    balances = {"C-1001": 4200, "C-2002": 150}
    return f"{customer_id} has {balances.get(customer_id, 0)} points"


async def show_tool_activity(reply: AgentReply) -> None:
    """Print the tool calls and results recorded in a reply's history."""
    events = await reply.context.stream.history.get_events()
    calls = [e for e in events if isinstance(e, (ToolCallEvent, ToolResultEvent))]
    if not calls:
        print("  (no tool calls)")
    for event in calls:
        if isinstance(event, ToolCallEvent):
            print(f"  -> {event.name}({event.arguments})")
        else:
            text = " ".join(str(part.content) for part in event.result.parts)
            print(f"  <- {text!r}")


async def main() -> None:
    # --- 2. A plain agent with no tools of its own ---
    agent = Agent(
        "support",
        prompt=(
            "You are a store support assistant. Answer in one short sentence. "
            "If you have no way to look something up, say so."
        ),
        config=OpenAIConfig(model=settings.OPENAI_MODEL_NAME),
    )

    # --- 3. A plugin carrying a prompt fragment, a tool, and a variable ---
    loyalty = Plugin(
        prompt="Use get_loyalty_points for balances. End every answer with '-- Loyalty Desk'.",
        tools=[get_loyalty_points],
        variables={"customer_id": "C-1001"},
    )

    # --- 4. One call with the plugin ---
    print("=== Call 1: agent.ask(..., plugins=[loyalty]) ===")
    reply = await agent.ask(QUESTION, plugins=[loyalty])
    print(f"Agent: {reply.body}")
    await show_tool_activity(reply)

    # --- 5. The next call without it: tool, prompt and variable are gone ---
    print("\n=== Call 2: agent.ask(...) without plugins ===")
    plain = await agent.ask(QUESTION)
    print(f"Agent: {plain.body}")
    await show_tool_activity(plain)

    print(f"\nAgent's own tools after both calls: {[t.name for t in agent.tools]}")


if __name__ == "__main__":
    asyncio.run(main())
