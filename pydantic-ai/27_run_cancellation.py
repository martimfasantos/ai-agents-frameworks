import asyncio

from dotenv import load_dotenv

from pydantic_ai import Agent, RunContext
from pydantic_ai.exceptions import RunCancelled

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- First-party run cancellation with AgentRun.cancel()
- RunContext.cancel() to stop a run from inside a tool
- RunCancelled raised when the run context exits

A run sometimes has to be abandoned mid-flight: the user navigated away, a
deadline passed, or a tool discovered the work is pointless. Cancelling tears
down the in-flight model request and cancels and drains running tool tasks,
then surfaces as RunCancelled when the agent.iter() context exits — so a
cancelled run is an explicit outcome rather than a silently truncated answer.

For more details, visit:
https://ai.pydantic.dev/agents/
-------------------------------------------------------
"""

# --- 1. A slow tool, so there is something to interrupt ---
outside_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Use slow_lookup for every question. Be brief.",
)


@outside_agent.tool_plain
async def slow_lookup(topic: str) -> str:
    """Look something up. Deliberately slow."""
    print(f"  [tool] slow_lookup({topic}) started")
    await asyncio.sleep(30)
    print("  [tool] slow_lookup finished")  # only if never cancelled
    return f"Everything about {topic}."


async def cancel_from_outside() -> None:
    """Cancel a run from the code driving it."""
    print("=== Cancelling from outside the run ===\n")

    async def cancel_soon(run) -> None:
        await asyncio.sleep(2)
        print("  [driver] calling run.cancel()")
        run.cancel()

    # CancelledError must be allowed out of the `async with` block: catching it
    # inside suppresses the teardown, and the slow tool then runs to completion.
    # It surfaces as RunCancelled once the context exits.
    try:
        async with outside_agent.iter("Tell me about Lisbon.") as run:
            asyncio.create_task(cancel_soon(run))
            async for _ in run:
                pass
    except RunCancelled:
        print("  RunCancelled — the in-flight tool was torn down")


# --- 2. A tool that decides the run is pointless and stops it ---
inside_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Use check_budget before answering. Be brief.",
)


@inside_agent.tool
async def check_budget(ctx: RunContext[None], estimate: int) -> str:
    """Check whether a job fits the remaining budget."""
    print(f"  [tool] estimate={estimate}, budget=100")
    if estimate > 100:
        print("  [tool] over budget — calling ctx.cancel()")
        ctx.cancel()
    return "within budget"


async def cancel_from_inside() -> None:
    """A tool cancels the run it is part of."""
    print("\n=== Cancelling from inside a tool ===\n")
    try:
        await inside_agent.run("Plan a job that will cost about 500 units.")
    except RunCancelled:
        print("  RunCancelled — the tool stopped its own run")


async def main() -> None:
    await cancel_from_outside()
    await cancel_from_inside()


if __name__ == "__main__":
    asyncio.run(main())
