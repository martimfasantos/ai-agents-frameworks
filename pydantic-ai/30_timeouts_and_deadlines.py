import asyncio

from dotenv import load_dotenv

from pydantic_ai import (
    Agent,
    ModelAPIError,
    ModelRequestTimeout,
    RunCancelled,
    RunContext,
    RunTimedOut,
)

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- agent.run(..., timeout=N) to bound the wall-clock time of a whole run
- RunTimedOut, which keeps everything the run completed in all_messages()
- RunContext.deadline / remaining_time() so a tool can fit its work to the time left
- model_settings={"request_timeout": ..., "max_retries": 0} and ModelRequestTimeout

Two different clocks are at play. `timeout=` bounds the whole run (every
model request, tool call and hook) and raises RunTimedOut, a RunCancelled
that is also a TimeoutError. `request_timeout` bounds a single model
request across all of the provider SDK's retries and raises
ModelRequestTimeout, a ModelAPIError -- so a FallbackModel can move on to
another model when one is too slow, while a run timeout never falls back.

For more details, visit:
https://pydantic.dev/docs/ai/core-concepts/timeouts/
-----------------------------------------------------------------------
"""


# --- 1. An agent whose only tool is far slower than the run's budget ---
slow_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Always use the archive_search tool before answering. Be brief.",
)


@slow_agent.tool_plain
async def archive_search(query: str) -> str:
    """Search the historical archive. Slow."""
    print(f"  [tool] archive_search({query!r}) started, needs 30s")
    await asyncio.sleep(30)
    return "Archive results."


# --- 2. An agent with a tool that reads the run's deadline ---
budget_agent = Agent(
    model=settings.OPENAI_MODEL_NAME,
    instructions="Always call the research tool, then answer in one sentence.",
)


@budget_agent.tool
async def research(ctx: RunContext, topic: str) -> str:
    """Research a topic, as deeply as the time left allows."""
    remaining = ctx.remaining_time()
    if remaining is None:
        print("  [tool] no deadline on this run -> deep research")
        return f"(deep) {topic}: capital of Portugal, rebuilt after the 1755 earthquake."

    print(f"  [tool] deadline={ctx.deadline:%H:%M:%S} UTC, remaining={remaining:.1f}s")
    if remaining < 10:
        print("  [tool] under 10s left -> quick answer from cache")
        return f"(cached) {topic}: capital of Portugal."

    print("  [tool] plenty of time -> deep research (1s)")
    await asyncio.sleep(1)
    return f"(deep) {topic}: capital of Portugal, rebuilt after the 1755 earthquake."


async def main() -> None:

    # ------------------------------------------------------------------
    # Example 1: timeout= on the run raises RunTimedOut
    # ------------------------------------------------------------------
    print("=== Example 1: agent.run(..., timeout=3) ===")

    try:
        await slow_agent.run("Find old records about Lisbon's 1755 earthquake.", timeout=3)
    except RunTimedOut as exc:
        print(f"RunTimedOut: {exc}")
        # One except clause covers both cancellation and asyncio-style timeouts
        print(f"  is RunCancelled: {isinstance(exc, RunCancelled)}")
        print(f"  is TimeoutError: {isinstance(exc, TimeoutError)}")
        # The completed work survives. The tool call that never returned is closed
        # out as 'interrupted' if this history is passed as message_history to resume.
        print("  exc.all_messages():")
        for message in exc.all_messages():
            for part in message.parts:
                print(f"    {type(message).__name__}.{type(part).__name__}")
    print()

    # ------------------------------------------------------------------
    # Example 2: a tool sizes its work with ctx.remaining_time()
    # ------------------------------------------------------------------
    print("=== Example 2: RunContext.deadline / remaining_time() ===")

    for timeout in (None, 30, 8):
        print(f"timeout={timeout}")
        result = await budget_agent.run("Tell me about Lisbon.", timeout=timeout)
        print(f"  Response: {result.output}")
    print()

    # ------------------------------------------------------------------
    # Example 3: request_timeout raises ModelRequestTimeout
    # ------------------------------------------------------------------
    print("=== Example 3: model_settings request_timeout ===")

    # 50ms is far below real API latency, so the request is guaranteed to miss
    # its deadline. max_retries=0 makes the OpenAI SDK try exactly once.
    impatient_agent = Agent(
        model=settings.OPENAI_MODEL_NAME,
        model_settings={"request_timeout": 0.05, "max_retries": 0},
    )

    try:
        await impatient_agent.run("What is the capital of France?")
    except ModelRequestTimeout as exc:
        print(f"ModelRequestTimeout: {exc}")
        print(f"  exc.model_name: {exc.model_name}")
        print(f"  exc.timeout:    {exc.timeout}s")
        # A model error, not a cancellation: FallbackModel falls back on it
        print(f"  is ModelAPIError: {isinstance(exc, ModelAPIError)}")
        print(f"  is RunCancelled:  {isinstance(exc, RunCancelled)}")


if __name__ == "__main__":
    asyncio.run(main())
