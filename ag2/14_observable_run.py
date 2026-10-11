import asyncio
import os

from ag2 import Agent, tool
from ag2.config import OpenAIConfig
from ag2.events import ModelMessageChunk, ModelResponse, ToolResultEvent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- agent.run() as an AgentRun async context manager
- run.start() + run.stream.join() to iterate events live
- run.enqueue() to steer a turn that is already running

ask() blocks until a turn finishes; run() lets you watch it unfold.
run.start() drives the turn in the background while run.stream.join()
yields events as they arrive, and run.enqueue() pushes a follow-up
message into the running turn's inbox — consumed at the next model
call, without starting a new turn.

For more details, visit:
https://github.com/ag2ai/ag2/blob/v1.0.1/website/docs/user-guide/agents.mdx
-------------------------------------------------------
"""


@tool
def search(query: str) -> str:
    """Look up a query in the knowledge base."""
    return (
        f"Results for {query!r}: Lisbon is the capital of Portugal, founded "
        "before Rome, and sits on the Tagus estuary."
    )


async def main() -> None:
    # --- 1. streaming=True makes the model emit token chunks ---
    config = OpenAIConfig(model=settings.OPENAI_MODEL_NAME, streaming=True)

    # No tools here, so the first ModelResponse is the final answer.
    narrator = Agent("narrator", prompt="You are a helpful assistant.", config=config)

    researcher = Agent(
        "researcher",
        prompt="You are a helpful assistant. Use the search tool when helpful.",
        config=config,
        tools=[search],
    )

    # --- 2. Watch a turn live, token by token ---
    print("=== Live token stream ===\n")
    async with narrator.run("Tell me about Paris in two sentences.") as run:
        run.start()  # drives the turn in the background; not awaited

        # join() stays open waiting for more events, so break on the turn's
        # terminal ModelResponse rather than expecting the iterator to end.
        with run.stream.join() as events:
            async for event in events:
                if isinstance(event, ModelMessageChunk):
                    print(event.content, end="", flush=True)
                elif isinstance(event, ModelResponse):
                    break

        reply = await run.result()

    print(f"\n\nFinal body length: {len(reply.body)} chars")

    # --- 3. Steer a running turn: wait for the first tool result, then inject ---
    print("\n=== Mid-turn steering with run.enqueue() ===\n")
    async with researcher.run("Search for Lisbon.") as run:
        run.start()

        # Bounded wait: if the model answers without searching, no
        # ToolResultEvent ever arrives.
        async with asyncio.timeout(60):
            with run.stream.where(ToolResultEvent).join(max_events=1) as results:
                async for result in results:
                    print("  [event] tool returned, injecting a follow-up instruction")
                    run.enqueue("Now compress that into exactly one short line.")

        steered = await run.result()

    print(f"\nSteered result: {steered.body}")


if __name__ == "__main__":
    asyncio.run(main())
