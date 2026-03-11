import os
import asyncio

from agents import Agent, RawResponsesStreamEvent, Runner, TResponseInputItem, trace
from openai.types.responses import ResponseContentPartDoneEvent, ResponseTextDeltaEvent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Routing / Handoffs between agents
- Streaming responses
- Tracing

The orchestrator agent receives a message and hands off to the
appropriate language-specific agent.  Responses are streamed
token-by-token to the console.  A short two-turn conversation
demonstrates agent continuity across turns.

For more details, visit:
https://openai.github.io/openai-agents-python/handoffs/
-------------------------------------------------------
"""

# --- 1. Define the language expert agents ---
french_agent = Agent(
    name="french_agent",
    instructions="You only speak French",
    handoff_description="A french speaking agent",
    model=settings.OPENAI_MODEL_NAME,
)

portuguese_agent = Agent(
    name="portuguese_agent",
    instructions="You only speak Portuguese",
    handoff_description="A portuguese speaking agent",
    model=settings.OPENAI_MODEL_NAME,
)

english_agent = Agent(
    name="english_agent",
    instructions="You only speak English",
    handoff_description="An english speaking agent",
    model=settings.OPENAI_MODEL_NAME,
)

# --- 2. Define the orchestrator agent that routes to the appropriate expert ---
orchestrator_agent = Agent(
    name="orchestrator_agent",
    instructions="Handoff to the appropriate agent based on the language of the request.",
    handoffs=[french_agent, portuguese_agent, english_agent],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # Demo messages instead of interactive input()
    demo_messages = [
        "Olá, como estás?",
        "Obrigado, adeus!",
    ]

    agent = orchestrator_agent
    inputs: list[TResponseInputItem] = [{"content": demo_messages[0], "role": "user"}]

    # --- 3. Run a short multi-turn conversation with streaming ---
    with trace("04_handoffs_and_streaming"):
        for i, msg in enumerate(demo_messages):
            if i > 0:
                inputs.append({"content": msg, "role": "user"})

            print(f"\n>>> User: {msg}")
            result = Runner.run_streamed(agent, input=inputs)
            async for event in result.stream_events():
                if not isinstance(event, RawResponsesStreamEvent):
                    continue
                data = event.data
                if isinstance(data, ResponseTextDeltaEvent):
                    print(data.delta, end="", flush=True)
                elif isinstance(data, ResponseContentPartDoneEvent):
                    print()

            inputs = result.to_input_list()
            agent = result.current_agent

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
