import asyncio

from strands import Agent, tool
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Custom callback handlers for streaming output
- Silent agents with callback_handler=None
- Async streaming with stream_async

Strands agents stream by default. You can customize the streaming behavior
with callback handlers that receive text chunks and tool-use events, or
use async iterators for fine-grained control over the event stream.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/streaming/
-------------------------------------------------------
"""


# --- 1. Define a custom callback handler ---


def custom_callback_handler(**kwargs):
    """A callback handler that tracks tool usage and formats output."""
    if "data" in kwargs:
        print(kwargs["data"], end="", flush=True)
    elif "current_tool_use" in kwargs:
        tool_info = kwargs["current_tool_use"]
        if tool_info.get("name"):
            print(f"\n  [Tool: {tool_info.get('name')}]", flush=True)


@tool
def get_fact(topic: str) -> str:
    """Get an interesting fact about a topic.

    Args:
        topic: The topic to get a fact about
    """
    facts = {
        "python": "Python was named after Monty Python's Flying Circus, not the snake.",
        "space": "A day on Venus is longer than a year on Venus.",
        "ocean": "The ocean produces over 50% of the world's oxygen.",
    }
    return facts.get(topic.lower(), f"Here's a fact: {topic} is fascinating!")


# --- 2. Use the custom callback handler ---
print("=== Custom Callback Handler ===\n")
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
agent_with_callback = Agent(
    model=openai_model,
    tools=[get_fact],
    callback_handler=custom_callback_handler,
)
result = agent_with_callback("Tell me a fact about Python and then about space.")
print(f"\n\n--- Final message length: {len(str(result.message))} chars ---\n")


# --- 3. Silent agent (no streaming output) ---
print("=== Silent Agent (callback_handler=None) ===\n")
silent_agent = Agent(
    model=openai_model,
    callback_handler=None,
)
result = silent_agent("What is 2 + 2? Answer in one word.")
print(f"Silent result: {result.message}\n")


# --- 4. Async streaming with stream_async ---
print("=== Async Streaming ===\n")


async def stream_example():
    agent = Agent(
        model=openai_model,
        tools=[get_fact],
        callback_handler=None,
    )

    async for event in agent.stream_async("Tell me a fun fact about the ocean."):
        if "data" in event:
            print(event["data"], end="", flush=True)
        elif "current_tool_use" in event and event["current_tool_use"].get("name"):
            print(f"\n  [Using tool: {event['current_tool_use']['name']}]")

    print()


asyncio.run(stream_example())
