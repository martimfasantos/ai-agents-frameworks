import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Custom tool definition
- Streaming responses via Console
- Token usage statistics and metrics

This example shows the different ways to stream responses and collect
metrics using Autogen. The Console utility formats streamed messages and
optionally prints token usage stats at the end.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/tutorial/agents.html
------------------------------------------------------------------------
"""


# --- 1. Define a tool that calculates the sum of a list of integers ---
def add_numbers(values: list[int]) -> int:
    """Calculate the sum of a list of integers."""
    return sum(values)


async def main() -> None:
    # --- 2. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 3. Define the agent ---
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
        tools=[add_numbers],
        system_message="Use tools to solve tasks.",
    )

    # --- 4. Stream the agent response with metrics ---
    # Option 1: read each message from the stream individually.
    # async for message in agent.run_stream(task="What is the result of 2 + 4?"):
    #     print(message)

    # Option 2: use Console to print all messages as they appear.
    await Console(
        agent.run_stream(task="What is the result of 2 + 4?"),
        output_stats=True,  # Enable stats/metrics printing
    )

    # --- 5. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
