import asyncio

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from settings import settings

"""
------------------------------------------------------------------------
In this example, we explore Autogen agents with the following features:
- Basic agent creation
- Running a simple task
- Streaming output to console

This example shows the simplest possible Autogen agent: an AssistantAgent
that responds to a single user message. We use Console() to display the
streamed output with formatting and token usage statistics.

For more details, visit:
https://microsoft.github.io/autogen/stable/user-guide/agentchat-user-guide/quickstart.html
------------------------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Define the model client ---
    model_client = OpenAIChatCompletionClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    # --- 2. Define the agent ---
    agent = AssistantAgent(
        name="assistant",
        model_client=model_client,
    )

    # --- 3. Run the agent with a user message ---
    await Console(agent.run_stream(task="Say 'Hello World!'"))

    # --- 4. Close the model client ---
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())
