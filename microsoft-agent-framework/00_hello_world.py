import asyncio

from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Creating a basic agent with OpenAIChatClient
- Running a simple single-turn query

This is the simplest possible agent — a single prompt
sent to an LLM through the Agent Framework abstraction.
It demonstrates the core pattern: create a client,
build an agent, and run it.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/overview/?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client and agent ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="hello-world",
        instructions="Be concise, reply with one sentence.",
    )

    # --- 2. Run the agent ---
    result = await agent.run('Where does "hello world" come from?')

    # --- 3. Print the result ---
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
