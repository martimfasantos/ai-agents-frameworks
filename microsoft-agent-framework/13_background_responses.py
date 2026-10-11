import asyncio
import time

from dotenv import load_dotenv

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Background responses with options={"background": True}
- Polling for completion using previous_response_id
- Async fire-and-forget pattern for long-running tasks

Background responses let agents process requests
asynchronously. The initial call returns immediately with
a continuation token, and you poll until the response
is ready — ideal for long-running agent tasks.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/background-responses?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create a client ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="background-agent",
        instructions="You are a thorough research assistant. Provide detailed answers.",
    )

    # --- 2. Start a background request ---
    print("Starting background request...")
    result = await agent.run(
        "Explain the key differences between REST and GraphQL APIs in 3 bullet points.",
        options={"background": True},
    )

    # --- 3. Check the initial response ---
    if result.text:
        print("Got immediate response (completed instantly):")
        print(result.text)
        return

    print(f"Request is processing in the background.")
    print(f"Response ID: {result.response_id}")
    print(f"Continuation token: {result.continuation_token}\n")

    # --- 4. Poll for completion with the continuation token ---
    max_attempts = 10
    for attempt in range(1, max_attempts + 1):
        time.sleep(2)
        print(f"Polling attempt {attempt}...")

        # The continuation token is what resumes a background response. Polling
        # by previous_response_id sends no messages, which the chat client
        # rejects with "Messages are required for chat completions".
        polled = await agent.run(
            options={"continuation_token": result.continuation_token},
        )

        if polled.text:
            print(f"\nBackground response ready after {attempt} poll(s):")
            print(polled.text)
            return

    print("Timed out waiting for background response.")


if __name__ == "__main__":
    asyncio.run(main())
