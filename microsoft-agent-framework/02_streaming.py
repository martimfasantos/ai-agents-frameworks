import asyncio

from dotenv import load_dotenv

from agent_framework import Agent, ResponseStream
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Streaming responses token-by-token
- Using the ResponseStream async iterator
- Accessing the final aggregated response after streaming

Streaming is essential for real-time UIs and chat
interfaces where users expect to see text appear
progressively rather than waiting for the full response.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/running-agents?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client and agent ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="streaming-agent",
        instructions="You are a helpful assistant. Keep responses concise.",
    )

    # --- 2. Run the agent with streaming enabled ---
    print("Streaming response:")
    print("-" * 40)

    response: ResponseStream = await agent.run(
        "Explain the concept of recursion in 3 sentences.",
        stream=True,
    )

    # --- 3. Iterate over streamed chunks ---
    async for update in response:
        if update.text:
            print(update.text, end="", flush=True)

    print()
    print("-" * 40)

    # --- 4. Access the final aggregated response ---
    final = await response.get_final_response()
    print(f"\nFull response length: {len(final.text)} characters")


if __name__ == "__main__":
    asyncio.run(main())
