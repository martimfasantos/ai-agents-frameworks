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
- Multi-turn conversations using AgentSession
- Maintaining context across multiple exchanges
- Creating and reusing sessions for stateful chat

Sessions allow agents to remember prior messages within
a conversation, enabling natural multi-turn dialogue.
This is the foundation for chatbot-style interactions
where context accumulates over time.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/conversations/session?pivots=programming-language-python
-------------------------------------------------------
"""


async def main() -> None:
    # --- 1. Create the client and agent ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="tutor",
        instructions=(
            "You are a patient programming tutor. "
            "Keep answers concise (2-3 sentences max). "
            "Build on prior context in the conversation."
        ),
    )

    # --- 2. Create a session for multi-turn conversation ---
    session = agent.create_session()

    # --- 3. First turn ---
    print("=== Turn 1 ===")
    result1 = await agent.run("What is a Python list?", session=session)
    print(f"User: What is a Python list?")
    print(f"Agent: {result1.text}\n")

    # --- 4. Second turn — builds on first ---
    print("=== Turn 2 ===")
    result2 = await agent.run("How do I add items to one?", session=session)
    print(f"User: How do I add items to one?")
    print(f"Agent: {result2.text}\n")

    # --- 5. Third turn — references earlier context ---
    print("=== Turn 3 ===")
    result3 = await agent.run("What about removing items?", session=session)
    print(f"User: What about removing items?")
    print(f"Agent: {result3.text}\n")

    # --- 6. Show session info ---
    print(f"Session ID: {session.session_id}")


if __name__ == "__main__":
    asyncio.run(main())
