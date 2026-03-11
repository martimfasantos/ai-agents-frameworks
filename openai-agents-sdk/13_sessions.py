import os
import asyncio

from agents import Agent, Runner, SQLiteSession

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- SQLiteSession for persistent conversation memory
- Multi-turn conversations without manual state management

Sessions automatically store and retrieve conversation history so the
agent remembers previous turns.  Here we use the built-in SQLiteSession
with an in-memory database to demonstrate a three-turn conversation
where the agent correctly recalls earlier context.

For more details, visit:
https://openai.github.io/openai-agents-python/sessions/
-------------------------------------------------------
"""

# --- 1. Create the agent ---
agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
    model=settings.OPENAI_MODEL_NAME,
)

# --- 2. Create an in-memory SQLite session ---
session = SQLiteSession("demo_conversation")


async def main() -> None:
    turns = [
        "What city is the Golden Gate Bridge in?",
        "What state is that city in?",
        "What's the approximate population of that state?",
    ]

    # --- 3. Run a multi-turn conversation ---
    for user_msg in turns:
        print(f"User: {user_msg}")
        result = await Runner.run(agent, user_msg, session=session)
        print(f"Assistant: {result.final_output}\n")

    print("The agent remembered context across all three turns using SQLiteSession.")


if __name__ == "__main__":
    asyncio.run(main())
