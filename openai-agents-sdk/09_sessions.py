import asyncio
import os
from agents import Agent, Runner, SQLiteSession
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------------------
In this example, we explore Sessions — built-in persistent memory
that automatically maintains conversation history across turns.

Features demonstrated:
- SQLiteSession for file-backed persistent conversation memory
- Multi-turn conversations where the agent remembers prior context
- No manual .to_input_list() — the SDK handles history automatically
- Session operations: get_items, pop_item, clear_session

This is a feature unique to the OpenAI Agents SDK. Other frameworks
require manual history management between conversation turns.
-------------------------------------------------------------------
"""


async def main():
    # 1. Define the agent
    agent = Agent(
        name="Assistant",
        instructions="Reply very concisely. You are a helpful assistant.",
        model=settings.OPENAI_MODEL_NAME,
    )

    # 2. Create a SQLiteSession with a session ID and file-backed persistence
    #    - In-memory: SQLiteSession("session_id")
    #    - File-backed: SQLiteSession("session_id", "path/to/db.sqlite")
    session = SQLiteSession("demo_conversation", "conversation_history.db")

    # 3. Clear any previous session data (for a fresh demo run)
    await session.clear_session()

    print("=== Sessions Example ===")
    print("The agent will remember previous messages automatically.\n")

    # 4. First turn — establish a topic
    print("--- Turn 1 ---")
    print("User: What city is the Golden Gate Bridge in?")
    result = await Runner.run(
        agent,
        "What city is the Golden Gate Bridge in?",
        session=session,
    )
    print(f"Assistant: {result.final_output}\n")

    # 5. Second turn — the agent remembers the previous context
    #    We never passed the first turn's output manually!
    print("--- Turn 2 ---")
    print("User: What state is it in?")
    result = await Runner.run(
        agent,
        "What state is it in?",
        session=session,
    )
    print(f"Assistant: {result.final_output}\n")

    # 6. Third turn — continuing the chain of context
    print("--- Turn 3 ---")
    print("User: What's the population of that state?")
    result = await Runner.run(
        agent,
        "What's the population of that state?",
        session=session,
    )
    print(f"Assistant: {result.final_output}\n")

    # 7. Inspect stored session items
    items = await session.get_items()
    print(f"=== Session now holds {len(items)} items ===")

    # 8. Demonstrate pop_item — remove the last item (useful for corrections)
    last_item = await session.pop_item()
    print(f"Popped last item (assistant response)")
    items_after_pop = await session.get_items()
    print(f"Session now holds {len(items_after_pop)} items after pop\n")

    print("=== Sessions Demo Complete ===")
    print("Notice how the agent remembered context from previous turns!")
    print("Sessions automatically handles conversation history persistence.")


if __name__ == "__main__":
    asyncio.run(main())
