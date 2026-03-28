import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.memory import InMemoryMemoryService
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import load_memory
from google.genai.types import Content, Part

from settings import settings
from utils import print_new_section

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- InMemoryMemoryService: stores completed sessions for later recall
- load_memory tool: lets agents search past interactions at runtime
- Cross-session memory: agents recall facts from prior conversations

ADK's memory system allows agents to persist information across separate
sessions and retrieve it on demand. This is essential for building
assistants that remember users across conversations without relying on
a single long-running session.

For more details, visit:
https://google.github.io/adk-docs/sessions/memory/
-------------------------------------------------------
"""

APP_NAME = "memory_demo"
USER_ID = "user"

# --- 1. Create the memory service (shared across sessions) ---
memory_service = InMemoryMemoryService()

# --- 2. Create the memory-enabled agent ---
# The load_memory tool lets the agent search past sessions
memory_agent = LlmAgent(
    name="MemoryAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a helpful assistant with access to past conversation memory. "
        "ALWAYS call the load_memory tool before answering any question about the user "
        "or prior conversations. Never answer from memory without calling the tool first. "
        "Respond in 1-3 sentences."
    ),
    tools=[load_memory],
)


async def run_session(
    session_id: str,
    query: str,
    session_service: InMemorySessionService,
    label: str,
) -> None:
    """Run a single agent session and return the completed session."""
    session = await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )

    runner = Runner(
        agent=memory_agent,
        app_name=APP_NAME,
        session_service=session_service,
        memory_service=memory_service,
    )

    content = Content(role="user", parts=[Part(text=query)])
    events = runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    )

    print(f"\n[{label}]")
    print(f"  Query: {query}")
    async for event in events:
        if event.is_final_response():
            if event.content and event.content.parts:
                print(f"  Response: {event.content.parts[0].text}")

    # Ingest the completed session into memory so future sessions can recall it
    completed_session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if completed_session:
        await memory_service.add_session_to_memory(completed_session)
        print(f"  (Session ingested into memory)")


async def main() -> None:
    session_service = InMemorySessionService()

    # --------------------------------------------------------------
    # Example 1: Establish facts in a first session
    # --------------------------------------------------------------
    print_new_section("1. First Session — Establishing Facts")
    await run_session(
        session_id="session-001",
        query=(
            "My name is Alice and I am a software engineer specialising in "
            "distributed systems. Please acknowledge that you have noted this."
        ),
        session_service=session_service,
        label="SESSION 1",
    )

    # --------------------------------------------------------------
    # Example 2: New session recalls facts from the first
    # --------------------------------------------------------------
    print_new_section("2. Second Session — Recalling from Memory")
    await run_session(
        session_id="session-002",
        query="What do you know about me from our previous conversation?",
        session_service=session_service,
        label="SESSION 2",
    )

    # --------------------------------------------------------------
    # Example 3: Third session retrieves specific detail
    # --------------------------------------------------------------
    print_new_section("3. Third Session — Specific Memory Retrieval")
    await run_session(
        session_id="session-003",
        query="Use load_memory to search for what I told you about my job. What is my area of specialisation?",
        session_service=session_service,
        label="SESSION 3",
    )


if __name__ == "__main__":
    asyncio.run(main())
