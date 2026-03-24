import os
import asyncio
import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

# Suppress the SDK's "non-text parts in response" informational warning
logging.getLogger("google_genai.types").setLevel(logging.ERROR)

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Google Search Grounding: connecting agents to live web search results
- Grounding metadata: accessing citation sources returned by the model
- GroundingChunk: structured web source data (title + URI) per cited fact
- Grounding supports: mapping text segments back to their source chunks

Grounding enriches agent responses with real-time web data and provides
transparent attribution. Each grounded response includes metadata that
identifies which web pages were consulted and which sentences are supported
by which sources. This example runs two queries and prints both the agent's
answer and the full set of grounding citations.

For more details, visit:
https://google.github.io/adk-docs/grounding/google_search_grounding/
-------------------------------------------------------
"""

APP_NAME = "grounding_demo"
USER_ID = "user"
SESSION_ID = "session_1"

# --- 1. Create agent with Google Search grounding ---

search_agent = LlmAgent(
    name="GroundedSearchAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a research assistant. Use Google Search to find accurate, "
        "up-to-date information. Summarize findings in 2-3 sentences."
    ),
    tools=[google_search],
)


# --- 2. Run queries and extract grounding metadata ---


async def run_demo() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(
        agent=search_agent, app_name=APP_NAME, session_service=session_service
    )

    async def ask_and_show_grounding(query: str) -> None:
        print(f"\n  Query: {query}")
        print("-" * 65)
        message = types.Content(role="user", parts=[types.Part(text=query)])

        answer = ""
        grounding_sources: list[tuple[str, str]] = []

        async for event in runner.run_async(
            user_id=USER_ID, session_id=SESSION_ID, new_message=message
        ):
            # Collect the final text answer
            if event.is_final_response() and event.content and event.content.parts:
                answer = event.content.parts[0].text or ""

            # Collect grounding citations from any event that carries them
            if event.grounding_metadata and event.grounding_metadata.grounding_chunks:
                for chunk in event.grounding_metadata.grounding_chunks:
                    if chunk.web and chunk.web.title and chunk.web.uri:
                        source = (chunk.web.title, chunk.web.uri)
                        if source not in grounding_sources:
                            grounding_sources.append(source)

        print(f"  Answer: {answer.strip()}")
        if grounding_sources:
            print(f"\n  Grounding sources ({len(grounding_sources)} cited):")
            for i, (title, uri) in enumerate(grounding_sources, 1):
                # Truncate long URIs for readability
                short_uri = uri[:80] + "..." if len(uri) > 80 else uri
                print(f"    [{i}] {title}")
                print(f"         {short_uri}")
        else:
            print("  (No grounding metadata returned for this response)")

    # --- 3. Demonstrate grounding with two real-time queries ---

    print("\n" + "=" * 65)
    print("  Google Search Grounding Demo")
    print("=" * 65)

    await ask_and_show_grounding("What is the current price of Bitcoin in USD?")
    await ask_and_show_grounding(
        "Who won the most recent FIFA World Cup and what was the final score?"
    )

    print()


if __name__ == "__main__":
    asyncio.run(run_demo())
