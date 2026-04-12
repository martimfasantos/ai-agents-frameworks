import asyncio
import os

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import FunctionTool
from google.genai import types

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Token usage tracking via Event.usage_metadata
- Prompt, candidate, and total token counts per event
- Accumulating usage across multiple events in a run

Google ADK events carry a usage_metadata field of type
GenerateContentResponseUsageMetadata with prompt_token_count,
candidates_token_count, and total_token_count. By iterating
over events during a run, you can track token consumption
for cost analysis and optimization.

For more details, visit:
https://google.github.io/adk-docs/events/
-------------------------------------------------------
"""


# --- 1. Define tools ---
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.

    Returns:
        A weather report string.
    """
    weather_data = {
        "london": "Cloudy, 14°C, light rain expected",
        "tokyo": "Sunny, 28°C, clear skies",
        "new york": "Partly cloudy, 22°C",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


# --- 2. Create the ADK agent ---
agent = LlmAgent(
    name="weather_agent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction="You are a concise weather assistant. Answer in 1-2 sentences.",
    tools=[FunctionTool(func=get_weather)],
)

# --- 3. Set up session and runner ---
session_service = InMemorySessionService()
runner = Runner(
    agent=agent,
    app_name="token_usage_demo",
    session_service=session_service,
)


# --- 4. Run and collect usage ---
async def main():
    session = await session_service.create_session(
        app_name="token_usage_demo",
        user_id="demo_user",
    )

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text="What's the weather in London and Tokyo?")],
    )

    print("=== Google ADK Token Usage ===\n")
    print("--- Running agent ---")

    total_prompt = 0
    total_candidates = 0
    total_tokens = 0
    event_count = 0
    final_response = ""

    async for event in runner.run_async(
        user_id="demo_user",
        session_id=session.id,
        new_message=content,
    ):
        event_count += 1

        # Collect final response text
        if event.is_final_response():
            for part in event.content.parts:
                if part.text:
                    final_response += part.text

        # Collect usage_metadata from each event
        if hasattr(event, "usage_metadata") and event.usage_metadata:
            um = event.usage_metadata
            prompt = getattr(um, "prompt_token_count", 0) or 0
            candidates = getattr(um, "candidates_token_count", 0) or 0
            total = getattr(um, "total_token_count", 0) or 0
            thoughts = getattr(um, "thoughts_token_count", 0) or 0

            total_prompt += prompt
            total_candidates += candidates
            total_tokens += total

            print(f"  Event {event_count} (author={event.author}):")
            print(f"    Prompt tokens:     {prompt}")
            print(f"    Candidate tokens:  {candidates}")
            if thoughts:
                print(f"    Thoughts tokens:   {thoughts}")
            print(f"    Total tokens:      {total}")

    print(f"\nResponse: {final_response}\n")

    # --- 5. Show accumulated totals ---
    print("--- Accumulated Token Usage ---")
    print(f"  Total events:      {event_count}")
    print(f"  Prompt tokens:     {total_prompt}")
    print(f"  Candidate tokens:  {total_candidates}")
    print(f"  Total tokens:      {total_tokens}")

    print("\n=== Token Usage Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
