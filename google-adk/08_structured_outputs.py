import os
import asyncio

from pydantic import BaseModel, Field

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai.types import Content, Part

from settings import settings
from utils import print_new_section

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- output_schema: enforce structured Pydantic model output from an LlmAgent
- output_key: store the structured result in session state automatically
- Session state inspection: read structured data after agent execution

Structured outputs guarantee that an agent always responds with a
well-typed JSON object matching a Pydantic schema, making it trivial
to post-process agent results in pipelines or APIs without fragile
text parsing.

Note: output_schema and tools cannot be used on the same agent.

For more details, visit:
https://google.github.io/adk-docs/agents/llm-agents/#structured-output
-------------------------------------------------------
"""

APP_NAME = "structured_output_demo"
USER_ID = "user"


# --- 1. Define Pydantic output schemas ---


class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: float = Field(description="Rating from 0.0 to 10.0")
    summary: str = Field(description="One-sentence summary of the review")
    recommended: bool = Field(description="Whether the reviewer recommends the movie")


class WeatherReport(BaseModel):
    city: str = Field(description="Name of the city")
    condition: str = Field(description="Current weather condition, e.g. Sunny, Rainy")
    temperature_celsius: float = Field(description="Temperature in degrees Celsius")
    advice: str = Field(description="One short piece of advice for the day")


# --- 2. Create agents with output_schema ---

movie_agent = LlmAgent(
    name="MovieReviewAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a film critic. When asked to review a movie, respond with a "
        "concise structured review. Always provide a numeric rating from 0 to 10."
    ),
    output_schema=MovieReview,
    output_key="movie_review",  # written to session state under this key
)

weather_agent = LlmAgent(
    name="WeatherReportAgent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a weather reporter. When asked about weather in a city, "
        "generate a plausible-sounding weather report for that city."
    ),
    output_schema=WeatherReport,
    output_key="weather_report",
)


async def run_structured_agent(
    agent: LlmAgent,
    query: str,
    state_key: str,
    session_id: str,
) -> None:
    """Run an agent with output_schema and inspect the structured result."""
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    runner = Runner(
        agent=agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    content = Content(role="user", parts=[Part(text=query)])
    events = runner.run_async(
        user_id=USER_ID, session_id=session_id, new_message=content
    )

    async for _ in events:
        pass  # consume events; structured output is stored in session state

    # Retrieve the structured result from session state
    session = await session_service.get_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=session_id
    )
    if session is None:
        return None
    result = session.state.get(state_key)
    return result


async def main() -> None:
    # --------------------------------------------------------------
    # Example 1: Movie review with structured output
    # --------------------------------------------------------------
    print_new_section("1. Movie Review — Structured Output")

    query = "Please review the movie Blade Runner 2049."
    print(f"  Query: {query}\n")

    review_data = await run_structured_agent(
        agent=movie_agent,
        query=query,
        state_key="movie_review",
        session_id="movie-session-001",
    )

    if review_data:
        review = MovieReview.model_validate(review_data)
        print(f"  Title       : {review.title}")
        print(f"  Rating      : {review.rating}/10")
        print(f"  Recommended : {review.recommended}")
        print(f"  Summary     : {review.summary}")

    # --------------------------------------------------------------
    # Example 2: Weather report with structured output
    # --------------------------------------------------------------
    print_new_section("2. Weather Report — Structured Output")

    query = "What is the weather like in Lisbon today?"
    print(f"  Query: {query}\n")

    weather_data = await run_structured_agent(
        agent=weather_agent,
        query=query,
        state_key="weather_report",
        session_id="weather-session-001",
    )

    if weather_data:
        report = WeatherReport.model_validate(weather_data)
        print(f"  City        : {report.city}")
        print(f"  Condition   : {report.condition}")
        print(f"  Temperature : {report.temperature_celsius}°C")
        print(f"  Advice      : {report.advice}")


if __name__ == "__main__":
    asyncio.run(main())
