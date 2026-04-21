import asyncio

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Structured output using Pydantic models
- Type-safe responses with response_format
- Accessing the parsed .value from the response

Structured output ensures the LLM returns data in a
precise, validated schema — essential for downstream
processing, API integrations, and data pipelines where
free-form text would require fragile parsing.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/structured-output/?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define Pydantic models for structured output ---
class Destination(BaseModel):
    """A single travel destination recommendation."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    reason: str = Field(description="Why this destination is recommended")
    best_month: str = Field(description="Best month to visit")
    budget_level: str = Field(description="Budget level: budget, mid-range, or luxury")


class TravelRecommendations(BaseModel):
    """A list of travel destination recommendations."""

    destinations: list[Destination] = Field(
        description="List of recommended destinations"
    )
    travel_tip: str = Field(description="A general travel tip")


async def main() -> None:
    # --- 2. Create the client and agent ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="travel-advisor",
        instructions="You are a travel advisor. Recommend exactly 3 destinations.",
    )

    # --- 3. Run with structured output ---
    result = await agent.run(
        "Suggest 3 affordable European destinations for a summer trip.",
        options={"response_format": TravelRecommendations},
    )

    # --- 4. Access the typed result ---
    recommendations: TravelRecommendations = result.value  # type: ignore[assignment]
    print(f"Travel tip: {recommendations.travel_tip}\n")

    for i, dest in enumerate(recommendations.destinations, 1):
        print(f"{i}. {dest.city}, {dest.country}")
        print(f"   Reason: {dest.reason}")
        print(f"   Best month: {dest.best_month}")
        print(f"   Budget: {dest.budget_level}")
        print()


if __name__ == "__main__":
    asyncio.run(main())
