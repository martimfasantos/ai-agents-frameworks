import asyncio
from typing import Annotated

from dotenv import load_dotenv
from pydantic import Field

from agent_framework import Agent, tool
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Defining custom function tools with plain functions
- Using the @tool decorator for explicit metadata
- Using Annotated + Field for parameter descriptions
- Passing tools to an agent

Function tools let agents call Python functions during
execution. The framework automatically generates JSON
schemas from type hints and docstrings, which the LLM
uses to decide when and how to invoke each tool.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/tools/function-tools?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define tools ---
def get_weather(
    city: Annotated[str, Field(description="The city to get weather for")],
) -> str:
    """Retrieves the current weather report for a specified city."""
    weather_data = {
        "lisbon": "Sunny, 25°C, light breeze from the west",
        "london": "Overcast, 14°C, chance of rain",
        "tokyo": "Clear skies, 28°C, humid",
    }
    return weather_data.get(
        city.lower(), f"Weather data for '{city}' is not available."
    )


@tool(name="get_time", description="Get the current time in a city")
def get_time(
    city: Annotated[str, Field(description="The city to get time for")],
) -> str:
    """Returns the current local time for a city (simulated)."""
    time_data = {
        "lisbon": "14:30 WET",
        "london": "13:30 GMT",
        "tokyo": "22:30 JST",
    }
    return time_data.get(city.lower(), f"Time data for '{city}' is not available.")


async def main() -> None:
    # --- 2. Create the client and agent with tools ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    agent = client.as_agent(
        name="weather-assistant",
        instructions="You are a helpful travel assistant. Use your tools to answer questions about cities.",
        tools=[get_weather, get_time],
    )

    # --- 3. Run the agent with a query that triggers tool use ---
    result = await agent.run("What's the weather and time in Lisbon right now?")

    # --- 4. Print the result ---
    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
