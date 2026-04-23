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
- Agent as Tool — composing agents via .as_tool()
- Specialist agents delegating to sub-agents
- Hierarchical agent architectures

Agent as Tool lets you wrap an entire agent as a callable
tool for another agent. This enables modular, composable
architectures where a coordinator delegates subtasks to
specialist agents, each with their own tools and expertise.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define specialist tools ---
def get_weather(city: str) -> str:
    """Gets the current weather for a city."""
    data = {
        "lisbon": "Sunny, 26°C",
        "london": "Cloudy, 15°C",
        "new york": "Partly cloudy, 22°C",
    }
    return data.get(city.lower(), f"No weather data for '{city}'.")


def get_restaurant(city: str, cuisine: str) -> str:
    """Finds a restaurant recommendation in a city."""
    restaurants = {
        ("lisbon", "seafood"): "Cervejaria Ramiro — famous for seafood platters",
        ("london", "indian"): "Dishoom — acclaimed Bombay-style cafe",
        ("new york", "italian"): "Carbone — classic Italian-American fine dining",
    }
    return restaurants.get(
        (city.lower(), cuisine.lower()),
        f"Try searching locally for {cuisine} restaurants in {city}.",
    )


async def main() -> None:
    # --- 2. Create specialist agents ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
    )

    weather_agent = client.as_agent(
        name="weather-specialist",
        description="Gets weather information for cities",
        instructions="You provide weather reports. Be concise.",
        tools=[get_weather],
    )

    restaurant_agent = client.as_agent(
        name="restaurant-specialist",
        description="Finds restaurant recommendations in cities",
        instructions="You recommend restaurants. Be concise.",
        tools=[get_restaurant],
    )

    # --- 3. Wrap specialist agents as tools ---
    weather_tool = weather_agent.as_tool()
    restaurant_tool = restaurant_agent.as_tool()

    # --- 4. Create a coordinator agent that uses specialist agents as tools ---
    coordinator = client.as_agent(
        name="travel-coordinator",
        instructions=(
            "You are a travel planning coordinator. "
            "Use your specialist tools to gather weather and restaurant info, "
            "then compile a helpful travel brief. Be concise."
        ),
        tools=[weather_tool, restaurant_tool],
    )

    # --- 5. Run the coordinator — it delegates to specialists ---
    print("Asking coordinator to plan a trip to Lisbon...")
    print("-" * 50)

    result = await coordinator.run(
        "I'm planning a trip to Lisbon. What's the weather like, "
        "and can you recommend a good seafood restaurant?"
    )

    print(result.text)


if __name__ == "__main__":
    asyncio.run(main())
