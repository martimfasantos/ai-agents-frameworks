from smolagents import ToolCallingAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents ToolCallingAgent:

- ToolCallingAgent vs CodeAgent differences
- JSON-based tool calling (standard function calling)
- Using ToolCallingAgent for structured tool invocation

While CodeAgent writes Python code to call tools, the
ToolCallingAgent uses the standard JSON tool-calling format
that most LLM providers support. This is closer to how
frameworks like LangChain or OpenAI function calling work.

For more details, visit:
https://huggingface.co/docs/smolagents/reference/agents
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define tools ---
@tool
def lookup_capital(country: str) -> str:
    """Look up the capital city of a country.

    Args:
        country: The name of the country.

    Returns:
        The capital city name.
    """
    capitals = {
        "france": "Paris",
        "japan": "Tokyo",
        "brazil": "Brasília",
        "portugal": "Lisbon",
        "germany": "Berlin",
    }
    return capitals.get(country.lower(), f"Capital not found for {country}")


@tool
def get_population(city: str) -> str:
    """Get the approximate population of a city.

    Args:
        city: The name of the city.

    Returns:
        A string with the population information.
    """
    populations = {
        "paris": "2.1 million",
        "tokyo": "13.9 million",
        "lisbon": "0.5 million",
        "berlin": "3.6 million",
        "brasília": "3.0 million",
    }
    return populations.get(city.lower(), f"Population data not available for {city}")


# --- 3. Create a ToolCallingAgent ---
agent = ToolCallingAgent(
    tools=[lookup_capital, get_population],
    model=model,
    max_steps=4,
)

# --- 4. Run queries ---
print("=== ToolCallingAgent Demo ===\n")

print("--- Query: Capital and population ---")
result = agent.run(
    "What is the capital of France and what is its population? Reply in one sentence."
)
print(f"Result: {result}")
