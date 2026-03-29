import os

from autogen import ConversableAgent, LLMConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore AG2 with the following features:
- Defining custom tool functions for agents
- Using the functions= parameter on ConversableAgent
- Automatic tool execution with run() and process()

Tools let agents call Python functions to retrieve data
or perform actions. AG2 handles the tool calling loop
automatically — the agent decides which tool to call,
and AG2 executes it and feeds results back.

For more details, visit:
https://docs.ag2.ai/latest/docs/user-guide/advanced-concepts/tools/basics/
-------------------------------------------------------
"""


# --- 1. Define custom tools ---
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "lisbon": "Sunny, 25°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"No weather data for {city}")


def get_population(city: str) -> str:
    """Get the population of a city."""
    pop_data = {
        "lisbon": "~550,000",
        "london": "~9,000,000",
        "tokyo": "~14,000,000",
    }
    return pop_data.get(city.lower(), f"No population data for {city}")


# --- 2. Create an agent with tools ---
llm_config = LLMConfig({"model": settings.OPENAI_MODEL_NAME})

assistant = ConversableAgent(
    name="assistant",
    system_message=(
        "You are a helpful city information assistant. "
        "Use tools to look up data, then give a brief summary. "
        "Reply in 2-3 sentences max."
    ),
    llm_config=llm_config,
    functions=[get_weather, get_population],
    human_input_mode="NEVER",
)

# --- 3. Run the agent with a query that triggers tools ---
result = assistant.run(
    message="What's the weather and population of Lisbon?",
    max_turns=2,
    user_input=False,
)
result.process()

# --- 4. Print the summary ---
print("\n=== Summary ===")
print(result.summary)
