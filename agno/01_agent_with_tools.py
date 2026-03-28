from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Creating custom tools with the @tool decorator
- Providing tool docstrings that the LLM reads as descriptions
- Attaching multiple tools to an agent
- Automatic tool selection and invocation by the LLM

Tools let agents interact with external systems. Agno uses
the @tool decorator to turn plain Python functions into
callable tools. The LLM reads each tool's name, description,
and parameter types to decide when and how to call it.

For more details, visit:
https://docs.agno.com/tools/introduction
-------------------------------------------------------
"""


# --- 1. Define custom tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a given city.

    Args:
        city: The name of the city to look up.

    Returns:
        A string with the weather report.
    """
    weather_data = {
        "lisbon": "Sunny, 25°C, light breeze from the west",
        "london": "Overcast, 14°C, chance of rain",
        "tokyo": "Clear skies, 22°C, low humidity",
    }
    return weather_data.get(
        city.lower(), f"Weather data for '{city}' is not available."
    )


@tool
def get_time(city: str) -> str:
    """Get the current local time for a given city.

    Args:
        city: The name of the city to look up.

    Returns:
        A string with the local time.
    """
    time_data = {
        "lisbon": "14:30 (UTC+1)",
        "london": "13:30 (UTC+0)",
        "tokyo": "22:30 (UTC+9)",
    }
    return time_data.get(city.lower(), f"Time data for '{city}' is not available.")


# --- 2. Create the agent with tools ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    tools=[get_weather, get_time],
    instructions="You are a travel assistant. Use your tools to answer questions about cities.",
    markdown=True,
)

# --- 3. Run the agent ---
run_output = agent.run("What's the weather and time in Lisbon right now?")

# --- 4. Print the result ---
pprint_run_response(run_output)
