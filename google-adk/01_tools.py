import os
import asyncio
import datetime
from zoneinfo import ZoneInfo

from google.adk.agents import LlmAgent
from google.adk.tools import FunctionTool

from settings import settings
from utils import call_agent_async

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Defining custom Python functions as agent tools
- Wrapping functions with FunctionTool for structured schemas
- Running an agent that selects and calls the right tool based on user queries

Google ADK agents can call Python functions as tools. The framework
reads the function's docstring and type hints to generate the tool schema
automatically. This example defines two tools — weather and time — and
lets the agent decide which one to invoke based on the user's question.

For more details, visit:
https://google.github.io/adk-docs/tools-custom/function-tools/
-------------------------------------------------------
"""


# --- 1. Define custom function tools ---


def get_weather(city: str) -> dict:
    """Retrieves the current weather report for a specified city.

    Args:
        city: The name of the city for which to retrieve the weather report.

    Returns:
        A dict with status and report or error message.
    """
    if city.lower() == "lisbon":
        return {
            "status": "success",
            "report": "The weather in Lisbon is sunny with a temperature of 25 degrees Celsius.",
        }
    return {
        "status": "error",
        "error_message": f"Weather information for '{city}' is not available.",
    }


def get_current_time(city: str) -> dict:
    """Returns the current time in a specified city.

    Args:
        city: The name of the city for which to retrieve the current time.

    Returns:
        A dict with status and report or error message.
    """
    if city.lower() == "new york":
        tz_identifier = "America/New_York"
    else:
        return {
            "status": "error",
            "error_message": f"Timezone information for '{city}' is not available.",
        }

    tz = ZoneInfo(tz_identifier)
    now = datetime.datetime.now(tz)
    return {
        "status": "success",
        "report": f"The current time in {city} is {now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}",
    }


weather_tool = FunctionTool(func=get_weather)
time_tool = FunctionTool(func=get_current_time)


# --- 2. Create the agent with tools ---
weather_time_agent = LlmAgent(
    name="weather_time_agent",
    model=settings.GOOGLE_MODEL_NAME,
    description="An agent that answers questions about weather and time.",
    instruction=(
        "You are a helpful assistant that provides weather and time information. "
        "Use the available tools to answer questions. Respond in 1-2 sentences."
    ),
    tools=[weather_tool, time_tool],
)


# --- 3. Run the agent with two different queries ---
query = "What is the weather in Lisbon?"
print(f"Query: {query}")
asyncio.run(call_agent_async(weather_time_agent, query, tool_calls=True))

query = "What is the current time in New York?"
print(f"Query: {query}")
asyncio.run(call_agent_async(weather_time_agent, query, tool_calls=True))
