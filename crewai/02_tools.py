import os
from pydantic import BaseModel, Field
from typing import Type

from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool, tool
from crewai_tools import ScrapeWebsiteTool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI agents with the following features:
- Custom tools using both @tool decorator and BaseTool class
- Tool usage for tasks

This example shows how to create agents with custom tools:
1. A custom function tool using @tool decorator to get weather information
2. A custom BaseTool class to get current time in cities
3. A built-in web scraping tool from crewai_tools

For more details, visit:
https://docs.crewai.com/en/concepts/tools#custom-tools
-------------------------------------------------------
"""


# --- 1. Define custom tools ---
# 1.1 Using @tool decorator
@tool("get_weather")
def get_weather(city: str) -> str:  # can be sync or async
    """Retrieves the current weather report for a specified city.

    Args:
        city (str): The name of the city for which to retrieve the weather report.

    Returns:
        str: Weather report or error message.
    """
    if city.lower() == "lisbon":
        return (
            "The weather in Lisbon is sunny with a temperature of 25 degrees Celsius."
        )
    elif city.lower() == "new york":
        return "The weather in New York is cloudy with a temperature of 18 degrees Celsius."
    else:
        return f"Weather information for '{city}' is not available."


# 1.2 Using BaseTool class
class TimeToolInput(BaseModel):
    """Input schema for TimeQueryTool."""

    city: str = Field(..., description="The name of the city to get time for.")


class TimeQueryTool(BaseTool):
    name: str = "time_query_tool"
    description: str = "Returns the current time in a specified city."
    args_schema: Type[BaseModel] = TimeToolInput

    def _run(self, city: str) -> str:  # can be sync or async
        """Get current time for a city."""
        time_zones = {
            "new york": "12:30 PM EST",
            "london": "5:30 PM GMT",
            "tokyo": "2:30 AM JST",
            "lisbon": "4:30 PM WET",
        }

        city_lower = city.lower()
        if city_lower in time_zones:
            return f"The current time in {city} is {time_zones[city_lower]}"
        else:
            return f"Sorry, I don't have timezone information for {city}."


# 1.3 Using built-in tool from crewai_tools
scrape_tool = ScrapeWebsiteTool()

# --- 2. Create specialized agents with tools ---
multi_tool_agent = Agent(
    role="Multi-Tool Specialist",
    goal="Utilize various tools to provide comprehensive information",
    backstory=(
        "You are a multi-tool specialist with access to weather, time, and web scraping tools. "
        "You provide accurate and helpful information to users using the best tool for the job."
    ),
    tools=[
        get_weather,  # @tool decorated function
        TimeQueryTool(),  # BaseTool subclass
        scrape_tool,  # Built-in tool from crewai_tools
    ],
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 3. Create tasks that use the tools ---
weather_task = Task(
    description="Get the current weather in Lisbon",
    expected_output="A detailed weather report for Lisbon",
    agent=multi_tool_agent,
)

time_task = Task(
    description="Find out what time it is in New York",
    expected_output="The current time in New York with timezone",
    agent=multi_tool_agent,
)

web_scrape_task = Task(
    description="Scrape the CrewAI website to find out what services they offer",
    expected_output="A phrase summarizing services offered by CrewAI",
    agent=multi_tool_agent,
)

# --- 3. Create the crew ---
crew = Crew(
    agents=[multi_tool_agent],
    tasks=[weather_task, time_task, web_scrape_task],
    process=Process.sequential,
    verbose=True,
)

# --- 4. Execute the crew ---
result = crew.kickoff()
# no need to print, as verbose=True will show the output in the terminal
