import os

from crewai import Agent, Task, Crew
from crewai import TaskOutput
from crewai.tools import tool

from settings import settings
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Task callbacks for post-processing
- Tool usage within tasks
- Integration of tools and callback mechanisms

This demonstrates how CrewAI agents can leverage callbacks mechanism
to perform actions after task completion, such as notifying users
or triggering subsequent workflows.

For more details, visit:
https://docs.crewai.com/en/concepts/tasks#callback-mechanism
-------------------------------------------------------
"""

# --- 1. Define a very simplified tool ---
@tool("get_weather")
def get_weather_lisbon(city: str) -> str:
    """Simple tool to get weather information for Lisbon""" 
    if city.lower() == "lisbon":
        return "The weather in Lisbon is sunny with a temperature of 25 degrees Celsius."
    else:
        return f"Weather information for '{city}' is not available."

# --- 2. Create a callback function to handle task completion ---
def callback_function(output: TaskOutput):
    # Do something after the task is completed
    # - Example: Send an email to the manager
    print(f"""
    ----------------------
    Callback:
        Task completed!
        Task: {output.description}
        Output: {output.raw}
    ----------------------
    """)
    
# --- 3. Create a simple agent ---
research_agent = Agent(
    role="Tool Agent",
    goal="Use tools to accomplish tasks efficiently. Avoid repetition.",
    backstory="You are a simple agent that uses tools to get information.",
    tools=[get_weather_lisbon],
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 4. Define a task that uses the tool and the callback ---
weather_task = Task(
    description="Get the current weather for a {city}",
    expected_output="The current weather report",
    agent=research_agent,
    tools=[get_weather_lisbon],
    callback=callback_function
)

# --- 5. Create the crew ---
crew = Crew(
    agents=[research_agent],
    tasks=[weather_task],
    verbose=True
)

# --- 6. Run the crew ---
result = crew.kickoff(
    inputs={"city": "Lisbon"}
)
