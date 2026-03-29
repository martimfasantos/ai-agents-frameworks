import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore a simple Hello World agent
with the following features:
- Basic agent creation with role, goal, and backstory
- Task definition with description and expected output
- Crew assembly and execution

This is the simplest possible CrewAI example: one agent,
one task, one crew. It demonstrates the core building
blocks of every CrewAI application.

For more details, visit:
https://docs.crewai.com/en/concepts/crews
-------------------------------------------------------
"""

# --- 1. Create an agent ---
agent = Agent(
    role="Greeter",
    goal="Say hello to the world",
    backstory="A friendly AI assistant",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 2. Create a task ---
task = Task(
    description="Say hello to the world",
    expected_output="A greeting message",
    agent=agent,
)

# --- 3. Create crew ---
crew = Crew(agents=[agent], tasks=[task])

# --- 4. Run the crew ---
result = crew.kickoff()
# no need to print, as verbose=True will show the output in the terminal
