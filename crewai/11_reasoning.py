import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Reasoning capabilities for complex tasks
- Limiting reasoning attempts
- Error handling during reasoning

This demonstrates how CrewAI agents can perform sophisticated
reasoning tasks with transparent thought processes.
Configuration options:
- reasoning: Enable or disable reasoning capabilities
- max_reasoning_attempts: Limit the number of reasoning attempts to avoid infinite loops

For more details, visit:
https://docs.crewai.com/en/concepts/agents#reasoning-agent
-------------------------------------------------------
"""

# --- 1. Create an agent with reasoning enabled ---
feedback_analyst = Agent(
    role="Customer Support Analyst",
    goal="Classify customer feedback as positive, negative, or neutral",
    backstory="You are skilled at understanding customer sentiment from short feedback messages.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
    reasoning=True,  # Enable reasoning capabilities
    max_reasoning_attempts=2,  # Simpler task, fewer attempts needed
)

# --- 2. Create a task ---
feedback_task = Task(
    description="Given a list of customer feedback messages, classify each as positive, negative, or neutral.",
    expected_output="A list of feedback messages with their sentiment classification.",
    agent=feedback_analyst,
)

# --- 3. Create a crew and run the task ---
crew = Crew(agents=[feedback_analyst], tasks=[feedback_task])
result = crew.kickoff()
