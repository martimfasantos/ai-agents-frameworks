import os
from pydantic import BaseModel

from crewai import Agent, Task, Crew
from crewai_tools import CodeInterpreterTool

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI tasks with the following features:
- Async task execution
- Task context (output of one task as input to another)
- Structured output with output_pydantic
- Task-specific tools
- Markdown output formatting

Tasks are the core units of work in CrewAI. This example shows
advanced task patterns including asynchronous execution, task
chaining via context, and structured Pydantic outputs.

For more details, visit:
https://docs.crewai.com/en/concepts/tasks
-------------------------------------------------------
"""


# --- 1. Define a Pydantic model for structured output ---
class CodeSnippet(BaseModel):
    code: str


# --- 2. Define the agent ---
agent = Agent(
    role="Multi Purpose Specialist",
    goal="You are an expert in multiple domains.",
    backstory=(
        "You are a master at understanding various topics and can provide "
        "detailed information and code snippets as needed."
    ),
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 3. Define the tasks ---
generate_code_snippet_task = Task(
    description="Generated simple code for this: {topic}",
    expected_output="A simple code snippet for {topic}",
    async_execution=True,  # This task will run asynchronously
    output_pydantic=CodeSnippet,  # Expecting structured output using Pydantic model
    agent=agent,
)

simplify_code_snippet_task = Task(
    description="Interpret and simplify code snippet using the tools",
    expected_output="A natural language description of the code snippet",
    agent=agent,
    context=[
        generate_code_snippet_task
    ],  # Other tasks whose outputs will be used as context for this task
    markdown=True,  # Enable automatic markdown formatting
    tools=[
        CodeInterpreterTool()
    ],  # Limit the agent to only use this tool for this task
    # (this tool will be added to the agent's tools automatically)
)

# --- 4. Create the crew ---
crew = Crew(
    agents=[agent],
    tasks=[generate_code_snippet_task, simplify_code_snippet_task],
)

# --- 5. Run the crew ---
result = crew.kickoff(
    inputs={"topic": "fibonacci sequence in python"},
)

# Show the results of the tasks
print("Task 1 Result (raw):\n", result.tasks_output[0].raw[:200], "...\n" + "-" * 50)
print(
    "Task 1 Result (pydantic):", type(result.tasks_output[0].pydantic), "\n" + "-" * 50
)
print("Task 2 Result (raw):\n", result.tasks_output[1].raw[:200])
