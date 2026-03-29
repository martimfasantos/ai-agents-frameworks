import os

from crewai import Agent, Task, Crew
from crewai.tasks.conditional_task import ConditionalTask
from crewai.tasks.task_output import TaskOutput

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- ConditionalTask for conditional task execution
- Condition functions that receive previous task output
- Skipping tasks based on runtime conditions

ConditionalTask lets you dynamically decide whether a task should
execute based on the output of a preceding task. This enables
branching logic within a sequential crew pipeline.

For more details, visit:
https://docs.crewai.com/en/learn/conditional-tasks
-------------------------------------------------------
"""

# --- 1. Create agents ---
analyst = Agent(
    role="Data Analyst",
    goal="Analyze data and determine if it needs further processing",
    backstory="You are a data analyst who evaluates datasets.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

reporter = Agent(
    role="Report Writer",
    goal="Write detailed reports based on analysis results",
    backstory="You are a technical writer who creates clear reports.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 2. Define the initial analysis task ---
analysis_task = Task(
    description=(
        "Analyze the following dataset summary and determine if there are "
        "any anomalies that require a detailed report. Dataset: 'Monthly "
        "sales for Q1 2025 show a 40% spike in March compared to January.'"
    ),
    expected_output=(
        "A brief analysis stating whether anomalies were found. "
        "Include the word 'anomaly' if issues were detected."
    ),
    agent=analyst,
)


# --- 3. Define a condition function ---
def should_generate_report(output: TaskOutput) -> bool:
    """Only generate a report if anomalies were found."""
    return "anomaly" in output.raw.lower()


# --- 4. Define a conditional task ---
report_task = ConditionalTask(
    description=(
        "Write a detailed report about the anomalies found in the dataset. "
        "Explain possible causes and recommend next steps."
    ),
    expected_output="A detailed anomaly report with causes and recommendations.",
    agent=reporter,
    condition=should_generate_report,  # Only runs if condition returns True
)

# --- 5. Create and run the crew ---
crew = Crew(
    agents=[analyst, reporter],
    tasks=[analysis_task, report_task],
    verbose=True,
)

result = crew.kickoff()
print("\nFinal result:", result.raw[:300])
