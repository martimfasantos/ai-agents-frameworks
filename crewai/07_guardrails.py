import os

from crewai import Agent, Task, Crew

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- Function-based task guardrails for output validation
- Automatic retry when guardrail validation fails
- Composing multiple validation checks in a single guardrail

Task guardrails let you validate and constrain agent outputs before
they are accepted. If a guardrail fails, the task is retried with
feedback. This ensures output quality without manual intervention.

For more details, visit:
https://docs.crewai.com/en/concepts/tasks#task-guardrails
-------------------------------------------------------
"""


# --- 1. Define guardrail functions ---
def must_contain_bullet_points(result) -> tuple[bool, str]:
    """Guardrail that checks if the output contains bullet points."""
    if "-" not in result.raw and "*" not in result.raw:
        return (
            False,
            "Output must contain bullet points (using - or *). Please reformat.",
        )
    return (True, result.raw)


def concise_bullet_guardrail(result) -> tuple[bool, str]:
    """Composite guardrail: must be under 500 chars AND contain bullet points."""
    if len(result.raw) > 500:
        return (False, "Output exceeds 500 characters. Please make it more concise.")
    if "-" not in result.raw and "*" not in result.raw:
        return (
            False,
            "Output must contain bullet points (using - or *). Please reformat.",
        )
    return (True, result.raw)


# --- 2. Create an agent ---
writer = Agent(
    role="Concise Writer",
    goal="Write clear, concise summaries using bullet points",
    backstory="You are a writer who specializes in brief, structured summaries.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 3. Create a task with a single guardrail ---
print("=== Example 1: Single guardrail ===")
task_single = Task(
    description="Summarize the benefits of remote work in a short bullet-point list.",
    expected_output="A concise bullet-point summary under 500 characters.",
    agent=writer,
    guardrail=must_contain_bullet_points,  # Single guardrail function
)

crew_single = Crew(agents=[writer], tasks=[task_single], verbose=True)
result1 = crew_single.kickoff()
print("Result 1:", result1.raw[:300])

# --- 4. Create a task with a composite guardrail ---
print("\n=== Example 2: Composite guardrail (length + format) ===")
task_composite = Task(
    description="List the top 3 programming languages for AI development with one-line descriptions.",
    expected_output="A short bullet-point list under 500 characters.",
    agent=writer,
    guardrail=concise_bullet_guardrail,  # Composite guardrail function
)

crew_composite = Crew(agents=[writer], tasks=[task_composite], verbose=True)
result2 = crew_composite.kickoff()
print("Result 2:", result2.raw[:300])
