import os

from crewai import Agent, Task, Crew
from crewai.agent.planning_config import PlanningConfig

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Planning/reasoning before execution via PlanningConfig
- reasoning_effort to trade adaptivity against latency
- Bounding the loop with max_steps and max_attempts

A planning agent drafts a plan before it starts, then optionally
observes each step and replans. PlanningConfig is the current API for
this — the older reasoning=True / max_reasoning_attempts pair is
deprecated and, because it leaves the per-step observation loop
uncapped, can keep replanning long after the work is done. Setting
reasoning_effort and max_steps keeps the run bounded and fast.

For more details, visit:
https://docs.crewai.com/en/concepts/agents#reasoning-agent
-------------------------------------------------------
"""

# --- 1. Create an agent that plans before executing ---
feedback_analyst = Agent(
    role="Customer Support Analyst",
    goal="Classify customer feedback as positive, negative, or neutral",
    backstory="You are skilled at understanding customer sentiment from short feedback messages.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
    # Replaces the deprecated reasoning=True / max_reasoning_attempts=2.
    planning_config=PlanningConfig(
        # "low" plans up front but skips the per-step LLM observation pass.
        # "medium"/"high" observe and replan after every step — far more
        # LLM calls, and on a short task they mostly re-litigate finished work.
        reasoning_effort="low",
        max_attempts=2,  # cap plan-refinement rounds
        max_steps=4,  # cap how long the generated plan can be
    ),
)

# --- 2. Create a task ---
# The messages are inlined. With nothing to classify and no tool to fetch
# anything, the agent plans endlessly against data that never arrives.
FEEDBACK = """
1. "Shipping was quick and the product works exactly as described."
2. "Third time the app has logged me out mid-order. Fed up."
3. "Arrived on the scheduled date."
"""

feedback_task = Task(
    description=(
        "Classify each of these customer feedback messages as positive, "
        f"negative, or neutral:\n{FEEDBACK}"
    ),
    expected_output="Each message with its sentiment, one per line.",
    agent=feedback_analyst,
)

# --- 3. Create a crew and run the task ---
crew = Crew(agents=[feedback_analyst], tasks=[feedback_task])
result = crew.kickoff()

print("\n=== Result ===")
print(result)
