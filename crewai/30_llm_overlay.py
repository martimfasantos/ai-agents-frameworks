import os

from crewai import Agent, Crew, Task
from crewai.llm_overlay import llm_overlay

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI with the following features:
- llm_overlay, a context manager that swaps models for one run (1.15.22)
- Role keys ({"Editor": model}) that move a single agent to another model
- Model keys ("model:gpt-4o-mini" or "model:*") that remap a model (1.15.24)
- Reading agent.llm.model and per-agent token usage to see what actually ran

Agents declare their model in code, so trying a stronger model for one role
usually means editing that code. llm_overlay applies a mapping for the
duration of a with block instead: agents built inside the block whose role
(or declared model) is a key run on the mapped model, and everything else
keeps its declared llm. The overlay is read when an agent is built, so the
crew factory has to be called inside the block.

For more details, visit:
https://docs.crewai.com/en/concepts/llms
-------------------------------------------------------
"""

# --- 1. A crew factory that declares every agent on the default model ---
RELEASE_NOTES = (
    "v2.3: checkout is 40% faster, dark mode added, "
    "CSV export fixed for files over 10 MB."
)


def build_crew() -> Crew:
    analyst = Agent(
        role="Analyst",
        goal="Spot the most user-visible change in release notes",
        backstory="You read release notes and answer in one short sentence.",
        llm=settings.OPENAI_MODEL_NAME,
    )
    editor = Agent(
        role="Editor",
        goal="Turn an analyst's pick into a customer announcement",
        backstory="You write one upbeat sentence, no hashtags.",
        llm=settings.OPENAI_MODEL_NAME,
    )
    pick = Task(
        description=f"Pick the most user-visible change in: {RELEASE_NOTES}",
        expected_output="One sentence naming the change.",
        agent=analyst,
    )
    announce = Task(
        description="Write a one-sentence customer announcement for that change.",
        expected_output="One sentence.",
        agent=editor,
    )
    return Crew(agents=[analyst, editor], tasks=[pick, announce])


def report(crew: Crew) -> None:
    for agent in crew.agents:
        usage = agent.llm.get_token_usage_summary()
        print(
            f"  {agent.role:<8} ran on {agent.llm.model:<12} "
            f"requests={usage.successful_requests} tokens={usage.total_tokens}"
        )


default_model = settings.OPENAI_MODEL_NAME
strong_model = settings.OPENAI_STRONG_MODEL_NAME

# --- 2. Baseline: no overlay, both agents on their declared model ---
print("=== 1. No overlay ===")
crew = build_crew()
result = crew.kickoff()
report(crew)
print(f"  Answer: {result.raw}\n")

# --- 3. Role key: only the Editor moves to the stronger model ---
print(f'=== 2. llm_overlay({{"Editor": "{strong_model}"}}) ===')
with llm_overlay({"Editor": strong_model}):
    crew = build_crew()
    result = crew.kickoff()
report(crew)
print(f"  Answer: {result.raw}\n")

# --- 4. Model key: every agent declared on the default model is remapped ---
# Agents read the overlay only when built, so a crew built before the block
# keeps its declared model even when kicked off inside it.
print(f'=== 3. llm_overlay({{"model:{default_model}": "{strong_model}"}}) ===')
built_before = build_crew()
with llm_overlay({f"model:{default_model}": strong_model}):
    crew = build_crew()
    result = crew.kickoff()
    built_before.kickoff()
print("Crew built inside the block:")
report(crew)
print(f"  Answer: {result.raw}")
print("Crew built before the block, kicked off inside it:")
report(built_before)
