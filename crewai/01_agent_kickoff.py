import os
from pydantic import BaseModel

from crewai import Agent

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agent kickoff with the following features:
- Direct agent interaction without creating a crew
- Agent.kickoff() for lightweight single-agent execution
- Structured output with response_format (Pydantic model)

The agent kickoff method lets you run an agent directly on a prompt
without wrapping it in a Task and Crew. This is ideal for quick,
single-agent use cases where full crew orchestration is unnecessary.

For more details, visit:
https://docs.crewai.com/en/concepts/agents#direct-agent-interaction-with-kickoff
-------------------------------------------------------
"""


# --- 1. Define a Pydantic model for structured output ---
class CityInfo(BaseModel):
    name: str
    country: str
    population: str
    fun_fact: str


# --- 2. Create an agent ---
city_expert = Agent(
    role="City Expert",
    goal="Provide detailed information about cities around the world",
    backstory="You are a geography expert with extensive knowledge of world cities.",
    llm=settings.OPENAI_MODEL_NAME,
    verbose=True,
)

# --- 3. Kickoff without structured output ---
print("=== Example 1: Simple kickoff ===")
result = city_expert.kickoff("Tell me about Lisbon, Portugal in one short paragraph.")
print("Result:", result)

# --- 4. Kickoff with structured output (response_format) ---
print("\n=== Example 2: Kickoff with structured output ===")
structured_result = city_expert.kickoff(
    "Tell me about Tokyo, Japan.",
    response_format=CityInfo,
)
print("Structured result:", structured_result)
print("Type:", type(structured_result))
