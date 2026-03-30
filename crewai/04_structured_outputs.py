import os
from pydantic import BaseModel

from crewai import Agent, Task, Crew, LLM

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore CrewAI's agents with the following features:
- Using Pydantic models for structured responses
- LLM with structured output
- Agent with structured output

This demonstrates how to ensure LLMs and agents return data in specific,
structured formats that can be easily processed by other systems.

For more details, visit:
https://docs.crewai.com/en/concepts/llms#structured-llm-calls
-------------------------------------------------------
"""


# --- 1. Define a Pydantic model for structured output ---
class Dog(BaseModel):
    name: str
    age: int
    breed: str


# --- 2. Create an LLM with structured output ---
llm = LLM(
    model=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
    temperature=0,
    response_format=Dog,  # Using Pydantic model for structured output
)

# --- 3. Test the LLM with structured output ---
response = llm.call(
    "Analyze the following messages and return the name, age, and breed. "
    "Meet Kona! She is 3 years old and is a black german shepherd."
)
print(response)
# Output:
# Dog(name='Kona', age=3, breed='black german shepherd')

# --- 4. Create an agent that uses the LLM with the structured output ---
agent = Agent(
    role="Dog Expert",
    goal="You know everything about dogs.",
    backstory=("You are a master at understanding dogs and their characteristics."),
    llm=llm,
    verbose=True,
)

# --- 5. Define a task for the agent ---
task = Task(
    description="Provide details about the dog named Max.",
    expected_output="Details about the dog named Max",
    output_pydantic=Dog,  # Expecting structured output using Pydantic model
    # output_json=Dog,    # Alternatively, expecting structured output in JSON format
    agent=agent,
)

# --- 6. Create and run the crew ---
crew = Crew(agents=[agent], tasks=[task])
result = crew.kickoff()

# Accessing Properties of the Result in Two Ways
print("Accessing Properties - Two Options")
name = result["name"]  # Dictionary-style indexing
age = result.pydantic.age  # Pydantic attribute access
print('Name: (results["name"])', name)
print("Age: (results.pydantic.age)", age)
