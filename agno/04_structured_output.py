from typing import List

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Structured output using output_schema with Pydantic models
- Type-safe, validated responses from the LLM
- Nested Pydantic models for complex data structures

Structured output forces the LLM to return data conforming to
a Pydantic schema instead of free-form text. This is ideal for
building pipelines where downstream code needs predictable
JSON. Agno validates the response against the schema
automatically.

For more details, visit:
https://docs.agno.com/agents/output-schema
-------------------------------------------------------
"""


# --- 1. Define the output schema ---
class Ingredient(BaseModel):
    name: str = Field(description="Name of the ingredient")
    quantity: str = Field(description="Amount needed, e.g. '2 cups'")


class Recipe(BaseModel):
    title: str = Field(description="Name of the recipe")
    description: str = Field(description="Brief description of the dish")
    ingredients: List[Ingredient] = Field(description="List of ingredients")
    steps: List[str] = Field(description="Step-by-step cooking instructions")
    prep_time_minutes: int = Field(description="Estimated preparation time in minutes")


# --- 2. Create the agent with output_schema ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    instructions="You are a professional chef. Provide detailed, accurate recipes.",
    output_schema=Recipe,
)

# --- 3. Run the agent ---
run_output = agent.run("Give me a recipe for a classic Portuguese pastéis de nata.")

# --- 4. Print the structured result ---
pprint_run_response(run_output)

# Access the parsed Pydantic object directly
recipe: Recipe = run_output.content
print(f"\nRecipe: {recipe.title}")
print(f"Prep time: {recipe.prep_time_minutes} minutes")
print(f"Number of ingredients: {len(recipe.ingredients)}")
print(f"Number of steps: {len(recipe.steps)}")
