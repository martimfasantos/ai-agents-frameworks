import os

from pydantic import BaseModel, Field

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Structured output with response_format and Pydantic models
- Defining output schemas with field descriptions
- Agent returning validated, typed responses

Structured output forces the agent's final response to conform
to a Pydantic model schema. This is essential when downstream
systems need predictable data formats rather than free-form text.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/structured-output
-------------------------------------------------------
"""


# --- 1. Define output schemas ---
class MovieReview(BaseModel):
    """Structured review of a movie."""

    title: str = Field(description="The movie title")
    year: int = Field(description="The release year")
    genre: str = Field(
        description="Primary genre: action, comedy, drama, sci-fi, horror, or other"
    )
    rating: float = Field(description="Rating from 0.0 to 10.0")
    summary: str = Field(description="One-sentence summary of the movie")


class CityInfo(BaseModel):
    """Structured information about a city."""

    name: str = Field(description="City name")
    country: str = Field(description="Country the city is in")
    population_millions: float = Field(description="Approximate population in millions")
    known_for: list[str] = Field(description="Top 3 things the city is known for")


# --- 2. Define a tool for research ---
@tool
def search_database(query: str) -> str:
    """Search a database for information."""
    # Simulated database responses
    data = {
        "inception": "Inception (2010), directed by Christopher Nolan. Sci-fi thriller about dream infiltration. Widely acclaimed.",
        "lisbon": "Lisbon is the capital of Portugal. Population ~550K (metro ~3M). Known for pastéis de nata, fado music, and tram 28.",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return f"No results found for: {query}"


# --- 3. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --------------------------------------------------------------
# Example 1: Movie review with structured output
# --------------------------------------------------------------
print("=== Example 1: Structured Movie Review ===")

# --- 4. Create agent with structured response format ---
movie_agent = create_agent(
    model=model,
    tools=[search_database],
    response_format=MovieReview,
    system_prompt="You review movies. Use the search tool to find info, then provide a structured review.",
)

result = movie_agent.invoke(
    {"messages": [{"role": "user", "content": "Review the movie Inception"}]}
)

review = result["structured_response"]
print(f"Title: {review.title}")
print(f"Year: {review.year}")
print(f"Genre: {review.genre}")
print(f"Rating: {review.rating}/10")
print(f"Summary: {review.summary}\n")

# --------------------------------------------------------------
# Example 2: City info with structured output
# --------------------------------------------------------------
print("=== Example 2: Structured City Info ===")

# --- 5. Create agent with different schema ---
city_agent = create_agent(
    model=model,
    tools=[search_database],
    response_format=CityInfo,
    system_prompt="You provide city information. Use the search tool, then return structured data.",
)

result = city_agent.invoke(
    {"messages": [{"role": "user", "content": "Tell me about Lisbon"}]}
)

city = result["structured_response"]
print(f"City: {city.name}, {city.country}")
print(f"Population: {city.population_millions}M")
print(f"Known for: {', '.join(city.known_for)}")
