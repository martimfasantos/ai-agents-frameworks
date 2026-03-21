from pydantic import BaseModel, Field

from strands import Agent
from strands.models.openai import OpenAIModel

from settings import settings

"""
-------------------------------------------------------
In this example, we explore Strands Agents SDK with the following features:
- Structured output with Pydantic models
- Type-safe, validated agent responses
- Multiple output schemas (PersonInfo, MovieReview)

Instead of raw text, you get validated Python objects back from the agent.
Pass a Pydantic model via structured_output_model and access the parsed
result through result.structured_output.

For more details, visit:
https://strandsagents.com/docs/user-guide/concepts/agents/structured-output/
-------------------------------------------------------
"""

# --- 1. Define Pydantic output models ---


class PersonInfo(BaseModel):
    """Model that contains information about a Person"""

    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    occupation: str = Field(description="Occupation of the person")
    skills: list[str] = Field(description="List of key skills", default_factory=list)


class MovieReview(BaseModel):
    """Model for a movie review"""

    title: str = Field(description="Movie title")
    year: int = Field(description="Release year")
    rating: float = Field(description="Rating out of 10", ge=0, le=10)
    genre: str = Field(description="Primary genre")
    summary: str = Field(description="Brief review summary")


# --- 2. Configure model and create the agent ---
openai_model = OpenAIModel(
    client_args={
        "api_key": settings.OPENAI_API_KEY.get_secret_value()
        if settings.OPENAI_API_KEY
        else ""
    },
    model_id=settings.OPENAI_MODEL_NAME,
)
# Default: Agent() uses Amazon Bedrock (requires AWS credentials)
agent = Agent(model=openai_model)

# --- 3. Extract person info (structured) ---
print("=== Person Info Extraction ===")
result = agent(
    "John Smith is a 35 year-old machine learning engineer who specializes in "
    "Python, PyTorch, and distributed systems.",
    structured_output_model=PersonInfo,
)
person: PersonInfo = result.structured_output
print(f"Name: {person.name}")
print(f"Age: {person.age}")
print(f"Occupation: {person.occupation}")
print(f"Skills: {', '.join(person.skills)}")

# --- 4. Extract movie review (structured) ---
print("\n=== Movie Review Extraction ===")
result = agent(
    "I just watched The Matrix from 1999. It's a sci-fi masterpiece! "
    "I'd give it a 9.2 out of 10. The blend of philosophy and action is unmatched.",
    structured_output_model=MovieReview,
)
review: MovieReview = result.structured_output
print(f"Title: {review.title}")
print(f"Year: {review.year}")
print(f"Rating: {review.rating}/10")
print(f"Genre: {review.genre}")
print(f"Summary: {review.summary}")
