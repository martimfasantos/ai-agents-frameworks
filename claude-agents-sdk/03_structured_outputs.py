import asyncio
import json

from dotenv import load_dotenv

from pydantic import BaseModel

from claude_agent_sdk import query, ClaudeAgentOptions, ResultMessage

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Claude Agent SDK with the following features:
- Structured JSON output via output_format option
- Using JSON Schema directly for output validation
- Using Pydantic model_json_schema() for type-safe schemas
- Accessing structured_output on the ResultMessage

Structured outputs force the agent to return valid JSON matching a
schema you define. This is useful when you need machine-readable
output for downstream processing. The result is available in
message.structured_output as a parsed Python object.

For more details, visit:
https://platform.claude.com/docs/en/agent-sdk/structured-outputs
-------------------------------------------------------
"""

# --------------------------------------------------------------
# Example 1: Raw JSON Schema
# --------------------------------------------------------------
print("=== Example 1: Raw JSON Schema ===")


async def example_raw_schema():
    options = ClaudeAgentOptions(
        output_format={
            "type": "json_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "City name"},
                    "country": {"type": "string", "description": "Country name"},
                    "population_millions": {"type": "number"},
                    "famous_for": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["name", "country", "population_millions", "famous_for"],
            },
        },
    )

    async for message in query(
        prompt="Give me information about Paris.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            print(
                f"Structured output: {json.dumps(message.structured_output, indent=2)}"
            )


asyncio.run(example_raw_schema())

# --------------------------------------------------------------
# Example 2: Pydantic Model Schema
# --------------------------------------------------------------
print("\n=== Example 2: Pydantic Model Schema ===")


class BookReview(BaseModel):
    title: str
    author: str
    rating: float
    summary: str
    themes: list[str]


async def example_pydantic_schema():
    options = ClaudeAgentOptions(
        output_format={
            "type": "json_schema",
            "schema": BookReview.model_json_schema(),
        },
    )

    async for message in query(
        prompt="Write a short review of '1984' by George Orwell.",
        options=options,
    ):
        if isinstance(message, ResultMessage) and message.subtype == "success":
            review = BookReview(**message.structured_output)
            print(f"Title: {review.title}")
            print(f"Author: {review.author}")
            print(f"Rating: {review.rating}/5")
            print(f"Summary: {review.summary}")
            print(f"Themes: {', '.join(review.themes)}")


asyncio.run(example_pydantic_schema())
