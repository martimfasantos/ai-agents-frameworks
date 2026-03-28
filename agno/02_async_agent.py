import asyncio

from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools import tool
from agno.utils.pprint import pprint_run_response

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Agno with the following features:
- Running agents asynchronously with agent.arun()
- Defining async tool functions
- Launching multiple agent calls concurrently with asyncio.gather

Async execution is essential for I/O-bound workloads. Agno
natively supports async agents via arun(), letting you
run multiple queries in parallel without blocking. This
example shows both an async tool and concurrent agent runs.

For more details, visit:
https://docs.agno.com/agents/run
-------------------------------------------------------
"""


# --- 1. Define an async tool ---
@tool
async def search_database(query: str) -> str:
    """Search a mock database for records matching a query.

    Args:
        query: The search term to look up.

    Returns:
        A string with the matching records.
    """
    # Simulate async I/O (e.g., a database call)
    await asyncio.sleep(0.1)
    mock_results = {
        "python": "Python 3.13 released — performance improvements and new features.",
        "agno": "Agno v2.5 — step-based workflows, teams, and improved tool support.",
        "ai": "AI adoption in enterprise software grew 40% year-over-year.",
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"No results found for '{query}'."


# --- 2. Create the agent ---
agent = Agent(
    model=OpenAIChat(id=settings.OPENAI_MODEL_NAME),
    tools=[search_database],
    instructions="You are a research assistant. Use the search tool to find information.",
    markdown=True,
)


# --- 3. Run multiple queries concurrently ---
async def main() -> None:
    results = await asyncio.gather(
        agent.arun("What's new with Python?"),
        agent.arun("Tell me about Agno framework updates."),
        agent.arun("What's happening in AI?"),
    )
    for i, run_output in enumerate(results, 1):
        print(f"\n=== Query {i} ===")
        pprint_run_response(run_output)


if __name__ == "__main__":
    asyncio.run(main())
