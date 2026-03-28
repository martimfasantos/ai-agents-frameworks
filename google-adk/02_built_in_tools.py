import os
import asyncio

from google.adk.agents import LlmAgent
from google.adk.tools import google_search

from settings import settings
from utils import call_agent_async

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- Using built-in tools provided by the ADK framework
- Attaching the google_search tool to an agent for live web search
- Observing tool call and result events during agent execution

Google ADK ships with built-in tools that agents can use without any
custom implementation. The google_search tool allows an agent to query
Google Search and incorporate real-time web results into its responses.
Note: only one built-in tool per agent is supported currently.

For more details, visit:
https://google.github.io/adk-docs/integrations/google-search/
-------------------------------------------------------
"""

# --- 1. Create the agent with the built-in google_search tool ---
search_agent = LlmAgent(
    name="search_agent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You are a search specialist. Use Google Search to find accurate, "
        "up-to-date information. Summarize findings in 2-3 sentences."
    ),
    tools=[google_search],
)

# --- 2. Run the agent ---
query = "What is the current price of Bitcoin in USD?"
print(f"Query: {query}")
asyncio.run(
    call_agent_async(search_agent, query, tool_calls=True, tool_call_results=True)
)
