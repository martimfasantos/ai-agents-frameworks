from smolagents import (
    CodeAgent,
    OpenAIModel,
    DuckDuckGoSearchTool,
    VisitWebpageTool,
    WikipediaSearchTool,
)

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents built-in tools:

- DuckDuckGoSearchTool for web search
- WikipediaSearchTool for encyclopedia lookups
- VisitWebpageTool for fetching webpage content
- Combining multiple built-in tools in one agent

smolagents ships with several ready-to-use tools that cover
common needs like web search, browsing, and knowledge lookup.
These require no API keys beyond the LLM provider.

For more details, visit:
https://huggingface.co/docs/smolagents/reference/default_tools
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)

# --- 2. Create an agent with built-in tools ---
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        WikipediaSearchTool(),
        VisitWebpageTool(),
    ],
    model=model,
    max_steps=4,
)

# --- 3. Run a query that uses search tools ---
print("=== Built-in Tools Demo ===\n")

print("--- Wikipedia lookup ---")
result = agent.run(
    "Use wikipedia to find out: what year was the Eiffel Tower completed? "
    "Reply with just the year and one sentence of context."
)
print(f"Result: {result}")
