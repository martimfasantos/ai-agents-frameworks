import asyncio

from dotenv import load_dotenv

from pydantic_ai import Agent
from pydantic_ai.builtin_tools import WebSearchTool, CodeExecutionTool
from pydantic_ai.models.openai import OpenAIResponsesModel

from settings import settings

load_dotenv()

"""
-----------------------------------------------------------------------
In this example, we explore Pydantic AI with the following features:
- Built-in WebSearchTool for grounding responses in live web results
- Built-in CodeExecutionTool for running Python code in a sandbox
- Combining multiple built-in tools in a single agent
- Using OpenAIResponsesModel (required for built-in tools)

Built-in tools are pre-configured capabilities provided by the model
provider (e.g. OpenAI). Unlike custom tools, they run server-side
and require no local implementation. WebSearchTool grounds answers in
real search results, while CodeExecutionTool lets the model write and
execute Python code to solve computational problems.

NOTE: Built-in tools require OpenAIResponsesModel (Responses API),
not the default ChatModel (Chat Completions API).

For more details, visit:
https://ai.pydantic.dev/builtin-tools/
-----------------------------------------------------------------------
"""

# Built-in tools require the Responses API model
responses_model = OpenAIResponsesModel(settings.OPENAI_MODEL_NAME)


# --------------------------------------------------------------
# Example 1: Web Search Tool
# --------------------------------------------------------------
print("=== Example 1: Web Search Tool ===")

# --- 1. Create agent with web search ---
search_agent = Agent(
    model=responses_model,
    instructions="Answer questions using web search. Be concise (1-2 sentences).",
    builtin_tools=[
        WebSearchTool(search_context_size="low"),
    ],
)

# --- 2. Run a query that benefits from live data ---
result1 = search_agent.run_sync("What is the current version of Pydantic AI?")
print(f"Response: {result1.output}")
print()


# --------------------------------------------------------------
# Example 2: Code Execution Tool
# --------------------------------------------------------------
print("=== Example 2: Code Execution Tool ===")

# --- 1. Create agent with code execution ---
code_agent = Agent(
    model=responses_model,
    instructions="Solve problems by writing and executing Python code. Show the result.",
    builtin_tools=[
        CodeExecutionTool(),
    ],
)

# --- 2. Ask a computational question ---
result2 = code_agent.run_sync(
    "Calculate the first 10 Fibonacci numbers and return them as a list."
)
print(f"Response: {result2.output}")
print()


# --------------------------------------------------------------
# Example 3: Combined Built-in Tools
# --------------------------------------------------------------
print("=== Example 3: Combined Web Search + Code Execution ===")


# --- 1. Create agent with both tools ---
async def run_combined():
    combined_agent = Agent(
        model=responses_model,
        instructions=(
            "You can search the web for information and execute Python code. "
            "Be concise."
        ),
        builtin_tools=[
            WebSearchTool(search_context_size="low"),
            CodeExecutionTool(),
        ],
    )

    # Ask something that benefits from code execution
    result3 = await combined_agent.run(
        "What is 2^100? Use code execution to compute the exact value."
    )
    print(f"Response: {result3.output}")


asyncio.run(run_combined())
