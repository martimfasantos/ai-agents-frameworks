import asyncio
import time

from dotenv import load_dotenv

from agent_framework import (
    Agent,
    AgentContext,
    ChatContext,
    FunctionInvocationContext,
    agent_middleware,
    chat_middleware,
    function_middleware,
)
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Agent middleware (wraps the full agent run)
- Chat middleware (wraps each LLM call)
- Function middleware (wraps each tool invocation)
- Decorator-based middleware with call_next pattern

Middleware provides hooks to intercept, log, modify, or
guard agent execution at three levels: the overall run,
individual LLM calls, and tool invocations. This is the
foundation for observability, cost tracking, and guardrails.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/middleware/?pivots=programming-language-python
-------------------------------------------------------
"""


# --- 1. Define agent-level middleware (wraps the full run) ---
@agent_middleware
async def timing_middleware(context: AgentContext, call_next) -> None:
    """Measures total agent run time."""
    start = time.perf_counter()
    print("[AgentMiddleware] Agent run started")
    await call_next()
    elapsed = time.perf_counter() - start
    print(f"[AgentMiddleware] Agent run completed in {elapsed:.2f}s")


# --- 2. Define chat-level middleware (wraps each LLM call) ---
@chat_middleware
async def logging_chat_middleware(context: ChatContext, call_next) -> None:
    """Logs each LLM call."""
    msg_count = len(context.messages) if context.messages else 0
    print(f"[ChatMiddleware] Sending {msg_count} messages to LLM")
    await call_next()
    print("[ChatMiddleware] LLM response received")


# --- 3. Define function-level middleware (wraps each tool call) ---
@function_middleware
async def tool_logging_middleware(
    context: FunctionInvocationContext, call_next
) -> None:
    """Logs each tool invocation."""
    print(f"[FunctionMiddleware] Calling tool: {context.function.name}")
    await call_next()
    result_str = str(context.result)[:80] if context.result else "None"
    print(f"[FunctionMiddleware] Tool '{context.function.name}' returned: {result_str}")


# --- 4. Define a simple tool to demonstrate function middleware ---
def get_population(city: str) -> str:
    """Gets the approximate population of a city."""
    populations = {
        "lisbon": "Lisbon has approximately 545,000 residents (2.9M metro area).",
        "paris": "Paris has approximately 2.2 million residents (12.4M metro area).",
        "tokyo": "Tokyo has approximately 14 million residents (37.4M metro area).",
    }
    return populations.get(
        city.lower(), f"Population data for '{city}' is not available."
    )


async def main() -> None:
    # --- 5. Create the client with chat + function middleware ---
    client = OpenAIChatClient(
        model=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        middleware=[logging_chat_middleware, tool_logging_middleware],
    )

    # --- 6. Create the agent with agent-level middleware ---
    agent = client.as_agent(
        name="middleware-demo",
        instructions="You are a helpful assistant. Use your tools when asked about cities.",
        tools=[get_population],
        middleware=[timing_middleware],
    )

    # --- 7. Run the agent — middleware intercepts at all levels ---
    print("=" * 50)
    result = await agent.run("What is the population of Lisbon?")
    print("=" * 50)
    print(f"\nFinal answer: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
