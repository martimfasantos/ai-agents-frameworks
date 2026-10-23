import asyncio
import os

from agents import (
    Agent,
    RunConfig,
    Runner,
    RunResult,
    ToolCallOutputItem,
    ToolErrorFormatterArgs,
    function_tool,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-----------------------------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- RunConfig(tool_error_formatter=...) applied to every tool in a run
- ToolErrorFormatterArgs with kind="tool_exception"
- Hiding exception details (e.g. credentials) from the model

When a function tool raises, the SDK's default handler hides the exception
and only tells the model "Please try again.". A run-wide tool_error_formatter
receives the exception in `args.error` and returns the message the model sees
instead; here it reports the error type but never the message, which holds a
database password. Returning None keeps the SDK default. A tool's own
`failure_error_function` still takes priority over the formatter.

For more details, visit:
https://openai.github.io/openai-agents-python/running_agents/#tool_error_formatter
-----------------------------------------------------------------------------
"""

FAKE_SECRET = "S3cr3t-Pa55"
ERROR_MESSAGE = f"could not connect to postgres://admin:{FAKE_SECRET}@db.internal:5432/orders"


# --- 1. A tool whose exception message contains a database password ---
class DatabaseConnectionError(Exception):
    pass


@function_tool
def get_order_status(order_id: str) -> str:
    """Look up the shipping status of an order."""
    raise DatabaseConnectionError(ERROR_MESSAGE)


# --- 2. A run-wide formatter that hides exception details ---
def format_tool_error(args: ToolErrorFormatterArgs[None]) -> str | None:
    if args.kind == "tool_exception":
        # args.error is the original exception, secret included — keep it server-side.
        error_type = type(args.error).__name__
        print(f"  [formatter] {error_type} from '{args.tool_name}' (message withheld)")
        return f"Tool '{args.tool_name}' failed with {error_type}. Details are hidden."
    # approval_rejected / tool_not_found: keep the SDK default message.
    return None


agent = Agent(
    name="Support Agent",
    instructions=(
        "Use get_order_status to answer. If the tool fails, tell the user in 1-2 "
        "sentences and include the error details you received."
    ),
    model=settings.OPENAI_MODEL_NAME,
    tools=[get_order_status],
)


# --- 3. Show exactly what the model saw as the tool output ---
def print_tool_outputs(result: RunResult) -> None:
    for item in result.new_items:
        if isinstance(item, ToolCallOutputItem):
            output = str(item.output)
            print(f"  Tool output seen by the model:\n    {output}")
            leaked = "YES" if FAKE_SECRET in output else "no"
            print(f"  Secret reached the model: {leaked}")
    print(f"  Final answer: {result.final_output}\n")


async def main() -> None:
    print("=== Tool error formatter ===\n")
    question = "What's the status of order ORD-1042?"
    print(f"User: {question}")
    print(f"The tool always raises: DatabaseConnectionError: {ERROR_MESSAGE}\n")

    # --- 4. Run 1: SDK default — a generic message, no detail at all ---
    print("--- Run 1: no tool_error_formatter (SDK default) ---")
    result = await Runner.run(agent, question)
    print_tool_outputs(result)

    # --- 5. Run 2: the formatter rewrites the failure before the model sees it ---
    print("--- Run 2: RunConfig(tool_error_formatter=format_tool_error) ---")
    result = await Runner.run(
        agent,
        question,
        run_config=RunConfig(tool_error_formatter=format_tool_error),
    )
    print_tool_outputs(result)


if __name__ == "__main__":
    asyncio.run(main())
