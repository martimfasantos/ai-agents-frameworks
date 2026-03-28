import asyncio
import os
from agents import Agent, Runner, RunState, function_tool
from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------------------------
In this example, we explore Human-in-the-Loop (HITL) — the ability to
pause agent execution until a person approves or rejects a tool call.

Features demonstrated:
- @function_tool(needs_approval=True) to always require approval
- Conditional approval with an async callback function
- result.interruptions to detect pending approvals
- RunState: serialize/deserialize paused runs (state.to_string / from_string)
- state.approve() and state.reject() to resume execution

This pattern is essential for high-stakes operations where you want
human oversight before the agent takes irreversible actions.
-------------------------------------------------------------------------
"""


# 1. Define a tool that ALWAYS requires approval
@function_tool(needs_approval=True)
async def delete_user_account(user_id: str) -> str:
    """Permanently deletes a user account. This action cannot be undone."""
    return f"Account {user_id} has been permanently deleted."


# 2. Define a tool with CONDITIONAL approval (only for large amounts)
async def needs_large_refund_approval(_ctx, params, _call_id) -> bool:
    """Only require approval for refunds over $100."""
    amount = params.get("amount", 0)
    return amount > 100


@function_tool(needs_approval=needs_large_refund_approval)
async def process_refund(order_id: str, amount: float) -> str:
    """Processes a refund for the given order."""
    return f"Refund of ${amount:.2f} processed for order {order_id}."


# 3. Define a safe tool that never needs approval
@function_tool
async def lookup_order(order_id: str) -> str:
    """Looks up order details."""
    return f"Order {order_id}: 2x Widget Pro, total $250.00, status: delivered."


# 4. Define the agent with all three tools
agent = Agent(
    name="Support Agent",
    instructions=(
        "You are a customer support agent. ALWAYS use the provided tools to "
        "fulfill requests. Never refuse to call a tool — the system handles "
        "approval automatically. When asked to delete an account, call "
        "delete_user_account. When asked to process a refund, call process_refund. "
        "When asked to look up an order, call lookup_order."
    ),
    tools=[delete_user_account, process_refund, lookup_order],
    model=settings.OPENAI_MODEL_NAME,
)


def prompt_user_approval(tool_name: str, arguments: str | None) -> bool:
    """Ask the human operator to approve or reject a tool call."""
    print(f"\n{'=' * 50}")
    print(f"  APPROVAL REQUIRED")
    print(f"  Tool: {tool_name}")
    print(f"  Arguments: {arguments}")
    print(f"{'=' * 50}")
    answer = input("  Approve? [y/N]: ").strip().lower()
    return answer in {"y", "yes"}


async def main():
    print("=== Human-in-the-Loop Example ===\n")

    # 5. Scenario 1: Tool that always needs approval (delete account)
    print("--- Scenario 1: Delete account (always needs approval) ---")
    print(
        "User: Delete the account for user_42 right now using the delete_user_account tool.\n"
    )

    result = await Runner.run(
        agent,
        "Delete the account for user_42 right now using the delete_user_account tool.",
    )

    # 6. Check for interruptions (pending approvals)
    while result.interruptions:
        print(f"Run paused with {len(result.interruptions)} pending approval(s)")

        # 7. Convert to RunState for serialization / approval handling
        state = result.to_state()

        # 8. (Optional) Serialize state — could be stored in a DB for async approval
        serialized = state.to_string()
        print(f"State serialized ({len(serialized)} chars) — could persist to DB")

        # 9. Deserialize (simulates loading from storage later)
        restored_state = await RunState.from_string(agent, serialized)

        # 10. Process each interruption
        for interruption in result.interruptions:
            approved = await asyncio.get_running_loop().run_in_executor(
                None,
                prompt_user_approval,
                interruption.name or "unknown_tool",
                interruption.arguments,
            )
            if approved:
                restored_state.approve(interruption)
            else:
                restored_state.reject(interruption)

        # 11. Resume the run with the approved/rejected state
        result = await Runner.run(agent, restored_state)

    print(f"\nAgent: {result.final_output}\n")

    # 12. Scenario 2: Conditional approval (small refund — no approval needed)
    print("--- Scenario 2: Small refund (no approval needed, amount <= $100) ---")
    print("User: Process a $50 refund for order ORD-123\n")

    result = await Runner.run(agent, "Process a $50 refund for order ORD-123")
    if not result.interruptions:
        print("(No approval needed — amount is under $100)")
    print(f"Agent: {result.final_output}\n")

    # 13. Scenario 3: Conditional approval (large refund — needs approval)
    print("--- Scenario 3: Large refund (needs approval, amount > $100) ---")
    print("User: Process a $250 refund for order ORD-456\n")

    result = await Runner.run(agent, "Process a $250 refund for order ORD-456")

    while result.interruptions:
        state = result.to_state()
        for interruption in result.interruptions:
            approved = await asyncio.get_running_loop().run_in_executor(
                None,
                prompt_user_approval,
                interruption.name or "unknown_tool",
                interruption.arguments,
            )
            if approved:
                state.approve(interruption)
            else:
                state.reject(interruption)
        result = await Runner.run(agent, state)

    print(f"Agent: {result.final_output}\n")
    print("=== Human-in-the-Loop Demo Complete ===")


if __name__ == "__main__":
    asyncio.run(main())
