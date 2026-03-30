import asyncio

from dotenv import load_dotenv

from agent_framework import Agent, tool, function_middleware, FunctionInvocationContext
from agent_framework.openai import OpenAIChatClient

from settings import settings

load_dotenv()

"""
-------------------------------------------------------
In this example, we explore Microsoft Agent Framework
with the following features:
- Function middleware as a tool approval gate
- Intercepting sensitive tool calls before execution
- Approving or rejecting tool invocations programmatically

Function middleware lets you intercept every tool call
before it executes. By checking the tool name or arguments,
you can build custom approval logic — approving safe
operations automatically while gating sensitive ones
through a policy engine, human review, or rule system.

For more details, visit:
https://learn.microsoft.com/en-us/agent-framework/agents/tools/tool-approval?pivots=programming-language-python
-------------------------------------------------------
"""

# A set of tools that require approval before execution.
SENSITIVE_TOOLS = {"transfer_funds", "delete_account"}


# --- 1. Define tools ---
@tool(name="check_balance", description="Check account balance")
def check_balance(account_id: str) -> str:
    """Checks the balance for an account (simulated)."""
    balances = {"ACC001": "$1,234.56", "ACC002": "$5,678.90"}
    return balances.get(account_id, f"Account '{account_id}' not found.")


@tool(name="transfer_funds", description="Transfer funds between accounts")
def transfer_funds(from_account: str, to_account: str, amount: float) -> str:
    """Transfers funds between two accounts (simulated)."""
    return f"Transferred ${amount:.2f} from {from_account} to {to_account}. Transaction complete."


# --- 2. Define function middleware that acts as an approval gate ---
@function_middleware
async def approval_middleware(
    context: FunctionInvocationContext,
    call_next,
) -> None:
    """
    Intercepts tool calls. Checks if the tool is in SENSITIVE_TOOLS
    and applies approval logic before allowing execution.
    In a real app, you would prompt a human or check a policy engine.
    """
    tool_name = context.function.name

    if tool_name in SENSITIVE_TOOLS:
        print(f"  [Approval] Tool '{tool_name}' requires approval.")
        print(f"  [Approval] Arguments: {context.arguments}")
        # Simulate approval (in production, prompt the user or check a policy)
        approved = True
        if approved:
            print("  [Approval] APPROVED — proceeding with tool call.\n")
            await call_next()
        else:
            print("  [Approval] REJECTED — blocking tool call.\n")
            context.result = "Tool call was rejected by the approval system."
    else:
        print(f"  [Auto] Tool '{tool_name}' — no approval needed.\n")
        await call_next()


async def main() -> None:
    # --- 3. Create the client with approval middleware ---
    client = OpenAIChatClient(
        model_id=settings.OPENAI_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY.get_secret_value(),
        middleware=[approval_middleware],
    )

    agent = client.as_agent(
        name="banking-assistant",
        instructions=(
            "You are a banking assistant. Use check_balance for balance inquiries "
            "and transfer_funds for transfers. Always call the appropriate tool "
            "directly when the user makes a request."
        ),
        tools=[check_balance, transfer_funds],
    )

    # --- 4. Test with a safe operation (no approval needed) ---
    print("=== Safe Operation: Check Balance ===")
    result = await agent.run("What is the balance for account ACC001?")
    print(f"Result: {result.text}\n")

    # --- 5. Test with a sensitive operation (goes through approval gate) ---
    print("=== Sensitive Operation: Transfer Funds ===")
    result = await agent.run("Transfer $100 from ACC001 to ACC002.")
    print(f"Result: {result.text}")


if __name__ == "__main__":
    asyncio.run(main())
