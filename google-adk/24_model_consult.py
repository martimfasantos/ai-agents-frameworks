import os
import asyncio
import textwrap

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import ModelConsultTool
from google.genai import types

from settings import settings

os.environ["GOOGLE_API_KEY"] = settings.GOOGLE_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore Google ADK with the following features:
- ModelConsultTool: an executor agent asks an advisor model for guidance mid-task
- max_uses / session_max_uses to cap advisor calls per turn and per session
- executor_instruction / advisor_instruction to steer when and how it consults

Routine steps (looking up the order, reading the policy) stay on the fast
executor model; only the decision where several policy clauses interact is
escalated. The advisor reads the whole session (instructions, tool calls and
results) and returns guidance, but the executor still makes the decision.
Both roles use the same model here to stay within free-tier quota; in
production the advisor is a stronger model (default: gemini-3.1-pro-preview).

For more details, visit:
https://google.github.io/adk-docs/integrations/model-consult/
-------------------------------------------------------
"""

APP_NAME = "consult_demo"
USER_ID = "user"
SESSION_ID = "s1"


# --- 1. Mock order tools for the executor's routine lookups ---
def lookup_order(order_id: str) -> dict:
    """Returns the details of an order."""
    return {
        "order_id": order_id,
        "item": "Noise-cancelling headphones",
        "category": "electronics",
        "price_usd": 249.00,
        "days_since_delivery": 34,
        "opened": True,
        "reported_issue": "left ear cup crackles at any volume",
        "payment_method": "credit_card",
        "goodwill_credit_already_issued_usd": 25.00,
        "customer_tier": "gold",
    }


def get_refund_policy() -> list[str]:
    """Returns the store's refund policy clauses."""
    return [
        "R1: Refunds are accepted within 30 days of delivery.",
        "R2: Gold-tier customers get a 45-day window instead.",
        "R3: Opened electronics carry a 15% restocking fee.",
        "R4: Defective items are exempt from the restocking fee.",
        "R5: Goodwill credit already issued on the order is deducted from the refund.",
        "R6: Refunds go to the original payment method.",
    ]


# --- 2. The advisor tool, capped at one consult ---
consult_tool = ModelConsultTool(
    model=settings.GOOGLE_MODEL_NAME,
    # thinking_level is a Gemini 3 option; None leaves Gemini 2.5's default alone.
    thinking_level=None,
    max_uses=1,
    session_max_uses=1,
    executor_instruction=(
        "After reading the order and the policy, call model_consult exactly once "
        "with the decision you need reviewed, then answer."
    ),
    advisor_instruction=(
        "You advise a refund agent mid-task. Reply in at most three short bullets: "
        "the clauses that apply, the correct outcome with the amount, and one "
        "pitfall to avoid."
    ),
)

agent = LlmAgent(
    name="refund_agent",
    model=settings.GOOGLE_MODEL_NAME,
    instruction=(
        "You decide refund requests. Call lookup_order and get_refund_policy in the "
        "same step. Give your decision in at most two sentences: approve or deny, "
        "the refund amount in USD, and where it goes."
    ),
    tools=[lookup_order, get_refund_policy, consult_tool],
)


def show(label: str, text: str) -> None:
    print(f"  {label}")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in lines:
        print(textwrap.fill(line, 88, initial_indent="    ", subsequent_indent="      "))


async def main() -> None:
    session_service = InMemorySessionService()
    await session_service.create_session(
        app_name=APP_NAME, user_id=USER_ID, session_id=SESSION_ID
    )
    runner = Runner(agent=agent, app_name=APP_NAME, session_service=session_service)

    # --- 3. Run the request and surface the consult as it happens ---
    request = "Order ORD-502 arrived broken. I want my money back."
    print(f"Customer: {request}\n")
    message = types.Content(role="user", parts=[types.Part(text=request)])
    async for event in runner.run_async(
        user_id=USER_ID, session_id=SESSION_ID, new_message=message
    ):
        for call in event.get_function_calls():
            if call.name == consult_tool.name:
                show("[executor -> advisor] question:", call.args["question"])
            else:
                print(f"  [executor] tool call: {call.name}({dict(call.args)})")
        for response in event.get_function_responses():
            if response.name != consult_tool.name:
                continue
            result = response.response
            budget = result["consults"]
            print(
                f"\n  [advisor] status={result['status']}  "
                f"model={result.get('advisor_model')}  "
                f"consults used={budget['used_this_turn']}/{budget['max_uses']}"
            )
            # "guidance" on success; "message" on limit_reached or error.
            advice = result.get("guidance") or result["message"]
            show("[advisor -> executor] guidance:", advice)
        if event.is_final_response() and event.content and event.content.parts:
            text = "".join(part.text or "" for part in event.content.parts)
            print()
            show("[executor] final decision:", text)


if __name__ == "__main__":
    asyncio.run(main())
