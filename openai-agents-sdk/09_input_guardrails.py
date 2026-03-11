import os
import asyncio
from pydantic import BaseModel

from agents import (
    Agent,
    GuardrailFunctionOutput,
    InputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    TResponseInputItem,
    input_guardrail,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Input guardrails
- Tripwire-based input validation
- Using a secondary agent inside a guardrail

Input guardrails run on the initial user input *before* the main agent
produces a response.  Here a cheap guardrail agent checks whether the
user is trying to get math homework help; if so, the tripwire fires
and the main agent never runs, saving cost.

For more details, visit:
https://openai.github.io/openai-agents-python/guardrails/#input-guardrails
-------------------------------------------------------
"""


# --- 1. Define the guardrail's output schema ---
class MathHomeworkCheck(BaseModel):
    is_math_homework: bool
    reasoning: str


# --- 2. Create a cheap guardrail agent ---
guardrail_agent = Agent(
    name="Guardrail check",
    instructions="Check if the user is asking you to do their math homework.",
    output_type=MathHomeworkCheck,
    model=settings.OPENAI_MODEL_NAME,
)


# --- 3. Define the input guardrail function ---
@input_guardrail
async def math_homework_guardrail(
    ctx: RunContextWrapper[None],
    agent: Agent,
    input: str | list[TResponseInputItem],
) -> GuardrailFunctionOutput:
    """Runs a secondary agent to detect math-homework requests."""
    result = await Runner.run(guardrail_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(
        output_info=result.final_output,
        tripwire_triggered=result.final_output.is_math_homework,
    )


# --- 4. Create the main agent with the input guardrail attached ---
agent = Agent(
    name="Customer support agent",
    instructions="You are a customer support agent. You help customers with their questions.",
    input_guardrails=[math_homework_guardrail],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 5. Test with a safe message ---
    result = await Runner.run(agent, "What are your store hours?")
    print(f"Safe message response: {result.final_output}")

    # --- 6. Test with a math homework message (should trip) ---
    try:
        await Runner.run(agent, "Hello, can you help me solve for x: 2x + 3 = 11?")
        print("Guardrail didn't trip - this is unexpected.")
    except InputGuardrailTripwireTriggered:
        print("Math homework guardrail tripped as expected!")


if __name__ == "__main__":
    asyncio.run(main())
