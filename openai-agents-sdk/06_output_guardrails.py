import os
import asyncio
import json
from pydantic import BaseModel, Field

from agents import (
    Agent,
    GuardrailFunctionOutput,
    OutputGuardrailTripwireTriggered,
    RunContextWrapper,
    Runner,
    output_guardrail,
)

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore OpenAI's Agents SDK with the following features:
- Output guardrails
- Tripwire-based safety checks

Output guardrails are checks that run on the final output of an agent.
They can verify the output doesn't contain sensitive data, meets quality
thresholds, or satisfies other constraints.  Here we check whether the
agent's response leaks a phone number.

For more details, visit:
https://openai.github.io/openai-agents-python/guardrails/
-------------------------------------------------------
"""


# --- 1. Define the output model for the agent's response ---
class MessageOutput(BaseModel):
    reasoning: str = Field(
        description="Thoughts on how to respond to the user's message"
    )
    response: str = Field(description="The response to the user's message")
    user_name: str | None = Field(
        description="The name of the user who sent the message, if known"
    )


# --- 2. Define the output guardrail function ---
@output_guardrail
async def sensitive_data_check(
    context: RunContextWrapper, agent: Agent, output: MessageOutput
) -> GuardrailFunctionOutput:
    """Checks if the agent's response or reasoning contains a phone number."""
    phone_number_in_response = "650" in output.response
    phone_number_in_reasoning = "650" in output.reasoning

    return GuardrailFunctionOutput(
        output_info={
            "phone_number_in_response": phone_number_in_response,
            "phone_number_in_reasoning": phone_number_in_reasoning,
        },
        tripwire_triggered=phone_number_in_response or phone_number_in_reasoning,
    )


# --- 3. Define the agent with the output model and output guardrail ---
agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant.",
    output_type=MessageOutput,
    output_guardrails=[sensitive_data_check],
    model=settings.OPENAI_MODEL_NAME,
)


async def main() -> None:
    # --- 4. Run with a safe message (guardrail should not trip) ---
    await Runner.run(agent, "What's the capital of California?")
    print("First message passed - guardrail didn't trip as expected.")

    try:
        # --- 5. Run with a message containing sensitive data ---
        result = await Runner.run(
            agent, "My phone number is 650-123-4567. Where do you think I live?"
        )
        print(
            f"Guardrail didn't trip - unexpected. Output: "
            f"{json.dumps(result.final_output.model_dump(), indent=2)}"
        )
    except OutputGuardrailTripwireTriggered as e:
        print(f"Guardrail tripped. Info: {e.guardrail_result.output.output_info}")


if __name__ == "__main__":
    asyncio.run(main())
