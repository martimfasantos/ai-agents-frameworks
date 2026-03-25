from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents streaming capabilities:

- Streaming agent steps in real time with stream=True
- Inspecting ActionStep and intermediate outputs
- Distinguishing between thinking steps and final answer

Streaming lets you observe the agent's reasoning process as
it happens, which is essential for UIs, debugging, and
understanding multi-step problem solving.

For more details, visit:
https://huggingface.co/docs/smolagents/guided_tour#agentic-loop
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define a simple tool ---
@tool
def calculate_compound_interest(principal: float, rate: float, years: int) -> str:
    """Calculate compound interest on an investment.

    Args:
        principal: The initial investment amount in dollars.
        rate: Annual interest rate as a percentage (e.g., 5 for 5%).
        years: Number of years to compound.

    Returns:
        A string with the final amount and interest earned.
    """
    final_amount = principal * (1 + rate / 100) ** years
    interest = final_amount - principal
    return (
        f"Principal: ${principal:,.2f}, Rate: {rate}%, Years: {years} → "
        f"Final: ${final_amount:,.2f}, Interest earned: ${interest:,.2f}"
    )


# --- 3. Create agent ---
agent = CodeAgent(
    tools=[calculate_compound_interest],
    model=model,
    max_steps=4,
)

# --- 4. Stream the agent's execution ---
print("=== Streaming Demo ===\n")
print("Streaming agent steps as they happen:\n")

step_count = 0
for step in agent.run(
    "If I invest $10,000 at 7% annual interest for 20 years, "
    "how much will I have? Reply in one sentence.",
    stream=True,
):
    # Each yielded item is a step or the final output
    step_type = type(step).__name__
    if step_type == "ActionStep":
        step_count += 1
        print(f"  [Step {step_count}] {step_type}")
        if hasattr(step, "tool_calls") and step.tool_calls:
            for tc in step.tool_calls:
                print(f"    Tool call: {tc.name}({tc.arguments})")
        if hasattr(step, "observations") and step.observations:
            obs_preview = str(step.observations)[:200]
            print(f"    Observation: {obs_preview}")
    elif step_type == "FinalAnswerStep":
        print(f"\n  [Final Answer] {step.output}")
    else:
        # Could be a text token or other step type
        if isinstance(step, str):
            pass  # Token-level streaming
        else:
            print(f"  [{step_type}]")

print(f"\nTotal steps: {step_count}")
