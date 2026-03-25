from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents memory management:

- Inspecting agent memory via agent.memory
- Reviewing past steps and tool calls
- Using agent.replay() to replay execution history
- Step callbacks for monitoring/modifying memory dynamically
- Resetting memory between runs

Agents maintain a memory of all steps taken during a run,
including code written, tool calls made, and observations.
This is useful for debugging, logging, and building
conversational agents.

For more details, visit:
https://huggingface.co/docs/smolagents/tutorials/memory
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define a tool ---
@tool
def lookup_fact(topic: str) -> str:
    """Look up an interesting fact about a topic.

    Args:
        topic: The topic to look up.

    Returns:
        An interesting fact string.
    """
    facts = {
        "python": "Python was named after Monty Python, not the snake.",
        "mars": "A day on Mars is about 24 hours and 37 minutes long.",
        "coffee": "Coffee was discovered by Ethiopian goat herders around 850 AD.",
    }
    return facts.get(topic.lower(), f"No fact available for '{topic}'.")


# --- 3. Define a step callback for monitoring ---
def step_monitor(step):
    """Callback that fires after each agent step."""
    step_num = getattr(step, "step_number", "?")
    print(f"  [Callback] Step {step_num} completed")


# --- 4. Create agent with step callback ---
agent = CodeAgent(
    tools=[lookup_fact],
    model=model,
    max_steps=4,
    step_callbacks=[step_monitor],
)

# --- 5. Run the agent ---
print("=== Memory Management Demo ===\n")

print("--- Running agent query ---")
result = agent.run(
    "Look up a fact about Python and a fact about Mars. "
    "Summarize both facts in 1-2 sentences."
)
print(f"\nAgent result: {result}\n")

# --- 6. Inspect memory ---
print("--- Inspecting agent memory ---")
memory = agent.memory
print(f"Number of steps in memory: {len(memory.steps)}")

for i, step in enumerate(memory.steps):
    step_type = type(step).__name__
    print(f"\n  Step {i}: {step_type}")
    if hasattr(step, "tool_calls") and step.tool_calls:
        for tc in step.tool_calls:
            print(f"    Tool: {tc.name}({tc.arguments})")
    if hasattr(step, "model_output") and step.model_output:
        output_preview = str(step.model_output)[:150]
        print(f"    Model output: {output_preview}...")

# --- 7. Show that memory persists between runs ---
print("\n--- Memory persists across runs ---")
print(f"Steps before second run: {len(agent.memory.steps)}")

result2 = agent.run("Look up a fact about coffee. Reply in one sentence.")
print(f"Second result: {result2}")
print(f"Steps after second run: {len(agent.memory.steps)}")

# --- 8. Reset memory ---
print("\n--- Resetting memory ---")
agent.memory.steps.clear()
print(f"Steps after reset: {len(agent.memory.steps)}")
