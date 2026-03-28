from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents callbacks and observability:

- Step callbacks for monitoring agent execution
- Logging tool calls, observations, and timing
- Custom callback functions for debugging
- Building an execution trace/log

Callbacks let you hook into the agent's execution loop to
monitor, log, or modify behavior at each step. Combined
with memory inspection, this provides full observability
into what the agent is doing and why.

For more details, visit:
https://huggingface.co/docs/smolagents/tutorials/inspect_runs
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define tools ---
@tool
def search_products(category: str) -> str:
    """Search for products in a given category.

    Args:
        category: The product category to search (e.g., electronics, books).

    Returns:
        A list of products in that category.
    """
    catalog = {
        "electronics": "1. Wireless Headphones ($79), 2. USB-C Hub ($35), 3. Portable Charger ($25)",
        "books": "1. 'Clean Code' ($30), 2. 'Design Patterns' ($45), 3. 'The Pragmatic Programmer' ($40)",
        "kitchen": "1. Coffee Maker ($60), 2. Blender ($45), 3. Toaster ($30)",
    }
    return catalog.get(
        category.lower(),
        f"No products found in category '{category}'",
    )


@tool
def get_product_details(product_name: str) -> str:
    """Get detailed information about a specific product.

    Args:
        product_name: The name of the product.

    Returns:
        Product details including rating and availability.
    """
    details = {
        "wireless headphones": "Rating: 4.5/5, In stock, Free shipping, Noise-canceling",
        "clean code": "Rating: 4.8/5, In stock, By Robert C. Martin, 464 pages",
        "coffee maker": "Rating: 4.2/5, In stock, 12-cup capacity, Programmable",
    }
    for key, val in details.items():
        if key in product_name.lower():
            return val
    return f"No details available for '{product_name}'"


# --- 3. Build a custom execution logger via callbacks ---
execution_log = []


def logging_callback(step):
    """Comprehensive step logger that builds an execution trace."""
    step_num = getattr(step, "step_number", len(execution_log))
    step_type = type(step).__name__

    entry = {
        "step": step_num,
        "type": step_type,
        "tools_called": [],
        "has_error": False,
    }

    # Log tool calls
    if hasattr(step, "tool_calls") and step.tool_calls:
        for tc in step.tool_calls:
            entry["tools_called"].append(
                {"name": tc.name, "args": str(tc.arguments)[:100]}
            )

    # Log errors
    if hasattr(step, "error") and step.error:
        entry["has_error"] = True
        entry["error_msg"] = str(step.error)[:100]

    # Log observations
    if hasattr(step, "observations") and step.observations:
        entry["observation_length"] = len(str(step.observations))

    execution_log.append(entry)


# --- 4. Create agent with logging callback ---
agent = CodeAgent(
    tools=[search_products, get_product_details],
    model=model,
    max_steps=5,
    step_callbacks=[logging_callback],
)

# --- 5. Run the agent ---
print("=== Callbacks & Observability Demo ===\n")

result = agent.run(
    "Search for electronics products, then get details on the wireless headphones. "
    "Give a brief 1-2 sentence summary."
)
print(f"Agent result: {result}\n")

# --- 6. Display the execution trace ---
print("--- Execution Trace ---")
for entry in execution_log:
    print(f"\n  Step {entry['step']} ({entry['type']}):")
    if entry["tools_called"]:
        for tc in entry["tools_called"]:
            print(f"    Tool: {tc['name']}({tc['args']})")
    if entry.get("observation_length"):
        print(f"    Observation length: {entry['observation_length']} chars")
    if entry["has_error"]:
        print(f"    ERROR: {entry.get('error_msg', 'unknown')}")

print(f"\n--- Summary ---")
print(f"Total steps: {len(execution_log)}")
total_tool_calls = sum(len(e["tools_called"]) for e in execution_log)
print(f"Total tool calls: {total_tool_calls}")
errors = sum(1 for e in execution_log if e["has_error"])
print(f"Errors: {errors}")
