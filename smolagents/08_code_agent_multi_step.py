from smolagents import CodeAgent, OpenAIModel, tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents CodeAgent multi-step
reasoning (CodeAct pattern):

- CodeAgent writing and executing Python code across steps
- Multi-step problem solving with intermediate variables
- Using additional_authorized_imports for stdlib modules
- Observing how CodeAgent chains computations in code

CodeAgent is smolagents' unique contribution: instead of JSON
tool calls, the agent writes actual Python code that gets
executed. This enables complex multi-step reasoning where
each step builds on previous results using real code.

For more details, visit:
https://huggingface.co/docs/smolagents/conceptual_guides/react
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define tools for the agent ---
@tool
def get_stock_price(ticker: str) -> float:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: The stock ticker symbol (e.g., AAPL, GOOGL).

    Returns:
        The current stock price as a float.
    """
    prices = {
        "AAPL": 178.50,
        "GOOGL": 141.25,
        "MSFT": 378.90,
        "AMZN": 185.60,
        "TSLA": 248.30,
    }
    return prices.get(ticker.upper(), 0.0)


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> float:
    """Get the exchange rate between two currencies.

    Args:
        from_currency: Source currency code (e.g., USD).
        to_currency: Target currency code (e.g., EUR).

    Returns:
        The exchange rate as a float.
    """
    rates = {
        ("USD", "EUR"): 0.92,
        ("USD", "GBP"): 0.79,
        ("USD", "JPY"): 149.50,
        ("EUR", "USD"): 1.09,
    }
    return rates.get((from_currency.upper(), to_currency.upper()), 0.0)


# --- 3. Create CodeAgent with additional imports ---
agent = CodeAgent(
    tools=[get_stock_price, get_exchange_rate],
    model=model,
    max_steps=5,
    additional_authorized_imports=["math"],
)

# --- 4. Run a multi-step reasoning task ---
print("=== CodeAgent Multi-Step (CodeAct) Demo ===\n")

result = agent.run(
    "I want to buy 10 shares each of AAPL and GOOGL. "
    "Calculate the total cost in USD, then convert it to EUR. "
    "Show the breakdown and final amount in 2-3 sentences."
)
print(f"\nFinal result: {result}")

# --- 5. Show the code the agent wrote ---
print("\n--- Code executed by the agent ---")
for i, step in enumerate(agent.memory.steps):
    if hasattr(step, "tool_calls") and step.tool_calls:
        for tc in step.tool_calls:
            print(f"\nStep {i} - Tool: {tc.name}({tc.arguments})")
    if hasattr(step, "model_output") and step.model_output:
        # The model_output contains the code the agent wrote
        output = str(step.model_output)
        if "```" in output or "=" in output:
            preview = output[:300]
            print(f"Step {i} - Code/output preview:\n{preview}")
