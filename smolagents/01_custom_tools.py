from smolagents import CodeAgent, OpenAIModel, tool, Tool

from settings import settings

"""
-------------------------------------------------------
In this example, we explore smolagents custom tool creation:

- Defining tools with the @tool decorator
- Defining tools by subclassing the Tool class
- Passing tools to a CodeAgent
- Agent autonomously selecting and calling the right tools

smolagents supports two patterns for creating custom tools:
the @tool decorator for simple functions, and the Tool base
class for more complex tools with state or custom schemas.

For more details, visit:
https://huggingface.co/docs/smolagents/tutorials/tools
-------------------------------------------------------
"""

# --- 1. Create the model ---
model = OpenAIModel(
    model_id=settings.OPENAI_MODEL_NAME,
    api_key=settings.OPENAI_API_KEY.get_secret_value(),
)


# --- 2. Define a tool using the @tool decorator ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city to get weather for.

    Returns:
        A string with the weather report.
    """
    weather_data = {
        "lisbon": "Sunny, 25°C, humidity 45%",
        "london": "Cloudy, 15°C, humidity 78%",
        "tokyo": "Rainy, 18°C, humidity 90%",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


# --- 3. Define a tool using the Tool subclass ---
class CurrencyConverterTool(Tool):
    name = "currency_converter"
    description = (
        "Convert an amount from one currency to another using fixed demo rates."
    )
    inputs = {
        "amount": {"type": "number", "description": "The amount to convert"},
        "from_currency": {
            "type": "string",
            "description": "Source currency code (e.g., USD)",
        },
        "to_currency": {
            "type": "string",
            "description": "Target currency code (e.g., EUR)",
        },
    }
    output_type = "string"

    # Fixed demo exchange rates
    rates = {
        ("USD", "EUR"): 0.92,
        ("EUR", "USD"): 1.09,
        ("USD", "GBP"): 0.79,
        ("GBP", "USD"): 1.27,
        ("EUR", "GBP"): 0.86,
        ("GBP", "EUR"): 1.16,
    }

    def forward(self, amount: float, from_currency: str, to_currency: str) -> str:
        key = (from_currency.upper(), to_currency.upper())
        if key in self.rates:
            converted = amount * self.rates[key]
            return f"{amount} {from_currency.upper()} = {converted:.2f} {to_currency.upper()}"
        return f"Conversion rate not available for {from_currency} to {to_currency}"


# --- 4. Create the agent with both tools ---
agent = CodeAgent(
    tools=[get_weather, CurrencyConverterTool()],
    model=model,
    max_steps=3,
)

# --- 5. Run queries that trigger each tool ---
print("=== Custom Tools Demo ===\n")

print("--- Query 1: Weather tool ---")
result1 = agent.run("What's the weather like in Lisbon? Reply in one sentence.")
print(f"Result: {result1}\n")

print("--- Query 2: Currency converter tool ---")
result2 = agent.run("Convert 100 USD to EUR. Reply in one sentence.")
print(f"Result: {result2}")
