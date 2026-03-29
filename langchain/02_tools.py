import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_openai import ChatOpenAI

from settings import settings

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY.get_secret_value()

"""
-------------------------------------------------------
In this example, we explore LangChain with the following features:
- Creating custom tools with the @tool decorator
- Passing tools to an agent via create_agent()
- Agent autonomously deciding which tools to call

Tools let agents interact with external systems. LangChain uses
the @tool decorator to convert Python functions into tools the
LLM can discover and call. The function name, docstring, and
parameter types are exposed to the model as the tool schema.

For more details, visit:
https://docs.langchain.com/oss/python/langchain/tools
-------------------------------------------------------
"""


# --- 1. Define custom tools ---
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    weather_data = {
        "lisbon": "Sunny, 25°C",
        "london": "Cloudy, 15°C",
        "tokyo": "Rainy, 18°C",
    }
    return weather_data.get(city.lower(), f"No weather data available for {city}")


@tool
def calculate(expression: str) -> str:
    """Evaluate a simple math expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- 2. Create the model ---
model = ChatOpenAI(model=settings.OPENAI_MODEL_NAME)

# --- 3. Create an agent with tools ---
agent = create_agent(
    model=model,
    tools=[get_weather, calculate],
    system_prompt="You are a helpful assistant with access to weather and math tools.",
)

# --- 4. Ask a question that requires a tool ---
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What's the weather like in Lisbon?"}]}
)
print(f"Weather query: {result['messages'][-1].content}\n")

# --- 5. Ask a question requiring the calculator ---
result = agent.invoke(
    {"messages": [{"role": "user", "content": "What is 42 * 17 + 3?"}]}
)
print(f"Math query: {result['messages'][-1].content}")
